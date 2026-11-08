"""Сборка выгрузок для кабинета.

Отдельный модуль, потому что здесь сходятся все слои сразу: массив из базы,
взвешивание, индекс, кодбук из анкеты, достижимость по AAPOR и отчёт. Держать
это в маршрутах значило бы получить контроллер на двести строк, который
невозможно проверить.

Пакет собирается в архив на диске, а не отдаётся потоком на лету. Причина
не в удобстве: §13.1 требует предоставить оперативный пакет не позднее трёх
рабочих дней, и «предоставить» означает файл, который можно скачать дважды,
сверить и приложить к переписке, а не результат сиюминутного запроса.
"""

from __future__ import annotations

import os
import tempfile
import uuid
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml
from sqlalchemy import func, select
from sqlalchemy.orm import Session as DbSession

from socmon.analysis.outcomes import (
    CallStatistics,
    Dispositions,
    ResponseRates,
    response_rates,
)
from socmon.analysis.wave import WaveSummary, load_valid_records, summarize
from socmon.analysis.weighting import Margin, WeightingError
from socmon.core.codes import AaporCode, CallStage, Disposition, Outcome, to_aapor
from socmon.core.questionnaire import Questionnaire
from socmon.db.models import CallAttempt, IdentityLink, Interview, QuestionnaireVersion, Wave
from socmon.domain.validity import Validity
from socmon.export import codebook as codebook_builder
from socmon.export import package as package_builder
from socmon.export.package import WavePackage
from socmon.export.report import write_report
from socmon.export.writers import validate_wave_code, write_package

EXPORTS_DIR = Path("exports")


class ArtifactError(RuntimeError):
    """Собрать выгрузку нельзя. Сообщение предназначено человеку."""


@dataclass(frozen=True, slots=True)
class Artifact:
    wave_code: str
    path: Path
    built_at: datetime
    files: tuple[str, ...]

    @property
    def size_bytes(self) -> int:
        return self.path.stat().st_size if self.path.exists() else 0


def questionnaire_of(version: QuestionnaireVersion) -> Questionnaire | None:
    """Разобрать сохранённую версию анкеты.

    Версия хранится исходным текстом, а не разобранной структурой: так
    её можно перечитать ровно в том виде, в каком её утверждал Заказчик,
    даже если модель данных с тех пор изменилась (ТЗ §7.3).
    """
    try:
        return Questionnaire.model_validate(yaml.safe_load(version.source_yaml))
    except Exception:
        return None


def wave_summary(db: DbSession, wave: Wave) -> WaveSummary:
    """Взвесить массив волны и посчитать индекс.

    Отказывается считать без эталонных распределений: показать невзвешенные
    цифры как взвешенные — худшее, что можно сделать (ТЗ §5.4).
    """
    if not wave.reference_distributions:
        raise ArtifactError(
            "не заданы эталонные распределения населения — взвешивание "
            "по ТЗ §5.4 невозможно"
        )
    version = db.scalar(
        select(QuestionnaireVersion).where(QuestionnaireVersion.wave_id == wave.id)
    )
    if version is None:
        raise ArtifactError("к волне не привязана версия анкеты")

    questionnaire = questionnaire_of(version)
    if questionnaire is None:
        raise ArtifactError("не удалось разобрать сохранённую версию анкеты")

    margins = [
        Margin(variable, dict(targets))
        for variable, targets in wave.reference_distributions.items()
    ]
    records = load_valid_records(
        db, wave.id, questionnaire, [m.variable for m in margins]
    )
    if not records:
        raise ArtifactError("в волне нет валидных интервью")

    try:
        return summarize(
            records, margins, wave_id=wave.id, target_min=wave.target_min
        )
    except WeightingError as exc:
        raise ArtifactError(str(exc)) from exc


def call_statistics(db: DbSession, wave_id: uuid.UUID) -> CallStatistics:
    """Достижимость по AAPOR (ТЗ §13.2).

    Считается по фактическим исходам, а не по счётчикам: счётчик можно
    забыть обновить, исходы — нет.
    """
    # Один итог на контакт; повторный недозвон не умножает знаменатель RR.
    final: dict[uuid.UUID, tuple[Outcome | None, Disposition | None]] = {}
    for contact_id, outcome, disposition in db.execute(
        select(CallAttempt.contact_id, Interview.outcome, CallAttempt.disposition)
        .outerjoin(IdentityLink, IdentityLink.attempt_id == CallAttempt.id)
        .outerjoin(Interview, Interview.id == IdentityLink.interview_id)
        .where(CallAttempt.wave_id == wave_id, CallAttempt.is_test.is_(False),
               CallAttempt.stage == CallStage.FINISHED,
               (Interview.id.is_(None)) | (Interview.is_test.is_(False)))
        .order_by(CallAttempt.attempt_no)
    ):
        previous = final.get(contact_id)
        if previous is None or previous[0] is None or outcome is not None:
            final[contact_id] = (outcome, disposition)
    codes: list[AaporCode] = [to_aapor(outcome, disposition)
                             for outcome, disposition in final.values()]

    dispositions = Dispositions.from_codes(codes)
    attempts = int(db.scalar(
        select(func.count()).select_from(CallAttempt).where(
            CallAttempt.wave_id == wave_id, CallAttempt.is_test.is_(False)
        )
    ) or 0)
    numbers = int(db.scalar(
        select(func.count(func.distinct(CallAttempt.contact_id))).where(
            CallAttempt.wave_id == wave_id, CallAttempt.is_test.is_(False)
        )
    ) or 0)

    def interviews(validity: Validity) -> int:
        return int(db.scalar(
            select(func.count()).select_from(Interview).where(
                Interview.wave_id == wave_id,
                Interview.is_test.is_(False),
                Interview.validity == validity,
            )
        ) or 0)

    reasons: dict[str, int] = {}
    for stored, in db.execute(
        select(Interview.validity_reasons).where(
            Interview.wave_id == wave_id,
            Interview.validity == Validity.INVALID, Interview.is_test.is_(False),
        )
    ).all():
        for reason in stored or []:
            reasons[reason] = reasons.get(reason, 0) + 1

    return CallStatistics(
        numbers_dialled=numbers,
        attempts_made=attempts,
        dispositions=dispositions,
        rates=response_rates(dispositions) if dispositions.total else _empty_rates(),
        valid_interviews=interviews(Validity.VALID),
        invalid_interviews=interviews(Validity.INVALID),
        exclusion_reasons=reasons,
    )


def _empty_rates() -> ResponseRates:
    """Нули, когда считать ещё не по чему.

    Не «недостающие данные», а честные нули: волна без единой попытки
    дозвона имеет достижимость ноль, и это осмысленное утверждение.
    """
    return ResponseRates(
        rr1=0.0, rr2=0.0, rr3=0.0, rr4=0.0, rr5=0.0, rr6=0.0,
        cooperation1=0.0, refusal1=0.0, contact1=0.0, eligibility_rate=0.0,
    )


def build_package(db: DbSession, wave: Wave) -> WavePackage:
    """Собрать оперативный пакет волны (ТЗ §13.2)."""
    version = db.scalar(
        select(QuestionnaireVersion).where(QuestionnaireVersion.wave_id == wave.id)
    )
    if version is None:
        raise ArtifactError("к волне не привязана версия анкеты")
    questionnaire = questionnaire_of(version)
    if questionnaire is None:
        raise ArtifactError("не удалось разобрать сохранённую версию анкеты")

    return package_builder.build(
        wave_summary(db, wave),
        codebook_builder.build(questionnaire),
        wave_code=wave.code,
        call_statistics=call_statistics(db, wave.id),
        methodology_notes=(
            f"Версия анкеты {version.version_id}. Правка любой формулировки "
            f"даёт новый идентификатор, и данные остаются привязаны "
            f"к своей редакции вопросов (ТЗ §7.3).",
        ),
    )


def build_archive(
    db: DbSession, wave: Wave, *, root: Path = EXPORTS_DIR, with_report: bool = True
) -> Artifact:
    """Собрать пакет и упаковать в архив.

    Каталог волны пересобирается целиком: остатки прошлой сборки в архиве
    хуже, чем её отсутствие — по ним нельзя понять, к какой выгрузке
    относится файл.
    """
    validate_wave_code(wave.code)
    package = build_package(db, wave)
    root.mkdir(parents=True, exist_ok=True)
    archive = root / f"wave-{wave.code}.zip"
    # Отдельный временный каталог для каждого запроса; готовый ZIP заменяется
    # атомарно. Параллельная сборка не удалит файлы другой сборки.
    with tempfile.TemporaryDirectory(prefix=".build-", dir=root) as directory:
        workdir = Path(directory)
        written = write_package(workdir, package)
        if with_report:
            written.append(write_report(workdir / f"wave-{wave.code}-отчёт.docx", package))
        temporary = workdir / "package.zip"
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in written:
                zf.write(path, arcname=path.name)
        temporary.chmod(0o600)
        os.replace(temporary, archive)

    return Artifact(
        wave_code=wave.code,
        path=archive,
        built_at=datetime.now(UTC),
        files=tuple(p.name for p in written),
    )


def existing_archive(wave_code: str, *, root: Path = EXPORTS_DIR) -> Artifact | None:
    """Найти уже собранный архив.

    Имя выводится из кода волны, а не берётся из запроса: путь, собранный
    из пользовательского ввода, — это обход каталогов, а рядом лежат
    выгрузки других волн.
    """
    validate_wave_code(wave_code)
    archive = root / f"wave-{wave_code}.zip"
    if archive.is_symlink() or not archive.is_file():
        return None
    with zipfile.ZipFile(archive) as zf:
        names = tuple(zf.namelist())
    return Artifact(
        wave_code=wave_code,
        path=archive,
        built_at=datetime.fromtimestamp(archive.stat().st_mtime, UTC),
        files=names,
    )
