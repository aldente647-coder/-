"""Загрузка базы контактов (ТЗ §4.2).

Требования раздела складываются в один список: безопасная загрузка,
техническая валидация формата, исключение дублей, случайный порядок обзвона
и ведение истории. Здесь всё это, и каждое требование стоит объяснения.

**Потоком, а не целиком.** Заказчик передаёт до миллиона номеров. Прочитать
их в память списком можно, но незачем: файл обрабатывается кусками, и
загрузка миллиона строк стоит столько же памяти, сколько загрузка тысячи.

**Случайный порядок — воспроизводимый.** §4.2 требует случайного порядка
обзвона, §5.1 — воспроизводимости отбора с фиксацией зерна. Порядок
выводится детерминированно из зерна и слепого индекса: он случаен по
отношению к любому признаку абонента, но восстанавливается по зерну
и предъявляется проверяющему.

**Дубли считаются раздельно.** Дубль внутри файла и совпадение с уже
загруженной базой — разные события: первое говорит о качестве файла,
второе о пересечении поставок. Свести их в одно число значит потерять
и то и другое.
"""

from __future__ import annotations

import hashlib
import secrets
import uuid
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import islice

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session as DbSession
from sqlalchemy.sql.elements import ColumnElement

from socmon.db.models import Contact, ContactBatch
from socmon.domain.identity import PhoneError, PhoneVault, normalize_phone

CHUNK_SIZE = 5_000
MAX_REJECT_SAMPLES = 50
"""Сколько отклонённых строк показать в отчёте. Полный список миллиона
ошибок никто не читает, а полсотни примеров объясняют, что не так с файлом."""


class ImportError_(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Reject:
    line: int
    value: str
    reason: str


@dataclass
class ImportReport:
    """Итог загрузки (ТЗ §4.2: отчёт о загрузке)."""

    batch_id: uuid.UUID
    seed: str
    total: int = 0
    loaded: int = 0
    duplicate_in_file: int = 0
    duplicate_in_base: int = 0
    rejected: int = 0
    samples: list[Reject] = field(default_factory=list)

    @property
    def usable_share(self) -> float:
        return self.loaded / self.total if self.total else 0.0

    @property
    def notes(self) -> tuple[str, ...]:
        notes: list[str] = []
        if self.rejected:
            notes.append(
                f"Отклонено строк: {self.rejected}. Это качество файла "
                f"Заказчика, а не сбой загрузки — номера не приводятся "
                f"к каноническому виду."
            )
        if self.duplicate_in_file:
            notes.append(
                f"Дублей внутри файла: {self.duplicate_in_file}. "
                f"Один абонент не должен попадать в выборку дважды (ТЗ §4.2)."
            )
        if self.duplicate_in_base:
            notes.append(
                f"Совпадений с уже загруженной базой: {self.duplicate_in_base}. "
                f"История звонков по ним сохранена, повторно они не заводятся."
            )
        if self.total and self.usable_share < 0.9:
            notes.append(
                f"Пригодных номеров {self.usable_share:.1%} — стоит вернуть "
                f"файл Заказчику до начала волны."
            )
        return tuple(notes)


def new_seed() -> str:
    """Зерно случайного порядка. Хранится на загрузке и не меняется."""
    return secrets.token_hex(16)


def dial_order(seed: str, blind_index: bytes) -> int:
    """Ключ порядка обзвона.

    Хеш зерна и слепого индекса: значение непредсказуемо без зерна и
    полностью восстановимо с ним. Берутся 63 бита — знаковый BIGINT
    не вмещает 64.
    """
    digest = hashlib.sha256(seed.encode("utf-8") + blind_index).digest()
    return int.from_bytes(digest[:8], "big") >> 1


def chunked(rows: Iterable[str], size: int = CHUNK_SIZE) -> Iterator[list[str]]:
    iterator = iter(rows)
    while chunk := list(islice(iterator, size)):
        yield chunk


def import_numbers(
    db: DbSession,
    vault: PhoneVault,
    rows: Iterable[str],
    *,
    name: str,
    source: str,
    seed: str | None = None,
    chunk_size: int = CHUNK_SIZE,
) -> ImportReport:
    """Загрузить номера в базу.

    Существующие контакты не перезаписываются: у них уже есть история
    звонков и, возможно, отметка стоп-листа. Затереть её новой поставкой
    значило бы позвонить человеку, который просил больше не звонить
    (ТЗ §6.2).
    """
    batch = ContactBatch(name=name, source=source, seed=seed or new_seed())
    db.add(batch)
    db.flush()

    report = ImportReport(batch_id=batch.id, seed=batch.seed)
    seen: set[bytes] = set()

    for lines in chunked(rows, chunk_size):
        payload: list[dict[str, object]] = []
        for raw in lines:
            report.total += 1
            index = report.total
            value = raw.strip()
            if not value:
                _reject(report, index, value, "пустая строка")
                continue
            try:
                normalized = normalize_phone(value)
            except PhoneError as exc:
                _reject(report, index, value, str(exc))
                continue

            blind = vault.blind_index(normalized)
            if blind in seen:
                report.duplicate_in_file += 1
                continue
            seen.add(blind)

            sealed = vault.seal(normalized)
            payload.append({
                "blind_index": blind,
                "phone_ciphertext": sealed.ciphertext,
                "batch_id": batch.id,
                "dial_order": dial_order(batch.seed, blind),
            })

        if payload:
            report.loaded += _insert(db, payload)

    report.duplicate_in_base = (
        report.total - report.rejected - report.duplicate_in_file - report.loaded
    )
    batch.rows_total = report.total
    batch.rows_loaded = report.loaded
    batch.rows_duplicate = report.duplicate_in_file + report.duplicate_in_base
    batch.rows_rejected = report.rejected
    db.flush()
    return report


def _insert(db: DbSession, payload: list[dict[str, object]]) -> int:
    """Вставить пачку, молча пропуская уже известные номера.

    Проверка «есть ли такой» отдельным запросом не годится: между проверкой
    и вставкой номер может появиться из параллельной загрузки. Уникальность
    слепого индекса решает это в базе, а не в приложении.
    """
    statement = (
        insert(Contact)
        .values(payload)
        .on_conflict_do_nothing(index_elements=[Contact.blind_index])
        .returning(Contact.id)
    )
    return len(db.execute(statement).all())


def _reject(report: ImportReport, line: int, value: str, reason: str) -> None:
    report.rejected += 1
    if len(report.samples) < MAX_REJECT_SAMPLES:
        report.samples.append(Reject(line=line, value=_mask(value), reason=reason))


def _mask(value: str) -> str:
    """Показать отклонённую строку, не выписывая номер целиком в отчёт."""
    if len(value) <= 6:
        return value
    return f"{value[:3]}…{value[-2:]}"


def dialable(db: DbSession, limit: int) -> list[Contact]:
    """Контакты, готовые к набору, в случайном порядке.

    Порядок задаётся ключом загрузки, а не `ORDER BY random()`: последний
    даёт новый порядок при каждом запросе и делает отбор непроверяемым,
    чего §5.1 не допускает.
    """
    return list(db.scalars(
        select(Contact)
        .where(
            Contact.do_not_call.is_(False),
            Contact.invalid_number.is_(False),
            Contact.never_answers.is_(False),
            Contact.deactivated_at.is_(None),
        )
        .order_by(Contact.dial_order)
        .limit(limit)
    ).all())


def base_summary(db: DbSession) -> dict[str, int]:
    """Состояние базы для кабинета."""
    def count(*conditions: ColumnElement[bool]) -> int:
        query = select(func.count()).select_from(Contact)
        for condition in conditions:
            query = query.where(condition)
        return int(db.scalar(query) or 0)

    return {
        "total": count(),
        "do_not_call": count(Contact.do_not_call.is_(True)),
        "invalid": count(Contact.invalid_number.is_(True)),
        "never_answers": count(Contact.never_answers.is_(True)),
        "dialable": count(
            Contact.do_not_call.is_(False),
            Contact.invalid_number.is_(False),
            Contact.never_answers.is_(False),
            Contact.deactivated_at.is_(None),
        ),
    }


def read_column(text: str, column: int | None = None) -> Iterator[str]:
    """Достать номера из CSV или простого списка.

    Разделитель определяется по первой строке: Заказчик присылает и точку
    с запятой, и запятую, и просто по номеру в строке, и спорить об этом
    в переписке дольше, чем определить.
    """
    lines = (line for line in text.splitlines() if line.strip())
    first = next(lines, None)
    if first is None:
        return

    delimiter = max(";,\t", key=first.count) if any(c in first for c in ";,\t") else ""

    def cells(line: str) -> list[str]:
        if not delimiter:
            return [line.strip()]
        return [c.strip().strip('"') for c in line.split(delimiter)]

    header = cells(first)
    index = column if column is not None else _guess_column(header)
    # Первая строка может быть заголовком, а может быть данными: если
    # в выбранной ячейке достаточно цифр, это номер, и терять его нельзя.
    if index < len(header) and _looks_like_number(header[index]):
        yield header[index]

    for line in lines:
        values = cells(line)
        if index < len(values):
            yield values[index]


def _guess_column(header: list[str]) -> int:
    for position, cell in enumerate(header):
        if _looks_like_number(cell) or "телефон" in cell.lower() or "phone" in cell.lower():
            return position
    return 0


def _looks_like_number(cell: str) -> bool:
    digits = sum(1 for c in cell if c.isdigit())
    return digits >= 10


def deactivate_never_answering(
    db: DbSession, *, min_attempts: int, since: datetime | None = None
) -> int:
    """Исключить систематически не отвечающих (ТЗ §4.2).

    Отметка ставится, а контакт остаётся: история звонков по нему — часть
    расчёта достижимости, и удалять её нельзя.
    """
    from socmon.db.models import CallAttempt

    stamp = since or datetime.now(UTC)
    silent = (
        select(CallAttempt.contact_id)
        .where(CallAttempt.ended_at.isnot(None))
        .group_by(CallAttempt.contact_id)
        .having(func.count() >= min_attempts)
        .having(func.count(CallAttempt.started_at) == 0)
    )
    contacts = db.scalars(
        select(Contact).where(
            Contact.id.in_(silent), Contact.never_answers.is_(False)
        )
    ).all()
    for contact in contacts:
        contact.never_answers = True
        contact.deactivated_at = stamp
    db.flush()
    return len(contacts)
