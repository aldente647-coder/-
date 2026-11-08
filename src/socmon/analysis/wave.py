"""Сводка по волне: от валидного массива до индекса.

Собирает в одном месте то, что ТЗ §13.2 требует отдавать в оперативном пакете:
объём и структуру достигнутой выборки, эталонные распределения, диагностику
весов, индекс с компонентным профилем и ограничения интерпретации.

Порядок шагов зафиксирован здесь намеренно: сначала веса считаются по признакам
структуры, и только потом по взвешенному массиву считаются содержательные
показатели. Обратный порядок — то есть подбор весов с оглядкой на результат —
это подгонка, прямо запрещённая §5.1.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session as DbSession

from socmon.analysis.dataset import Dataset, Record, with_weights
from socmon.analysis.index import (
    DEFAULT_COMPONENTS,
    Component,
    IndexResult,
    MissingPolicy,
)
from socmon.analysis.index import compute as compute_index
from socmon.analysis.weighting import (
    DEFAULT_POLICY,
    Margin,
    RakingPolicy,
    WeightDiagnostics,
    rake,
)
from socmon.core.questionnaire import Questionnaire, ScreenerNode, SingleNode
from socmon.db.models import Answer, Interview
from socmon.domain.validity import Validity


@dataclass(frozen=True, slots=True)
class WaveSummary:
    """Итог волны. Всё, что нужно, чтобы опубликовать цифру и защитить её."""

    wave_id: uuid.UUID
    valid_interviews: int
    target_min: int
    weights: WeightDiagnostics
    index: IndexResult
    records: tuple[Record, ...]

    @property
    def meets_target(self) -> bool:
        """ТЗ §5.2: минимум 1500 валидных интервью на волну."""
        return self.valid_interviews >= self.target_min

    @property
    def notes(self) -> tuple[str, ...]:
        """Ограничения интерпретации — единым списком в отчёт."""
        notes = list(self.weights.warnings) + list(self.index.notes)
        if not self.meets_target:
            notes.insert(0, (
                f"валидных интервью {self.valid_interviews} при минимуме "
                f"{self.target_min}: волна не добирает целевой объём, "
                f"фактические интервалы шире расчётных"
            ))
        return tuple(notes)


def summarize(
    records: Dataset,
    margins: Sequence[Margin],
    *,
    wave_id: uuid.UUID,
    target_min: int = 1500,
    components: Sequence[Component] = DEFAULT_COMPONENTS,
    raking_policy: RakingPolicy = DEFAULT_POLICY,
    missing_policy: MissingPolicy = MissingPolicy.EXCLUDE,
    neutral_substitution_approved_by: str | None = None,
) -> WaveSummary:
    """Взвесить массив и посчитать индекс."""
    raked = rake(records, margins, raking_policy)
    weighted = with_weights(records, raked.weights)
    index = compute_index(
        weighted,
        components,
        missing_policy=missing_policy,
        neutral_substitution_approved_by=neutral_substitution_approved_by,
    )
    return WaveSummary(
        wave_id=wave_id,
        valid_interviews=len(records),
        target_min=target_min,
        weights=raked.diagnostics,
        index=index,
        records=tuple(weighted),
    )


def load_valid_records(
    db: DbSession,
    wave_id: uuid.UUID,
    questionnaire: Questionnaire,
    strata_keys: Sequence[str],
) -> list[Record]:
    """Собрать валидный аналитический массив волны.

    В выборку идут только интервью со статусом «валидно» и снятым признаком
    тестового звонка — контрольные звонки не попадают в массив никогда.

    Наблюдения, у которых не заполнен хотя бы один признак структуры, сюда
    не доходят: взвешивание откажется их обрабатывать, и это правильно —
    молча выбросить их значило бы отобрать выборку руками (§5.1).
    """
    labels = _strata_labels(questionnaire, strata_keys)

    interviews = db.scalars(
        select(Interview).where(
            Interview.wave_id == wave_id,
            Interview.validity == Validity.VALID,
            Interview.is_test.is_(False),
        )
    ).all()
    if not interviews:
        return []

    rows = db.scalars(
        select(Answer).where(Answer.interview_id.in_([i.id for i in interviews]))
    ).all()

    by_interview: dict[uuid.UUID, dict[str, int | None]] = {}
    for answer in rows:
        if answer.core_key:
            by_interview.setdefault(answer.interview_id, {})[answer.core_key] = answer.code

    records: list[Record] = []
    for interview in interviews:
        answers = by_interview.get(interview.id, {})
        records.append(Record(
            id=str(interview.id),
            strata={
                key: labels[key][code]
                for key in strata_keys
                if (code := answers.get(key)) is not None and code in labels[key]
            },
            answers=answers,
            weight=float(interview.weight) if interview.weight is not None else 1.0,
        ))
    return records


def _strata_labels(
    questionnaire: Questionnaire, strata_keys: Sequence[str]
) -> dict[str, dict[int, str]]:
    """Соответствие «код варианта → подпись» для признаков структуры.

    Подписи берутся из анкеты, а не дублируются константами: иначе эталонные
    распределения Заказчика и категории массива однажды разойдутся
    в написании, и взвешивание упадёт на ровном месте.
    """
    labels: dict[str, dict[int, str]] = {}
    for node in questionnaire.nodes:
        if not isinstance(node, SingleNode | ScreenerNode) or node.core_key is None:
            continue
        if node.core_key in strata_keys:
            labels[node.core_key] = {o.code: o.label for o in node.options}

    missing = set(strata_keys) - set(labels)
    if missing:
        raise ValueError(
            f"в анкете нет вопросов с ключами {sorted(missing)}: "
            f"взвешивать по этим признакам нечем"
        )
    return labels
