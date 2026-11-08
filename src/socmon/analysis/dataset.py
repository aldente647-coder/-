"""Массив волны в том виде, в каком его считает аналитика."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Record:
    """Одно валидное интервью.

    Идентификатор — обезличенный: аналитический слой номера не видит
    в принципе (ТЗ §14.2).
    """

    id: str
    strata: Mapping[str, str] = field(default_factory=dict)
    """Признаки структуры выборки: пол, возраст, территория и прочее (ТЗ §5.4)."""

    answers: Mapping[str, int | None] = field(default_factory=dict)
    """Ответы по ключам постоянного ядра, а не по идентификаторам узлов:
    формулировка вопроса между волнами может меняться, ключ показателя — нет."""

    weight: float = 1.0

    def get(self, core_key: str) -> int | None:
        return self.answers.get(core_key)


Dataset = Sequence[Record]


def subgroup(records: Dataset, **criteria: str) -> list[Record]:
    """Подвыборка по признакам структуры.

    ТЗ §5.2 требует по малым территориям и подгруппам указывать фактический
    объём подвыборки и ограничение на интерпретацию — поэтому подвыборка
    здесь обычный список, а предупреждение о её размере выдаёт оценка.
    """
    return [
        record
        for record in records
        if all(record.strata.get(key) == value for key, value in criteria.items())
    ]


def with_weights(records: Dataset, weights: Iterable[float]) -> list[Record]:
    """Приписать массиву рассчитанные веса."""
    weighted = [
        Record(id=r.id, strata=r.strata, answers=r.answers, weight=w)
        for r, w in zip(records, weights, strict=True)
    ]
    return weighted
