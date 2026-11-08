"""Таблицы оперативного пакета (ТЗ §13.2).

Одномерные распределения и базовые кросс-табуляции. Взвешенные и невзвешенные
доли всегда стоят рядом: расхождение между ними — это и есть вклад
взвешивания, и читатель отчёта должен видеть его, а не догадываться о нём.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from socmon.analysis.dataset import Dataset
from socmon.analysis.estimates import Estimate, weighted_share
from socmon.export.codebook import Codebook, CodebookEntry


@dataclass(frozen=True, slots=True)
class FrequencyRow:
    code: int
    label: str
    count: int
    unweighted_share: float
    weighted_share: float
    estimate: Estimate | None
    is_service: bool
    """Служебный код: отказ, затруднение, нераспознанная речь. Считается
    отдельно и в базу содержательных ответов не входит."""


@dataclass(frozen=True, slots=True)
class FrequencyTable:
    variable: str
    question: str
    rows: tuple[FrequencyRow, ...]
    substantive_n: int
    total_n: int

    @property
    def missing_share(self) -> float:
        if not self.total_n:
            return 0.0
        return 1 - self.substantive_n / self.total_n


def frequency(records: Dataset, entry: CodebookEntry) -> FrequencyTable:
    """Одномерное распределение по переменной.

    Проценты содержательных вариантов считаются от базы содержательных
    ответов, а служебные коды — от всей подвыборки. Смешивать эти базы
    нельзя: иначе доля «затрудняюсь» разбавляет распределение мнений
    и все цифры оказываются занижены.
    """
    counts: Counter[int] = Counter()
    present = 0
    for record in records:
        code = record.get(entry.variable)
        if code is None:
            continue
        present += 1
        counts[code] += 1

    service = set(entry.service_values)
    substantive = sum(n for code, n in counts.items() if code not in service)

    rows: list[FrequencyRow] = []
    for code in sorted(counts):
        is_service = code in service
        base = present if is_service else substantive
        estimate = None
        if not is_service and substantive:
            estimate = weighted_share(
                records, lambda r, c=code: _substantive_match(r, entry, c, service)  # type: ignore[misc]
            )
        rows.append(FrequencyRow(
            code=code,
            label=entry.value_label(code),
            count=counts[code],
            unweighted_share=counts[code] / base if base else 0.0,
            weighted_share=estimate.value if estimate else _weighted(records, entry, code, service),
            estimate=estimate,
            is_service=is_service,
        ))

    return FrequencyTable(
        variable=entry.variable,
        question=entry.question,
        rows=tuple(rows),
        substantive_n=substantive,
        total_n=len(records),
    )


def _substantive_match(
    record: object, entry: CodebookEntry, code: int, service: set[int]
) -> bool | None:
    """Служебные ответы выпадают из базы, а не засчитываются в «не выбрал»."""
    value = record.get(entry.variable)  # type: ignore[attr-defined]
    if value is None or value in service:
        return None
    return bool(value == code)


def _weighted(records: Dataset, entry: CodebookEntry, code: int, service: set[int]) -> float:
    total = sum(r.weight for r in records if r.get(entry.variable) is not None)
    if total <= 0:
        return 0.0
    hit = sum(r.weight for r in records if r.get(entry.variable) == code)
    return hit / total


@dataclass(frozen=True, slots=True)
class CrossTable:
    row_variable: str
    column_variable: str
    row_labels: tuple[str, ...]
    column_labels: tuple[str, ...]
    counts: tuple[tuple[int, ...], ...]
    shares: tuple[tuple[float, ...], ...]
    """Взвешенные доли по строке: сколько процентов внутри каждой группы."""

    row_totals: tuple[int, ...]
    unreliable_rows: tuple[str, ...]
    """Строки, где подвыборка слишком мала для содержательных выводов
    (ТЗ §5.2). Их нужно печатать с оговоркой, а не молча."""


MIN_CELL_BASE = 100


def crosstab(
    records: Dataset,
    row_entry: CodebookEntry,
    column_entry: CodebookEntry,
) -> CrossTable:
    """Базовая кросс-табуляция: распределение переменной внутри групп."""
    row_codes = sorted(row_entry.values)
    column_codes = sorted(column_entry.values)
    service = set(column_entry.service_values)

    counts: list[tuple[int, ...]] = []
    shares: list[tuple[float, ...]] = []
    totals: list[int] = []
    unreliable: list[str] = []

    for row_code in row_codes:
        group = [r for r in records if r.get(row_entry.variable) == row_code]
        eligible = [r for r in group if (v := r.get(column_entry.variable)) is not None
                    and v not in service]
        weight_total = sum(r.weight for r in eligible)

        counts.append(tuple(
            sum(1 for r in eligible if r.get(column_entry.variable) == code)
            for code in column_codes
        ))
        shares.append(tuple(
            (sum(r.weight for r in eligible if r.get(column_entry.variable) == code)
             / weight_total) if weight_total else 0.0
            for code in column_codes
        ))
        totals.append(len(eligible))
        if len(eligible) < MIN_CELL_BASE:
            unreliable.append(row_entry.value_label(row_code))

    return CrossTable(
        row_variable=row_entry.variable,
        column_variable=column_entry.variable,
        row_labels=tuple(row_entry.value_label(c) for c in row_codes),
        column_labels=tuple(column_entry.value_label(c) for c in column_codes),
        counts=tuple(counts),
        shares=tuple(shares),
        row_totals=tuple(totals),
        unreliable_rows=tuple(unreliable),
    )


def all_frequencies(records: Dataset, book: Codebook) -> list[FrequencyTable]:
    """Одномерные распределения по всем закрытым вопросам анкеты."""
    return [
        frequency(records, entry)
        for entry in book
        if entry.kind != "open" and any(
            r.get(entry.variable) is not None for r in records
        )
    ]
