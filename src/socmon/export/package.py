"""Оперативный пакет волны (ТЗ §13.2).

Собирает в одну структуру всё, что уходит Заказчику не позднее трёх рабочих
дней после закрытия волны. Дальше пакет только раскладывается по файлам —
никакой логики в писателях нет, иначе цифры в XLSX и в DOCX однажды разойдутся.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

from socmon.analysis.dataset import Dataset
from socmon.analysis.outcomes import CallStatistics
from socmon.analysis.wave import WaveSummary
from socmon.export.codebook import Codebook, CodebookEntry
from socmon.export.tables import CrossTable, FrequencyTable, all_frequencies, crosstab

#: Признаки, в разрезе которых индекс и распределения считаются всегда
#: (ТЗ §12: «расчёт индекса в целом, по территориям, социальным группам»).
DEFAULT_BREAKDOWNS = ("territory", "gender", "age_group")


@dataclass(frozen=True, slots=True)
class WavePackage:
    wave_code: str
    generated_at: datetime
    summary: WaveSummary
    codebook: Codebook
    records: tuple  # type: ignore[type-arg]
    frequencies: tuple[FrequencyTable, ...]
    crosstabs: tuple[CrossTable, ...]
    index_breakdowns: dict[str, list[tuple[str, int, float | None]]]
    call_statistics: CallStatistics | None = None
    methodology_notes: tuple[str, ...] = field(default=())

    @property
    def limitations(self) -> tuple[str, ...]:
        """Методологические ограничения единым списком (ТЗ §13.2).

        Сводятся из всех источников сразу: ограничения весов, ограничения
        индекса, недобор объёма, оговорки по достижимости. Разбросанные по
        разделам, они не читаются; собранные вместе — читаются.
        """
        notes = list(self.summary.notes)
        if self.call_statistics is not None:
            notes.extend(self.call_statistics.notes)
        notes.extend(self.methodology_notes)
        return tuple(notes)


def build(
    summary: WaveSummary,
    codebook: Codebook,
    *,
    wave_code: str,
    call_statistics: CallStatistics | None = None,
    breakdowns: Sequence[str] = DEFAULT_BREAKDOWNS,
    crosstab_target: str = "local_problem",
    methodology_notes: Sequence[str] = (),
    now: datetime | None = None,
) -> WavePackage:
    """Собрать пакет по результатам волны."""
    records: Dataset = summary.records

    crosstabs: list[CrossTable] = []
    target = codebook.by_variable(crosstab_target)
    if target is not None:
        for variable in breakdowns:
            entry = codebook.by_variable(variable)
            if entry is not None and entry.values:
                crosstabs.append(crosstab(records, entry, target))

    index_breakdowns: dict[str, list[tuple[str, int, float | None]]] = {}
    for variable in breakdowns:
        entry = codebook.by_variable(variable)
        if entry is not None and entry.values:
            index_breakdowns[variable] = _index_by(records, entry)

    return WavePackage(
        wave_code=wave_code,
        generated_at=now or datetime.now(UTC),
        summary=summary,
        codebook=codebook,
        records=tuple(records),
        frequencies=tuple(all_frequencies(records, codebook)),
        crosstabs=tuple(crosstabs),
        index_breakdowns=index_breakdowns,
        call_statistics=call_statistics,
        methodology_notes=tuple(methodology_notes),
    )


def _index_by(
    records: Dataset, entry: CodebookEntry
) -> list[tuple[str, int, float | None]]:
    from socmon.analysis.index import compute
    from socmon.export.tables import MIN_CELL_BASE

    rows: list[tuple[str, int, float | None]] = []
    for code in sorted(entry.values):
        group = [r for r in records if r.get(entry.variable) == code]
        label = entry.value_label(code)
        if len(group) < MIN_CELL_BASE:
            # Посчитать индекс по горстке анкет технически можно.
            # Публиковать — нет (ТЗ §5.2).
            rows.append((label, len(group), None))
            continue
        rows.append((label, len(group), compute(group).aggregate))
    return rows
