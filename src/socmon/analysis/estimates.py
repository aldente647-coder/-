"""Оценки с доверительными интервалами (ТЗ §5.2).

Интервалы считаются с учётом весов и эффекта дизайна, а не по формуле для
простого случайного отбора. Разница не косметическая: при эффекте дизайна 1,5
настоящий интервал шире наивного примерно на четверть, и оценка «различие
значимо» может смениться на противоположную.

Отдельная обязанность модуля — не давать молча опубликовать оценку по горстке
наблюдений. По малым территориям и подгруппам ТЗ §5.2 требует указывать
фактический объём подвыборки, доверительный интервал и ограничение
на интерпретацию, поэтому каждая оценка несёт их в себе.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from socmon.analysis.dataset import Dataset, Record
from socmon.analysis.weighting import design_effect, effective_sample_size

Z_95 = 1.959963985
"""Квантиль стандартного нормального распределения для 95% (ТЗ §5.2)."""

MIN_RELIABLE_N = 100
"""Ниже этого объёма подвыборки оценка сопровождается ограничением
на интерпретацию. Порог методологический и подлежит согласованию
с Заказчиком, поэтому вынесен константой, а не зашит в формулу."""


class EstimationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Estimate:
    """Оценка со всем, что нужно, чтобы её честно опубликовать."""

    value: float
    ci_low: float
    ci_high: float
    standard_error: float
    n: int
    """Фактический объём подвыборки — тот, по которому реально считали."""

    effective_n: float
    design_effect: float
    base_n: int
    """Объём подвыборки до отсева пропусков. Разница с `n` — доля тех,
    кто на этот вопрос не ответил."""

    reliable: bool
    notes: tuple[str, ...] = ()

    @property
    def margin_of_error(self) -> float:
        """Половина ширины интервала — «плюс-минус» в процентных пунктах."""
        return (self.ci_high - self.ci_low) / 2

    @property
    def response_rate(self) -> float:
        return self.n / self.base_n if self.base_n else 0.0

    def formatted(self, *, percent: bool = False) -> str:
        scale = 100 if percent else 1
        unit = "%" if percent else ""
        return (
            f"{self.value * scale:.1f}{unit} "
            f"(95% ДИ {self.ci_low * scale:.1f}–{self.ci_high * scale:.1f}{unit}, n={self.n})"
        )


def weighted_share(
    records: Dataset,
    predicate: Callable[[Record], bool | None],
) -> Estimate:
    """Взвешенная доля наблюдений, удовлетворяющих условию.

    Предикат вправе вернуть None — «признак неизвестен». Такие наблюдения
    выпадают из базы расчёта, а не засчитываются в отрицательный ответ:
    иначе отказ отвечать превратился бы в содержательное «нет».
    """
    kept: list[tuple[float, float]] = []
    for record in records:
        outcome = predicate(record)
        if outcome is None:
            continue
        kept.append((1.0 if outcome else 0.0, record.weight))

    if not kept:
        raise EstimationError("ни одного наблюдения с известным значением признака")

    values = [v for v, _ in kept]
    weights = [w for _, w in kept]
    share = _weighted_mean(values, weights)
    variance = share * (1 - share)
    return _assemble(share, variance, weights, base_n=len(records), clamp=(0.0, 1.0))


def weighted_mean(
    records: Dataset,
    core_key: str,
    *,
    valid: Callable[[int], bool] | None = None,
) -> Estimate:
    """Взвешенное среднее по числовому показателю ядра.

    По умолчанию содержательными считаются коды 0-94: служебная зона
    (отказ, затруднение, нераспознанная речь) в среднее не входит.
    Подставлять вместо них середину шкалы нельзя без явного согласования —
    ТЗ §12 запрещает заменять пропуски средними значениями.
    """
    is_valid = valid or (lambda code: 0 <= code <= 94)

    values: list[float] = []
    weights: list[float] = []
    for record in records:
        code = record.get(core_key)
        if code is None or not is_valid(code):
            continue
        values.append(float(code))
        weights.append(record.weight)

    if not values:
        raise EstimationError(f"{core_key}: ни одного содержательного ответа")

    mean = _weighted_mean(values, weights)
    variance = _weighted_variance(values, weights, mean)
    return _assemble(mean, variance, weights, base_n=len(records))


def _assemble(
    point: float,
    variance: float,
    weights: Sequence[float],
    *,
    base_n: int,
    clamp: tuple[float, float] | None = None,
) -> Estimate:
    n = len(weights)
    deff = design_effect(weights) if n > 1 else 1.0
    n_eff = effective_sample_size(weights) if n > 1 else float(n)

    # Стандартная ошибка через эффективный объём: se = √(σ²·deff / n) = √(σ²/n_eff).
    standard_error = math.sqrt(variance / n_eff) if n_eff > 0 else float("inf")
    low, high = point - Z_95 * standard_error, point + Z_95 * standard_error
    if clamp is not None:
        low, high = max(clamp[0], low), min(clamp[1], high)

    notes: list[str] = []
    reliable = True
    if n < MIN_RELIABLE_N:
        reliable = False
        notes.append(
            f"объём подвыборки {n} меньше {MIN_RELIABLE_N}: оценка приводится "
            f"справочно, содержательные выводы по ней делать нельзя"
        )
    if base_n and n / base_n < 0.5:
        notes.append(
            f"содержательный ответ дали {n} из {base_n} — больше половины "
            f"подвыборки отказались или затруднились"
        )
    if deff > 2.0:
        notes.append(f"эффект дизайна {deff:.2f}: интервал шире наивного")

    return Estimate(
        value=point,
        ci_low=low,
        ci_high=high,
        standard_error=standard_error,
        n=n,
        effective_n=n_eff,
        design_effect=deff,
        base_n=base_n,
        reliable=reliable,
        notes=tuple(notes),
    )


def _weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    total = sum(weights)
    if total <= 0:
        raise EstimationError("сумма весов не положительна")
    return sum(v * w for v, w in zip(values, weights, strict=True)) / total


def _weighted_variance(
    values: Sequence[float], weights: Sequence[float], mean: float
) -> float:
    total = sum(weights)
    if total <= 0:
        raise EstimationError("сумма весов не положительна")
    return sum(w * (v - mean) ** 2 for v, w in zip(values, weights, strict=True)) / total


def margin_of_error_srs(n: int, share: float = 0.5) -> float:
    """Ошибка выборки при простом случайном отборе — ориентир из ТЗ §5.2.

    При n=1500 и доле 0,5 даёт те самые ±2,5 процентного пункта. Это
    справочная величина: фактические интервалы считаются с учётом весов
    и дизайна и почти всегда шире.
    """
    if n <= 0:
        raise EstimationError("объём выборки должен быть положительным")
    return Z_95 * math.sqrt(share * (1 - share) / n)
