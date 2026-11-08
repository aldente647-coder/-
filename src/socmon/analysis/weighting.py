"""Статистическое взвешивание массива волны (ТЗ §5.4).

Метод — итеративное пропорциональное подгонное взвешивание (raking, IPF):
веса подгоняются так, чтобы взвешенные распределения массива сошлись
с эталонными распределениями Заказчика по каждому признаку в отдельности.
Совместного распределения признаков raking не требует — что важно, потому что
Заказчик обычно даёт именно одномерные эталоны.

Граница, которую нельзя переходить: §5.1 запрещает подгонять результаты.
Взвешивание правит структуру выборки по заранее согласованным признакам —
и только по ним. Ни один содержательный ответ в расчёт весов не входит,
и это видно из сигнатуры: сюда передаются `strata`, а не `answers`.

По каждой волне отдаётся полный диагностический набор, перечисленный
в §5.4: взвешенное и эталонное распределения, минимальный и максимальный
вес, доля наблюдений с экстремальными весами.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from socmon.analysis.dataset import Dataset

#: Признаки, без которых взвешивание не имеет смысла (ТЗ §5.4: «базовыми
#: параметрами структуры выборки должны являться (min) пол, возраст, территория»).
BASE_VARIABLES = ("gender", "age_group", "territory")


class WeightingError(ValueError):
    """Взвешивание невозможно. Лучше отказ, чем молча кривые веса."""


@dataclass(frozen=True, slots=True)
class Margin:
    """Эталонное распределение по одному признаку.

    Доли нормируются на входе: Заказчик одинаково законно может прислать
    и проценты, и абсолютную численность.
    """

    variable: str
    targets: Mapping[str, float]

    def __post_init__(self) -> None:
        if not self.targets:
            raise WeightingError(f"{self.variable}: пустое эталонное распределение")
        if any(v < 0 for v in self.targets.values()):
            raise WeightingError(f"{self.variable}: отрицательная доля в эталоне")
        total = sum(self.targets.values())
        if total <= 0:
            raise WeightingError(f"{self.variable}: сумма эталона равна нулю")
        object.__setattr__(
            self, "targets", {k: v / total for k, v in self.targets.items()}
        )

    @property
    def categories(self) -> frozenset[str]:
        return frozenset(self.targets)


@dataclass(frozen=True, slots=True)
class RakingPolicy:
    max_iterations: int = 200
    tolerance: float = 1e-7
    """Сходимость по максимальному отклонению взвешенной доли от эталонной."""

    max_weight_ratio: float | None = 5.0
    """Потолок и пол веса относительно среднего.

    Обрезка — компромисс: без неё одно редкое наблюдение получает вес в
    десятки единиц и начинает единолично двигать оценку по подгруппе.
    С ней взвешенное распределение чуть не дотягивает до эталона, и это
    отклонение обязано быть видно в диагностике, а не замолчано.
    """

    require_base_variables: bool = True
    """ТЗ §5.4: остальные признаки допускаются только при надёжных эталонах
    и методологическом обосновании — но пол, возраст и территория обязательны."""


@dataclass(frozen=True, slots=True)
class WeightDiagnostics:
    """Всё, что ТЗ §5.4 требует предоставлять по каждой волне."""

    n: int
    min_weight: float
    max_weight: float
    mean_weight: float
    extreme_share: float
    """Доля наблюдений, упёршихся в потолок или пол веса."""

    design_effect: float
    """Эффект дизайна по Кишу. Показывает, во сколько раз взвешивание
    раздуло дисперсию оценок против простого случайного отбора."""

    effective_n: float
    """Эффективный объём выборки. Именно он, а не n, определяет реальную
    ширину доверительных интервалов (ТЗ §5.2)."""

    iterations: int
    converged: bool
    max_margin_deviation: float
    achieved: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    reference: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    unweighted: Mapping[str, Mapping[str, float]] = field(default_factory=dict)

    @property
    def warnings(self) -> tuple[str, ...]:
        """Ограничения интерпретации — в отчёт по волне (ТЗ §5.4)."""
        notes: list[str] = []
        if not self.converged:
            notes.append(
                f"взвешивание не сошлось за {self.iterations} итераций; "
                f"максимальное отклонение от эталона {self.max_margin_deviation:.4f}"
            )
        if self.extreme_share > 0:
            # Даже одно наблюдение на границе означает, что до эталона
            # не дотянули: его вклад ограничен искусственно, а не данными.
            tail = (
                " — структура достигнутой выборки заметно расходится с эталонной"
                if self.extreme_share > 0.05
                else ""
            )
            notes.append(
                f"экстремальные веса у {self.extreme_share:.1%} наблюдений: "
                f"их вклад ограничен обрезкой{tail}"
            )
        if self.design_effect > 2.0:
            notes.append(
                f"эффект дизайна {self.design_effect:.2f}: доверительные интервалы "
                f"шире, чем при простом случайном отборе, в {self.design_effect ** 0.5:.2f} раза"
            )
        return tuple(notes)


@dataclass(frozen=True, slots=True)
class RakingResult:
    weights: tuple[float, ...]
    diagnostics: WeightDiagnostics


DEFAULT_POLICY = RakingPolicy()


def rake(
    records: Dataset,
    margins: Sequence[Margin],
    policy: RakingPolicy = DEFAULT_POLICY,
) -> RakingResult:
    """Рассчитать веса методом IPF.

    Веса нормируются так, что их сумма равна числу наблюдений: средний вес
    единица, и невзвешенные частоты сравнимы со взвешенными напрямую.
    """
    if not records:
        raise WeightingError("пустой массив")
    if not margins:
        raise WeightingError("не заданы эталонные распределения")

    _validate(records, margins, policy)

    n = len(records)
    weights = [1.0] * n
    lo, hi = _bounds(policy)

    converged = False
    iteration = 0
    deviation = float("inf")

    while iteration < policy.max_iterations:
        iteration += 1
        for margin in margins:
            _fit_margin(records, weights, margin)
        _normalize(weights, n)
        if lo is not None and hi is not None:
            _clamp(weights, lo, hi, n)
        deviation = _max_deviation(records, weights, margins)
        if deviation <= policy.tolerance:
            converged = True
            break

    return RakingResult(
        weights=tuple(weights),
        diagnostics=_diagnose(
            records, weights, margins, iteration, converged, deviation, policy
        ),
    )


# --------------------------------------------------------------------- шаги


def _validate(records: Dataset, margins: Sequence[Margin], policy: RakingPolicy) -> None:
    names = [m.variable for m in margins]
    if len(names) != len(set(names)):
        raise WeightingError("признак задан эталоном дважды")

    if policy.require_base_variables:
        missing = set(BASE_VARIABLES) - set(names)
        if missing:
            raise WeightingError(
                "нет эталонов по базовым признакам структуры выборки: "
                + ", ".join(sorted(missing))
            )

    for margin in margins:
        observed: set[str] = set()
        for index, record in enumerate(records):
            value = record.strata.get(margin.variable)
            if value is None:
                raise WeightingError(
                    f"наблюдение {record.id or index}: не заполнен признак "
                    f"«{margin.variable}». Наблюдения с пропусками по признакам "
                    f"взвешивания исключаются на этапе валидности, а не молча здесь"
                )
            if value not in margin.targets:
                raise WeightingError(
                    f"наблюдение {record.id or index}: категория «{value}» признака "
                    f"«{margin.variable}» отсутствует в эталонном распределении"
                )
            observed.add(value)

        empty = margin.categories - observed
        if empty:
            raise WeightingError(
                f"{margin.variable}: в массиве нет наблюдений категорий "
                f"{sorted(empty)}, довесить их до эталона невозможно"
            )


def _fit_margin(records: Dataset, weights: list[float], margin: Margin) -> None:
    achieved = _weighted_shares(records, weights, margin.variable)
    factors = {
        category: margin.targets[category] / share
        for category, share in achieved.items()
        if share > 0
    }
    for index, record in enumerate(records):
        weights[index] *= factors.get(record.strata[margin.variable], 1.0)


def _clamp(weights: list[float], lo: float, hi: float, n: int) -> None:
    """Прижать веса к границам и вернуть сумму к n.

    В один проход это не делается: обрезав верхние веса, мы убираем часть
    массы, нормировка поднимает все веса обратно, и часть снова уходит за
    потолок. Поэтому обрезка и нормировка чередуются до стабилизации.

    Если границы недостижимы (эталон требует довесить группу сильнее, чем
    разрешает потолок), цикл выходит по лимиту, и веса остаются в границах,
    а недотяг до эталона показывает `max_margin_deviation`.
    """
    epsilon = 1e-12
    for _ in range(50):
        clamped = [min(max(w, lo), hi) for w in weights]
        weights[:] = clamped
        total = sum(weights)
        if abs(total - n) <= epsilon * n:
            return
        _normalize(weights, n)
        if all(lo - epsilon <= w <= hi + epsilon for w in weights):
            return
    weights[:] = [min(max(w, lo), hi) for w in weights]


def _normalize(weights: list[float], n: int) -> None:
    total = sum(weights)
    if total <= 0:
        raise WeightingError("сумма весов обнулилась")
    scale = n / total
    for index in range(n):
        weights[index] *= scale


def _bounds(policy: RakingPolicy) -> tuple[float | None, float | None]:
    if policy.max_weight_ratio is None:
        return None, None
    if policy.max_weight_ratio <= 1:
        raise WeightingError("потолок веса должен быть больше единицы")
    return 1.0 / policy.max_weight_ratio, policy.max_weight_ratio


def _weighted_shares(
    records: Dataset, weights: Sequence[float], variable: str
) -> dict[str, float]:
    totals: defaultdict[str, float] = defaultdict(float)
    for record, weight in zip(records, weights, strict=True):
        totals[record.strata[variable]] += weight
    grand = sum(totals.values())
    return {category: value / grand for category, value in totals.items()}


def _unweighted_shares(records: Dataset, variable: str) -> dict[str, float]:
    counts = Counter(record.strata[variable] for record in records)
    n = sum(counts.values())
    return {category: value / n for category, value in counts.items()}


def _max_deviation(
    records: Dataset, weights: Sequence[float], margins: Sequence[Margin]
) -> float:
    worst = 0.0
    for margin in margins:
        achieved = _weighted_shares(records, weights, margin.variable)
        for category, target in margin.targets.items():
            worst = max(worst, abs(achieved.get(category, 0.0) - target))
    return worst


def _diagnose(
    records: Dataset,
    weights: Sequence[float],
    margins: Sequence[Margin],
    iterations: int,
    converged: bool,
    deviation: float,
    policy: RakingPolicy,
) -> WeightDiagnostics:
    n = len(weights)
    lo, hi = _bounds(policy)
    extreme = 0
    if lo is not None and hi is not None:
        epsilon = 1e-9
        extreme = sum(1 for w in weights if w <= lo + epsilon or w >= hi - epsilon)

    return WeightDiagnostics(
        n=n,
        min_weight=min(weights),
        max_weight=max(weights),
        mean_weight=sum(weights) / n,
        extreme_share=extreme / n,
        design_effect=design_effect(weights),
        effective_n=effective_sample_size(weights),
        iterations=iterations,
        converged=converged,
        max_margin_deviation=deviation,
        achieved={m.variable: _weighted_shares(records, weights, m.variable) for m in margins},
        reference={m.variable: dict(m.targets) for m in margins},
        unweighted={m.variable: _unweighted_shares(records, m.variable) for m in margins},
    )


# ------------------------------------------------------------- характеристики


def design_effect(weights: Sequence[float]) -> float:
    """Эффект дизайна по Кишу: n·Σw² / (Σw)².

    Единица означает, что взвешивание ничего не стоило по точности. Всё, что
    выше, — плата за приведение структуры к эталону, и её нужно учитывать
    в доверительных интервалах, иначе они окажутся уже настоящих.
    """
    total = sum(weights)
    squares = sum(w * w for w in weights)
    if total <= 0 or squares <= 0:
        raise WeightingError("веса вырождены")
    return len(weights) * squares / (total * total)


def effective_sample_size(weights: Sequence[float]) -> float:
    """(Σw)² / Σw² — объём выборки, эквивалентный по точности невзвешенной."""
    total = sum(weights)
    squares = sum(w * w for w in weights)
    if squares <= 0:
        raise WeightingError("веса вырождены")
    return total * total / squares
