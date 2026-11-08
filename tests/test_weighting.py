from __future__ import annotations

import pytest

from socmon.analysis.dataset import Record
from socmon.analysis.weighting import (
    Margin,
    RakingPolicy,
    WeightingError,
    design_effect,
    effective_sample_size,
    rake,
)

BASE_MARGINS = [
    Margin("gender", {"m": 0.46, "f": 0.54}),
    Margin("age_group", {"18-34": 0.30, "35-54": 0.36, "55+": 0.34}),
    Margin("territory", {"north": 0.4, "south": 0.6}),
]


def population(counts: dict[tuple[str, str, str], int]) -> list[Record]:
    records: list[Record] = []
    for (gender, age, territory), how_many in counts.items():
        for _ in range(how_many):
            records.append(Record(
                id=f"{gender}{age}{territory}{len(records)}",
                strata={"gender": gender, "age_group": age, "territory": territory},
            ))
    return records


def balanced(n_each: int = 40) -> list[Record]:
    return population({
        (g, a, t): n_each
        for g in ("m", "f")
        for a in ("18-34", "35-54", "55+")
        for t in ("north", "south")
    })


def shares(records: list[Record], weights: tuple[float, ...], variable: str) -> dict[str, float]:
    totals: dict[str, float] = {}
    for record, weight in zip(records, weights, strict=True):
        totals[record.strata[variable]] = totals.get(record.strata[variable], 0.0) + weight
    grand = sum(totals.values())
    return {k: v / grand for k, v in totals.items()}


# ------------------------------------------------------------------ сходимость


def test_raking_reaches_the_reference_distribution() -> None:
    records = balanced()
    result = rake(records, BASE_MARGINS)

    assert result.diagnostics.converged
    for margin in BASE_MARGINS:
        achieved = shares(records, result.weights, margin.variable)
        for category, target in margin.targets.items():
            assert achieved[category] == pytest.approx(target, abs=1e-4)


def test_weights_average_to_one() -> None:
    """Сумма весов равна числу наблюдений: взвешенные частоты сравнимы
    с невзвешенными напрямую."""
    records = balanced()
    result = rake(records, BASE_MARGINS)
    assert sum(result.weights) == pytest.approx(len(records))
    assert result.diagnostics.mean_weight == pytest.approx(1.0)


def test_already_representative_sample_gets_flat_weights() -> None:
    """Если структура уже совпадает с эталоном, взвешивание не должно
    ничего выдумывать."""
    records = balanced()
    margins = [
        Margin("gender", {"m": 0.5, "f": 0.5}),
        Margin("age_group", {"18-34": 1 / 3, "35-54": 1 / 3, "55+": 1 / 3}),
        Margin("territory", {"north": 0.5, "south": 0.5}),
    ]
    result = rake(records, margins)
    assert all(w == pytest.approx(1.0) for w in result.weights)
    assert result.diagnostics.design_effect == pytest.approx(1.0)


def test_underrepresented_group_is_weighted_up() -> None:
    records = population({
        ("m", "18-34", "north"): 10,   # молодых мужчин недобрали
        ("f", "35-54", "south"): 190,
    })
    margins = [
        Margin("gender", {"m": 0.5, "f": 0.5}),
        Margin("age_group", {"18-34": 0.5, "35-54": 0.5}),
        Margin("territory", {"north": 0.5, "south": 0.5}),
    ]
    result = rake(records, margins, RakingPolicy(max_weight_ratio=None))
    scarce = [w for r, w in zip(records, result.weights, strict=True)
              if r.strata["gender"] == "m"]
    plenty = [w for r, w in zip(records, result.weights, strict=True)
              if r.strata["gender"] == "f"]
    assert min(scarce) > max(plenty)


# --------------------------------------------------------------------- отказы


def test_missing_base_variable_is_refused() -> None:
    """ТЗ §5.4: пол, возраст и территория — базовые параметры структуры."""
    with pytest.raises(WeightingError, match="базовым признакам"):
        rake(balanced(), [Margin("gender", {"m": 0.5, "f": 0.5})])


def test_record_without_a_stratum_is_refused_loudly() -> None:
    """Молча выбросить наблюдение нельзя: это уже отбор, а §5.1 его запрещает."""
    records = balanced()
    records.append(Record(id="broken", strata={"gender": "m", "age_group": "18-34"}))
    with pytest.raises(WeightingError, match="не заполнен признак"):
        rake(records, BASE_MARGINS)


def test_unknown_category_is_refused() -> None:
    records = balanced()
    records.append(Record(
        id="alien",
        strata={"gender": "x", "age_group": "18-34", "territory": "north"},
    ))
    with pytest.raises(WeightingError, match="отсутствует в эталонном"):
        rake(records, BASE_MARGINS)


def test_category_absent_from_the_sample_is_refused() -> None:
    """Довесить до эталона категорию, которой в массиве нет, невозможно —
    и делать вид, что получилось, нельзя."""
    records = balanced()
    margins = [*BASE_MARGINS[:2], Margin("territory", {"north": 0.3, "south": 0.3, "east": 0.4})]
    with pytest.raises(WeightingError, match="довесить"):
        rake(records, margins)


def test_empty_reference_is_refused() -> None:
    with pytest.raises(WeightingError, match="пустое эталонное"):
        Margin("gender", {})


def test_reference_is_normalised_from_any_units() -> None:
    """Заказчик одинаково законно присылает и проценты, и численность."""
    percents = Margin("gender", {"m": 46.0, "f": 54.0})
    counts = Margin("gender", {"m": 460_000, "f": 540_000})
    assert dict(percents.targets) == pytest.approx(dict(counts.targets))
    assert sum(percents.targets.values()) == pytest.approx(1.0)


# ---------------------------------------------------------------- диагностика


def test_extreme_weights_are_trimmed_and_reported() -> None:
    """ТЗ §5.4 требует показывать долю наблюдений с экстремальными весами."""
    records = population({
        ("m", "18-34", "north"): 4,
        ("f", "55+", "south"): 396,
    })
    margins = [
        Margin("gender", {"m": 0.5, "f": 0.5}),
        Margin("age_group", {"18-34": 0.5, "55+": 0.5}),
        Margin("territory", {"north": 0.5, "south": 0.5}),
    ]
    result = rake(records, margins, RakingPolicy(max_weight_ratio=5.0))

    assert result.diagnostics.max_weight <= 5.0 + 1e-6
    assert result.diagnostics.min_weight >= 0.2 - 1e-6
    assert result.diagnostics.extreme_share > 0
    assert any("экстремальные веса" in w for w in result.diagnostics.warnings)
    # Довесить 1% выборки до 50% потолок в пятикратный вес не позволяет —
    # и делать вид, что позволил, система не должна.
    assert not result.diagnostics.converged


def test_trimming_is_honest_about_not_reaching_the_reference() -> None:
    """Обрезка — компромисс, и недотяг до эталона обязан быть виден."""
    records = population({("m", "18-34", "north"): 4, ("f", "55+", "south"): 396})
    margins = [
        Margin("gender", {"m": 0.5, "f": 0.5}),
        Margin("age_group", {"18-34": 0.5, "55+": 0.5}),
        Margin("territory", {"north": 0.5, "south": 0.5}),
    ]
    result = rake(records, margins, RakingPolicy(max_weight_ratio=2.0, max_iterations=50))
    assert not result.diagnostics.converged
    assert result.diagnostics.max_margin_deviation > 0.01
    assert any("не сошлось" in w for w in result.diagnostics.warnings)


def test_diagnostics_carry_all_three_distributions() -> None:
    """ТЗ §5.4: взвешенное распределение, эталонное и алгоритм расчёта."""
    records = balanced()
    diagnostics = rake(records, BASE_MARGINS).diagnostics

    assert set(diagnostics.achieved) == {"gender", "age_group", "territory"}
    assert diagnostics.reference["gender"]["m"] == pytest.approx(0.46)
    assert diagnostics.unweighted["gender"]["m"] == pytest.approx(0.5)
    assert diagnostics.achieved["gender"]["m"] == pytest.approx(0.46, abs=1e-4)


# ------------------------------------------------------- эффект дизайна


def test_design_effect_of_flat_weights_is_one() -> None:
    assert design_effect([1.0] * 100) == pytest.approx(1.0)
    assert effective_sample_size([1.0] * 100) == pytest.approx(100.0)


def test_uneven_weights_cost_precision() -> None:
    """Взвешивание не бесплатно: эффективный объём выборки всегда меньше
    фактического, и именно он определяет ширину интервалов."""
    weights = [0.2] * 50 + [1.8] * 50
    assert design_effect(weights) > 1.0
    assert effective_sample_size(weights) < 100


def test_design_effect_matches_its_definition() -> None:
    weights = [0.5, 1.0, 1.5, 2.0]
    n, total, squares = 4, sum(weights), sum(w * w for w in weights)
    assert design_effect(weights) == pytest.approx(n * squares / total**2)


def test_empty_dataset_is_refused() -> None:
    with pytest.raises(WeightingError, match="пустой массив"):
        rake([], BASE_MARGINS)
