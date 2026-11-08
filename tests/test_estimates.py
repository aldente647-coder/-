from __future__ import annotations

from collections.abc import Sequence

import pytest

from socmon.analysis.dataset import Record, subgroup
from socmon.analysis.estimates import (
    EstimationError,
    margin_of_error_srs,
    weighted_mean,
    weighted_share,
)
from socmon.core.codes import SpecialCode


def sample(codes: Sequence[int | None], weight: float = 1.0, **strata: str) -> list[Record]:
    return [
        Record(id=str(i), answers={"anxiety": c}, weight=weight, strata=strata)
        for i, c in enumerate(codes)
    ]


def test_sampling_error_matches_the_figure_in_the_tor() -> None:
    """ТЗ §5.2 обещает ≈±2,5 п.п. при n≈1500. Проверяем, что обещание
    арифметически верно — это опорная величина всей отчётности."""
    assert margin_of_error_srs(1500) == pytest.approx(0.0253, abs=0.0002)


def test_weighted_mean_of_flat_weights_is_the_plain_mean() -> None:
    estimate = weighted_mean(sample([2, 4, 6, 8] * 50), "anxiety")
    assert estimate.value == pytest.approx(5.0)
    assert estimate.design_effect == pytest.approx(1.0)


def test_weights_move_the_estimate() -> None:
    records = [
        *sample([10] * 50, weight=1.5),
        *sample([0] * 50, weight=0.5),
    ]
    estimate = weighted_mean(records, "anxiety")
    assert estimate.value == pytest.approx(7.5)


def test_refusals_are_excluded_not_counted_as_zero() -> None:
    """Подставить вместо отказа ноль — значит выдать молчание
    за содержательный ответ (ТЗ §12)."""
    codes: Sequence[int | None] = [8] * 100 + [int(SpecialCode.REFUSED)] * 50
    estimate = weighted_mean(sample(codes), "anxiety")
    assert estimate.value == pytest.approx(8.0)
    assert estimate.n == 100
    assert estimate.base_n == 150


def test_high_refusal_rate_is_flagged() -> None:
    codes: Sequence[int | None] = [8] * 40 + [int(SpecialCode.DONT_KNOW)] * 110
    estimate = weighted_mean(sample(codes), "anxiety")
    assert any("отказались или затруднились" in n for n in estimate.notes)


def test_all_refusals_is_an_error_not_a_zero() -> None:
    codes: Sequence[int | None] = [int(SpecialCode.UNRECOGNIZED)] * 20
    with pytest.raises(EstimationError, match="ни одного содержательного"):
        weighted_mean(sample(codes), "anxiety")


# ------------------------------------------------------- доверительные интервалы


def test_confidence_interval_widens_with_the_design_effect() -> None:
    """Ключевое требование ТЗ §5.2: интервалы считаются с учётом весов,
    а не по формуле простого случайного отбора."""
    values: list[int | None] = [0, 10] * 100
    flat = weighted_mean(sample(values), "anxiety")

    # Вес меняется парами, а не поэлементно: так среднее и дисперсия остаются
    # прежними, и на ширину интервала влияет только неравномерность весов.
    # Иначе тест мерил бы не эффект дизайна, а сдвинувшееся среднее.
    uneven = [
        Record(id=str(i), answers={"anxiety": c}, weight=1.7 if (i // 2) % 2 == 0 else 0.3)
        for i, c in enumerate(values)
    ]
    weighted = weighted_mean(uneven, "anxiety")

    assert weighted.value == pytest.approx(flat.value)

    assert weighted.design_effect > flat.design_effect
    assert weighted.effective_n < flat.effective_n
    assert weighted.margin_of_error > flat.margin_of_error


def test_interval_brackets_the_point_estimate() -> None:
    estimate = weighted_mean(sample([3, 5, 7] * 50), "anxiety")
    assert estimate.ci_low < estimate.value < estimate.ci_high


def test_share_interval_never_leaves_zero_one() -> None:
    """Доля 98% с наивным интервалом ушла бы за 100% — обрезаем."""
    records = sample([10] * 98 + [0] * 2)
    estimate = weighted_share(records, lambda r: r.get("anxiety") == 10)
    assert 0.0 <= estimate.ci_low <= estimate.ci_high <= 1.0


def test_share_ignores_unknown_values() -> None:
    """Предикат вправе сказать «не знаю» — и такое наблюдение выпадает
    из базы, а не засчитывается в отрицательный ответ."""
    records = sample([10] * 50 + [0] * 50 + [None] * 100)

    def high_anxiety(record: Record) -> bool | None:
        code = record.get("anxiety")
        return None if code is None else code >= 8

    estimate = weighted_share(records, high_anxiety)
    assert estimate.value == pytest.approx(0.5)
    assert estimate.n == 100
    assert estimate.base_n == 200


# ------------------------------------------------------------- малые подгруппы


def test_small_subgroup_is_marked_unreliable() -> None:
    """ТЗ §5.2: по малым территориям указывать объём подвыборки,
    интервал и ограничение на интерпретацию."""
    estimate = weighted_mean(sample([5] * 30), "anxiety")
    assert estimate.reliable is False
    assert estimate.n == 30
    assert any("содержательные выводы по ней делать нельзя" in n for n in estimate.notes)


def test_large_subgroup_is_reliable() -> None:
    assert weighted_mean(sample([5] * 300), "anxiety").reliable is True


def test_subgroup_selects_by_strata() -> None:
    records = [
        *sample([8] * 150, territory="north"),
        *sample([3] * 150, territory="south"),
    ]
    north = subgroup(records, territory="north")
    assert len(north) == 150
    assert weighted_mean(north, "anxiety").value == pytest.approx(8.0)


def test_formatted_output_carries_the_interval_and_the_base() -> None:
    text = weighted_mean(sample([5] * 200), "anxiety").formatted()
    assert "95% ДИ" in text
    assert "n=200" in text
