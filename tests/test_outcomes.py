from __future__ import annotations

import pytest

from socmon.analysis.outcomes import CallStatistics, Dispositions, OutcomeError, response_rates
from socmon.core.codes import AaporCode as A


def wave(**counts: int) -> Dispositions:
    codes: list[A] = []
    for name, number in counts.items():
        codes.extend([getattr(A, name.upper())] * number)
    return Dispositions.from_codes(codes)


def test_dispositions_group_by_aapor_family() -> None:
    d = wave(complete=100, partial=10, refusal=50, break_off=5,
             non_contact=20, unknown_eligibility=300, not_eligible=15,
             out_of_sample=40)
    assert d.interviews == 100
    assert d.refusals == 55  # отказ и прерывание — одна группа
    assert d.eligible == 100 + 10 + 55 + 20
    assert d.total == d.eligible + 300 + 15


def test_dead_numbers_are_not_part_of_the_sample() -> None:
    """Недействующий номер — это качество базы, а не отказ респондента,
    и в знаменатель достижимости он не входит."""
    with_dead = wave(complete=100, refusal=100, unknown_eligibility=100, out_of_sample=500)
    without = wave(complete=100, refusal=100, unknown_eligibility=100)
    assert response_rates(with_dead).rr3 == pytest.approx(response_rates(without).rr3)


def test_rr5_is_always_the_flattering_one() -> None:
    """RR5 исключает номера неизвестной пригодности и потому всегда выше.
    Публиковать его как «долю ответивших» — лукавство, ради которого
    и разведены шесть коэффициентов."""
    r = response_rates(wave(complete=100, refusal=200, non_contact=100,
                            unknown_eligibility=5000, not_eligible=50))
    assert r.rr5 > r.rr3 >= r.rr1
    assert r.rr2 >= r.rr1
    assert r.rr6 >= r.rr5


def test_rr1_matches_its_definition() -> None:
    d = wave(complete=100, partial=20, refusal=200, non_contact=80,
             other_eligible=10, unknown_eligibility=1000, not_eligible=90)
    r = response_rates(d)
    denominator = (100 + 20) + (200 + 80 + 10) + 1000
    assert r.rr1 == pytest.approx(100 / denominator)
    assert r.rr2 == pytest.approx(120 / denominator)


def test_rr3_uses_the_estimated_eligibility() -> None:
    d = wave(complete=100, refusal=200, unknown_eligibility=1000, not_eligible=100)
    r = response_rates(d)
    e = 300 / 400
    assert r.eligibility_rate == pytest.approx(e)
    assert r.rr3 == pytest.approx(100 / (300 + e * 1000))


def test_cooperation_counts_only_those_actually_reached() -> None:
    r = response_rates(wave(complete=100, refusal=100, non_contact=800,
                            unknown_eligibility=100, not_eligible=10))
    assert r.cooperation1 == pytest.approx(100 / 200)
    assert r.contact1 < r.cooperation1


def test_realistic_robo_survey_lands_around_two_percent() -> None:
    """Опорная прикидка проекта: 1,5-3% валидных на набранный номер.
    У ФОМ на мобильных с живым интервьюером RR3 ≈ 13%."""
    d = wave(complete=1700, partial=200, refusal=4000, non_contact=1000,
             unknown_eligibility=80_000, not_eligible=300, out_of_sample=2000)
    assert 0.015 <= response_rates(d).rr3 <= 0.03


def test_eligibility_cannot_be_estimated_without_data() -> None:
    with pytest.raises(OutcomeError, match="не установлена пригодность"):
        _ = wave(unknown_eligibility=100).eligibility_rate


def test_empty_wave_is_refused() -> None:
    with pytest.raises(OutcomeError, match="пустой массив"):
        response_rates(Dispositions())


def test_mostly_unknown_eligibility_is_flagged() -> None:
    d = wave(complete=100, refusal=100, unknown_eligibility=50_000, not_eligible=10)
    statistics = CallStatistics(
        numbers_dialled=d.total, attempts_made=d.total * 3,
        dispositions=d, rates=response_rates(d),
        valid_interviews=100, invalid_interviews=20, exclusion_reasons={},
    )
    assert any("пригодность установить не удалось" in n for n in statistics.notes)
