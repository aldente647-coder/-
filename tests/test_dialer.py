"""Диалер: темп набора, карусель номеров, предохранитель."""

from __future__ import annotations

import uuid

import pytest

from socmon.core.codes import Disposition
from socmon.runtime.dialer import (
    BreakerState,
    Dialer,
    DialerPolicy,
    RoundRobinCallerIds,
)
from socmon.telephony.provider import TelephonyError
from tests.fakes import FakeClock, FakeContactSource, FakeProvider

WAVE = uuid.uuid4()


def build(
    provider: FakeProvider,
    source: FakeContactSource,
    *,
    policy: DialerPolicy | None = None,
    numbers: list[str] | None = None,
    clock: FakeClock | None = None,
) -> tuple[Dialer, list[uuid.UUID]]:
    started: list[uuid.UUID] = []
    dialer = Dialer(
        provider, source, RoundRobinCallerIds(numbers or ["+74951110001", "+74951110002"]),
        wave_id=WAVE,
        on_originated=lambda attempt, handle: started.append(attempt),
        policy=policy or DialerPolicy(calls_per_second=100, burst=100),
        clock=clock or FakeClock(step_sec=0.0),
    )
    return dialer, started


# ------------------------------------------------------------------- базовый ход


async def test_tick_dials_the_claimed_contacts() -> None:
    provider, source = FakeProvider(), FakeContactSource()
    source.fill(5)
    dialer, started = build(provider, source)

    report = await dialer.tick()
    assert report.originated == 5
    assert len(provider.originated) == 5
    assert len(started) == 5
    assert dialer.active == 5


async def test_attempt_id_travels_to_the_engine() -> None:
    """По нему события сшиваются с базой — без него звонок анонимен."""
    provider, source = FakeProvider(), FakeContactSource()
    source.fill(1)
    dialer, _ = build(provider, source)
    await dialer.tick()

    claimed = provider.originated[0]
    assert uuid.UUID(claimed.attempt_id)
    assert claimed.destination.startswith("+7949")


async def test_concurrency_ceiling_is_respected() -> None:
    provider, source = FakeProvider(), FakeContactSource()
    source.fill(50)
    dialer, _ = build(provider, source, policy=DialerPolicy(
        max_concurrent=10, calls_per_second=100, burst=100
    ))

    assert (await dialer.tick()).originated == 10
    assert (await dialer.tick()).reason == "потолок одновременных звонков"


async def test_finished_call_frees_a_slot() -> None:
    provider, source = FakeProvider(), FakeContactSource()
    source.fill(12)
    dialer, started = build(provider, source, policy=DialerPolicy(
        max_concurrent=10, calls_per_second=100, burst=100
    ))
    await dialer.tick()
    dialer.release(started[0], Disposition.NO_ANSWER)

    assert (await dialer.tick()).originated == 1


# --------------------------------------------------------------------- темп


async def test_pacing_limits_the_burst() -> None:
    """На полной скорости можно упереться в ограничения оператора связи
    и сорвать волну."""
    provider, source = FakeProvider(), FakeContactSource()
    source.fill(100)
    clock = FakeClock(step_sec=0.0)
    dialer, _ = build(provider, source, clock=clock, policy=DialerPolicy(
        calls_per_second=2.0, burst=5, max_concurrent=100
    ))

    assert (await dialer.tick()).originated == 5   # всплеск израсходован
    assert (await dialer.tick()).reason == "темп набора"

    clock.advance(5)
    assert (await dialer.tick()).originated == 5   # накопилось за 5 секунд


# ------------------------------------------------------------ карусель номеров


async def test_calls_are_spread_across_the_number_pool() -> None:
    """С одного номера операторы связи блокируют быстро (CONCEPT.md §8.1)."""
    provider, source = FakeProvider(), FakeContactSource()
    source.fill(6)
    dialer, _ = build(provider, source, numbers=["+7495000001", "+7495000002", "+7495000003"])
    await dialer.tick()

    used = [r.caller_id for r in provider.originated]
    assert len(set(used)) == 3


def test_burned_number_leaves_the_rotation() -> None:
    pool = RoundRobinCallerIds(["a", "b"])
    pool.burn("a")
    assert pool.available == 1
    assert {pool.take() for _ in range(4)} == {"b"}


def test_empty_pool_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="пустой пул"):
        RoundRobinCallerIds([])


async def test_dialling_stops_when_every_number_is_burned() -> None:
    provider, source = FakeProvider(), FakeContactSource()
    source.fill(3)
    pool = RoundRobinCallerIds(["a"])
    pool.burn("a")
    dialer = Dialer(
        provider, source, pool, wave_id=WAVE,
        on_originated=lambda a, h: None, clock=FakeClock(step_sec=0.0),
    )
    report = await dialer.tick()
    assert report.originated == 0
    assert report.reason is not None and "живых номеров" in report.reason


def test_taking_from_a_fully_burned_pool_raises() -> None:
    pool = RoundRobinCallerIds(["a"])
    pool.burn("a")
    with pytest.raises(TelephonyError, match="выведены из ротации"):
        pool.take()


# ------------------------------------------------------------- предохранитель


async def test_mass_technical_failures_trip_the_breaker() -> None:
    """Продолжать звонить, пока лежит транк, — прожигать попытки дозвона."""
    provider, source = FakeProvider(), FakeContactSource()
    source.fill(40)
    dialer, started = build(provider, source, policy=DialerPolicy(
        calls_per_second=100, burst=100, min_samples=10, failure_threshold=0.5,
    ))
    await dialer.tick()
    for attempt in started[:12]:
        dialer.release(attempt, Disposition.TECH_ERROR)

    assert dialer.breaker.state is BreakerState.OPEN
    report = await dialer.tick()
    assert report.originated == 0
    assert report.reason is not None and "предохранитель" in report.reason


async def test_busy_numbers_never_trip_the_breaker() -> None:
    """Занято и недозвон — нормальная жизнь обзвона, а не отказ инфраструктуры."""
    provider, source = FakeProvider(), FakeContactSource()
    source.fill(40)
    dialer, started = build(provider, source)
    await dialer.tick()
    for attempt in started[:30]:
        dialer.release(attempt, Disposition.BUSY)

    assert dialer.breaker.state is BreakerState.CLOSED


async def test_a_few_failures_do_not_stop_the_wave() -> None:
    """На трёх неудачах подряд останавливать волну — значит останавливать
    её постоянно."""
    provider, source = FakeProvider(), FakeContactSource()
    source.fill(40)
    dialer, started = build(provider, source, policy=DialerPolicy(
        calls_per_second=100, burst=100, min_samples=10,
    ))
    await dialer.tick()
    for attempt in started[:3]:
        dialer.release(attempt, Disposition.TECH_ERROR)

    assert dialer.breaker.state is BreakerState.CLOSED


async def test_probe_call_after_the_cooldown_resumes_the_wave() -> None:
    provider, source = FakeProvider(), FakeContactSource()
    source.fill(100)
    clock = FakeClock(step_sec=0.0)
    dialer, started = build(provider, source, clock=clock, policy=DialerPolicy(
        calls_per_second=100, burst=100, min_samples=10, cooldown_sec=120,
    ))
    await dialer.tick()
    for attempt in started[:12]:
        dialer.release(attempt, Disposition.TECH_ERROR)
    assert dialer.breaker.state is BreakerState.OPEN

    clock.advance(121)
    probe = await dialer.tick()
    assert probe.originated == 1
    assert probe.breaker is BreakerState.PROBING

    dialer.release(started[-1], None)  # пробный прошёл
    assert not dialer.breaker.is_open
    assert (await dialer.tick()).originated > 1


async def test_failed_probe_extends_the_pause() -> None:
    provider, source = FakeProvider(), FakeContactSource()
    source.fill(100)
    clock = FakeClock(step_sec=0.0)
    dialer, started = build(provider, source, clock=clock, policy=DialerPolicy(
        calls_per_second=100, burst=100, min_samples=10, cooldown_sec=120,
    ))
    await dialer.tick()
    for attempt in started[:12]:
        dialer.release(attempt, Disposition.TECH_ERROR)

    clock.advance(121)
    await dialer.tick()
    dialer.release(started[-1], Disposition.TECH_ERROR)  # пробный тоже упал

    assert dialer.breaker.is_open
    assert (await dialer.tick()).originated == 0


# ------------------------------------------------------------- сбой при наборе


async def test_failed_originate_returns_the_contact_to_the_queue() -> None:
    """Иначе сбой транка съедал бы лимит попыток на номер,
    отведённый регламентом (ТЗ §6.2)."""
    provider, source = FakeProvider(fail_originate=True), FakeContactSource()
    source.fill(3)
    dialer, started = build(provider, source)

    report = await dialer.tick()
    assert report.originated == 0
    assert report.failed == 3
    assert len(source.released) == 3
    assert started == []
    assert dialer.active == 0
