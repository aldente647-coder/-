from __future__ import annotations

from datetime import UTC, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pytest

from socmon.domain.schedule import (
    CallPolicy,
    DailyWindow,
    plan_next_attempt,
)

MSK = ZoneInfo("Europe/Moscow")
POLICY = CallPolicy()


def msk(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=MSK)


# 2026-08-20 — четверг, 2026-08-22 — суббота.


@pytest.mark.parametrize(
    ("moment", "expected"),
    [
        (msk(2026, 8, 20, 18, 30), True),   # будни, внутри окна
        (msk(2026, 8, 20, 17, 59), False),  # будни, рано
        (msk(2026, 8, 20, 20, 0), False),   # верхняя граница исключительная
        (msk(2026, 8, 20, 3, 0), False),    # ночь
        (msk(2026, 8, 22, 13, 0), True),    # суббота, внутри окна
        (msk(2026, 8, 22, 11, 0), False),   # суббота, рано
    ],
)
def test_window_boundaries(moment: datetime, expected: bool) -> None:
    assert POLICY.is_open(moment) is expected


def test_windows_are_local_not_utc() -> None:
    """20:00 в Москве и 20:00 в UTC — разные вещи. Путаница здесь даёт
    ровно ночные звонки, запрещённые ТЗ §6.2."""
    utc_evening = datetime(2026, 8, 20, 19, 0, tzinfo=UTC)  # 22:00 МСК
    assert POLICY.is_open(utc_evening) is False


def test_naive_datetime_is_refused() -> None:
    with pytest.raises(ValueError, match="с зоной"):
        POLICY.is_open(datetime(2026, 8, 20, 18, 30))


def test_next_open_returns_now_when_window_is_already_open() -> None:
    moment = msk(2026, 8, 20, 18, 30)
    assert POLICY.next_open(moment) == moment


def test_next_open_jumps_to_the_evening() -> None:
    assert POLICY.next_open(msk(2026, 8, 20, 9, 0)) == msk(2026, 8, 20, 18, 0)


def test_next_open_rolls_over_to_the_next_day() -> None:
    assert POLICY.next_open(msk(2026, 8, 20, 21, 0)) == msk(2026, 8, 21, 18, 0)


def test_next_open_uses_the_weekend_window() -> None:
    """Суббота открывается в 12:00, а не в 18:00."""
    assert POLICY.next_open(msk(2026, 8, 22, 9, 0)) == msk(2026, 8, 22, 12, 0)


def test_retry_respects_the_48_hour_interval() -> None:
    plan = plan_next_attempt(
        contact_key="abc",
        attempts_made=1,
        last_attempt_at=msk(2026, 8, 20, 18, 30),
        last_disposition="no_answer",
    )
    assert plan.at is not None
    assert plan.at - msk(2026, 8, 20, 18, 30) >= timedelta(hours=48)
    assert POLICY.is_open(plan.at)


def test_retry_never_lands_outside_the_window() -> None:
    """Главное свойство модуля: что бы ни подали на вход, звонка ночью не будет."""
    for hour in range(24):
        for day in range(20, 27):
            plan = plan_next_attempt(
                contact_key=f"k{day}{hour}",
                attempts_made=1,
                last_attempt_at=msk(2026, 8, day, hour),
                last_disposition="busy",
            )
            assert plan.at is not None
            assert POLICY.is_open(plan.at), f"попытка вне окна: {plan.at}"


def test_attempts_land_on_different_times_of_day() -> None:
    """ТЗ §6.2: попытки выполняются в различные временные точки внутри окна."""
    times = set()
    for attempt in range(4):
        plan = plan_next_attempt(
            contact_key="one-and-the-same-contact",
            attempts_made=attempt,
            last_attempt_at=msk(2026, 8, 20, 18, 30),
            last_disposition="no_answer",
        )
        assert plan.at is not None
        times.add(plan.at.astimezone(MSK).time())
    assert len(times) > 1


def test_plan_is_reproducible() -> None:
    """Разброс детерминирован: тот же вход — тот же план. Нужно и для разбора
    инцидентов, и для воспроизводимости расчётов (ТЗ §12)."""
    kwargs = {
        "contact_key": "stable",
        "attempts_made": 2,
        "last_attempt_at": msk(2026, 8, 20, 19, 0),
        "last_disposition": "no_answer",
    }
    assert plan_next_attempt(**kwargs).at == plan_next_attempt(**kwargs).at  # type: ignore[arg-type]


def test_attempt_limit_stops_retries() -> None:
    plan = plan_next_attempt(
        contact_key="abc",
        attempts_made=5,
        last_attempt_at=msk(2026, 8, 20, 18, 30),
        last_disposition="no_answer",
    )
    assert plan.scheduled is False
    assert plan.reason is not None
    assert "лимит" in plan.reason


@pytest.mark.parametrize("disposition", ["tech_error", "invalid_number", "refusal"])
def test_non_retryable_dispositions_are_not_repeated(disposition: str) -> None:
    """Технический сбой не повторяем намеренно: повтор маскирует поломку
    вместо того, чтобы о ней сигнализировать."""
    plan = plan_next_attempt(
        contact_key="abc",
        attempts_made=1,
        last_attempt_at=msk(2026, 8, 20, 18, 30),
        last_disposition=disposition,
    )
    assert plan.scheduled is False


def test_custom_policy_is_honoured() -> None:
    policy = CallPolicy(
        weekday=DailyWindow(time(10, 0), time(12, 0)),
        max_attempts=3,
        min_interval_hours=72,
    )
    plan = plan_next_attempt(
        contact_key="abc",
        attempts_made=1,
        last_attempt_at=msk(2026, 8, 20, 11, 0),
        last_disposition="no_answer",
        policy=policy,
    )
    assert plan.at is not None
    assert plan.at - msk(2026, 8, 20, 11, 0) >= timedelta(hours=72)
    assert policy.is_open(plan.at)


def test_window_must_be_ordered() -> None:
    with pytest.raises(ValueError, match="раньше конца"):
        DailyWindow(time(20, 0), time(18, 0))
