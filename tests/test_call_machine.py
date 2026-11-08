"""Машина состояний звонка.

Блюпринт VOIF: не делать первый звонок, пока она не готова. Поэтому здесь
проверяется не «в целом работает», а каждое запрещённое ребро.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from socmon.core.codes import CallStage, Disposition
from socmon.telephony.machine import (
    CallEvent,
    CallState,
    IllegalTransition,
    WatchdogPolicy,
    apply,
    retryable,
    stale_event,
)

NOW = datetime(2026, 8, 20, 18, 30, tzinfo=UTC)


def drive(*events: CallEvent) -> CallState:
    state = CallState()
    for event in events:
        state = apply(state, event).state
    return state


# ---------------------------------------------------------------- нормальный ход


def test_full_interview_path() -> None:
    state = drive(CallEvent.RING, CallEvent.ANSWER, CallEvent.COMPLETE)
    assert state.stage is CallStage.FINISHED
    assert state.disposition is None  # связь отработала штатно
    assert state.answered is True


def test_transfer_to_operator_path() -> None:
    state = drive(
        CallEvent.RING, CallEvent.ANSWER,
        CallEvent.TRANSFER_REQUEST, CallEvent.OPERATOR_ANSWER, CallEvent.HANGUP,
    )
    assert state.stage is CallStage.FINISHED
    assert state.transferred is True


def test_no_answer_path() -> None:
    state = drive(CallEvent.RING, CallEvent.NO_ANSWER)
    assert state.disposition is Disposition.NO_ANSWER
    assert state.answered is False


def test_answering_machine_ends_the_call_without_talking() -> None:
    """Разговаривать с автоответчиком — прожигать минуты связи впустую."""
    state = drive(CallEvent.RING, CallEvent.ANSWERING_MACHINE)
    assert state.stage is CallStage.FINISHED
    assert state.disposition is Disposition.ANSWERING_MACHINE
    assert state.answered is False


def test_missing_operator_is_not_a_respondent_refusal() -> None:
    state = drive(CallEvent.RING, CallEvent.ANSWER,
                  CallEvent.TRANSFER_REQUEST, CallEvent.NO_OPERATOR)
    assert state.disposition is Disposition.NO_OPERATOR


# ------------------------------------------------------------ запрещённые рёбра


@pytest.mark.parametrize(
    ("path", "forbidden"),
    [
        ((), CallEvent.OPERATOR_ANSWER),                  # оператор без перевода
        ((), CallEvent.COMPLETE),                         # интервью без ответа
        ((CallEvent.RING,), CallEvent.COMPLETE),          # то же на гудках
        ((CallEvent.RING,), CallEvent.TRANSFER_REQUEST),  # перевод до ответа
        ((CallEvent.RING, CallEvent.ANSWER), CallEvent.OPERATOR_ANSWER),
        ((CallEvent.RING, CallEvent.ANSWER), CallEvent.BUSY),   # «занято» после ответа
        ((CallEvent.RING, CallEvent.ANSWER), CallEvent.NO_ANSWER),
    ],
)
def test_illegal_transitions_are_refused(
    path: tuple[CallEvent, ...], forbidden: CallEvent
) -> None:
    state = drive(*path)
    with pytest.raises(IllegalTransition):
        apply(state, forbidden)


def test_terminal_state_absorbs_everything() -> None:
    """Движок вправе прислать «канал уничтожен» после того, как мы закрыли
    звонок сами. Это не ошибка, а нормальная гонка."""
    finished = drive(CallEvent.RING, CallEvent.NO_ANSWER)
    for event in (CallEvent.HANGUP, CallEvent.ANSWER, CallEvent.TECH_ERROR):
        transition = apply(finished, event)
        assert transition.changed is False
        assert transition.state == finished


def test_repeated_event_is_idempotent() -> None:
    """Доставка «хотя бы один раз» — норма для вебхуков; повтор обязан быть
    безобидным, иначе один звонок дважды займёт оператора."""
    state = drive(CallEvent.RING)
    again = apply(state, CallEvent.RING)
    assert again.changed is False
    assert again.state == state


# ------------------------------------------------ обрыв в разговоре и повторы


def test_hangup_during_interview_is_not_a_technical_failure() -> None:
    """Респондент положил трубку — связь не подвела. Повторять такой номер
    регламент не должен (ТЗ §6.2)."""
    state = drive(CallEvent.RING, CallEvent.ANSWER, CallEvent.HANGUP)
    assert state.disposition is None
    assert retryable(state.disposition) is False


def test_hangup_before_answer_is_a_dropped_call() -> None:
    state = drive(CallEvent.RING, CallEvent.HANGUP)
    assert state.disposition is Disposition.DROPPED
    assert retryable(state.disposition) is True


@pytest.mark.parametrize(
    ("disposition", "expected"),
    [
        (Disposition.NO_ANSWER, True),
        (Disposition.BUSY, True),
        (Disposition.ANSWERING_MACHINE, True),
        (Disposition.TECH_ERROR, False),
        (Disposition.INVALID_NUMBER, False),
        (Disposition.NO_OPERATOR, False),
        (None, False),
    ],
)
def test_retry_rules_match_the_dialling_regulation(
    disposition: Disposition | None, expected: bool
) -> None:
    """Технический сбой не повторяем: повтор маскирует поломку вместо того,
    чтобы о ней сигнализировать."""
    assert retryable(disposition) is expected


# --------------------------------------------------------------------- сторож


def test_watchdog_ignores_a_healthy_call() -> None:
    state = drive(CallEvent.RING, CallEvent.ANSWER)
    assert stale_event(state, stage_since=NOW, now=NOW + timedelta(seconds=120),
                       last_media_at=NOW + timedelta(seconds=118)) is None


def test_watchdog_closes_a_stuck_ringing_call() -> None:
    state = drive(CallEvent.RING)
    assert stale_event(state, stage_since=NOW, now=NOW + timedelta(minutes=5)) is (
        CallEvent.TIMEOUT
    )


def test_watchdog_closes_an_interview_that_will_not_end() -> None:
    """Анкета рассчитана на 10 минут (ТЗ §3.3). Полчаса — застрявший диалог,
    а не разговорчивый респондент."""
    state = drive(CallEvent.RING, CallEvent.ANSWER)
    assert stale_event(state, stage_since=NOW, now=NOW + timedelta(minutes=30)) is (
        CallEvent.TIMEOUT
    )


def test_silence_on_a_live_channel_is_a_technical_failure() -> None:
    """Звук пропал — канал мёртв. Это поломка, а не поведение респондента,
    и повторять номер по такой причине нельзя."""
    state = drive(CallEvent.RING, CallEvent.ANSWER)
    assert stale_event(
        state, stage_since=NOW, now=NOW + timedelta(seconds=200),
        last_media_at=NOW + timedelta(seconds=10),
    ) is CallEvent.TECH_ERROR


def test_watchdog_leaves_finished_calls_alone() -> None:
    finished = drive(CallEvent.RING, CallEvent.NO_ANSWER)
    assert stale_event(finished, stage_since=NOW, now=NOW + timedelta(days=1)) is None


def test_watchdog_limits_are_configurable() -> None:
    state = drive(CallEvent.RING, CallEvent.ANSWER)
    strict = WatchdogPolicy(interview_sec=60.0)
    assert stale_event(state, stage_since=NOW, now=NOW + timedelta(seconds=90),
                       policy=strict) is CallEvent.TIMEOUT
