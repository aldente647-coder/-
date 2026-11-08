"""Смена состояния попытки в базе: машина, дедупликация, блокировка."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session
from sqlalchemy.orm.exc import StaleDataError

from socmon.core.codes import CallStage, Disposition
from socmon.db.calls import Applied, record_transition
from socmon.db.models import CallAttempt, CallEventRecord
from socmon.telephony.machine import CallEvent, IllegalTransition
from tests.conftest import needs_db
from tests.test_db_constraints import make_contact, make_wave

pytestmark = [needs_db, pytest.mark.usefixtures("engine")]

NOW = datetime(2026, 8, 20, 18, 30, tzinfo=UTC)


@pytest.fixture
def attempt(session: Session) -> CallAttempt:
    contact = make_contact(session, index=int(uuid.uuid4().int % 10**9))
    wave = make_wave(session, f"w-{uuid.uuid4().hex[:6]}")
    row = CallAttempt(
        contact_id=contact.id, wave_id=wave.id, attempt_no=1, scheduled_at=NOW
    )
    session.add(row)
    session.flush()
    return row


def push(
    session: Session, attempt: CallAttempt, event: CallEvent,
    *, seconds: int = 0, event_id: str | None = None,
) -> Applied:
    return record_transition(
        session, attempt, event,
        at=NOW + timedelta(seconds=seconds),
        provider_event_id=event_id or f"{event.value}-{uuid.uuid4()}",
    )


def test_ring_moves_the_attempt_to_ringing(
    session: Session, attempt: CallAttempt
) -> None:
    push(session, attempt, CallEvent.RING)
    assert attempt.stage is CallStage.RINGING


def test_answer_starts_the_interview_clock(
    session: Session, attempt: CallAttempt
) -> None:
    push(session, attempt, CallEvent.RING)
    push(session, attempt, CallEvent.ANSWER, seconds=6)
    assert attempt.stage is CallStage.IN_INTERVIEW
    assert attempt.started_at == NOW + timedelta(seconds=6)


def test_finished_attempt_gets_its_duration(
    session: Session, attempt: CallAttempt
) -> None:
    push(session, attempt, CallEvent.RING)
    push(session, attempt, CallEvent.ANSWER, seconds=6)
    push(session, attempt, CallEvent.COMPLETE, seconds=306)

    assert attempt.stage is CallStage.FINISHED
    assert attempt.ended_at == NOW + timedelta(seconds=306)
    assert float(attempt.duration_sec or 0) == pytest.approx(300.0)


def test_no_answer_leaves_no_start_time(
    session: Session, attempt: CallAttempt
) -> None:
    push(session, attempt, CallEvent.RING)
    push(session, attempt, CallEvent.NO_ANSWER, seconds=40)

    assert attempt.disposition is Disposition.NO_ANSWER
    assert attempt.started_at is None
    assert attempt.duration_sec is None


def test_redelivered_event_changes_nothing(
    session: Session, attempt: CallAttempt
) -> None:
    """Доставка «хотя бы один раз» — норма. Без дедупликации один звонок
    дважды займёт оператора."""
    push(session, attempt, CallEvent.RING, event_id="e-1")
    first = attempt.stage

    again = push(session, attempt, CallEvent.RING, event_id="e-1")
    assert again.duplicate is True
    assert attempt.stage is first

    stored = session.scalar(
        select(func.count()).select_from(CallEventRecord)
        .where(CallEventRecord.attempt_id == attempt.id)
    )
    assert stored == 1


def test_every_event_is_journalled_even_when_nothing_changes(
    session: Session, attempt: CallAttempt
) -> None:
    """Шкала времени попытки (ТЗ §10.1) должна отражать то, что происходило
    на линии, а не только удачные переходы."""
    push(session, attempt, CallEvent.RING, event_id="a")
    push(session, attempt, CallEvent.RING, event_id="b")  # тот же звонок, другое событие

    rows = session.scalars(
        select(CallEventRecord).where(CallEventRecord.attempt_id == attempt.id)
    ).all()
    assert len(rows) == 2
    assert rows[1].stage_before == rows[1].stage_after == CallStage.RINGING.value


def test_impossible_event_is_refused(session: Session, attempt: CallAttempt) -> None:
    """«Интервью завершено» из стадии набора пометило бы недозвон как
    состоявшееся интервью и испортило бы расчёт достижимости."""
    with pytest.raises(IllegalTransition):
        push(session, attempt, CallEvent.COMPLETE)


def test_events_after_the_end_are_absorbed(
    session: Session, attempt: CallAttempt
) -> None:
    push(session, attempt, CallEvent.RING)
    push(session, attempt, CallEvent.NO_ANSWER, seconds=40)

    late = push(session, attempt, CallEvent.HANGUP, seconds=41)
    assert late.transition.changed is False
    assert attempt.disposition is Disposition.NO_ANSWER


def test_concurrent_update_is_detected(
    session: Session, attempt: CallAttempt
) -> None:
    """Диалер и вебхуки движка ходят в одну строку. Без счётчика версии
    один затирает другого молча."""
    session.commit()
    session.execute(
        update(CallAttempt)
        .where(CallAttempt.id == attempt.id)
        .values(version=attempt.version + 1)
        .execution_options(synchronize_session=False)
    )
    with pytest.raises(StaleDataError):
        push(session, attempt, CallEvent.RING)


def test_transfer_path_is_recorded(session: Session, attempt: CallAttempt) -> None:
    push(session, attempt, CallEvent.RING)
    push(session, attempt, CallEvent.ANSWER, seconds=5)
    push(session, attempt, CallEvent.TRANSFER_REQUEST, seconds=90)
    push(session, attempt, CallEvent.OPERATOR_ANSWER, seconds=95)

    assert attempt.stage is CallStage.WITH_OPERATOR
    kinds = [
        r.kind for r in session.scalars(
            select(CallEventRecord)
            .where(CallEventRecord.attempt_id == attempt.id)
            .order_by(CallEventRecord.at)
        ).all()
    ]
    assert kinds == ["ring", "answer", "transfer_request", "operator_answer"]
