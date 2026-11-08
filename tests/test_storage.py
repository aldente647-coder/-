"""Полный путь: интервью в движке → строки в базе → вердикт валидности."""

from __future__ import annotations

import dataclasses
import uuid

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session as DbSession

from socmon.core.codes import Outcome
from socmon.core.engine import InterviewSession, Utterance
from socmon.core.questionnaire import Questionnaire
from socmon.db.models import Answer, IdentityLink, Interview, InterviewEvent
from socmon.db.storage import CallContext, save_interview
from socmon.domain.validity import Reason, Validity
from tests.conftest import FakeClock, needs_db
from tests.test_db_constraints import make_contact, make_version, make_wave
from tests.test_engine import HAPPY_PATH, drive

pytestmark = [needs_db, pytest.mark.usefixtures("engine")]


@pytest.fixture
def context(session: DbSession) -> CallContext:
    from socmon.db.models import CallAttempt

    wave = make_wave(session, f"w-{uuid.uuid4().hex[:6]}")
    # Только версия анкеты: заводить здесь лишнее интервью нельзя — оно
    # попало бы в аналитический массив волны и испортило бы тесты сводки.
    version = make_version(session, wave).id
    contact = make_contact(session, index=int(uuid.uuid4().int % 10**9))
    attempt = CallAttempt(
        contact_id=contact.id, wave_id=wave.id, attempt_no=1,
        scheduled_at=wave.starts_at,
    )
    session.add(attempt)
    session.flush()
    return CallContext(
        wave_id=wave.id,
        questionnaire_version_id=version,
        contact_id=contact.id,
        attempt_id=attempt.id,
    )


def run(core: Questionnaire, script: dict[str, str] | None = None) -> InterviewSession:
    interview = InterviewSession(core, clock=FakeClock(step_sec=8.0))
    drive(interview, script or HAPPY_PATH)
    return interview


def test_completed_interview_is_stored_and_valid(
    session: DbSession, core: Questionnaire, context: CallContext
) -> None:
    stored = save_interview(session, run(core), context)

    assert stored.outcome is Outcome.COMPLETED
    assert stored.validity is Validity.VALID
    assert stored.validity_reasons == []
    assert stored.consent is True
    assert stored.duration_sec > 0

    answers = session.scalars(
        select(Answer).where(Answer.interview_id == stored.id)
    ).all()
    assert len(answers) == len(run(core).answers)
    assert {a.core_key for a in answers} >= {"anxiety", "institutional_trust"}

    events = session.scalar(
        select(func.count()).select_from(InterviewEvent)
        .where(InterviewEvent.interview_id == stored.id)
    )
    assert events and events > 20


def test_identity_link_is_the_only_bridge(
    session: DbSession, core: Questionnaire, context: CallContext
) -> None:
    """Из аналитической строки к абоненту дороги нет — только через
    отдельную таблицу в контактном контуре (ТЗ §14.2)."""
    stored = save_interview(session, run(core), context)

    link = session.scalar(
        select(IdentityLink).where(IdentityLink.interview_id == stored.id)
    )
    assert link is not None
    assert link.contact_id == context.contact_id
    assert not any(c.name.endswith("contact_id") for c in Interview.__table__.columns)


def test_a_second_interview_in_the_same_wave_is_refused(
    session: DbSession, core: Questionnaire, context: CallContext
) -> None:
    """ТЗ §4.2 исполняется базой, а не добросовестностью кода."""
    save_interview(session, run(core), context)
    with pytest.raises(IntegrityError):
        save_interview(session, run(core), context)


def test_refusal_is_stored_as_invalid_with_reasons(
    session: DbSession, core: Questionnaire, context: CallContext
) -> None:
    interview = InterviewSession(core, clock=FakeClock())
    interview.begin()
    interview.submit(Utterance.said("нет, не хочу"))

    stored = save_interview(session, interview, context)
    assert stored.validity is Validity.INVALID
    assert Reason.NO_CONSENT.value in (stored.validity_reasons or [])
    assert Reason.NOT_COMPLETED.value in (stored.validity_reasons or [])


def test_test_call_never_enters_the_dataset(
    session: DbSession, core: Questionnaire, context: CallContext
) -> None:
    """Контрольный звонок проходит весь путь как настоящий, но в массив
    не попадает — приём из блюпринта VOIF."""
    marked = dataclasses.replace(context, is_test=True)
    stored = save_interview(session, run(core), marked)

    assert stored.is_test is True
    assert stored.validity is Validity.INVALID
    assert stored.validity_reasons == [Reason.TEST_CALL.value]


def test_screened_out_respondent_is_invalid(
    session: DbSession, core: Questionnaire, context: CallContext
) -> None:
    interview = InterviewSession(core, clock=FakeClock())
    interview.begin()
    interview.submit(Utterance.said("да"))
    interview.submit(Utterance.said("нет, мне меньше"))

    stored = save_interview(session, interview, context)
    assert stored.outcome is Outcome.INELIGIBLE_AGE
    assert Reason.AGE_NOT_CONFIRMED.value in (stored.validity_reasons or [])


def test_unfinished_interview_is_not_saved(core: Questionnaire, context: CallContext) -> None:
    interview = InterviewSession(core, clock=FakeClock())
    interview.begin()
    with pytest.raises(ValueError, match="не завершено"):
        save_interview(None, interview, context)  # type: ignore[arg-type]


def test_silent_open_answer_is_not_stored_as_an_empty_string(
    session: DbSession, core: Questionnaire, context: CallContext
) -> None:
    script = dict(HAPPY_PATH)
    interview = InterviewSession(core, clock=FakeClock(step_sec=8.0))
    turn = interview.begin()
    while (node_id := turn.awaiting) is not None:
        if node_id.startswith("q_open"):
            turn = interview.submit(Utterance.quiet())
        else:
            turn = interview.submit(Utterance.said(script[node_id]))

    stored = save_interview(session, interview, context)
    open_rows = session.scalars(
        select(Answer).where(Answer.interview_id == stored.id, Answer.node_id.like("q_open%"))
    ).all()
    assert open_rows == []
    assert stored.validity is Validity.VALID
