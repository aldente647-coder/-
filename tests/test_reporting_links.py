"""Отчётность использует конкретные попытки, без произведения строк волны."""
from __future__ import annotations

import uuid

from sqlalchemy.orm import Session

from socmon.core.codes import CallStage, Disposition
from socmon.db.journal import DbJournal
from socmon.db.models import CallAttempt, IdentityLink
from socmon.web.artifacts import call_statistics
from tests.conftest import needs_db
from tests.test_db_constraints import NOW, make_contact, make_interview, make_version, make_wave

pytestmark = needs_db


def test_final_outcomes_are_counted_once_per_contact(session: Session) -> None:
    wave = make_wave(session)
    first = make_contact(session, 1)
    second = make_contact(session, 2)
    rows = []
    for contact, number, disposition, is_test in [
        (first, 1, Disposition.NO_ANSWER, False),
        (first, 2, None, False),
        (second, 1, Disposition.BUSY, False),
        (second, 2, Disposition.NO_ANSWER, False),
        (second, 3, Disposition.NO_ANSWER, True),
    ]:
        row = CallAttempt(
            contact_id=contact.id, wave_id=wave.id, attempt_no=number,
            scheduled_at=NOW, stage=CallStage.FINISHED,
            disposition=disposition, is_test=is_test,
        )
        session.add(row)
        rows.append(row)
    session.flush()
    interview = make_interview(session, wave)
    session.add(IdentityLink(
        contact_id=first.id, wave_id=wave.id, attempt_id=rows[1].id,
        interview_id=interview.id,
    ))
    session.flush()
    stats = call_statistics(session, wave.id)
    assert stats.numbers_dialled == 2
    assert stats.attempts_made == 4
    assert stats.dispositions.total == 2
    assert stats.dispositions.interviews == 1


def test_journal_does_not_return_another_interview_in_same_wave(session: Session) -> None:
    wave = make_wave(session)
    version = make_version(session, wave)
    journal = DbJournal(session, wave_id=wave.id, questionnaire_version_id=version.id)
    expected = {}
    for index in (1, 2):
        contact = make_contact(session, index)
        attempt = CallAttempt(contact_id=contact.id, wave_id=wave.id,
                              attempt_no=1, scheduled_at=NOW)
        session.add(attempt)
        session.flush()
        interview = make_interview(session, wave)
        session.add(IdentityLink(contact_id=contact.id, wave_id=wave.id,
                                 attempt_id=attempt.id, interview_id=interview.id))
        expected[attempt.id] = interview.id
    session.flush()
    for attempt_id, interview_id in expected.items():
        found = journal.interview_of(attempt_id)
        assert found and found.id == interview_id
    assert journal.interview_of(uuid.uuid4()) is None
