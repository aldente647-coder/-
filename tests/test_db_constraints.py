"""Ограничения целостности на живой PostgreSQL.

Проверяем не ORM, а то, что база физически не даст нарушить требования ТЗ,
даже если ошибётся прикладной код.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.orm.exc import StaleDataError

from socmon.core.codes import AaporCode, Outcome
from socmon.db.models import (
    Answer,
    CallAttempt,
    Contact,
    IdentityLink,
    Interview,
    OpenAnswerCoding,
    QuestionnaireVersion,
    Wave,
)
from socmon.domain.validity import Validity
from tests.conftest import needs_db

pytestmark = needs_db

NOW = datetime(2026, 8, 20, 18, 0, tzinfo=UTC)


def make_contact(session: Session, index: int = 0) -> Contact:
    contact = Contact(
        blind_index=index.to_bytes(32, "big"),
        phone_ciphertext=b"opaque",
    )
    session.add(contact)
    session.flush()
    return contact


def make_wave(session: Session, code: str = "w-2026-17") -> Wave:
    wave = Wave(
        code=code, name="Волна", starts_at=NOW, ends_at=NOW + timedelta(days=14)
    )
    session.add(wave)
    session.flush()
    return wave


def make_version(
    session: Session, wave: Wave, *, source_yaml: str | None = None
) -> QuestionnaireVersion:
    """Версия анкеты волны.

    Переиспользует уже существующую: у волны она одна утверждённая,
    а уникальность по паре «волна + версия» не даст завести вторую.
    """
    existing = session.scalar(
        select(QuestionnaireVersion).where(QuestionnaireVersion.wave_id == wave.id)
    )
    if existing is not None:
        return existing
    version = QuestionnaireVersion(
        wave_id=wave.id, name="Ядро", version_id="abc123",
        source_yaml=source_yaml or "...", core_keys=["anxiety"],
    )
    session.add(version)
    session.flush()
    return version


def make_interview(session: Session, wave: Wave, **kwargs: object) -> Interview:
    version = make_version(session, wave)
    defaults: dict[str, object] = {
        "wave_id": wave.id,
        "questionnaire_version_id": version.id,
        "started_at": NOW,
        "timezone": "Europe/Moscow",
        "outcome": Outcome.COMPLETED,
        "aapor_code": AaporCode.COMPLETE,
        "validity": Validity.VALID,
    }
    defaults.update(kwargs)
    interview = Interview(**defaults)
    session.add(interview)
    session.flush()
    return interview


# --------------------------------------------------------------- дедупликация


def test_the_same_number_cannot_be_loaded_twice(session: Session) -> None:
    """ТЗ §4.2: исключение дублей. Слепой индекс уникален, поэтому один
    абонент физически не может попасть в базу дважды."""
    make_contact(session, 1)
    session.add(Contact(blind_index=(1).to_bytes(32, "big"), phone_ciphertext=b"other"))
    with pytest.raises(IntegrityError):
        session.flush()


# ------------------------------------------------------- повторный обзвон


def test_contact_cannot_be_interviewed_twice_in_one_wave(session: Session) -> None:
    """ТЗ §4.2: после состоявшегося интервью номер не включается в текущую
    волну повторно."""
    contact, wave = make_contact(session), make_wave(session)
    attempt = CallAttempt(
        contact_id=contact.id, wave_id=wave.id, attempt_no=1, scheduled_at=NOW
    )
    session.add(attempt)
    session.flush()

    session.add(IdentityLink(
        interview_id=uuid.uuid4(), contact_id=contact.id,
        attempt_id=attempt.id, wave_id=wave.id,
    ))
    session.flush()

    session.add(IdentityLink(
        interview_id=uuid.uuid4(), contact_id=contact.id,
        attempt_id=attempt.id, wave_id=wave.id,
    ))
    with pytest.raises(IntegrityError):
        session.flush()


def test_the_same_contact_may_be_interviewed_in_the_next_wave(session: Session) -> None:
    """Запрет действует внутри волны, а не навсегда: между волнами абонент
    снова попадает в рамку выборки."""
    contact = make_contact(session)
    for code in ("w-17", "w-18"):
        wave = make_wave(session, code)
        attempt = CallAttempt(
            contact_id=contact.id, wave_id=wave.id, attempt_no=1, scheduled_at=NOW
        )
        session.add(attempt)
        session.flush()
        session.add(IdentityLink(
            interview_id=uuid.uuid4(), contact_id=contact.id,
            attempt_id=attempt.id, wave_id=wave.id,
        ))
        session.flush()


def test_attempt_numbers_are_unique_within_a_wave(session: Session) -> None:
    contact, wave = make_contact(session), make_wave(session)
    for _ in range(2):
        session.add(CallAttempt(
            contact_id=contact.id, wave_id=wave.id, attempt_no=1, scheduled_at=NOW
        ))
    with pytest.raises(IntegrityError):
        session.flush()


def test_attempt_number_starts_at_one(session: Session) -> None:
    contact, wave = make_contact(session), make_wave(session)
    session.add(CallAttempt(
        contact_id=contact.id, wave_id=wave.id, attempt_no=0, scheduled_at=NOW
    ))
    with pytest.raises(IntegrityError):
        session.flush()


def test_call_cannot_end_before_it_started(session: Session) -> None:
    contact, wave = make_contact(session), make_wave(session)
    session.add(CallAttempt(
        contact_id=contact.id, wave_id=wave.id, attempt_no=1, scheduled_at=NOW,
        started_at=NOW, ended_at=NOW - timedelta(minutes=5),
    ))
    with pytest.raises(IntegrityError):
        session.flush()


# ------------------------------------------------------------------- ответы


def test_answer_code_stays_within_the_code_space(session: Session) -> None:
    wave = make_wave(session)
    interview = make_interview(session, wave)
    session.add(Answer(
        interview_id=interview.id, node_id="q_life", code=500, answered_at=NOW
    ))
    with pytest.raises(IntegrityError):
        session.flush()


def test_empty_answer_is_rejected(session: Session) -> None:
    """Ячейка без кода и без текста — это дырка в массиве, а не ответ."""
    wave = make_wave(session)
    interview = make_interview(session, wave)
    session.add(Answer(interview_id=interview.id, node_id="q_life", answered_at=NOW))
    with pytest.raises(IntegrityError):
        session.flush()


def test_one_answer_per_node(session: Session) -> None:
    wave = make_wave(session)
    interview = make_interview(session, wave)
    for code in (7, 8):
        session.add(Answer(
            interview_id=interview.id, node_id="q_life", code=code, answered_at=NOW
        ))
    with pytest.raises(IntegrityError):
        session.flush()


# ------------------------------------------------------- маркировка моделей


def test_model_output_cannot_shed_its_marking(session: Session) -> None:
    """ТЗ §11.3: автоматически сгенерированные результаты маркируются.
    Ограничение не даёт выдать вывод модели за первичные данные."""
    wave = make_wave(session)
    interview = make_interview(session, wave)
    session.add(OpenAnswerCoding(
        interview_id=interview.id, node_id="q_open_leader",
        classifier_version="v1", model="qwen3-32b",
        themes=["доверие"], entities=[], is_model_generated=False,
    ))
    with pytest.raises(IntegrityError):
        session.flush()


def test_recoding_adds_a_row_instead_of_overwriting(session: Session) -> None:
    """Новая версия классификатора не затирает старую разметку — иначе
    исторические выводы задним числом меняются (ТЗ §11.3)."""
    wave = make_wave(session)
    interview = make_interview(session, wave)
    for version in ("v1", "v2"):
        session.add(OpenAnswerCoding(
            interview_id=interview.id, node_id="q_open_leader",
            classifier_version=version, model="qwen3-32b", themes=[], entities=[],
        ))
    session.flush()

    session.add(OpenAnswerCoding(
        interview_id=interview.id, node_id="q_open_leader",
        classifier_version="v1", model="qwen3-32b", themes=[], entities=[],
    ))
    with pytest.raises(IntegrityError):
        session.flush()


# ------------------------------------------------------------------- прочее


def test_weight_must_be_positive(session: Session) -> None:
    wave = make_wave(session)
    with pytest.raises(IntegrityError):
        make_interview(session, wave, weight=0)


def test_wave_period_must_be_ordered(session: Session) -> None:
    session.add(Wave(
        code="broken", name="Волна", starts_at=NOW, ends_at=NOW - timedelta(days=1)
    ))
    with pytest.raises(IntegrityError):
        session.flush()


def test_optimistic_locking_detects_a_lost_update(session: Session) -> None:
    """Приём из блюпринта VOIF: без счётчика версии конкурентные обновления
    затирают друг друга и один контакт уходит в набор дважды."""
    contact = make_contact(session)
    session.commit()

    stale_version = contact.version

    # Имитируем другой процесс: synchronize_session=False оставляет объект
    # в памяти со старой версией — ровно так это и выглядит при гонке.
    session.execute(
        update(Contact)
        .where(Contact.id == contact.id)
        .values(never_answers=True, version=stale_version + 1)
        .execution_options(synchronize_session=False)
    )
    contact.do_not_call = True
    with pytest.raises(StaleDataError):
        session.flush()
