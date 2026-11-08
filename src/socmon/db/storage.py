"""Сохранение интервью.

Единственное место, где движок интервью встречается с базой. Сервисы не пишут
SQL напрямую — только через этот слой; так проще и тестировать, и аудировать,
кто и что записал.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy.orm import Session as DbSession

from socmon.core.codes import Outcome, to_aapor
from socmon.core.engine import InterviewSession
from socmon.db.models import Answer, IdentityLink, Interview, InterviewEvent
from socmon.domain.facts import derive_facts
from socmon.domain.validity import DEFAULT_POLICY, ValidityPolicy, evaluate


@dataclass(frozen=True, slots=True)
class CallContext:
    """Что известно о звонке за пределами анкеты."""

    wave_id: uuid.UUID
    questionnaire_version_id: uuid.UUID
    contact_id: uuid.UUID
    attempt_id: uuid.UUID
    timezone: str = "Europe/Moscow"
    recording_uri: str | None = None
    recording_respondent_uri: str | None = None
    recording_robot_uri: str | None = None
    transcript_uri: str | None = None
    is_test: bool = False
    answering_machine: bool = False
    synthetic_respondent: bool = False
    respondent_substituted: bool = False
    technical_error: bool = False


def save_interview(
    db: DbSession,
    interview_session: InterviewSession,
    context: CallContext,
    *,
    policy: ValidityPolicy = DEFAULT_POLICY,
) -> Interview:
    """Записать завершённое интервью и вынести решение о валидности.

    Идентификатор интервью служит псевдонимом респондента и генерируется здесь
    же; связь с абонентом уходит единственной строкой в ``identity_link``,
    в контактный контур (ТЗ §14.2).
    """
    if not interview_session.finished:
        raise ValueError("интервью не завершено: сохранять нечего")

    outcome = interview_session.outcome
    assert outcome is not None
    verdict = evaluate(derive_facts(
        interview_session,
        is_test=context.is_test,
        answering_machine=context.answering_machine,
        synthetic_respondent=context.synthetic_respondent,
        respondent_substituted=context.respondent_substituted,
        technical_error=context.technical_error,
    ), policy)

    interview = Interview(
        id=uuid.uuid4(),
        wave_id=context.wave_id,
        questionnaire_version_id=context.questionnaire_version_id,
        started_at=interview_session.events[0].at,
        finished_at=interview_session.events[-1].at,
        duration_sec=round(interview_session.duration_sec, 2),
        timezone=context.timezone,
        consent=interview_session.consented,
        outcome=outcome,
        aapor_code=to_aapor(outcome, None),
        transferred_to_operator=outcome is Outcome.TRANSFERRED_TO_OPERATOR,
        unrecognized_count=_unrecognized(interview_session),
        recording_uri=context.recording_uri,
        recording_respondent_uri=context.recording_respondent_uri,
        recording_robot_uri=context.recording_robot_uri,
        transcript_uri=context.transcript_uri,
        is_test=context.is_test,
        validity=verdict.status,
        validity_reasons=[r.value for r in verdict.reasons],
        validity_details=list(verdict.details),
    )
    db.add(interview)
    db.flush()

    db.add_all(
        Answer(
            interview_id=interview.id,
            node_id=answer.node_id,
            core_key=answer.core_key,
            code=answer.code,
            codes=list(answer.codes) or None,
            text=answer.text,
            attempts=answer.attempts,
            answered_at=answer.at,
        )
        for answer in interview_session.answers.values()
        # Ячейка без кода и без текста — дырка в массиве, а не ответ:
        # молчание на открытом вопросе не хранится как пустая строка.
        if answer.code is not None or answer.text
    )
    db.add_all(
        InterviewEvent(
            interview_id=interview.id,
            at=event.at,
            kind=event.kind,
            node_id=event.node_id,
            detail=event.detail,
        )
        for event in interview_session.events
    )
    db.add(IdentityLink(
        interview_id=interview.id,
        contact_id=context.contact_id,
        attempt_id=context.attempt_id,
        wave_id=context.wave_id,
    ))
    db.flush()
    return interview


def _unrecognized(interview_session: InterviewSession) -> int:
    from socmon.core.codes import SpecialCode

    return sum(
        1 for a in interview_session.answers.values() if a.code == SpecialCode.UNRECOGNIZED
    )
