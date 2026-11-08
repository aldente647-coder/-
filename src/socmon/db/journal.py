"""Журнал рантайма поверх базы.

Реализация порта `socmon.runtime.ports.Journal`. До неё рантайм умел вести
разговор, но записывать его было некуда: тесты кормили подставной журнал
в памяти. Здесь та же самая логика ложится на PostgreSQL — с машиной
состояний, дедупликацией событий и оптимистичной блокировкой,
которые уже написаны в `db/calls.py`.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session as DbSession

from socmon.core.engine import InterviewSession
from socmon.db.calls import record_transition
from socmon.db.models import CallAttempt, IdentityLink, Interview
from socmon.db.storage import CallContext, save_interview
from socmon.telephony.machine import CallEvent


class JournalError(RuntimeError):
    pass


class DbJournal:
    """Пишет ход звонка в базу.

    Контекст волны и версии анкеты задаётся один раз при создании: он
    одинаков для всех звонков волны, а тащить его через каждый вызов
    значило бы дать возможность перепутать.
    """

    def __init__(
        self,
        db: DbSession,
        *,
        wave_id: uuid.UUID,
        questionnaire_version_id: uuid.UUID,
        timezone: str = "Europe/Moscow",
    ) -> None:
        self._db = db
        self._wave_id = wave_id
        self._version_id = questionnaire_version_id
        self._timezone = timezone
        self._recordings: dict[uuid.UUID, tuple[str | None, str | None]] = {}

    def transition(
        self,
        attempt_id: uuid.UUID,
        event: CallEvent,
        *,
        at: datetime,
        provider_event_id: str,
        detail: str | None = None,
    ) -> bool:
        attempt = self._attempt(attempt_id)
        applied = record_transition(
            self._db, attempt, event,
            at=at, provider_event_id=provider_event_id, detail=detail,
        )
        return not applied.duplicate

    def attach_recording(
        self, attempt_id: uuid.UUID, respondent: str | None, robot: str | None
    ) -> None:
        """Запомнить ссылки на записи.

        Пишутся не сразу: интервью в этот момент ещё не создано в базе,
        а держать ссылку на несуществующую строку незачем. Сохраняется
        всё разом в `save_interview`.
        """
        self._recordings[attempt_id] = (respondent, robot)

    def save_interview(self, attempt_id: uuid.UUID, session: InterviewSession) -> None:
        attempt = self._attempt(attempt_id)
        respondent, robot = self._recordings.pop(attempt_id, (None, None))
        save_interview(self._db, session, CallContext(
            wave_id=self._wave_id,
            questionnaire_version_id=self._version_id,
            contact_id=attempt.contact_id,
            attempt_id=attempt_id,
            timezone=self._timezone,
            recording_respondent_uri=respondent,
            recording_robot_uri=robot,
            is_test=attempt.is_test,
        ))

    def _attempt(self, attempt_id: uuid.UUID) -> CallAttempt:
        attempt = self._db.get(CallAttempt, attempt_id)
        if attempt is None:
            raise JournalError(f"попытка {attempt_id} не найдена")
        return attempt

    def interview_of(self, attempt_id: uuid.UUID) -> Interview | None:
        return self._db.scalar(
            select(Interview).join(IdentityLink, IdentityLink.interview_id == Interview.id)
            .where(IdentityLink.attempt_id == attempt_id, Interview.wave_id == self._wave_id)
        )
