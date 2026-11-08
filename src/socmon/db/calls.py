"""Смена состояния попытки дозвона.

Единственная точка, где стадия звонка меняется в базе. Прямая запись
`attempt.stage = ...` в обход этой функции запрещена конвенцией — ровно как
в блюпринте VOIF: без единой точки перехода звонки начинают мерцать между
статусами, а один контакт уходит в набор дважды.

Три защиты работают вместе, и ни одна не заменяет другие:

* **машина состояний** отвергает переход, невозможный по жизненному циклу;
* **уникальный идентификатор события** гасит повторную доставку от движка;
* **оптимистичная блокировка** ловит конкурентное обновление той же строки.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session as DbSession

from socmon.core.codes import CallStage
from socmon.db.models import CallAttempt, CallEventRecord
from socmon.telephony.machine import CallEvent, CallState, Transition, apply


@dataclass(frozen=True, slots=True)
class Applied:
    transition: Transition
    duplicate: bool
    """Событие уже обрабатывалось. Не ошибка: доставка «хотя бы один раз» —
    норма, и повтор обязан быть безобидным."""


def state_of(attempt: CallAttempt) -> CallState:
    return CallState(
        stage=attempt.stage,
        disposition=attempt.disposition,
        answered=attempt.started_at is not None
        and attempt.stage
        in (CallStage.IN_INTERVIEW, CallStage.TRANSFERRING, CallStage.WITH_OPERATOR),
        transferred=attempt.stage is CallStage.WITH_OPERATOR,
    )


def record_transition(
    db: DbSession,
    attempt: CallAttempt,
    event: CallEvent,
    *,
    at: datetime,
    provider_event_id: str,
    detail: str | None = None,
) -> Applied:
    """Применить событие движка к попытке дозвона.

    Событие журналируется всегда — даже когда оно ничего не меняет: шкала
    времени попытки (ТЗ §10.1) должна отражать то, что действительно
    происходило на линии, а не только удачные переходы.
    """
    before = state_of(attempt)
    transition = apply(before, event)

    # Вставка идёт внутри точки сохранения, и объект создаётся там же.
    # Добавить его снаружи нельзя: откат точки сохранения не отцепит запись
    # от сессии, и следующий flush упадёт тем же нарушением уникальности.
    savepoint = db.begin_nested()
    db.add(CallEventRecord(
        attempt_id=attempt.id,
        provider_event_id=provider_event_id,
        at=at,
        kind=event.value,
        stage_before=before.stage.value,
        stage_after=transition.state.stage.value,
        detail=detail,
    ))
    try:
        db.flush()
    except IntegrityError:
        # Такое событие уже обработано: движок доставил его повторно.
        savepoint.rollback()
        return Applied(Transition(before, changed=False), duplicate=True)
    savepoint.commit()

    if not transition.changed:
        return Applied(transition, duplicate=False)

    attempt.stage = transition.state.stage
    attempt.disposition = transition.state.disposition
    if transition.state.stage is CallStage.IN_INTERVIEW and attempt.started_at is None:
        attempt.started_at = at
    if transition.state.finished:
        attempt.ended_at = at
        if attempt.started_at is not None:
            attempt.duration_sec = round((at - attempt.started_at).total_seconds(), 2)
    db.flush()
    return Applied(transition, duplicate=False)
