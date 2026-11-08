"""Машина состояний звонка.

Блюпринт VOIF заканчивается прямым предупреждением: не делать первый звонок,
пока она не готова, — «без неё система разваливается в первый же день обзвона,
и чинить это по факту дороже, чем заложить сразу». Речь о гонках между
диалером и вебхуками движка: звонки начинают мерцать между статусами,
операторы получают перевод дважды, один контакт набирается параллельно.

Здесь машина чистая: ни базы, ни сети. Она отвечает на единственный вопрос —
допустим ли переход, — и потому проверяется полностью.

Три оси из блюпринта разведены и тут: событие меняет **стадию**, а причину
обрыва (**disposition**) выставляет только там, где связь действительно
подвела. Исход интервью — третья ось — живёт в движке анкеты и сюда
не попадает вовсе.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from enum import StrEnum

from socmon.core.codes import CallStage, Disposition


class IllegalTransition(RuntimeError):
    """Переход запрещён жизненным циклом. Это ошибка кода, не данных."""


class CallEvent(StrEnum):
    """События, приходящие от движка телефонии."""

    RING = "ring"
    ANSWER = "answer"
    ANSWERING_MACHINE = "answering_machine"
    """Автоответчик распознан на линии — первая линия защиты.
    Вторая, разбор транскрипта задним числом, работает в ночном пайплайне."""

    TRANSFER_REQUEST = "transfer_request"
    OPERATOR_ANSWER = "operator_answer"
    NO_OPERATOR = "no_operator"
    COMPLETE = "complete"
    BUSY = "busy"
    NO_ANSWER = "no_answer"
    INVALID_NUMBER = "invalid_number"
    TECH_ERROR = "tech_error"
    HANGUP = "hangup"
    TIMEOUT = "timeout"


@dataclass(frozen=True, slots=True)
class CallState:
    stage: CallStage = CallStage.DIALING
    disposition: Disposition | None = None
    answered: bool = False
    """Дошло ли до живого разговора. Нужно, чтобы отличить недозвон
    от прерванного разговора: для регламента повторов это разные вещи."""

    transferred: bool = False

    @property
    def finished(self) -> bool:
        return self.stage is CallStage.FINISHED


@dataclass(frozen=True, slots=True)
class Transition:
    state: CallState
    changed: bool
    """False означает, что событие пришло повторно и ничего не меняет.

    Доставка «хотя бы один раз» — норма для вебхуков, поэтому повтор обязан
    быть безобидным. Ошибкой считается не повтор, а переход, невозможный
    по жизненному циклу.
    """


#: Куда можно уйти из каждой стадии.
_ALLOWED: dict[CallStage, frozenset[CallStage]] = {
    CallStage.DIALING: frozenset({
        CallStage.RINGING, CallStage.IN_INTERVIEW, CallStage.FINISHED,
    }),
    CallStage.RINGING: frozenset({CallStage.IN_INTERVIEW, CallStage.FINISHED}),
    CallStage.IN_INTERVIEW: frozenset({CallStage.TRANSFERRING, CallStage.FINISHED}),
    CallStage.TRANSFERRING: frozenset({CallStage.WITH_OPERATOR, CallStage.FINISHED}),
    CallStage.WITH_OPERATOR: frozenset({CallStage.FINISHED}),
    CallStage.FINISHED: frozenset(),
}

#: Событие → (новая стадия, причина обрыва). Причина пуста там, где связь
#: отработала штатно и разговор закончился по существу.
_EFFECT: dict[CallEvent, tuple[CallStage, Disposition | None]] = {
    CallEvent.RING: (CallStage.RINGING, None),
    CallEvent.ANSWER: (CallStage.IN_INTERVIEW, None),
    CallEvent.ANSWERING_MACHINE: (CallStage.FINISHED, Disposition.ANSWERING_MACHINE),
    CallEvent.TRANSFER_REQUEST: (CallStage.TRANSFERRING, None),
    CallEvent.OPERATOR_ANSWER: (CallStage.WITH_OPERATOR, None),
    CallEvent.NO_OPERATOR: (CallStage.FINISHED, Disposition.NO_OPERATOR),
    CallEvent.COMPLETE: (CallStage.FINISHED, None),
    CallEvent.BUSY: (CallStage.FINISHED, Disposition.BUSY),
    CallEvent.NO_ANSWER: (CallStage.FINISHED, Disposition.NO_ANSWER),
    CallEvent.INVALID_NUMBER: (CallStage.FINISHED, Disposition.INVALID_NUMBER),
    CallEvent.TECH_ERROR: (CallStage.FINISHED, Disposition.TECH_ERROR),
    CallEvent.TIMEOUT: (CallStage.FINISHED, Disposition.TIMEOUT),
    CallEvent.HANGUP: (CallStage.FINISHED, Disposition.DROPPED),
}


#: Из каких стадий событие вообще осмысленно.
#:
#: Одной таблицы разрешённых стадий мало. Она пропускает, например,
#: «интервью завершено» из стадии набора: формально переход в FINISHED
#: допустим, а по смыслу это пометило бы недозвон как состоявшееся интервью
#: и испортило бы расчёт достижимости.
_VALID_FROM: dict[CallEvent, frozenset[CallStage]] = {
    CallEvent.RING: frozenset({CallStage.DIALING}),
    CallEvent.ANSWER: frozenset({CallStage.DIALING, CallStage.RINGING}),
    CallEvent.ANSWERING_MACHINE: frozenset(
        {CallStage.DIALING, CallStage.RINGING, CallStage.IN_INTERVIEW}
    ),
    CallEvent.TRANSFER_REQUEST: frozenset({CallStage.IN_INTERVIEW}),
    CallEvent.OPERATOR_ANSWER: frozenset({CallStage.TRANSFERRING}),
    CallEvent.NO_OPERATOR: frozenset({CallStage.TRANSFERRING}),
    CallEvent.COMPLETE: frozenset(
        {CallStage.IN_INTERVIEW, CallStage.TRANSFERRING, CallStage.WITH_OPERATOR}
    ),
    CallEvent.BUSY: frozenset({CallStage.DIALING, CallStage.RINGING}),
    CallEvent.NO_ANSWER: frozenset({CallStage.DIALING, CallStage.RINGING}),
    CallEvent.INVALID_NUMBER: frozenset({CallStage.DIALING, CallStage.RINGING}),
    # Сбой, тайм-аут и обрыв возможны на любой живой стадии.
    CallEvent.TECH_ERROR: frozenset(_ALLOWED) - {CallStage.FINISHED},
    CallEvent.TIMEOUT: frozenset(_ALLOWED) - {CallStage.FINISHED},
    CallEvent.HANGUP: frozenset(_ALLOWED) - {CallStage.FINISHED},
}


def apply(state: CallState, event: CallEvent) -> Transition:
    """Применить событие движка к состоянию звонка.

    Единственная точка смены стадии. Прямая запись стадии в обход этой
    функции запрещена конвенцией и код-ревью — ровно как в блюпринте.
    """
    target, disposition = _EFFECT[event]

    if state.finished:
        # Терминальное состояние поглощает всё: движок вправе прислать
        # «канал уничтожен» после того, как мы уже закрыли звонок сами.
        return Transition(state, changed=False)

    if target is state.stage and target is not CallStage.FINISHED:
        return Transition(state, changed=False)

    if state.stage not in _VALID_FROM[event] or target not in _ALLOWED[state.stage]:
        raise IllegalTransition(
            f"событие «{event.value}» невозможно в стадии «{state.stage.value}»"
        )

    if event is CallEvent.HANGUP and state.answered:
        # Респондент положил трубку в разговоре. Связь не подвела — это
        # прерванное интервью, а не технический обрыв, и повторять его
        # регламент повторов не должен.
        disposition = None

    return Transition(
        replace(
            state,
            stage=target,
            disposition=disposition,
            answered=state.answered or target is CallStage.IN_INTERVIEW,
            transferred=state.transferred or target is CallStage.WITH_OPERATOR,
        ),
        changed=True,
    )


# --------------------------------------------------------------------- сторож


@dataclass(frozen=True, slots=True)
class WatchdogPolicy:
    """Пределы, после которых звонок считается зависшим.

    Приём из блюпринта: без сторожа зависший звонок занимает оператора
    и засоряет отчёты, а закрыть его некому — движок о нём уже забыл.
    """

    dialing_sec: float = 15.0
    ringing_sec: float = 45.0
    interview_sec: float = 900.0
    """Полтора лимита интервью: анкета рассчитана на 10 минут (ТЗ §3.3),
    всё сверх — признак застрявшего диалога, а не разговорчивого респондента."""

    transferring_sec: float = 45.0
    with_operator_sec: float = 1800.0
    silence_sec: float = 60.0
    """Сколько можно не слышать звук в активном разговоре, прежде чем
    считать канал мёртвым."""


DEFAULT_WATCHDOG = WatchdogPolicy()


_LIMITS: dict[CallStage, str] = {
    CallStage.DIALING: "dialing_sec",
    CallStage.RINGING: "ringing_sec",
    CallStage.IN_INTERVIEW: "interview_sec",
    CallStage.TRANSFERRING: "transferring_sec",
    CallStage.WITH_OPERATOR: "with_operator_sec",
}


def stale_event(
    state: CallState,
    *,
    stage_since: datetime,
    now: datetime,
    last_media_at: datetime | None = None,
    policy: WatchdogPolicy = DEFAULT_WATCHDOG,
) -> CallEvent | None:
    """Проверить, не завис ли звонок. Возвращает событие, которым его закрыть.

    Разведены две причины: превышение лимита стадии даёт TIMEOUT, пропавший
    звук в активном разговоре — TECH_ERROR. Смешивать их нельзя: первое
    бывает штатно, второе всегда означает поломку, и повторять такой номер
    регламент не должен.
    """
    if state.finished:
        return None

    limit = getattr(policy, _LIMITS[state.stage])
    if now - stage_since > timedelta(seconds=limit):
        return CallEvent.TIMEOUT

    if state.stage in (CallStage.IN_INTERVIEW, CallStage.WITH_OPERATOR):
        if last_media_at is None:
            return None
        if now - last_media_at > timedelta(seconds=policy.silence_sec):
            return CallEvent.TECH_ERROR

    return None


#: Причины обрыва, по которым повтор осмыслен (ТЗ §6.2).
#:
#: Дублирует RETRYABLE из domain.schedule намеренной проверкой: список причин
#: живёт в регламенте дозвона, а здесь — утверждение, что телефонный слой
#: не изобретает своих.
def retryable(disposition: Disposition | None) -> bool:
    from socmon.domain.schedule import RETRYABLE

    return disposition is not None and disposition.value in RETRYABLE
