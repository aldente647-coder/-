"""Цикл одного звонка.

Переводит события телефонии в ходы движка анкеты и обратно — команды движку
телефонии. Ни одного решения о содержании разговора здесь нет: что спросить
дальше, знает только `socmon.core.engine`, а рантайм лишь озвучивает
его решение и передаёт ответ.

Три вещи, которые легко сделать неправильно и которые тут сделаны явно:

* **перебивание** ловится на промежуточной гипотезе распознавания, но не
  раньше, чем через заданную задержку от начала реплики — иначе кашель
  обрывает вступление, и люди принимают это за поломку;
* **тишина** отличается от нераспознанной речи: в журнал они идут разными
  событиями, хотя движок анкеты трактует их одинаково;
* **исход интервью не выводится из состояния звонка**. Респондент, положивший
  трубку на середине, даёт прерванное интервью, а не «завершённое с обрывом».
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from socmon.core.codes import Outcome
from socmon.core.engine import Action, Answer, InterviewSession, Turn
from socmon.core.engine import Utterance as EngineUtterance
from socmon.core.questionnaire import Questionnaire
from socmon.runtime.ports import Journal
from socmon.telephony.machine import CallEvent, CallState, apply
from socmon.telephony.provider import (
    CallHandle,
    ProviderEvent,
    RecognizedSpeech,
    TelephonyProvider,
)
from socmon.voice.delivery import (
    Catalogue,
    Delivery,
    Gender,
    Kind,
    Segment,
    repeat_reason,
    take_media,
)
from socmon.voice.render import VoiceManifest

Clock = Callable[[], datetime]


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class RuntimePolicy:
    barge_in_delay_sec: float = 3.0
    """Сколько робот договаривает, прежде чем реагировать на перебивание."""

    silence_timeout_sec: float = 6.0
    nudge_after_sec: float = 3.5
    """Через сколько секунд тишины окликнуть: «алло, вы меня слышите?».

    Раньше тайм-аута: живой интервьюер не ждёт шесть секунд молча,
    а после оклика человеку легче ответить, чем после повтора вопроса."""

    min_confidence: float = 0.0
    """Ниже этой уверенности гипотеза считается нераспознанной.

    Ноль означает «доверять распознаванию». Порог подбирается на пилоте:
    §5.3 бракует интервью с систематически нераспознанной речью, и цена
    ошибки здесь — потерянное интервью, а не просто переспрос.
    """


DEFAULT_POLICY = RuntimePolicy()


@dataclass
class LiveCall:
    """Состояние одного идущего звонка."""

    attempt_id: uuid.UUID
    handle: CallHandle
    call: CallState = field(default_factory=CallState)
    interview: InterviewSession | None = None
    delivery: Delivery | None = None
    """Подача этого звонка: отклики, паузы, дубли. None — без подачи,
    реплики играют впритык, как до неё."""

    last_answer: Answer | None = None
    nudged: bool = False
    speaking_since: datetime | None = None
    """Когда робот начал текущую реплику. Нужно для задержки перебивания."""

    awaiting_since: datetime | None = None
    """Когда робот замолчал и стал ждать ответа. Нужно для тайм-аута тишины."""

    prompts_pending: int = 0

    @property
    def finished(self) -> bool:
        return self.call.finished

    def interview_awaiting(self) -> str | None:
        """Узел, ответа на который ждёт движок анкеты."""
        if self.interview is None or self.interview.finished:
            return None
        return self.interview.current_node_id


class CallRuntime:
    """Ведёт активные звонки."""

    def __init__(
        self,
        provider: TelephonyProvider,
        questionnaire: Questionnaire,
        journal: Journal,
        *,
        policy: RuntimePolicy = DEFAULT_POLICY,
        operator_endpoint: str | None = None,
        clock: Clock = _utcnow,
        catalogue: Catalogue | None = None,
        library: VoiceManifest | None = None,
        media_dir: Path | None = None,
        gender: Gender = "f",
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        """`catalogue` включает подачу. `library` и `media_dir` — чтобы знать,
        сколько дублей есть у реплики и есть ли запись отклика вовсе: подача
        не должна просить у АТС файл, которого нет. `sleep` подменяется
        в тестах — ждать настоящие паузы там незачем."""
        self._provider = provider
        self._questionnaire = questionnaire
        self._journal = journal
        self._policy = policy
        self._operator_endpoint = operator_endpoint
        self._clock = clock
        self._catalogue = catalogue
        self._library = library
        self._media_dir = media_dir
        self._gender = gender
        self._sleep = sleep
        self._calls: dict[uuid.UUID, LiveCall] = {}

    @property
    def active(self) -> int:
        return len(self._calls)

    def call_of(self, attempt_id: uuid.UUID) -> LiveCall | None:
        return self._calls.get(attempt_id)

    def register(self, attempt_id: uuid.UUID, handle: CallHandle) -> LiveCall:
        """Взять звонок под наблюдение сразу после набора."""
        live = LiveCall(attempt_id=attempt_id, handle=handle)
        if self._catalogue is not None:
            live.delivery = Delivery(
                self._catalogue, self._questionnaire, gender=self._gender,
                seed=str(attempt_id), takes=self._max_takes(),
            )
        self._calls[attempt_id] = live
        return live

    async def abort_call(self, attempt_id: uuid.UUID, event: CallEvent) -> None:
        """Закрыть незавершённый звонок при потере транспорта или тайм-ауте."""
        live = self._calls.get(attempt_id)
        if live is not None:
            if not live.finished:
                await self._journal_event(live, event, self._clock())
            await self._close(live, self._clock())

    async def interrupt_speech(self, speech: RecognizedSpeech) -> None:
        """Прервать воспроизведение из читателя звука, до обработки ответа."""
        live = self._calls.get(uuid.UUID(speech.attempt_id))
        if live is not None:
            await self._maybe_barge_in(live, speech)

    def cancel_prompts(self, attempt_id: uuid.UUID) -> None:
        live = self._calls.get(attempt_id)
        if live is not None:
            live.prompts_pending = 0

    # ------------------------------------------------------- события телефонии

    async def on_call_event(self, event: ProviderEvent) -> None:
        """Применить событие движка телефонии."""
        attempt_id = uuid.UUID(event.attempt_id)
        live = self._calls.get(attempt_id)
        if live is None:
            return

        fresh = self._journal.transition(
            attempt_id,
            event.event,
            at=event.at,
            provider_event_id=event.event_id or f"{event.event.value}:{event.at.isoformat()}",
            detail=event.detail,
        )
        if not fresh:
            # Движок доставил событие повторно — это норма, а не ошибка.
            return

        transition = apply(live.call, event.event)
        live.call = transition.state
        if not transition.changed:
            return

        if event.event is CallEvent.ANSWER:
            await self._start_interview(live, event.at)
        elif live.finished:
            await self._close(live, event.at)

    async def _start_interview(self, live: LiveCall, at: datetime) -> None:
        """Респондент снял трубку.

        Запись включается до первой реплики: ТЗ §10.2 требует записи всего
        разговора, а вступление с информированием о согласии — его часть
        и юридически самая важная.
        """
        await self._provider.start_recording(live.handle)
        live.interview = InterviewSession(self._questionnaire, clock=self._clock)
        await self._speak(live, live.interview.begin(), at)

    # ---------------------------------------------------------------- речь

    async def on_speech(self, speech: RecognizedSpeech) -> None:
        """Обработать результат распознавания."""
        attempt_id = uuid.UUID(speech.attempt_id)
        live = self._calls.get(attempt_id)
        if live is None or live.interview is None or live.finished:
            return

        if not speech.final:
            await self._maybe_barge_in(live, speech)
            return
        if live.prompts_pending:
            # Робот ещё говорит, а гипотеза уже финальная: перебивание
            # обработается ниже, реплику надо оборвать.
            await self._stop_speaking(live)

        await self._answer(live, self._to_engine(speech), speech.at)

    async def _maybe_barge_in(self, live: LiveCall, speech: RecognizedSpeech) -> None:
        """Оборвать реплику, если респондент действительно заговорил.

        Задержка от начала фразы обязательна: мгновенная реакция на любой
        звук обрывает приветствие на кашле, и это читается как поломка.
        """
        if not live.prompts_pending or live.speaking_since is None:
            return
        if speech.silence or not (speech.text or "").strip():
            return
        elapsed = (speech.at - live.speaking_since).total_seconds()
        if elapsed < self._policy.barge_in_delay_sec:
            return
        await self._stop_speaking(live)

    def _to_engine(self, speech: RecognizedSpeech) -> EngineUtterance:
        """Перевести гипотезу распознавания в реплику для движка анкеты.

        Низкая уверенность приравнивается к молчанию: движок переспросит
        ровно один раз (§8.1) и, если снова не разберёт, поставит код
        нераспознанной речи — вместо того чтобы записать догадку как ответ.
        """
        if speech.silence or speech.text is None:
            return EngineUtterance.quiet()
        if (
            speech.confidence is not None
            and speech.confidence < self._policy.min_confidence
        ):
            return EngineUtterance.quiet()
        return EngineUtterance.said(speech.text)

    async def on_tick(self, now: datetime | None = None) -> None:
        """Проверить тайм-ауты тишины по всем идущим звонкам."""
        moment = now or self._clock()
        for live in list(self._calls.values()):
            if live.interview is None or live.finished or live.prompts_pending:
                continue
            if live.awaiting_since is None:
                continue
            waited = moment - live.awaiting_since
            if waited > timedelta(seconds=self._policy.silence_timeout_sec):
                await self._answer(live, EngineUtterance.quiet(), moment)
            elif (
                live.delivery is not None and not live.nudged
                and waited > timedelta(seconds=self._policy.nudge_after_sec)
            ):
                live.nudged = True
                nudge = live.delivery.nudge()
                if self._has_media(nudge.key):
                    live.prompts_pending = 1
                    live.speaking_since = self._clock()
                    await self._provider.play(live.handle, self._media(nudge))
                    live.prompts_pending = 0

    # ------------------------------------------------------------- продвижение

    async def _answer(
        self, live: LiveCall, utterance: EngineUtterance, at: datetime
    ) -> None:
        assert live.interview is not None
        before = set(live.interview.answers)
        turn = live.interview.submit(utterance)
        fresh = [a for k, a in live.interview.answers.items() if k not in before]
        live.last_answer = fresh[-1] if fresh else None
        await self._speak(live, turn, at)

    async def _speak(self, live: LiveCall, turn: Turn, at: datetime) -> None:
        """Произнести реплики хода и решить, что делать дальше."""
        live.awaiting_since = None
        live.nudged = False
        if live.delivery is None:
            live.prompts_pending = len(turn.prompts)
            for prompt in turn.prompts:
                if not prompt.text.strip():
                    continue
                live.speaking_since = self._clock()
                await self._provider.play(live.handle, prompt.audio_key or prompt.node_id)
                if not live.prompts_pending:
                    break
        else:
            assert live.interview is not None
            segments = live.delivery.plan(
                turn, previous=live.last_answer,
                reason=repeat_reason(live.interview.events),
            )
            # Отклик без записи в библиотеке молча пропускается;
            # вопрос анкеты играет всегда — его отсутствие ловит проверка
            # готовности до запуска волны.
            playable = [
                s for s in segments if s.kind is Kind.PROMPT or self._has_media(s.key)
            ]
            live.prompts_pending = len(playable)
            for segment in playable:
                if segment.gap_ms:
                    await self._sleep(segment.gap_ms / 1000)
                if not live.prompts_pending:
                    break  # перебили во время паузы
                live.speaking_since = self._clock()
                await self._provider.play(live.handle, self._media(segment))
                if not live.prompts_pending:
                    break
        live.prompts_pending = 0

        if turn.action is Action.TRANSFER:
            await self._transfer(live, at)
            return
        if turn.action is Action.END:
            await self._finish(live, at)
            return
        live.awaiting_since = self._clock()

    def _has_media(self, key: str) -> bool:
        if self._library is None:
            return True
        return key in self._library.entries

    def _media(self, segment: Segment) -> str:
        entry = self._library.entries.get(segment.key) if self._library else None
        takes = entry.takes if entry is not None else ()
        return take_media(segment.key, segment.take, takes, self._media_dir)

    def _max_takes(self) -> int:
        if self._library is None:
            return 1
        return max((len(e.takes) + 1 for e in self._library.entries.values()), default=1)

    async def _stop_speaking(self, live: LiveCall) -> None:
        await self._provider.stop_playback(live.handle)
        live.prompts_pending = 0

    async def _transfer(self, live: LiveCall, at: datetime) -> None:
        """Перевести на дежурного оператора (ТЗ §9.2)."""
        # Перевод фиксируется всегда, даже когда оператора не нашлось: иначе
        # «нет оператора» приходит из стадии интервью, минуя стадию перевода,
        # и в шкале времени звонка появляется дыра.
        await self._journal_event(live, CallEvent.TRANSFER_REQUEST, at)
        if self._operator_endpoint is None:
            # Оператора нет — честно завершаем звонок, а не молчим в трубку.
            await self._journal_event(live, CallEvent.NO_OPERATOR, at)
            await self._provider.hangup(live.handle)
            await self._close(live, at)
            return
        await self._provider.transfer_to_operator(live.handle, self._operator_endpoint)

    async def _finish(self, live: LiveCall, at: datetime) -> None:
        await self._journal_event(live, CallEvent.COMPLETE, at)
        await self._provider.hangup(live.handle)
        await self._close(live, at)

    async def _journal_event(
        self, live: LiveCall, event: CallEvent, at: datetime
    ) -> None:
        """Записать переход, инициированный нами, а не движком телефонии."""
        transition = apply(live.call, event)
        self._journal.transition(
            live.attempt_id,
            event,
            at=at,
            provider_event_id=f"runtime:{live.attempt_id}:{event.value}",
        )
        live.call = transition.state

    async def _close(self, live: LiveCall, at: datetime) -> None:
        """Снять звонок с наблюдения и сохранить интервью.

        Интервью сохраняется и незавершённое: прерванный разговор — это
        данные, а не мусор. Валидность решается правилами §5.3 отдельно.
        """
        if live.interview is not None:
            if not live.interview.finished:
                live.interview.abort(Outcome.PARTIAL)
            recording = await self._provider.recording(live.handle)
            self._journal.attach_recording(
                live.attempt_id, recording.respondent_uri, recording.robot_uri
            )
            self._journal.save_interview(live.attempt_id, live.interview)
        self._calls.pop(live.attempt_id, None)
