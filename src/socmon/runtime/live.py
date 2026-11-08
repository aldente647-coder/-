"""Живой звонок: связка движка анкеты, телефонии и распознавания.

До этого модуля части работали порознь: станция звонила, реплики играли,
распознавание мерилось, движок вёл разговор — но в одном настоящем звонке
вместе не собирались. Здесь они собираются.

Две вещи, которые в тестах не проявляются и всплывают только на живой линии:

**Реплику надо дожидаться.** Команда «проиграй» возвращается сразу, а звук
идёт ещё секунды. Начать слушать в этот момент — значит принять собственный
голос робота за ответ респондента и уйти по случайной ветке анкеты.

**Звук приходит отдельным соединением.** Asterisk сам подключается к нам
по TCP и шлёт разговор кадрами; распознавание живёт в этом потоке, а не
в потоке событий. Сшивается всё по идентификатору попытки.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import websockets

from socmon.asr.audiosocket import AudioSocketServer
from socmon.asr.contract import Recognizer, VadPolicy
from socmon.core.questionnaire import Questionnaire
from socmon.runtime.call import CallRuntime, RuntimePolicy
from socmon.runtime.ports import Journal
from socmon.telephony.asterisk import AsteriskProvider
from socmon.telephony.machine import CallEvent
from socmon.telephony.provider import (
    CallHandle,
    OriginateRequest,
    ProviderEvent,
    RecognizedSpeech,
    TelephonyError,
)

TICK_SEC = 1.0


@dataclass
class PlaybackWaiter:
    """Ожидание конца реплики.

    Asterisk ставит воспроизведения в очередь и по каждому присылает
    `PlaybackFinished`. Пока не пришло — робот ещё говорит.
    """

    _pending: dict[str, asyncio.Event] = field(default_factory=dict)
    _early: dict[str, None] = field(default_factory=dict)

    def expect(self, playback_id: str) -> None:
        self._pending.setdefault(playback_id, asyncio.Event())
        if playback_id in self._early:
            self._early.pop(playback_id)
            self._pending[playback_id].set()

    def finish(self, playback_id: str) -> None:
        event = self._pending.get(playback_id)
        if event is not None:
            event.set()
        else:
            # WebSocket может обогнать HTTP-ответ на команду play.
            self._early[playback_id] = None
            if len(self._early) > 128:
                self._early.pop(next(iter(self._early)))

    async def wait(self, playback_id: str, timeout: float = 60.0) -> None:
        event = self._pending.setdefault(playback_id, asyncio.Event())
        try:
            async with asyncio.timeout(timeout):
                await event.wait()
        except TimeoutError as exc:
            raise TelephonyError("не получено окончание аудиореплики") from exc
        finally:
            self._pending.pop(playback_id, None)

    def cancel_all(self) -> None:
        for event in self._pending.values():
            event.set()
        self._pending.clear()
        self._early.clear()


class SpeakingProvider:
    """Обёртка над движком телефонии: `play` ждёт, пока реплика договорит.

    Обёртка, а не правка провайдера: провайдер отвечает за команды АТС,
    а знание о том, что мы ждём окончания, принадлежит ходу разговора.
    """

    def __init__(self, inner: AsteriskProvider, waiter: PlaybackWaiter) -> None:
        self._inner = inner
        self._waiter = waiter

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def play(self, call: CallHandle, media: str) -> str | None:
        playback_id = await self._inner.play(call, media)
        if playback_id:
            self._waiter.expect(playback_id)
            try:
                await self._waiter.wait(playback_id)
            finally:
                self._inner.playback_finished(call, playback_id)
        return playback_id

    async def stop_playback(self, call: CallHandle) -> None:
        self._waiter.cancel_all()
        await self._inner.stop_playback(call)


class LiveCall:
    """Один живой звонок от набора до завершения."""

    def __init__(
        self,
        provider: AsteriskProvider,
        questionnaire: Questionnaire,
        journal: Journal,
        recognizer: Recognizer,
        *,
        audio_host: str = "127.0.0.1",
        audio_port: int = 0,
        policy: RuntimePolicy | None = None,
        vad: VadPolicy | None = None,
        operator_endpoint: str | None = None,
        on_event: Callable[[str], None] | None = None,
    ) -> None:
        self._provider = provider
        self._waiter = PlaybackWaiter()
        self._speaking = SpeakingProvider(provider, self._waiter)
        self._runtime = CallRuntime(
            self._speaking,  # type: ignore[arg-type]
            questionnaire,
            journal,
            policy=policy or RuntimePolicy(),
            operator_endpoint=operator_endpoint,
        )
        self._recognizer = recognizer
        self._audio_host = audio_host
        self._audio_port = audio_port
        self._vad = vad or VadPolicy()
        self._say = on_event or (lambda _: None)
        self._finished = asyncio.Event()
        self._inbox: asyncio.Queue[ProviderEvent | RecognizedSpeech | datetime] = asyncio.Queue(256)

    async def dial(
        self, destination: str, caller_id: str, *, timeout_sec: float = 900.0,
        attempt_id: uuid.UUID | None = None,
    ) -> uuid.UUID:
        """Набрать номер и провести интервью."""
        if timeout_sec <= 0:
            raise ValueError("тайм-аут звонка должен быть положительным")
        attempt_id = attempt_id or uuid.uuid4()
        self._finished.clear()
        self._inbox = asyncio.Queue(256)
        heard: asyncio.Queue[RecognizedSpeech] = asyncio.Queue(64)

        async def on_speech(speech: RecognizedSpeech) -> None:
            await heard.put(speech)

        audio = AudioSocketServer(
            self._recognizer, on_speech,
            host=self._audio_host, port=self._audio_port, vad=self._vad,
        )
        handle: CallHandle | None = None
        tasks: list[asyncio.Task[Any]] = []
        abort_event = CallEvent.TECH_ERROR
        try:
            await audio.start()
            stream_id = audio.expect(str(attempt_id))
            config = self._provider.config
            async with websockets.connect(
                config.websocket_url, additional_headers=config.websocket_headers,
            ) as socket:
                handle = await self._provider.originate(OriginateRequest(
                    attempt_id=str(attempt_id), destination=destination,
                    caller_id=caller_id, timeout_sec=30,
                ))
                self._runtime.register(attempt_id, handle)
                self._say(f"начата попытка {attempt_id}")
                tasks = [
                    asyncio.create_task(self._pump_events(socket, attempt_id, audio, stream_id)),
                    asyncio.create_task(self._pump_speech(heard)),
                    asyncio.create_task(self._tick()),
                    asyncio.create_task(self._dispatch(attempt_id)),
                    asyncio.create_task(self._finished.wait()),
                ]
                done, _ = await asyncio.wait(
                    tasks, timeout=timeout_sec, return_when=asyncio.FIRST_COMPLETED,
                )
                if not done:
                    abort_event = CallEvent.TIMEOUT
                    raise TelephonyError("время звонка вышло")
                for task in done:
                    task.result()
                if not self._finished.is_set():
                    raise TelephonyError("поток событий звонка закрылся")
        finally:
            self._runtime.cancel_prompts(attempt_id)
            self._waiter.cancel_all()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            try:
                if handle is not None:
                    try:
                        await self._provider.hangup(handle)
                    finally:
                        await self._runtime.abort_call(attempt_id, abort_event)
            finally:
                await audio.stop()
        return attempt_id

    async def _pump_events(
        self, socket: Any, attempt_id: uuid.UUID, audio: AudioSocketServer, stream_id: str
    ) -> None:
        async for raw in socket:
            payload = json.loads(raw)
            kind = payload.get("type")

            if kind == "PlaybackFinished":
                self._waiter.finish(str((payload.get("playback") or {}).get("id")))
                continue

            live = self._runtime.call_of(attempt_id)
            if live is None:
                return
            # События снупов, внешнего медиа и чужих звонков не являются
            # ответом или завершением основного канала.
            event = await self._provider.event_for_call(payload, live.handle)
            if event is None:
                continue

            if event.event.value == "answer":
                # Звук подключаем до первой реплики: вступление с
                # информированием о согласии — часть записи (ТЗ §10.2).
                await self._provider.attach_audio(
                    live.handle, host=self._audio_host,
                    port=audio.port, stream_id=stream_id,
                )
            if event.event in {
                CallEvent.HANGUP, CallEvent.TECH_ERROR, CallEvent.TIMEOUT,
                CallEvent.BUSY, CallEvent.NO_ANSWER, CallEvent.INVALID_NUMBER,
            }:
                self._runtime.cancel_prompts(attempt_id)
                self._waiter.cancel_all()
            self._say(f"событие: {event.event.value}")
            await self._inbox.put(event)

        raise TelephonyError("соединение событий АТС закрыто")

    async def _pump_speech(self, heard: asyncio.Queue[RecognizedSpeech]) -> None:
        while True:
            speech = await heard.get()
            await self._runtime.interrupt_speech(speech)
            await self._inbox.put(speech)

    async def _dispatch(self, attempt_id: uuid.UUID) -> None:
        """Изменения интервью последовательны; чтение WebSocket независимо."""
        while not self._finished.is_set():
            item = await self._inbox.get()
            if isinstance(item, ProviderEvent):
                await self._runtime.on_call_event(item)
            elif isinstance(item, RecognizedSpeech):
                await self._runtime.on_speech(item)
            else:
                await self._runtime.on_tick(item)
            if self._runtime.call_of(attempt_id) is None:
                self._finished.set()

    async def _tick(self) -> None:
        """Тайм-аут тишины: респондент молчит, значит надо переспросить."""
        while True:
            await asyncio.sleep(TICK_SEC)
            if self._inbox.empty():
                await self._inbox.put(datetime.now(UTC))
