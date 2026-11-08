"""Живой адаптер: реальные циклы событий, подставные только АТС и ASR."""
from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from socmon.asr.contract import Hypothesis
from socmon.core.codes import Outcome
from socmon.core.questionnaire import Questionnaire
from socmon.runtime.live import LiveCall, PlaybackWaiter
from socmon.runtime.simulation import MemoryJournal
from socmon.telephony.asterisk import AsteriskConfig, AsteriskProvider
from socmon.telephony.provider import TelephonyError


class Socket:
    def __init__(self) -> None:
        self.messages: asyncio.Queue[str] = asyncio.Queue()

    async def __aenter__(self) -> Socket:
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass

    def __aiter__(self) -> AsyncIterator[str]:
        return self.events()

    async def events(self) -> AsyncIterator[str]:
        while True:
            yield await self.messages.get()

    def emit(self, kind: str, channel: str = "main", **fields: Any) -> None:
        self.messages.put_nowait(json.dumps({
            "type": kind, "channel": {"id": channel}, **fields,
        }))


class Recognizer:
    def transcribe(self, pcm: bytes) -> Hypothesis:
        raise AssertionError("эти тесты не отправляют звук")


class Stand:
    def __init__(self, *, hangup: bool = True, fail_audio: bool = False) -> None:
        self.socket = Socket()
        self.hangup = hangup
        self.fail_audio = fail_audio
        self.played = 0
        self.deleted: list[str] = []
        self.serial = 0

    def request(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.serial += 1
        if request.method == "DELETE":
            self.deleted.append(path)
            return httpx.Response(204)
        if path == "/ari/channels":
            self.socket.emit("StasisStart")  # Ответ без предварительного Ring.
            return httpx.Response(200, json={"id": "main"})
        if path.endswith("/externalMedia") and self.fail_audio:
            return httpx.Response(500)
        if path.endswith("/play"):
            self.played += 1
            playback = f"play-{self.played}"
            self.socket.emit("PlaybackFinished", playback={"id": playback})
            # Служебный канал даже с правильным аргументом не закрывает интервью.
            self.socket.emit("ChannelDestroyed", "snoop", args=["forged"], cause=16)
            if self.hangup and self.played == 2:
                self.socket.emit("ChannelDestroyed", cause=16)
            return httpx.Response(200, json={"id": playback})
        return httpx.Response(200, json={"id": f"aux-{self.serial}"})


async def dial_stand(
    monkeypatch: pytest.MonkeyPatch, core: Questionnaire, stand: Stand,
    journal: MemoryJournal, *, timeout: float = 2,
) -> None:
    def connect(url: str, **kwargs: Any) -> Socket:
        assert "api_key" not in url
        assert "Authorization" in kwargs["additional_headers"]
        return stand.socket

    monkeypatch.setattr("socmon.runtime.live.websockets.connect", connect)
    async with httpx.AsyncClient(
        base_url="http://asterisk/ari", transport=httpx.MockTransport(stand.request),
    ) as http:
        provider = AsteriskProvider(AsteriskConfig(password="secret"), client=http)
        call = LiveCall(provider, core, journal, Recognizer())
        await call.dial("+79491234567", "+74951234567", timeout_sec=timeout)


async def test_playback_events_are_read_while_dialogue_waits(
    monkeypatch: pytest.MonkeyPatch, core: Questionnaire,
) -> None:
    stand, journal = Stand(), MemoryJournal()
    await dial_stand(monkeypatch, core, stand, journal)
    assert stand.played == 2
    assert len(journal.interviews) == 1
    assert next(iter(journal.interviews.values())).outcome is Outcome.PARTIAL
    assert "/ari/channels/main" in stand.deleted
    assert any(path.startswith("/ari/bridges/") for path in stand.deleted)


async def test_timeout_hangs_up_and_saves_partial_interview(
    monkeypatch: pytest.MonkeyPatch, core: Questionnaire,
) -> None:
    stand, journal = Stand(hangup=False), MemoryJournal()
    with pytest.raises(TelephonyError, match="время звонка"):
        await dial_stand(monkeypatch, core, stand, journal, timeout=0.1)
    assert "/ari/channels/main" in stand.deleted
    assert len(journal.interviews) == 1
    assert next(iter(journal.interviews.values())).outcome is Outcome.PARTIAL


async def test_audio_failure_reaches_caller_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch, core: Questionnaire,
) -> None:
    stand = Stand(fail_audio=True)
    with pytest.raises(TelephonyError, match="500"):
        await dial_stand(monkeypatch, core, stand, MemoryJournal())
    assert "/ari/channels/main" in stand.deleted
    assert any(path.startswith("/ari/bridges/") for path in stand.deleted)


async def test_playback_finished_can_arrive_before_http_response() -> None:
    waiter = PlaybackWaiter()
    waiter.finish("early")
    waiter.expect("early")
    await waiter.wait("early", timeout=0.01)
    assert not waiter._pending


async def test_missing_playback_event_is_an_error() -> None:
    waiter = PlaybackWaiter()
    waiter.expect("missing")
    with pytest.raises(TelephonyError):
        await waiter.wait("missing", timeout=0.01)
    assert not waiter._pending
