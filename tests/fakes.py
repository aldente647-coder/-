"""Подставные реализации портов.

Позволяют прогнать весь цикл звонка и набора без Asterisk и без базы:
именно то, что можно проверить до аренды транка.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from socmon.core.engine import InterviewSession
from socmon.runtime.ports import Claim
from socmon.telephony.machine import CallEvent
from socmon.telephony.provider import (
    CallHandle,
    Capabilities,
    OriginateRequest,
    ProviderEvent,
    RecognizedSpeech,
    Recording,
    TelephonyError,
)

START = datetime(2026, 8, 20, 18, 30, tzinfo=UTC)


class FakeClock:
    def __init__(self, start: datetime = START, step_sec: float = 1.0) -> None:
        self.now = start
        self.step = timedelta(seconds=step_sec)

    def __call__(self) -> datetime:
        current = self.now
        self.now += self.step
        return current

    def advance(self, seconds: float) -> datetime:
        self.now += timedelta(seconds=seconds)
        return self.now


@dataclass
class FakeProvider:
    """Движок телефонии, который всё записывает и ничего не звонит."""

    played: list[tuple[str, str]] = field(default_factory=list)
    recorded: list[str] = field(default_factory=list)
    transferred: list[tuple[str, str]] = field(default_factory=list)
    hungup: list[str] = field(default_factory=list)
    stopped: list[str] = field(default_factory=list)
    originated: list[OriginateRequest] = field(default_factory=list)
    fail_originate: bool = False

    def capabilities(self) -> Capabilities:
        return Capabilities(True, True, True, True, True)

    async def originate(self, request: OriginateRequest) -> CallHandle:
        if self.fail_originate:
            raise TelephonyError("транк недоступен")
        self.originated.append(request)
        return CallHandle(request.attempt_id, f"chan-{len(self.originated)}")

    async def play(self, call: CallHandle, media: str) -> None:
        self.played.append((call.attempt_id, media))

    async def stop_playback(self, call: CallHandle) -> None:
        self.stopped.append(call.attempt_id)

    async def start_recording(self, call: CallHandle) -> None:
        self.recorded.append(call.attempt_id)

    async def recording(self, call: CallHandle) -> Recording:
        return Recording(
            respondent_uri=f"rec:{call.attempt_id}-respondent.wav",
            robot_uri=f"rec:{call.attempt_id}-robot.wav",
        )

    async def transfer_to_operator(self, call: CallHandle, endpoint: str) -> None:
        self.transferred.append((call.attempt_id, endpoint))

    async def hangup(self, call: CallHandle) -> None:
        self.hungup.append(call.attempt_id)

    async def events(self) -> AsyncIterator[ProviderEvent]:  # pragma: no cover
        """Поток не используется: рантайм в тестах кормится вручную."""
        empty: tuple[ProviderEvent, ...] = ()
        for event in empty:
            yield event

    async def utterances(self) -> AsyncIterator[RecognizedSpeech]:  # pragma: no cover
        empty: tuple[RecognizedSpeech, ...] = ()
        for speech in empty:
            yield speech

    def media_for(self, attempt_id: str) -> list[str]:
        return [m for a, m in self.played if a == attempt_id]


@dataclass
class FakeJournal:
    """Журнал в памяти с той же дедупликацией, что и в базе."""

    seen: set[str] = field(default_factory=set)
    transitions: list[tuple[uuid.UUID, CallEvent]] = field(default_factory=list)
    interviews: dict[uuid.UUID, InterviewSession] = field(default_factory=dict)
    recordings: dict[uuid.UUID, tuple[str | None, str | None]] = field(
        default_factory=dict
    )

    def transition(
        self, attempt_id: uuid.UUID, event: CallEvent, *,
        at: datetime, provider_event_id: str, detail: str | None = None,
    ) -> bool:
        if provider_event_id in self.seen:
            return False
        self.seen.add(provider_event_id)
        self.transitions.append((attempt_id, event))
        return True

    def save_interview(self, attempt_id: uuid.UUID, session: InterviewSession) -> None:
        self.interviews[attempt_id] = session

    def attach_recording(
        self, attempt_id: uuid.UUID, respondent: str | None, robot: str | None
    ) -> None:
        self.recordings[attempt_id] = (respondent, robot)

    def events_for(self, attempt_id: uuid.UUID) -> list[CallEvent]:
        return [e for a, e in self.transitions if a == attempt_id]


@dataclass
class FakeContactSource:
    """Источник контактов с атомарной пометкой «в наборе»."""

    queue: list[Claim] = field(default_factory=list)
    released: list[uuid.UUID] = field(default_factory=list)

    def claim(self, wave_id: uuid.UUID, limit: int, now: datetime) -> Sequence[Claim]:
        taken, self.queue = self.queue[:limit], self.queue[limit:]
        return taken

    def release(self, attempt_id: uuid.UUID, now: datetime) -> None:
        self.released.append(attempt_id)

    def fill(self, count: int, attempt_no: int = 1) -> None:
        self.queue.extend(
            Claim(
                attempt_id=uuid.uuid4(),
                contact_id=uuid.uuid4(),
                phone=f"+7949000{i:04d}",
                attempt_no=attempt_no,
            )
            for i in range(count)
        )
