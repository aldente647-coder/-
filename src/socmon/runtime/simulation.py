"""Подставной движок телефонии для прогона всего цикла без Asterisk.

Нужен не только для тестов. Из него же вырастает контрольный звонок из
блюпринта VOIF — искусственный звонок, проходящий весь путь как настоящий,
чтобы проверить цепочку перед волной и после обновления. Такие звонки
помечаются тестовыми и не попадают ни в статистику, ни в аналитический
массив: тестовое интервью в валидном массиве нарушает ТЗ §5.3.
"""

from __future__ import annotations

import random
import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from socmon.core.codes import Disposition
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
)

#: Как отвечает «средний» респондент. Ключ — идентификатор узла анкеты.
SCRIPT: dict[str, list[str]] = {
    "consent": ["да", "хорошо", "ладно"],
    "screen_age": ["да"],
    "screen_territory": ["да", "проживаю"],
    "q_life": ["шесть", "семь", "пять"],
    "q_family_change": ["не изменилось", "ухудшилось"],
    "q_material": ["четыре", "пять", "три"],
    "q_safety": ["семь", "шесть"],
    "q_anxiety": ["восемь", "семь", "девять"],
    "q_trust": ["четыре", "три"],
    "q_fairness": ["четыре", "пять"],
    "q_conflict": ["шесть", "пять"],
    "q_authorities": ["пять", "четыре"],
    "q_local_problem": ["цены", "медицина", "дороги"],
    "q_collective": ["скорее готов", "не готов", "готов"],
    "q_open_leader": ["затрудняюсь назвать", "местный депутат"],
    "q_open_worry": ["цены растут, зарплата стоит"],
    "d_gender": ["женский", "мужской"],
    "d_age_group": ["мне сорок два", "мне двадцать пять", "мне шестьдесят"],
    "d_territory": ["центральный", "северный", "южный"],
    "d_settlement": ["город", "село"],
    "d_education": ["высшее", "среднее специальное"],
    "d_employment": ["работаю", "на пенсии"],
    "d_nationality": ["русский", "затрудняюсь"],
    "d_politics": ["иногда", "не интересуюсь"],
}


@dataclass
class SimulatedProvider:
    """Движок, который «звонит» в память."""

    seed: int = 20260820
    answer_rate: float = 0.30
    """Доля номеров, где вообще снимают трубку."""

    machine_rate: float = 0.15
    """Доля снявших, оказавшихся автоответчиком."""

    played: list[tuple[str, str]] = field(default_factory=list)
    hungup: list[str] = field(default_factory=list)
    _rng: random.Random = field(init=False)
    _pending: list[ProviderEvent] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def capabilities(self) -> Capabilities:
        return Capabilities(True, True, True, True, True)

    async def originate(self, request: OriginateRequest) -> CallHandle:
        return CallHandle(request.attempt_id, f"sim-{request.attempt_id[:8]}")

    async def play(self, call: CallHandle, media: str) -> None:
        self.played.append((call.attempt_id, media))

    async def stop_playback(self, call: CallHandle) -> None: ...
    async def start_recording(self, call: CallHandle) -> None: ...

    async def recording(self, call: CallHandle) -> Recording:
        return Recording(
            respondent_uri=f"sim:{call.attempt_id}-respondent.wav",
            robot_uri=f"sim:{call.attempt_id}-robot.wav",
        )

    async def transfer_to_operator(self, call: CallHandle, endpoint: str) -> None: ...

    async def hangup(self, call: CallHandle) -> None:
        self.hungup.append(call.attempt_id)

    async def events(self) -> AsyncIterator[ProviderEvent]:  # pragma: no cover
        for event in list(self._pending):
            yield event

    async def utterances(self) -> AsyncIterator[RecognizedSpeech]:  # pragma: no cover
        empty: tuple[RecognizedSpeech, ...] = ()
        for speech in empty:
            yield speech

    # ------------------------------------------------------------ сценарий линии

    def outcome_for(self, attempt_id: str, at: datetime) -> list[ProviderEvent]:
        """Что произойдёт на линии по этому номеру.

        Доли недозвона взяты из порядка величин, ожидаемого для робо-обзвона
        (RESEARCH.md §2): подавляющее большинство номеров не отвечает.
        """
        def build(kind: CallEvent, offset: float) -> ProviderEvent:
            return ProviderEvent(
                attempt_id=attempt_id, event=kind,
                at=at + timedelta(seconds=offset),
                event_id=f"{attempt_id}:{kind.value}",
            )

        roll = self._rng.random()
        events = [build(CallEvent.RING, 1)]
        if roll > self.answer_rate:
            events.append(build(
                CallEvent.BUSY if roll > 0.9 else CallEvent.NO_ANSWER, 25
            ))
        elif self._rng.random() < self.machine_rate:
            events.append(build(CallEvent.ANSWERING_MACHINE, 6))
        else:
            events.append(build(CallEvent.ANSWER, 6))
        return events

    def reply_for(self, node_id: str) -> str:
        options = SCRIPT.get(node_id)
        return self._rng.choice(options) if options else "затрудняюсь"

    def hangs_up(self, probability: float) -> bool:
        return self._rng.random() < probability


def sample_claims(count: int, rng: random.Random) -> list[Claim]:
    return [
        Claim(
            attempt_id=uuid.uuid4(),
            contact_id=uuid.uuid4(),
            phone=f"+7949{rng.randint(1000000, 9999999)}",
            attempt_no=1,
        )
        for _ in range(count)
    ]


@dataclass
class MemoryContactSource:
    queue: list[Claim] = field(default_factory=list)
    released: list[uuid.UUID] = field(default_factory=list)

    def claim(self, wave_id: uuid.UUID, limit: int, now: datetime) -> Sequence[Claim]:
        taken, self.queue = self.queue[:limit], self.queue[limit:]
        return taken

    def release(self, attempt_id: uuid.UUID, now: datetime) -> None:
        self.released.append(attempt_id)


@dataclass
class MemoryJournal:
    """Журнал в памяти: та же дедупликация, что и в базе."""

    seen: set[str] = field(default_factory=set)
    dispositions: dict[uuid.UUID, Disposition | None] = field(default_factory=dict)
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
        return True

    def save_interview(self, attempt_id: uuid.UUID, session: InterviewSession) -> None:
        self.interviews[attempt_id] = session

    def attach_recording(
        self, attempt_id: uuid.UUID, respondent: str | None, robot: str | None
    ) -> None:
        self.recordings[attempt_id] = (respondent, robot)
