"""Порты рантайма.

Рантайм не импортирует ни SQLAlchemy, ни Asterisk: и то и другое приходит
через узкие контракты. Это не украшение — иначе цикл звонка невозможно
проверить, не подняв базу и АТС.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from socmon.core.engine import InterviewSession
from socmon.telephony.machine import CallEvent


@dataclass(frozen=True, slots=True)
class Claim:
    """Контакт, взятый в набор.

    Номер уже расшифрован: диалеру он нужен, чтобы набрать. Дальше по
    системе едет только `attempt_id` — в аналитический контур номер
    не попадает никогда (ТЗ §14.2).
    """

    attempt_id: uuid.UUID
    contact_id: uuid.UUID
    phone: str
    attempt_no: int


class ContactSource(Protocol):
    """Откуда диалер берёт контакты."""

    def claim(self, wave_id: uuid.UUID, limit: int, now: datetime) -> Sequence[Claim]:
        """Атомарно взять до `limit` контактов, готовых к набору.

        Атомарность обязательна: без неё два процесса наберут один номер
        одновременно. Реализация помечает контакты «в наборе» тем же
        запросом, которым выбирает.
        """
        ...

    def release(self, attempt_id: uuid.UUID, now: datetime) -> None:
        """Вернуть контакт в очередь: набрать не удалось до соединения."""
        ...


class Journal(Protocol):
    """Куда рантайм пишет ход звонка."""

    def transition(
        self,
        attempt_id: uuid.UUID,
        event: CallEvent,
        *,
        at: datetime,
        provider_event_id: str,
        detail: str | None = None,
    ) -> bool:
        """Записать переход. Возвращает False, если событие уже обработано."""
        ...

    def save_interview(self, attempt_id: uuid.UUID, session: InterviewSession) -> None:
        """Сохранить ответы и исход завершённого интервью."""
        ...

    def attach_recording(
        self, attempt_id: uuid.UUID, respondent: str | None, robot: str | None
    ) -> None:
        ...


class CallerIdPool(Protocol):
    """Карусель исходящих номеров.

    С одного номера операторы связи блокируют быстро — это подтверждено
    и расчётом (CONCEPT.md §8.1), и блюпринтом VOIF.
    """

    def take(self) -> str: ...

    def burn(self, caller_id: str) -> None:
        """Вывести номер из ротации: похоже, его заблокировали."""
        ...

    @property
    def available(self) -> int: ...
