"""Интерфейс движка телефонии.

Бизнес-логика вызывает «набери номер», «проиграй реплику», «переведи на
оператора» и не знает имени провайдера. Имя конкретного движка не должно
встречаться в коде за пределами этого пакета — быстрая проверка чистоты слоя:
`grep asterisk` вне `telephony/` должен молчать.

Зачем это нам, а не «как в блюпринте»: договор с оператором связи, возможно,
придётся переоформлять на Заказчика (RESEARCH.md §4), а номерная ёмкость —
менять при выгорании номеров. Всё это должно быть сменой реализации,
а не переписыванием системы.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, runtime_checkable

from socmon.telephony.machine import CallEvent


@dataclass(frozen=True, slots=True)
class Capabilities:
    """Что умеет конкретная реализация.

    Ядро не должно строить догадок: если движок не разделяет каналы записи,
    это ограничение выгрузки по ТЗ §10.2, и знать о нём надо до волны,
    а не после.
    """

    separate_recording_channels: bool
    """Раздельная запись «робот» и «респондент» (ТЗ §10.2)."""

    answering_machine_detection: bool
    barge_in: bool
    operator_transfer: bool
    caller_id_rotation: bool
    """Карусель исходящих номеров: с одного номера операторы связи блокируют
    быстро (CONCEPT.md §8.1)."""


@dataclass(frozen=True, slots=True)
class OriginateRequest:
    """Задание на звонок."""

    attempt_id: str
    """Наш идентификатор попытки. Уходит движку и возвращается в событиях —
    по нему события сшиваются с базой без догадок."""

    destination: str
    caller_id: str
    timeout_sec: int = 30
    detect_answering_machine: bool = True
    variables: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CallHandle:
    attempt_id: str
    provider_call_id: str


@dataclass(frozen=True, slots=True)
class Recording:
    """Записанный разговор (ТЗ §10.1: ссылка на аудиозапись)."""

    respondent_uri: str | None
    robot_uri: str | None
    mixed_uri: str | None = None

    @property
    def channels_separated(self) -> bool:
        return self.respondent_uri is not None and self.robot_uri is not None


@dataclass(frozen=True, slots=True)
class ProviderEvent:
    """Событие движка, переведённое на язык нашей машины состояний."""

    attempt_id: str
    event: CallEvent
    at: datetime
    provider_call_id: str | None = None
    event_id: str | None = None
    """Идентификатор события у провайдера. Нужен для дедупликации: доставка
    «хотя бы один раз» — норма, и повтор не должен сработать дважды."""

    detail: str | None = None


@dataclass(frozen=True, slots=True)
class RecognizedSpeech:
    """Что респондент сказал, по мнению распознавания.

    Назван не Utterance, чтобы не путаться с репликой, которую принимает
    движок анкеты: там это уже решение о ходе диалога, здесь — сырой
    результат распознавания с гипотезой и уверенностью.
    """

    attempt_id: str
    text: str | None
    at: datetime
    final: bool = True
    """Промежуточная гипотеза или окончательная.

    Перебивание ловится именно на промежуточных: ждать финальной — значит
    реагировать на секунду позже и звучать как глухой собеседник.
    """

    silence: bool = False
    confidence: float | None = None


class TelephonyError(RuntimeError):
    pass


@runtime_checkable
class TelephonyProvider(Protocol):
    """Контракт движка телефонии."""

    def capabilities(self) -> Capabilities: ...

    async def originate(self, request: OriginateRequest) -> CallHandle: ...

    async def play(self, call: CallHandle, media: str) -> None:
        """Проиграть заранее отрендеренную реплику.

        Именно файл, а не синтез на лету: ТЗ §8.1 требует дословного
        воспроизведения утверждённых формулировок, §8.3 — стабильности голоса
        между волнами, и то и другое даёт только пререндер.
        """

    async def stop_playback(self, call: CallHandle) -> None:
        """Оборвать реплику: респондент перебил (ТЗ §8.1)."""

    async def start_recording(self, call: CallHandle) -> None: ...

    async def recording(self, call: CallHandle) -> Recording: ...

    async def transfer_to_operator(self, call: CallHandle, endpoint: str) -> None: ...

    async def hangup(self, call: CallHandle) -> None: ...

    def events(self) -> AsyncIterator[ProviderEvent]: ...

    def utterances(self) -> AsyncIterator[RecognizedSpeech]:
        """Поток распознанной речи.

        Здесь всплывают результаты MRCP-распознавания: Asterisk отдаёт их
        через UniMRCP, а движок анкеты решает, что с ними делать. Сама
        интерпретация ответа сюда не попадает — это граница слоя.
        """
        ...
