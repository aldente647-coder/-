"""Мост от Asterisk к распознаванию: протокол AudioSocket.

Взят вместо UniMRCP сознательно. UniMRCP — это отдельный сервер, обёртка
модели в MRCP и ещё один демон в контуре; AudioSocket встроен в Asterisk
и представляет собой TCP с кадрами из четырёх байт заголовка. Меньше
движущихся частей — меньше мест, где волна встанет ночью.

Схема работы: рантайм просит Asterisk подключить к каналу внешнее медиа,
Asterisk сам приходит к нам TCP-клиентом и первым же кадром называет
идентификатор потока. По нему звук связывается с попыткой дозвона.

Звук идёт сырыми отсчётами 8 кГц — ровно тот формат, в котором озвучена
анкета и в котором работает распознавание. Ни одного перекодирования
на пути.
"""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import IntEnum

from socmon.asr.contract import (
    DEFAULT_VAD,
    Recognizer,
    SpeechSegmenter,
    VadPolicy,
)
from socmon.telephony.provider import RecognizedSpeech

HEADER_BYTES = 3
MAX_PAYLOAD = 65535


class FrameType(IntEnum):
    TERMINATE = 0x00
    IDENTIFIER = 0x01
    """Первый кадр соединения: 16 байт идентификатора потока."""

    SILENCE = 0x02
    AUDIO = 0x10
    ERROR = 0xFF


class AudioSocketError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Frame:
    type: FrameType
    payload: bytes

    def encode(self) -> bytes:
        if len(self.payload) > MAX_PAYLOAD:
            raise AudioSocketError(f"кадр длиннее {MAX_PAYLOAD} байт")
        return bytes([self.type]) + len(self.payload).to_bytes(2, "big") + self.payload


def decode_header(header: bytes) -> tuple[FrameType, int]:
    """Разобрать заголовок кадра: тип и длина полезной нагрузки.

    Неизвестный тип не считается ошибкой соединения: Asterisk вправе
    прислать служебный кадр, о котором мы не знаем, и рвать из-за этого
    живой звонок — плохой обмен.
    """
    if len(header) != HEADER_BYTES:
        raise AudioSocketError(f"заголовок должен быть {HEADER_BYTES} байта")
    raw, length = header[0], int.from_bytes(header[1:3], "big")
    try:
        return FrameType(raw), length
    except ValueError:
        return FrameType.ERROR, length


SpeechHandler = Callable[[RecognizedSpeech], Awaitable[None]]


@dataclass
class AudioStream:
    """Один звуковой поток — один разговор."""

    attempt_id: str
    segmenter: SpeechSegmenter
    frames: int = 0
    bytes_received: int = 0
    segments: int = 0


class AudioSocketServer:
    """Принимает звук от Asterisk и отдаёт распознанные реплики.

    Распознавание выполняется в отдельном потоке: модель считает
    десятки-сотни миллисекунд, и держать на это время цикл событий значит
    остановить все остальные звонки разом.
    """

    def __init__(
        self,
        recognizer: Recognizer,
        on_speech: SpeechHandler,
        *,
        host: str = "127.0.0.1",
        port: int = 8090,
        vad: VadPolicy = DEFAULT_VAD,
    ) -> None:
        self._recognizer = recognizer
        self._on_speech = on_speech
        self._host = host
        self._port = port
        self._vad = vad
        self._connections: set[asyncio.Task[None]] = set()
        self._streams: dict[str, str] = {}
        self._server: asyncio.Server | None = None
        self.active: dict[str, AudioStream] = {}

    def expect(self, attempt_id: str) -> str:
        """Зарегистрировать поток до того, как Asterisk подключится.

        Идентификатор передаётся Asterisk при создании внешнего медиа
        и возвращается первым кадром — так звук связывается с попыткой,
        а не угадывается по порядку подключений.
        """
        stream_id = str(uuid.uuid4())
        self._streams[stream_id] = attempt_id
        return stream_id

    def forget(self, stream_id: str) -> None:
        self._streams.pop(stream_id, None)

    async def start(self) -> None:
        self._server = await asyncio.start_server(
            self._serve, host=self._host, port=self._port
        )

    @property
    def port(self) -> int:
        """Фактический порт. Нужен, когда сервер поднят на нулевом:
        Asterisk должен знать, куда подключаться."""
        sockets = getattr(self._server, "sockets", None)
        if not sockets:
            raise AudioSocketError("сервер не запущен")
        port: int = sockets[0].getsockname()[1]
        return port

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        for task in self._connections:
            task.cancel()
        await asyncio.gather(*self._connections, return_exceptions=True)
        self._streams.clear()

    # ------------------------------------------------------------ соединение

    async def _serve(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        stream: AudioStream | None = None
        task = asyncio.current_task()
        assert task is not None
        self._connections.add(task)
        try:
            async with asyncio.timeout(5):
                stream = await self._handshake(reader)
            if stream is None:
                return
            self.active[stream.attempt_id] = stream
            await self._pump(reader, stream)
        except (AudioSocketError, asyncio.IncompleteReadError, ConnectionError, TimeoutError):
            # Обрыв звукового потока не должен ронять весь сервер: остальные
            # разговоры продолжаются, а этот звонок закроет сторож по времени.
            pass
        finally:
            try:
                if stream is not None:
                    if not task.cancelling():
                        await self._finish(stream)
                    self.active.pop(stream.attempt_id, None)
            finally:
                self._connections.discard(task)
                writer.close()
                with contextlib.suppress(ConnectionError):
                    await writer.wait_closed()

    async def _handshake(self, reader: asyncio.StreamReader) -> AudioStream | None:
        """Прочитать первый кадр с идентификатором потока."""
        kind, length = decode_header(await reader.readexactly(HEADER_BYTES))
        if kind is not FrameType.IDENTIFIER or length != 16:
            raise AudioSocketError("первым кадром ожидался UUID длиной 16 байт")
        payload = await reader.readexactly(length)

        stream_id = str(uuid.UUID(bytes=payload))
        attempt_id = self._streams.pop(stream_id, None)
        if attempt_id is None:
            # Чужой поток. Молча закрываем: подключиться мог кто угодно,
            # и обрабатывать неизвестный звук нельзя.
            return None
        return AudioStream(attempt_id=attempt_id, segmenter=SpeechSegmenter(self._vad))

    async def _pump(self, reader: asyncio.StreamReader, stream: AudioStream) -> None:
        while True:
            async with asyncio.timeout(30):
                kind, length = decode_header(await reader.readexactly(HEADER_BYTES))
                payload = await reader.readexactly(length) if length else b""

            if kind is FrameType.TERMINATE:
                return
            if kind is not FrameType.AUDIO or not payload:
                continue
            if length % 2:
                raise AudioSocketError("кадр PCM содержит неполный отсчёт")

            stream.frames += 1
            stream.bytes_received += len(payload)
            for segment in stream.segmenter.feed(payload):
                await self._recognize(stream, segment)

    async def _finish(self, stream: AudioStream) -> None:
        """Дораспознать реплику, оборванную закрытием канала."""
        tail = stream.segmenter.flush()
        if tail:
            await self._recognize(stream, tail)

    async def _recognize(self, stream: AudioStream, segment: bytes) -> None:
        stream.segments += 1
        hypothesis = await asyncio.to_thread(self._recognizer.transcribe, segment)
        text = hypothesis.text.strip()
        await self._on_speech(RecognizedSpeech(
            attempt_id=stream.attempt_id,
            text=text or None,
            at=datetime.now(UTC),
            final=True,
            silence=not text,
            confidence=hypothesis.confidence,
        ))
