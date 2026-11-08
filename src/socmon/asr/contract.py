"""Контракт распознавателя и сегментация речи.

Важная оговорка о замысле. Раньше в концепции говорилось о «пограмматичном
распознавании» — ограничении словаря на каждом вопросе. С GigaAM так
не выйдет: это end-to-end модель, внешнюю грамматику в её декодер
не подставить.

Поэтому схема другая и, пожалуй, честнее: распознаём свободно, а ограничение
применяем к тексту — грамматиками узлов из `core/grammar.py`. Не совпало
ни с одним вариантом — переспрашиваем один раз и ставим код нераспознанной
речи, а не записываем догадку. Побочный выигрыш: сырой текст сохраняется
целиком и уходит в ночной разбор, чего при грамматичном декодировании
не было бы.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

#: Телефонный канал: 8 кГц, моно, 16 бит. То же, подо что озвучена анкета.
SAMPLE_RATE = 8000
SAMPLE_WIDTH = 2
FRAME_MS = 20
FRAME_BYTES = SAMPLE_RATE * SAMPLE_WIDTH * FRAME_MS // 1000


class RecognitionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Hypothesis:
    """Что услышала модель."""

    text: str
    confidence: float | None = None
    final: bool = True


@runtime_checkable
class Recognizer(Protocol):
    """Модель распознавания.

    Принимает целый сегмент речи, а не поток: сегментацию делает VAD,
    и это разделение позволяет менять модель, не трогая логику диалога.
    """

    def transcribe(self, pcm: bytes) -> Hypothesis: ...


@dataclass(frozen=True, slots=True)
class VadPolicy:
    """Пороги нарезки речи на реплики."""

    energy_threshold: int = 500
    """Порог энергии кадра. Подбирается на пилоте под конкретный транк:
    у разных операторов заметно разный уровень сигнала."""

    speech_frames_to_start: int = 5
    """Сколько подряд «громких» кадров считать началом речи. Сто миллисекунд
    отсекают щелчки линии, но не отсекают короткое «да»."""

    silence_ms_to_end: int = 700
    """Пауза, после которой реплика считается законченной.

    Значение — компромисс. Меньше — робот перебивает задумавшегося
    респондента. Больше — заметная задержка после каждого ответа,
    и интервью перестаёт укладываться в лимит ТЗ §3.3.
    """

    max_utterance_ms: int = 45_000
    """Потолок на одну реплику. Нужен для открытых вопросов (ТЗ §7.5),
    где респондент может говорить долго, и как страховка от «залипшего»
    канала, который шумит без остановки."""

    min_utterance_ms: int = 200
    """Короче этого — щелчок или вздох, а не ответ.

    Меряется по звучащей части, а не по длине куска. Иначе щелчок линии
    вместе с приклеенной к нему паузой проходит порог и уходит
    в распознавание как полноценная реплика.
    """

    keep_tail_ms: int = 200
    """Сколько тишины оставить в конце сегмента.

    Немного нужно: модели легче на затухании слова. Всю паузу тащить
    незачем — это лишние сотни миллисекунд счёта на каждой реплике.
    """


DEFAULT_VAD = VadPolicy()


def frame_energy(frame: bytes) -> int:
    """Средняя амплитуда кадра.

    Считается вручную по 16-битным отсчётам: тянуть numpy ради модуля
    и деления незачем, а на 50 кадрах в секунду разница незаметна.
    """
    if not frame:
        return 0
    total = 0
    for index in range(0, len(frame) - 1, SAMPLE_WIDTH):
        sample = int.from_bytes(frame[index:index + SAMPLE_WIDTH], "little", signed=True)
        total += abs(sample)
    return total // (len(frame) // SAMPLE_WIDTH)


@dataclass
class SpeechSegmenter:
    """Нарезает поток кадров на реплики по паузам.

    Простой энергетический детектор, а не нейросетевой. Причина
    прагматичная: он работает на телефонном канале предсказуемо, ничего
    не грузит и его поведение объяснимо на разборе инцидента. Если
    на пилоте выяснится, что шумные линии его обманывают, заменяется
    на Silero VAD без изменения остального кода.
    """

    policy: VadPolicy = DEFAULT_VAD
    _buffer: bytearray = None  # type: ignore[assignment]
    _speaking: bool = False
    _loud_run: int = 0
    _silent_ms: int = 0
    _spoken_ms: int = 0
    _loud_ms: int = 0

    def __post_init__(self) -> None:
        self._buffer = bytearray()

    @property
    def speaking(self) -> bool:
        return self._speaking

    def feed(self, frame: bytes) -> Iterator[bytes]:
        """Скормить кадр. Отдаёт законченные реплики."""
        loud = frame_energy(frame) >= self.policy.energy_threshold

        if not self._speaking:
            self._loud_run = self._loud_run + 1 if loud else 0
            if self._loud_run >= self.policy.speech_frames_to_start:
                self._speaking = True
                self._silent_ms = 0
                self._spoken_ms = self._loud_run * FRAME_MS
                self._loud_ms = self._spoken_ms
                self._buffer.extend(frame)
            return

        self._buffer.extend(frame)
        self._spoken_ms += FRAME_MS
        if loud:
            self._loud_ms += FRAME_MS
            self._silent_ms = 0
        else:
            self._silent_ms += FRAME_MS

        ended = self._silent_ms >= self.policy.silence_ms_to_end
        overlong = self._spoken_ms >= self.policy.max_utterance_ms
        if ended or overlong:
            segment = self._close()
            if segment is not None:
                yield segment

    def flush(self) -> bytes | None:
        """Забрать недоговорённое: канал закрылся посреди реплики."""
        return self._close() if self._speaking else None

    def _close(self) -> bytes | None:
        segment = bytes(self._buffer)
        loud_ms, trailing_ms = self._loud_ms, self._silent_ms

        self._buffer.clear()
        self._speaking = False
        self._loud_run = 0
        self._silent_ms = 0
        self._spoken_ms = 0
        self._loud_ms = 0

        if loud_ms < self.policy.min_utterance_ms:
            return None

        # Обрезаем хвостовую тишину, оставляя немного на затухание слова.
        extra_ms = trailing_ms - self.policy.keep_tail_ms
        if extra_ms > 0:
            cut = extra_ms * SAMPLE_RATE * SAMPLE_WIDTH // 1000
            segment = segment[:-cut] if cut < len(segment) else segment
        return segment
