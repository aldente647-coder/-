"""Синтезатор речи.

Контракт отделён от реализации намеренно: логика озвучки анкеты обязана
проверяться без torch и без скачивания моделей. Silero подключается адаптером.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable
from xml.sax.saxutils import escape

#: Частота дискретизации телефонного канала.
#:
#: Восемь килогерц — не экономия, а честность: ровно это услышит респондент.
#: Демонстрировать Заказчику 24 кГц значит показывать звук, которого в системе
#: не будет, и получить утверждение голоса, который на линии зазвучит иначе.
TELEPHONY_SAMPLE_RATE = 8000

#: Частоты, которые поддерживает Silero.
SUPPORTED_RATES = (8000, 24000, 48000)

#: Голоса Silero для русского языка (модель v4_ru).
#:
#: ТЗ §8.3 требует представить до пилота не менее двух вариантов — мужской
#: и женский, нейтрального среднего возраста, в спокойной манере.
RUSSIAN_VOICES: dict[str, str] = {
    "aidar": "мужской",
    "eugene": "мужской",
    "baya": "женский",
    "kseniya": "женский",
    "xenia": "женский",
}

DEFAULT_FEMALE = "baya"
DEFAULT_MALE = "eugene"

_WHITESPACE = re.compile(r"\s+")


class SynthesisError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Speech:
    """Синтезированная реплика."""

    wav: bytes
    sample_rate: int
    duration_sec: float


@runtime_checkable
class SpeechSynthesizer(Protocol):
    def synthesize(self, text: str, *, voice: str, sample_rate: int) -> Speech: ...


def prepare(text: str) -> str:
    """Привести формулировку к виду, пригодному для синтеза.

    Схлопывает переносы, оставшиеся от YAML. Ничего, кроме пробелов,
    не трогает: любая правка текста здесь означала бы, что робот произносит
    не то, что утверждено (ТЗ §8.1).
    """
    return _WHITESPACE.sub(" ", text).strip()


class SileroSynthesizer:
    """Silero TTS v4 для русского языка.

    Модель грузится один раз и держится в памяти: на CPU одна секунда звука
    считается быстрее, чем произносится, так что вся анкета озвучивается
    за минуты и без видеокарты.
    """

    def __init__(
        self, *, model_id: str = "v4_ru", threads: int = 4,
        model_path: Path | None = None, natural: bool = False,
    ) -> None:
        self._model_id = model_id
        self._model_path = model_path
        self._natural = natural
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest() if model_path else "hub"
        self.fingerprint = f"silero:{model_id}:{digest}:" + (
            "calm-v2" if natural else "plain-v1"
        )
        self._threads = threads
        self._model: Any | None = None
        self._loaded_rate: int | None = None

    def _load(self, sample_rate: int) -> Any:
        if self._model is not None:
            return self._model
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - зависит от окружения
            raise SynthesisError(
                "не установлен torch: синтез недоступен. "
                "Поставьте зависимости группы voice"
            ) from exc

        torch.set_num_threads(self._threads)
        if self._model_path is not None:
            model = torch.package.PackageImporter(  # type: ignore[attr-defined]
                str(self._model_path)
            ).load_pickle(
                "tts_models", "model"
            )
        else:
            model, _ = torch.hub.load(  # type: ignore[no-untyped-call]
                repo_or_dir="snakers4/silero-models",
                model="silero_tts",
                language="ru",
                speaker=self._model_id,
                trust_repo=True,
            )
        model.to(torch.device("cpu"))
        self._model = model
        self._loaded_rate = sample_rate
        return model

    def synthesize(self, text: str, *, voice: str, sample_rate: int) -> Speech:
        if voice not in RUSSIAN_VOICES:
            raise SynthesisError(
                f"неизвестный голос «{voice}»; доступны: {', '.join(RUSSIAN_VOICES)}"
            )
        if sample_rate not in SUPPORTED_RATES:
            raise SynthesisError(
                f"частота {sample_rate} не поддерживается; "
                f"доступны: {', '.join(map(str, SUPPORTED_RATES))}"
            )
        prepared = prepare(text)
        if not prepared:
            raise SynthesisError("пустая реплика")

        model = self._load(sample_rate)
        options: dict[str, Any] = {"speaker": voice, "sample_rate": sample_rate}
        if self._natural:
            options.update(ssml_text=calm_ssml(prepared), put_accent=True, put_yo=True)
        else:
            options["text"] = prepared
        audio = model.apply_tts(**options)
        return Speech(
            wav=to_wav(audio.numpy(), sample_rate),
            sample_rate=sample_rate,
            duration_sec=len(audio) / sample_rate,
        )


def to_wav(samples: Any, sample_rate: int) -> bytes:
    """Собрать 16-битный моно WAV.

    Пишем вручную, а не через torchaudio: лишняя зависимость ради одного
    заголовка не нужна, а Asterisk ждёт именно такой формат.
    """
    import io
    import wave

    scaled = (samples.clip(-1.0, 1.0) * 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(scaled.tobytes())
    return buffer.getvalue()


def calm_ssml(text: str) -> str:
    """Небольшие смысловые паузы; слова и высота голоса остаются исходными.

    Слишком медленный SSML rate=slow растягивает русскую речь. Используем
    естественный темп диктора и явную короткую паузу только между фразами.
    XML из текста никогда не становится управляющей разметкой.
    """
    spoken = prepare(text)
    # Русское произношение названий; исходная анкета и её хеш не меняются.
    # В v5_5_ru латиница внутри SSML вызывает ошибку ext_alph=None.
    for name, pronunciation in (("Telegram", "Телеграм"), ("Max", "Макс")):
        spoken = re.sub(r"\b" + name + r"\b", pronunciation, spoken, flags=re.IGNORECASE)
    sentences = re.split(r"(?<=[.!?])\s+(?=[А-ЯЁ])", spoken)
    return '<speak>' + '<break time="250ms"/>'.join(map(escape, sentences)) + '</speak>'
