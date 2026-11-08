"""Предварительная озвучка Qwen3-TTS на Apple Silicon через MLX.

На телефонном сервере этот модуль не вызывается: он проигрывает готовые WAV.
Референсы — созданные синтетические голоса, а не записи реальных людей.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any

from socmon.voice.clean import at_rate, clean_master
from socmon.voice.spoken import spoken
from socmon.voice.synth import SUPPORTED_RATES, Speech, SynthesisError, prepare, to_wav

QWEN_VOICES = {"denis": "мужской", "anna": "женский", "alex": "мужской"}


def files_digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode())
        with path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
    return digest.hexdigest()


class QwenMlxSynthesizer:
    """Стабильный тембр по фиксированному синтетическому референсу."""

    def __init__(
        self, model: Path, references: Path, cache: Path, retakes: Path | None = None,
        *,
        takes: int = 1,
        temperature: float = 0.7,
        shaped: bool = False,
        keep_breath: bool = False,
        room_tone_db: float | None = None,
    ) -> None:
        """`takes` — сколько дублей просить на одну реплику; `shaped` — читать
        формулировку по речевой разметке (паузы в перечислениях, произношение
        названий), слова при этом те же; `keep_breath` и `room_tone_db` —
        чистка, оставляющая вдох и комнатный тон вместо цифрового нуля.

        Всё это входит в отпечаток рецепта: дубль, собранный по другим
        правилам, — другая запись, и из кэша он не подойдёт."""
        if not 1 <= takes <= 8:
            raise SynthesisError("дублей на реплику — от одного до восьми")
        self.takes = takes
        self._temperature = temperature
        self._shaped = shaped
        self._keep_breath = keep_breath
        self._room_tone_db = room_tone_db
        if not (model / "config.json").is_file():
            raise SynthesisError("нужен локальный каталог модели Qwen3-TTS Base MLX")
        refs = [
            references / f"{voice}.{ext}" for voice in QWEN_VOICES for ext in ("wav", "txt")
            if (references / f"{voice}.wav").is_file()
        ]
        if not refs:
            raise SynthesisError(
                "в каталоге референсов нет ни одной пары <голос>.wav + <голос>.txt"
            )
        self.available = {p.stem for p in refs if p.suffix == ".wav"}
        weights = list(model.rglob("*.safetensors"))
        if not weights:
            raise SynthesisError("в каталоге модели нет весов")
        version = importlib.metadata.version("mlx-audio")
        cleaning = "clean-v2" if (keep_breath or room_tone_db is not None) else "clean-v1"
        self.fingerprint = (
            f"qwen3-tts-1.7b-base:mlx-audio-{version}:"
            f"{files_digest(weights + list(model.glob('*.json')))}:"
            f"{files_digest(refs)}:{cleaning}:temp-{temperature:g}"
            + (":shaped-v1" if shaped else "")
            + (f":breath:tone{room_tone_db:g}" if cleaning == "clean-v2" else "")
            + (f":takes-{takes}" if takes > 1 else "")
        )
        self._recipe = self.fingerprint
        self._retakes: dict[str, int] = json.loads(retakes.read_text()) if retakes else {}
        if not isinstance(self._retakes, dict) or any(
            not isinstance(k, str) or type(v) is not int or not 1 <= v <= 100
            for k, v in self._retakes.items()
        ):
            raise SynthesisError("карта дублей должна содержать пары голос:хеш → число 1–100")
        if self._retakes:
            digest = hashlib.sha256(json.dumps(self._retakes, sort_keys=True).encode()).hexdigest()
            self.fingerprint += f":retakes-{digest}"
        self._model_path = model
        self._references = references
        self._cache = cache
        self._model: Any = None

    def synthesize(
        self, text: str, *, voice: str, sample_rate: int, take: int = 0,
    ) -> Speech:
        if voice not in self.available or sample_rate not in SUPPORTED_RATES:
            raise SynthesisError("нет референса для этого голоса или частота не поддерживается")
        if not 0 <= take < self.takes:
            raise SynthesisError(f"дубль {take} вне запрошенных {self.takes}")
        prepared = prepare(text)
        if not prepared:
            raise SynthesisError("пустая реплика")
        import numpy as np

        # Хеш и карта дублей — по утверждённой формулировке; произносится
        # размеченный вариант с теми же словами.
        to_speak = spoken(prepared) if self._shaped else prepared
        text_digest = hashlib.sha256(prepared.encode()).hexdigest()[:16]
        retake = self._retakes.get(f"{voice}:{text_digest}", 0)
        recipe = (
            self._recipe + voice + to_speak
            + (f":take-{retake}" if retake else "")
            + (f":alt-{take}" if take else "")
        )
        key = hashlib.sha256(recipe.encode()).hexdigest()
        cached = self._cache / f"{key}.npy"
        if cached.is_file():
            master = np.load(cached, allow_pickle=False)
        else:
            import mlx.core as mx
            from mlx_audio.tts.utils import load_model

            if self._model is None:
                self._model = load_model(str(self._model_path))
            mx.random.seed(int(key[:8], 16))
            parts = list(self._model.generate(
                text=to_speak, lang_code="Russian",
                ref_audio=str(self._references / f"{voice}.wav"),
                ref_text=prepare((self._references / f"{voice}.txt").read_text()),
                temperature=self._temperature, repetition_penalty=1.05,
                max_tokens=min(1800, max(300, len(to_speak) * 5)),
            ))
            if not parts or any(r.sample_rate != 24000 for r in parts):
                raise SynthesisError("модель вернула неожиданный формат аудио")
            raw = np.concatenate([np.array(r.audio).reshape(-1) for r in parts])
            # Ограничение ловит зацикливание генерации, а не обрезает запись.
            if len(raw) / 24000 > max(12, len(to_speak.split()) * 1.4 + 5):
                raise SynthesisError("аномально длинная запись: требуется повторная проверка")
            # Дубли чуть разной громкости: у живого человека они не одинаковые.
            jitter = ((int(key[8:12], 16) % 15) - 7) / 10 if self.takes > 1 else 0.0
            master = clean_master(
                raw, keep_breath=self._keep_breath, room_tone_db=self._room_tone_db,
                gain_db=jitter, seed=int(key[:8], 16),
            )
            self._cache.mkdir(parents=True, exist_ok=True)
            temporary = cached.with_suffix(".tmp")
            with temporary.open("wb") as handle:
                np.save(handle, master, allow_pickle=False)
            temporary.replace(cached)
            mx.clear_cache()
        audio = at_rate(master, 24000, sample_rate)
        return Speech(to_wav(audio, sample_rate), sample_rate, len(audio) / sample_rate)
