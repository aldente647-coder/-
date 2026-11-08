"""F5-TTS с русской дообучкой: клон голоса, который слушается ударений.

Второй движок клонирования рядом с Qwen. Разница принципиальная: русская
дообучка F5-TTS обучена на тексте с отметками ударения («+» перед ударной
гласной) и читает их. Qwen отметок не понимает вовсе — проверено на
минимальных парах. Поэтому здесь каждая формулировка проходит через словарь
ударений (`socmon.voice.accent`) до синтеза; слова при этом те же самые.

Работает на PyTorch: на Маке через Metal (около восьми секунд синтеза
на секунду речи), на сервере — на процессоре, в шесть раз медленнее.
Мастера кэшируются по рецепту, как у Qwen: повторная сборка не синтезирует
то, что уже есть.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
from pathlib import Path
from typing import Any

from socmon.voice.accent import Accentuator, strip_marks
from socmon.voice.clean import at_rate, clean_master
from socmon.voice.spoken import spoken
from socmon.voice.synth import SUPPORTED_RATES, Speech, SynthesisError, prepare, to_wav

MODEL_RATE = 24000


class F5Synthesizer:
    """Клон по референсу; ударения из словаря; дубли — разными зёрнами."""

    def __init__(
        self,
        model: Path,
        references: Path,
        cache: Path,
        *,
        takes: int = 1,
        steps: int = 32,
        device: str = "auto",
        shaped: bool = True,
        accentuator: Accentuator | None = None,
    ) -> None:
        for name in ("model_v1.safetensors", "vocab.txt"):
            if not (model / name).is_file():
                raise SynthesisError(f"в каталоге модели нет {name}")
        if not 1 <= takes <= 8:
            raise SynthesisError("дублей на реплику — от одного до восьми")
        self.takes = takes
        self._model_dir = model
        self._references = references
        self._cache = cache
        self._steps = steps
        self._device = device
        self._shaped = shaped
        self._accent = accentuator or Accentuator()
        self._tts: Any = None
        self._ref_text: dict[str, str] = {}
        self.available = {
            p.stem for p in references.glob("*.wav")
            if (references / f"{p.stem}.txt").is_file()
        }
        if not self.available:
            raise SynthesisError(
                "в каталоге референсов нет ни одной пары <голос>.wav + <голос>.txt"
            )
        digest = hashlib.sha256()
        for path in sorted([model / "model_v1.safetensors", model / "vocab.txt"]):
            digest.update(path.read_bytes()[:1 << 20])
            digest.update(str(path.stat().st_size).encode())
        for voice in sorted(self.available):
            digest.update((references / f"{voice}.wav").read_bytes())
            digest.update((references / f"{voice}.txt").read_bytes())
        try:
            version = importlib.metadata.version("f5-tts")
        except importlib.metadata.PackageNotFoundError:
            version = "?"
        self.fingerprint = (
            f"f5-tts-ru:{version}:{digest.hexdigest()[:16]}:accent-v1:steps-{steps}"
            + (":shaped-v1" if shaped else "") + (f":takes-{takes}" if takes > 1 else "")
        )

    def _load(self) -> Any:
        if self._tts is None:
            import torch
            from f5_tts.api import F5TTS

            device = self._device
            if device == "auto":
                device = "mps" if torch.backends.mps.is_available() else "cpu"
            if device == "cpu":
                torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))
            self._tts = F5TTS(
                model="F5TTS_v1_Base",
                ckpt_file=str(self._model_dir / "model_v1.safetensors"),
                vocab_file=str(self._model_dir / "vocab.txt"),
                device=device,
            )
        return self._tts

    def _reference_text(self, voice: str) -> str:
        if voice not in self._ref_text:
            plain = prepare((self._references / f"{voice}.txt").read_text(encoding="utf-8"))
            self._ref_text[voice] = self._accent.mark(plain)
        return self._ref_text[voice]

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

        to_speak = self._accent.mark(spoken(prepared) if self._shaped else prepared)
        key = hashlib.sha256(
            f"{self.fingerprint}:{voice}:{to_speak}:{take}".encode()
        ).hexdigest()
        cached = self._cache / f"{key}.npy"
        if cached.is_file():
            master = np.load(cached, allow_pickle=False)
        else:
            tts = self._load()
            wav, rate, _ = tts.infer(
                ref_file=str(self._references / f"{voice}.wav"),
                ref_text=self._reference_text(voice),
                gen_text=to_speak,
                nfe_step=self._steps, cfg_strength=2.0, speed=1.0,
                seed=int(key[:8], 16) % (2**31), remove_silence=False,
            )
            if rate != MODEL_RATE:
                raise SynthesisError(f"модель вернула {rate} Гц вместо {MODEL_RATE}")
            raw = np.asarray(wav, dtype=np.float64).reshape(-1)
            # Защита от зацикливания: речь не может быть длиннее разумного
            # для этого числа слов.
            if len(raw) / MODEL_RATE > max(12, len(strip_marks(to_speak).split()) * 1.4 + 5):
                raise SynthesisError("аномально длинная запись: требуется повторная проверка")
            jitter = ((int(key[8:12], 16) % 15) - 7) / 10 if self.takes > 1 else 0.0
            master = clean_master(raw, MODEL_RATE, gain_db=jitter)
            self._cache.mkdir(parents=True, exist_ok=True)
            temporary = cached.with_suffix(".tmp")
            with temporary.open("wb") as handle:
                np.save(handle, master, allow_pickle=False)
            temporary.replace(cached)
        audio = at_rate(master, MODEL_RATE, sample_rate)
        return Speech(to_wav(audio, sample_rate), sample_rate, len(audio) / sample_rate)
