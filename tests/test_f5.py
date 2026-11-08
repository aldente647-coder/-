"""F5-синтезатор: словарь ударений перед синтезом, кэш мастеров, дубли."""

from __future__ import annotations

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from socmon.voice.accent import Accentuator  # noqa: E402
from socmon.voice.f5 import F5Synthesizer  # noqa: E402
from socmon.voice.synth import SynthesisError  # noqa: E402


class FakeTts:
    """Подставная модель: пишет, что её попросили произнести, и отдаёт тон."""

    def __init__(self) -> None:
        self.asked: list[str] = []

    def infer(
        self, *, ref_file: str, ref_text: str, gen_text: str, seed: int, **_: object
    ) -> tuple:
        self.asked.append(gen_text)
        t = np.arange(int(24000 * 1.2)) / 24000
        audio = 0.4 * np.sin(2 * np.pi * (150 + seed % 50) * t)
        return audio.astype(np.float32), 24000, None


class FakeAccent(Accentuator):
    def __init__(self) -> None:
        super().__init__(exceptions={})

    def mark(self, text: str) -> str:  # словарь не грузим: ставим «+» перед первой гласной
        out = []
        for word in text.split(" "):
            for i, ch in enumerate(word):
                if ch.lower() in "аеёиоуыэюя":
                    word = word[:i] + "+" + word[i:]
                    break
            out.append(word)
        return " ".join(out)


@pytest.fixture
def synth(tmp_path: Path) -> F5Synthesizer:
    model = tmp_path / "model"
    model.mkdir()
    (model / "model_v1.safetensors").write_bytes(b"\0" * 64)
    (model / "vocab.txt").write_text("a\nб\n+\n", encoding="utf-8")
    refs = tmp_path / "refs"
    refs.mkdir()
    (refs / "denis.wav").write_bytes(b"RIFF")
    (refs / "denis.txt").write_text("Здравствуйте, меня зовут Денис.", encoding="utf-8")
    s = F5Synthesizer(model, refs, tmp_path / "cache", takes=2, accentuator=FakeAccent())
    s._tts = FakeTts()
    return s


def test_text_reaches_the_model_with_stress_marks(synth: F5Synthesizer) -> None:
    speech = synth.synthesize("Вы согласны принять участие?", voice="denis", sample_rate=8000)
    assert speech.sample_rate == 8000 and speech.duration_sec > 1.0
    asked = synth._tts.asked[-1]  # type: ignore[union-attr]
    assert "+" in asked and asked.replace("+", "") == "Вы согласны принять участие?"
    # Референс тоже размечен.
    assert "+" in synth._reference_text("denis")


def test_masters_are_cached_per_take(synth: F5Synthesizer) -> None:
    synth.synthesize("Вы согласны?", voice="denis", sample_rate=8000)
    synth.synthesize("Вы согласны?", voice="denis", sample_rate=8000)
    synth.synthesize("Вы согласны?", voice="denis", sample_rate=8000, take=1)
    assert len(synth._tts.asked) == 2  # type: ignore[union-attr]
    with pytest.raises(SynthesisError):
        synth.synthesize("Вы согласны?", voice="denis", sample_rate=8000, take=2)


def test_unknown_voice_is_refused(synth: F5Synthesizer) -> None:
    with pytest.raises(SynthesisError):
        synth.synthesize("Вы согласны?", voice="anna", sample_rate=8000)
