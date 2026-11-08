"""Аудиопроверки запускаются в окружении синтеза с numpy/scipy."""
from __future__ import annotations

import io
import wave
from pathlib import Path

import pytest

from socmon.voice.clean import at_rate, clean_master
from socmon.voice.synth import SynthesisError, to_wav

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")


def test_clean_edges_level_and_phone_copy() -> None:
    rate = 24000
    tone = 0.5 * np.sin(2 * np.pi * 440 * np.arange(rate) / rate)
    raw = np.concatenate([np.zeros(rate), tone, np.zeros(rate)])
    master = clean_master(raw)
    # Убрана внешняя тишина, но осталась вся секунда сигнала.
    assert 1.2 < len(master) / rate < 1.6
    assert np.max(np.abs(master)) <= 0.851
    assert np.count_nonzero(master[:1920]) == 0
    assert np.count_nonzero(master[-3360:]) == 0
    phone = at_rate(master, 24000, 8000)
    assert abs(len(phone) / 8000 - len(master) / rate) < 0.001
    with wave.open(io.BytesIO(to_wav(phone, 8000))) as handle:
        assert (handle.getframerate(), handle.getnchannels(), handle.getsampwidth()) == (8000, 1, 2)


def test_resampling_removes_frequencies_above_phone_band() -> None:
    t = np.arange(24000) / 24000
    low = at_rate(np.sin(2 * np.pi * 1000 * t), 24000, 8000)
    high = at_rate(np.sin(2 * np.pi * 7000 * t), 24000, 8000)
    assert np.sqrt(np.mean(high[100:-100] ** 2)) < np.sqrt(np.mean(low ** 2)) * 0.01


@pytest.mark.parametrize("bad", [np.zeros(24000), np.full(24000, np.nan), np.zeros(20)])
def test_invalid_audio_is_rejected(bad: object) -> None:
    with pytest.raises(SynthesisError):
        clean_master(bad)


def test_internal_pause_is_retained() -> None:
    tone = 0.2 * np.sin(2 * np.pi * 440 * np.arange(12000) / 24000)
    separated = clean_master(np.concatenate([tone, np.zeros(24000), tone]))
    joined = clean_master(np.concatenate([tone, tone]))
    assert len(separated) - len(joined) == 24000


def test_retakes_preserve_other_cached_recordings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hashlib

    from socmon.voice.qwen import QwenMlxSynthesizer

    model, refs, cache = (tmp_path / name for name in ('model', 'refs', 'cache'))
    for directory in (model, refs, cache):
        directory.mkdir()
    (model / 'config.json').write_text('{}')
    (model / 'model.safetensors').write_bytes(b'test weights, never loaded')
    for voice in ('anna', 'alex'):
        (refs / f'{voice}.wav').write_bytes(b'test reference, never loaded')
        (refs / f'{voice}.txt').write_text('Эталон.')
    monkeypatch.setattr('socmon.voice.qwen.importlib.metadata.version', lambda _: 'test')
    original = QwenMlxSynthesizer(model, refs, cache)
    text = 'Вы согласны принять участие?'
    key = hashlib.sha256((original.fingerprint + 'anna' + text).encode()).hexdigest()
    np.save(cache / f'{key}.npy', np.ones(24000, dtype=np.float32) * 0.1)
    first = original.synthesize(text, voice='anna', sample_rate=8000)
    retakes = tmp_path / 'retakes.json'
    retakes.write_text('{"alex:unrelated": 1}')
    changed = QwenMlxSynthesizer(model, refs, cache, retakes)
    assert changed.fingerprint != original.fingerprint
    assert changed.synthesize(text, voice='anna', sample_rate=8000).wav == first.wav


# ------------------------------------------------ вдох, комнатный тон, громкость


def _speech_like(rate: int = 24000) -> np.ndarray:
    t = np.arange(rate) / rate
    return 0.5 * np.sin(2 * np.pi * 220 * t)


def test_breath_before_speech_survives_when_asked() -> None:
    """Вдох тише речи на порядок; жадный порог принимает его за тишину."""
    rate = 24000
    # Вдох на 50 дБ тише речи: ниже жадного порога, выше чуткого.
    breath = 0.0011 * np.random.default_rng(1).uniform(-1, 1, int(rate * 0.3))
    raw = np.concatenate([np.zeros(rate), breath, _speech_like(rate), np.zeros(rate)])
    strict = clean_master(raw)
    gentle = clean_master(raw, keep_breath=True)
    # Обычная чистка вдох срезает, чуткая — оставляет и даёт запас спереди.
    assert len(gentle) - len(strict) > int(rate * 0.25)


def test_room_tone_replaces_digital_silence() -> None:
    rate = 24000
    raw = np.concatenate([np.zeros(rate), _speech_like(rate), np.zeros(rate)])
    master = clean_master(raw, room_tone_db=-60.0)
    edge = master[: int(rate * 0.06)]
    assert np.count_nonzero(edge) == edge.size
    level = 20 * np.log10(np.sqrt(np.mean(edge ** 2)))
    assert -64 < level < -56
    # Тон воспроизводим: одно зерно — один и тот же шум.
    again = clean_master(raw, room_tone_db=-60.0)
    assert np.array_equal(master, again)
    assert not np.array_equal(master, clean_master(raw, room_tone_db=-60.0, seed=7))


def test_gain_shifts_level_but_never_clips() -> None:
    rate = 24000
    raw = np.concatenate([np.zeros(rate), _speech_like(rate), np.zeros(rate)])
    quiet = clean_master(raw, gain_db=-1.0)
    loud = clean_master(raw, gain_db=1.0)
    ratio = np.sqrt(np.mean(loud ** 2)) / np.sqrt(np.mean(quiet ** 2))
    assert 1.2 < ratio < 1.3  # ровно два децибела
    assert np.max(np.abs(clean_master(raw, gain_db=5.0))) <= 0.9
