"""Просодические измерения на синтетических сигналах."""

from __future__ import annotations

import pytest

from socmon.voice.prosody import Prosody, best_take, measure, syllables

np = pytest.importorskip("numpy")
RATE = 24000


def tone(freq_hz: float, seconds: float, amp: float = 0.4) -> np.ndarray:
    t = np.arange(int(RATE * seconds)) / RATE
    # Несколько гармоник — ближе к голосу, чем чистый синус.
    return amp * (np.sin(2 * np.pi * freq_hz * t) + 0.4 * np.sin(4 * np.pi * freq_hz * t))


def test_syllables_count_vowels() -> None:
    assert syllables("Вы согласны принять участие?") == 10


def test_monotone_signal_has_no_pitch_movement() -> None:
    result = measure(tone(180, 1.5), RATE, "раз два три четыре пять шесть")
    assert 170 < result.pitch_median_hz < 190
    assert result.pitch_std_semitones < 0.5
    assert result.longest_pause_sec < 0.05


def test_gliding_pitch_is_seen_as_movement() -> None:
    t = np.arange(int(RATE * 1.5)) / RATE
    glide = 0.4 * np.sin(2 * np.pi * (140 + 60 * t / 1.5) * t)
    result = measure(glide, RATE, "раз два три четыре пять шесть")
    assert result.pitch_std_semitones > 1.5
    assert result.pitch_range_semitones > 4


def test_internal_gap_is_measured() -> None:
    audio = np.concatenate([tone(160, 0.6), np.zeros(int(RATE * 0.8)), tone(160, 0.6)])
    result = measure(audio, RATE, "слово слово")
    assert 0.7 < result.longest_pause_sec < 0.95
    assert 1.1 < result.voiced_sec < 1.3


def test_score_prefers_lively_speech_and_punishes_faults() -> None:
    lively = Prosody(3.0, 2.5, 12, 200, 3.0, 8, 0.4, 0.8)
    monotone = Prosody(3.0, 2.5, 12, 200, 0.3, 1, 0.4, 0.8)
    stalled = Prosody(6.0, 2.5, 12, 200, 3.0, 8, 2.5, 0.8)
    clipped = Prosody(3.0, 2.5, 12, 200, 3.0, 8, 0.4, 1.0)
    rushed = Prosody(1.5, 1.4, 12, 200, 3.0, 8, 0.2, 0.8)  # 8,6 слога в секунду
    calm = Prosody(3.2, 2.6, 12, 200, 3.0, 8, 0.4, 0.8)    # 4,6 слога в секунду
    assert lively.score > monotone.score
    assert lively.score > stalled.score
    assert lively.score > clipped.score
    assert lively.score > rushed.score
    assert calm.score > lively.score
    assert best_take([(0, monotone), (1, lively), (2, stalled)]) == 1


def test_best_take_breaks_ties_towards_the_earlier_take() -> None:
    same = Prosody(3.0, 2.5, 12, 200, 3.0, 8, 0.4, 0.8)
    assert best_take([(2, same), (0, same), (1, same)]) == 0
