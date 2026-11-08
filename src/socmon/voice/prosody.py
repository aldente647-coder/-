"""Просодические измерения записи — чтобы выбирать дубль не наугад.

Из нескольких дублей одной реплики один звучит живее других: с движением
высоты голоса, в человеческом темпе, без провалов. Слушать сотни дублей
некому, поэтому измеряем то, что измеряется: темп, размах и изменчивость
высоты, длину внутренних пауз, перегруз. Метрики грубые — они не отличат
хорошую интонацию от плохой, но надёжно отсеивают монотонный, скомканный
или сбойный дубль. Окончательное слово остаётся за прослушиванием.

Только numpy: без librosa, без torch. Высота оценивается автокорреляцией
по кадрам — для выбора между дублями этой точности достаточно.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_VOWELS = re.compile(r"[аеёиоуыэюя]", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class Prosody:
    duration_sec: float
    voiced_sec: float
    syllables: int
    pitch_median_hz: float
    pitch_std_semitones: float
    """Изменчивость высоты. Ноль — монотон; живая речь — от полутора
    до пяти-шести полутонов."""

    pitch_range_semitones: float
    longest_pause_sec: float
    """Самая длинная внутренняя тишина. Больше полутора секунд —
    скорее сбой генерации, чем смысловая пауза."""

    peak: float

    @property
    def rate(self) -> float:
        """Слогов в секунду озвученной речи. Спокойная русская речь —
        четыре-шесть."""
        return self.syllables / self.voiced_sec if self.voiced_sec > 0 else 0.0

    @property
    def score(self) -> float:
        """Чем выше, тем живее и надёжнее. Штрафы за то, что выдаёт машину
        или сбой; премия за движение голоса в человеческом диапазоне."""
        value = 0.0
        # Темп считается по всем гласным, включая безударные, поэтому числа
        # выше «школьных» четырёх-шести. Замер на Qwen3-TTS: 6,7–7,3.
        # В допустимом окне чуть спокойнее — чуть лучше: интервьюер не спешит.
        if self.rate < 4.0:
            value -= (4.0 - self.rate) * 2.0
        elif self.rate > 7.4:
            value -= (self.rate - 7.4) * 2.5
        else:
            value += (7.4 - self.rate) * 0.3
        if self.pitch_std_semitones < 1.2:
            value -= (1.2 - self.pitch_std_semitones) * 3.0
        elif self.pitch_std_semitones <= 5.5:
            value += min(self.pitch_std_semitones, 4.0)
        else:
            value -= (self.pitch_std_semitones - 5.5) * 1.5
        if self.longest_pause_sec > 1.5:
            value -= (self.longest_pause_sec - 1.5) * 4.0
        if self.peak > 0.99:
            value -= 5.0
        if self.voiced_sec < 0.25:
            value -= 10.0
        return value


def syllables(text: str) -> int:
    return len(_VOWELS.findall(text))


def measure(samples: Any, rate: int, text: str) -> Prosody:
    """Измерить запись. `samples` — float в [-1, 1], моно."""
    import numpy as np

    audio = np.asarray(samples, dtype=np.float64).reshape(-1)
    if audio.size < rate // 10:
        return Prosody(audio.size / rate, 0.0, syllables(text), 0.0, 0.0, 0.0, 0.0,
                       float(np.max(np.abs(audio))) if audio.size else 0.0)

    frame = int(rate * 0.030)
    hop = int(rate * 0.010)
    count = max(1, (audio.size - frame) // hop + 1)
    frames = np.lib.stride_tricks.as_strided(
        audio, shape=(count, frame), strides=(audio.strides[0] * hop, audio.strides[0]),
    )
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    loud = rms > max(1e-4, float(np.max(rms)) * 10 ** (-32 / 20))

    pitches = _pitch_track(frames, rate, loud)
    voiced = pitches > 0
    hop_sec = hop / rate

    semis = 12 * np.log2(pitches[voiced] / 100.0) if voiced.any() else np.zeros(0)
    return Prosody(
        duration_sec=audio.size / rate,
        voiced_sec=float(loud.sum()) * hop_sec,
        syllables=syllables(text),
        pitch_median_hz=float(np.median(pitches[voiced])) if voiced.any() else 0.0,
        pitch_std_semitones=float(np.std(semis)) if semis.size > 2 else 0.0,
        pitch_range_semitones=float(np.percentile(semis, 95) - np.percentile(semis, 5))
        if semis.size > 2 else 0.0,
        longest_pause_sec=_longest_gap(loud) * hop_sec,
        peak=float(np.max(np.abs(audio))),
    )


def _pitch_track(frames: Any, rate: int, loud: Any) -> Any:
    """Автокорреляционная оценка F0 по кадру, 60–400 Гц. Ноль — глухой кадр."""
    import numpy as np

    low, high = int(rate / 400), int(rate / 60)
    window = np.hanning(frames.shape[1])
    out = np.zeros(frames.shape[0])
    for index in np.flatnonzero(loud):
        segment = (frames[index] - frames[index].mean()) * window
        energy = float(np.dot(segment, segment))
        if energy <= 0:
            continue
        spectrum = np.fft.rfft(segment, n=2 * segment.size)
        corr = np.fft.irfft(spectrum * np.conj(spectrum))[: segment.size] / energy
        span = corr[low:high]
        if span.size == 0:
            continue
        lag = int(np.argmax(span)) + low
        # Достаточно выраженный пик — периодический (звонкий) кадр.
        if corr[lag] > 0.45:
            out[index] = rate / lag
    return out


def _longest_gap(loud: Any) -> int:
    """Самая длинная тишина между озвученными кадрами, в кадрах."""
    import numpy as np

    active = np.flatnonzero(loud)
    if active.size < 2:
        return 0
    return int(np.max(np.diff(active)) - 1)


def best_take(candidates: list[tuple[int, Prosody]]) -> int:
    """Номер лучшего дубля. При равенстве — с меньшим номером: он уже
    мог быть проверен раньше."""
    if not candidates:
        raise ValueError("нет дублей для выбора")
    return min(candidates, key=lambda c: (-c[1].score, c[0]))[0]
