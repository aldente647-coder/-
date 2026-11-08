"""Подготовка чистого мастера и телефонной копии без изменения темпа речи.

Режимы `keep_breath` и `room_tone_db` оставлены как опции, но по умолчанию
выключены. Прослушивание 09.09.2026 показало: комнатный тон −60 дБ слышен
как помеха, а чуткий порог обрезки сохраняет не вдох, а шум модели перед
речью на −23 дБ. Чистая тишина на краях звучит лучше любого «оживления».
"""
from __future__ import annotations

from typing import Any

from socmon.voice.synth import SynthesisError


def clean_master(
    samples: Any,
    rate: int = 24000,
    *,
    keep_breath: bool = False,
    room_tone_db: float | None = None,
    gain_db: float = 0.0,
    seed: int = 0,
) -> Any:
    """Убрать DC, лишние края и скачки на границе; оставить внутренние паузы.

    `keep_breath` — более чуткий порог и запас перед речью, чтобы уцелел вдох.
    `room_tone_db` — уровень комнатного тона в дБ относительно полной шкалы;
    None — края нулевые, как раньше. `gain_db` — небольшой сдвиг громкости:
    у живого человека дубли не одинаковой громкости.
    """
    import numpy as np
    from scipy.signal import butter, sosfilt

    audio = np.asarray(samples, dtype=np.float64).reshape(-1)
    if rate < 8000 or audio.size < rate // 10 or not np.isfinite(audio).all():
        raise SynthesisError("повреждённая или слишком короткая запись")
    audio = sosfilt(butter(2, 45, btype="highpass", fs=rate, output="sos"), audio)
    peak = float(np.max(np.abs(audio)))
    if peak < 0.001:
        raise SynthesisError("модель вернула тишину")

    # Только внешняя тишина; запас защищает тихие согласные и окончания.
    # Со вдохом порог ниже и запас спереди больше: вдох на порядок тише речи.
    floor = max(0.0003, peak * (0.0012 if keep_breath else 0.003))
    active = np.flatnonzero(np.abs(audio) > floor)
    before = int(rate * (0.22 if keep_breath else 0.08))
    after = int(rate * 0.08)
    audio = audio[max(0, int(active[0]) - before):int(active[-1]) + after + 1]

    speech = audio[np.abs(audio) > peak * 0.02]
    rms = float(np.sqrt(np.mean(speech ** 2)))
    gain = min(10 ** (-21 / 20) / rms, 0.85 / peak, 3.0)
    gain *= 10 ** (max(-2.0, min(2.0, gain_db)) / 20)
    audio *= gain
    if float(np.max(np.abs(audio))) > 0.9:
        audio *= 0.9 / float(np.max(np.abs(audio)))

    fade = min(int(rate * 0.006), len(audio) // 2)
    audio[:fade] *= np.linspace(0, 1, fade)
    audio[-fade:] *= np.linspace(1, 0, fade)
    padded = np.pad(audio, (int(rate * 0.08), int(rate * 0.14)))

    if room_tone_db is not None:
        padded = padded + room_tone(padded.size, rate, room_tone_db, seed=seed)
    return padded.astype(np.float32)


def room_tone(length: int, rate: int, level_db: float, *, seed: int = 0) -> Any:
    """Тихий шум помещения: белый шум, приглушённый сверху, — ближе к тому,
    что слышно в трубке между словами, чем чистая тишина.

    Воспроизводим по зерну: два рендера одного дубля обязаны совпасть."""
    import numpy as np
    from scipy.signal import butter, sosfilt

    generator = np.random.default_rng(seed)
    noise = generator.standard_normal(length)
    noise = sosfilt(butter(2, 3400, btype="lowpass", fs=rate, output="sos"), noise)
    target = 10 ** (level_db / 20)
    current = float(np.sqrt(np.mean(noise ** 2))) or 1.0
    return noise * (target / current)


def at_rate(master: Any, source_rate: int, target_rate: int) -> Any:
    """Антиалиасинг при сужении полосы; мастер и телефон — одна запись."""
    from math import gcd

    from scipy.signal import resample_poly

    if source_rate == target_rate:
        return master
    factor = gcd(source_rate, target_rate)
    return resample_poly(master, target_rate // factor, source_rate // factor)
