"""Распознавание речи моделью GigaAM (SberDevices, лицензия MIT).

Почему именно она — RESEARCH.md §5. Коротко: на домене колл-центра WER
около 9,5% против 20,8% у Whisper в среднем по бенчмарку; обучена на
700 тыс. часов русской речи; энкодер 220M параметров, то есть влезает
в одну видеокарту вместе с офлайн-пайплайном. И главное — работает
в своём контуре: ТЗ §14.1 запрещает зарубежные API, и облачные
распознавалки нам недоступны по условию, а не по вкусу.

Две модели под две задачи:

* **онлайн** — быстрый вариант для хода диалога. Точность здесь важна
  меньше скорости: ответ всё равно проверяется грамматикой узла, и цена
  ошибки — переспрос, а не испорченные данные;
* **офлайн** — точный вариант с готовой пунктуацией и числами. Работает
  ночью по записям, его результат идёт в транскрипты (ТЗ §10.2)
  и в разбор открытых ответов (§11.2).

Расхождение двух проходов даром даёт метрику качества распознавания,
которую требует §11.3.
"""

from __future__ import annotations

import os
import tempfile
import wave
from contextlib import suppress
from dataclasses import dataclass
from io import BytesIO
from typing import Any

from socmon.asr.contract import SAMPLE_RATE, SAMPLE_WIDTH, Hypothesis, RecognitionError


@dataclass(frozen=True, slots=True)
class WordTiming:
    """Слово с привязкой ко времени записи (ТЗ §10.2, §11.1)."""

    text: str
    start: float
    end: float

#: Модель для хода диалога. RNNT предназначена для онлайн-декодирования.
ONLINE_MODEL = "v3_rnnt"

#: Модель для ночного прохода: end-to-end, отдаёт готовый текст
#: с пунктуацией, заглавными и числами.
OFFLINE_MODEL = "v3_e2e_rnnt"

#: Предел длины для распознавания одним куском, секунд.
#:
#: В пакете это `LONGFORM_THRESHOLD = 25 * SAMPLE_RATE`; выше — исключение
#: «Too long wav file». Наш сегментатор допускает реплики до 45 секунд:
#: открытый вопрос (ТЗ §7.5) может звучать долго, и обрезать его было бы
#: потерей данных.
LONGFORM_THRESHOLD_SEC = 25.0

#: Длина куска, на которые режется длинная запись.
#:
#: Режем сами, а не зовём `transcribe_longform` из пакета: тот тянет
#: pyannote и скачивает его модель с HuggingFace. Это зарубежный сервис,
#: и §14.1 запрещает обращаться к таким из нашего контура — вне
#: зависимости от того, удобно это или нет.
CHUNK_SEC = 20.0

#: В каком окне вокруг границы искать паузу для разреза.
SPLIT_SEARCH_SEC = 2.0

#: Что пакет вообще умеет загрузить. Список проверен на установленном
#: пакете, а не выписан из статьи.
KNOWN_MODELS = frozenset({
    "v1_ctc", "v1_rnnt", "v1_ssl",
    "v2_ctc", "v2_rnnt", "v2_ssl",
    "v3_ctc", "v3_rnnt", "v3_ssl", "v3_e2e_ctc", "v3_e2e_rnnt",
    "emo",
})


@dataclass(frozen=True, slots=True)
class GigaamConfig:
    model_name: str = ONLINE_MODEL
    device: str = "cpu"
    """«cuda» на боевом контуре. На CPU модель работает, но медленно:
    для стенда годится, для сотни линий — нет."""

    def __post_init__(self) -> None:
        if self.model_name not in KNOWN_MODELS:
            raise RecognitionError(
                f"неизвестная модель «{self.model_name}»; "
                f"доступны: {', '.join(sorted(KNOWN_MODELS))}"
            )

    @classmethod
    def from_env(cls, *, offline: bool = False) -> GigaamConfig:
        default = OFFLINE_MODEL if offline else ONLINE_MODEL
        return cls(
            model_name=os.environ.get("SOCMON_ASR_MODEL", default),
            device=os.environ.get("SOCMON_ASR_DEVICE", "cpu"),
        )


def to_wav(pcm: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Обернуть сырые отсчёты в WAV.

    Модель принимает файл или путь, а из телефонного канала приходит голый
    поток. Заголовок дешевле, чем тащить ради него звуковую библиотеку.
    """
    buffer = BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(SAMPLE_WIDTH)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    return buffer.getvalue()


def split_audio(pcm: bytes) -> list[tuple[float, bytes]]:
    """Нарезать запись на куски по границам тишины.

    Граница ищется не жёстко по времени, а в окне вокруг него — в самом
    тихом месте. Разрез посреди слова стоит одного-двух неверных слов
    на каждом стыке, и на длинном открытом ответе это заметно.

    Возвращает пары «смещение в секундах, звук».
    """
    from socmon.asr.contract import FRAME_BYTES, frame_energy

    bytes_per_sec = SAMPLE_RATE * SAMPLE_WIDTH
    chunk_bytes = int(CHUNK_SEC * bytes_per_sec)
    search = int(SPLIT_SEARCH_SEC * bytes_per_sec)

    pieces: list[tuple[float, bytes]] = []
    start = 0
    while start < len(pcm):
        if start + chunk_bytes >= len(pcm):
            pieces.append((start / bytes_per_sec, pcm[start:]))
            break

        target = start + chunk_bytes
        window = range(
            max(start + FRAME_BYTES, target - search),
            min(len(pcm) - FRAME_BYTES, target + search),
            FRAME_BYTES,
        )
        cut = min(window, key=lambda i: frame_energy(pcm[i:i + FRAME_BYTES]), default=target)
        pieces.append((start / bytes_per_sec, pcm[start:cut]))
        start = cut
    return pieces


class GigaamRecognizer:
    """Адаптер к модели.

    Веса грузятся при первом обращении и держатся в памяти: перезагружать
    их на каждую реплику — гарантированный срыв темпа разговора.
    """

    def __init__(self, config: GigaamConfig | None = None) -> None:
        self._config = config or GigaamConfig()
        self._model: Any | None = None

    @property
    def model_name(self) -> str:
        return self._config.model_name

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            import gigaam
        except ImportError as exc:  # pragma: no cover - зависит от окружения
            raise RecognitionError(
                "не установлен пакет gigaam: распознавание недоступно. "
                "Поставьте зависимости группы asr"
            ) from exc
        try:
            self._model = gigaam.load_model(
                self._config.model_name, device=self._config.device
            )
        except Exception as exc:  # pragma: no cover - зависит от весов
            raise RecognitionError(
                f"не удалось загрузить модель {self._config.model_name}: {exc}"
            ) from exc
        return self._model

    def transcribe(self, pcm: bytes) -> Hypothesis:
        """Распознать сегмент речи.

        Пустой результат — это не ошибка, а «ничего не разобрал». Движок
        анкеты трактует его как молчание и переспросит один раз (ТЗ §8.1);
        записывать догадку вместо ответа нельзя.
        """
        if not pcm:
            return Hypothesis(text="", confidence=0.0)
        return Hypothesis(text=self._run(pcm, word_timestamps=False)[0])

    def transcribe_with_timestamps(self, pcm: bytes) -> tuple[str, list[WordTiming]]:
        """Распознать с временными метками слов.

        ТЗ §10.2 требует транскрипт с временными метками, §11.1 — дословных
        цитат с привязкой ко времени. Модель отдаёт это сама, поэтому
        выравнивать текст с записью отдельным проходом не нужно.
        """
        return self._run(pcm, word_timestamps=True)

    def _run(self, pcm: bytes, *, word_timestamps: bool) -> tuple[str, list[WordTiming]]:
        """Прогнать модель.

        Пакет принимает путь к файлу, а не отсчёты, поэтому сегмент кладётся
        во временный файл. Он удаляется сразу: телефонный разговор в /tmp —
        это персональные данные вне защищённого контура (ТЗ §14.2).
        """
        if len(pcm) / (SAMPLE_RATE * SAMPLE_WIDTH) > LONGFORM_THRESHOLD_SEC:
            return self._run_chunked(pcm, word_timestamps=word_timestamps)
        return self._run_once(pcm, word_timestamps=word_timestamps, offset=0.0)

    def _run_once(
        self, pcm: bytes, *, word_timestamps: bool, offset: float
    ) -> tuple[str, list[WordTiming]]:
        model = self._load()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            handle.write(to_wav(pcm))
            path = handle.name
        try:
            result = model.transcribe(path, word_timestamps=word_timestamps)
        except Exception as exc:  # pragma: no cover - зависит от весов
            raise RecognitionError(f"сбой распознавания: {exc}") from exc
        finally:
            with suppress(OSError):
                os.unlink(path)

        words = [
            WordTiming(
                text=str(w.word),
                start=float(w.start) + offset,
                end=float(w.end) + offset,
            )
            for w in (getattr(result, "words", None) or [])
        ]
        return str(getattr(result, "text", result) or "").strip(), words

    def _run_chunked(
        self, pcm: bytes, *, word_timestamps: bool
    ) -> tuple[str, list[WordTiming]]:
        """Распознать длинную запись по кускам.

        Метки времени сдвигаются на смещение куска, поэтому снаружи
        результат неотличим от распознавания целиком — а этого и требует
        ТЗ §10.2 и §11.1.
        """
        texts: list[str] = []
        words: list[WordTiming] = []
        for offset, chunk in split_audio(pcm):
            text, chunk_words = self._run_once(
                chunk, word_timestamps=word_timestamps, offset=offset
            )
            if text:
                texts.append(text)
            words.extend(chunk_words)
        return " ".join(texts).strip(), words
