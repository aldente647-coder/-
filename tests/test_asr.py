"""Сегментация речи и мост AudioSocket. Без модели и без Asterisk."""

from __future__ import annotations

import asyncio
import struct
import uuid

import pytest

from socmon.asr.audiosocket import (
    HEADER_BYTES,
    AudioSocketError,
    AudioSocketServer,
    Frame,
    FrameType,
    decode_header,
)
from socmon.asr.contract import (
    FRAME_BYTES,
    SAMPLE_RATE,
    Hypothesis,
    Recognizer,
    SpeechSegmenter,
    VadPolicy,
)
from socmon.asr.gigaam import (
    KNOWN_MODELS,
    OFFLINE_MODEL,
    ONLINE_MODEL,
    GigaamConfig,
    GigaamRecognizer,
    to_wav,
)
from socmon.telephony.provider import RecognizedSpeech

QUIET = b"\x00\x00" * (FRAME_BYTES // 2)


def loud(amplitude: int = 8000) -> bytes:
    """Кадр «речи»: пила заданной амплитуды."""
    samples = [amplitude if i % 2 else -amplitude for i in range(FRAME_BYTES // 2)]
    return struct.pack(f"<{len(samples)}h", *samples)


class FakeRecognizer:
    """Отдаёт заранее заданный текст и запоминает длины сегментов."""

    def __init__(self, text: str = "да") -> None:
        self.text = text
        self.segments: list[int] = []

    def transcribe(self, pcm: bytes) -> Hypothesis:
        self.segments.append(len(pcm))
        return Hypothesis(text=self.text, confidence=0.9)


# ------------------------------------------------------------------ сегментация


def test_silence_alone_produces_no_utterance() -> None:
    segmenter = SpeechSegmenter()
    assert [s for _ in range(100) for s in segmenter.feed(QUIET)] == []


def test_speech_then_pause_closes_an_utterance() -> None:
    segmenter = SpeechSegmenter()
    produced: list[bytes] = []
    for _ in range(50):
        produced.extend(segmenter.feed(loud()))
    assert produced == []  # пока говорит — не отдаём

    for _ in range(40):
        produced.extend(segmenter.feed(QUIET))
    assert len(produced) == 1


def test_short_click_is_not_an_answer() -> None:
    """Щелчок линии или вздох не должны уходить в распознавание."""
    segmenter = SpeechSegmenter(VadPolicy(speech_frames_to_start=1, min_utterance_ms=400))
    produced: list[bytes] = []
    for _ in range(3):
        produced.extend(segmenter.feed(loud()))
    for _ in range(40):
        produced.extend(segmenter.feed(QUIET))
    assert produced == []


def test_two_answers_separated_by_a_pause_are_two_utterances() -> None:
    segmenter = SpeechSegmenter()
    produced: list[bytes] = []
    for _ in range(2):
        for _ in range(30):
            produced.extend(segmenter.feed(loud()))
        for _ in range(40):
            produced.extend(segmenter.feed(QUIET))
    assert len(produced) == 2


def test_endless_speech_is_cut_at_the_ceiling() -> None:
    """Открытый вопрос допускает долгий ответ, но «залипший» шумящий канал
    не должен копиться в памяти бесконечно (ТЗ §7.5)."""
    segmenter = SpeechSegmenter(VadPolicy(max_utterance_ms=1000))
    produced: list[bytes] = []
    for _ in range(200):
        produced.extend(segmenter.feed(loud()))
    assert len(produced) >= 2


def test_pause_length_is_a_tunable_compromise() -> None:
    """Короче — робот перебивает задумавшегося. Длиннее — заметная задержка
    после каждого ответа, и лимит ТЗ §3.3 перестаёт выполняться."""
    patient = SpeechSegmenter(VadPolicy(silence_ms_to_end=2000))
    produced: list[bytes] = []
    for _ in range(30):
        produced.extend(patient.feed(loud()))
    for _ in range(40):  # 800 мс тишины
        produced.extend(patient.feed(QUIET))
    assert produced == []


def test_flush_returns_speech_cut_off_by_a_dropped_line() -> None:
    segmenter = SpeechSegmenter()
    for _ in range(30):
        list(segmenter.feed(loud()))
    assert segmenter.flush() is not None
    assert segmenter.flush() is None


# ------------------------------------------------------------------- кадры


def test_frame_round_trip() -> None:
    frame = Frame(FrameType.AUDIO, b"\x01\x02\x03")
    encoded = frame.encode()
    kind, length = decode_header(encoded[:HEADER_BYTES])
    assert kind is FrameType.AUDIO
    assert encoded[HEADER_BYTES:HEADER_BYTES + length] == b"\x01\x02\x03"


def test_unknown_frame_type_does_not_break_the_call() -> None:
    """Asterisk вправе прислать служебный кадр, о котором мы не знаем.
    Рвать из-за этого живой звонок — плохой обмен."""
    kind, length = decode_header(bytes([0x77, 0x00, 0x04]))
    assert kind is FrameType.ERROR
    assert length == 4


def test_oversized_frame_is_refused() -> None:
    with pytest.raises(AudioSocketError, match="длиннее"):
        Frame(FrameType.AUDIO, b"x" * 70000).encode()


def test_truncated_header_is_refused() -> None:
    with pytest.raises(AudioSocketError, match="заголовок"):
        decode_header(b"\x10\x00")


# ------------------------------------------------------------------- сервер


async def drive_server(
    recognizer: FakeRecognizer, frames: list[Frame], *, register: bool = True
) -> list[RecognizedSpeech]:
    heard: list[RecognizedSpeech] = []

    async def collect(speech: RecognizedSpeech) -> None:
        heard.append(speech)

    server = AudioSocketServer(recognizer, collect, port=0)
    await server.start()
    port = server.port

    stream_id = server.expect("attempt-1") if register else str(uuid.uuid4())
    _, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(Frame(FrameType.IDENTIFIER, uuid.UUID(stream_id).bytes).encode())
    for frame in frames:
        writer.write(frame.encode())
    await writer.drain()
    writer.close()
    await asyncio.sleep(0.15)
    await server.stop()
    return heard


async def test_audio_becomes_recognized_speech() -> None:
    recognizer = FakeRecognizer("да, согласен")
    frames = (
        [Frame(FrameType.AUDIO, loud()) for _ in range(30)]
        + [Frame(FrameType.AUDIO, QUIET) for _ in range(40)]
    )
    heard = await drive_server(recognizer, frames)

    assert len(heard) == 1
    assert heard[0].text == "да, согласен"
    assert heard[0].attempt_id == "attempt-1"
    assert heard[0].silence is False


async def test_unknown_stream_is_dropped_silently() -> None:
    """Подключиться к порту может кто угодно; обрабатывать неизвестный
    звук нельзя."""
    recognizer = FakeRecognizer()
    frames = [Frame(FrameType.AUDIO, loud()) for _ in range(30)]
    assert await drive_server(recognizer, frames, register=False) == []
    assert recognizer.segments == []


async def test_empty_recognition_is_reported_as_silence() -> None:
    """«Ничего не разобрал» — это молчание, а не ответ: движок переспросит
    один раз и поставит код нераспознанной речи (ТЗ §8.1, §5.3)."""
    frames = (
        [Frame(FrameType.AUDIO, loud()) for _ in range(30)]
        + [Frame(FrameType.AUDIO, QUIET) for _ in range(40)]
    )
    heard = await drive_server(FakeRecognizer(""), frames)
    assert heard[0].silence is True
    assert heard[0].text is None


async def test_terminate_frame_closes_the_stream() -> None:
    recognizer = FakeRecognizer()
    frames = [
        *[Frame(FrameType.AUDIO, loud()) for _ in range(30)],
        Frame(FrameType.TERMINATE, b""),
        *[Frame(FrameType.AUDIO, loud()) for _ in range(30)],
    ]
    heard = await drive_server(recognizer, frames)
    # Речь, оборванная закрытием, дораспознаётся один раз.
    assert len(heard) == 1


# ------------------------------------------------------------------ модель


def test_chosen_models_are_the_third_generation() -> None:
    """RESEARCH.md §5: v3 обучена на 700 тыс. часов и даёт 9,5% WER
    на домене колл-центра."""
    assert ONLINE_MODEL == "v3_rnnt"
    assert OFFLINE_MODEL == "v3_e2e_rnnt"
    assert {ONLINE_MODEL, OFFLINE_MODEL} <= KNOWN_MODELS


def test_emotion_model_is_available_for_the_night_pass() -> None:
    """ТЗ §11.1 требует признаков раздражения и тревоги. В v3 распознавания
    эмоций нет, но в линейке есть отдельная модель."""
    assert "emo" in KNOWN_MODELS


def test_unknown_model_is_refused_at_configuration_time() -> None:
    from socmon.asr.contract import RecognitionError

    with pytest.raises(RecognitionError, match="неизвестная модель"):
        GigaamConfig(model_name="whisper-large")


def test_wav_header_matches_the_telephone_channel() -> None:
    import io
    import wave

    with wave.open(io.BytesIO(to_wav(b"\x00\x00" * 800))) as handle:
        assert handle.getframerate() == SAMPLE_RATE
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2


def test_empty_audio_never_reaches_the_model() -> None:
    assert GigaamRecognizer().transcribe(b"").text == ""


def test_fake_recognizer_satisfies_the_contract() -> None:
    assert isinstance(FakeRecognizer(), Recognizer)


def test_long_answers_do_not_hit_the_model_length_limit() -> None:
    """У GigaAM `transcribe` ограничен 25 секундами, а открытый вопрос
    (ТЗ §7.5) может звучать дольше — сегментатор допускает 45. Обрезать
    ответ было бы потерей данных, поэтому длинные куски идут другим
    методом. Поймано на живом сервере: «Too long wav file»."""
    from socmon.asr.contract import DEFAULT_VAD
    from socmon.asr.gigaam import LONGFORM_THRESHOLD_SEC

    assert DEFAULT_VAD.max_utterance_ms / 1000 > LONGFORM_THRESHOLD_SEC


def test_long_audio_is_split_by_ourselves_not_by_the_package() -> None:
    """`transcribe_longform` тянет pyannote и качает его модель
    с HuggingFace. Это зарубежный сервис, а §14.1 запрещает обращаться
    к таким из нашего контура. Режем сами."""
    from socmon.asr.gigaam import CHUNK_SEC, split_audio

    minute = loud() * (60 * 1000 // 20)
    pieces = split_audio(minute)
    assert len(pieces) >= 3
    for offset, chunk in pieces:
        seconds = len(chunk) / (SAMPLE_RATE * 2)
        assert seconds <= CHUNK_SEC + 2.5   # окно поиска паузы
        assert offset >= 0


def test_split_covers_the_recording_without_gaps() -> None:
    """Потерять кусок открытого ответа нельзя: это данные."""
    from socmon.asr.gigaam import split_audio

    audio = loud() * 2000
    pieces = split_audio(audio)
    assert b"".join(chunk for _, chunk in pieces) == audio


def test_short_audio_is_not_split() -> None:
    from socmon.asr.gigaam import split_audio

    short = loud() * 50   # одна секунда
    assert len(split_audio(short)) == 1


def test_split_prefers_a_quiet_boundary() -> None:
    """Разрез посреди слова стоит неверных слов на каждом стыке."""
    from socmon.asr.contract import FRAME_BYTES, frame_energy
    from socmon.asr.gigaam import split_audio

    speech = loud() * (19 * 50)      # 19 секунд речи
    pause = QUIET * 50               # секунда тишины
    audio = speech + pause + loud() * (19 * 50)

    pieces = split_audio(audio)
    assert len(pieces) >= 2
    boundary = len(pieces[0][1])
    assert frame_energy(audio[boundary:boundary + FRAME_BYTES]) < 100
