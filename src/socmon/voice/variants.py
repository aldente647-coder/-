"""Воспроизводимая сборка полных голосовых библиотек и примеров."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from socmon.core.questionnaire import Questionnaire
from socmon.voice.catalog import SAMPLES, SILERO_OPTIONS, VoiceOption
from socmon.voice.render import RenderError, render, speakable, text_hash, verify
from socmon.voice.synth import SpeechSynthesizer


def build_variants(
    directory: Path, synthesizer: SpeechSynthesizer, *,
    questionnaires: Path | None = None,
    voices: Sequence[VoiceOption] = SILERO_OPTIONS,
    progress: Callable[[str], None] = print,
) -> None:
    """directory — отдельный каталог variants; действующий голос не заменяется."""
    source = questionnaires or Path(__file__).resolve().parents[1] / "questionnaires"
    reference = Questionnaire.from_yaml(source / "wave-base.yaml")
    waves = [Questionnaire.from_yaml(p) for p in sorted(source.glob("wave-*.yaml"))]
    # Историческая core-v1 имеет другие тексты при тех же audio_key:
    # в библиотеку актуальных волн её нельзя смешивать.
    hashes: dict[str, str] = {}
    for questionnaire in waves:
        for node in speakable(questionnaire):
            key, digest = node.audio_key or node.id, text_hash(node.text)
            if key in hashes and hashes[key] != digest:
                raise RenderError(f"разные формулировки используют audio_key {key}")
            hashes[key] = digest
    for voice in voices:
        target = directory / voice.id
        for questionnaire in waves:
            report = render(questionnaire, synthesizer, target, voice=voice.id)
            progress(f"{voice.id}: {questionnaire.name}, {len(report.rendered)} новых реплик")
        if not all(verify(q, target).ready for q in waves):
            raise RuntimeError(f"неполная библиотека {voice.id}")
        for node_id in SAMPLES:
            node = reference.node(node_id)
            # Телефонный пример — тот же файл, который проигрывает демо.
            phone = target / f"{node.audio_key or node.id}.wav"
            (target / f"sample_{node_id}_8000.wav").write_bytes(phone.read_bytes())
            studio = synthesizer.synthesize(node.text, voice=voice.id, sample_rate=24000)
            (target / f"sample_{node_id}_24000.wav").write_bytes(studio.wav)
        progress(f"{voice.id}: готово")
