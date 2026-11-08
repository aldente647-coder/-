"""Озвучка анкеты. Проверяется без torch — на подставном синтезаторе."""

from __future__ import annotations

import io
import json
import wave
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from socmon.core.questionnaire import Questionnaire
from socmon.voice.render import (
    MANIFEST_NAME,
    RenderError,
    VoiceManifest,
    VoiceReport,
    render,
    speakable,
    takes_available,
    text_hash,
    verify,
)
from socmon.voice.synth import (
    RUSSIAN_VOICES,
    TELEPHONY_SAMPLE_RATE,
    Speech,
    SpeechSynthesizer,
    SynthesisError,
    prepare,
)

NOW = datetime(2026, 8, 20, 18, 0, tzinfo=UTC)


class FakeSynthesizer:
    """Пишет настоящий WAV, длительность выводит из длины текста."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int]] = []

    def synthesize(self, text: str, *, voice: str, sample_rate: int) -> Speech:
        self.calls.append((prepare(text), voice, sample_rate))
        frames = max(1, len(prepare(text)) * sample_rate // 15)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(b"\x00\x00" * frames)
        return Speech(buffer.getvalue(), sample_rate, frames / sample_rate)


@pytest.fixture
def synth() -> FakeSynthesizer:
    return FakeSynthesizer()


def render_core(
    core: Questionnaire, synth: FakeSynthesizer, out: Path, **kwargs: object
) -> VoiceReport:
    return render(core, synth, out, voice="baya", now=NOW, **kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------- подготовка текста


def test_yaml_line_wrapping_is_collapsed() -> None:
    assert prepare("Здравствуйте.\n  Проводится   опрос. ") == (
        "Здравствуйте. Проводится опрос."
    )


def test_nothing_but_whitespace_is_touched() -> None:
    """Любая правка текста означала бы, что робот произносит не то,
    что утверждено (ТЗ §8.1)."""
    original = "Ноль — совсем не доволен, десять — полностью доволен."
    assert prepare(original) == original


def test_reformatting_does_not_change_the_hash() -> None:
    assert text_hash("вопрос\n  из двух строк") == text_hash("вопрос из двух строк")


# ------------------------------------------------------------------------ рендер


def test_every_spoken_node_gets_a_file(
    core: Questionnaire, synth: FakeSynthesizer, tmp_path: Path
) -> None:
    report = render_core(core, synth, tmp_path)
    assert report.total == len(speakable(core))
    assert len(list(tmp_path.glob("*.wav"))) == report.total
    assert (tmp_path / MANIFEST_NAME).exists()


def test_silent_terminal_nodes_are_not_voiced(core: Questionnaire) -> None:
    """Узел с пустым текстом существует только чтобы зафиксировать исход."""
    keys = {n.id for n in speakable(core)}
    assert "end_completed" not in keys


def test_telephony_sample_rate_is_the_default(
    core: Questionnaire, synth: FakeSynthesizer, tmp_path: Path
) -> None:
    """Восемь килогерц — то, что реально услышит респондент. Демонстрировать
    Заказчику 24 кГц значит утверждать голос, который на линии зазвучит иначе."""
    report = render_core(core, synth, tmp_path)
    assert report.manifest.sample_rate == TELEPHONY_SAMPLE_RATE
    with wave.open(str(tmp_path / f"{report.manifest.entries['intro_v1'].file}")) as handle:
        assert handle.getframerate() == TELEPHONY_SAMPLE_RATE
        assert handle.getnchannels() == 1


def test_manifest_is_pinned_to_the_questionnaire_version(
    core: Questionnaire, synth: FakeSynthesizer, tmp_path: Path
) -> None:
    report = render_core(core, synth, tmp_path)
    assert report.manifest.version_id == core.version_id
    assert report.manifest.voice == "baya"


def test_unknown_voice_is_refused(
    core: Questionnaire, synth: FakeSynthesizer, tmp_path: Path
) -> None:
    with pytest.raises(RenderError, match="неизвестный голос"):
        render(core, synth, tmp_path, voice="скарлетт")


def test_uploaded_revision_uses_its_own_voice_library(
    core: Questionnaire, synth: FakeSynthesizer, tmp_path: Path,
) -> None:
    from socmon.voice.catalog import questionnaire_directory

    assert questionnaire_directory(tmp_path, core.version_id) == tmp_path
    dedicated = tmp_path / 'questionnaires' / core.version_id / 'anna'
    render(core, synth, dedicated, voice='anna')
    assert questionnaire_directory(tmp_path, core.version_id) == dedicated
    assert verify(core, questionnaire_directory(tmp_path, core.version_id)).ready
    # Мужской вариант не подменяется женским, если ещё не собран.
    assert questionnaire_directory(tmp_path, core.version_id, 'alex') == tmp_path / 'voices/alex'


# ------------------------------------------------------------ переиспользование


def test_second_run_renders_nothing_new(
    core: Questionnaire, synth: FakeSynthesizer, tmp_path: Path
) -> None:
    """§8.3 требует стабильности голоса между волнами, а каждый прогон синтеза
    звучит чуть иначе — значит, переозвучивать без нужды нельзя."""
    render_core(core, synth, tmp_path)
    synth.calls.clear()

    again = render_core(core, synth, tmp_path)
    assert synth.calls == []
    assert again.rendered == ()
    assert len(again.reused) == again.total


def test_only_the_changed_prompt_is_re_rendered(
    core: Questionnaire, synth: FakeSynthesizer, tmp_path: Path
) -> None:
    render_core(core, synth, tmp_path)
    synth.calls.clear()

    edited = core.model_copy(update={
        "nodes": tuple(
            n.model_copy(update={"text": "Совсем другой вопрос."})
            if n.id == "q_anxiety" else n
            for n in core.nodes
        )
    })
    report = render(edited, synth, tmp_path, voice="baya", now=NOW)
    assert report.rendered == ("q_anxiety_v1",)
    assert len(synth.calls) == 1


def test_force_re_renders_everything(
    core: Questionnaire, synth: FakeSynthesizer, tmp_path: Path
) -> None:
    render_core(core, synth, tmp_path)
    report = render_core(core, synth, tmp_path, force=True)
    assert report.reused == ()


def test_changing_the_voice_invalidates_the_library(
    core: Questionnaire, synth: FakeSynthesizer, tmp_path: Path
) -> None:
    """Смешивать два голоса в одной анкете нельзя (ТЗ §8.3)."""
    render_core(core, synth, tmp_path)
    report = render(core, synth, tmp_path, voice="eugene", now=NOW)
    assert report.reused == ()
    assert report.manifest.voice == "eugene"


def test_deleted_file_is_rendered_again(
    core: Questionnaire, synth: FakeSynthesizer, tmp_path: Path
) -> None:
    report = render_core(core, synth, tmp_path)
    (tmp_path / report.manifest.entries["intro_v1"].file).unlink()
    again = render_core(core, synth, tmp_path)
    assert "intro_v1" in again.rendered


# ------------------------------------------------------------------- проверка


def test_fresh_render_is_ready(
    core: Questionnaire, synth: FakeSynthesizer, tmp_path: Path
) -> None:
    render_core(core, synth, tmp_path)
    check = verify(core, tmp_path)
    assert check.ready
    assert check.problems == ()
    assert check.version_mismatch is False


def test_absent_library_reports_everything_missing(
    core: Questionnaire, tmp_path: Path
) -> None:
    check = verify(core, tmp_path)
    assert not check.ready
    assert len(check.missing) == len(speakable(core))


def test_edited_wording_makes_the_recording_stale(
    core: Questionnaire, synth: FakeSynthesizer, tmp_path: Path
) -> None:
    """Самый опасный случай: сценарий выглядит озвученным, робот читает
    прошлую редакцию вопроса, а в массив пишется код по новой."""
    render_core(core, synth, tmp_path)
    edited = core.model_copy(update={
        "nodes": tuple(
            n.model_copy(update={"text": "Переформулированный вопрос."})
            if n.id == "q_trust" else n
            for n in core.nodes
        )
    })
    check = verify(edited, tmp_path)
    assert check.stale == ("q_trust_v1",)
    assert not check.ready
    assert any("формулировка изменилась" in p for p in check.problems)


def test_missing_file_is_detected_even_with_a_manifest(
    core: Questionnaire, synth: FakeSynthesizer, tmp_path: Path
) -> None:
    report = render_core(core, synth, tmp_path)
    (tmp_path / report.manifest.entries["q_life_v1"].file).unlink()
    check = verify(core, tmp_path)
    assert check.missing == ("q_life_v1",)


def test_orphaned_recordings_are_reported_but_not_blocking(
    core: Questionnaire, synth: FakeSynthesizer, tmp_path: Path
) -> None:
    render_core(core, synth, tmp_path)
    manifest = json.loads((tmp_path / MANIFEST_NAME).read_text(encoding="utf-8"))
    manifest["entries"]["old_question_v1"] = {
        "audio_key": "old_question_v1", "node_id": "gone",
        "text_hash": "deadbeef", "file": "old_question_v1.wav",
        "duration_sec": 3.0, "rendered_at": NOW.isoformat(),
    }
    (tmp_path / MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    check = verify(core, tmp_path)
    assert check.orphaned == ("old_question_v1",)
    assert check.ready


def test_manifest_survives_a_round_trip(
    core: Questionnaire, synth: FakeSynthesizer, tmp_path: Path
) -> None:
    original = render_core(core, synth, tmp_path).manifest
    assert VoiceManifest.load(tmp_path / MANIFEST_NAME) == original


# ------------------------------------------------------------------- синтез


def test_duration_is_estimated_from_real_prompt_lengths(
    core: Questionnaire, synth: FakeSynthesizer, tmp_path: Path
) -> None:
    """ТЗ §3.3: не более 10 минут у 90% завершённых интервью. Оценка нижняя —
    ветвление и переспросы её только удлиняют."""
    from socmon.voice.render import estimate_duration

    manifest = render_core(core, synth, tmp_path).manifest
    estimate = estimate_duration(core, manifest)

    assert estimate.questions > 15
    assert estimate.robot_sec > 0
    assert estimate.total_sec > estimate.robot_sec
    assert "мин" in estimate.note


def test_straight_path_skips_unused_terminals(core: Questionnaire) -> None:
    """Из десяти завершений в одном интервью звучит ровно одно."""
    from socmon.voice.render import straight_path

    ids = [n.id for n in straight_path(core)]
    assert ids[0] == "intro"
    assert ids[-1] == "end_completed"
    assert "end_dont_call" not in ids


def test_two_voices_are_offered_for_approval() -> None:
    """ТЗ §8.3: до пилота представить не менее двух вариантов —
    мужской и женский."""
    genders = set(RUSSIAN_VOICES.values())
    assert {"мужской", "женский"} <= genders


def test_silero_refuses_an_unknown_voice() -> None:
    from socmon.voice.synth import SileroSynthesizer

    with pytest.raises(SynthesisError, match="неизвестный голос"):
        SileroSynthesizer().synthesize("текст", voice="нет такого", sample_rate=8000)


def test_silero_refuses_an_unsupported_rate() -> None:
    from socmon.voice.synth import SileroSynthesizer

    with pytest.raises(SynthesisError, match="не поддерживается"):
        SileroSynthesizer().synthesize("текст", voice="baya", sample_rate=16000)


def test_fake_synthesizer_satisfies_the_contract(synth: FakeSynthesizer) -> None:
    assert isinstance(synth, SpeechSynthesizer)


# ----------------------------------------- одна библиотека на несколько волн

QUESTIONNAIRES = Path(__file__).resolve().parents[1] / "src/socmon/questionnaires"


def test_rendering_a_second_wave_keeps_the_first_one_voiced(
    synth: FakeSynthesizer, tmp_path: Path
) -> None:
    """Волны делят одну библиотеку реплик, и рендер второй не должен
    обесточивать первую.

    Ядро у всех волн общее и переозвучке не подлежит — §8.3 требует
    стабильности голоса. Если бы манифест описывал только последнюю
    анкету, файлы предыдущей оставались бы на диске, а проверка готовности
    сообщала бы, что волну запускать нельзя.
    """
    social = Questionnaire.from_yaml(QUESTIONNAIRES / "wave-social.yaml")
    governance = Questionnaire.from_yaml(QUESTIONNAIRES / "wave-governance.yaml")

    render(social, synth, tmp_path, voice="baya", now=NOW)
    assert verify(social, tmp_path).ready

    render(governance, synth, tmp_path, voice="baya", now=NOW)
    assert verify(governance, tmp_path).ready
    assert verify(social, tmp_path).ready


def test_shared_core_is_voiced_once_for_all_waves(
    synth: FakeSynthesizer, tmp_path: Path
) -> None:
    """Общие реплики переиспользуются, а не синтезируются заново."""
    base = Questionnaire.from_yaml(QUESTIONNAIRES / "wave-base.yaml")
    social = Questionnaire.from_yaml(QUESTIONNAIRES / "wave-social.yaml")

    render(base, synth, tmp_path, voice="baya", now=NOW)
    second = render(social, synth, tmp_path, voice="baya", now=NOW)

    # Заново озвучен только модуль волны, ядро взято готовым.
    assert set(second.rendered) == {
        n.audio_key or n.id for n in social.nodes
        if (n.audio_key or n.id) not in {m.audio_key or m.id for m in base.nodes}
        and n.text.strip()
    }
    assert len(second.reused) > len(second.rendered)


def test_changing_the_voice_still_starts_a_clean_library(
    synth: FakeSynthesizer, tmp_path: Path
) -> None:
    """Накопление записей не должно смешивать два голоса в одной анкете."""
    base = Questionnaire.from_yaml(QUESTIONNAIRES / "wave-base.yaml")
    render(base, synth, tmp_path, voice="baya", now=NOW)
    report = render(base, synth, tmp_path, voice="eugene", now=NOW)
    assert not report.reused
    assert {e.audio_key for e in report.manifest.entries.values()} == set(report.rendered)


def test_ssml_preserves_words_and_escapes_markup() -> None:
    import xml.etree.ElementTree as ET

    from socmon.voice.synth import calm_ssml

    text = 'Здравствуйте. Вы согласны? Условие < 5 & 6 > 3.'
    root = ET.fromstring(calm_ssml(text))
    assert ' '.join(' '.join(root.itertext()).split()) == text
    assert len(root.findall('break')) == 2
    assert root.find('prosody') is None


def test_ssml_pronounces_latin_brand_names_in_russian() -> None:
    from socmon.voice.synth import calm_ssml

    assert 'Макс; Телеграм-каналы' in calm_ssml('Max; Telegram-каналы')


def test_synthesis_recipe_change_invalidates_cached_audio(
    core: Questionnaire, tmp_path: Path,
) -> None:
    class VersionedSynth(FakeSynthesizer):
        fingerprint = 'model-v1'

    synth = VersionedSynth()
    first = render_core(core, synth, tmp_path)
    assert not render_core(core, synth, tmp_path).rendered
    synth.fingerprint = 'model-v2'
    updated = render_core(core, synth, tmp_path)
    assert updated.rendered == first.rendered
    assert updated.manifest.synthesis_id == 'model-v2'


def test_old_voice_manifest_still_loads(core: Questionnaire, tmp_path: Path) -> None:
    report = render_core(core, FakeSynthesizer(), tmp_path)
    payload = json.loads(report.manifest.to_json())
    payload.pop('synthesis_id')
    path = tmp_path / MANIFEST_NAME
    path.write_text(json.dumps(payload))
    assert VoiceManifest.load(path).synthesis_id == 'legacy'


def test_variant_builder_rejects_conflicting_wave_audio_keys(
    tmp_path: Path,
) -> None:
    import shutil

    from socmon.voice.variants import build_variants

    source = tmp_path / 'questionnaires'
    original = Path(__file__).resolve().parents[1] / 'src/socmon/questionnaires'
    shutil.copytree(original, source)
    # Два исторически разных текста, названные одним audio_key.
    shutil.copyfile(source / 'core-v1.yaml', source / 'wave-conflict.yaml')
    with pytest.raises(RenderError, match='разные формулировки'):
        build_variants(tmp_path / 'voices', FakeSynthesizer(), questionnaires=source)
    assert not (tmp_path / 'voices').exists()


# ------------------------------------------------------ дубли и служебные реплики


class TakingSynthesizer(FakeSynthesizer):
    """Три дубля на реплику; второй — с движением высоты, остальные ровные.

    Так лучший дубль известен заранее, и выбор по просодии проверяется
    точно, а не «какой-нибудь»."""

    takes = 3

    def synthesize(
        self, text: str, *, voice: str, sample_rate: int, take: int = 0,
    ) -> Speech:
        self.calls.append((prepare(text), voice, sample_rate))
        seconds = max(0.6, len(prepare(text)) / 15)
        t = np.arange(int(sample_rate * seconds)) / sample_rate
        freq = 180 + (60 * t / seconds if take == 1 else 0)
        audio = 0.4 * np.sin(2 * np.pi * freq * t)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes((audio * 32767).astype("<i2").tobytes())
        return Speech(buffer.getvalue(), sample_rate, seconds)


def test_render_keeps_every_take_and_puts_the_liveliest_first(tmp_path: Path) -> None:
    base = Questionnaire.from_yaml(QUESTIONNAIRES / "wave-base.yaml")
    synth = TakingSynthesizer()
    report = render(base, synth, tmp_path, voice="anna", now=NOW)

    entry = report.manifest.entries["consent_v1"]
    assert entry.file == "consent_v1.wav"
    assert entry.takes == ("consent_v1-t2.wav", "consent_v1-t3.wav")
    assert takes_available(entry, tmp_path) == 3
    # Три вызова синтеза на реплику, а не один.
    assert sum(1 for c in synth.calls if c[0] == "Вы согласны принять участие?") == 3

    # Основной файл — тот, где голос движется: второй дубль синтезатора.
    from socmon.voice.prosody import measure

    with wave.open(str(tmp_path / "consent_v1.wav")) as handle:
        frames = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2") / 32768
    assert measure(frames, 8000, "Вы согласны принять участие?").pitch_std_semitones > 1.5

    # Повторный рендер ничего не пересинтезирует и не теряет дубли.
    again = render(base, TakingSynthesizer(), tmp_path, voice="anna", now=NOW)
    assert not again.rendered
    assert again.manifest.entries["consent_v1"].takes == entry.takes
    assert verify(base, tmp_path).ready


def test_manifest_round_trips_takes(tmp_path: Path) -> None:
    base = Questionnaire.from_yaml(QUESTIONNAIRES / "wave-base.yaml")
    render(base, TakingSynthesizer(), tmp_path, voice="anna", now=NOW)
    loaded = VoiceManifest.load(tmp_path / MANIFEST_NAME)
    assert loaded.entries["consent_v1"].takes == ("consent_v1-t2.wav", "consent_v1-t3.wav")


def test_fewer_takes_remove_the_leftovers(tmp_path: Path) -> None:
    """Дубли прошлой сборки сверх нынешнего числа подача сочла бы живыми."""
    base = Questionnaire.from_yaml(QUESTIONNAIRES / "wave-base.yaml")
    render(base, TakingSynthesizer(), tmp_path, voice="anna", now=NOW)
    assert (tmp_path / "consent_v1-t3.wav").exists()
    report = render(base, FakeSynthesizer(), tmp_path, voice="anna", now=NOW)
    assert "consent_v1" in report.rendered
    assert report.manifest.entries["consent_v1"].takes == ()
    assert not (tmp_path / "consent_v1-t3.wav").exists()


def test_service_phrases_join_the_same_library(tmp_path: Path) -> None:
    from socmon.voice.delivery import Catalogue
    from socmon.voice.render import render_catalogue, verify_delivery

    base = Questionnaire.from_yaml(QUESTIONNAIRES / "wave-base.yaml")
    catalogue = Catalogue.from_yaml()
    render(base, FakeSynthesizer(), tmp_path, voice="alex", now=NOW)
    assert not verify_delivery(catalogue, tmp_path, "m").ready

    synth = FakeSynthesizer()
    report = render_catalogue(catalogue, synth, tmp_path, voice="alex", gender="m", now=NOW)
    assert (tmp_path / "ack_ugu.wav").exists()
    assert report.manifest.entries["rep_ne_rasslyshal"].node_id == "service"
    # Мужской голос получил мужскую форму.
    assert any("не расслышал." in text for text, _, _ in synth.calls)
    assert verify_delivery(catalogue, tmp_path, "m").ready
    # Вопросы анкеты при этом не потерялись из манифеста.
    assert verify(base, tmp_path).ready
    # А женская форма для этой библиотеки — устаревшая.
    assert verify_delivery(catalogue, tmp_path, "f").stale
