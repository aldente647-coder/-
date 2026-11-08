"""Проверка библиотеки распознаванием и отсев провалившихся дублей."""

from __future__ import annotations

from pathlib import Path

from socmon.core.questionnaire import Questionnaire
from socmon.voice.qa import audit, compare
from socmon.voice.render import MANIFEST_NAME, VoiceManifest, render, speakable
from tests.test_voice import NOW, QUESTIONNAIRES, TakingSynthesizer


def test_compare_ignores_what_recognition_cannot_reproduce() -> None:
    exact, similarity, missing, extra = compare(
        "Каналы в Max, Telegram-каналы.", "каналы в макс телеграм каналы"
    )
    assert exact and similarity == 1.0 and not missing and not extra


def test_recogniser_slips_in_endings_and_numerals_still_pass() -> None:
    """«18» вместо «восемнадцать», «напряжения» вместо «напряжение» —
    промахи распознавателя, а не записи."""
    from socmon.voice.qa import Check

    for expected, heard in [
        ("вам уже исполнилось восемнадцать лет?", "вам уже исполнилось 18 лет"),
        ("ваша личная тревога, напряжение выросли", "ваша личная тревога напряжения выросли"),
        ("вы доверяете или не доверяете?", "вы доверяете или недоверяете"),
        ("образования — колледжей, техникумов?", "образования в колледже и техникумов"),
    ]:
        exact, similarity, missing, extra = compare(expected, heard)
        check = Check("k", "f", expected, heard, exact, similarity, missing, extra)
        assert check.passed, (expected, heard, similarity)
    # А настоящий повтор и пропуск слова — нет.
    exact, similarity, *_ = compare("Вы согласны принять участие?", "вы вы согласны участие")
    assert not exact and similarity < 0.9


def test_compare_reports_dropped_and_repeated_words() -> None:
    exact, _, missing, extra = compare(
        "Вы согласны принять участие?", "вы согласны согласны участие"
    )
    assert not exact
    assert missing == ("принять",)
    assert extra == ("согласны",)


def test_audit_prunes_takes_that_fail_and_keeps_the_rest(tmp_path: Path) -> None:
    base = Questionnaire.from_yaml(QUESTIONNAIRES / "wave-base.yaml")
    render(base, TakingSynthesizer(), tmp_path, voice="anna", now=NOW)
    texts = {n.audio_key or n.id: n.text for n in speakable(base)}

    def fake_transcribe(path: Path) -> str:
        # Третий дубль согласия «заикается», всё остальное — идеально.
        if path.name == "consent_v1-t3.wav":
            return "вы согласны согласны принять участие"
        return texts[path.stem.split("-t")[0]]

    report = audit(tmp_path, texts, fake_transcribe, prune=True, keys=["consent_v1", "intro_v1"])
    assert report.pruned == ("consent_v1-t3.wav",)
    assert not report.failed_primary
    assert not (tmp_path / "consent_v1-t3.wav").exists()
    manifest = VoiceManifest.load(tmp_path / MANIFEST_NAME)
    assert manifest.entries["consent_v1"].takes == ("consent_v1-t2.wav",)
    assert manifest.entries["intro_v1"].takes == ("intro_v1-t2.wav", "intro_v1-t3.wav")
    assert report.passed == len(report.checks) - 1
    assert report.exact == report.passed


def test_audit_promotes_a_good_take_when_the_primary_fails(tmp_path: Path) -> None:
    base = Questionnaire.from_yaml(QUESTIONNAIRES / "wave-base.yaml")
    render(base, TakingSynthesizer(), tmp_path, voice="anna", now=NOW)
    texts = {n.audio_key or n.id: n.text for n in speakable(base)}

    def fake_transcribe(path: Path) -> str:
        if path.name == "consent_v1.wav":
            return "вы согласны участие"
        return texts[path.stem.split("-t")[0]]

    report = audit(tmp_path, texts, fake_transcribe, prune=True, keys=["consent_v1"])
    assert not report.failed_primary
    manifest = VoiceManifest.load(tmp_path / MANIFEST_NAME)
    assert (tmp_path / "consent_v1.wav").exists()
    assert manifest.entries["consent_v1"].takes == ("consent_v1-t3.wav",)


def test_audit_flags_a_prompt_with_no_good_recording(tmp_path: Path) -> None:
    base = Questionnaire.from_yaml(QUESTIONNAIRES / "wave-base.yaml")
    render(base, TakingSynthesizer(), tmp_path, voice="anna", now=NOW)
    texts = {n.audio_key or n.id: n.text for n in speakable(base)}
    report = audit(
        tmp_path, texts, lambda _: "ничего не разобрать", prune=True, keys=["consent_v1"]
    )
    assert report.failed_primary == ("consent_v1",)
    # Ничего не удалено: заменить нечем, решать человеку.
    assert (tmp_path / "consent_v1.wav").exists()
