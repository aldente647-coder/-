"""Демо играет подачу: отклики, паузы и дубли — как настоящий звонок."""

from __future__ import annotations

from pathlib import Path

from socmon.core.questionnaire import Questionnaire
from socmon.voice.delivery import Catalogue, Pool
from socmon.voice.render import MANIFEST_NAME, VoiceManifest, render, render_catalogue
from socmon.web.demo import DemoStudio
from tests.test_voice import NOW, QUESTIONNAIRES, FakeSynthesizer, TakingSynthesizer


def library(tmp_path: Path) -> tuple[Questionnaire, Catalogue, VoiceManifest]:
    base = Questionnaire.from_yaml(QUESTIONNAIRES / "wave-base.yaml")
    catalogue = Catalogue.from_yaml()
    render(base, TakingSynthesizer(), tmp_path, voice="anna", now=NOW)
    render_catalogue(catalogue, FakeSynthesizer(), tmp_path, voice="anna", gender="f", now=NOW)
    return base, catalogue, VoiceManifest.load(tmp_path / MANIFEST_NAME)


def test_demo_thanks_for_an_open_answer_and_pauses(tmp_path: Path) -> None:
    base, catalogue, manifest = library(tmp_path)
    studio = DemoStudio(base, catalogue=catalogue, gender="f", library=manifest, directory=tmp_path)
    token, step = studio.start()
    assert step.lines[0].kind == "prompt" and step.lines[0].gap_ms == 0

    replies = ["да", "да", "да", "доволен", "не изменилась", "доверяю", "доверяю",
               "не доверяю", "хорошо", "хорошо", "справедливо",
               "хватает только на самое необходимое", "скорее в безопасности",
               "не изменились", "удовлетворительное"]
    for reply in replies:
        step = studio.answer(token, reply)  # type: ignore[assignment]
    assert step.node_id == "q11_problems"
    step = studio.answer(token, "цены выросли и дороги разбиты")  # type: ignore[assignment]

    thanks = {p.key for p in catalogue.pools[Pool.ACK_OPEN]}
    assert step.lines[0].kind == "ack"
    assert step.lines[0].media in thanks
    assert 500 <= step.lines[0].gap_ms <= 900
    assert step.lines[-1].kind == "prompt"
    assert step.lines[-1].media.startswith("q12_prices_v1")


def test_demo_spreads_takes_across_sessions(tmp_path: Path) -> None:
    """Два прохода одной анкеты не обязаны звучать одинаково."""
    base, catalogue, manifest = library(tmp_path)
    studio = DemoStudio(base, catalogue=catalogue, gender="f", library=manifest, directory=tmp_path)
    media = {studio.start()[1].lines[0].media for _ in range(40)}
    assert media == {"intro_v1", "intro_v1-t2", "intro_v1-t3"}


def test_demo_without_a_library_plays_only_the_questionnaire(tmp_path: Path) -> None:
    base = Questionnaire.from_yaml(QUESTIONNAIRES / "wave-base.yaml")
    studio = DemoStudio(base, catalogue=Catalogue.from_yaml(), gender="f", library=None)
    token, step = studio.start()
    for reply in ("да", "да", "да"):
        step = studio.answer(token, reply)  # type: ignore[assignment]
    # Без библиотеки откликов подача их не просит, но паузы выдерживает.
    assert all(line.kind == "prompt" for line in step.lines)
    assert step.lines[0].gap_ms > 0
