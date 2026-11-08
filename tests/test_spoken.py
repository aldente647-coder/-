"""Речевая разметка не меняет слов — только паузы и произношение."""

from __future__ import annotations

from pathlib import Path

import pytest

from socmon.core.questionnaire import Questionnaire
from socmon.voice.delivery import Catalogue
from socmon.voice.render import speakable
from socmon.voice.spoken import same_words, spoken, words

WAVES = sorted(Path("src/socmon/questionnaires").glob("wave-*.yaml"))


def test_enumeration_becomes_separate_phrases() -> None:
    text = (
        "Как вы оцениваете положение: денег хватает на всё; хватает на повседневные "
        "расходы; или не хватает даже на самое необходимое?"
    )
    result = spoken(text)
    assert ";" not in result
    assert "всё. Хватает" in result
    assert ", или не хватает" in result
    assert same_words(text, result)


def test_alternative_question_pauses_before_or() -> None:
    assert spoken("Вы довольны или не довольны?") == "Вы довольны, или не довольны?"
    # Уже стоящая запятая не удваивается.
    assert spoken("хорошо, или плохо?") == "хорошо, или плохо?"


def test_latin_names_are_pronounced_in_russian() -> None:
    result = spoken("Каналы в Max; Telegram-каналы; мессенджеры WhatsApp и Viber.")
    assert "Макс" in result and "Телеграм" in result and "Ватсап" in result
    assert "Max" not in result and "Telegram" not in result


def test_dash_becomes_a_comma_pause() -> None:
    assert spoken("Жильё — возможность снять?") == "Жильё, возможность снять?"


@pytest.mark.parametrize("path", WAVES, ids=[p.stem for p in WAVES])
def test_every_questionnaire_prompt_keeps_its_words(path: Path) -> None:
    """ТЗ §8.1: робот произносит утверждённую формулировку. Разметка
    обязана оставить каждое слово на месте — во всех четырёх волнах."""
    questionnaire = Questionnaire.from_yaml(path)
    for node in speakable(questionnaire):
        assert same_words(node.text, spoken(node.text)), node.id


def test_service_phrases_keep_their_words() -> None:
    for phrase in Catalogue.from_yaml().phrases():
        for gender in ("f", "m"):
            text = phrase.spoken(gender)  # type: ignore[arg-type]
            assert same_words(text, spoken(text)), phrase.key


def test_words_are_compared_without_case_and_yo() -> None:
    assert words("Жильё, ЁЖИК") == words("жилье ежик")
