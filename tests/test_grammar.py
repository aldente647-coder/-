from __future__ import annotations

import pytest

from socmon.core.grammar import (
    ControlIntent,
    MatchKind,
    OptionGrammar,
    match_control,
    match_options,
)

YES = OptionGrammar.build(1, ["да", "согласен"])
NO = OptionGrammar.build(2, ["нет", "не согласен"])
RATHER_YES = OptionGrammar.build(3, ["скорее да"])


def test_longest_match_wins() -> None:
    """«скорее да» должно выиграть у «да», иначе шкалы согласия схлопнутся."""
    result = match_options("ну скорее да", (YES, NO, RATHER_YES))
    assert result.kind is MatchKind.ANSWER
    assert result.code == 3


def test_word_boundaries() -> None:
    """Без границ слова «да» ловилось бы в «дальше», а «нет» — в «нету»."""
    assert match_options("дальше что", (YES, NO)).kind is MatchKind.NONE
    assert match_options("нету времени", (YES, NO)).kind is MatchKind.NONE


def test_contradiction_is_not_an_answer() -> None:
    """Два варианта на непересекающихся участках фразы — противоречие."""
    result = match_options("да нет", (YES, NO))
    assert result.kind is MatchKind.AMBIGUOUS
    assert result.candidates == (1, 2)


def test_nested_match_is_not_a_contradiction() -> None:
    """«скорее да» содержит «да» — это вложение, а не спор двух вариантов."""
    result = match_options("скорее да", (YES, NO, RATHER_YES))
    assert result.kind is MatchKind.ANSWER
    assert result.code == 3


def test_unknown_returns_none() -> None:
    assert match_options("сегодня дождь", (YES, NO)).kind is MatchKind.NONE


@pytest.mark.parametrize(
    ("phrase", "intent"),
    [
        ("больше не звоните мне", ControlIntent.DONT_CALL),
        ("удалите мой номер", ControlIntent.DONT_CALL),
        ("дайте оператора", ControlIntent.OPERATOR),
        ("повторите пожалуйста", ControlIntent.REPEAT),
        ("перезвоните позже", ControlIntent.CALLBACK),
        ("сейчас неудобно", ControlIntent.CALLBACK),
    ],
)
def test_control_intents(phrase: str, intent: ControlIntent) -> None:
    result = match_control(phrase)
    assert result.kind is MatchKind.CONTROL
    assert result.intent is intent


@pytest.mark.parametrize("phrase", ["не интересно", "нет", "не хочу", "не надо"])
def test_refusal_is_not_a_do_not_call_request(phrase: str) -> None:
    """Ключевой урок блюпринта: «не интересно» — отказ от текущего разговора,
    звонить можно позже. «Не звоните мне» — запрет и стоп-лист навсегда.
    Смешать их — значит либо жечь базу, либо нарушать прямой запрет (ТЗ §6.2)."""
    assert match_control(phrase).kind is MatchKind.NONE


def test_longest_control_phrase_wins() -> None:
    result = match_control("больше не звоните")
    assert result.phrase == "больше не звоните"
