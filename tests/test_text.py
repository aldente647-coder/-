from __future__ import annotations

import pytest

from socmon.core.text import normalize, parse_number, tokenize


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Ещё раз, пожалуйста!", "еще раз пожалуйста"),
        ("ДА, КОНЕЧНО.", "да конечно"),
        ("тридцать   пять", "тридцать пять"),
        ("Всё-таки нет", "все-таки нет"),
    ],
)
def test_normalize(raw: str, expected: str) -> None:
    assert normalize(raw) == expected


def test_yo_survives_decomposition() -> None:
    """ё заменяется до NFKD: иначе разложение оставит висящий диерезис."""
    assert normalize("ёлка") == "елка"
    assert tokenize(normalize("ёлка")) == ["елка"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("восемь", 8),
        ("8", 8),
        ("где-то восьмерка", 8),
        ("ну наверное семь", 7),
        ("ноль", 0),
        ("десять", 10),
        ("десятку поставлю", 10),
    ],
)
def test_parse_number_ok(raw: str, expected: int) -> None:
    assert parse_number(raw, 0, 10) == expected


@pytest.mark.parametrize("raw", ["то ли семь то ли восемь", "не знаю", "сто", "-3"])
def test_parse_number_rejects(raw: str) -> None:
    """Два разных числа — не ответ, а повод переспросить."""
    assert parse_number(raw, 0, 10) is None


def test_parse_number_repeated_same_value() -> None:
    assert parse_number("семь, семь конечно", 0, 10) == 7


# ------------------------------------------------- составные числительные

@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("восемнадцать", 18),
        ("двадцать один", 21),
        ("тридцать два", 32),
        ("сорок пять", 45),
        ("семьдесят", 70),
        ("шестьдесят восемь", 68),
        ("девяносто четыре", 94),
        ("мне 32 года", 32),
        ("мне тридцать два года", 32),
    ],
)
def test_parse_age_in_words(raw: str, expected: int) -> None:
    """Возраст называют словами чаще, чем цифрами.

    Без склейки «сорок пять» разбиралось бы на два разных числа и
    отбраковывалось правилом неоднозначности — то есть возраст словами
    не принимался бы вовсе.
    """
    assert parse_number(raw, 18, 94) == expected


@pytest.mark.parametrize("raw", ["мне уже много", "не скажу", "то ли сорок то ли пятьдесят"])
def test_parse_age_rejects_non_answers(raw: str) -> None:
    assert parse_number(raw, 18, 94) is None


def test_tens_do_not_leak_into_the_short_scale() -> None:
    """Шкала 0-10 от появления десятков не пострадала: «сорок пять»
    в неё не попадает, а «семь» по-прежнему разбирается."""
    assert parse_number("сорок пять", 0, 10) is None
    assert parse_number("семь", 0, 10) == 7
