"""Речевая разметка: как формулировка анкеты произносится, а не читается.

Текст анкеты написан для бумаги. Перечисление вариантов ответа через точку
с запятой глаз читает как список; синтез читает его как одну длинную фразу
на одном дыхании — и это звучит машиной сильнее любого тембра. Живой
интервьюер делает паузу перед каждым вариантом и ещё одну перед «или».

Разметка меняет только знаки препинания и произношение латинских названий.
Слова остаются те же самые, что утверждены Заказчиком (ТЗ §8.1), и это
проверяется тестом: набор слов до и после разметки обязан совпасть.
"""

from __future__ import annotations

import re

#: Произношение латинских названий. Хеш текста считается от исходной
#: формулировки, поэтому анкета и её версия не меняются.
PRONUNCIATION: dict[str, str] = {
    "Telegram": "Телеграм",
    "Max": "Макс",
    "WhatsApp": "Ватсап",
    "Viber": "Вайбер",
}

_WORD = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)
_SPACES = re.compile(r"\s+")


def pronounce(text: str) -> str:
    """Латинские названия — по-русски. Синтез иначе читает их по буквам."""
    for name, spoken in PRONUNCIATION.items():
        text = re.sub(rf"\b{name}\b", spoken, text, flags=re.IGNORECASE)
    return text


def spoken(text: str) -> str:
    """Формулировка в том виде, в каком её стоит произносить."""
    result = _SPACES.sub(" ", pronounce(text)).strip()

    # Перечисление через точку с запятой: каждый вариант — своей фразой,
    # с полноценной паузой. Точка вместо «;» даёт синтезу законченную
    # интонацию варианта, а не бесконечную запятую.
    if ";" in result:
        # «; или» закрывает перечисление — это пауза перед «или»,
        # а не начало нового предложения.
        result = re.sub(r"\s*;\s*(?=или\b)", ", ", result)
        result = re.sub(r"\s*;\s*", ". ", result)
        result = _capitalize_after_period(result)

    # «Или» в конце перечисления — после паузы: «…хорошо, или плохо?»
    result = re.sub(r"(?<=[а-яё])\s+или\s+(?=[а-яё])", ", или ", result)
    result = result.replace(",, или", ", или").replace(", , или", ", или")

    # Тире — пауза с продолжением мысли. Синтез тире часто проглатывает,
    # запятая с многоточием читается надёжнее.
    result = re.sub(r"\s+[—–-]\s+", ", ", result)
    return result


def _capitalize_after_period(text: str) -> str:
    parts = text.split(". ")
    return ". ".join(p[:1].upper() + p[1:] if p else p for p in parts)


def words(text: str) -> list[str]:
    """Слова без учёта регистра, «ё» и пунктуации — для сверки.

    Числа приводятся к словам: распознаватель пишет «18», анкета говорит
    «восемнадцать», и это одно и то же."""
    result: list[str] = []
    for token in _WORD.findall(pronounce(text).lower()):
        if token.isdigit() and int(token) < 100:
            result.extend(number_words(int(token)))
        else:
            result.append(token.replace("ё", "е"))
    return result


def number_words(value: int) -> list[str]:
    """Число 0–99 словами, в именительном падеже: для сверки достаточно."""
    from socmon.core.text import TEENS, TENS, UNITS

    if value == 0:
        return ["ноль"]
    if value == 10:
        return ["десять"]
    if value < 10:
        return [next(w for w, n in UNITS.items() if n == value)]
    if value < 20:
        return [next(w for w, n in TEENS.items() if n == value)]
    tens, units = divmod(value, 10)
    out = [next(w for w, n in TENS.items() if n == tens * 10)]
    if units:
        out.append(next(w for w, n in UNITS.items() if n == units))
    return out


def same_words(original: str, rendered: str) -> bool:
    """Разметка не имела права тронуть ни одного слова."""
    return words(original) == words(rendered)
