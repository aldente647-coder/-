"""Подписи для кабинета.

Внутри системы состояния — короткие английские коды: они попадают в базу,
в выгрузки и в кодбук, и менять их нельзя. Но человеку на экране нужен
русский: «draft» и «valid» понятны разработчику, а не методологу
и не Заказчику.

Перевод живёт здесь одним списком, а не разбросан по шаблонам: иначе
одно и то же состояние однажды назовут по-разному на двух страницах.
"""

from __future__ import annotations

WAVE_STATUS: dict[str, str] = {
    "draft": "черновик",
    "active": "идёт",
    "paused": "пауза",
    "closed": "закрыта",
}

VALIDITY: dict[str, str] = {
    "valid": "валидно",
    "invalid": "забраковано",
    "pending": "не проверено",
}

OUTCOME: dict[str, str] = {
    "completed": "завершено",
    "partial": "прервано",
    "no_consent": "нет согласия",
    "refusal_before": "отказ до начала",
    "refusal_during": "отказ в ходе",
    "callback_request": "просил перезвонить",
    "dont_call_request": "просил не звонить",
    "transferred_to_operator": "передано оператору",
    "ineligible_age": "младше 18",
    "ineligible_territory": "другая территория",
    "quota_closed": "квота закрыта",
    "speech_unusable": "речь не разобрать",
    "not_respondent": "не тот собеседник",
}

DISPOSITION: dict[str, str] = {
    "no_answer": "не ответили",
    "busy": "занято",
    "invalid_number": "неверный номер",
    "answering_machine": "автоответчик",
    "tech_error": "сбой связи",
    "no_operator": "нет оператора",
    "timeout": "тайм-аут",
    "dropped": "обрыв",
}

STAGE: dict[str, str] = {
    "dialing": "набираем",
    "ringing": "гудки",
    "in_interview": "разговор",
    "transferring": "перевод",
    "with_operator": "у оператора",
    "finished": "завершён",
}

TONE: dict[str, str] = {
    "positive": "позитивная",
    "neutral": "нейтральная",
    "negative": "негативная",
}

_ALL: dict[str, str] = {
    **WAVE_STATUS, **VALIDITY, **OUTCOME, **DISPOSITION, **STAGE, **TONE
}


def ru(value: object) -> str:
    """Русская подпись для значения состояния.

    Незнакомое значение возвращается как есть: лучше показать код,
    чем пустоту, — по коду хотя бы понятно, что искать.
    """
    if value is None:
        return "—"
    key = getattr(value, "value", value)
    return _ALL.get(str(key), str(key))


def plural(count: int, one: str, few: str, many: str) -> str:
    """Число с правильной формой слова.

    «1 препятствий» на экране читается как недоделка, и справедливо.
    """
    tail100, tail10 = count % 100, count % 10
    if 11 <= tail100 <= 14:
        word = many
    elif tail10 == 1:
        word = one
    elif 2 <= tail10 <= 4:
        word = few
    else:
        word = many
    return f"{count} {word}"
