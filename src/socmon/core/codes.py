"""Коды ответов и исходов.

Разделение на три независимые оси (стадия / техническая причина обрыва / исход)
взято из блюпринта VOIF: ТЗ §6.3 сваливает их в один плоский список, из-за чего
достижимость по AAPOR приходится считать вручную. Здесь оси разведены, а исход
интервью явно смаплен на AAPOR final disposition codes.
"""

from __future__ import annotations

from enum import IntEnum, StrEnum

# --------------------------------------------------------------------------
# Специальные коды ответа на отдельный вопрос
# --------------------------------------------------------------------------


class SpecialCode(IntEnum):
    """Зарезервированные коды. Обычные варианты ответа нумеруются с 1.

    ТЗ §5.3 требует, чтобы обязательные вопросы имели либо допустимый ответ,
    либо *корректно зарегистрированный* код отказа или затруднения. Поэтому
    отказ, затруднение и нераспознанная речь — три разных кода, а не один.
    """

    NOT_ASKED = 95
    """Вопрос не задавался: пропущен по логике анкеты."""

    UNRECOGNIZED = 96
    """Речь не распознана после исчерпания попыток (ТЗ §5.3, §10.1)."""

    DONT_KNOW = 97
    """«Затрудняюсь ответить» — обязательный вариант для чувствительных
    вопросов (ТЗ §7.6)."""

    REFUSED = 98
    """Отказ отвечать на конкретный вопрос. Не является основанием
    для браковки интервью (ТЗ §7.6)."""

    NO_ANSWER = 99
    """Ответа нет: интервью прервано до того, как вопрос был отвечен."""


#: Коды, которые не являются содержательным ответом.
NON_SUBSTANTIVE = frozenset(SpecialCode)

#: Максимальный код обычного варианта ответа. Всё выше — служебная зона.
MAX_OPTION_CODE = 94


# --------------------------------------------------------------------------
# Ось 1: стадия звонка (что происходит прямо сейчас)
# --------------------------------------------------------------------------


class CallStage(StrEnum):
    """Единственное финальное состояние — FINISHED; что именно произошло,
    объясняют две другие оси."""

    DIALING = "dialing"
    RINGING = "ringing"
    IN_INTERVIEW = "in_interview"
    TRANSFERRING = "transferring"
    WITH_OPERATOR = "with_operator"
    FINISHED = "finished"


# --------------------------------------------------------------------------
# Ось 2: техническая причина обрыва (пусто, если связь отработала штатно)
# --------------------------------------------------------------------------


class Disposition(StrEnum):
    NO_ANSWER = "no_answer"
    BUSY = "busy"
    INVALID_NUMBER = "invalid_number"
    ANSWERING_MACHINE = "answering_machine"
    TECH_ERROR = "tech_error"
    NO_OPERATOR = "no_operator"
    TIMEOUT = "timeout"
    DROPPED = "dropped"


# --------------------------------------------------------------------------
# Ось 3: исход интервью
# --------------------------------------------------------------------------


class Outcome(StrEnum):
    COMPLETED = "completed"
    """Интервью завершено по утверждённой логике анкеты (ТЗ §5.3)."""

    PARTIAL = "partial"
    """Начато и прервано после согласия, но до финального узла."""

    NO_CONSENT = "no_consent"
    """Респондент не дал согласия на вступлении (ТЗ §8.4)."""

    REFUSAL_BEFORE = "refusal_before"
    REFUSAL_DURING = "refusal_during"
    CALLBACK_REQUEST = "callback_request"
    DONT_CALL_REQUEST = "dont_call_request"
    """Явная просьба не звонить: немедленно в стоп-лист (ТЗ §6.2)."""

    TRANSFERRED_TO_OPERATOR = "transferred_to_operator"
    INELIGIBLE_AGE = "ineligible_age"
    INELIGIBLE_TERRITORY = "ineligible_territory"
    QUOTA_CLOSED = "quota_closed"
    SPEECH_UNUSABLE = "speech_unusable"
    """Систематически нераспознанная речь — интервью непригодно (ТЗ §5.3)."""

    NOT_RESPONDENT = "not_respondent"
    """Соединение без участия респондента / подмена собеседника."""


# --------------------------------------------------------------------------
# Маппинг на AAPOR Standard Definitions
# --------------------------------------------------------------------------


class AaporCode(StrEnum):
    """Final disposition codes. Группа определяется первой цифрой:
    1 — интервью, 2 — контакт был, интервью нет, 3 — пригодность неизвестна,
    4 — непригоден для опроса."""

    COMPLETE = "1.1"
    PARTIAL = "1.2"
    REFUSAL = "2.11"
    BREAK_OFF = "2.12"
    NON_CONTACT = "2.20"
    LANGUAGE_BARRIER = "2.33"
    OTHER_ELIGIBLE = "2.30"
    UNKNOWN_ELIGIBILITY = "3.20"
    NOT_ELIGIBLE = "4.70"
    OUT_OF_SAMPLE = "4.10"


_OUTCOME_TO_AAPOR: dict[Outcome, AaporCode] = {
    Outcome.COMPLETED: AaporCode.COMPLETE,
    Outcome.PARTIAL: AaporCode.PARTIAL,
    Outcome.NO_CONSENT: AaporCode.REFUSAL,
    Outcome.REFUSAL_BEFORE: AaporCode.REFUSAL,
    Outcome.REFUSAL_DURING: AaporCode.BREAK_OFF,
    Outcome.DONT_CALL_REQUEST: AaporCode.REFUSAL,
    Outcome.CALLBACK_REQUEST: AaporCode.OTHER_ELIGIBLE,
    Outcome.TRANSFERRED_TO_OPERATOR: AaporCode.OTHER_ELIGIBLE,
    Outcome.NOT_RESPONDENT: AaporCode.OTHER_ELIGIBLE,
    Outcome.SPEECH_UNUSABLE: AaporCode.LANGUAGE_BARRIER,
    Outcome.INELIGIBLE_AGE: AaporCode.NOT_ELIGIBLE,
    Outcome.INELIGIBLE_TERRITORY: AaporCode.NOT_ELIGIBLE,
    Outcome.QUOTA_CLOSED: AaporCode.OUT_OF_SAMPLE,
}

_DISPOSITION_TO_AAPOR: dict[Disposition, AaporCode] = {
    # Пригодность абонента неизвестна: мы так и не узнали, кто там и подходит ли он.
    Disposition.NO_ANSWER: AaporCode.UNKNOWN_ELIGIBILITY,
    Disposition.BUSY: AaporCode.UNKNOWN_ELIGIBILITY,
    Disposition.ANSWERING_MACHINE: AaporCode.UNKNOWN_ELIGIBILITY,
    Disposition.TIMEOUT: AaporCode.UNKNOWN_ELIGIBILITY,
    Disposition.TECH_ERROR: AaporCode.UNKNOWN_ELIGIBILITY,
    Disposition.NO_OPERATOR: AaporCode.UNKNOWN_ELIGIBILITY,
    Disposition.DROPPED: AaporCode.NON_CONTACT,
    Disposition.INVALID_NUMBER: AaporCode.OUT_OF_SAMPLE,
}


def to_aapor(outcome: Outcome | None, disposition: Disposition | None) -> AaporCode:
    """Свести две оси к одному коду AAPOR.

    Исход интервью приоритетнее технической причины: если разговор состоялся
    и завершился отказом, звонок нельзя считать недозвоном, даже если связь
    потом оборвалась.
    """
    if outcome is not None:
        return _OUTCOME_TO_AAPOR[outcome]
    if disposition is not None:
        return _DISPOSITION_TO_AAPOR[disposition]
    return AaporCode.UNKNOWN_ELIGIBILITY
