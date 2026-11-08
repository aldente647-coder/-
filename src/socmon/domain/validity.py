"""Критерии валидного интервью (ТЗ §5.3).

Решение о включении интервью в аналитический массив принимается правилами, а
не человеком и не моделью. Это принципиально: §5.1 прямо запрещает отбирать
интервью под желаемый результат, поэтому критерии должны быть выписаны один
раз, применяться ко всем одинаково и давать перечень оснований исключения для
отчёта (§13.2).

Здесь же — граница с блюпринтом VOIF. Там итог разговора ставит оператор
вручную, и это верно для продаж. У нас оператора в массовом обзвоне нет, а
«результат» — это сама анкета, поэтому валидность вычисляется.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from socmon.core.codes import Outcome, SpecialCode


class Validity(StrEnum):
    VALID = "valid"
    INVALID = "invalid"


class Reason(StrEnum):
    """Основания исключения. Идут в выгрузку как есть (ТЗ §13.2)."""

    TEST_CALL = "test_call"
    NOT_COMPLETED = "not_completed"
    NO_CONSENT = "no_consent"
    AGE_NOT_CONFIRMED = "age_not_confirmed"
    TERRITORY_NOT_CONFIRMED = "territory_not_confirmed"
    CORE_ANSWER_MISSING = "core_answer_missing"
    SPEECH_UNUSABLE = "speech_unusable"
    TOO_SHORT = "too_short"
    ANSWERING_MACHINE = "answering_machine"
    SYNTHETIC_RESPONDENT = "synthetic_respondent"
    RESPONDENT_SUBSTITUTED = "respondent_substituted"
    TECHNICAL_ERROR = "technical_error"
    MANUAL_REJECTION = "manual_rejection"


@dataclass(frozen=True, slots=True)
class ValidityPolicy:
    """Пороги контроля качества."""

    min_duration_sec: float | None = None
    """ТЗ §5.3: «длительность интервью не ниже порога, утверждённого по
    результатам пилота». До пилота порога нет, и выдумывать его нельзя —
    None означает, что критерий не применяется, и это фиксируется явно."""

    max_unrecognized_core: int = 2
    """Сколько ключевых вопросов может уйти в «нераспознано», прежде чем
    интервью перестаёт поддаваться интерпретации (ТЗ §5.3)."""


#: Пороги по умолчанию: порога длительности до пилота нет (ТЗ §5.3).
DEFAULT_POLICY = ValidityPolicy()


@dataclass(frozen=True, slots=True)
class InterviewFacts:
    """Всё, что нужно знать об интервью, чтобы вынести решение.

    Плоская структура вместо ссылки на ORM-объект: правила валидности должны
    проверяться без базы, а состав входных данных — быть виден целиком.
    """

    outcome: Outcome
    consented: bool
    age_confirmed: bool
    territory_confirmed: bool
    duration_sec: float
    answers: Mapping[str, int | None]
    required_nodes: frozenset[str]
    is_test: bool = False
    answering_machine: bool = False
    synthetic_respondent: bool = False
    respondent_substituted: bool = False
    technical_error: bool = False
    manual_rejection: str | None = None
    """Ручная браковка по методике контроля качества (ТЗ §5.3, §11.3).
    Хранится вместе с обоснованием, иначе это скрытая правка массива."""


@dataclass(frozen=True, slots=True)
class Verdict:
    status: Validity
    reasons: tuple[Reason, ...] = ()
    details: tuple[str, ...] = field(default=())

    @property
    def is_valid(self) -> bool:
        return self.status is Validity.VALID


def evaluate(facts: InterviewFacts, policy: ValidityPolicy = DEFAULT_POLICY) -> Verdict:
    """Проверить интервью по всем критериям ТЗ §5.3.

    Проверки не прерываются на первом провале: отчёт об исключённых интервью
    полезнее, когда видно все основания сразу, а не первое попавшееся.
    """
    reasons: list[Reason] = []
    details: list[str] = []

    if facts.is_test:
        # Контрольный звонок проходит весь путь как настоящий, но в массив
        # и в статистику не попадает никогда (приём из блюпринта VOIF).
        return Verdict(Validity.INVALID, (Reason.TEST_CALL,), ("контрольный звонок",))

    if not facts.consented:
        reasons.append(Reason.NO_CONSENT)
    if not facts.age_confirmed:
        reasons.append(Reason.AGE_NOT_CONFIRMED)
    if not facts.territory_confirmed:
        reasons.append(Reason.TERRITORY_NOT_CONFIRMED)
    if facts.outcome is not Outcome.COMPLETED:
        reasons.append(Reason.NOT_COMPLETED)
        details.append(f"исход: {facts.outcome.value}")

    missing = _missing_core(facts)
    if missing:
        reasons.append(Reason.CORE_ANSWER_MISSING)
        details.append("нет отметки по узлам: " + ", ".join(missing))

    unrecognized = _unrecognized_core(facts)
    if len(unrecognized) > policy.max_unrecognized_core:
        reasons.append(Reason.SPEECH_UNUSABLE)
        details.append(f"нераспознано ключевых ответов: {len(unrecognized)}")

    if policy.min_duration_sec is not None and facts.duration_sec < policy.min_duration_sec:
        reasons.append(Reason.TOO_SHORT)
        details.append(f"{facts.duration_sec:.0f} с при пороге {policy.min_duration_sec:.0f} с")

    if facts.answering_machine:
        reasons.append(Reason.ANSWERING_MACHINE)
    if facts.synthetic_respondent:
        reasons.append(Reason.SYNTHETIC_RESPONDENT)
    if facts.respondent_substituted:
        reasons.append(Reason.RESPONDENT_SUBSTITUTED)
    if facts.technical_error:
        reasons.append(Reason.TECHNICAL_ERROR)
    if facts.manual_rejection:
        reasons.append(Reason.MANUAL_REJECTION)
        details.append(facts.manual_rejection)

    if reasons:
        return Verdict(Validity.INVALID, tuple(reasons), tuple(details))
    return Verdict(Validity.VALID)


def _missing_core(facts: InterviewFacts) -> list[str]:
    """Ключевые вопросы без всякой отметки.

    Отказ и затруднение отметкой являются: §5.3 требует «допустимый ответ
    ЛИБО корректно зарегистрированный код отказа/затруднения», а §7.6 прямо
    говорит, что отказ от чувствительного вопроса интервью не бракует.
    Не хватает только того, чего в массиве нет вовсе или что помечено
    как «не задавался».
    """
    return [
        node_id
        for node_id in sorted(facts.required_nodes)
        if facts.answers.get(node_id, SpecialCode.NOT_ASKED) == SpecialCode.NOT_ASKED
    ]


def _unrecognized_core(facts: InterviewFacts) -> list[str]:
    return [
        node_id
        for node_id in sorted(facts.required_nodes)
        if facts.answers.get(node_id) == SpecialCode.UNRECOGNIZED
    ]
