from __future__ import annotations

import pytest

from socmon.core.codes import Outcome, SpecialCode
from socmon.domain.validity import (
    InterviewFacts,
    Reason,
    Validity,
    ValidityPolicy,
    evaluate,
)

CORE = frozenset({"q_life", "q_material", "q_anxiety", "q_trust"})


def facts(**overrides: object) -> InterviewFacts:
    base: dict[str, object] = {
        "outcome": Outcome.COMPLETED,
        "consented": True,
        "age_confirmed": True,
        "territory_confirmed": True,
        "duration_sec": 420.0,
        "answers": {"q_life": 7, "q_material": 4, "q_anxiety": 8, "q_trust": 3},
        "required_nodes": CORE,
    }
    base.update(overrides)
    return InterviewFacts(**base)  # type: ignore[arg-type]


def test_clean_interview_is_valid() -> None:
    assert evaluate(facts()).is_valid


def test_refusal_on_a_single_question_does_not_reject_the_interview() -> None:
    """ТЗ §7.6 прямо: отказ от ответа на отдельный чувствительный вопрос
    не является основанием для браковки."""
    verdict = evaluate(facts(answers={
        "q_life": 7, "q_material": SpecialCode.REFUSED,
        "q_anxiety": 8, "q_trust": SpecialCode.DONT_KNOW,
    }))
    assert verdict.is_valid


def test_missing_core_answer_rejects() -> None:
    verdict = evaluate(facts(answers={"q_life": 7, "q_material": 4}))
    assert verdict.status is Validity.INVALID
    assert Reason.CORE_ANSWER_MISSING in verdict.reasons
    assert "q_anxiety" in verdict.details[0]


def test_not_asked_counts_as_missing() -> None:
    """«Не задавался» — это дырка в ядре, а не зарегистрированный отказ."""
    verdict = evaluate(facts(answers={
        "q_life": 7, "q_material": 4, "q_anxiety": 8,
        "q_trust": SpecialCode.NOT_ASKED,
    }))
    assert Reason.CORE_ANSWER_MISSING in verdict.reasons


def test_systematic_unrecognized_speech_rejects() -> None:
    verdict = evaluate(facts(answers={
        "q_life": SpecialCode.UNRECOGNIZED, "q_material": SpecialCode.UNRECOGNIZED,
        "q_anxiety": SpecialCode.UNRECOGNIZED, "q_trust": 3,
    }))
    assert Reason.SPEECH_UNUSABLE in verdict.reasons


def test_a_couple_of_unrecognized_answers_are_tolerated() -> None:
    verdict = evaluate(facts(answers={
        "q_life": SpecialCode.UNRECOGNIZED, "q_material": SpecialCode.UNRECOGNIZED,
        "q_anxiety": 8, "q_trust": 3,
    }))
    assert verdict.is_valid


def test_test_call_is_excluded_and_short_circuits() -> None:
    """Контрольный звонок в массив не попадает никогда — и разбирать его
    по остальным критериям бессмысленно."""
    verdict = evaluate(facts(is_test=True, consented=False, outcome=Outcome.PARTIAL))
    assert verdict.reasons == (Reason.TEST_CALL,)


def test_duration_threshold_is_absent_until_the_pilot_sets_it() -> None:
    """ТЗ §5.3: порог утверждается по результатам пилота. Пока его нет,
    придумывать нельзя — критерий не применяется."""
    assert evaluate(facts(duration_sec=5.0)).is_valid


def test_duration_threshold_applies_once_configured() -> None:
    verdict = evaluate(facts(duration_sec=45.0), ValidityPolicy(min_duration_sec=120.0))
    assert Reason.TOO_SHORT in verdict.reasons
    assert "45 с при пороге 120 с" in verdict.details[0]


@pytest.mark.parametrize(
    ("flag", "reason"),
    [
        ("answering_machine", Reason.ANSWERING_MACHINE),
        ("synthetic_respondent", Reason.SYNTHETIC_RESPONDENT),
        ("respondent_substituted", Reason.RESPONDENT_SUBSTITUTED),
        ("technical_error", Reason.TECHNICAL_ERROR),
    ],
)
def test_integrity_flags_reject(flag: str, reason: Reason) -> None:
    assert reason in evaluate(facts(**{flag: True})).reasons


def test_manual_rejection_carries_its_justification() -> None:
    """Ручная браковка без обоснования — это скрытая правка массива (ТЗ §11.3)."""
    verdict = evaluate(facts(manual_rejection="прослушано: отвечал не респондент"))
    assert Reason.MANUAL_REJECTION in verdict.reasons
    assert "прослушано" in verdict.details[0]


def test_all_reasons_are_collected_not_just_the_first() -> None:
    """Отчёт об исключённых интервью (ТЗ §13.2) полезнее, когда видно
    все основания сразу."""
    verdict = evaluate(facts(
        consented=False, age_confirmed=False, outcome=Outcome.PARTIAL, answers={},
    ))
    assert {
        Reason.NO_CONSENT, Reason.AGE_NOT_CONFIRMED,
        Reason.NOT_COMPLETED, Reason.CORE_ANSWER_MISSING,
    } <= set(verdict.reasons)


def test_partial_interview_is_not_valid() -> None:
    assert Reason.NOT_COMPLETED in evaluate(facts(outcome=Outcome.PARTIAL)).reasons
