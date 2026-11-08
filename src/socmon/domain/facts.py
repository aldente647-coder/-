"""Мост между движком интервью и правилами валидности.

Отдельный модуль, потому что это ровно то место, где легко незаметно
смошенничать: «подкрутить» вывод факта из ответов — и массив поедет, а
никакого следа в правилах не останется. Здесь вывод один, он явный
и покрыт тестами.
"""

from __future__ import annotations

from socmon.core.codes import Outcome
from socmon.core.engine import InterviewSession
from socmon.core.questionnaire import Questionnaire
from socmon.domain.validity import InterviewFacts


def derive_facts(
    session: InterviewSession,
    *,
    is_test: bool = False,
    answering_machine: bool = False,
    synthetic_respondent: bool = False,
    respondent_substituted: bool = False,
    technical_error: bool = False,
    manual_rejection: str | None = None,
) -> InterviewFacts:
    """Собрать факты об интервью для проверки по ТЗ §5.3.

    Признаки автоответчика, синтетического собеседника и подмены респондента
    движок не выводит — их приносят детектор на линии и ночной постанализ.
    Поэтому они приходят параметрами, а не берутся из воздуха.
    """
    questionnaire = session.questionnaire
    codes = {node_id: answer.code for node_id, answer in session.answers.items()}

    return InterviewFacts(
        outcome=session.outcome or Outcome.PARTIAL,
        consented=session.consented,
        age_confirmed=_passed(questionnaire, codes, Outcome.INELIGIBLE_AGE),
        territory_confirmed=_passed(questionnaire, codes, Outcome.INELIGIBLE_TERRITORY),
        duration_sec=session.duration_sec,
        answers=codes,
        required_nodes=questionnaire.required_core_nodes,
        is_test=is_test,
        answering_machine=answering_machine,
        synthetic_respondent=synthetic_respondent,
        respondent_substituted=respondent_substituted,
        technical_error=technical_error,
        manual_rejection=manual_rejection,
    )


def _passed(
    questionnaire: Questionnaire, codes: dict[str, int | None], outcome: Outcome
) -> bool:
    """Прошёл ли респондент фильтр, отбраковывающий по этому основанию.

    Молчание фильтром не считается: если вопрос не задан или ответ не
    распознан, подтверждения нет. Признать «наверное, совершеннолетний»
    значило бы протащить в массив непроверенного респондента.
    """
    node = questionnaire.screener_for(outcome)
    if node is None:
        return True
    answer = codes.get(node.id)
    return answer is not None and answer in node.pass_codes
