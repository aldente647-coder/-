from __future__ import annotations

from pathlib import Path

import pytest

from socmon.analysis.dataset import Record
from socmon.analysis.index import (
    DEFAULT_COMPONENTS,
    MONITORING_COMPONENTS,
    CategoricalComponent,
    IndexError_,
    MissingPolicy,
    ScaleComponent,
    compute,
    recipe_hash,
)
from socmon.core.codes import SpecialCode

CALM = {
    "life_situation": 10, "material_stability": 10, "anxiety": 0,
    "fairness": 10, "institutional_trust": 10, "perceived_conflict": 0,
    "collective_action": 4,
}
TENSE = {
    "life_situation": 0, "material_stability": 0, "anxiety": 10,
    "fairness": 0, "institutional_trust": 0, "perceived_conflict": 10,
    "collective_action": 1,
}


def people(answers: dict[str, int], n: int = 200, **strata: str) -> list[Record]:
    return [Record(id=f"{id(answers)}-{i}", answers=dict(answers), strata=strata)
            for i in range(n)]


# ------------------------------------------------------------ шкала и полюса


def test_calm_population_scores_zero() -> None:
    assert compute(people(CALM)).aggregate == pytest.approx(0.0)


def test_tense_population_scores_ten() -> None:
    assert compute(people(TENSE)).aggregate == pytest.approx(10.0)


def test_index_is_the_mean_of_seven_components() -> None:
    """Формула ТЗ §12 буквально: I = (E+F+A+J+D+C+P)/7."""
    result = compute(people(CALM, 100) + people(TENSE, 100))
    assert result.aggregate == pytest.approx(
        sum(c.value for c in result.components) / 7
    )
    assert len(result.components) == 7


def test_orientation_is_uniform_higher_means_more_tension() -> None:
    """Самая дорогая ошибка модуля: перепутанное направление шкалы даёт
    правдоподобный индекс, поехавший в противоположную сторону."""
    satisfied = compute(people({**TENSE, "life_situation": 10}))
    dissatisfied = compute(people({**TENSE, "life_situation": 0}))
    assert dissatisfied.profile["E"] > satisfied.profile["E"]

    trusting = compute(people({**TENSE, "institutional_trust": 10}))
    assert compute(people(TENSE)).profile["D"] > trusting.profile["D"]

    anxious = compute(people({**CALM, "anxiety": 10}))
    assert anxious.profile["A"] > compute(people(CALM)).profile["A"]


def test_component_profile_is_reported_alongside_the_score() -> None:
    """ТЗ §12: предоставлять компонентный профиль, а не только единый балл."""
    profile = compute(people(CALM, 100) + people(TENSE, 100)).profile
    assert set(profile) == {"E", "F", "A", "J", "D", "C", "P"}


def test_categorical_component_uses_explicit_scores() -> None:
    ready = compute(people({**CALM, "collective_action": 1}))
    unready = compute(people({**CALM, "collective_action": 4}))
    assert ready.profile["P"] == pytest.approx(10.0)
    assert unready.profile["P"] == pytest.approx(0.0)


# ------------------------------------------------------------------ пропуски


def test_refusals_are_excluded_by_default() -> None:
    """ТЗ §12: не заменять пропуски средними значениями."""
    mixed = people({**TENSE, "anxiety": int(SpecialCode.REFUSED)}, 100) + people(TENSE, 100)
    result = compute(mixed)
    assert result.profile["A"] == pytest.approx(10.0)
    anxiety = next(c for c in result.components if c.key == "A")
    assert anxiety.missing_share == pytest.approx(0.5)


def test_neutral_substitution_requires_named_approval() -> None:
    with pytest.raises(IndexError_, match="явного согласования"):
        compute(people(TENSE), missing_policy=MissingPolicy.NEUTRAL)


def test_neutral_substitution_works_once_approved() -> None:
    mixed = people({**TENSE, "anxiety": int(SpecialCode.DONT_KNOW)}, 100)
    result = compute(
        mixed,
        missing_policy=MissingPolicy.NEUTRAL,
        neutral_substitution_approved_by="методолог Заказчика, протокол от 20.08.2026",
    )
    assert result.profile["A"] == pytest.approx(5.0)


def test_high_refusal_rate_on_a_component_is_flagged() -> None:
    mixed = people({**TENSE, "fairness": int(SpecialCode.REFUSED)}, 60) + people(TENSE, 140)
    result = compute(mixed)
    assert any("проверить формулировку вопроса" in n for n in result.notes)


def test_component_without_any_answer_is_an_error() -> None:
    blind = people({**TENSE, "anxiety": int(SpecialCode.UNRECOGNIZED)})
    with pytest.raises(IndexError_, match="посчитать нельзя"):
        compute(blind)


# ------------------------------------------- агрегатный и индивидуальный индекс


def test_individual_index_carries_a_confidence_interval() -> None:
    result = compute(people(CALM, 100) + people(TENSE, 100))
    assert result.individual is not None
    assert result.individual.ci_low < result.individual.value < result.individual.ci_high
    assert result.complete_cases == 200


def test_partial_answers_do_not_enter_the_individual_index() -> None:
    """У респондента, ответившего на три компонента из семи, среднее было бы
    по другой шкале — такие наблюдения в индивидуальный индекс не идут."""
    partial = people({**TENSE, "anxiety": int(SpecialCode.REFUSED)}, 50)
    result = compute(partial + people(TENSE, 150))
    assert result.complete_cases == 150
    assert result.n == 200


def test_divergence_between_the_two_indices_is_flagged() -> None:
    """Расхождение означает, что отказы распределены неслучайно —
    это ограничение интерпретации, а не мелочь."""
    calm_but_silent = people({**CALM, "anxiety": int(SpecialCode.REFUSED)}, 100)
    result = compute(calm_but_silent + people(TENSE, 100))
    assert result.individual is not None
    assert any("расходятся" in n for n in result.notes)


def test_weights_move_the_index() -> None:
    tense = [Record(id=f"t{i}", answers=dict(TENSE), weight=1.8) for i in range(100)]
    calm = [Record(id=f"c{i}", answers=dict(CALM), weight=0.2) for i in range(100)]
    assert compute(tense + calm).aggregate > 8.0


# ------------------------------------------------------------ воспроизводимость


def test_recipe_hash_pins_the_methodology() -> None:
    """ТЗ §12: обеспечить воспроизводимость расчёта в выгрузке."""
    baseline = recipe_hash(DEFAULT_COMPONENTS, MissingPolicy.EXCLUDE)
    assert baseline == recipe_hash(DEFAULT_COMPONENTS, MissingPolicy.EXCLUDE)
    assert baseline != recipe_hash(DEFAULT_COMPONENTS, MissingPolicy.NEUTRAL)

    flipped = tuple(
        ScaleComponent(c.key, c.name, c.core_key, c.low, c.high, invert=not c.invert)
        if isinstance(c, ScaleComponent) else c
        for c in DEFAULT_COMPONENTS
    )
    assert baseline != recipe_hash(flipped, MissingPolicy.EXCLUDE)


def test_result_carries_its_own_recipe() -> None:
    result = compute(people(TENSE))
    assert result.recipe_hash == recipe_hash(DEFAULT_COMPONENTS, MissingPolicy.EXCLUDE)


# ----------------------------------------------------------------- валидация


def test_score_outside_the_index_scale_is_refused() -> None:
    with pytest.raises(IndexError_, match="вне шкалы 0-10"):
        CategoricalComponent("X", "Плохой", "k", scores={1: 42.0})


def test_service_code_cannot_be_a_variant_score() -> None:
    """Код 97 — «затрудняюсь», а не вариант ответа."""
    with pytest.raises(IndexError_, match="вне зоны вариантов"):
        CategoricalComponent("X", "Плохой", "k", scores={97: 5.0})


def test_code_outside_the_source_scale_is_refused() -> None:
    with pytest.raises(IndexError_, match="вне шкалы"):
        ScaleComponent("A", "Тревожность", "anxiety").normalize(11)


def test_empty_dataset_is_refused() -> None:
    with pytest.raises(IndexError_, match="пустой массив"):
        compute([])


# ------------------------------------------- индекс на анкете Заказчика

def test_monitoring_components_span_the_whole_scale() -> None:
    """Компоненты анкеты Заказчика дают полный размах 0-10 на полюсах.

    Проверка не формальная: у категориальных компонентов баллы вариантов
    назначаются руками, и перепутанное направление одного из них здесь же
    и всплывёт — индекс поедет в другую сторону, оставшись правдоподобным.
    """
    calm = {
        "region_satisfaction": 1, "material_status": 1, "anxiety_dynamics": 3,
        "fairness": 1, "trust_republic": 1, "safety": 1,
    }
    tense = {
        "region_satisfaction": 2, "material_status": 4, "anxiety_dynamics": 1,
        "fairness": 2, "trust_republic": 2, "safety": 2,
    }
    assert compute(people(calm), MONITORING_COMPONENTS).aggregate == pytest.approx(0.0)
    assert compute(people(tense), MONITORING_COMPONENTS).aggregate == pytest.approx(10.0)


def test_monitoring_index_is_six_components_by_agreement() -> None:
    """Формула из шести слагаемых согласована с Заказчиком.

    ТЗ §12 приводит семь, включая готовность к коллективным действиям,
    но такого вопроса в анкете нет. Состав закреплён тестом, чтобы отличие
    от формулы ТЗ оставалось видимым решением, а не случайностью, и чтобы
    седьмой компонент нельзя было дописать молча.
    """
    keys = [c.key for c in MONITORING_COMPONENTS]
    assert keys == ["E", "F", "A", "J", "D", "C"]
    assert "P" not in keys


def test_monitoring_recipe_differs_from_the_reference() -> None:
    """Методика другая — значит, и хеш методики другой."""
    assert recipe_hash(MONITORING_COMPONENTS, MissingPolicy.EXCLUDE) != recipe_hash(
        DEFAULT_COMPONENTS, MissingPolicy.EXCLUDE
    )


def test_every_monitoring_component_comes_from_the_core() -> None:
    """Компонент индекса не может стоять в переменном блоке.

    Переменный блок Заказчик вправе менять от волны к волне. Вопрос,
    от которого зависит индекс, оттуда однажды исчезнет — и ряд оборвётся
    без единой ошибки в логах. Ради этого справедливость и перенесена
    в постоянное ядро.
    """
    from socmon.core.questionnaire import Questionnaire

    path = Path("src/socmon/questionnaires/wave-base.yaml")
    section_of = {
        n.core_key: n.section
        for n in Questionnaire.from_yaml(path).nodes
        if n.core_key
    }
    for component in MONITORING_COMPONENTS:
        assert section_of[component.core_key] == "core", (
            f"компонент {component.key} снимается вопросом "
            f"вне постоянного ядра"
        )
