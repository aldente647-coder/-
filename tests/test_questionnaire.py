from __future__ import annotations

import copy
from typing import Any

import pytest
import yaml

from socmon.core.questionnaire import Questionnaire, QuestionnaireError
from tests.conftest import REFERENCE


def _raw() -> dict[str, Any]:
    """Сырой YAML для порчи в негативных тестах. Тип намеренно Any: смысл
    этих тестов — подсунуть валидатору заведомо неправильную структуру."""
    loaded = yaml.safe_load(REFERENCE.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def test_reference_loads(core: Questionnaire) -> None:
    assert core.entry_id == "intro"
    assert len(core.nodes) > 20


def test_version_id_is_content_addressed(core: Questionnaire) -> None:
    """ТЗ §7.3: любая правка формулировки обязана давать новую версию."""
    same = Questionnaire.from_yaml(REFERENCE)
    assert core.version_id == same.version_id

    raw = _raw()
    raw["nodes"][0]["text"] += " "  # один-единственный пробел
    assert Questionnaire.parse(raw).version_id != core.version_id


def test_core_keys_cover_index_components(core: Questionnaire) -> None:
    """Компоненты индекса ТЗ §12 должны иметь стабильные ключи ядра,
    иначе динамический ряд между волнами не построить."""
    expected = {
        "life_situation", "material_stability", "anxiety",
        "fairness", "institutional_trust", "perceived_conflict", "collective_action",
    }
    assert expected <= set(core.core_keys)


def test_rejects_dangling_transition() -> None:
    raw = _raw()
    raw["nodes"][0]["next"] = "nowhere"
    with pytest.raises(QuestionnaireError, match="несуществующий узел"):
        Questionnaire.parse(raw)


def test_rejects_unreachable_node() -> None:
    raw = _raw()
    orphan = copy.deepcopy(raw["nodes"][5])
    orphan["id"] = "orphan"
    raw["nodes"].append(orphan)
    with pytest.raises(QuestionnaireError, match="недостижимые узлы"):
        Questionnaire.parse(raw)


def test_rejects_node_that_cannot_reach_an_end() -> None:
    """Узел без выхода к завершению — это звонок, повисший на живом человеке."""
    raw = _raw()
    nodes = raw["nodes"]
    trap = {
        "id": "trap", "kind": "info", "section": "core",
        "text": "тупик", "next": "trap2",
    }
    trap2 = {
        "id": "trap2", "kind": "info", "section": "core",
        "text": "тупик", "next": "trap",
    }
    # Ветка-тупик подвешивается как маршрут одного из вариантов ответа,
    # чтобы основной путь остался цел и первым сработала именно проверка
    # достижимости завершения, а не достижимости узлов.
    for node in nodes:
        if node["id"] == "q_local_problem":
            node["routes"] = {5: "trap"}
    nodes.extend([trap, trap2])
    with pytest.raises(QuestionnaireError, match="не достичь завершения"):
        Questionnaire.parse(raw)


def test_rejects_missing_terminal() -> None:
    raw = _raw()
    del raw["terminals"]["dont_call_request"]
    with pytest.raises(QuestionnaireError, match="не заданы терминалы"):
        Questionnaire.parse(raw)


def test_terminal_outcome_must_agree_with_node() -> None:
    raw = _raw()
    raw["terminals"]["dont_call_request"] = "end_callback"
    with pytest.raises(QuestionnaireError, match="объявляет исход"):
        Questionnaire.parse(raw)


def test_sensitive_question_must_allow_dont_know() -> None:
    """ТЗ §7.6: на чувствительный вопрос всегда можно не отвечать."""
    raw = _raw()
    for node in raw["nodes"]:
        if node["id"] == "d_nationality":
            node["allow_dont_know"] = False
    with pytest.raises(QuestionnaireError, match="затрудняюсь"):
        Questionnaire.parse(raw)


def test_rejects_duplicate_option_codes() -> None:
    raw = _raw()
    for node in raw["nodes"]:
        if node["id"] == "q_family_change":
            node["options"][1]["code"] = 1
    with pytest.raises(QuestionnaireError, match="не уникальны"):
        Questionnaire.parse(raw)


def test_rejects_scale_overlapping_service_codes() -> None:
    """Верх шкалы не должен заезжать в служебную зону: иначе «95» окажется
    одновременно оценкой и кодом «вопрос не задавался»."""
    raw = _raw()
    for node in raw["nodes"]:
        if node["id"] == "q_life":
            node["high"] = 100
    with pytest.raises(QuestionnaireError, match="служебную зону"):
        Questionnaire.parse(raw)


@pytest.mark.parametrize("key", ["../../outside", "/tmp/outside", "x\\y", "a'b", "x\n"])
def test_audio_key_cannot_escape_media_directory(core: Questionnaire, key: str) -> None:
    raw = core.model_dump(mode="json")
    raw["nodes"][0]["audio_key"] = key
    with pytest.raises(ValueError):
        Questionnaire.model_validate(raw)
