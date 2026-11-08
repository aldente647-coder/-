"""Анкета Заказчика и механизмы, которых она потребовала.

Файл проверяет три вещи разом: что все четыре волны собираются из модулей
и проходятся целиком, что вопрос «до трёх ответов» пишет в массив все
названные варианты, и что фильтр по предыдущему ответу действительно
пропускает вопрос, а не задаёт его молча.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from socmon.core.codes import Outcome, SpecialCode
from socmon.core.engine import Action, InterviewSession, Turn, Utterance
from socmon.core.grammar import OptionGrammar, match_multi
from socmon.core.questionnaire import Questionnaire, QuestionnaireError

QUESTIONNAIRES = Path(__file__).resolve().parents[1] / "src/socmon/questionnaires"
WAVES = ("wave-base", "wave-social", "wave-environment", "wave-governance")

#: Ответы, ведущие по «зелёному» маршруту до конца любой из волн.
SCRIPT: dict[str, str] = {
    "consent": "да, согласен",
    "screen_age": "да, исполнилось",
    "screen_territory": "живу постоянно",
    "q01_region": "не доволен",
    "q02_dynamics": "не изменилась",
    "q03a_federal": "доверяю",
    "q03b_republic": "доверяю",
    "q03c_local": "не доверяю",
    "q04_president": "хорошо",
    "q05_head": "хорошо",
    "q07_material": "хватает только на самое необходимое",
    "q08_safety": "скорее в безопасности",
    "q09_anxiety": "не изменились",
    "q10_health": "удовлетворительное",
    "q11_problems": "цены выросли и дороги разбитые",
    "q12_prices": "на продукты и на лекарства",
    "q13_normalization": "дороги починить",
    "q14_sources": "телевизор, телеграм и разговоры с людьми",
    "q15_trusted": "нет таких",
    "q16_svo": "положительно",
    "p01a_relatives": "по разному",
    "p01b_tv": "в основном положительные",
    "p01c_social": "в основном отрицательные",
    "p02_media_trust": "местные сми и телеграм",
    "p03_info_gaps": "про выплаты",
    "p04_tv_frequency": "реже",
    "p05_tv_less_reason": "много негатива",
    "q_fairness": "несправедливо",
    "d_gender": "мужской",
    "d_age": "тридцать два",
    "d_settlement": "Донецк, Киевский район",
    "d_education": "высшее",
    "d_svo_participant": "нет",
    "d_veteran": "нет",
    "d_svo_family": "да, брат",
    "d_employment": "работаю",
    "d_nationality": "русский",
    "d01_filter": "да, есть",
    "d01_preschool": "скорее не удовлетворен",
    "d02_filter": "нет",
    "d02_school": "скорее удовлетворен",
    "d03_college": "не сталкивался",
    "d04_university": "скорее удовлетворен",
    "d05_filter": "да, обращались",
    "d05_medicine": "совсем не удовлетворен",
    "d06_youth_region": "скорее плохие",
    "d07_youth_russia": "скорее хорошие",
    "d08_culture": "не пользуюсь",
    "d09_childbirth": "да, планирую",
    "d10_children_conditions": "скорее удовлетворен",
    "d11_amenities": "скорее не удовлетворен",
    "d12_housing": "скорее недоступно",
    "d13_roads": "совсем не удовлетворен",
    "d14_city_transport": "скорее удовлетворен",
    "d15_intercity_transport": "не езжу",
    "d16_ecology": "скорее плохая",
    "d17_waste": "скорее удовлетворен",
    "d18_utilities": "скорее не удовлетворен",
    "d25_sport": "не занимаюсь",
    "d19_officials": "не обращался",
    "d20_egov": "скорее удовлетворен",
    "d21_communications": "совсем не удовлетворен",
    "d22_tourism": "не путешествую",
    "d23_business": "скорее плохие",
    "d24_retraining": "не сталкивался",
    "d26_volunteering": "готов участвовать",
}


def drive(session: InterviewSession, **overrides: str) -> Turn:
    """Проиграть интервью целиком. Незаданный ответ — ошибка теста,
    а не повод молча остановиться."""
    script = dict(SCRIPT, **overrides)
    turn = session.begin()
    for _ in range(150):
        if turn.action not in (Action.ASK, Action.REPEAT):
            return turn
        reply = script.get(turn.awaiting)
        assert reply is not None, f"нет ответа для узла {turn.awaiting}"
        turn = session.submit(Utterance.said(reply))
    raise AssertionError("интервью не завершилось за 150 шагов")


@pytest.fixture(scope="module", params=WAVES)
def wave(request: pytest.FixtureRequest) -> Questionnaire:
    return Questionnaire.from_yaml(QUESTIONNAIRES / f"{request.param}.yaml")


# ------------------------------------------------------------ сборка волн


def test_every_wave_assembles_and_completes(wave: Questionnaire) -> None:
    """Все четыре волны собираются из модулей и проходятся до конца."""
    session = InterviewSession(questionnaire=wave)
    turn = drive(session)
    assert turn.outcome is Outcome.COMPLETED
    assert session.consented


def test_waves_share_the_same_core_wording() -> None:
    """Формулировки ядра совпадают во всех волнах посимвольно.

    Ради этого модульная сборка и заводилась: разошедшийся текст вопроса
    рвёт динамический ряд §3.1 незаметно — данные продолжают собираться,
    но сравнивать их уже нельзя.
    """
    texts: list[dict[str, str]] = []
    for name in WAVES:
        q = Questionnaire.from_yaml(QUESTIONNAIRES / f"{name}.yaml")
        texts.append({n.id: n.text for n in q.nodes if n.id.startswith(("q0", "q1", "p0"))})
    first = texts[0]
    assert first, "ядро не найдено — изменились идентификаторы узлов"
    for other in texts[1:]:
        assert other == first


def test_waves_differ_only_by_their_module() -> None:
    """Волны — разные анкеты: у каждой свой version_id."""
    versions = {
        name: Questionnaire.from_yaml(QUESTIONNAIRES / f"{name}.yaml").version_id
        for name in WAVES
    }
    assert len(set(versions.values())) == len(WAVES)


def test_module_reference_past_the_last_module_is_rejected(tmp_path: Path) -> None:
    """`@next` в последнем модуле — переход в никуда, и это ошибка загрузки,
    а не сюрприз в звонке."""
    (tmp_path / "only.yaml").write_text(
        yaml.safe_dump({"nodes": [
            {"id": "a", "kind": "info", "section": "intro", "text": "…", "next": "@next"},
        ]}, allow_unicode=True), encoding="utf-8")
    (tmp_path / "wave.yaml").write_text(
        yaml.safe_dump({"name": "n", "wave": "w", "modules": ["only.yaml"]},
                       allow_unicode=True), encoding="utf-8")
    with pytest.raises(QuestionnaireError, match="дальше идти некуда"):
        Questionnaire.from_yaml(tmp_path / "wave.yaml")


def test_modules_and_nodes_together_are_rejected(tmp_path: Path) -> None:
    (tmp_path / "wave.yaml").write_text(
        yaml.safe_dump({"name": "n", "wave": "w", "modules": ["x.yaml"], "nodes": []},
                       allow_unicode=True), encoding="utf-8")
    with pytest.raises(QuestionnaireError, match="оставьте что-то одно"):
        Questionnaire.from_yaml(tmp_path / "wave.yaml")


# ------------------------------------------------------- множественный выбор


def test_multi_records_every_named_option() -> None:
    """Вопрос «до трёх ответов» пишет все названные варианты, а не первый."""
    q = Questionnaire.from_yaml(QUESTIONNAIRES / "wave-base.yaml")
    session = InterviewSession(questionnaire=q)
    drive(session, q14_sources="телевизор, телеграм и разговоры с людьми")
    answer = session.answers["q14_sources"]
    assert answer.codes == (1, 6, 2)
    # Порядок — по месту в реплике: первым назван первый ответ.
    assert answer.code == 1


def test_multi_keeps_at_most_the_allowed_number() -> None:
    """«До трёх ответов» — ограничение методики, а не пожелание."""
    grammars = tuple(
        OptionGrammar.build(code, phrases)
        for code, phrases in ((1, ("телевидение",)), (2, ("радио",)),
                              (3, ("газеты",)), (4, ("интернет",)))
    )
    said = "телевидение радио газеты интернет"
    assert match_multi(said, grammars, 3) == (1, 2, 3)
    assert match_multi(said, grammars, 1) == (1,)


def test_multi_prefers_the_wider_match_on_the_same_words() -> None:
    """Один участок речи не должен засчитаться дважды."""
    grammars = (
        OptionGrammar.build(1, ("интернет",)),
        OptionGrammar.build(2, ("интернет новостные сайты",)),
    )
    assert match_multi("узнаю интернет новостные сайты", grammars, 3) == (2,)


def test_exclusive_option_cancels_the_rest() -> None:
    """«Никому не доверяю» вместе с названным источником — противоречие."""
    q = Questionnaire.from_yaml(QUESTIONNAIRES / "wave-base.yaml")
    session = InterviewSession(questionnaire=q)
    drive(session, p02_media_trust="местные сми, хотя вообще никому не доверяю")
    answer = session.answers["p02_media_trust"]
    assert answer.codes == (10,)
    assert any(e.kind == "exclusive_option" for e in session.events)


def test_multi_accepts_refusal_as_a_single_answer() -> None:
    q = Questionnaire.from_yaml(QUESTIONNAIRES / "wave-base.yaml")
    session = InterviewSession(questionnaire=q)
    drive(session, q14_sources="затрудняюсь ответить")
    assert session.answers["q14_sources"].code == SpecialCode.DONT_KNOW


# --------------------------------------------------------------- фильтры


def test_filter_skips_the_question_and_marks_it_not_asked() -> None:
    """Отфильтрованный вопрос не пропадает из массива: у него код 95.

    Пустая ячейка и ячейка «не задавался» — разные вещи. Смешать их
    значит посчитать долю от неверной базы.
    """
    q = Questionnaire.from_yaml(QUESTIONNAIRES / "wave-social.yaml")
    session = InterviewSession(questionnaire=q)
    drive(session, d01_filter="нет")
    assert session.answers["d01_preschool"].code == SpecialCode.NOT_ASKED
    assert session.answers["d01_preschool"].attempts == 0
    assert any(
        e.kind == "filtered_out" and e.node_id == "d01_preschool"
        for e in session.events
    )


def test_filter_lets_the_question_through_when_it_holds() -> None:
    q = Questionnaire.from_yaml(QUESTIONNAIRES / "wave-social.yaml")
    session = InterviewSession(questionnaire=q)
    drive(session, d01_filter="да, есть")
    assert session.answers["d01_preschool"].code == 3


def test_age_filter_uses_the_numeric_answer() -> None:
    """Планы на ребёнка спрашивают до сорока пяти. Семидесятилетнему
    такой вопрос не задаётся вовсе."""
    q = Questionnaire.from_yaml(QUESTIONNAIRES / "wave-social.yaml")

    young = InterviewSession(questionnaire=q)
    drive(young, d_age="тридцать два")
    assert young.answers["d09_childbirth"].code == 1

    old = InterviewSession(questionnaire=q)
    drive(old, d_age="семьдесят")
    assert old.answers["d09_childbirth"].code == SpecialCode.NOT_ASKED


def test_filter_does_not_hold_on_a_refusal() -> None:
    """Отказ назвать возраст — не основание считать, что он подходящий."""
    q = Questionnaire.from_yaml(QUESTIONNAIRES / "wave-social.yaml")
    session = InterviewSession(questionnaire=q)
    drive(session, d_age="не хочу отвечать")
    assert session.answers["d_age"].code == SpecialCode.REFUSED
    assert session.answers["d09_childbirth"].code == SpecialCode.NOT_ASKED


def test_filter_on_an_unknown_indicator_is_rejected(tmp_path: Path) -> None:
    body = {
        "name": "n", "wave": "w",
        "terminals": {
            "dont_call_request": "e", "callback_request": "e",
            "refusal_before": "e", "refusal_during": "e",
            "transferred_to_operator": "e", "speech_unusable": "e",
        },
        "nodes": [
            {"id": "q", "kind": "info", "section": "core", "text": "…", "next": "e",
             "ask_if": {"core_key": "нет_такого", "codes": [1]}},
            {"id": "e", "kind": "end", "section": "outro", "text": "",
             "outcome": "completed"},
        ],
    }
    (tmp_path / "q.yaml").write_text(yaml.safe_dump(body, allow_unicode=True),
                                     encoding="utf-8")
    with pytest.raises(QuestionnaireError, match="которого нет в анкете"):
        Questionnaire.from_yaml(tmp_path / "q.yaml")


def test_filter_pointing_forward_is_rejected() -> None:
    """Фильтр по ответу, который ещё не получен, — вопрос, который
    не задастся никогда."""
    q = Questionnaire.from_yaml(QUESTIONNAIRES / "wave-social.yaml")
    raw = q.model_dump(mode="json")
    for node in raw["nodes"]:
        if node["id"] == "q01_region":
            node["ask_if"] = {"core_key": "age", "between": [18, 45]}
    with pytest.raises(QuestionnaireError, match="спрашивают позже"):
        Questionnaire.parse(raw)


# ------------------------------------------------------------- содержание


def test_core_indicators_are_present_in_every_wave(wave: Questionnaire) -> None:
    """Ключи ядра — то, по чему строятся ряды между волнами."""
    required = {
        "region_satisfaction", "settlement_dynamics", "trust_federal",
        "trust_republic", "trust_local", "rating_president", "rating_head",
        "material_status", "safety", "anxiety_dynamics", "health",
        "news_sources", "svo_attitude", "fairness",
        "gender", "age", "education", "employment",
    }
    assert required <= set(wave.core_keys)


def test_sensitive_questions_allow_declining(wave: Questionnaire) -> None:
    """ТЗ §7.6: на чувствительный вопрос можно не отвечать, и отметка
    по нему не требуется для валидного интервью."""
    sensitive = [n for n in wave.nodes if getattr(n, "sensitive", False)]
    assert sensitive
    for node in sensitive:
        assert node.allow_dont_know
        assert node.id not in wave.required_core_nodes
