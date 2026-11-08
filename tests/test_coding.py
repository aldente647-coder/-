"""Разбор открытых ответов: темы, сущности, тональность."""

from __future__ import annotations

from pathlib import Path

import pytest

from socmon.coding.analysis import (
    InterviewInput,
    OpenAnswer,
    Tone,
    analyse,
    detect_signals,
    estimate_tone,
)
from socmon.coding.classifier import (
    Theme,
    ThemeClassifier,
    classify,
    unmatched_fragments,
)
from socmon.coding.entities import (
    EntityKind,
    Gazetteer,
    GazetteerEntry,
    extract,
    tally,
)

REFERENCE = Path("src/socmon/classifiers/themes-v1.yaml")


@pytest.fixture
def themes() -> ThemeClassifier:
    return ThemeClassifier.from_yaml(REFERENCE)


@pytest.fixture
def book() -> Gazetteer:
    return Gazetteer(entries=(
        GazetteerEntry("leader", "Пушилин Денис", EntityKind.PERSON, ("Пушилин", "Пушилину")),
        GazetteerEntry("tg", "Донецк Live", EntityKind.CHANNEL, ("Донецк Лайв",)),
        GazetteerEntry("ivanov_a", "Иванов Андрей", EntityKind.PERSON, ("Иванов",)),
        GazetteerEntry("ivanov_b", "Иванов Сергей", EntityKind.PERSON, ("Иванов",)),
    ))


# --------------------------------------------------------------- классификатор


def test_one_answer_can_carry_several_themes(themes: ThemeClassifier) -> None:
    """Схлопывать в одну тему значит терять то, ради чего открытый вопрос
    и задавался."""
    coding = classify("цены выросли, а до больницы не доехать, дороги разбитые", themes)
    assert set(coding.codes) >= {"prices", "healthcare", "transport"}


def test_exclusion_silences_a_match_not_the_whole_theme(themes: ThemeClassifier) -> None:
    """«Цены выросли, а дороги разбитые» — это законно обе темы.
    Исключение, отменяющее тему целиком, съедало бы половину ответов."""
    coding = classify("цены выросли, а дороги разбитые", themes)
    assert "prices" in coding.codes
    assert "transport" in coding.codes


def test_road_is_transport_not_prices(themes: ThemeClassifier) -> None:
    coding = classify("дорога до садика совсем разбита", themes)
    assert coding.codes == ("transport",)


def test_alarming_themes_are_marked(themes: ThemeClassifier) -> None:
    """ТЗ §11.1: перечень тревожных тем."""
    assert classify("страшно за детей", themes).alarming == ("safety",)
    assert classify("автобус не ходит", themes).alarming == ()


def test_unmatched_answer_is_flagged_not_forced(themes: ThemeClassifier) -> None:
    coding = classify("соседи мусорят во дворе", themes)
    assert coding.unmatched
    assert coding.codes == ()


def test_coding_carries_its_classifier_version(themes: ThemeClassifier) -> None:
    """ТЗ §11.3: скрытое изменение исторических данных запрещено."""
    assert classify("цены", themes).classifier_version == themes.version_id


def test_editing_a_rule_changes_the_version(themes: ThemeClassifier) -> None:
    edited = themes.model_copy(update={
        "themes": (*themes.themes[:-1], Theme(
            code="nothing", name="Ничего", phrases=("совсем ничего",)
        )),
    })
    assert edited.version_id != themes.version_id


def test_human_revision_note_does_not_affect_the_version(
    themes: ThemeClassifier
) -> None:
    """Пометка «утверждено протоколом» — для людей; идентификатор считается
    от правил, иначе он менялся бы без единой правки в разметке."""
    renamed = themes.model_copy(update={"revision": "утверждено 21.08.2026"})
    assert renamed.version_id == themes.version_id


def test_every_coding_is_marked_as_generated(themes: ThemeClassifier) -> None:
    assert classify("цены", themes).is_model_generated is True


def test_match_evidence_is_kept_for_checking(themes: ThemeClassifier) -> None:
    """«Тема такая-то, потому что модель так решила» проверке не поддаётся."""
    hit = classify("всё подорожало", themes).themes[0]
    assert hit.matched == "подорожало"


def test_theme_without_rules_is_refused() -> None:
    # pydantic заворачивает ошибку валидатора в ValidationError, но та
    # наследует ValueError и несёт наше сообщение.
    with pytest.raises(ValueError, match="не задано ни одного правила"):
        Theme(code="x", name="Пустая")


def test_broken_pattern_is_refused_at_load_time() -> None:
    with pytest.raises(ValueError, match="плохое выражение"):
        Theme(code="x", name="Кривая", patterns=("[unclosed",))


def test_duplicate_codes_are_refused() -> None:
    theme = Theme(code="same", name="Тема", phrases=("слово",))
    with pytest.raises(ValueError, match="не уникальны"):
        ThemeClassifier(name="битый", themes=(theme, theme))


def test_new_theme_candidates_need_repetition(themes: ThemeClassifier) -> None:
    """Тема, прозвучавшая единожды, темой пока не является."""
    answers = ["соседи мусорят"] * 4 + ["интернет пропадает"]
    words = dict(unmatched_fragments(answers, themes, min_count=3))
    assert "соседи" in words
    assert "интернет" not in words


# -------------------------------------------------------------------- сущности


def test_gazetteer_resolves_a_known_alias(book: Gazetteer) -> None:
    """Распознавание пишет иначе, чем справочник, и без вариантов написания
    упоминание просто потеряется."""
    mentions = extract("Доверяю Пушилину, он хоть что-то делает", book)
    assert mentions[0].key == "leader"
    assert mentions[0].ambiguous is False


def test_two_people_with_one_surname_stay_unresolved(book: Gazetteer) -> None:
    """ТЗ §7.5 запрещает объединять неоднозначные имена автоматически."""
    mention = extract("Иванов у нас нормальный", book)[0]
    assert mention.ambiguous is True
    assert mention.key is None
    assert set(mention.candidates) == {"ivanov_a", "ivanov_b"}


def test_longer_name_wins_over_its_part(book: Gazetteer) -> None:
    """Иначе полное название развалится на два ложных упоминания."""
    mentions = extract("смотрю Донецк Лайв", book)
    assert [m.key for m in mentions] == ["tg"]


def test_unknown_name_is_typed_by_context(book: Gazetteer) -> None:
    mention = extract("есть такой депутат Ковальчук, к нему и ходим", book)[0]
    assert mention.kind is EntityKind.PERSON
    assert mention.ambiguous is True


def test_everything_outside_the_gazetteer_is_ambiguous(book: Gazetteer) -> None:
    """Мы не знаем, кто такой «Петров» из ответа, и делать вид,
    что знаем, нельзя."""
    mentions = extract("нравится как Петров работает", book)
    assert all(m.ambiguous for m in mentions)
    assert all(m.key is None for m in mentions)


def test_tally_never_merges_ambiguous_with_resolved(book: Gazetteer) -> None:
    """Иначе ручная проверка теряет смысл: сливать будет уже нечего."""
    mentions = [
        *extract("Пушилин молодец", book),
        *extract("Пушилин снова выступал", book),
        *extract("а Ковальчук ничего не делает", book),
    ]
    rows = tally(mentions, book)
    resolved = {r.key: r for r in rows if not r.ambiguous}
    assert resolved["leader"].count == 2
    assert any(r.ambiguous and r.count == 1 for r in rows)


def test_ambiguous_row_carries_a_warning(book: Gazetteer) -> None:
    rows = tally(list(extract("депутат Сидоров обещал", book)), book)
    assert rows[0].note is not None and "ручной проверки" in rows[0].note


# ----------------------------------------------------------------- тональность


def test_required_signals_are_detected() -> None:
    """ТЗ §11.1: признаки раздражения, одобрения, похвалы, недоверия, тревоги."""
    names = {s.name for s in detect_signals(
        "надоело, врут они всё, страшно, но спасибо что хоть отлично сделали дорогу"
    )}
    assert names == {"раздражение", "недоверие", "тревога", "одобрение", "похвала"}


def test_negative_answer_reads_as_negative() -> None:
    estimate = estimate_tone("надоело, страшно, не верю никому")
    assert estimate.tone is Tone.NEGATIVE
    assert estimate.confidence > 0.6


def test_mixed_answer_lowers_confidence_instead_of_faking_neutrality() -> None:
    """«Спасибо, но надоело уже» — это не нейтральный ответ, а неуверенная
    оценка, и разница в выгрузке должна быть видна."""
    mixed = estimate_tone("спасибо, но надоело уже")
    plain = estimate_tone("обычный ответ без признаков")
    assert mixed.tone is Tone.NEUTRAL
    assert mixed.confidence < plain.confidence


def test_tone_is_marked_as_a_model_estimate() -> None:
    """ТЗ §11.1: не диагностика и не основание для выводов о респонденте."""
    estimate = estimate_tone("страшно")
    assert estimate.is_model_generated is True
    assert "Не является диагностикой" in estimate.disclaimer


def test_signal_evidence_is_kept() -> None:
    assert detect_signals("мне страшно")[0].matched == "страшно"


# ---------------------------------------------------------------- разбор целиком


def test_full_analysis_collects_everything_the_tor_asks_for(
    themes: ThemeClassifier, book: Gazetteer
) -> None:
    result = analyse(InterviewInput(
        open_answers=(
            OpenAnswer(
                "q_open_leader",
                "Читаю Донецк Лайв, там депутат Ковальчук пишет", 312.4, 319.1,
            ),
            OpenAnswer("q_open_worry", "Цены выросли, надоело, страшно за детей", 344.0, 351.2),
        ),
        closed_total=20, closed_unrecognized=1,
    ), themes, book)

    assert result.summary
    assert "prices" in result.themes
    assert "safety" in result.alarming_themes
    assert result.tone.tone is Tone.NEGATIVE
    assert all(q.timed for q in result.quotes)
    assert result.classifier_version == themes.version_id
    assert result.is_model_generated is True


def test_quotes_keep_their_timestamps(themes: ThemeClassifier) -> None:
    """ТЗ §11.1: дословные цитаты с временными метками. Модель отдаёт
    их сама, выравнивать отдельным проходом не нужно."""
    result = analyse(InterviewInput(
        open_answers=(OpenAnswer("q", "цены выросли", 12.5, 15.0),),
    ), themes)
    assert result.quotes[0].start_sec == 12.5
    assert result.quotes[0].end_sec == 15.0


def test_ambiguous_mentions_send_the_interview_to_review(
    themes: ThemeClassifier, book: Gazetteer
) -> None:
    result = analyse(InterviewInput(
        open_answers=(OpenAnswer("q", "депутат Ковальчук обещал", None, None),),
    ), themes, book)
    assert result.quality.needs_review
    assert any("ручной проверки" in r for r in result.review_notes)


def test_many_unrecognized_answers_send_the_interview_to_review(
    themes: ThemeClassifier
) -> None:
    result = analyse(InterviewInput(
        open_answers=(OpenAnswer("q", "цены", None, None),),
        closed_total=20, closed_unrecognized=9,
    ), themes)
    assert result.quality.needs_review
    assert result.quality.unrecognized_share > 0.4


def test_totally_unclassified_interview_hints_at_a_gap(
    themes: ThemeClassifier
) -> None:
    """Пробел в классификаторе — повод его расширить, а не молча
    выбросить ответ (ТЗ §11.2)."""
    result = analyse(InterviewInput(
        open_answers=(OpenAnswer("q", "соседи мусорят во дворе", None, None),),
    ), themes)
    assert any("пробел" in r for r in result.review_notes)


def test_clean_interview_needs_no_review(themes: ThemeClassifier, book: Gazetteer) -> None:
    result = analyse(InterviewInput(
        open_answers=(OpenAnswer("q", "цены выросли, но в целом устраивает", None, None),),
        closed_total=20, closed_unrecognized=0,
    ), themes, book)
    assert not result.quality.needs_review


def test_answer_about_a_person_is_not_a_classifier_gap(
    themes: ThemeClassifier, book: Gazetteer
) -> None:
    """Вопрос об источниках доверия (ТЗ §7.5) вообще не про проблемы.
    Считать «доверяю такому-то» пробелом классификатора значит слать
    на ручную проверку каждое второе интервью."""
    result = analyse(InterviewInput(
        open_answers=(OpenAnswer("q_open_leader", "Доверяю Пушилину", None, None),),
        closed_total=20, closed_unrecognized=0,
    ), themes, book)
    assert not any("пробел" in r for r in result.review_notes)


# ------------------------------------------ рамка кодировки Заказчика

CUSTOMER = Path("src/socmon/classifiers/themes-v2.yaml")


@pytest.fixture
def customer_themes() -> ThemeClassifier:
    return ThemeClassifier.from_yaml(CUSTOMER)


def test_customer_frame_keeps_every_category(customer_themes: ThemeClassifier) -> None:
    """Перечень к вопросу 11 воспроизведён целиком.

    Категории Заказчика не объединяются по нашему усмотрению: «лекарства»
    отдельно от медицины, «состояние дорог» отдельно от транспорта,
    «низкие доходы» отдельно от роста цен. Свести их обратно — потерять
    разрез, ради которого перечень и составлялся.
    """
    assert {t.code for t in customer_themes.themes} == {
        "prices", "incomes", "utilities", "communications", "healthcare",
        "medicines", "unemployment", "roads", "transport", "housing",
        "safety", "education", "documents", "payments", "nothing",
    }


@pytest.mark.parametrize(
    ("said", "expected"),
    [
        ("цены выросли, а дороги разбитые", {"prices", "roads"}),
        ("лекарств в аптеке нет и к врачу не записаться", {"medicines", "healthcare"}),
        ("пенсия маленькая, на жизнь не хватает", {"incomes"}),
        ("света нет второй день и вода по графику", {"utilities"}),
        ("обстрелы каждую ночь, страшно", {"safety"}),
        ("связь плохая, интернет не ловит", {"communications"}),
        ("выплаты задерживают уже третий месяц", {"payments"}),
        ("автобусы редко ходят и в садик мест нет", {"transport", "education"}),
        ("справки собирать замучился", {"documents"}),
        ("работы нет, безработица", {"unemployment"}),
        ("квартиру разбило, жить негде", {"housing"}),
        ("все хорошо, проблем нет", {"nothing"}),
    ],
)
def test_customer_frame_reads_inflected_speech(
    customer_themes: ThemeClassifier, said: str, expected: set[str]
) -> None:
    """Живая речь идёт в падежах.

    Фразы ловят только именительный: «лекарства» находится, «лекарств
    в аптеке нет» — нет. Основы слов шаблонами закрывают именно это,
    и проверять их нужно на склонённых формах, а не на словарных.
    """
    assert set(classify(said, customer_themes).codes) == expected


def test_prices_do_not_swallow_roads(customer_themes: ThemeClassifier) -> None:
    """«Дорого» и «дорога» — общий корень, разные темы. Ошибка здесь
    переносит жалобы на дороги в рост цен и наоборот."""
    assert set(classify("все очень дорого", customer_themes).codes) == {"prices"}
    assert set(classify("дороги разбиты", customer_themes).codes) == {"roads"}
