"""Подача: отклики, паузы и дубли между вопросами."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pytest

from socmon.core.codes import SpecialCode
from socmon.core.engine import Action, Answer, Event, InterviewSession, Prompt, Turn, Utterance
from socmon.core.questionnaire import Questionnaire
from socmon.voice.delivery import (
    Catalogue,
    Delivery,
    DeliveryError,
    Kind,
    Pool,
    media_name,
    repeat_reason,
    take_filename,
    take_media,
)

QUESTIONNAIRE = "src/socmon/questionnaires/wave-base.yaml"
NOW = datetime(2026, 9, 9, 18, 0, tzinfo=UTC)


@pytest.fixture(scope="module")
def catalogue() -> Catalogue:
    return Catalogue.from_yaml()


@pytest.fixture(scope="module")
def wave() -> Questionnaire:
    return Questionnaire.from_yaml(QUESTIONNAIRE)


def answer(node_id: str, code: int | None, text: str | None = None) -> Answer:
    return Answer(node_id=node_id, core_key=None, code=code, text=text, attempts=1, at=NOW)


def ask(*node_ids: str, action: Action = Action.ASK) -> Turn:
    return Turn(action=action, prompts=tuple(
        Prompt(n, f"Текст {n}", f"{n}_v1") for n in node_ids
    ), awaiting=node_ids[-1])


# ------------------------------------------------------------------ каталог


def test_catalogue_has_every_pool_and_stable_version(catalogue: Catalogue) -> None:
    assert set(catalogue.pools) == set(Pool)
    assert catalogue.version_id == Catalogue.from_yaml().version_id
    assert len(catalogue.version_id) == 16


def test_gendered_phrases_follow_the_voice(catalogue: Catalogue) -> None:
    phrase = catalogue.phrase("rep_ne_rasslyshal")
    assert "расслышала" in phrase.spoken("f")
    assert "расслышал." in phrase.spoken("m")


def test_catalogue_rejects_duplicate_keys(tmp_path: pytest.TempPathFactory) -> None:
    import yaml

    raw = yaml.safe_load(Path("src/socmon/voice/delivery.yaml").read_text(encoding="utf-8"))
    raw["pools"]["nudge"].append({"key": "ack_ugu", "text": "Угу?"})
    path = tmp_path / "bad.yaml"  # type: ignore[operator]
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    with pytest.raises(DeliveryError, match="не уникальны"):
        Catalogue.from_yaml(path)


# --------------------------------------------------------------------- план


def test_first_turn_starts_immediately_without_acknowledgement(
    catalogue: Catalogue, wave: Questionnaire
) -> None:
    delivery = Delivery(catalogue, wave, seed="call-1")
    segments = delivery.plan(ask("intro", "consent"))
    assert [s.kind for s in segments] == [Kind.PROMPT, Kind.PROMPT]
    assert segments[0].gap_ms == 0
    # Между вступлением и вопросом — короткая пауза, а не стык впритык.
    assert 150 <= segments[1].gap_ms <= 350


def test_no_acknowledgement_after_consent_or_screener(
    catalogue: Catalogue, wave: Questionnaire
) -> None:
    for gate in ("consent", "screen_age", "screen_territory"):
        for seed in range(40):
            delivery = Delivery(catalogue, wave, seed=f"{gate}-{seed}")
            segments = delivery.plan(ask("q01_region"), previous=answer(gate, 1))
            assert all(s.kind is Kind.PROMPT for s in segments), gate


def test_closed_answers_get_acknowledged_sometimes_not_always(
    catalogue: Catalogue, wave: Questionnaire
) -> None:
    """Откликаться на каждый ответ — тоже механика, только другая."""
    acked = 0
    for seed in range(400):
        delivery = Delivery(catalogue, wave, seed=f"s{seed}")
        segments = delivery.plan(ask("q02_dynamics"), previous=answer("q01_region", 2))
        acked += segments[0].kind is Kind.ACK
    assert 0.4 < acked / 400 < 0.7


def test_open_answer_is_always_acknowledged(catalogue: Catalogue, wave: Questionnaire) -> None:
    for seed in range(30):
        delivery = Delivery(catalogue, wave, seed=f"o{seed}")
        segments = delivery.plan(
            ask("q12_prices"), previous=answer("q11_problems", None, "цены и дороги")
        )
        assert segments[0].kind is Kind.ACK
        assert segments[0].key in {p.key for p in catalogue.pools[Pool.ACK_OPEN]}
        assert 500 <= segments[0].gap_ms <= 900


def test_silent_open_answer_is_not_thanked(catalogue: Catalogue, wave: Questionnaire) -> None:
    delivery = Delivery(catalogue, wave, seed="quiet")
    segments = delivery.plan(ask("q12_prices"), previous=answer("q11_problems", None, None))
    assert segments[0].kind is Kind.PROMPT


def test_refusal_gets_the_soft_pool(catalogue: Catalogue, wave: Questionnaire) -> None:
    soft = {p.key for p in catalogue.pools[Pool.ACK_SOFT]}
    for code in (SpecialCode.REFUSED, SpecialCode.DONT_KNOW):
        delivery = Delivery(catalogue, wave, seed=str(code))
        segments = delivery.plan(ask("q08_safety"), previous=answer("q07_material", code))
        assert segments[0].kind is Kind.ACK
        assert segments[0].key in soft


def test_unrecognized_answer_is_not_acknowledged(
    catalogue: Catalogue, wave: Questionnaire
) -> None:
    delivery = Delivery(catalogue, wave, seed="u")
    segments = delivery.plan(
        ask("q08_safety"), previous=answer("q07_material", SpecialCode.UNRECOGNIZED)
    )
    assert segments[0].kind is Kind.PROMPT


def test_same_acknowledgement_never_twice_in_a_row(
    catalogue: Catalogue, wave: Questionnaire
) -> None:
    delivery = Delivery(catalogue, wave, seed="row")
    previous_key: str | None = None
    for _ in range(200):
        segments = delivery.plan(ask("q02_dynamics"), previous=answer("q01_region", 1))
        acks = [s for s in segments if s.kind is Kind.ACK]
        if acks:
            assert acks[0].key != previous_key
            previous_key = acks[0].key


def test_repeat_is_prefaced_according_to_its_reason(
    catalogue: Catalogue, wave: Questionnaire
) -> None:
    asked = {p.key for p in catalogue.pools[Pool.REPEAT_ASKED]}
    unheard = {p.key for p in catalogue.pools[Pool.REPEAT_UNHEARD]}
    delivery = Delivery(catalogue, wave, gender="m", seed="rep")

    polite = delivery.plan(ask("q01_region", action=Action.REPEAT), reason="repeat_requested")
    assert polite[0].kind is Kind.PREFACE and polite[0].key in asked
    assert polite[1].kind is Kind.PROMPT and 250 <= polite[1].gap_ms <= 400

    sorry = delivery.plan(ask("q01_region", action=Action.REPEAT), reason="not_recognized")
    assert sorry[0].kind is Kind.PREFACE and sorry[0].key in unheard
    assert "расслышал." in sorry[0].text or "разобрал." in sorry[0].text


def test_repeat_reason_is_read_from_the_journal() -> None:
    events = [
        Event(NOW, "q1", "asked"),
        Event(NOW, "q1", "control_intent", "repeat"),
        Event(NOW, "q1", "repeat", "repeat_requested"),
    ]
    assert repeat_reason(events) == "repeat_requested"
    events[-1] = Event(NOW, "q1", "repeat", "not_recognized")
    assert repeat_reason(events) == "not_recognized"
    assert repeat_reason([Event(NOW, "q1", "asked")]) is None


def test_plan_is_reproducible_from_the_seed(catalogue: Catalogue, wave: Questionnaire) -> None:
    """Подача звонка восстанавливается по его идентификатору."""
    first = Delivery(catalogue, wave, seed="attempt-42", takes=3)
    second = Delivery(catalogue, wave, seed="attempt-42", takes=3)
    turns = [ask("q02_dynamics"), ask("q03_intro", "q03a_federal"), ask("q04_president")]
    answers = [answer("q01_region", 1), answer("q02_dynamics", 2), answer("q03c_local", 2)]
    for turn, prev in zip(turns, answers, strict=True):
        assert first.plan(turn, previous=prev) == second.plan(turn, previous=prev)


def test_takes_spread_across_the_library(catalogue: Catalogue, wave: Questionnaire) -> None:
    delivery = Delivery(catalogue, wave, seed="takes", takes=3)
    chosen = Counter(
        delivery.plan(ask("q02_dynamics"))[0].take for _ in range(300)
    )
    assert set(chosen) == {0, 1, 2}
    assert min(chosen.values()) > 50


def test_take_filenames_are_asterisk_safe() -> None:
    assert take_filename("q01_region_v1", 0, 3) == "q01_region_v1.wav"
    assert take_filename("q01_region_v1", 1, 3) == "q01_region_v1-t2.wav"
    assert take_filename("q01_region_v1", 2, 3) == "q01_region_v1-t3.wav"
    # Дубль сверх имеющихся сворачивается, а не падает.
    assert take_filename("q01_region_v1", 4, 3) == "q01_region_v1-t2.wav"
    assert take_filename("q01_region_v1", 2, 1) == "q01_region_v1.wav"
    assert media_name("q01_region_v1", 1, 2) == "q01_region_v1-t2"


def test_delivery_follows_a_real_interview(catalogue: Catalogue, wave: Questionnaire) -> None:
    """Сквозной проход: план строится на каждом ходу и никогда не пуст."""
    from tests.test_monitoring_questionnaire import SCRIPT

    session = InterviewSession(questionnaire=wave)
    delivery = Delivery(catalogue, wave, seed="e2e", takes=3)
    turn = session.begin()
    previous: Answer | None = None
    kinds: Counter[Kind] = Counter()
    for _ in range(80):
        segments = delivery.plan(
            turn, previous=previous, reason=repeat_reason(session.events)
        )
        assert segments and segments[-1].kind is Kind.PROMPT
        kinds.update(s.kind for s in segments)
        if turn.action in (Action.END, Action.TRANSFER):
            break
        before = set(session.answers)
        reply = "затрудняюсь" if turn.awaiting == "q05_head" else SCRIPT[turn.awaiting]
        turn = session.submit(Utterance.said(reply))
        fresh = [a for k, a in session.answers.items() if k not in before]
        previous = fresh[-1] if fresh else None
    assert session.finished
    assert kinds[Kind.ACK] >= 8
    assert kinds[Kind.PROMPT] >= 37


def test_take_media_follows_the_manifest_list_not_the_number(tmp_path: Path) -> None:
    """После отсева дублей список может быть «t3 без t2»: номер указал бы
    на файл, которого нет."""
    for name in ("q_v1.wav", "q_v1-t3.wav"):
        (tmp_path / name).write_bytes(b"RIFF")
    takes = ("q_v1-t3.wav",)
    assert take_media("q_v1", 0, takes, tmp_path) == "q_v1"
    assert take_media("q_v1", 1, takes, tmp_path) == "q_v1-t3"
    assert take_media("q_v1", 2, takes, tmp_path) == "q_v1"
    # Дубль, которого нет на диске, не предлагается.
    assert take_media("q_v1", 1, ("q_v1-t2.wav",), tmp_path) == "q_v1"
    # Без каталога — верим манифесту.
    assert take_media("q_v1", 1, ("q_v1-t2.wav",), None) == "q_v1-t2"
