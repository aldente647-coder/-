"""Нестандартные ответы респондентов — корпус из tests/scenarios.yaml.

Каждый сценарий доводит интервью до нужного узла по «зелёному» сценарию,
подаёт реплику и проверяет, что робот сделал ровно то, что сделал бы живой
интервьюер: уточнил, переспросил, объяснил, записал код или закончил.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from socmon.core.codes import Outcome, SpecialCode
from socmon.core.engine import Action, InterviewSession, Turn, Utterance
from socmon.core.questionnaire import Questionnaire
from socmon.voice.delivery import repeat_reason
from tests.test_monitoring_questionnaire import SCRIPT

QUESTIONNAIRE = Path("src/socmon/questionnaires/wave-base.yaml")
CASES = yaml.safe_load(Path("tests/scenarios.yaml").read_text(encoding="utf-8"))


def reach(session: InterviewSession, node_id: str) -> Turn:
    turn = session.begin()
    for _ in range(80):
        if turn.awaiting == node_id:
            return turn
        assert turn.action is Action.ASK, f"не дошли до {node_id}: {turn}"
        turn = session.submit(Utterance.said(SCRIPT[turn.awaiting]))
    raise AssertionError(f"узел {node_id} недостижим по сценарию")


def utter(text: str) -> Utterance:
    return Utterance.said(text) if text else Utterance.quiet()


@pytest.mark.parametrize("case", CASES, ids=[c["title"] for c in CASES])
def test_scenario(case: dict) -> None:  # type: ignore[type-arg]
    session = InterviewSession(Questionnaire.from_yaml(QUESTIONNAIRE))
    node = case["node"]
    reach(session, node)
    turn = session.submit(utter(case["said"]))
    expect: str = case["expect"]
    reason = repeat_reason(session.events)

    if expect == "code":
        assert session.answers[node].code == case["code"], session.answers.get(node)
    elif expect == "probe":
        assert turn.action is Action.REPEAT and reason == "hedged", (turn, reason)
    elif expect == "repeat":
        assert turn.action is Action.REPEAT and reason == "not_recognized", (turn, reason)
    elif expect == "asked_repeat":
        assert turn.action is Action.REPEAT and reason == "repeat_requested", (turn, reason)
    elif expect == "who":
        assert turn.action is Action.REPEAT and reason == "who_asked", (turn, reason)
    elif expect == "dont_know":
        assert session.answers[node].code == SpecialCode.DONT_KNOW
    elif expect == "dont_know_consent":
        # Уклончивое согласие — не согласие: уточняем, а при повторе — отказ.
        assert turn.action is Action.REPEAT and reason == "hedged"
    elif expect == "refused":
        assert session.answers[node].code == SpecialCode.REFUSED
    elif expect == "unrecognized":
        assert session.answers[node].code == SpecialCode.UNRECOGNIZED
    elif expect == "open":
        assert session.answers[node].code is None
        assert session.answers[node].text == case["said"]
    elif expect.startswith("end:"):
        assert session.finished
        assert session.outcome is Outcome(expect.removeprefix("end:"))
    else:
        raise AssertionError(f"неизвестное ожидание {expect}")

    if "then" in case:
        assert turn.action is Action.REPEAT, "вторая реплика возможна только после переспроса"
        session.submit(utter(case["then"]))
        assert session.answers[node].code == case["code"], session.answers.get(node)
