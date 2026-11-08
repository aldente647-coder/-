from __future__ import annotations

import pytest

from socmon.core.codes import AaporCode, Outcome, SpecialCode, to_aapor
from socmon.core.engine import Action, InterviewError, InterviewSession, Turn, Utterance
from socmon.core.questionnaire import Questionnaire
from tests.conftest import FakeClock

# Ответы на каждый узел, ведущие по «зелёному» маршруту до конца анкеты.
HAPPY_PATH: dict[str, str] = {
    "consent": "да, конечно",
    "screen_age": "да, исполнилось",
    "screen_territory": "да, живу",
    "q_life": "шесть",
    "q_family_change": "не изменилось",
    "q_material": "четыре",
    "q_safety": "семь",
    "q_anxiety": "восемь",
    "q_trust": "три",
    "q_fairness": "четыре",
    "q_conflict": "шесть",
    "q_authorities": "пять",
    "q_local_problem": "цены",
    "q_collective": "скорее да",
    "q_open_leader": "доверяю нашему местному блогеру Иванову",
    "q_open_worry": "цены растут, хотелось бы стабильности",
    "d_gender": "женский",
    "d_age_group": "мне 42 года",
    "d_territory": "центральный",
    "d_settlement": "в городе",
    "d_education": "высшее",
    "d_employment": "работаю",
    "d_nationality": "русская",
    "d_politics": "иногда",
}


def drive(session: InterviewSession, script: dict[str, str]) -> Turn:
    """Проиграть интервью целиком по сценарию ответов."""
    turn = session.begin()
    guard = 0
    while turn.awaiting is not None:
        guard += 1
        if guard > 200:
            raise AssertionError(f"интервью зациклилось на {turn.awaiting}")
        reply = script.get(turn.awaiting)
        if reply is None:
            raise AssertionError(f"в сценарии нет ответа на узел {turn.awaiting}")
        turn = session.submit(Utterance.said(reply))
    return turn


def test_happy_path_completes(core: Questionnaire, clock: FakeClock) -> None:
    session = InterviewSession(core, clock=clock)
    turn = drive(session, HAPPY_PATH)

    assert turn.action is Action.END
    assert turn.outcome is Outcome.COMPLETED
    assert session.consented is True
    assert to_aapor(session.outcome, None) is AaporCode.COMPLETE


def test_happy_path_records_every_answer(core: Questionnaire, clock: FakeClock) -> None:
    session = InterviewSession(core, clock=clock)
    drive(session, HAPPY_PATH)

    assert session.answers["q_anxiety"].code == 8
    assert session.answers["q_family_change"].code == 2
    assert session.answers["q_local_problem"].code == 1
    assert session.answers["d_age_group"].code == 2  # «сорок два» → группа 35-54
    assert session.answers["q_open_leader"].text is not None
    assert session.answers["q_life"].core_key == "life_situation"


def test_intro_is_spoken_before_the_first_question(core: Questionnaire, clock: FakeClock) -> None:
    """Реплики без вопроса проигрываются подряд и не ждут ответа."""
    session = InterviewSession(core, clock=clock)
    turn = session.begin()

    assert turn.action is Action.ASK
    assert turn.awaiting == "consent"
    assert [p.node_id for p in turn.prompts] == ["intro", "consent"]


def test_refusal_of_consent_ends_interview(core: Questionnaire, clock: FakeClock) -> None:
    session = InterviewSession(core, clock=clock)
    session.begin()
    turn = session.submit(Utterance.said("нет, не хочу"))

    assert turn.action is Action.END
    assert turn.outcome is Outcome.NO_CONSENT
    assert session.consented is False
    assert to_aapor(session.outcome, None) is AaporCode.REFUSAL


def test_silence_at_consent_is_never_read_as_agreement(
    core: Questionnaire, clock: FakeClock
) -> None:
    """ТЗ §8.2 прямо запрещает трактовать отсутствие ответа как согласие."""
    session = InterviewSession(core, clock=clock)
    session.begin()

    repeat = session.submit(Utterance.quiet())
    assert repeat.action is Action.REPEAT
    assert repeat.awaiting == "consent"

    end = session.submit(Utterance.quiet())
    assert end.outcome is Outcome.NO_CONSENT
    assert session.consented is False


def test_question_is_repeated_exactly_once(core: Questionnaire, clock: FakeClock) -> None:
    """ТЗ §8.1: повторить вопрос не более одного раза, потом — код."""
    session = InterviewSession(core, clock=clock)
    session.begin()
    session.submit(Utterance.said("да"))
    session.submit(Utterance.said("да"))
    session.submit(Utterance.said("да"))  # → q_life

    first = session.submit(Utterance.said("кхм что-то непонятное"))
    assert first.action is Action.REPEAT
    assert first.awaiting == "q_life"

    second = session.submit(Utterance.said("снова непонятное"))
    assert second.action is Action.ASK
    assert second.awaiting == "q_family_change"
    assert session.answers["q_life"].code == SpecialCode.UNRECOGNIZED
    assert session.answers["q_life"].attempts == 2


def test_three_unrecognized_in_a_row_abort_the_interview(
    core: Questionnaire, clock: FakeClock
) -> None:
    """ТЗ §5.3: систематически нераспознанная речь делает интервью непригодным."""
    session = InterviewSession(core, clock=clock)
    session.begin()
    for _ in range(3):
        session.submit(Utterance.said("да"))

    turn: Turn | None = None
    for _ in range(10):
        turn = session.submit(Utterance.said("шшшш"))
        if turn.outcome is not None:
            break
    assert turn is not None
    assert turn.outcome is Outcome.SPEECH_UNUSABLE
    assert to_aapor(session.outcome, None) is AaporCode.LANGUAGE_BARRIER


def test_recognized_answer_resets_the_unrecognized_streak(
    core: Questionnaire, clock: FakeClock
) -> None:
    session = InterviewSession(core, clock=clock)
    session.begin()
    for _ in range(3):
        session.submit(Utterance.said("да"))

    session.submit(Utterance.said("шшш"))          # q_life, переспрос
    session.submit(Utterance.said("шшш"))          # q_life → нераспознано
    session.submit(Utterance.said("ухудшилось"))   # q_family_change — разобрали
    turn = session.submit(Utterance.said("пять"))  # q_material

    assert turn.outcome is None
    assert session.answers["q_family_change"].code == 3


def test_dont_call_request_terminates_with_approved_farewell(
    core: Questionnaire, clock: FakeClock
) -> None:
    """ТЗ §6.2: явная просьба не звонить — немедленный стоп-лист."""
    session = InterviewSession(core, clock=clock)
    session.begin()
    session.submit(Utterance.said("да"))
    turn = session.submit(Utterance.said("больше не звоните мне"))

    assert turn.action is Action.END
    assert turn.outcome is Outcome.DONT_CALL_REQUEST
    assert turn.prompts[-1].node_id == "end_dont_call"


def test_operator_request_transfers(core: Questionnaire, clock: FakeClock) -> None:
    """ТЗ §8.1 и §9.2: перевод на оператора по запросу респондента."""
    session = InterviewSession(core, clock=clock)
    session.begin()
    turn = session.submit(Utterance.said("дайте оператора"))

    assert turn.action is Action.TRANSFER
    assert turn.outcome is Outcome.TRANSFERRED_TO_OPERATOR


def test_repeat_request_does_not_consume_the_retry_budget(
    core: Questionnaire, clock: FakeClock
) -> None:
    """Просьба повторить — право респондента, а не промах распознавания."""
    session = InterviewSession(core, clock=clock)
    session.begin()
    session.submit(Utterance.said("да"))
    session.submit(Utterance.said("да"))
    session.submit(Utterance.said("да"))  # → q_life

    again = session.submit(Utterance.said("повторите пожалуйста"))
    assert again.action is Action.REPEAT

    misheard = session.submit(Utterance.said("бу-бу-бу"))
    assert misheard.action is Action.REPEAT  # бюджет переспроса ещё цел

    answered = session.submit(Utterance.said("девять"))
    assert answered.awaiting == "q_family_change"
    assert session.answers["q_life"].code == 9


def test_screener_routes_minors_out(core: Questionnaire, clock: FakeClock) -> None:
    session = InterviewSession(core, clock=clock)
    session.begin()
    session.submit(Utterance.said("да"))
    turn = session.submit(Utterance.said("нет, мне меньше"))

    assert turn.outcome is Outcome.INELIGIBLE_AGE
    assert to_aapor(session.outcome, None) is AaporCode.NOT_ELIGIBLE


def test_dont_know_is_accepted_and_coded(core: Questionnaire, clock: FakeClock) -> None:
    """Отказ от ответа на отдельный вопрос не бракует интервью (ТЗ §7.6)."""
    script = dict(HAPPY_PATH)
    script["d_nationality"] = "затрудняюсь ответить"
    script["q_trust"] = "не хочу отвечать"

    session = InterviewSession(core, clock=clock)
    turn = drive(session, script)

    assert turn.outcome is Outcome.COMPLETED
    assert session.answers["d_nationality"].code == SpecialCode.DONT_KNOW
    assert session.answers["q_trust"].code == SpecialCode.REFUSED
    assert session.answers["q_trust"].is_substantive is False


def test_not_interested_is_an_answer_not_a_stop_list_request(
    core: Questionnaire, clock: FakeClock
) -> None:
    """«Не интересуюсь политикой» — содержательный ответ, а не просьба
    исключить номер. Это ровно та ошибка, о которой предупреждает блюпринт."""
    script = dict(HAPPY_PATH)
    script["d_politics"] = "не интересуюсь"

    session = InterviewSession(core, clock=clock)
    turn = drive(session, script)

    assert turn.outcome is Outcome.COMPLETED
    assert session.answers["d_politics"].code == 3


def test_hangup_before_consent_is_not_a_break_off(core: Questionnaire, clock: FakeClock) -> None:
    session = InterviewSession(core, clock=clock)
    session.begin()
    turn = session.submit(Utterance.dropped())

    assert turn.outcome is Outcome.REFUSAL_BEFORE
    assert to_aapor(session.outcome, None) is AaporCode.REFUSAL


def test_hangup_after_consent_is_a_break_off(core: Questionnaire, clock: FakeClock) -> None:
    session = InterviewSession(core, clock=clock)
    session.begin()
    session.submit(Utterance.said("да"))
    session.submit(Utterance.said("да"))
    session.submit(Utterance.said("да"))
    turn = session.submit(Utterance.dropped())

    assert turn.outcome is Outcome.REFUSAL_DURING
    assert to_aapor(session.outcome, None) is AaporCode.BREAK_OFF


def test_events_form_a_timeline(core: Questionnaire, clock: FakeClock) -> None:
    """ТЗ §10.1: по каждому интервью нужна шкала времени."""
    session = InterviewSession(core, clock=clock)
    drive(session, HAPPY_PATH)

    kinds = [e.kind for e in session.events]
    assert kinds[0] == "interview_started"
    assert kinds[-1] == "interview_finished"
    assert "consent_granted" in kinds
    assert all(
        earlier.at <= later.at
        for earlier, later in zip(session.events, session.events[1:], strict=False)
    )


def test_duration_is_measured(core: Questionnaire) -> None:
    slow = FakeClock(step_sec=30.0)
    session = InterviewSession(core, clock=slow)
    drive(session, HAPPY_PATH)

    assert session.duration_sec > 0
    assert session.within_target_duration is False  # 30 с на реплику — заведомо мимо


def test_submitting_after_the_end_is_a_bug_not_a_no_op(
    core: Questionnaire, clock: FakeClock
) -> None:
    session = InterviewSession(core, clock=clock)
    session.begin()
    session.submit(Utterance.said("нет"))
    with pytest.raises(RuntimeError, match="уже завершено"):
        session.submit(Utterance.said("да"))


# --------------------------------------------------------------- обрыв снаружи


def test_aborted_interview_keeps_the_answers_given(core: Questionnaire) -> None:
    """Прерванный разговор — это данные, а не мусор."""
    session = InterviewSession(core, clock=FakeClock())
    session.begin()
    for reply in ("да", "да", "да", "семь"):
        session.submit(Utterance.said(reply))
    session.abort(Outcome.PARTIAL)

    assert session.outcome is Outcome.PARTIAL
    assert session.answers["q_life"].code == 7
    assert any(e.kind == "interview_aborted" for e in session.events)


def test_abort_after_finish_changes_nothing(core: Questionnaire) -> None:
    session = InterviewSession(core, clock=FakeClock())
    session.begin()
    session.submit(Utterance.said("нет"))
    original = session.outcome

    session.abort(Outcome.PARTIAL)
    assert session.outcome is original


def test_abort_before_start_is_an_error(core: Questionnaire) -> None:
    with pytest.raises(InterviewError, match="не начиналось"):
        InterviewSession(core, clock=FakeClock()).abort(Outcome.PARTIAL)
