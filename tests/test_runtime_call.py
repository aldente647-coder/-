"""Цикл звонка: события телефонии ↔ движок анкеты."""

from __future__ import annotations

import uuid
from datetime import timedelta

import pytest

from socmon.core.codes import Outcome
from socmon.core.questionnaire import Questionnaire
from socmon.runtime.call import CallRuntime, RuntimePolicy
from socmon.telephony.machine import CallEvent
from socmon.telephony.provider import CallHandle, ProviderEvent, RecognizedSpeech
from tests.fakes import START, FakeClock, FakeJournal, FakeProvider

ATTEMPT = uuid.UUID("6f1c2a10-0000-4000-8000-000000000001")


@pytest.fixture
def provider() -> FakeProvider:
    return FakeProvider()


@pytest.fixture
def journal() -> FakeJournal:
    return FakeJournal()


@pytest.fixture
def runtime(
    provider: FakeProvider, journal: FakeJournal, core: Questionnaire
) -> CallRuntime:
    engine = CallRuntime(
        provider, core, journal,
        policy=RuntimePolicy(barge_in_delay_sec=3.0, silence_timeout_sec=6.0),
        operator_endpoint="PJSIP/operator1",
        clock=FakeClock(step_sec=2.0),
    )
    engine.register(ATTEMPT, CallHandle(str(ATTEMPT), "chan-1"))
    return engine


def event(kind: CallEvent, seconds: float = 0, event_id: str | None = None) -> ProviderEvent:
    return ProviderEvent(
        attempt_id=str(ATTEMPT), event=kind,
        at=START + timedelta(seconds=seconds),
        event_id=event_id or f"{kind.value}-{seconds}",
    )


def speech(
    text: str | None, seconds: float, *, final: bool = True, silence: bool = False
) -> RecognizedSpeech:
    return RecognizedSpeech(
        attempt_id=str(ATTEMPT), text=text,
        at=START + timedelta(seconds=seconds), final=final, silence=silence,
    )


async def answer(runtime: CallRuntime, *, at: float = 1.0) -> None:
    await runtime.on_call_event(event(CallEvent.RING, 0))
    await runtime.on_call_event(event(CallEvent.ANSWER, at))


# ---------------------------------------------------------------- начало звонка


async def test_recording_starts_before_the_first_word(
    runtime: CallRuntime, provider: FakeProvider
) -> None:
    """Вступление с информированием о согласии — часть разговора
    и юридически самая важная (ТЗ §10.2, §8.4)."""
    await answer(runtime)
    assert provider.recorded == [str(ATTEMPT)]
    assert provider.played[0][1] == "intro_v1"


async def test_intro_and_question_are_played_in_order(
    runtime: CallRuntime, provider: FakeProvider
) -> None:
    await answer(runtime)
    assert provider.media_for(str(ATTEMPT)) == ["intro_v1", "consent_v1"]


async def test_no_answer_never_starts_an_interview(
    runtime: CallRuntime, provider: FakeProvider, journal: FakeJournal
) -> None:
    await runtime.on_call_event(event(CallEvent.RING, 0))
    await runtime.on_call_event(event(CallEvent.NO_ANSWER, 30))
    assert provider.recorded == []
    assert journal.interviews == {}


async def test_events_for_unknown_calls_are_ignored(
    runtime: CallRuntime, provider: FakeProvider
) -> None:
    """Чужой канал в том же приложении не должен ничего запустить."""
    stranger = ProviderEvent(
        attempt_id=str(uuid.uuid4()), event=CallEvent.ANSWER, at=START, event_id="x"
    )
    await runtime.on_call_event(stranger)
    assert provider.recorded == []


async def test_redelivered_event_does_not_restart_the_interview(
    runtime: CallRuntime, provider: FakeProvider
) -> None:
    await answer(runtime)
    await runtime.on_call_event(event(CallEvent.ANSWER, 1, event_id="answer-1.0"))
    assert provider.recorded == [str(ATTEMPT)]


# --------------------------------------------------------------------- диалог


async def test_answers_advance_the_questionnaire(
    runtime: CallRuntime, provider: FakeProvider
) -> None:
    await answer(runtime)
    await runtime.on_speech(speech("да", 10))
    assert provider.media_for(str(ATTEMPT))[-1] == "screen_age_v1"


async def test_refusal_ends_the_call_with_the_approved_farewell(
    runtime: CallRuntime, provider: FakeProvider, journal: FakeJournal
) -> None:
    await answer(runtime)
    await runtime.on_speech(speech("нет, не хочу", 10))

    assert provider.media_for(str(ATTEMPT))[-1] == "end_no_consent_v1"
    assert provider.hungup == [str(ATTEMPT)]
    assert journal.interviews[ATTEMPT].outcome is Outcome.NO_CONSENT


async def test_finished_call_is_saved_with_its_recording(
    runtime: CallRuntime, journal: FakeJournal
) -> None:
    await answer(runtime)
    await runtime.on_speech(speech("нет", 10))

    respondent, robot = journal.recordings[ATTEMPT]
    assert respondent is not None and "respondent" in respondent
    assert robot is not None and "robot" in robot
    assert runtime.active == 0


async def test_request_for_a_human_is_transferred(
    runtime: CallRuntime, provider: FakeProvider
) -> None:
    """ТЗ §9.2: респондент должен иметь возможность запросить оператора."""
    await answer(runtime)
    await runtime.on_speech(speech("позовите человека", 10))
    assert provider.transferred == [(str(ATTEMPT), "PJSIP/operator1")]


async def test_without_an_operator_the_call_ends_honestly(
    provider: FakeProvider, journal: FakeJournal, core: Questionnaire
) -> None:
    """Молчать в трубку, когда оператора нет, нельзя."""
    runtime = CallRuntime(
        provider, core, journal, operator_endpoint=None, clock=FakeClock(step_sec=2.0)
    )
    runtime.register(ATTEMPT, CallHandle(str(ATTEMPT), "chan-1"))
    await answer(runtime)
    await runtime.on_speech(speech("оператора", 10))

    assert provider.transferred == []
    assert provider.hungup == [str(ATTEMPT)]
    assert CallEvent.NO_OPERATOR in journal.events_for(ATTEMPT)


async def test_dont_call_request_reaches_the_journal(
    runtime: CallRuntime, journal: FakeJournal
) -> None:
    """ТЗ §6.2: немедленно в стоп-лист, действующий на все волны."""
    await answer(runtime)
    await runtime.on_speech(speech("больше не звоните мне", 10))
    assert journal.interviews[ATTEMPT].outcome is Outcome.DONT_CALL_REQUEST


# ---------------------------------------------------------------- перебивание


async def test_nothing_is_interrupted_when_the_robot_is_silent(
    runtime: CallRuntime, provider: FakeProvider
) -> None:
    """Гипотеза пришла, когда реплика уже доиграла: обрывать нечего."""
    await answer(runtime)
    await runtime.on_speech(speech("да я согласен", 10, final=False))
    assert provider.stopped == []


async def test_cough_during_the_intro_does_not_cut_it_off(
    provider: FakeProvider, journal: FakeJournal, core: Questionnaire
) -> None:
    """Мгновенная реакция на любой звук обрывает приветствие на кашле,
    и это читается как поломка."""
    runtime = CallRuntime(
        provider, core, journal,
        policy=RuntimePolicy(barge_in_delay_sec=3.0),
        clock=FakeClock(step_sec=2.0),
    )
    live = runtime.register(ATTEMPT, CallHandle(str(ATTEMPT), "chan-1"))
    await answer(runtime)

    live.prompts_pending = 1
    live.speaking_since = START
    await runtime.on_speech(speech("кхм", 1, final=False))
    assert provider.stopped == []


async def test_speaking_over_the_robot_after_the_delay_interrupts_it(
    provider: FakeProvider, journal: FakeJournal, core: Questionnaire
) -> None:
    runtime = CallRuntime(
        provider, core, journal,
        policy=RuntimePolicy(barge_in_delay_sec=3.0),
        clock=FakeClock(step_sec=2.0),
    )
    live = runtime.register(ATTEMPT, CallHandle(str(ATTEMPT), "chan-1"))
    await answer(runtime)

    live.prompts_pending = 1
    live.speaking_since = START
    await runtime.on_speech(speech("да я вас понял, согласен", 5, final=False))
    assert provider.stopped == [str(ATTEMPT)]


async def test_silence_hypothesis_never_interrupts(
    provider: FakeProvider, journal: FakeJournal, core: Questionnaire
) -> None:
    runtime = CallRuntime(provider, core, journal, clock=FakeClock(step_sec=2.0))
    live = runtime.register(ATTEMPT, CallHandle(str(ATTEMPT), "chan-1"))
    await answer(runtime)
    live.prompts_pending = 1
    live.speaking_since = START
    await runtime.on_speech(speech(None, 30, final=False, silence=True))
    assert provider.stopped == []


# ------------------------------------------------------------------- тишина


async def test_silence_timeout_makes_the_robot_repeat(
    runtime: CallRuntime, provider: FakeProvider
) -> None:
    """ТЗ §8.1: переспросить не более одного раза."""
    await answer(runtime)
    before = len(provider.played)
    await runtime.on_tick(START + timedelta(seconds=60))
    assert provider.media_for(str(ATTEMPT))[-1] == "consent_v1"
    assert len(provider.played) > before


async def test_tick_before_the_timeout_does_nothing(
    runtime: CallRuntime, provider: FakeProvider
) -> None:
    await answer(runtime)
    before = len(provider.played)
    await runtime.on_tick(START + timedelta(seconds=2))
    assert len(provider.played) == before


async def test_low_confidence_is_treated_as_silence(
    provider: FakeProvider, journal: FakeJournal, core: Questionnaire
) -> None:
    """Записать догадку как ответ хуже, чем переспросить: §5.3 бракует
    интервью по систематически нераспознанной речи, а не по переспросам."""
    runtime = CallRuntime(
        provider, core, journal,
        policy=RuntimePolicy(min_confidence=0.6), clock=FakeClock(step_sec=2.0),
    )
    runtime.register(ATTEMPT, CallHandle(str(ATTEMPT), "chan-1"))
    await answer(runtime)

    unsure = RecognizedSpeech(
        attempt_id=str(ATTEMPT), text="да", at=START + timedelta(seconds=10),
        confidence=0.2,
    )
    await runtime.on_speech(unsure)
    assert provider.media_for(str(ATTEMPT))[-1] == "consent_v1"  # переспросил


# ------------------------------------------------------------- обрыв снаружи


async def test_hangup_mid_interview_saves_what_was_answered(
    runtime: CallRuntime, journal: FakeJournal
) -> None:
    """Прерванный разговор — это данные, а не мусор."""
    await answer(runtime)
    await runtime.on_speech(speech("да", 10))
    await runtime.on_speech(speech("да", 20))
    await runtime.on_call_event(event(CallEvent.HANGUP, 30))

    session = journal.interviews[ATTEMPT]
    assert session.outcome is Outcome.PARTIAL
    assert session.answers["consent"].code == 1
    assert runtime.active == 0


async def test_technical_failure_also_saves_the_interview(
    runtime: CallRuntime, journal: FakeJournal
) -> None:
    await answer(runtime)
    await runtime.on_speech(speech("да", 10))
    await runtime.on_call_event(event(CallEvent.TECH_ERROR, 25))
    assert ATTEMPT in journal.interviews


# ------------------------------------------------------------------ подача


@pytest.fixture
def delivered(
    provider: FakeProvider, journal: FakeJournal, core: Questionnaire
) -> tuple[CallRuntime, list[float]]:
    """Рантайм с подачей и подменённым сном: паузы записываются, а не ждутся."""
    from socmon.voice.delivery import Catalogue

    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    engine = CallRuntime(
        provider, core, journal,
        policy=RuntimePolicy(barge_in_delay_sec=3.0, silence_timeout_sec=6.0,
                             nudge_after_sec=3.5),
        operator_endpoint="PJSIP/operator1",
        clock=FakeClock(step_sec=1.0),
        catalogue=Catalogue.from_yaml(),
        sleep=fake_sleep,
    )
    engine.register(ATTEMPT, CallHandle(str(ATTEMPT), "chan-1"))
    return engine, slept


async def test_open_answer_is_thanked_before_the_next_question(
    delivered: tuple[CallRuntime, list[float]], provider: FakeProvider
) -> None:
    """После развёрнутого ответа — отклик, пауза, и только потом вопрос."""
    from socmon.voice.delivery import Catalogue, Pool

    runtime, slept = delivered
    await answer(runtime)
    for reply in ("да", "да", "да"):  # согласие и два фильтра
        await runtime.on_speech(speech(reply, 10))
    await runtime.on_speech(speech("шесть", 12))       # q_life
    await runtime.on_speech(speech("не изменилось", 14))
    for reply in ("четыре", "семь", "восемь", "три", "четыре", "шесть", "пять"):
        await runtime.on_speech(speech(reply, 16))
    await runtime.on_speech(speech("цены", 18))
    await runtime.on_speech(speech("скорее да", 20))
    played_before = len(provider.played)
    await runtime.on_speech(speech("доверяю нашему блогеру Иванову", 22))  # открытый

    fresh = [m for _, m in provider.played[played_before:]]
    thanks = {p.key for p in Catalogue.from_yaml().pools[Pool.ACK_OPEN]}
    assert fresh[0] in thanks, fresh
    assert fresh[-1] == "q_open_worry_v1"
    # Паузы выдержаны: перед откликом и после него.
    assert len(slept) >= 2 and all(0.2 <= s <= 0.9 for s in slept[-2:])


async def test_silence_is_nudged_before_the_question_repeats(
    delivered: tuple[CallRuntime, list[float]], provider: FakeProvider
) -> None:
    from socmon.voice.delivery import Catalogue, Pool

    runtime, _ = delivered
    await answer(runtime)
    live = runtime.call_of(ATTEMPT)
    assert live is not None and live.awaiting_since is not None
    nudges = {p.key for p in Catalogue.from_yaml().pools[Pool.NUDGE]}

    await runtime.on_tick(live.awaiting_since + timedelta(seconds=2))
    assert not any(m in nudges for _, m in provider.played)

    await runtime.on_tick(live.awaiting_since + timedelta(seconds=4))
    assert any(m in nudges for _, m in provider.played)
    count = len(provider.played)
    # Второй оклик не следует: один раз на вопрос.
    await runtime.on_tick(live.awaiting_since + timedelta(seconds=5))
    assert len(provider.played) == count

    # Тайм-аут: движок повторяет вопрос, подача предваряет его извинением.
    await runtime.on_tick(live.awaiting_since + timedelta(seconds=7))
    unheard = {p.key for p in Catalogue.from_yaml().pools[Pool.REPEAT_UNHEARD]}
    tail = [m for _, m in provider.played[count:]]
    assert tail[0] in unheard and tail[-1] == "consent_v1"


async def test_barge_in_during_a_pause_cancels_the_rest(
    delivered: tuple[CallRuntime, list[float]], provider: FakeProvider
) -> None:
    """Перебили во время паузы перед вопросом — вопрос не звучит."""
    runtime, _ = delivered
    live = runtime.call_of(ATTEMPT)
    assert live is not None
    await answer(runtime)
    # Имитируем перебивание, сработавшее внутри паузы: сон обнуляет очередь.

    async def interrupting_sleep(seconds: float) -> None:
        live.prompts_pending = 0

    runtime._sleep = interrupting_sleep
    before = len(provider.played)
    await runtime.on_speech(speech("да", 10))
    # Согласие принято, но следующий вопрос не проигран.
    assert len(provider.played) == before


async def test_without_a_catalogue_the_runtime_behaves_as_before(
    runtime: CallRuntime, provider: FakeProvider
) -> None:
    await answer(runtime)
    await runtime.on_speech(speech("да", 10))
    keys = [m for _, m in provider.played]
    assert all(not k.startswith(("ack_", "rep_", "nudge_")) for k in keys)
