"""Перевод событий ARI и построение запросов к Asterisk.

Живой Asterisk не нужен: перевод — чистая функция, а запросы проверяются
подставным транспортом. Это то, что можно проверить до аренды транка.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from socmon.telephony.asterisk import (
    ATTEMPT_VARIABLE,
    AsteriskConfig,
    AsteriskProvider,
    translate,
)
from socmon.telephony.machine import CallEvent
from socmon.telephony.provider import CallHandle, OriginateRequest, TelephonyError

ATTEMPT = "6f1c2a10-0000-4000-8000-000000000001"
CHANNEL = "1755700000.42"


def ari(kind: str, **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": kind,
        "timestamp": "2026-08-20T18:30:00.000+0000",
        "channel": {
            "id": CHANNEL,
            "state": extra.pop("state", "Up"),
            "channelvars": {ATTEMPT_VARIABLE: ATTEMPT, **extra.pop("vars", {})},
        },
    }
    payload.update(extra)
    return payload


def by_channel(channel_id: str) -> str | None:
    return ATTEMPT if channel_id == CHANNEL else None


# ------------------------------------------------------------------- перевод


def test_ringing_becomes_a_ring_event() -> None:
    event = translate(ari("ChannelStateChange", state="Ringing"), by_channel)
    assert event is not None
    assert event.event is CallEvent.RING
    assert event.attempt_id == ATTEMPT


def test_entering_the_application_means_someone_picked_up() -> None:
    event = translate(ari("StasisStart"), by_channel)
    assert event is not None and event.event is CallEvent.ANSWER


def test_amd_verdict_short_circuits_the_call() -> None:
    """AMD отрабатывает в дилплане до Stasis и кладёт вердикт в переменную
    канала. Разговаривать с автоответчиком — прожигать минуты впустую."""
    event = translate(ari("StasisStart", vars={"AMDSTATUS": "MACHINE"}), by_channel)
    assert event is not None
    assert event.event is CallEvent.ANSWERING_MACHINE
    assert event.detail is not None and "автоответчик" in event.detail


def test_human_verdict_is_a_normal_answer() -> None:
    event = translate(ari("StasisStart", vars={"AMDSTATUS": "HUMAN"}), by_channel)
    assert event is not None and event.event is CallEvent.ANSWER


@pytest.mark.parametrize(
    ("cause", "expected"),
    [
        (1, CallEvent.INVALID_NUMBER),
        (16, CallEvent.HANGUP),
        (17, CallEvent.BUSY),
        (19, CallEvent.NO_ANSWER),
        (21, CallEvent.NO_ANSWER),
        (34, CallEvent.TECH_ERROR),
        (42, CallEvent.TECH_ERROR),
    ],
)
def test_hangup_causes_are_split_by_meaning(cause: int, expected: CallEvent) -> None:
    """Недозвон, занято и технический сбой — разные вещи: регламент повторов
    обращается с ними по-разному (ТЗ §6.2)."""
    event = translate(ari("ChannelDestroyed", cause=cause), by_channel)
    assert event is not None and event.event is expected


def test_unknown_cause_is_a_technical_failure_not_a_no_answer() -> None:
    """Неизвестную причину безопаснее считать поломкой: тогда номер
    не будет перебираться впустую, а сбой станет заметен."""
    event = translate(ari("ChannelDestroyed", cause=99), by_channel)
    assert event is not None and event.event is CallEvent.TECH_ERROR


@pytest.mark.parametrize(
    ("status", "expected"),
    [("BUSY", CallEvent.BUSY), ("NOANSWER", CallEvent.NO_ANSWER),
     ("CONGESTION", CallEvent.TECH_ERROR)],
)
def test_dial_status_is_translated(status: str, expected: CallEvent) -> None:
    event = translate(ari("Dial", dialstatus=status), by_channel)
    assert event is not None and event.event is expected


def test_irrelevant_events_are_dropped_not_guessed() -> None:
    assert translate(ari("PlaybackStarted"), by_channel) is None
    assert translate(ari("Dial", dialstatus="ANSWER"), by_channel) is None
    assert translate(ari("ChannelStateChange", state="Down"), by_channel) is None


def test_event_without_a_known_attempt_is_dropped() -> None:
    """Чужой канал в том же приложении не должен превратиться в переход
    по нашему звонку."""
    payload = ari("StasisStart")
    payload["channel"]["id"] = "someone-else"
    payload["channel"]["channelvars"] = {}
    assert translate(payload, by_channel) is None


def test_attempt_id_survives_without_the_channel_registry() -> None:
    """Переменная канала едет вместе с событием и переживает перезапуск
    управляющего процесса — реестр в памяти лишь запасной путь."""
    event = translate(ari("StasisStart"), lambda _: None)
    assert event is not None and event.attempt_id == ATTEMPT


def test_app_args_carry_the_attempt_when_variables_are_absent() -> None:
    payload = ari("StasisStart")
    payload["channel"]["channelvars"] = {}
    payload["args"] = [ATTEMPT]
    event = translate(payload, lambda _: None)
    assert event is not None and event.attempt_id == ATTEMPT


def test_timestamp_is_parsed_with_a_timezone() -> None:
    event = translate(ari("StasisStart"), by_channel)
    assert event is not None and event.at.tzinfo is not None


# ------------------------------------------------------------------- запросы


class Recorder:
    """Подставной транспорт: записывает запросы и отдаёт заготовленные ответы."""

    def __init__(self, responses: dict[str, dict[str, Any]] | None = None) -> None:
        self.calls: list[httpx.Request] = []
        self._responses = responses or {}

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        body = self._responses.get(request.url.path, {"id": "chan-1"})
        return httpx.Response(200, json=body)

    def paths(self) -> list[str]:
        return [r.url.path for r in self.calls]


def provider(recorder: Recorder) -> AsteriskProvider:
    client = httpx.AsyncClient(
        base_url="http://asterisk/ari",
        transport=httpx.MockTransport(recorder.handler),
    )
    return AsteriskProvider(AsteriskConfig(password="secret"), client=client)


@pytest.mark.asyncio
async def test_originate_goes_through_the_dialplan_not_straight_to_stasis() -> None:
    """AMD — приложение дилплана. При прямом входе в Stasis он не выполнится,
    и робот начнёт зачитывать анкету голосовой почте."""
    recorder = Recorder()
    handle = await provider(recorder).originate(OriginateRequest(
        attempt_id=ATTEMPT, destination="+79491234567", caller_id="+74951234567",
    ))

    body = json.loads(recorder.calls[0].content)
    assert body["context"] == "socmon-outbound"
    assert body["extension"] == "s"
    assert "app" not in body
    assert body["variables"][ATTEMPT_VARIABLE] == ATTEMPT
    assert body["endpoint"] == "PJSIP/79491234567@trunk"
    assert body["callerId"] == "+74951234567"
    assert handle.provider_call_id == "chan-1"


def test_endpoint_template_is_configuration_not_code() -> None:
    """Оператор связи может смениться вместе с юрлицом-инициатором вызовов
    (RESEARCH.md §4) — точка назначения обязана быть настройкой."""
    config = AsteriskConfig(endpoint_template="SIP/{number}@backup")
    assert config.endpoint_for("+79491234567") == "SIP/79491234567@backup"


@pytest.mark.asyncio
async def test_recording_uses_two_snoop_channels() -> None:
    """ТЗ §10.2: раздельные каналы «робот» и «респондент». spy=in слышит
    респондента, spy=out — робота."""
    recorder = Recorder()
    await provider(recorder).start_recording(
        CallHandle(attempt_id=ATTEMPT, provider_call_id=CHANNEL)
    )
    spies = [
        r.url.params.get("spy") for r in recorder.calls if r.url.path.endswith("/snoop")
    ]
    assert spies == ["in", "out"]

    names = [
        r.url.params.get("name") for r in recorder.calls if r.url.path.endswith("/record")
    ]
    assert names == [f"{ATTEMPT}-respondent", f"{ATTEMPT}-robot", f"{ATTEMPT}-mixed"]


@pytest.mark.asyncio
async def test_recording_reports_separated_channels() -> None:
    recording = await provider(Recorder()).recording(
        CallHandle(attempt_id=ATTEMPT, provider_call_id=CHANNEL)
    )
    assert recording.channels_separated
    assert recording.mixed_uri is not None


@pytest.mark.asyncio
async def test_transfer_bridges_instead_of_redirecting() -> None:
    """При переадресации канал уходит из-под контроля, а с ним запись,
    события и возможность закрыть зависший звонок."""
    recorder = Recorder({
        "/ari/bridges": {"id": "bridge-1"}, "/ari/channels": {"id": "op-1"},
    })
    await provider(recorder).transfer_to_operator(
        CallHandle(attempt_id=ATTEMPT, provider_call_id=CHANNEL), "PJSIP/operator7"
    )
    assert recorder.paths() == [
        "/ari/bridges", "/ari/bridges/bridge-1/addChannel", "/ari/channels",
    ]


@pytest.mark.asyncio
async def test_audio_stream_is_attached_without_transcoding() -> None:
    """slin на 8 кГц — тот же формат, в котором озвучена анкета и работает
    распознавание. Ни одного перекодирования на пути."""
    recorder = Recorder()
    await provider(recorder).attach_audio(
        CallHandle(ATTEMPT, CHANNEL), host="127.0.0.1", port=8090, stream_id="s-1",
    )
    request = next(r for r in recorder.calls if r.url.path.endswith("/externalMedia"))
    assert request.url.path == "/ari/channels/externalMedia"
    assert request.url.params.get("encapsulation") == "audiosocket"
    assert request.url.params.get("format") == "slin"
    assert request.url.params.get("data") == "s-1"
    assert request.url.params.get("external_host") == "127.0.0.1:8090"


@pytest.mark.asyncio
async def test_barge_in_stops_the_prompt() -> None:
    recorder = Recorder()
    engine = provider(recorder)
    call = CallHandle(attempt_id=ATTEMPT, provider_call_id=CHANNEL)
    playback_id = await engine.play(call, "intro_v1")
    await engine.stop_playback(call)
    assert recorder.calls[-1].method == "DELETE"
    assert recorder.calls[-1].url.path == f"/ari/playbacks/{playback_id}"


@pytest.mark.asyncio
async def test_asterisk_errors_are_wrapped_with_context() -> None:
    def failing(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="Channel not found")

    client = httpx.AsyncClient(
        base_url="http://asterisk/ari", transport=httpx.MockTransport(failing)
    )
    engine = AsteriskProvider(AsteriskConfig(password="x"), client=client)
    with pytest.raises(TelephonyError, match="404"):
        await engine.play(CallHandle(ATTEMPT, CHANNEL), "intro_v1")


def test_password_must_come_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SOCMON_ARI_PASSWORD", raising=False)
    with pytest.raises(TelephonyError, match="SOCMON_ARI_PASSWORD"):
        AsteriskConfig.from_env()


def test_websocket_url_matches_the_scheme() -> None:
    assert AsteriskConfig(password="p").websocket_url.startswith("ws://")
    assert AsteriskConfig(
        base_url="https://pbx.example", password="p"
    ).websocket_url.startswith("wss://")


@pytest.mark.parametrize("destination", ["123&456", "PJSIP/user", "123@other", "123\n"])
def test_destination_cannot_inject_dialplan_syntax(destination: str) -> None:
    with pytest.raises(TelephonyError):
        AsteriskConfig().endpoint_for(destination)


async def test_auxiliary_channel_cannot_forge_main_call_event() -> None:
    engine = provider(Recorder())
    call = CallHandle(ATTEMPT, CHANNEL)
    event = await engine.event_for_call({
        "type": "ChannelDestroyed", "channel": {
            "id": "snoop-1", "channelvars": {ATTEMPT_VARIABLE: ATTEMPT},
        }, "args": [ATTEMPT], "cause": 16,
    }, call)
    assert event is None


async def test_operator_is_bridged_only_after_answer() -> None:
    recorder = Recorder({"/ari/bridges": {"id": "b1"}, "/ari/channels": {"id": "op1"}})
    engine = provider(recorder)
    call = CallHandle(ATTEMPT, CHANNEL)
    await engine.transfer_to_operator(call, "PJSIP/operator1")
    before = len(recorder.calls)
    event = await engine.event_for_call({
        "type": "StasisStart", "channel": {"id": "op1"}, "args": [ATTEMPT],
    }, call)
    assert event and event.event is CallEvent.OPERATOR_ANSWER
    assert len(recorder.calls) == before + 1
    assert recorder.calls[-1].url.params["channel"] == "op1"
