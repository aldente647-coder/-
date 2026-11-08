"""Реализация телефонии на Asterisk через ARI.

Почему ARI, а не AGI или дилплан: нам нужен внешний управляющий контур.
Логика диалога живёт в движке анкеты (`socmon.core.engine`), а Asterisk
исполняет команды — играет реплику, слушает, переводит. Держать граф анкеты
в дилплане нельзя: он версионируется вместе с анкетой и меняется каждую волну.

Разделение записи сделано снуп-каналами: один со `spy=in` пишет респондента,
второй со `spy=out` — робота. Смешанная запись тоже сохраняется. ТЗ §10.2
просит «раздельные аудиоканалы либо иной механизм однозначного определения
говорящего», и снупы дают именно первое: диаризация становится тривиальной,
а на наложенной речи не остаётся главного источника ошибок транскрипции.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import AsyncIterator, Callable
from contextlib import suppress
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlencode

import httpx
import websockets

from socmon.telephony.machine import CallEvent
from socmon.telephony.provider import (
    CallHandle,
    Capabilities,
    OriginateRequest,
    ProviderEvent,
    Recording,
    TelephonyError,
)

#: Переменная канала, в которой едет наш идентификатор попытки.
ATTEMPT_VARIABLE = "SOCMON_ATTEMPT"

#: Коды разъединения ISDN (Q.850), которые Asterisk отдаёт в ChannelDestroyed.
#: Разложены по нашим событиям: недозвон, занято и технический сбой — разные
#: вещи, и регламент повторов (ТЗ §6.2) обращается с ними по-разному.
HANGUP_CAUSES: dict[int, CallEvent] = {
    1: CallEvent.INVALID_NUMBER,   # номер не распределён
    2: CallEvent.INVALID_NUMBER,   # нет маршрута до сети
    3: CallEvent.INVALID_NUMBER,   # нет маршрута до абонента
    16: CallEvent.HANGUP,          # нормальное завершение
    17: CallEvent.BUSY,
    18: CallEvent.NO_ANSWER,       # абонент не отвечает
    19: CallEvent.NO_ANSWER,       # нет ответа после сигнала
    20: CallEvent.NO_ANSWER,       # абонент недоступен
    21: CallEvent.NO_ANSWER,       # вызов отклонён
    22: CallEvent.INVALID_NUMBER,  # номер изменён
    27: CallEvent.INVALID_NUMBER,  # направление вне обслуживания
    28: CallEvent.INVALID_NUMBER,  # неверный формат номера
    34: CallEvent.TECH_ERROR,      # нет свободных каналов
    38: CallEvent.TECH_ERROR,      # сеть неисправна
    42: CallEvent.TECH_ERROR,      # перегрузка оборудования
}

#: Итоги набора из события Dial.
DIAL_STATUSES: dict[str, CallEvent] = {
    "BUSY": CallEvent.BUSY,
    "NOANSWER": CallEvent.NO_ANSWER,
    "CANCEL": CallEvent.NO_ANSWER,
    "CONGESTION": CallEvent.TECH_ERROR,
    "CHANUNAVAIL": CallEvent.TECH_ERROR,
}


DEFAULT_BASE_URL = "http://127.0.0.1:8088"
DEFAULT_USER = "socmon"
DEFAULT_APP = "socmon"
DEFAULT_ENDPOINT = "PJSIP/{number}@trunk"
DEFAULT_OUTBOUND_CONTEXT = "socmon-outbound"


@dataclass(frozen=True, slots=True)
class AsteriskConfig:
    base_url: str = DEFAULT_BASE_URL
    username: str = DEFAULT_USER
    password: str = field(default="", repr=False)
    app: str = DEFAULT_APP
    endpoint_template: str = DEFAULT_ENDPOINT
    """Шаблон точки назначения. Транк задаётся конфигурацией, а не кодом:
    оператор связи может смениться вместе с юрлицом-инициатором вызовов."""

    outbound_context: str = DEFAULT_OUTBOUND_CONTEXT
    """Контекст дилплана, куда попадает отвеченный канал.

    Звонок заводится через дилплан, а не напрямую в Stasis: детектор
    автоответчика AMD — приложение дилплана, и при прямом входе в Stasis
    он просто не выполнится. Тогда робот начнёт зачитывать анкету
    голосовой почте, прожигая минуты связи (ТЗ §6.3).
    """

    recording_format: str = "wav"
    originate_timeout_sec: int = 30

    @classmethod
    def from_env(cls) -> AsteriskConfig:
        try:
            password = os.environ["SOCMON_ARI_PASSWORD"]
        except KeyError as exc:
            raise TelephonyError(
                "не задана переменная окружения SOCMON_ARI_PASSWORD"
            ) from exc
        return cls(
            base_url=os.environ.get("SOCMON_ARI_URL", DEFAULT_BASE_URL),
            username=os.environ.get("SOCMON_ARI_USER", DEFAULT_USER),
            password=password,
            app=os.environ.get("SOCMON_ARI_APP", DEFAULT_APP),
            endpoint_template=os.environ.get("SOCMON_ARI_ENDPOINT", DEFAULT_ENDPOINT),
            outbound_context=os.environ.get(
                "SOCMON_ARI_CONTEXT", DEFAULT_OUTBOUND_CONTEXT
            ),
        )

    def endpoint_for(self, number: str) -> str:
        if not re.fullmatch(r"\+?[0-9]{3,15}", number):
            raise TelephonyError("назначение должно быть телефонным номером из 3–15 цифр")
        return self.endpoint_template.format(number=number.lstrip("+"))

    @property
    def websocket_url(self) -> str:
        scheme = "wss" if self.base_url.startswith("https") else "ws"
        host = self.base_url.split("://", 1)[1]
        return (
            f"{scheme}://{host.rstrip('/')}/ari/events?"
            + urlencode({"app": self.app})
        )

    @property
    def websocket_headers(self) -> dict[str, str]:
        # Пароль в query string попадает в access log прокси и исключения.
        request = httpx.Request("GET", self.base_url)
        auth = httpx.BasicAuth(self.username, self.password)
        authorized = next(auth.auth_flow(request))
        return {"Authorization": authorized.headers["Authorization"]}


# ------------------------------------------------------------------- перевод


def translate(
    payload: dict[str, Any],
    attempt_for_channel: Callable[[str], str | None],
) -> ProviderEvent | None:
    """Перевести событие ARI на язык нашей машины состояний.

    Чистая функция: всё, что зависит от состояния клиента, приходит
    отдельным поиском по каналу. Благодаря этому перевод проверяется
    на записанных событиях, без живого Asterisk.

    События, не относящиеся к жизненному циклу звонка, отбрасываются —
    возвращается None, а не выдуманный переход.
    """
    kind = payload.get("type")
    channel = payload.get("channel") or {}
    channel_id = channel.get("id", "")
    attempt_id = _attempt_id(payload, attempt_for_channel)
    if attempt_id is None:
        return None

    at = _timestamp(payload)

    def event(name: CallEvent, detail: str | None = None) -> ProviderEvent:
        return ProviderEvent(
            attempt_id=attempt_id,
            event=name,
            at=at,
            provider_call_id=channel_id or None,
            event_id=payload.get("application_uuid") or f"{kind}:{channel_id}:{at.isoformat()}",
            detail=detail,
        )

    match kind:
        case "ChannelStateChange" if channel.get("state") == "Ringing":
            return event(CallEvent.RING)

        case "StasisStart":
            # Канал дошёл до нашего приложения — значит, кто-то снял трубку.
            # Отличить человека от автоответчика на этом шаге уже можно:
            # AMD отрабатывает в дилплане до Stasis и кладёт вердикт
            # в переменную канала.
            if _amd_verdict(payload) == "MACHINE":
                return event(CallEvent.ANSWERING_MACHINE, "AMD: автоответчик")
            return event(CallEvent.ANSWER)

        case "Dial":
            status = str(payload.get("dialstatus") or "")
            if status in DIAL_STATUSES:
                return event(DIAL_STATUSES[status], f"dialstatus={status}")
            return None

        case "ChannelDestroyed":
            cause = int(payload.get("cause") or 0)
            mapped = HANGUP_CAUSES.get(cause, CallEvent.TECH_ERROR)
            return event(mapped, f"cause={cause}")

        case _:
            return None


def _attempt_id(
    payload: dict[str, Any], attempt_for_channel: Callable[[str], str | None]
) -> str | None:
    """Найти нашу попытку.

    Сначала переменная канала — она едет вместе с событием и переживает
    перезапуск управляющего процесса. Реестр каналов в памяти — запасной путь.
    """
    channel = payload.get("channel") or {}
    variables = channel.get("channelvars") or {}
    if variables.get(ATTEMPT_VARIABLE):
        return str(variables[ATTEMPT_VARIABLE])
    args = payload.get("args") or []
    if args:
        return str(args[0])
    return attempt_for_channel(channel.get("id", ""))


def _amd_verdict(payload: dict[str, Any]) -> str | None:
    channel = payload.get("channel") or {}
    variables = channel.get("channelvars") or {}
    verdict = variables.get("AMDSTATUS")
    if verdict:
        return str(verdict)
    args = payload.get("args") or []
    return str(args[1]) if len(args) > 1 else None


def _timestamp(payload: dict[str, Any]) -> datetime:
    raw = payload.get("timestamp")
    if not raw:
        return datetime.now(UTC)
    try:
        parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return datetime.now(UTC)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


# -------------------------------------------------------------------- клиент


class AsteriskProvider:
    """Движок телефонии поверх Asterisk ARI."""

    def __init__(self, config: AsteriskConfig, client: httpx.AsyncClient | None = None) -> None:
        self._config = config
        self._client = client or httpx.AsyncClient(
            base_url=f"{config.base_url}/ari",
            auth=(config.username, config.password),
            timeout=httpx.Timeout(10.0),
        )
        self._channels: dict[str, str] = {}
        self._snoops: dict[str, list[str]] = {}
        self._bridges: dict[str, str] = {}
        self._audio_bridges: dict[str, str] = {}
        self._playbacks: dict[str, set[str]] = {}
        self._operators: dict[str, str] = {}
        self._operator_answered: set[str] = set()

    @property
    def config(self) -> AsteriskConfig:
        return self._config

    def capabilities(self) -> Capabilities:
        return Capabilities(
            separate_recording_channels=True,
            answering_machine_detection=True,
            barge_in=True,
            operator_transfer=True,
            caller_id_rotation=True,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------ управление

    async def originate(self, request: OriginateRequest) -> CallHandle:
        """Начать звонок.

        Идентификатор попытки уходит переменной канала, а дилплан передаёт
        его же аргументом в Stasis. Дублирование намеренное: переменная
        переживает перезапуск управляющего процесса, аргумент приходит
        в первом же событии — вместе они не дают потерять связь события
        с базой.
        """
        payload = {
            "endpoint": self._config.endpoint_for(request.destination),
            # Через дилплан, а не сразу в Stasis: см. `outbound_context`.
            "context": self._config.outbound_context,
            "extension": "s",
            "priority": 1,
            "callerId": request.caller_id,
            "timeout": request.timeout_sec,
            "variables": {
                ATTEMPT_VARIABLE: request.attempt_id,
                **request.variables,
            },
        }
        data = await self._post("/channels", json=payload)
        channel_id = str(data["id"])
        self._channels[channel_id] = request.attempt_id
        return CallHandle(attempt_id=request.attempt_id, provider_call_id=channel_id)

    async def play(self, call: CallHandle, media: str) -> str | None:
        """Поставить реплику в очередь воспроизведения канала.

        Реплики именно ставятся в очередь, а не перебивают друг друга:
        вступление и первый вопрос звучат подряд, как одна речь.
        """
        data = await self._post(
            f"/channels/{call.provider_call_id}/play",
            params={"media": f"sound:{media}"},
        )
        playback_id = str(data["id"]) if data.get("id") else None
        if playback_id:
            self._playbacks.setdefault(call.attempt_id, set()).add(playback_id)
        return playback_id

    def playback_finished(self, call: CallHandle, playback_id: str) -> None:
        self._playbacks.get(call.attempt_id, set()).discard(playback_id)

    async def stop_playback(self, call: CallHandle) -> None:
        """Оборвать реплику робота: респондент перебил (ТЗ §8.1)."""
        for playback_id in self._playbacks.pop(call.attempt_id, set()):
            await self._delete(f"/playbacks/{playback_id}", missing_ok=True)

    async def hangup(self, call: CallHandle) -> None:
        try:
            await self._delete(f"/channels/{call.provider_call_id}", missing_ok=True)
        finally:
            self._channels.pop(call.provider_call_id, None)
            self._playbacks.pop(call.attempt_id, None)
            self._operators.pop(call.attempt_id, None)
            self._operator_answered.discard(call.attempt_id)
            resources = [f"/channels/{channel}" for channel in
                         self._snoops.pop(call.attempt_id, [])]
            for bridges in (self._audio_bridges, self._bridges):
                bridge = bridges.pop(call.attempt_id, None)
                if bridge:
                    resources.append(f"/bridges/{bridge}")
            results = await asyncio.gather(
                *(self._delete(path, missing_ok=True) for path in resources),
                return_exceptions=True,
            )
            for result in results:
                if isinstance(result, Exception):
                    raise result

    # -------------------------------------------------------------- запись

    async def start_recording(self, call: CallHandle) -> None:
        """Начать запись двумя снуп-каналами плюс общий микс.

        `spy=in` слышит респондента, `spy=out` — то, что говорит робот.
        Раздельные дорожки нужны не для красоты: на них диаризация
        тривиальна, а транскрипция не путает говорящих на перебиваниях.
        """
        self._snoops.setdefault(call.attempt_id, [])
        for direction, side in (("in", "respondent"), ("out", "robot")):
            snoop = await self._post(
                f"/channels/{call.provider_call_id}/snoop",
                params={"spy": direction, "app": self._config.app},
            )
            snoop_id = str(snoop["id"])
            self._snoops[call.attempt_id].append(snoop_id)
            await self._post(
                f"/channels/{snoop_id}/record",
                params={
                    "name": f"{call.attempt_id}-{side}",
                    "format": self._config.recording_format,
                    "ifExists": "overwrite",
                },
            )
        await self._post(
            f"/channels/{call.provider_call_id}/record",
            params={
                "name": f"{call.attempt_id}-mixed",
                "format": self._config.recording_format,
                "ifExists": "overwrite",
            },
        )

    async def recording(self, call: CallHandle) -> Recording:
        suffix = self._config.recording_format
        return Recording(
            respondent_uri=f"recording:{call.attempt_id}-respondent.{suffix}",
            robot_uri=f"recording:{call.attempt_id}-robot.{suffix}",
            mixed_uri=f"recording:{call.attempt_id}-mixed.{suffix}",
        )

    async def attach_audio(
        self, call: CallHandle, *, host: str, port: int, stream_id: str
    ) -> str:
        """Подключить к каналу поток звука для распознавания.

        Asterisk сам приходит к нам TCP-клиентом и первым кадром называет
        `stream_id` — по нему звук связывается с попыткой дозвона. Формат
        slin16 на 8 кГц: ровно тот, в котором работает распознавание,
        поэтому перекодирования на пути нет ни одного.
        """
        if call.attempt_id in self._audio_bridges:
            return self._audio_bridges[call.attempt_id]
        bridge = await self._post("/bridges", params={"type": "mixing,proxy_media"})
        bridge_id = str(bridge["id"])
        self._audio_bridges[call.attempt_id] = bridge_id
        # В ASR уходит только входящая дорожка; голос робота не распознаётся.
        snoop = await self._post(
            f"/channels/{call.provider_call_id}/snoop",
            params={"spy": "in", "app": self._config.app},
        )
        snoop_id = str(snoop["id"])
        self._snoops.setdefault(call.attempt_id, []).append(snoop_id)
        data = await self._post(
            "/channels/externalMedia",
            params={
                "app": self._config.app,
                "external_host": f"{host}:{port}",
                "encapsulation": "audiosocket",
                "transport": "tcp",
                "connection_type": "client",
                "format": "slin",
                "direction": "both",
                "data": stream_id,
            },
        )
        media_id = str(data["id"])
        self._snoops[call.attempt_id].append(media_id)
        await self._post(
            f"/bridges/{bridge_id}/addChannel",
            params={"channel": f"{snoop_id},{media_id}"},
        )
        return media_id

    # ------------------------------------------------------------- перевод

    async def transfer_to_operator(self, call: CallHandle, endpoint: str) -> None:
        """Соединить респондента с оператором через мост.

        Мост, а не переадресация: при переадресации мы теряем канал из-под
        контроля и вместе с ним запись, события и возможность закрыть
        зависший звонок.
        """
        bridge = await self._post("/bridges", params={"type": "mixing"})
        bridge_id = str(bridge["id"])
        self._bridges[call.attempt_id] = bridge_id

        await self._post(
            f"/bridges/{bridge_id}/addChannel",
            params={"channel": call.provider_call_id},
        )
        operator = await self._post(
            "/channels",
            json={
                "endpoint": endpoint,
                "app": self._config.app,
                "appArgs": call.attempt_id,
                "timeout": 30,
                "variables": {ATTEMPT_VARIABLE: call.attempt_id},
            },
        )
        operator_id = str(operator["id"])
        self._operators[call.attempt_id] = operator_id
        self._snoops.setdefault(call.attempt_id, []).append(operator_id)

    async def event_for_call(
        self, payload: dict[str, Any], call: CallHandle,
    ) -> ProviderEvent | None:
        """Отфильтровать служебные каналы и подключить ответившего оператора."""
        channel_id = str((payload.get("channel") or {}).get("id", ""))
        operator_id = self._operators.get(call.attempt_id)
        if channel_id not in {call.provider_call_id, operator_id}:
            return None
        event = translate(payload, lambda _: call.attempt_id)
        if event is None:
            return None
        event = replace(event, attempt_id=call.attempt_id)
        if channel_id == operator_id:
            if event.event is CallEvent.ANSWER:
                if call.attempt_id not in self._operator_answered:
                    await self._post(
                        f"/bridges/{self._bridges[call.attempt_id]}/addChannel",
                        params={"channel": operator_id},
                    )
                    self._operator_answered.add(call.attempt_id)
                return replace(event, event=CallEvent.OPERATOR_ANSWER)
            if event.event in {
                CallEvent.HANGUP, CallEvent.NO_ANSWER, CallEvent.BUSY, CallEvent.TECH_ERROR,
            }:
                kind = (CallEvent.HANGUP if call.attempt_id in self._operator_answered
                        else CallEvent.NO_OPERATOR)
                return replace(event, event=kind)
            return None
        return event

    # -------------------------------------------------------------- события

    async def events(self) -> AsyncIterator[ProviderEvent]:
        """Поток событий движка.

        Соединение переустанавливается само: разрыв websocket не должен
        останавливать волну. Пока связи нет, зависшие звонки закрывает
        сторож — он работает по времени, а не по событиям, и именно
        поэтому нужен отдельно.
        """
        while True:
            try:
                async with websockets.connect(
                    self._config.websocket_url,
                    additional_headers=self._config.websocket_headers,
                ) as socket:
                    async for raw in socket:
                        payload = json.loads(raw)
                        translated = translate(payload, self._channels.get)
                        self._remember(payload)
                        if translated is not None:
                            yield translated
            except (OSError, websockets.WebSocketException):
                await asyncio.sleep(3)

    def _remember(self, payload: dict[str, Any]) -> None:
        channel = payload.get("channel") or {}
        channel_id = channel.get("id")
        if payload.get("type") == "ChannelDestroyed" and channel_id:
            with suppress(KeyError):
                del self._channels[str(channel_id)]

    # -------------------------------------------------------------- запросы

    async def _post(self, path: str, **kwargs: Any) -> dict[str, Any]:
        return await self._request("POST", path, **kwargs)

    async def _delete(
        self, path: str, *, missing_ok: bool = False, **kwargs: Any,
    ) -> dict[str, Any]:
        return await self._request("DELETE", path, missing_ok=missing_ok, **kwargs)

    async def _request(
        self, method: str, path: str, *, missing_ok: bool = False, **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            response = await self._client.request(method, path, **kwargs)
            if missing_ok and response.status_code == 404:
                return {}
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise TelephonyError(
                f"{method} {path}: Asterisk ответил {exc.response.status_code}"
            ) from exc
        except httpx.HTTPError as exc:
            raise TelephonyError(f"{method} {path}: ошибка соединения с АТС") from exc

        if not response.content:
            return {}
        parsed = response.json()
        return parsed if isinstance(parsed, dict) else {"result": parsed}
