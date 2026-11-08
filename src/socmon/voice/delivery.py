"""Подача: что робот говорит и сколько молчит между вопросами анкеты.

Робота выдаёт не тембр, а поведение. Следующий вопрос вылетает в ту же
миллисекунду, как распознан ответ; между вопросами — ни «угу», ни вдоха;
переспрос — тот же файл слово в слово; каждый звонок звучит как предыдущий.
Живой интервьюер откликается на ответ, выдерживает разную паузу, а
переспрашивая, сначала извиняется.

Этот модуль решает ровно это и ничего больше. Он не трогает ни граф анкеты,
ни формулировки вопросов: служебные реплики берутся из утверждаемого
каталога, а вопросы произносятся дословно (ТЗ §8.1). Языковой модели здесь
нет — все решения принимает генератор случайных чисел с известным зерном,
поэтому подача любого звонка воспроизводима по его идентификатору.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from socmon.core.codes import SpecialCode
from socmon.core.engine import Action, Answer, Event, Turn
from socmon.core.questionnaire import (
    ConsentNode,
    InfoNode,
    Node,
    OpenNode,
    Questionnaire,
    ScreenerNode,
    Section,
)

CATALOGUE_PATH = Path(__file__).with_name("delivery.yaml")

Gender = Literal["f", "m"]

#: Род голоса — для форм вроде «расслышала»/«расслышал».
VOICE_GENDER: dict[str, Gender] = {
    "anna": "f", "baya": "f", "kseniya": "f", "xenia": "f",
    "denis": "m", "alex": "m", "aidar": "m", "eugene": "m",
}


class DeliveryError(ValueError):
    """Каталог служебных реплик не прошёл проверку."""


# ------------------------------------------------------------------ каталог


class _Base(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class GenderedText(_Base):
    f: str
    m: str


class Phrase(_Base):
    key: str
    text: str | GenderedText

    def spoken(self, gender: Gender) -> str:
        if isinstance(self.text, GenderedText):
            return self.text.f if gender == "f" else self.text.m
        return self.text


class Pool(StrEnum):
    ACK_CLOSED = "ack_closed"
    ACK_OPEN = "ack_open"
    ACK_SOFT = "ack_soft"
    REPEAT_ASKED = "repeat_asked"
    REPEAT_UNHEARD = "repeat_unheard"
    PROBE = "probe"
    WHO = "who"
    NUDGE = "nudge"


class Catalogue(_Base):
    """Служебные реплики. Утверждаются как анкета, версионируются как анкета."""

    name: str
    pools: dict[Pool, tuple[Phrase, ...]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> Self:
        missing = set(Pool) - set(self.pools)
        if missing:
            raise DeliveryError(f"нет пулов: {sorted(p.value for p in missing)}")
        keys: list[str] = []
        for pool, phrases in self.pools.items():
            if not phrases:
                raise DeliveryError(f"пул {pool.value} пуст")
            keys.extend(p.key for p in phrases)
        if len(keys) != len(set(keys)):
            raise DeliveryError("ключи реплик не уникальны")
        return self

    @property
    def version_id(self) -> str:
        canonical = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    def phrases(self) -> tuple[Phrase, ...]:
        return tuple(p for phrases in self.pools.values() for p in phrases)

    def phrase(self, key: str) -> Phrase:
        for candidate in self.phrases():
            if candidate.key == key:
                return candidate
        raise KeyError(key)

    @classmethod
    def from_yaml(cls, path: str | Path = CATALOGUE_PATH) -> Catalogue:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        try:
            return cls.model_validate(raw)
        except ValidationError as exc:
            raise DeliveryError(str(exc)) from exc


# ----------------------------------------------------------------- сегменты


class Kind(StrEnum):
    PROMPT = "prompt"
    """Реплика анкеты: вопрос, вступление, прощание."""

    ACK = "ack"
    """Отклик на ответ."""

    PREFACE = "preface"
    """Пояснение перед повтором вопроса."""

    NUDGE = "nudge"
    """Оклик в тишину."""


@dataclass(frozen=True, slots=True)
class Segment:
    """Одно воспроизведение: что играть и сколько молчать перед этим."""

    kind: Kind
    key: str
    gap_ms: int = 0
    take: int = 0
    """Номер дубля, 0 — основной. Реальный номер ограничится тем, сколько
    дублей есть в библиотеке; выбор здесь — намерение, а не факт."""

    text: str = ""
    """Для демонстрации и журнала. В звонке не используется."""


#: Паузы в миллисекундах: (от, до). Живой человек не отвечает мгновенно
#: и не выдерживает одинаковую паузу дважды.
GAPS: dict[str, tuple[int, int]] = {
    "before_ack_closed": (250, 500),
    "before_ack_open": (500, 900),
    "before_ack_soft": (300, 600),
    "after_ack": (200, 450),
    "before_prompt_plain": (350, 700),
    "before_screener": (300, 500),
    "between_prompts": (150, 350),
    "after_preface": (250, 400),
}

#: Вероятность отклика после закрытого ответа. Не единица: откликаться
#: на каждый ответ — тоже механика, только другая.
ACK_PROBABILITY: dict[Section, float] = {
    Section.CORE: 0.55,
    Section.VARIABLE: 0.55,
    Section.BLITZ: 0.35,
    Section.DEMOGRAPHY: 0.4,
    Section.INTRO: 0.0,
    Section.OUTRO: 0.0,
}

Reason = Literal["repeat_requested", "not_recognized", "hedged", "who_asked"]

#: Почему вопрос повторяется — и какой пул это объясняет респонденту.
REPEAT_POOLS: dict[str, Pool] = {
    "repeat_requested": Pool.REPEAT_ASKED,
    "not_recognized": Pool.REPEAT_UNHEARD,
    "hedged": Pool.PROBE,
    "who_asked": Pool.WHO,
}


def repeat_reason(events: Sequence[Event]) -> Reason | None:
    """Почему движок повторяет вопрос: по просьбе или потому что не разобрал.

    Берётся из журнала интервью, а не из хода: у хода есть только действие
    REPEAT, причину он не сообщает — ему она безразлична, подаче нет.
    """
    for event in reversed(events):
        if event.kind == "repeat":
            if event.detail in ("repeat_requested", "hedged", "who_asked"):
                return event.detail  # type: ignore[return-value]
            return "not_recognized"
        if event.kind in ("asked", "answered"):
            return None
    return None


class Delivery:
    """Подача одного разговора.

    Экземпляр живёт один звонок: помнит последний отклик, чтобы не повторять
    его дважды подряд, и держит свой генератор случайности с зерном
    из идентификатора звонка — воспроизводимость важнее непредсказуемости.
    """

    def __init__(
        self,
        catalogue: Catalogue,
        questionnaire: Questionnaire,
        *,
        gender: Gender = "f",
        seed: str = "",
        takes: int = 1,
    ) -> None:
        self._catalogue = catalogue
        self._questionnaire = questionnaire
        self._gender = gender
        self._takes = max(1, takes)
        self._rng = random.Random(hashlib.sha256(seed.encode("utf-8")).digest())
        self._last_ack: str | None = None

    # ---------------------------------------------------------------- план

    def plan(
        self,
        turn: Turn,
        *,
        previous: Answer | None = None,
        reason: Reason | None = None,
    ) -> tuple[Segment, ...]:
        """Что сыграть на этом ходу и с какими паузами."""
        segments: list[Segment] = []
        prompts = [p for p in turn.prompts if p.text.strip()]

        if turn.action is Action.REPEAT:
            pool = REPEAT_POOLS.get(reason or "", Pool.REPEAT_UNHEARD)
            segments.append(self._service(Kind.PREFACE, pool, self._gap("before_prompt_plain")))
            gap = self._gap("after_preface")
        else:
            ack = self._acknowledge(previous)
            if ack is not None:
                segments.append(ack)
                gap = self._gap("after_ack")
            elif previous is None:
                gap = 0
            elif self._is_gate(previous):
                gap = self._gap("before_screener")
            else:
                gap = self._gap("before_prompt_plain")

        for index, prompt in enumerate(prompts):
            segments.append(Segment(
                kind=Kind.PROMPT,
                key=prompt.audio_key or prompt.node_id,
                gap_ms=gap if index == 0 else self._gap("between_prompts"),
                take=self._rng.randrange(self._takes),
                text=" ".join(prompt.text.split()),
            ))
        return tuple(segments)

    def nudge(self) -> Segment:
        """Оклик в тишину."""
        return self._service(Kind.NUDGE, Pool.NUDGE, 0)

    # ------------------------------------------------------------ отклики

    def _acknowledge(self, previous: Answer | None) -> Segment | None:
        if previous is None:
            return None
        node = self._node(previous.node_id)
        if node is None or self._is_gate(previous):
            return None
        code = previous.code

        if code in (SpecialCode.REFUSED, SpecialCode.DONT_KNOW):
            return self._service(Kind.ACK, Pool.ACK_SOFT, self._gap("before_ack_soft"))
        if code in (SpecialCode.UNRECOGNIZED, SpecialCode.NOT_ASKED, SpecialCode.NO_ANSWER):
            return None
        if isinstance(node, OpenNode):
            if not previous.text:
                return None
            return self._service(Kind.ACK, Pool.ACK_OPEN, self._gap("before_ack_open"))
        if code is None:
            return None
        if self._rng.random() >= ACK_PROBABILITY.get(node.section, 0.0):
            return None
        return self._service(Kind.ACK, Pool.ACK_CLOSED, self._gap("before_ack_closed"))

    def _service(self, kind: Kind, pool: Pool, gap_ms: int) -> Segment:
        phrases = self._catalogue.pools[pool]
        candidates = [p for p in phrases if p.key != self._last_ack] or list(phrases)
        phrase = self._rng.choice(candidates)
        if kind is Kind.ACK:
            self._last_ack = phrase.key
        return Segment(
            kind=kind, key=phrase.key, gap_ms=gap_ms,
            take=self._rng.randrange(self._takes), text=phrase.spoken(self._gender),
        )

    # ------------------------------------------------------------ служебное

    def _gap(self, name: str) -> int:
        low, high = GAPS[name]
        return self._rng.randint(low, high)

    def _node(self, node_id: str) -> Node | None:
        try:
            return self._questionnaire.node(node_id)
        except KeyError:
            return None

    def _is_gate(self, answer: Answer) -> bool:
        """Согласие и фильтры — не вопросы, на которые откликаются «угу»."""
        node = self._node(answer.node_id)
        return isinstance(node, ConsentNode | ScreenerNode | InfoNode)


def take_media(key: str, take: int, takes: Sequence[str], directory: Path | None) -> str:
    """Имя записи для АТС (без расширения) по номеру дубля из плана подачи.

    Дубли берутся из списка манифеста, а не по номеру: после отсева
    провалившихся дублей список может быть «t3 без t2», и номер указал бы
    на файл, которого нет. Отсутствующие на диске дубли пропускаются;
    нет ни одного — играет основной."""
    present = [
        name for name in takes
        if directory is None or (directory / name).exists()
    ]
    options = [f"{key}.wav", *present]
    return options[take % len(options)].removesuffix(".wav")


def take_filename(key: str, take: int, available: int) -> str:
    """Имя файла дубля: основной — `key.wav`, остальные — `key-t2.wav`.

    Дефис, а не точка: Asterisk считает всё после точки расширением
    и файла `key.t2` не найдёт. Номер сверх имеющихся дублей сворачивается
    в имеющиеся — намерение подачи не обязано совпадать с библиотекой.
    """
    if available <= 1:
        return f"{key}.wav"
    index = take % available
    return f"{key}.wav" if index == 0 else f"{key}-t{index + 1}.wav"


def media_name(key: str, take: int, available: int) -> str:
    """То же без расширения — так реплику просит Asterisk (`sound:имя`)."""
    return take_filename(key, take, available).removesuffix(".wav")
