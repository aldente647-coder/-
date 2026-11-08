"""Публичная демонстрация опроса.

Страница, на которой любой человек может послушать робота и пройти анкету
сам — без учётной записи и без телефона. Реплики звучат те же самые файлы,
что робот проигрывает в звонке, и логика перехода по вопросам та же.

Смысл не в развлечении: Заказчику нужно утвердить голос и формулировки
(ТЗ §8.3, §7.1), а по описанию этого не сделать — надо услышать.

Демонстрация ничего не пишет в базу. Ответы живут в памяти процесса и
исчезают через час: это витрина, а не сбор данных.
"""

from __future__ import annotations

import secrets
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from socmon.core.codes import SpecialCode
from socmon.core.engine import Action, Answer, InterviewSession, Turn, Utterance
from socmon.core.questionnaire import (
    ConsentNode,
    MultiNode,
    Node,
    OpenNode,
    Questionnaire,
    ScaleNode,
    ScreenerNode,
    SingleNode,
)
from socmon.voice.delivery import (
    Catalogue,
    Delivery,
    Gender,
    Kind,
    Segment,
    repeat_reason,
    take_media,
)
from socmon.voice.render import VoiceManifest

SESSION_TTL_SEC = 3600
MAX_SESSIONS = 500


@dataclass(frozen=True, slots=True)
class Choice:
    """Кнопка ответа на экране."""

    label: str
    value: str


@dataclass(frozen=True, slots=True)
class Line:
    """Одно воспроизведение на странице демо."""

    text: str
    media: str
    """Имя файла без расширения — с учётом дубля."""

    gap_ms: int
    kind: str
    """prompt, ack, preface — чтобы отклик рисовался иначе, чем вопрос."""


@dataclass(frozen=True, slots=True)
class Step:
    """Что показать человеку прямо сейчас."""

    prompts: tuple[tuple[str, str], ...]
    """Пары «что сказать» и «какой файл проиграть»."""

    node_id: str | None
    choices: tuple[Choice, ...]
    free_text: bool
    finished: bool
    outcome: str | None
    asked: int
    """Сколько вопросов робот успел задать. Не то же самое, что число
    вопросов в анкете: часть отсеивается фильтрами и переходами — если
    человек не стал смотреть новости реже, вопроса «почему реже» не будет."""

    answered: int
    """На сколько из них человек ответил. Открытые считаются наравне
    с закрытыми: там нет кода, но ответ есть.

    Раньше здесь были только закодированные ответы, а рядом стояло общее
    число вопросов анкеты. Числитель и знаменатель считались по разным
    правилам, и на безупречно пройденном опросе счётчик всё равно
    показывал недобор — ровно на число открытых вопросов."""

    unrecognized: int
    """Ответы, которых робот не разобрал. В демонстрации это самое
    интересное число: оно показывает, где грамматика вопроса слаба."""
    heard: str | None = None
    understood: str | None = None
    lines: tuple[Line, ...] = ()
    """То же с подачей: отклики, паузы, дубли. Демо играет именно это —
    как в настоящем звонке, а не реплики впритык."""


@dataclass
class _Session:
    interview: InterviewSession
    turn: Turn
    touched_at: float = field(default_factory=time.monotonic)
    heard: str | None = None
    understood: str | None = None
    delivery: Delivery | None = None
    last_answer: Answer | None = None


class DemoStudio:
    """Хранит идущие демонстрации.

    В памяти, а не в базе: демонстрация — витрина, и складывать ответы
    случайных посетителей рядом с массивом волны нельзя ни при каких
    обстоятельствах (ТЗ §5.3).
    """

    def __init__(
        self,
        questionnaire: Questionnaire,
        *,
        catalogue: Catalogue | None = None,
        gender: Gender = "f",
        library: VoiceManifest | None = None,
        directory: Path | None = None,
    ) -> None:
        self._questionnaire = questionnaire
        self._catalogue = catalogue
        self._gender = gender
        self._library = library
        self._directory = directory
        self._sessions: dict[str, _Session] = {}

    @property
    def questionnaire(self) -> Questionnaire:
        return self._questionnaire

    def start(self) -> tuple[str, Step]:
        self._evict()
        token = secrets.token_urlsafe(16)
        interview = InterviewSession(self._questionnaire)
        turn = interview.begin()
        delivery = None
        if self._catalogue is not None:
            delivery = Delivery(
                self._catalogue, self._questionnaire, gender=self._gender,
                seed=token, takes=self._max_takes(),
            )
        self._sessions[token] = _Session(interview=interview, turn=turn, delivery=delivery)
        return token, self._step(self._sessions[token])

    def answer(self, token: str, reply: str) -> Step | None:
        if len(reply) > 4000:
            return None
        session = self._sessions.get(token)
        if session is None:
            return None
        if time.monotonic() - session.touched_at > SESSION_TTL_SEC:
            self._sessions.pop(token, None)
            return None
        session.touched_at = time.monotonic()

        before = dict(session.interview.answers)
        utterance = Utterance.said(reply) if reply.strip() else Utterance.quiet()
        session.turn = session.interview.submit(utterance)
        session.heard = reply.strip() or None
        session.understood = self._understood(session, before)
        fresh = [a for k, a in session.interview.answers.items() if k not in before]
        session.last_answer = fresh[-1] if fresh else None

        if session.interview.finished:
            step = self._step(session)
            self._sessions.pop(token, None)
            return step
        return self._step(session)

    def _understood(self, session: _Session, before: Mapping[str, Answer]) -> str | None:
        """Что робот записал в массив после этого ответа.

        Показываем именно код, а не пересказ: это то самое, что попадёт
        в данные, и Заказчику полезно увидеть связь между сказанным
        и записанным.
        """
        fresh = [
            answer for node_id, answer in session.interview.answers.items()
            if node_id not in before
        ]
        if not fresh:
            return None
        answer = fresh[-1]
        if answer.code is None:
            return "записан текст ответа"
        node = self._questionnaire.node(answer.node_id)
        if len(answer.codes) > 1:
            named = ", ".join(
                f"{code} — {_label_for(node, code)}".rstrip(" —")
                for code in answer.codes
            )
            return f"коды {named}"
        label = _label_for(node, answer.code)
        return f"код {answer.code}" + (f" — {label}" if label else "")

    def _step(self, session: _Session) -> Step:
        interview = session.interview
        turn = session.turn
        prompts = tuple(
            (" ".join(p.text.split()), p.audio_key or p.node_id)
            for p in turn.prompts if p.text.strip()
        )
        node = (
            self._questionnaire.node(turn.awaiting) if turn.awaiting else None
        )
        return Step(
            prompts=prompts,
            lines=self._lines(session),
            node_id=turn.awaiting,
            choices=_choices(node) if node else (),
            free_text=isinstance(node, OpenNode | ScaleNode),
            finished=turn.action in (Action.END, Action.TRANSFER),
            outcome=interview.outcome.value if interview.outcome else None,
            asked=sum(
                1 for a in interview.answers.values()
                if a.code != SpecialCode.NOT_ASKED
            ),
            answered=sum(
                1 for a in interview.answers.values()
                if a.code not in (SpecialCode.NOT_ASKED, SpecialCode.UNRECOGNIZED)
                and (a.code is not None or a.text)
            ),
            unrecognized=sum(
                1 for a in interview.answers.values()
                if a.code == SpecialCode.UNRECOGNIZED
            ),
            heard=session.heard,
            understood=session.understood,
        )

    def _lines(self, session: _Session) -> tuple[Line, ...]:
        """Строки подачи для страницы: с откликами, паузами и дублями."""
        turn = session.turn
        if session.delivery is None:
            return tuple(
                Line(" ".join(p.text.split()), p.audio_key or p.node_id, 0, "prompt")
                for p in turn.prompts if p.text.strip()
            )
        segments = session.delivery.plan(
            turn, previous=session.last_answer,
            reason=repeat_reason(session.interview.events),
        )
        lines: list[Line] = []
        for segment in segments:
            if segment.kind is not Kind.PROMPT and not self._has_media(segment.key):
                continue
            lines.append(Line(
                text=segment.text, media=self._media(segment),
                gap_ms=segment.gap_ms, kind=segment.kind.value,
            ))
        return tuple(lines)

    def _has_media(self, key: str) -> bool:
        return self._library is None or key in self._library.entries

    def _media(self, segment: Segment) -> str:
        entry = self._library.entries.get(segment.key) if self._library else None
        takes = entry.takes if entry is not None else ()
        return take_media(segment.key, segment.take, takes, self._directory)

    def _max_takes(self) -> int:
        if self._library is None:
            return 1
        return max((len(e.takes) + 1 for e in self._library.entries.values()), default=1)

    def _evict(self) -> None:
        """Убрать заброшенные демонстрации.

        Без этого память процесса растёт от каждого посетителя, который
        открыл страницу и закрыл вкладку.
        """
        now = time.monotonic()
        stale = [
            token for token, s in self._sessions.items()
            if now - s.touched_at > SESSION_TTL_SEC
        ]
        for token in stale:
            self._sessions.pop(token, None)
        while len(self._sessions) >= MAX_SESSIONS:
            oldest = min(self._sessions, key=lambda t: self._sessions[t].touched_at)
            self._sessions.pop(oldest, None)


def _choices(node: Node) -> tuple[Choice, ...]:
    """Кнопки для узла. Слова берём те же, что понимает робот в звонке."""
    if isinstance(node, ConsentNode):
        return (
            Choice(node.accept.label, node.accept.phrases[0] if node.accept.phrases else "да"),
            Choice(node.decline.label, node.decline.phrases[0] if node.decline.phrases else "нет"),
        )
    if isinstance(node, SingleNode | ScreenerNode):
        return tuple(
            Choice(option.label, option.phrases[0] if option.phrases else option.label)
            for option in node.options
        )
    if isinstance(node, ScaleNode):
        return tuple(
            Choice(str(value), str(value)) for value in range(node.low, node.high + 1)
        )
    return ()


def _label_for(node: Node, code: int) -> str:
    for special in SpecialCode:
        if code == special:
            return {
                SpecialCode.DONT_KNOW: "затрудняюсь ответить",
                SpecialCode.REFUSED: "отказ отвечать",
                SpecialCode.UNRECOGNIZED: "не разобрал",
                SpecialCode.NO_ANSWER: "нет ответа",
                SpecialCode.NOT_ASKED: "не задавался",
            }[special]
    if isinstance(node, ConsentNode):
        return node.accept.label if code == node.accept.code else node.decline.label
    if isinstance(node, SingleNode | MultiNode | ScreenerNode):
        for option in node.options:
            if option.code == code:
                return option.label
    return ""
