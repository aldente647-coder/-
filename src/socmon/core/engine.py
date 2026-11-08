"""Детерминированный движок интервью.

ТЗ §8: полностью свободный генеративный агент не допускается. Ход разговора
определяется исключительно графом анкеты и грамматиками узлов; языковая модель
в звонке не участвует вовсе. Всё, что движок не смог разобрать, становится
явным кодом, а не догадкой.

Движок не знает ни про телефонию, ни про базу: на вход — реплика, на выход —
что произнести дальше. Благодаря этому интервью проигрывается в тестах целиком,
без единого звонка.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

from socmon.core.codes import Outcome, SpecialCode
from socmon.core.grammar import (
    ControlIntent,
    MatchKind,
    OptionGrammar,
    match_control,
    match_multi,
    match_options,
)
from socmon.core.questionnaire import (
    ConsentNode,
    EndNode,
    InfoNode,
    MultiNode,
    Node,
    OpenNode,
    Questionnaire,
    ScaleNode,
    ScreenerNode,
    SingleNode,
)
from socmon.core.text import parse_number

Clock = Callable[[], datetime]


def _utcnow() -> datetime:
    return datetime.now(UTC)


# ---------------------------------------------------------------- грамматики

#: Затруднение и отказ принимаются на любом закрытом вопросе.
#: Отказ от ответа на отдельный вопрос не бракует интервью (ТЗ §7.6).
_DONT_KNOW = OptionGrammar.build(
    SpecialCode.DONT_KNOW,
    (
        "затрудняюсь", "затрудняюсь ответить", "не знаю", "трудно сказать",
        "сложно сказать", "затрудняюсь сказать",
    ),
)
_REFUSED = OptionGrammar.build(
    SpecialCode.REFUSED,
    (
        "не буду отвечать", "не хочу отвечать", "не хочу говорить",
        "пропустим", "пропустите", "следующий вопрос", "без комментариев",
        "это личное", "не ваше дело", "не скажу", "не буду говорить",
        "не отвечу", "секрет", "воздержусь",
    ),
)
_SPECIAL = (_REFUSED, _DONT_KNOW)


# -------------------------------------------------------------------- модели


class Action(StrEnum):
    ASK = "ask"
    REPEAT = "repeat"
    TRANSFER = "transfer"
    END = "end"


@dataclass(frozen=True, slots=True)
class Utterance:
    """Реплика респондента, как её увидел движок.

    `text is None` означает тишину: истёк таймаут ожидания. Для логики это то же
    самое, что нераспознанная речь, но в журнал попадает отдельным событием.
    """

    text: str | None = None
    silence: bool = False
    hangup: bool = False

    @staticmethod
    def said(text: str) -> Utterance:
        return Utterance(text=text)

    @staticmethod
    def quiet() -> Utterance:
        return Utterance(silence=True)

    @staticmethod
    def dropped() -> Utterance:
        return Utterance(hangup=True)


@dataclass(frozen=True, slots=True)
class Prompt:
    node_id: str
    text: str
    audio_key: str | None = None


@dataclass(frozen=True, slots=True)
class Turn:
    """Что робот делает после реплики респондента."""

    action: Action
    prompts: tuple[Prompt, ...] = ()
    awaiting: str | None = None
    outcome: Outcome | None = None


@dataclass(frozen=True, slots=True)
class Answer:
    """Одна ячейка будущего массива данных."""

    node_id: str
    core_key: str | None
    code: int | None
    text: str | None
    attempts: int
    at: datetime
    codes: tuple[int, ...] = ()
    """Все отмеченные варианты. У обычного вопроса здесь ровно один код —
    тот же, что в `code`; у вопроса «до трёх ответов» их столько, сколько
    назвал респондент, в порядке называния.

    `code` при этом остаётся первым из них. Так вопрос с множественным
    выбором не ломает ни одну выгрузку и ни один расчёт, написанные до его
    появления: они видят главный ответ, а полный набор берут те, кому он нужен."""

    @property
    def is_substantive(self) -> bool:
        return self.code is not None and self.code <= SpecialCode.NOT_ASKED - 1


@dataclass(frozen=True, slots=True)
class Event:
    """Запись в шкале времени интервью (ТЗ §10.1)."""

    at: datetime
    node_id: str
    kind: str
    detail: str | None = None


class InterviewError(RuntimeError):
    pass


# -------------------------------------------------------------------- движок


@dataclass
class InterviewSession:
    """Одно интервью. Экземпляр живёт ровно один звонок."""

    questionnaire: Questionnaire
    clock: Clock = _utcnow

    answers: dict[str, Answer] = field(default_factory=dict)
    events: list[Event] = field(default_factory=list)
    outcome: Outcome | None = None
    consented: bool = False

    _current: str | None = None
    _repeats: int = 0
    _clarifications: int = 0
    _hedged: bool = False
    """На этом узле уже уточняли после уклончивого ответа. Второй уклончивый
    ответ — это «затрудняюсь», а не «не разобрал»: человек ответил, просто
    не выбрал."""
    _unrecognized_streak: int = 0
    _started_at: datetime | None = None
    _finished_at: datetime | None = None

    # ------------------------------------------------------------- жизненный цикл

    def begin(self) -> Turn:
        if self._started_at is not None:
            raise InterviewError("интервью уже начато")
        self._started_at = self.clock()
        self._log(self.questionnaire.entry_id, "interview_started")
        return self._advance_to(self.questionnaire.entry_id)

    @property
    def finished(self) -> bool:
        return self.outcome is not None

    @property
    def current_node_id(self) -> str | None:
        """Узел, ответа на который ждёт движок."""
        return self._current

    def abort(self, outcome: Outcome) -> None:
        """Закрыть интервью, оборванное снаружи: связь пропала, респондент
        бросил трубку, сторож закрыл зависший звонок.

        Прощальная реплика не проигрывается — некому. Ответы, данные до
        обрыва, сохраняются: это данные, а не мусор. Валидность решается
        правилами §5.3 отдельно, и частичное интервью может оказаться
        как валидным, так и нет.
        """
        if self.finished:
            return
        if self._started_at is None:
            raise InterviewError("интервью не начиналось")
        self.outcome = outcome
        self._finished_at = self.clock()
        self._current = None
        self._log("", "interview_aborted", outcome.value)

    @property
    def duration_sec(self) -> float:
        if self._started_at is None:
            return 0.0
        end = self._finished_at or self.clock()
        return (end - self._started_at).total_seconds()

    @property
    def within_target_duration(self) -> bool:
        """ТЗ §3.3: не менее 90% завершённых интервью должны укладываться
        в лимит. Флаг считается по каждому интервью, доля — по волне."""
        return self.duration_sec <= self.questionnaire.policy.target_duration_sec

    # ------------------------------------------------------------------- ход

    def submit(self, utterance: Utterance) -> Turn:
        if self.finished:
            raise InterviewError("интервью уже завершено")
        if self._current is None:
            raise InterviewError("не вызван begin()")

        node = self.questionnaire.node(self._current)

        if utterance.hangup:
            self._log(node.id, "hangup")
            return self._terminate_via(
                Outcome.REFUSAL_DURING if self.consented else Outcome.REFUSAL_BEFORE
            )

        if utterance.text is not None:
            control = match_control(utterance.text)
            if control.kind is MatchKind.CONTROL:
                assert control.intent is not None
                self._log(node.id, "control_intent", control.intent.value)
                handled = self._handle_control(control.intent, node)
                if handled is not None:
                    return handled

        return self._answer(node, utterance)

    # -------------------------------------------------- управляющие намерения

    def _handle_control(self, intent: ControlIntent, node: Node) -> Turn | None:
        match intent:
            case ControlIntent.DONT_CALL:
                return self._terminate_via(Outcome.DONT_CALL_REQUEST)
            case ControlIntent.CALLBACK:
                return self._terminate_via(Outcome.CALLBACK_REQUEST)
            case ControlIntent.OPERATOR:
                return self._terminate_via(Outcome.TRANSFERRED_TO_OPERATOR)
            case ControlIntent.HANGUP:
                return self._terminate_via(
                    Outcome.REFUSAL_DURING if self.consented else Outcome.REFUSAL_BEFORE
                )
            case ControlIntent.REPEAT:
                # Просьба повторить — законное право респондента, а не промах
                # распознавания, поэтому она не тратит бюджет переспросов.
                # Лимит нужен лишь на случай, когда «повторите» слышится в цикле.
                if self._clarifications >= 2:
                    return None
                self._clarifications += 1
                return self._repeat(node, "repeat_requested")
            case ControlIntent.WHO:
                # «А вы кто?» — объяснить и вернуться к вопросу. Тоже право,
                # а не сбой; бюджет переспросов не тратит.
                if self._clarifications >= 2:
                    return None
                self._clarifications += 1
                return self._repeat(node, "who_asked")

    # --------------------------------------------------------------- ответы

    def _answer(self, node: Node, utterance: Utterance) -> Turn:
        match node:
            case OpenNode():
                return self._answer_open(node, utterance)
            case ConsentNode():
                return self._answer_consent(node, utterance)
            case ScreenerNode():
                return self._answer_screener(node, utterance)
            case SingleNode():
                return self._answer_single(node, utterance)
            case MultiNode():
                return self._answer_multi(node, utterance)
            case ScaleNode():
                return self._answer_scale(node, utterance)
            case _:
                raise InterviewError(f"узел {node.id} не ожидает ответа")

    def _answer_open(self, node: OpenNode, utterance: Utterance) -> Turn:
        """Открытый ответ пишем как есть.

        Никаких уточнений и переспросов: смысловая обработка — ночным батчем
        (ТЗ §7.5, §11.2). Тишина здесь — валидный «нет, никого не назову».
        """
        text = None if utterance.silence else utterance.text
        self._record(node, code=None, text=text, attempts=1)
        return self._advance_to(node.next)

    def _answer_consent(self, node: ConsentNode, utterance: Utterance) -> Turn:
        match, hedged = self._match_closed(
            utterance, (node.accept.grammar(), node.decline.grammar())
        )
        if match is None:
            return self._retry_or_give_up(node, on_exhausted=node.refuse_next, hedged=hedged)
        if match == SpecialCode.DONT_KNOW:
            # «Не знаю, может быть» на предложение участвовать — не отказ
            # и не согласие. Уточняем один раз; без явного «да» после этого
            # интервью не начинается (ТЗ §8.2).
            return self._retry_or_give_up(node, on_exhausted=node.refuse_next, hedged=True)
        if match == node.accept.code:
            self.consented = True
            self._record(node, code=match, text=utterance.text, attempts=self._repeats + 1)
            self._log(node.id, "consent_granted")
            return self._advance_to(node.next)
        self._record(node, code=match, text=utterance.text, attempts=self._repeats + 1)
        self._log(node.id, "consent_refused")
        return self._advance_to(node.refuse_next)

    def _answer_screener(self, node: ScreenerNode, utterance: Utterance) -> Turn:
        grammars = tuple(o.grammar() for o in node.options)
        match, hedged = self._match_closed(utterance, grammars)
        if match is None:
            return self._retry_or_give_up(node, on_exhausted=node.fail_next, hedged=hedged)
        self._record(node, code=match, text=utterance.text, attempts=self._repeats + 1)
        if match in node.pass_codes:
            return self._advance_to(node.next)
        self._log(node.id, "screened_out")
        return self._advance_to(node.fail_next)

    def _answer_single(self, node: SingleNode, utterance: Utterance) -> Turn:
        grammars = tuple(o.grammar() for o in node.options)
        match, hedged = self._match_closed(utterance, grammars)
        if match is None:
            return self._retry_or_unrecognized(node, hedged=hedged)
        self._record(node, code=match, text=utterance.text, attempts=self._repeats + 1)
        target = node.routes.get(match, node.next)
        return self._advance_to(target)

    def _answer_multi(self, node: MultiNode, utterance: Utterance) -> Turn:
        """Вопрос «до N ответов»: в одной реплике может быть названо несколько.

        Отказ и затруднение проверяются раньше вариантов и отменяют разбор
        целиком: «затрудняюсь ответить» — это один ответ, а не ноль.
        """
        if utterance.text is None:
            return self._retry_or_unrecognized(node)

        special = match_options(utterance.text, _SPECIAL)
        if special.kind is MatchKind.ANSWER:
            assert special.code is not None
            self._record(node, code=special.code, text=utterance.text,
                         attempts=self._repeats + 1)
            return self._advance_to(node.next)

        grammars = tuple(o.grammar() for o in node.options)
        codes = match_multi(utterance.text, grammars, node.max_answers)
        if not codes:
            hedged = match_options(utterance.text, grammars).kind is MatchKind.HEDGED
            return self._retry_or_unrecognized(node, hedged=hedged)

        exclusive = next((c for c in codes if c in node.exclusive_codes), None)
        if exclusive is not None and len(codes) > 1:
            self._log(node.id, "exclusive_option", str(exclusive))
            codes = (exclusive,)

        self._record(node, code=codes[0], codes=codes, text=utterance.text,
                     attempts=self._repeats + 1)
        return self._advance_to(node.next)

    def _answer_scale(self, node: ScaleNode, utterance: Utterance) -> Turn:
        if utterance.text is not None:
            special = match_options(utterance.text, _SPECIAL)
            if special.kind is MatchKind.ANSWER:
                assert special.code is not None
                self._record(node, code=special.code, text=utterance.text,
                             attempts=self._repeats + 1)
                return self._advance_to(node.next)
            value = parse_number(utterance.text, node.low, node.high)
            if value is not None:
                self._record(node, code=value, text=utterance.text, attempts=self._repeats + 1)
                return self._advance_to(node.next)
        return self._retry_or_unrecognized(node)

    def _match_closed(
        self, utterance: Utterance, grammars: tuple[OptionGrammar, ...]
    ) -> tuple[int | None, bool]:
        """Ступень грамматик: сначала отказ и затруднение, затем варианты узла.

        Порядок важен: «не знаю» не должно случайно пойматься вариантом,
        в фразах которого встретилось слово «не».

        Второе значение — уклончивость: человек ответил («средне», «не очень»,
        «и да и нет»), но не выбрал. Это не молчание и не шум — уточняем.
        """
        if utterance.text is None:
            return None, False
        special = match_options(utterance.text, _SPECIAL)
        if special.kind is MatchKind.ANSWER:
            return special.code, False
        result = match_options(utterance.text, grammars)
        if result.kind is MatchKind.ANSWER:
            return result.code, False
        return None, result.kind in (MatchKind.HEDGED, MatchKind.AMBIGUOUS)

    # ------------------------------------------------------ переспрос и сдача

    def _retry_or_unrecognized(self, node: Node, *, hedged: bool = False) -> Turn:
        """Не разобрали — переспрашиваем ровно один раз (ТЗ §8.1), потом
        фиксируем код нераспознанной речи и идём дальше.

        Уклончивый ответ — отдельный случай: переспрос идёт с уточнением
        «и всё же, если выбирать», а если человек и после этого не выбрал,
        записывается «затрудняюсь» — он ответил, просто не выбрал."""
        if self._repeats < self.questionnaire.policy.max_repeats:
            self._repeats += 1
            self._hedged = hedged
            return self._repeat(node, "hedged" if hedged else "not_recognized")

        if hedged or self._hedged:
            self._record(node, code=SpecialCode.DONT_KNOW, text=None, attempts=self._repeats + 1)
            self._log(node.id, "hedged_twice")
            return self._advance_to(node.next)

        self._record(node, code=SpecialCode.UNRECOGNIZED, text=None, attempts=self._repeats + 1)
        self._unrecognized_streak += 1
        if self._unrecognized_streak >= self.questionnaire.policy.max_unrecognized_before_abort:
            self._log(node.id, "speech_unusable")
            return self._terminate_via(Outcome.SPEECH_UNUSABLE)
        return self._advance_to(node.next)

    def _retry_or_give_up(
        self, node: Node, *, on_exhausted: str, hedged: bool = False
    ) -> Turn:
        """Для согласия и фильтров нераспознанный ответ нельзя трактовать
        как «да»: ТЗ §8.2 прямо запрещает считать отсутствие ответа согласием."""
        if self._repeats < self.questionnaire.policy.max_repeats:
            self._repeats += 1
            return self._repeat(node, "hedged" if hedged else "not_recognized")
        self._record(node, code=SpecialCode.UNRECOGNIZED, text=None, attempts=self._repeats + 1)
        self._log(node.id, "no_explicit_consent")
        return self._advance_to(on_exhausted)

    def _repeat(self, node: Node, reason: str) -> Turn:
        self._log(node.id, "repeat", reason)
        return Turn(
            action=Action.REPEAT,
            prompts=(Prompt(node.id, node.text, node.audio_key),),
            awaiting=node.id,
        )

    # ------------------------------------------------------------- продвижение

    def _advance_to(self, node_id: str | None) -> Turn:
        """Проиграть подряд все реплики без вопроса и остановиться там,
        где нужен ответ респондента."""
        if node_id is None:
            raise InterviewError("некуда переходить: у узла не задан next")

        prompts: list[Prompt] = []
        current = node_id
        seen: set[str] = set()

        while True:
            if current in seen:
                raise InterviewError(f"цикл из реплик без вопроса на узле {current}")
            seen.add(current)

            node = self.questionnaire.node(current)
            prompts.append(Prompt(node.id, node.text, node.audio_key))

            if isinstance(node, EndNode):
                return self._finish(node, tuple(prompts))
            if isinstance(node, InfoNode):
                self._log(node.id, "info")
                if node.next is None:
                    raise InterviewError(f"{node.id}: у реплики не задан next")
                current = node.next
                continue

            if not self._filter_passes(node):
                self._log(node.id, "filtered_out", node.ask_if.core_key if node.ask_if else None)
                self._record(node, code=SpecialCode.NOT_ASKED, text=None, attempts=0)
                prompts.pop()
                if node.next is None:
                    raise InterviewError(f"{node.id}: отфильтрованному узлу некуда идти")
                current = node.next
                continue

            self._current = node.id
            self._repeats = 0
            self._clarifications = 0
            self._hedged = False
            self._log(node.id, "asked")
            return Turn(action=Action.ASK, prompts=tuple(prompts), awaiting=node.id)

    def _filter_passes(self, node: Node) -> bool:
        """Проверить фильтр узла по уже полученным ответам."""
        if node.ask_if is None:
            return True
        for answer in self.answers.values():
            if answer.core_key == node.ask_if.core_key:
                return node.ask_if.holds_for(answer.code)
        return False

    def _terminate_via(self, outcome: Outcome) -> Turn:
        """Уйти на утверждённое прощание, соответствующее исходу."""
        target = self.questionnaire.terminals.get(outcome)
        if target is None:
            raise InterviewError(f"нет утверждённого завершения для исхода {outcome.value}")
        return self._advance_to(target)

    def _finish(self, node: EndNode, prompts: tuple[Prompt, ...]) -> Turn:
        self._current = None
        self.outcome = self._effective_outcome(node.outcome)
        self._finished_at = self.clock()
        self._log(node.id, "interview_finished", self.outcome.value)
        action = Action.TRANSFER if self.outcome is Outcome.TRANSFERRED_TO_OPERATOR else Action.END
        return Turn(action=action, prompts=prompts, outcome=self.outcome)

    def _effective_outcome(self, declared: Outcome) -> Outcome:
        """Интервью считается завершённым только если респондент дошёл до конца
        после согласия. Ранний выход по фильтру завершением не является."""
        if declared is Outcome.COMPLETED and not self.consented:
            return Outcome.PARTIAL
        return declared

    # ---------------------------------------------------------------- журнал

    def _record(
        self,
        node: Node,
        *,
        code: int | None,
        text: str | None,
        attempts: int,
        codes: tuple[int, ...] | None = None,
    ) -> None:
        self.answers[node.id] = Answer(
            node_id=node.id,
            core_key=node.core_key,
            code=code,
            text=text,
            attempts=attempts,
            at=self.clock(),
            codes=codes if codes is not None else (() if code is None else (code,)),
        )
        if code is not None and code != SpecialCode.UNRECOGNIZED:
            self._unrecognized_streak = 0
        self._log(node.id, "answered", None if code is None else str(code))

    def _log(self, node_id: str, kind: str, detail: str | None = None) -> None:
        self.events.append(Event(at=self.clock(), node_id=node_id, kind=kind, detail=detail))
