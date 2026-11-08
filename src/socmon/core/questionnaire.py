"""Анкета как версионируемый неизменяемый артефакт.

ТЗ §7.3 требует версионирования каждой анкеты, §3.1 — сопоставимости
постоянного ядра между волнами, §11.3 — запрета скрытого изменения
исторических данных. Отсюда конструкция: анкета грузится из YAML, валидируется
целиком и получает `version_id` — хеш собственного содержимого. Любая правка
формулировки даёт новый `version_id`, поэтому «та же анкета, но чуть-чуть
другая» технически невозможна.
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from socmon.core.codes import MAX_OPTION_CODE, Outcome, SpecialCode
from socmon.core.grammar import OptionGrammar


class QuestionnaireError(ValueError):
    """Анкета не прошла валидацию. Поднимается на загрузке, а не в звонке."""


#: Исходы, для которых анкета обязана содержать утверждённое завершение.
REQUIRED_TERMINALS: frozenset[Outcome] = frozenset(
    {
        Outcome.DONT_CALL_REQUEST,
        Outcome.CALLBACK_REQUEST,
        Outcome.REFUSAL_BEFORE,
        Outcome.REFUSAL_DURING,
        Outcome.TRANSFERRED_TO_OPERATOR,
        Outcome.SPEECH_UNUSABLE,
    }
)


class Section(StrEnum):
    """Части анкеты по ТЗ §7.1."""

    INTRO = "intro"
    CORE = "core"
    VARIABLE = "variable"
    BLITZ = "blitz"
    DEMOGRAPHY = "demography"
    OUTRO = "outro"


class _Base(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Option(_Base):
    """Вариант ответа. `code` попадает в массив данных и в кодбук."""

    code: Annotated[int, Field(ge=1, le=MAX_OPTION_CODE)]
    label: str
    phrases: tuple[str, ...] = ()
    patterns: tuple[str, ...] = ()

    def grammar(self) -> OptionGrammar:
        return OptionGrammar.build(self.code, self.phrases, self.patterns)


class AskIf(_Base):
    """Условие показа вопроса — фильтр анкеты.

    Анкета Заказчика фильтрует часть вопросов по уже полученным ответам:
    про детский сад спрашивают тех, у кого есть дошкольники, про планы
    на ребёнка — людей до сорока пяти. Без этого робот задаёт пенсионеру
    вопрос о планах завести ребёнка, и это не курьёз, а брошенная трубка.

    Незаданный вопрос не пропадает из массива: он получает код 95
    «не задавался». Пустая ячейка и ячейка «отфильтрован» — разные вещи,
    и при расчёте долей их нельзя смешивать.
    """

    core_key: str
    """Показатель, по которому проверяем. Именно core_key, а не id узла:
    условие переживает переименование узлов и смену версии анкеты."""

    codes: tuple[int, ...] = ()
    """Условие выполнено, если ответ — один из перечисленных кодов."""

    between: tuple[int, int] | None = None
    """Условие выполнено, если числовой ответ попал в отрезок включительно."""

    @model_validator(mode="after")
    def _check_one_form(self) -> Self:
        if bool(self.codes) == (self.between is not None):
            raise QuestionnaireError(
                f"ask_if по {self.core_key}: задайте либо codes, либо between"
            )
        if self.between is not None and self.between[0] > self.between[1]:
            raise QuestionnaireError(f"ask_if по {self.core_key}: пустой отрезок between")
        return self

    def holds_for(self, code: int | None) -> bool:
        """Выполнено ли условие при таком ответе на опорный вопрос.

        Отсутствие ответа, отказ и «затрудняюсь» условие не выполняют:
        нельзя считать, что у молчащего респондента есть дети.
        """
        if code is None or code >= SpecialCode.NOT_ASKED:
            return False
        if self.between is not None:
            return self.between[0] <= code <= self.between[1]
        return code in self.codes


class _NodeBase(_Base):
    id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    section: Section
    text: str
    """Дословная формулировка. Робот обязан воспроизводить её без изменений
    (ТЗ §8.1), поэтому в проде она заранее отрендерена в аудиофайл."""

    audio_key: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,64}$")
    next: str | None = None
    core_key: str | None = None
    """Стабильный ключ показателя постоянного ядра. Живёт между волнами и
    версиями анкеты — по нему строятся динамические ряды (ТЗ §3.1)."""

    ask_if: AskIf | None = None
    """Фильтр: вопрос задаётся только при выполнении условия."""


class InfoNode(_NodeBase):
    """Реплика без вопроса: вступление, предупреждение перед чувствительным
    блоком, благодарность."""

    kind: Literal["info"] = "info"


class ConsentNode(_NodeBase):
    """Согласие на участие (ТЗ §8.4). Без явного «да» интервью не начинается."""

    kind: Literal["consent"] = "consent"
    accept: Option
    """Названы accept/decline, а не yes/no: в YAML 1.1 голые `yes` и `no`
    разрешаются в булевы значения и ключ теряется."""

    decline: Option
    refuse_next: str
    """Куда уйти при отказе. Указывает на терминал с утверждённой прощальной
    формулировкой — робот не должен импровизировать при отказе (ТЗ §8.1)."""


class ScreenerNode(_NodeBase):
    """Фильтр пригодности: возраст 18+, проживание на территории (ТЗ §5.3)."""

    kind: Literal["screener"] = "screener"
    options: tuple[Option, ...]
    pass_codes: tuple[int, ...]
    fail_next: str

    @model_validator(mode="after")
    def _check_pass_codes(self) -> Self:
        known = {o.code for o in self.options}
        unknown = set(self.pass_codes) - known
        if unknown:
            raise QuestionnaireError(f"{self.id}: pass_codes {sorted(unknown)} нет среди вариантов")
        return self


class SingleNode(_NodeBase):
    """Выбор одного варианта."""

    kind: Literal["single"] = "single"
    options: tuple[Option, ...]
    routes: dict[int, str] = Field(default_factory=dict)
    required: bool = True
    sensitive: bool = False
    allow_dont_know: bool = True
    """Озвучивать ли вариант «затрудняюсь ответить». На приём ответа не влияет:
    отказ и затруднение принимаются всегда — принуждать к ответу нельзя."""

    @model_validator(mode="after")
    def _check_options(self) -> Self:
        codes = [o.code for o in self.options]
        if len(codes) != len(set(codes)):
            raise QuestionnaireError(f"{self.id}: коды вариантов не уникальны")
        if self.sensitive and not self.allow_dont_know:
            raise QuestionnaireError(
                f"{self.id}: чувствительный вопрос обязан допускать "
                f"«затрудняюсь ответить» (ТЗ §7.6)"
            )
        return self


class MultiNode(_NodeBase):
    """Выбор нескольких вариантов одной репликой: «до трёх ответов».

    В телефонном интервью такой вопрос нельзя разбить на серию «да/нет» по
    каждому варианту — на пятнадцати позициях это минута лишнего времени и
    гарантированный обрыв. Респондент называет несколько подряд, движок
    разбирает реплику целиком.
    """

    kind: Literal["multi"] = "multi"
    options: tuple[Option, ...]
    max_answers: int = 3
    """Сколько ответов засчитывается. Лишние отбрасываются — «до трёх ответов»
    в анкете это ограничение методики, и превышать его нельзя даже если
    респондент назвал пять."""

    exclusive_codes: tuple[int, ...] = ()
    """Варианты, отменяющие все прочие: «никому не доверяю», «проблем нет».
    Названные вместе с содержательными, они означают, что респондента
    не поняли или он передумал, — оставляем только исключающий."""

    required: bool = True
    sensitive: bool = False
    allow_dont_know: bool = True

    @model_validator(mode="after")
    def _check_options(self) -> Self:
        codes = [o.code for o in self.options]
        if len(codes) != len(set(codes)):
            raise QuestionnaireError(f"{self.id}: коды вариантов не уникальны")
        if self.max_answers < 1:
            raise QuestionnaireError(f"{self.id}: max_answers должен быть не меньше 1")
        unknown = set(self.exclusive_codes) - set(codes)
        if unknown:
            raise QuestionnaireError(
                f"{self.id}: exclusive_codes {sorted(unknown)} нет среди вариантов"
            )
        if self.sensitive and not self.allow_dont_know:
            raise QuestionnaireError(
                f"{self.id}: чувствительный вопрос обязан допускать "
                f"«затрудняюсь ответить» (ТЗ §7.6)"
            )
        return self


class ScaleNode(_NodeBase):
    """Числовая шкала. Диапазон 0-10 — то, что нормируется в компонент индекса."""

    kind: Literal["scale"] = "scale"
    low: int = 0
    high: int = 10
    low_label: str = ""
    high_label: str = ""
    required: bool = True
    sensitive: bool = False
    allow_dont_know: bool = True

    @model_validator(mode="after")
    def _check_range(self) -> Self:
        if self.low >= self.high:
            raise QuestionnaireError(f"{self.id}: low должен быть меньше high")
        if self.high > MAX_OPTION_CODE:
            raise QuestionnaireError(
                f"{self.id}: верх шкалы {self.high} заходит в служебную зону кодов"
            )
        return self


class OpenNode(_NodeBase):
    """Открытый вопрос. В моменте ничего не распознаём и не уточняем: пишем
    аудио и стенограмму, кодирование идёт ночным батчем (ТЗ §7.5, §11.2)."""

    kind: Literal["open"] = "open"
    max_seconds: int = 45
    required: bool = False
    max_mentions: int = 1
    """Сколько упоминаний засчитывается при кодировании: «до трёх ответов»,
    «до двух упоминаний». В звонке ни на что не влияет — робот всё равно
    пишет реплику целиком; ограничение применяется ночным батчем (ТЗ §11.2),
    поэтому и живёт в анкете, а не в коде кодировщика."""

    @model_validator(mode="after")
    def _check_mentions(self) -> Self:
        if self.max_mentions < 1:
            raise QuestionnaireError(f"{self.id}: max_mentions должен быть не меньше 1")
        return self


class EndNode(_NodeBase):
    """Терминал. Каждый путь по анкете обязан приводить сюда."""

    kind: Literal["end"] = "end"
    outcome: Outcome


Node = Annotated[
    InfoNode | ConsentNode | ScreenerNode | SingleNode | MultiNode | ScaleNode
    | OpenNode | EndNode,
    Field(discriminator="kind"),
]


class DialogPolicy(_Base):
    """Поведение робота, не зависящее от конкретного узла (ТЗ §8.1)."""

    max_repeats: int = 1
    """«Повторять вопрос не более одного раза при нераспознанном ответе,
    если Заказчик не утвердит иной алгоритм» — ТЗ §8.1."""

    silence_timeout_sec: float = 6.0
    barge_in_delay_sec: float = 3.0
    """Задержка перед реакцией на перебивание. Мгновенная реакция на любой звук
    обрывает вступление на кашле — число из блюпринта VOIF."""

    max_unrecognized_before_abort: int = 3
    """Сколько вопросов подряд может уйти в «нераспознано», прежде чем интервью
    признаётся непригодным (ТЗ §5.3)."""

    target_duration_sec: int = 600
    """ТЗ §3.3: не более 10 минут у 90% завершённых интервью."""


class Questionnaire(_Base):
    """Валидированная анкета. Неизменяема; идентифицируется хешем содержимого."""

    name: str
    wave: str
    policy: DialogPolicy = DialogPolicy()
    nodes: tuple[Node, ...]
    terminals: dict[Outcome, str] = Field(default_factory=dict)
    """Утверждённые завершения на ситуации, приходящие не из логики анкеты:
    просьба не звонить, просьба перезвонить, требование оператора и т.д.
    Формулировка каждого прощания утверждается Заказчиком (ТЗ §8.1)."""

    @model_validator(mode="after")
    def _validate_graph(self) -> Self:
        by_id: dict[str, Node] = {}
        for node in self.nodes:
            if node.id in by_id:
                raise QuestionnaireError(f"дублирующийся id узла: {node.id}")
            by_id[node.id] = node

        if not self.nodes:
            raise QuestionnaireError("анкета пуста")
        if not any(isinstance(n, EndNode) for n in self.nodes):
            raise QuestionnaireError("нет ни одного терминального узла")

        for node in self.nodes:
            for target in self._targets(node):
                if target not in by_id:
                    raise QuestionnaireError(f"{node.id}: переход на несуществующий узел {target}")
            if not isinstance(node, EndNode) and not self._targets(node):
                raise QuestionnaireError(f"{node.id}: нет ни одного перехода")

        self._check_filters()

        missing = REQUIRED_TERMINALS - set(self.terminals)
        if missing:
            raise QuestionnaireError(
                f"не заданы терминалы для: {sorted(m.value for m in missing)}"
            )
        for outcome, target in self.terminals.items():
            if target not in by_id:
                raise QuestionnaireError(f"терминал {outcome.value}: нет узла {target}")
            node = by_id[target]
            if not isinstance(node, EndNode):
                raise QuestionnaireError(f"терминал {outcome.value}: узел {target} не терминальный")
            if node.outcome is not outcome:
                raise QuestionnaireError(
                    f"терминал {outcome.value}: узел {target} объявляет исход {node.outcome.value}"
                )

        self._check_reachability(by_id)
        return self

    def _check_filters(self) -> None:
        """Фильтр обязан ссылаться на вопрос, который к этому моменту уже задан.

        Порядок берём объявленный: узлы перечислены в порядке интервью.
        Ссылка вперёд — это фильтр по ответу, которого ещё нет, то есть
        вопрос, который не задастся никогда.
        """
        defined_at: dict[str, int] = {}
        for position, node in enumerate(self.nodes):
            if node.core_key and node.core_key not in defined_at:
                defined_at[node.core_key] = position

        for position, node in enumerate(self.nodes):
            if node.ask_if is None:
                continue
            source = defined_at.get(node.ask_if.core_key)
            if source is None:
                raise QuestionnaireError(
                    f"{node.id}: фильтр ссылается на показатель "
                    f"{node.ask_if.core_key}, которого нет в анкете"
                )
            if source >= position:
                raise QuestionnaireError(
                    f"{node.id}: фильтр ссылается на показатель "
                    f"{node.ask_if.core_key}, который спрашивают позже"
                )

    @staticmethod
    def _targets(node: Node) -> list[str]:
        targets = [node.next] if node.next else []
        if isinstance(node, SingleNode):
            targets.extend(node.routes.values())
        elif isinstance(node, ConsentNode):
            targets.append(node.refuse_next)
        elif isinstance(node, ScreenerNode):
            targets.append(node.fail_next)
        return [t for t in targets if t]

    def _check_reachability(self, by_id: dict[str, Node]) -> None:
        """Все узлы достижимы от входа, и от каждого достижим терминал.

        Второе важнее первого: узел, из которого нельзя выйти, — это звонок,
        который повиснет на живом респонденте.
        """
        entry = self.nodes[0].id
        seen: set[str] = set()
        stack = [entry, *self.terminals.values()]
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(self._targets(by_id[current]))

        orphans = sorted(set(by_id) - seen)
        if orphans:
            raise QuestionnaireError(f"недостижимые узлы: {orphans}")

        can_finish = {n.id for n in self.nodes if isinstance(n, EndNode)}
        changed = True
        while changed:
            changed = False
            for node in self.nodes:
                if node.id in can_finish:
                    continue
                if any(t in can_finish for t in self._targets(node)):
                    can_finish.add(node.id)
                    changed = True
        stuck = sorted(set(by_id) - can_finish)
        if stuck:
            raise QuestionnaireError(f"из этих узлов не достичь завершения: {stuck}")

    # ---------------------------------------------------------------- доступ

    def node(self, node_id: str) -> Node:
        for n in self.nodes:
            if n.id == node_id:
                return n
        raise KeyError(node_id)

    @property
    def entry_id(self) -> str:
        return self.nodes[0].id

    @property
    def core_keys(self) -> tuple[str, ...]:
        """Показатели постоянного ядра — то, что обязано быть сопоставимо
        между волнами."""
        return tuple(n.core_key for n in self.nodes if n.core_key)

    @property
    def required_core_nodes(self) -> frozenset[str]:
        """Узлы, отметка по которым обязательна для валидного интервью (ТЗ §5.3).

        Чувствительные вопросы сюда не входят: §7.6 разрешает не отвечать на
        них, и требовать по ним отметку значило бы браковать интервью за
        законное молчание респондента.
        """
        return frozenset(
            n.id
            for n in self.nodes
            if isinstance(n, SingleNode | MultiNode | ScaleNode)
            and n.required
            and not n.sensitive
        )

    def screener_for(self, outcome: Outcome) -> ScreenerNode | None:
        """Найти фильтр, отбраковывающий по заданному основанию.

        Так критерии §5.3 («подтвердил возраст», «проживает на территории»)
        привязываются к анкете, а не к именам узлов, зашитым в код.
        """
        for node in self.nodes:
            if not isinstance(node, ScreenerNode):
                continue
            target = self.node(node.fail_next)
            if isinstance(target, EndNode) and target.outcome is outcome:
                return node
        return None

    @property
    def version_id(self) -> str:
        """Хеш содержимого. Правка любой формулировки меняет идентификатор,
        и старые данные остаются привязанными к старой версии."""
        canonical = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, ensure_ascii=False, separators=(",", ":")
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    # -------------------------------------------------------------- загрузка

    @classmethod
    def parse(cls, raw: object) -> Questionnaire:
        """Разобрать и провалидировать анкету.

        Ошибки схемы и ошибки графа приводятся к одному типу: тому, кто правит
        анкету, безразлично, на каком уровне он ошибся, — ему нужен один
        понятный отказ на границе загрузки.
        """
        try:
            return cls.model_validate(raw)
        except ValidationError as exc:
            raise QuestionnaireError(str(exc)) from exc

    @classmethod
    def from_yaml(cls, path: str | Path) -> Questionnaire:
        """Загрузить анкету, при необходимости собрав её из модулей.

        ТЗ §7.1 делит анкету на постоянное ядро и переменный блок, а по объёму
        дополнительные модули в одну волну не помещаются и раскладываются по
        разным волнам. Держать из-за этого несколько почти одинаковых копий
        анкеты нельзя: ядро обязано быть посимвольно одним и тем же, иначе
        динамические ряды §3.1 рвутся на ровном месте.

        Поэтому волна описывается списком `modules`, а ядро лежит одним файлом
        и подключается всеми волнами. `version_id` считается уже по собранной
        анкете — состав модулей входит в него наравне с формулировками.
        """
        source = Path(path)
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and "modules" in raw:
            raw = _assemble_modules(raw, source.parent)
        return cls.parse(raw)


#: Ссылка на вход следующего модуля. Модуль не знает, что идёт после него,
#: и знать не должен: тем же ядром пользуются все волны с разными
#: продолжениями.
NEXT_MODULE = "@next"


def _assemble_modules(raw: dict[str, object], base: Path) -> dict[str, object]:
    """Склеить анкету из модулей, разрешив ссылки `@next`."""
    names = raw.get("modules")
    if not isinstance(names, list) or not names:
        raise QuestionnaireError("modules: ожидался непустой список файлов")
    if "nodes" in raw:
        raise QuestionnaireError("анкета задаёт и modules, и nodes — оставьте что-то одно")

    loaded: list[list[dict[str, object]]] = []
    for name in names:
        if not isinstance(name, str):
            raise QuestionnaireError(f"modules: {name!r} — не имя файла")
        path = (base / name).resolve()
        if not path.is_relative_to(base.resolve()):
            raise QuestionnaireError("модуль должен находиться внутри каталога анкеты")
        try:
            body = yaml.safe_load(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise QuestionnaireError(f"модуль {name}: {exc}") from exc
        nodes = body.get("nodes") if isinstance(body, dict) else body
        if not isinstance(nodes, list) or not nodes:
            raise QuestionnaireError(f"модуль {name}: нет узлов")
        loaded.append(nodes)

    for position, nodes in enumerate(loaded):
        following = loaded[position + 1 :]
        entry = following[0][0].get("id") if following else None
        for node in nodes:
            _rewire(node, entry, names[position])

    return {k: v for k, v in raw.items() if k != "modules"} | {
        "nodes": [node for nodes in loaded for node in nodes]
    }


def _rewire(node: dict[str, object], entry: object, module: str) -> None:
    """Заменить `@next` на вход следующего модуля."""
    if node.get("next") == NEXT_MODULE:
        if entry is None:
            raise QuestionnaireError(
                f"модуль {module}, узел {node.get('id')}: ссылка {NEXT_MODULE} "
                f"в последнем модуле — дальше идти некуда"
            )
        node["next"] = entry
    for key in ("refuse_next", "fail_next"):
        if node.get(key) == NEXT_MODULE:
            if entry is None:
                raise QuestionnaireError(
                    f"модуль {module}, узел {node.get('id')}: ссылка {NEXT_MODULE} "
                    f"в последнем модуле — дальше идти некуда"
                )
            node[key] = entry
    routes = node.get("routes")
    if isinstance(routes, dict):
        for code, target in routes.items():
            if target == NEXT_MODULE:
                if entry is None:
                    raise QuestionnaireError(
                        f"модуль {module}, узел {node.get('id')}: ссылка {NEXT_MODULE} "
                        f"в последнем модуле — дальше идти некуда"
                    )
                routes[code] = entry


def reserved_codes() -> dict[str, int]:
    """Служебные коды — для кодбука и выгрузок (ТЗ §13.2)."""
    return {c.name.lower(): int(c) for c in SpecialCode}
