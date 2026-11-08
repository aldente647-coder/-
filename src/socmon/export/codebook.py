"""Кодбук переменных (ТЗ §13.2).

Кодбук строится из самой анкеты, а не ведётся отдельным файлом. Отдельный файл
неизбежно расходится с анкетой — и расхождение обнаруживается тогда, когда
Заказчик уже посчитал что-то по неверным подписям.

Служебные коды выписываются по каждой переменной явно: без этого «97» в
выгрузке выглядит как оценка по шкале, а не как «затрудняюсь ответить».
"""

from __future__ import annotations

from dataclasses import dataclass, field

from socmon.core.codes import SpecialCode
from socmon.core.questionnaire import (
    ConsentNode,
    OpenNode,
    Questionnaire,
    ScaleNode,
    ScreenerNode,
    SingleNode,
)

KIND_LABELS = {
    "single": "выбор одного варианта",
    "scale": "числовая шкала",
    "open": "открытый ответ",
    "screener": "фильтр",
    "consent": "согласие",
}

SERVICE_LABELS: dict[int, str] = {
    SpecialCode.NOT_ASKED: "не задавался по логике анкеты",
    SpecialCode.UNRECOGNIZED: "речь не распознана",
    SpecialCode.DONT_KNOW: "затрудняюсь ответить",
    SpecialCode.REFUSED: "отказ от ответа на вопрос",
    SpecialCode.NO_ANSWER: "интервью прервано до этого вопроса",
}


@dataclass(frozen=True, slots=True)
class CodebookEntry:
    variable: str
    """Имя переменной в выгрузке. Для показателей ядра — стабильный ключ,
    переживающий смену формулировки между волнами."""

    node_id: str
    section: str
    question: str
    kind: str
    values: dict[int, str] = field(default_factory=dict)
    required: bool = False
    sensitive: bool = False
    core_key: str | None = None

    @property
    def kind_label(self) -> str:
        return KIND_LABELS.get(self.kind, self.kind)

    @property
    def service_values(self) -> dict[int, str]:
        """Служебные коды. У открытых вопросов их нет — там хранится текст."""
        if self.kind == "open":
            return {}
        return dict(SERVICE_LABELS)

    def value_label(self, code: int | None) -> str:
        if code is None:
            return ""
        return self.values.get(code) or SERVICE_LABELS.get(code) or str(code)


@dataclass(frozen=True, slots=True)
class Codebook:
    questionnaire_name: str
    version_id: str
    entries: tuple[CodebookEntry, ...]

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def by_variable(self, variable: str) -> CodebookEntry | None:
        for entry in self.entries:
            if entry.variable == variable:
                return entry
        return None

    @property
    def variables(self) -> tuple[str, ...]:
        return tuple(e.variable for e in self.entries)


def build(questionnaire: Questionnaire) -> Codebook:
    """Собрать кодбук по анкете."""
    entries: list[CodebookEntry] = []
    for node in questionnaire.nodes:
        entry = _entry_for(node)
        if entry is not None:
            entries.append(entry)
    return Codebook(
        questionnaire_name=questionnaire.name,
        version_id=questionnaire.version_id,
        entries=tuple(entries),
    )


def _entry_for(node: object) -> CodebookEntry | None:
    def base(kind: str, values: dict[int, str], **extra: object) -> CodebookEntry:
        assert isinstance(node, SingleNode | ScaleNode | OpenNode | ScreenerNode | ConsentNode)
        return CodebookEntry(
            variable=node.core_key or node.id,
            node_id=node.id,
            section=node.section.value,
            question=" ".join(node.text.split()),
            kind=kind,
            values=values,
            core_key=node.core_key,
            **extra,  # type: ignore[arg-type]
        )

    match node:
        case SingleNode():
            return base(
                "single",
                {o.code: o.label for o in node.options},
                required=node.required,
                sensitive=node.sensitive,
            )
        case ScaleNode():
            # У шкалы подписаны только полюса — так её и задают респонденту.
            values = {node.low: node.low_label, node.high: node.high_label}
            return base(
                "scale",
                {k: v for k, v in values.items() if v},
                required=node.required,
                sensitive=node.sensitive,
            )
        case ScreenerNode():
            return base("screener", {o.code: o.label for o in node.options}, required=True)
        case ConsentNode():
            return base(
                "consent",
                {node.accept.code: node.accept.label, node.decline.code: node.decline.label},
                required=True,
            )
        case OpenNode():
            return base("open", {})
        case _:
            return None
