"""Упомянутые персоны, медиа, каналы и сообщества (ТЗ §11.1, §11.2).

Самое опасное место всего разбора — не пропустить упоминание, а **склеить
разные**. ТЗ §7.5 запрещает объединять неоднозначные имена автоматически:
«Иванов» из двух интервью может быть депутатом и соседом, а «Первый»
телеканалом и порядковым номером. Ошибка тут не техническая: из неё
рождается вывод «доверие к такому-то выросло», которого в данных нет.

Поэтому здесь два раздельных механизма:

* **справочник** — перечни персоналий, медиа и сообществ, которые Заказчик
  передаёт на каждую волну (§7.3). Совпадение со справочником даёт
  уверенную привязку;
* **эвристика** — упоминания вне справочника ловятся по контекстным
  маркерам («канал», «депутат», «газета») и по заглавной букве.
  Такие сущности помечаются неоднозначными и **не сливаются** ни с чем
  до ручной проверки.

Заглавные буквы в тексте есть потому, что ночной проход распознавания
идёт end-to-end моделью с готовой пунктуацией. На онлайн-гипотезе
из хода диалога этот механизм не работал бы.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from socmon.core.text import normalize


class EntityKind(StrEnum):
    PERSON = "person"
    MEDIA = "media"
    CHANNEL = "channel"
    COMMUNITY = "community"
    ORGANISATION = "organisation"
    UNKNOWN = "unknown"


#: Слова, после которых обычно идёт имя или название.
#: Разложены по видам, чтобы не звать всё подряд «персоной».
CONTEXT_MARKERS: dict[EntityKind, tuple[str, ...]] = {
    EntityKind.PERSON: (
        "депутат", "депутата", "глава", "главы", "мэр", "мэра",
        "губернатор", "губернатора", "министр", "министра",
        "journalist", "журналист", "журналиста", "блогер", "блогера",
        "эксперт", "эксперта", "активист", "активиста", "мужик", "женщина",
    ),
    EntityKind.MEDIA: (
        "газета", "газете", "телеканал", "канал", "радио", "издание",
        "сайт", "новости", "программа", "передача",
    ),
    EntityKind.CHANNEL: (
        "телеграм", "телеграмм", "тг", "паблик", "группа", "группе", "чат",
    ),
    EntityKind.COMMUNITY: (
        "сообщество", "движение", "объединение", "организация", "фонд",
    ),
}

_CAPITALIZED = re.compile(r"(?<![.!?]\s)(?<!^)\b([А-ЯЁ][а-яё]{2,}(?:\s+[А-ЯЁ][а-яё]{2,})?)")
_SENTENCE_START = re.compile(r"(?:^|[.!?]\s+)([А-ЯЁ][а-яё]+)")


@dataclass(frozen=True, slots=True)
class GazetteerEntry:
    """Запись справочника Заказчика."""

    key: str
    """Канонический идентификатор. По нему сущность считается в отчётах."""

    name: str
    kind: EntityKind
    aliases: tuple[str, ...] = ()
    """Варианты написания и произношения. Распознавание пишет «Ковальчюк»
    там, где в справочнике «Ковальчук», и без списка вариантов упоминание
    просто потеряется."""

    @property
    def surfaces(self) -> tuple[str, ...]:
        return (self.name, *self.aliases)


@dataclass(frozen=True, slots=True)
class Mention:
    """Упоминание в тексте."""

    surface: str
    """Как прозвучало в ответе. Хранится дословно: §11.3 требует
    возможности проверить любой автоматический вывод."""

    kind: EntityKind
    key: str | None = None
    """Ключ справочника. None означает, что сущности там нет."""

    ambiguous: bool = False
    """Привязать однозначно не удалось. Такие упоминания **не сливаются**
    ни с чем до ручной проверки (ТЗ §7.5)."""

    candidates: tuple[str, ...] = ()
    """Между какими записями справочника не удалось выбрать."""

    is_model_generated: bool = True


@dataclass
class Gazetteer:
    """Справочник персоналий, медиа и сообществ волны."""

    entries: tuple[GazetteerEntry, ...] = ()
    _index: dict[str, list[GazetteerEntry]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self._index = {}
        for entry in self.entries:
            for surface in entry.surfaces:
                self._index.setdefault(normalize(surface), []).append(entry)

    def lookup(self, surface: str) -> list[GazetteerEntry]:
        return self._index.get(normalize(surface), [])

    @property
    def surfaces(self) -> tuple[str, ...]:
        """Все написания, отсортированные от длинных к коротким.

        Порядок важен: «Иван Петров» должен победить «Иван», иначе
        полное имя развалится на два ложных упоминания.
        """
        return tuple(sorted(self._index, key=len, reverse=True))


def extract(text: str, gazetteer: Gazetteer | None = None) -> tuple[Mention, ...]:
    """Найти упоминания в ответе."""
    book = gazetteer or Gazetteer()
    found: list[Mention] = []
    claimed: list[tuple[int, int]] = []
    normalized = normalize(text)

    # Сначала справочник: у него приоритет над догадками.
    for surface in book.surfaces:
        for match in re.finditer(rf"(?<!\w){re.escape(surface)}(?!\w)", normalized):
            if _overlaps(match.span(), claimed):
                continue
            claimed.append(match.span())
            matches = book.lookup(surface)
            keys = sorted({e.key for e in matches})
            found.append(Mention(
                surface=surface,
                kind=matches[0].kind if len(keys) == 1 else EntityKind.UNKNOWN,
                key=keys[0] if len(keys) == 1 else None,
                ambiguous=len(keys) > 1,
                candidates=tuple(keys) if len(keys) > 1 else (),
            ))

    found.extend(_heuristic(text, normalized, claimed))
    return tuple(found)


def _heuristic(
    text: str, normalized: str, claimed: list[tuple[int, int]]
) -> list[Mention]:
    """Упоминания вне справочника.

    Всё найденное здесь помечается неоднозначным без исключений. Мы не
    знаем, кто такой «Петров» из ответа, и делать вид, что знаем, нельзя.
    """
    mentions: list[Mention] = []
    seen: set[str] = set()

    for match in _CAPITALIZED.finditer(text):
        surface = match.group(1).strip()
        key = normalize(surface)
        if key in seen:
            continue
        position = normalized.find(key)
        if position >= 0 and _overlaps((position, position + len(key)), claimed):
            continue
        seen.add(key)
        mentions.append(Mention(
            surface=surface,
            kind=_kind_from_context(text, match.start()),
            ambiguous=True,
        ))
    return mentions


def _kind_from_context(text: str, position: int) -> EntityKind:
    """Определить вид по словам перед упоминанием."""
    prefix = normalize(text[max(0, position - 40):position])
    words = prefix.split()[-4:]
    for kind, markers in CONTEXT_MARKERS.items():
        if any(word in markers for word in words):
            return kind
    return EntityKind.UNKNOWN


def _overlaps(span: tuple[int, int], claimed: Iterable[tuple[int, int]]) -> bool:
    return any(span[0] < end and start < span[1] for start, end in claimed)


@dataclass(frozen=True, slots=True)
class MentionTally:
    """Свод упоминаний по волне (ТЗ §13.2)."""

    key: str
    name: str
    kind: EntityKind
    count: int
    ambiguous: bool

    @property
    def note(self) -> str | None:
        if not self.ambiguous:
            return None
        return (
            "упоминание не привязано к справочнику: требует ручной проверки "
            "перед публикацией (ТЗ §7.5)"
        )


def tally(
    mentions: Sequence[Mention], gazetteer: Gazetteer | None = None
) -> tuple[MentionTally, ...]:
    """Свести упоминания в перечень для отчёта.

    Неоднозначные считаются отдельно и **не суммируются** с записями
    справочника, даже если написание похоже. Иначе ручная проверка
    теряет смысл: сливать будет уже нечего, всё слито.
    """
    from collections import Counter

    book = gazetteer or Gazetteer()
    resolved: Counter[str] = Counter()
    unresolved: Counter[tuple[str, EntityKind]] = Counter()

    for mention in mentions:
        if mention.key is not None and not mention.ambiguous:
            resolved[mention.key] += 1
        else:
            unresolved[(normalize(mention.surface), mention.kind)] += 1

    names = {e.key: e for e in book.entries}
    rows = [
        MentionTally(
            key=key, name=names[key].name if key in names else key,
            kind=names[key].kind if key in names else EntityKind.UNKNOWN,
            count=count, ambiguous=False,
        )
        for key, count in resolved.most_common()
    ]
    rows.extend(
        MentionTally(key=surface, name=surface, kind=kind, count=count, ambiguous=True)
        for (surface, kind), count in unresolved.most_common()
    )
    return tuple(rows)
