"""Тематический классификатор открытых ответов (ТЗ §11.2).

Правила лежат в YAML рядом с анкетой и правятся так же — руками методолога,
а не переобучением. Версия классификатора считается от содержимого: правку
одного ключевого слова видно по идентификатору, и старые размеченные ответы
остаются привязаны к своей версии.

Это прямое требование §11.3: скрытое изменение исторических данных без
версионирования запрещено. Если бы разметка хранилась без версии, вчерашняя
и сегодняшняя выглядели бы одинаково, а означали разное.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from socmon.core.text import normalize


class ClassifierError(ValueError):
    """Классификатор не прошёл проверку. Поднимается на загрузке."""


class _Base(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Theme(_Base):
    """Одна тема классификатора."""

    code: str
    name: str
    phrases: tuple[str, ...] = ()
    """Слова и обороты, по которым тема опознаётся. Совпадение ищется
    по границам слова: без этого «дом» ловится внутри «домогательства»."""

    patterns: tuple[str, ...] = ()
    """Регулярные выражения для случаев, где перечислить формы невозможно:
    «подорожа-ло/ли/ет», «не хватает денег/средств»."""

    excludes: tuple[str, ...] = ()
    """Обороты, гасящие ложное срабатывание **в этом месте текста**.

    Отменяют не тему целиком, а конкретное совпадение: «цены выросли,
    а дороги разбитые» — это законно и цены, и транспорт. Исключение,
    отменяющее тему целиком, съедало бы половину содержательных ответов.
    """

    alarming: bool = False
    """Тема относится к тревожным (ТЗ §11.1: «перечень тревожных тем»)."""

    priority: int = 0
    """При равной длине совпадения выигрывает более высокий приоритет."""

    @model_validator(mode="after")
    def _check_has_rules(self) -> Self:
        if not self.phrases and not self.patterns:
            raise ClassifierError(f"тема {self.code}: не задано ни одного правила")
        for pattern in self.patterns:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ClassifierError(f"тема {self.code}: плохое выражение {pattern!r}") from exc
        return self


class ThemeClassifier(_Base):
    """Утверждённый Заказчиком перечень тем."""

    name: str
    revision: str = ""
    """Человекочитаемая пометка Заказчика: «утверждено протоколом от…».
    На идентификатор версии не влияет — тот считается от правил."""

    themes: tuple[Theme, ...]

    @model_validator(mode="after")
    def _check_codes(self) -> Self:
        codes = [t.code for t in self.themes]
        if not codes:
            raise ClassifierError("классификатор пуст")
        if len(codes) != len(set(codes)):
            raise ClassifierError("коды тем не уникальны")
        return self

    @property
    def version_id(self) -> str:
        """Отпечаток правил. Правка любого слова даёт новый идентификатор."""
        canonical = json.dumps(
            [t.model_dump(mode="json") for t in self.themes],
            sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    @property
    def alarming_codes(self) -> frozenset[str]:
        return frozenset(t.code for t in self.themes if t.alarming)

    def theme(self, code: str) -> Theme:
        for theme in self.themes:
            if theme.code == code:
                return theme
        raise KeyError(code)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ThemeClassifier:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(raw)


Match = Annotated[tuple[str, int], Field()]


class ThemeHit(_Base):
    """Сработавшая тема с обоснованием.

    Обоснование хранится не для красоты: §11.3 требует возможности
    проверить любой автоматический вывод, а «тема такая-то, потому что
    модель так решила» проверке не поддаётся.
    """

    code: str
    name: str
    matched: str
    alarming: bool


class Coding(_Base):
    """Результат разметки одного открытого ответа."""

    classifier_version: str
    themes: tuple[ThemeHit, ...]
    unmatched: bool
    """Ни одна тема не сработала. Кандидат в новые темы (ТЗ §11.2)."""

    is_model_generated: bool = True
    """ТЗ §11.3: автоматически сгенерированные результаты маркируются.
    Поле неизменно и стоит на каждом выводе — чтобы отличить разметку
    от первичного ответа было можно всегда, а не по памяти аналитика."""

    @property
    def codes(self) -> tuple[str, ...]:
        return tuple(hit.code for hit in self.themes)

    @property
    def alarming(self) -> tuple[str, ...]:
        return tuple(hit.code for hit in self.themes if hit.alarming)


def _word_regex(phrase: str) -> re.Pattern[str]:
    words = normalize(phrase).split()
    body = r"\s+".join(re.escape(w) for w in words)
    return re.compile(rf"(?<!\w){body}(?!\w)")


def classify(text: str, classifier: ThemeClassifier) -> Coding:
    """Разметить ответ темами.

    Тем может быть несколько: «цены выросли, и до больницы не доехать» —
    это и материальное положение, и медицина, и транспорт. Схлопывать
    в одну тему значит терять то, ради чего открытый вопрос и задавался.
    """
    normalized = normalize(text)
    hits: list[ThemeHit] = []

    for theme in classifier.themes:
        matched = _first_match(normalized, theme)
        if matched is not None:
            hits.append(ThemeHit(
                code=theme.code, name=theme.name,
                matched=matched, alarming=theme.alarming,
            ))

    ordered = sorted(
        hits, key=lambda h: (-classifier.theme(h.code).priority, h.code)
    )
    return Coding(
        classifier_version=classifier.version_id,
        themes=tuple(ordered),
        unmatched=not ordered,
    )


def _first_match(normalized: str, theme: Theme) -> str | None:
    """Первое совпадение темы, не попавшее в зону исключения.

    Зоны считаются по позициям в тексте, а не по факту наличия слова:
    исключение гасит совпадение, наложившееся на него, и только его.
    """
    blocked = [
        found.span()
        for phrase in theme.excludes
        for found in _word_regex(phrase).finditer(normalized)
    ]

    def clear(span: tuple[int, int]) -> bool:
        return not any(span[0] < end and start < span[1] for start, end in blocked)

    for phrase in theme.phrases:
        for found in _word_regex(phrase).finditer(normalized):
            if clear(found.span()):
                return found.group(0)
    for pattern in theme.patterns:
        for found in re.finditer(pattern, normalized):
            if clear(found.span()):
                return found.group(0)
    return None


def unmatched_fragments(
    answers: Sequence[str], classifier: ThemeClassifier, *, min_count: int = 3
) -> tuple[tuple[str, int], ...]:
    """Слова из неразмеченных ответов — кандидаты в новые темы (ТЗ §11.2).

    Не «новые темы», а именно кандидаты: решение о расширении классификатора
    принимает Заказчик, а не система. Порог по частоте отсекает случайные
    оговорки: тема, прозвучавшая единожды, темой пока не является.
    """
    from collections import Counter

    counter: Counter[str] = Counter()
    for answer in answers:
        if not classify(answer, classifier).unmatched:
            continue
        for word in normalize(answer).split():
            if len(word) > 4:
                counter[word] += 1
    return tuple(
        (word, count) for word, count in counter.most_common() if count >= min_count
    )
