"""Разбор одного интервью (ТЗ §11.1).

Собирает то, что требует ТЗ по каждому содержательно завершённому интервью:
краткое резюме, тематические метки, тревожные темы, тональность, признаки
раздражения и одобрения, дословные цитаты с временными метками, перечень
упомянутых персон и сообществ, оценку качества распознавания и разметки.

Одно правило проходит через весь модуль и важнее остального кода:

> Оценки тональности, эмоций и интонации — **модельные вероятностные
> оценки**. Они не являются клинической, психологической или иной
> диагностикой и не могут использоваться для вывода о чувствительных
> характеристиках конкретного респондента (ТЗ §11.1).

Отсюда две вещи в конструкции. Первая: каждый вывод несёт `is_model_generated`
и уверенность — маркировка не приписывается потом, её нельзя забыть.
Вторая: разбор **никогда не трогает первичный ответ**. Коды, которые
respondent дал в анкете, разбором не переписываются — только дополняются
производными полями.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from socmon.coding.classifier import Coding, ThemeClassifier, classify
from socmon.coding.entities import Gazetteer, Mention, extract
from socmon.core.text import normalize

#: Лексиконы признаков из ТЗ §11.1.
#:
#: Словарь, а не модель, выбран сознательно: §11.3 требует, чтобы любой
#: автоматический вывод можно было проверить, а «модель так решила»
#: проверке не поддаётся. Когда на пилоте станет ясно, чего словарю
#: не хватает, рядом встаёт модель `emo` из линейки GigaAM — контракт
#: тот же, маркировка та же.
SIGNAL_LEXICON: dict[str, tuple[str, ...]] = {
    "раздражение": (
        "надоело", "достало", "сколько можно", "безобразие", "возмутительно",
        "хватит", "терпение", "бесит", "злит", "отвратительно",
    ),
    "одобрение": (
        "спасибо", "хорошо", "молодцы", "правильно", "нравится",
        "довольны", "рад", "рада", "порядок", "устраивает",
    ),
    "похвала": (
        "отлично", "прекрасно", "замечательно", "лучше стало", "наладилось",
    ),
    "недоверие": (
        "не верю", "обманывают", "врут", "показуха", "для галочки",
        "все равно ничего", "бесполезно", "не сделают",
    ),
    "тревога": (
        "страшно", "боюсь", "боимся", "тревожно", "беспокоюсь",
        "опасаюсь", "переживаю", "неспокойно", "что будет дальше",
    ),
}

NEGATIVE_SIGNALS = frozenset({"раздражение", "недоверие", "тревога"})
POSITIVE_SIGNALS = frozenset({"одобрение", "похвала"})


class Tone(StrEnum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


@dataclass(frozen=True, slots=True)
class SignalHit:
    """Признак с обоснованием."""

    name: str
    matched: str
    """Что именно сработало. Без этого проверить вывод невозможно."""


@dataclass(frozen=True, slots=True)
class ToneEstimate:
    """Оценка тональности.

    Именно оценка: поля названы так, чтобы в выгрузке нельзя было принять
    её за факт (ТЗ §11.1).
    """

    tone: Tone
    signals: tuple[SignalHit, ...]
    confidence: float
    is_model_generated: bool = True

    @property
    def disclaimer(self) -> str:
        return (
            "Модельная вероятностная оценка. Не является диагностикой "
            "и не может использоваться для вывода о характеристиках "
            "конкретного респондента (ТЗ §11.1)."
        )


@dataclass(frozen=True, slots=True)
class Quote:
    """Дословная цитата с привязкой ко времени (ТЗ §11.1)."""

    text: str
    node_id: str
    start_sec: float | None = None
    end_sec: float | None = None

    @property
    def timed(self) -> bool:
        return self.start_sec is not None


@dataclass(frozen=True, slots=True)
class QualityEstimate:
    """Оценка качества транскрипции и классификации (ТЗ §11.1, §11.3)."""

    answers_total: int
    answers_unrecognized: int
    open_answers: int
    open_unclassified: int
    ambiguous_mentions: int
    needs_review: bool
    reasons: tuple[str, ...] = ()

    @property
    def unrecognized_share(self) -> float:
        if not self.answers_total:
            return 0.0
        return self.answers_unrecognized / self.answers_total


@dataclass(frozen=True, slots=True)
class InterviewAnalysis:
    """Полный разбор интервью."""

    summary: str
    codings: tuple[Coding, ...]
    themes: tuple[str, ...]
    alarming_themes: tuple[str, ...]
    tone: ToneEstimate
    quotes: tuple[Quote, ...]
    mentions: tuple[Mention, ...]
    quality: QualityEstimate
    classifier_version: str
    is_model_generated: bool = True

    @property
    def review_notes(self) -> tuple[str, ...]:
        return self.quality.reasons


@dataclass(frozen=True, slots=True)
class OpenAnswer:
    """Открытый ответ, как он выходит из ночного распознавания."""

    node_id: str
    text: str
    start_sec: float | None = None
    end_sec: float | None = None


@dataclass(frozen=True, slots=True)
class InterviewInput:
    """Что подаётся на разбор.

    Закрытые ответы приходят только счётчиками: разбору незачем знать,
    что респондент выбрал в шкале, а вот доля нераспознанного нужна
    для оценки качества.
    """

    open_answers: tuple[OpenAnswer, ...] = ()
    closed_total: int = 0
    closed_unrecognized: int = 0
    extra_text: tuple[str, ...] = field(default_factory=tuple)


def detect_signals(text: str) -> tuple[SignalHit, ...]:
    """Найти признаки из §11.1."""
    normalized = normalize(text)
    hits: list[SignalHit] = []
    for name, phrases in SIGNAL_LEXICON.items():
        for phrase in phrases:
            marker = normalize(phrase)
            if marker and marker in normalized:
                hits.append(SignalHit(name=name, matched=marker))
                break
    return tuple(hits)


def estimate_tone(text: str) -> ToneEstimate:
    """Оценить тональность по найденным признакам.

    Уверенность растёт с числом сработавших признаков и падает, когда
    они противоречат друг другу. Ответ «спасибо, но надоело уже» —
    это не нейтральный ответ, а неуверенная оценка, и разница
    в выгрузке должна быть видна.
    """
    signals = detect_signals(text)
    negative = sum(1 for s in signals if s.name in NEGATIVE_SIGNALS)
    positive = sum(1 for s in signals if s.name in POSITIVE_SIGNALS)

    if negative == positive:
        tone = Tone.NEUTRAL
        confidence = 0.3 if signals else 0.5
    elif negative > positive:
        tone = Tone.NEGATIVE
        confidence = min(0.9, 0.5 + 0.15 * (negative - positive))
    else:
        tone = Tone.POSITIVE
        confidence = min(0.9, 0.5 + 0.15 * (positive - negative))

    return ToneEstimate(tone=tone, signals=signals, confidence=round(confidence, 2))


def summarize(codings: Sequence[Coding], tone: ToneEstimate) -> str:
    """Собрать краткое резюме (ТЗ §11.1).

    Резюме собирается из уже размеченного, а не пересказывает ответ своими
    словами. Пересказ — работа языковой модели, и он подключается сверху;
    но даже без модели резюме обязано быть, иначе оперативный пакет
    неполон.
    """
    names: list[str] = []
    for coding in codings:
        for hit in coding.themes:
            if hit.name not in names:
                names.append(hit.name)

    if not names:
        return "Открытые ответы не размечены ни одной темой классификатора."
    tone_word = {
        Tone.NEGATIVE: "негативная", Tone.POSITIVE: "позитивная",
        Tone.NEUTRAL: "нейтральная",
    }[tone.tone]
    return (
        f"Темы: {', '.join(names)}. Тональность {tone_word} "
        f"(модельная оценка, уверенность {tone.confidence:.2f})."
    )


def analyse(
    interview: InterviewInput,
    classifier: ThemeClassifier,
    gazetteer: Gazetteer | None = None,
    *,
    max_quotes: int = 5,
) -> InterviewAnalysis:
    """Разобрать интервью."""
    codings = tuple(classify(a.text, classifier) for a in interview.open_answers)
    corpus = " ".join(
        [a.text for a in interview.open_answers] + list(interview.extra_text)
    )
    tone = estimate_tone(corpus)

    themes: list[str] = []
    alarming: list[str] = []
    for coding in codings:
        for hit in coding.themes:
            if hit.code not in themes:
                themes.append(hit.code)
            if hit.alarming and hit.code not in alarming:
                alarming.append(hit.code)

    mentions = tuple(
        mention
        for answer in interview.open_answers
        for mention in extract(answer.text, gazetteer)
    )
    quotes = tuple(
        Quote(
            text=answer.text.strip(), node_id=answer.node_id,
            start_sec=answer.start_sec, end_sec=answer.end_sec,
        )
        for answer in interview.open_answers
        if answer.text.strip()
    )[:max_quotes]

    return InterviewAnalysis(
        summary=summarize(codings, tone),
        codings=codings,
        themes=tuple(themes),
        alarming_themes=tuple(alarming),
        tone=tone,
        quotes=quotes,
        mentions=mentions,
        quality=_quality(interview, codings, mentions),
        classifier_version=classifier.version_id,
    )


def _quality(
    interview: InterviewInput, codings: Sequence[Coding], mentions: Sequence[Mention]
) -> QualityEstimate:
    """Оценить, годится ли разбор без человека.

    Сомнительные случаи не отбрасываются и не «дочищаются» автоматически —
    они помечаются для ручной проверки. ТЗ §11.3 требует ручной проверки
    случайной выборки; эти интервью попадают в неё не случайно,
    а по признаку.
    """
    unclassified = sum(1 for c in codings if c.unmatched)
    ambiguous = sum(1 for m in mentions if m.ambiguous)
    reasons: list[str] = []

    if interview.closed_total and interview.closed_unrecognized / interview.closed_total > 0.2:
        reasons.append(
            f"не распознано {interview.closed_unrecognized} из "
            f"{interview.closed_total} закрытых ответов — проверить запись"
        )
    resolved = sum(1 for m in mentions if m.key is not None and not m.ambiguous)
    if codings and unclassified == len(codings) and not resolved:
        # Оговорка про упоминания существенна: вопрос об источниках доверия
        # (ТЗ §7.5) вообще не про проблемы, и ответ «доверяю такому-то»
        # законно не попадает ни в одну тему. Считать это пробелом
        # классификатора значит слать на ручную проверку каждое второе
        # интервью и обесценить сам признак.
        reasons.append(
            "ни один открытый ответ не размечен: возможен пробел "
            "в классификаторе (ТЗ §11.2)"
        )
    if ambiguous:
        reasons.append(
            f"неоднозначных упоминаний: {ambiguous} — требуют ручной "
            f"проверки перед сведением (ТЗ §7.5)"
        )

    return QualityEstimate(
        answers_total=interview.closed_total,
        answers_unrecognized=interview.closed_unrecognized,
        open_answers=len(codings),
        open_unclassified=unclassified,
        ambiguous_mentions=ambiguous,
        needs_review=bool(reasons),
        reasons=tuple(reasons),
    )
