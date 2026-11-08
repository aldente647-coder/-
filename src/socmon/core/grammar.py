"""Распознавание ответа по грамматике узла.

Ступень 1 каскада из блюпринта VOIF (keyword → эмбеддинги → LLM), но третья
ступень у нас отсутствует принципиально: ТЗ §8.2 запрещает генеративной модели
интерпретировать ответ в ходе интервью. Не разобрали грамматикой и семантикой —
переспрашиваем ровно один раз и ставим код «нераспознанная речь».

Ограниченный словарь на каждый вопрос вместо свободного ASR — то, что вытягивает
точность на телефонном канале 8 кГц.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from socmon.core.text import normalize


class ControlIntent(StrEnum):
    """Управляющие намерения. Проверяются раньше вариантов ответа и действуют
    на любом узле анкеты (ТЗ §8.1)."""

    REPEAT = "repeat"
    OPERATOR = "operator"
    CALLBACK = "callback"
    DONT_CALL = "dont_call"
    HANGUP = "hangup"
    WHO = "who"
    """«А вы кто такие?» — человек хочет понять, с кем говорит, прежде чем
    отвечать. Робот объясняет и возвращается к вопросу; это не отказ."""


#: Управляющие фразы намеренно многословны и явны.
#:
#: Урок блюпринта: маркеры перенаправления нельзя мешать с отрицательным
#: настроением. «Не интересно» — отказ от текущего предложения, звонить можно
#: позже; «не звоните мне» — запрет и стоп-лист навсегда (ТЗ §6.2). Смешать
#: эти два случая — значит либо терять базу впустую, либо нарушать прямой
#: запрет абонента. Поэтому одиночное «нет» сюда не попадает никогда.
CONTROL_PHRASES: dict[ControlIntent, tuple[str, ...]] = {
    ControlIntent.DONT_CALL: (
        "не звоните", "больше не звоните", "не надо мне звонить",
        "удалите мой номер", "уберите мой номер", "исключите мой номер",
        "не беспокойте", "больше не беспокойте",
    ),
    ControlIntent.OPERATOR: (
        "оператор", "оператора", "с оператором", "живой человек",
        "живого человека", "с человеком", "позовите человека",
    ),
    ControlIntent.REPEAT: (
        "повторите", "повторить", "еще раз", "не расслышал", "не расслышала",
        "не понял вопрос", "не поняла вопрос", "что вы сказали",
        "не понял", "не поняла", "не понимаю", "не расслышала вопрос",
        "какие варианты", "какие есть варианты", "повторите варианты",
        "какие были варианты", "а какие варианты", "что за варианты",
    ),
    ControlIntent.WHO: (
        "кто вы", "вы кто", "кто это", "кто говорит", "кто звонит",
        "кто такие", "вы кто такие", "кто вы такие", "что вам нужно",
        "откуда у вас мой номер", "откуда номер", "откуда мой номер",
        "что за опрос", "какой опрос", "что это за опрос", "зачем это",
        "для чего это", "кто вас прислал", "вы откуда",
    ),
    ControlIntent.CALLBACK: (
        "перезвоните", "перезвоните позже", "позвоните позже",
        "сейчас неудобно", "неудобно говорить", "я за рулем", "я на работе",
    ),
    ControlIntent.HANGUP: (
        "до свидания", "всего доброго", "закончим на этом", "прекратите",
        "я кладу трубку",
    ),
}


class MatchKind(StrEnum):
    ANSWER = "answer"
    CONTROL = "control"
    NONE = "none"
    AMBIGUOUS = "ambiguous"
    HEDGED = "hedged"
    """Уклончивый ответ: «средне», «так себе», «не очень доволен». Человек
    ответил, но не выбрал. Живой интервьюер уточняет один раз: «и всё же —
    ближе к чему?» — и лишь потом записывает «затрудняюсь»."""


#: Отрицание перед вариантом меняет его смысл на противоположный или
#: уклончивый: «не очень доволен» — это не «доволен». Между отрицанием
#: и словом допускаются усилители и связки: «не совсем», «не то чтобы».
NEGATIONS: frozenset[str] = frozenset({"не", "ни"})
NEGATION_BRIDGES: frozenset[str] = frozenset({
    "очень", "совсем", "особо", "особенно", "то", "чтобы", "так", "уж",
    "слишком", "прям", "прямо", "сказать", "чтоб", "сильно", "вполне",
})

#: Уклончивые ответы: середина там, где её не предлагали.
HEDGE_PHRASES: tuple[str, ...] = (
    "средне", "среднее", "так себе", "ну такое", "такое себе", "по разному",
    "когда как", "зависит", "смотря", "и так и так", "и так и сяк",
    "пятьдесят на пятьдесят", "50 на 50", "ни то ни другое", "ни то ни се",
    "ни да ни нет", "серединка на половинку", "не то и не другое",
    "где то посередине", "посередине", "между", "и да и нет",
)

#: Маркеры поправки: человек назвал одно, потом другое — считается последнее.
CORRECTION_MARKERS: tuple[str, ...] = (
    "хотя", "вернее", "точнее", "в смысле", "то есть", "все таки", "все же",
    "нет", "пожалуй", "ой", "или нет", "хотя нет", "скорее",
)


@dataclass(frozen=True, slots=True)
class Match:
    kind: MatchKind
    code: int | None = None
    intent: ControlIntent | None = None
    phrase: str | None = None
    candidates: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class OptionGrammar:
    """Грамматика одного варианта ответа."""

    code: int
    phrases: tuple[str, ...] = ()
    patterns: tuple[str, ...] = ()
    _compiled: tuple[re.Pattern[str], ...] = field(default=(), repr=False, compare=False)

    @staticmethod
    def build(code: int, phrases: Iterable[str], patterns: Iterable[str] = ()) -> OptionGrammar:
        pats = tuple(patterns)
        return OptionGrammar(
            code=code,
            phrases=tuple(phrases),
            patterns=pats,
            _compiled=tuple(re.compile(p) for p in pats),
        )


def _phrase_regex(phrase: str) -> re.Pattern[str]:
    """Фраза как последовательность слов на границах слова.

    Без границ «да» ловилось бы внутри «дальше», а «нет» — внутри «нету
    времени». На односложных русских ответах это ошибка на каждом третьем звонке.
    """
    words = normalize(phrase).split()
    body = r"\s+".join(re.escape(w) for w in words)
    return re.compile(rf"(?<!\w){body}(?!\w)")


_CONTROL_REGEX: dict[ControlIntent, tuple[tuple[str, re.Pattern[str]], ...]] = {
    intent: tuple((p, _phrase_regex(p)) for p in phrases)
    for intent, phrases in CONTROL_PHRASES.items()
}


def match_control(utterance: str) -> Match:
    """Найти управляющее намерение. Выигрывает самая длинная сработавшая фраза:
    «больше не звоните» должно победить «не звоните», а не наоборот."""
    text = normalize(utterance)
    best: tuple[int, ControlIntent, str] | None = None
    for intent, compiled in _CONTROL_REGEX.items():
        for phrase, rx in compiled:
            if rx.search(text) and (best is None or len(phrase) > best[0]):
                best = (len(phrase), intent, phrase)
    if best is None:
        return Match(kind=MatchKind.NONE)
    return Match(kind=MatchKind.CONTROL, intent=best[1], phrase=best[2])


def match_options(utterance: str, options: Sequence[OptionGrammar]) -> Match:
    """Сопоставить реплику с вариантами ответа узла.

    Два случая, которые легко перепутать:

    * **Вложение.** «скорее да» содержит «да». Побеждает объемлющее совпадение,
      иначе любая шкала согласия схлопнется в «да»/«нет».
    * **Противоречие.** В «да нет» два разных варианта сработали на
      *непересекающихся* участках фразы. Это не ответ: возвращаем AMBIGUOUS,
      и движок переспрашивает вместо того, чтобы угадывать.

    Поэтому сравниваются не длины, а участки текста.
    """
    text = normalize(utterance)
    spans: dict[int, tuple[int, int]] = {}
    negated: list[int] = []
    for opt in options:
        best: tuple[int, int] | None = None
        for phrase in opt.phrases:
            found = _phrase_regex(phrase).search(text)
            if found and (best is None or _width(found.span()) > _width(best)):
                best = found.span()
        for rx in opt._compiled:
            found = rx.search(text)
            if found and (best is None or _width(found.span()) > _width(best)):
                best = found.span()
        if best is None:
            continue
        # «Не очень доволен»: вариант назван, но под отрицанием. Сам вариант,
        # начинающийся с «не» («не доверяю»), отрицанием не считается.
        if _negated(text, best):
            negated.append(opt.code)
            continue
        current = spans.get(opt.code)
        if current is None or _width(best) > _width(current):
            spans[opt.code] = best

    if not spans:
        if negated or _hedged(text):
            return Match(kind=MatchKind.HEDGED, candidates=tuple(sorted(negated)))
        return Match(kind=MatchKind.NONE)
    if len(spans) == 1:
        return Match(kind=MatchKind.ANSWER, code=next(iter(spans)))

    winner = max(spans, key=lambda code: _width(spans[code]))
    if all(_contains(spans[winner], span) for code, span in spans.items() if code != winner):
        return Match(kind=MatchKind.ANSWER, code=winner)
    corrected = _last_after_correction(text, spans)
    if corrected is not None:
        return Match(kind=MatchKind.ANSWER, code=corrected)
    return Match(kind=MatchKind.AMBIGUOUS, candidates=tuple(sorted(spans)))


def _negated(text: str, span: tuple[int, int]) -> bool:
    """Стоит ли перед совпадением отрицание — прямо или через усилитель."""
    before = text[: span[0]].split()
    if text[span[0]:].split()[:1] and text[span[0]:].split()[0] in NEGATIONS:
        return False
    for word in reversed(before[-3:]):
        if word in NEGATIONS:
            return True
        if word not in NEGATION_BRIDGES:
            return False
    return False


def _hedged(text: str) -> bool:
    return any(_phrase_regex(p).search(text) for p in HEDGE_PHRASES)


def _last_after_correction(text: str, spans: dict[int, tuple[int, int]]) -> int | None:
    """Два варианта на разных участках, между ними «хотя нет» — берём
    последний. Без маркера поправки это противоречие, а не выбор."""
    ordered = sorted(spans.items(), key=lambda item: item[1][0])
    first_end = ordered[0][1][1]
    last_start = ordered[-1][1][0]
    between = text[first_end:last_start]
    if any(_phrase_regex(m).search(between) for m in CORRECTION_MARKERS):
        return ordered[-1][0]
    return None


def match_multi(
    utterance: str, options: Sequence[OptionGrammar], limit: int
) -> tuple[int, ...]:
    """Собрать все варианты, названные в одной реплике (вопросы «до N ответов»).

    Отличие от :func:`match_options` принципиальное. Там несовпадающие участки
    текста — противоречие: «да нет» не ответ. Здесь ровно наоборот: назвать
    несколько разных вариантов и есть правильное поведение респондента.

    Перекрытия разрешаются в пользу более широкого совпадения: во фразе
    «интернет, новостные сайты» сработают и «интернет — соцсети», и
    «интернет — новостные сайты», но участок один, и победить должен тот,
    что длиннее. Иначе один и тот же кусок речи засчитается дважды.

    Порядок сохраняется по позиции в реплике — первым назван первый ответ.
    Лишнее сверх `limit` отбрасывается: инструкция «до трёх ответов» —
    часть методики, а не пожелание.
    """
    text = normalize(utterance)
    hits: list[tuple[tuple[int, int], int]] = []
    for opt in options:
        best: tuple[int, int] | None = None
        for phrase in opt.phrases:
            found = _phrase_regex(phrase).search(text)
            if found and (best is None or _width(found.span()) > _width(best)):
                best = found.span()
        for rx in opt._compiled:
            found = rx.search(text)
            if found and (best is None or _width(found.span()) > _width(best)):
                best = found.span()
        if best is not None:
            hits.append((best, opt.code))

    hits.sort(key=lambda h: (-_width(h[0]), h[0][0]))
    taken: list[tuple[int, int]] = []
    chosen: list[tuple[tuple[int, int], int]] = []
    for span, code in hits:
        if any(_overlaps(span, other) for other in taken):
            continue
        taken.append(span)
        chosen.append((span, code))

    chosen.sort(key=lambda c: c[0][0])
    return tuple(code for _, code in chosen[:limit])


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _width(span: tuple[int, int]) -> int:
    return span[1] - span[0]


def _contains(outer: tuple[int, int], inner: tuple[int, int]) -> bool:
    return outer[0] <= inner[0] and inner[1] <= outer[1]
