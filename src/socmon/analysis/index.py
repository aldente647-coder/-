"""Индекс социального напряжения (ТЗ §12).

Формула из ТЗ: ``I = (E + F + A + J + D + C + P) / 7``, компоненты
предварительно приводятся к шкале 0-10. Здесь она реализована буквально,
но с тремя обязательными по §12 добавками:

* **методика нормирования зафиксирована** — каждый компонент несёт своё
  правило приведения к 0-10, и правило это описано данными, а не кодом;
* **правила обработки отказов зафиксированы** — по умолчанию отказ
  и «затрудняюсь» исключаются из базы компонента. Замена пропусков средним
  требует явного согласования и включается отдельным флагом;
* **расчёт воспроизводим** — методика сворачивается в хеш, который уходит
  в выгрузку вместе с числом.

Ориентация шкалы у всех компонентов одна: **выше значение — выше напряжение**.
Поэтому «удовлетворённость жизнью» и «доверие институтам» инвертируются,
а «тревожность» и «конфликтность» берутся как есть. Перепутать направление —
самая дорогая ошибка в этом модуле: индекс поедет в противоположную сторону
и при этом останется правдоподобным.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from socmon.analysis.dataset import Dataset, Record
from socmon.analysis.estimates import (
    Estimate,
    EstimationError,
    _assemble,
    _weighted_mean,
    _weighted_variance,
)
from socmon.core.codes import MAX_OPTION_CODE

INDEX_SCALE_LOW = 0.0
INDEX_SCALE_HIGH = 10.0


class IndexError_(ValueError):
    """Индекс посчитать нельзя."""


class MissingPolicy(StrEnum):
    EXCLUDE = "exclude"
    """Отказ и затруднение выпадают из базы компонента. По умолчанию."""

    NEUTRAL = "neutral"
    """Пропуски приравниваются к середине шкалы.

    ТЗ §12: «не заменять пропуски средними значениями без явного
    согласования». Включается только осознанно и попадает в методику
    расчёта, то есть меняет хеш.
    """


@dataclass(frozen=True, slots=True)
class ScaleComponent:
    """Компонент из числовой шкалы. Линейное приведение к 0-10."""

    key: str
    name: str
    core_key: str
    low: int = 0
    high: int = 10
    invert: bool = False
    """True, если высокое значение исходной шкалы означает НИЗКОЕ напряжение."""

    def normalize(self, code: int) -> float:
        if not self.low <= code <= self.high:
            raise IndexError_(f"{self.core_key}: код {code} вне шкалы {self.low}-{self.high}")
        position = (code - self.low) / (self.high - self.low)
        if self.invert:
            position = 1.0 - position
        return INDEX_SCALE_LOW + position * (INDEX_SCALE_HIGH - INDEX_SCALE_LOW)

    def accepts(self, code: int) -> bool:
        return self.low <= code <= self.high

    def recipe(self) -> dict[str, object]:
        return {
            "key": self.key, "core_key": self.core_key, "kind": "scale",
            "low": self.low, "high": self.high, "invert": self.invert,
        }


@dataclass(frozen=True, slots=True)
class CategoricalComponent:
    """Компонент из вопроса с вариантами. Балл каждого варианта задан явно.

    Именно явно, а не формулой: расстояния между «скорее готов» и «скорее не
    готов» не выводятся из номеров вариантов, их назначает методолог, и это
    решение должно быть видно в методике.
    """

    key: str
    name: str
    core_key: str
    scores: Mapping[int, float]

    def __post_init__(self) -> None:
        if not self.scores:
            raise IndexError_(f"{self.core_key}: не заданы баллы вариантов")
        for code, score in self.scores.items():
            if not 1 <= code <= MAX_OPTION_CODE:
                raise IndexError_(f"{self.core_key}: код {code} вне зоны вариантов ответа")
            if not INDEX_SCALE_LOW <= score <= INDEX_SCALE_HIGH:
                raise IndexError_(f"{self.core_key}: балл {score} вне шкалы 0-10")

    def normalize(self, code: int) -> float:
        try:
            return self.scores[code]
        except KeyError as exc:
            raise IndexError_(f"{self.core_key}: для кода {code} не задан балл") from exc

    def accepts(self, code: int) -> bool:
        return code in self.scores

    def recipe(self) -> dict[str, object]:
        return {
            "key": self.key, "core_key": self.core_key, "kind": "categorical",
            "scores": {str(k): v for k, v in sorted(self.scores.items())},
        }


Component = ScaleComponent | CategoricalComponent


#: Компонентный состав индекса по ТЗ §12.
#:
#: Подлежит окончательному утверждению Заказчиком — как и сама формула.
#: До утверждения это рабочая методика, а не догма: у семи разнородных
#: компонентов равные веса методологически спорны (см. RESEARCH.md §3).
DEFAULT_COMPONENTS: tuple[Component, ...] = (
    ScaleComponent("E", "Негативная оценка общей ситуации", "life_situation", invert=True),
    ScaleComponent("F", "Материальная уязвимость", "material_stability", invert=True),
    ScaleComponent("A", "Тревожность", "anxiety"),
    ScaleComponent("J", "Ощущение несправедливости", "fairness", invert=True),
    ScaleComponent("D", "Недоверие к институтам", "institutional_trust", invert=True),
    ScaleComponent("C", "Воспринимаемая конфликтность", "perceived_conflict"),
    CategoricalComponent(
        "P", "Готовность к законным коллективным действиям", "collective_action",
        # 1 готов, 2 скорее готов, 3 скорее не готов, 4 не готов — равными шагами.
        scores={1: 10.0, 2: 20 / 3, 3: 10 / 3, 4: 0.0},
    ),
)


#: Компоненты индекса на анкете Заказчика.
#:
#: В отличие от эталонной анкеты, где каждый компонент снимался шкалой 0-10,
#: здесь всё меряется вариантами ответа, и балл каждого варианта назначен
#: явно. Два следствия, о которых нужно знать, читая число:
#:
#: * **Компонент из двух вариантов даёт только 0 или 10.** Среднее по нему —
#:   это доля негативных ответов, умноженная на десять, и меняется оно
#:   скачками. Плавную динамику по такому компоненту ловить бессмысленно.
#: * **Слагаемых шесть, а не семь.** ТЗ §12 приводит формулу с компонентом P —
#:   готовностью к законным коллективным действиям. Вопроса про неё в анкете
#:   Заказчика нет, и Заказчик согласовал расчёт по шести компонентам.
#:   Делитель, соответственно, шесть: доливать седьмое слагаемое нулём или
#:   средним значило бы занижать или размывать индекс молча.
#:
#: Все шесть компонентов взяты из постоянного ядра. Это не совпадение:
#: компонент индекса не может стоять в переменном блоке, где вопрос вправе
#: исчезнуть в очередной волне и оборвать ряд. Ради этого вопрос
#: о справедливости и перенесён из переменного блока в ядро.
MONITORING_COMPONENTS: tuple[Component, ...] = (
    CategoricalComponent(
        "E", "Негативная оценка общей ситуации", "region_satisfaction",
        scores={1: 0.0, 2: 10.0},
    ),
    CategoricalComponent(
        "F", "Материальная уязвимость", "material_status",
        # 1 хватает на всё, 2 на повседневное, 3 только на необходимое,
        # 4 не хватает и на необходимое — равными шагами.
        scores={1: 0.0, 2: 10 / 3, 3: 20 / 3, 4: 10.0},
    ),
    CategoricalComponent(
        "A", "Тревожность", "anxiety_dynamics",
        scores={1: 10.0, 2: 5.0, 3: 0.0},
    ),
    CategoricalComponent(
        "J", "Ощущение несправедливости", "fairness",
        scores={1: 0.0, 2: 10.0},
    ),
    CategoricalComponent(
        "D", "Недоверие к властям Республики", "trust_republic",
        scores={1: 0.0, 2: 10.0},
    ),
    CategoricalComponent(
        "C", "Ощущение небезопасности", "safety",
        scores={1: 0.0, 2: 10.0},
    ),
)


@dataclass(frozen=True, slots=True)
class ComponentResult:
    key: str
    name: str
    core_key: str
    estimate: Estimate
    missing_share: float
    """Доля отказов и затруднений по этому компоненту. Высокая доля —
    сигнал не о напряжении, а о неудачной формулировке вопроса."""

    @property
    def value(self) -> float:
        return self.estimate.value


@dataclass(frozen=True, slots=True)
class IndexResult:
    """Индекс, компонентный профиль и всё, что нужно для §12 и §13.2."""

    aggregate: float
    """I по формуле ТЗ: среднее компонентных средних. Каждый компонент
    считается по своей максимальной базе."""

    individual: Estimate | None
    """Тот же индекс, посчитанный по полным наблюдениям. Даёт доверительный
    интервал и позволяет разрезы по подгруппам, но база у него меньше."""

    components: tuple[ComponentResult, ...]
    n: int
    complete_cases: int
    missing_policy: MissingPolicy
    recipe_hash: str
    notes: tuple[str, ...] = ()

    @property
    def profile(self) -> dict[str, float]:
        """Компонентный профиль — то, что ТЗ §12 требует давать наряду
        с единым баллом."""
        return {c.key: round(c.value, 3) for c in self.components}


def compute(
    records: Dataset,
    components: Sequence[Component] = DEFAULT_COMPONENTS,
    *,
    missing_policy: MissingPolicy = MissingPolicy.EXCLUDE,
    neutral_substitution_approved_by: str | None = None,
) -> IndexResult:
    """Рассчитать индекс и компонентный профиль.

    Замена пропусков серединой шкалы требует имени согласовавшего: ТЗ §12
    запрещает такую замену «без явного согласования», а согласование без
    следа в данных согласованием не является.
    """
    if not records:
        raise IndexError_("пустой массив")
    if not components:
        raise IndexError_("не задан состав компонентов")
    if missing_policy is MissingPolicy.NEUTRAL and not neutral_substitution_approved_by:
        raise IndexError_(
            "замена пропусков серединой шкалы требует явного согласования "
            "Заказчика (ТЗ §12): укажите, кем оно дано"
        )

    scored = [_score_records(records, c, missing_policy) for c in components]
    results = tuple(
        _summarize(component, values, len(records))
        for component, values in zip(components, scored, strict=True)
    )

    aggregate = sum(r.value for r in results) / len(results)
    individual, complete = _individual_index(records, components, missing_policy)

    return IndexResult(
        aggregate=aggregate,
        individual=individual,
        components=results,
        n=len(records),
        complete_cases=complete,
        missing_policy=missing_policy,
        recipe_hash=recipe_hash(components, missing_policy),
        notes=_notes(results, aggregate, individual, complete, len(records)),
    )


# ------------------------------------------------------------------ внутреннее

_NEUTRAL = (INDEX_SCALE_LOW + INDEX_SCALE_HIGH) / 2


def _score_one(
    record: Record, component: Component, policy: MissingPolicy
) -> float | None:
    code = record.get(component.core_key)
    if code is not None and component.accepts(code):
        return component.normalize(code)
    return _NEUTRAL if policy is MissingPolicy.NEUTRAL else None


def _score_records(
    records: Dataset, component: Component, policy: MissingPolicy
) -> list[tuple[float, float]]:
    scored: list[tuple[float, float]] = []
    for record in records:
        score = _score_one(record, component, policy)
        if score is not None:
            scored.append((score, record.weight))
    return scored


def _summarize(
    component: Component, scored: Sequence[tuple[float, float]], base_n: int
) -> ComponentResult:
    if not scored:
        raise IndexError_(
            f"{component.core_key}: ни одного содержательного ответа, "
            f"компонент «{component.name}» посчитать нельзя"
        )
    values = [v for v, _ in scored]
    weights = [w for _, w in scored]
    estimate = _estimate_of(values, weights, base_n)
    return ComponentResult(
        key=component.key,
        name=component.name,
        core_key=component.core_key,
        estimate=estimate,
        missing_share=1 - len(scored) / base_n if base_n else 0.0,
    )


def _estimate_of(values: Sequence[float], weights: Sequence[float], base_n: int) -> Estimate:
    mean = _weighted_mean(values, weights)
    variance = _weighted_variance(values, weights, mean)
    return _assemble(
        mean, variance, weights,
        base_n=base_n, clamp=(INDEX_SCALE_LOW, INDEX_SCALE_HIGH),
    )


def _individual_index(
    records: Dataset, components: Sequence[Component], policy: MissingPolicy
) -> tuple[Estimate | None, int]:
    """Индекс по полным наблюдениям.

    Наблюдение считается полным, когда есть балл по каждому компоненту.
    Собирать индивидуальный индекс из частично отвеченных анкет нельзя:
    у респондента, ответившего на три компонента из семи, получилось бы
    среднее по другой шкале.
    """
    values: list[float] = []
    weights: list[float] = []
    for record in records:
        scores = [_score_one(record, c, policy) for c in components]
        if any(s is None for s in scores):
            continue
        values.append(sum(s for s in scores if s is not None) / len(scores))
        weights.append(record.weight)

    if not values:
        return None, 0
    try:
        return _estimate_of(values, weights, len(records)), len(values)
    except EstimationError:
        return None, len(values)


def _notes(
    components: Sequence[ComponentResult],
    aggregate: float,
    individual: Estimate | None,
    complete: int,
    n: int,
) -> tuple[str, ...]:
    notes: list[str] = []
    for component in components:
        if component.missing_share > 0.2:
            notes.append(
                f"компонент {component.key} «{component.name}»: "
                f"{component.missing_share:.1%} отказов и затруднений — "
                f"проверить формулировку вопроса, а не только сам показатель"
            )
    if n and complete / n < 0.7:
        notes.append(
            f"полных наблюдений {complete} из {n}: индекс по индивидуальным "
            f"данным опирается на заметно суженную базу"
        )
    if individual is not None and abs(individual.value - aggregate) > 0.3:
        notes.append(
            f"агрегатный индекс {aggregate:.2f} и индивидуальный "
            f"{individual.value:.2f} расходятся: отказы распределены "
            f"неслучайно по компонентам"
        )
    return tuple(notes)


def recipe_hash(
    components: Sequence[Component], missing_policy: MissingPolicy
) -> str:
    """Отпечаток методики расчёта.

    Уходит в выгрузку рядом с числом (ТЗ §12: «обеспечить воспроизводимость
    расчёта в выгрузке»). Изменили направление шкалы или балл варианта —
    хеш другой, и старые цифры не спутаешь с новыми.
    """
    payload = {
        "formula": "mean",
        "scale": [INDEX_SCALE_LOW, INDEX_SCALE_HIGH],
        "missing_policy": missing_policy.value,
        "components": [c.recipe() for c in components],
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
