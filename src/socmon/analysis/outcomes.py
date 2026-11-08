"""Достижимость респондентов по стандарту AAPOR.

ТЗ §13.2 требует давать по каждой волне сведения о числе попыток, дозвонах,
отказах, прерываниях, валидных и невалидных интервью. Считать это по
самодельным формулам нельзя: цифру «доля ответивших» невозможно сопоставить
ни с чем, если не сказано, как она получена.

Поэтому берётся AAPOR Standard Definitions — тот же стандарт, на который
ссылаются российские полстеры. ВЦИОМ публикует достижимость по каждой волне;
делаем так же: цифра, зафиксированная и объяснённая заранее, снимает половину
будущих споров о качестве выборки.

Ориентир: у ФОМ на всероссийском опросе по мобильным телефонам RR3 ≈ 13%,
и это живой интервьюер. Робот даёт кратно меньше — сравнивать наши цифры
надо с этим ориентиром, а не с академическими опросами лицом к лицу.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from socmon.core.codes import AaporCode

#: Группы итоговых кодов AAPOR.
INTERVIEW = frozenset({AaporCode.COMPLETE})
PARTIAL = frozenset({AaporCode.PARTIAL})
REFUSAL = frozenset({AaporCode.REFUSAL, AaporCode.BREAK_OFF})
NON_CONTACT = frozenset({AaporCode.NON_CONTACT})
OTHER_ELIGIBLE = frozenset({AaporCode.OTHER_ELIGIBLE, AaporCode.LANGUAGE_BARRIER})
UNKNOWN = frozenset({AaporCode.UNKNOWN_ELIGIBILITY})
NOT_ELIGIBLE = frozenset({AaporCode.NOT_ELIGIBLE})
OUT_OF_SAMPLE = frozenset({AaporCode.OUT_OF_SAMPLE})


class OutcomeError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Dispositions:
    """Итоговые исходы обзвона, сгруппированные по AAPOR."""

    interviews: int = 0
    partials: int = 0
    refusals: int = 0
    non_contacts: int = 0
    other_eligible: int = 0
    unknown_eligibility: int = 0
    not_eligible: int = 0
    out_of_sample: int = 0

    @classmethod
    def from_codes(cls, codes: Iterable[AaporCode]) -> Dispositions:
        counter = Counter(codes)

        def take(group: frozenset[AaporCode]) -> int:
            return sum(counter[code] for code in group)

        return cls(
            interviews=take(INTERVIEW),
            partials=take(PARTIAL),
            refusals=take(REFUSAL),
            non_contacts=take(NON_CONTACT),
            other_eligible=take(OTHER_ELIGIBLE),
            unknown_eligibility=take(UNKNOWN),
            not_eligible=take(NOT_ELIGIBLE),
            out_of_sample=take(OUT_OF_SAMPLE),
        )

    @property
    def eligible(self) -> int:
        """Заведомо пригодные: с ними состоялся контакт и они подошли."""
        return (
            self.interviews + self.partials + self.refusals
            + self.non_contacts + self.other_eligible
        )

    @property
    def total(self) -> int:
        """Все номера в работе. Недействующие номера (4.10) сюда не входят:
        они не часть выборки, а брак базы."""
        return self.eligible + self.unknown_eligibility + self.not_eligible

    @property
    def eligibility_rate(self) -> float:
        """Доля пригодных среди тех, чью пригодность удалось установить (`e`).

        Оценка по методу CASRO: предполагается, что среди номеров неизвестной
        пригодности доля подходящих такая же, как среди установленных. Это
        предположение, а не факт, и в отчёте оно должно быть названо.
        """
        known = self.eligible + self.not_eligible
        if known == 0:
            raise OutcomeError("ни по одному номеру не установлена пригодность")
        return self.eligible / known


@dataclass(frozen=True, slots=True)
class ResponseRates:
    """Коэффициенты AAPOR. Каждый отвечает на свой вопрос, и подменять
    один другим — самый простой способ приукрасить достижимость."""

    rr1: float
    """Строгая доля ответивших: только полные интервью, все номера
    неизвестной пригодности считаются пригодными."""

    rr2: float
    """То же, но полные и частичные интервью вместе."""

    rr3: float
    """Доля ответивших с оценкой пригодности `e`. Основной публикуемый
    показатель — его же приводит ФОМ."""

    rr4: float
    rr5: float
    """Верхняя граница: номера неизвестной пригодности исключены вовсе.
    Публиковать её как «долю ответивших» — лукавство."""

    rr6: float
    cooperation1: float
    """Доля согласившихся среди тех, с кем реально поговорили."""

    refusal1: float
    contact1: float
    """Доля номеров, по которым удалось выйти на человека."""

    eligibility_rate: float


def response_rates(d: Dispositions) -> ResponseRates:
    """Рассчитать коэффициенты AAPOR."""
    if d.total == 0:
        raise OutcomeError("пустой массив исходов")

    i, p = d.interviews, d.partials
    contacted_refusals = d.refusals
    r_nc_o = d.refusals + d.non_contacts + d.other_eligible
    unknown = d.unknown_eligibility
    e = d.eligibility_rate

    full = (i + p) + r_nc_o + unknown
    estimated = (i + p) + r_nc_o + e * unknown
    eligible_only = (i + p) + r_nc_o
    contacted = (i + p) + contacted_refusals + d.other_eligible

    return ResponseRates(
        rr1=_ratio(i, full),
        rr2=_ratio(i + p, full),
        rr3=_ratio(i, estimated),
        rr4=_ratio(i + p, estimated),
        rr5=_ratio(i, eligible_only),
        rr6=_ratio(i + p, eligible_only),
        cooperation1=_ratio(i, contacted),
        refusal1=_ratio(contacted_refusals, full),
        contact1=_ratio(contacted, full),
        eligibility_rate=e,
    )


def _ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


@dataclass(frozen=True, slots=True)
class CallStatistics:
    """Сведения о ходе обзвона (ТЗ §13.2)."""

    numbers_dialled: int
    attempts_made: int
    dispositions: Dispositions
    rates: ResponseRates
    valid_interviews: int
    invalid_interviews: int
    exclusion_reasons: dict[str, int]

    @property
    def attempts_per_number(self) -> float:
        if self.numbers_dialled == 0:
            return 0.0
        return self.attempts_made / self.numbers_dialled

    @property
    def notes(self) -> tuple[str, ...]:
        notes = [
            "Достижимость рассчитана по AAPOR Standard Definitions. "
            "Основной показатель — RR3; он учитывает оценочную долю пригодных "
            f"среди номеров неизвестной пригодности (e = {self.rates.eligibility_rate:.3f}).",
        ]
        if self.dispositions.unknown_eligibility > self.dispositions.eligible:
            notes.append(
                "По большинству номеров пригодность установить не удалось: "
                "оценка RR3 сильно зависит от допущения о доле пригодных среди них."
            )
        if self.dispositions.out_of_sample:
            notes.append(
                f"Недействующих номеров в базе: {self.dispositions.out_of_sample}. "
                f"В расчёт достижимости они не входят — это качество базы, "
                f"а не отказ респондентов."
            )
        return tuple(notes)
