"""Жизненный цикл волны и готовность к запуску.

Волна — единица отчётности по ТЗ §3.1, и её главное свойство в том, что она
**закрывается**. Внутрь идущей волны нельзя досыпать номера: рамка выборки
перестанет соответствовать той, по которой считались веса, а сравнение
с прошлой волной — тому, что обещано в §3.1.

Именно поэтому «вечная кампания» из блюпринта VOIF, подхватывающая новые
контакты на ходу, у нас запрещена по построению: из активной волны переход
ведёт только в паузу или в закрытие.

Проверка готовности — второй приём оттуда же, и его стоит взять дословно:
перед запуском система обязана показать, чего не хватает, а не молча
не поехать. Цена ошибки здесь высокая — запустив волну с устаревшей
озвучкой, мы обзвоним тысячу человек не тем вопросом.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class WaveError(RuntimeError):
    pass


#: Сколько номеров нужно на одно валидное интервью.
#:
#: Взято из порядка величин, ожидаемого для робо-обзвона (RESEARCH.md §2):
#: у ФОМ на всероссийском опросе по мобильным RR3 ≈ 13%, и это живой
#: интервьюер. Робот даёт кратно меньше — рабочая гипотеза 1,5-3% валидных
#: на набранный номер, отсюда сорокакратный запас.
#:
#: Число — допущение до пилота, а не измерение. Пилот его и заменит (ТЗ §15).
CONTACTS_PER_INTERVIEW = 40


class WaveStatus(StrEnum):
    DRAFT = "draft"
    """Готовится. Правится свободно, звонков нет."""

    ACTIVE = "active"
    """Идёт обзвон."""

    PAUSED = "paused"
    """Набор остановлен, волна не закрыта. Идущие разговоры доводятся
    до конца — обрывать живого респондента на середине нельзя."""

    CLOSED = "closed"
    """Закрыта. Состояние финальное: массив зафиксирован, и дописать
    в него интервью задним числом невозможно."""


_ALLOWED: dict[WaveStatus, frozenset[WaveStatus]] = {
    WaveStatus.DRAFT: frozenset({WaveStatus.ACTIVE, WaveStatus.CLOSED}),
    WaveStatus.ACTIVE: frozenset({WaveStatus.PAUSED, WaveStatus.CLOSED}),
    WaveStatus.PAUSED: frozenset({WaveStatus.ACTIVE, WaveStatus.CLOSED}),
    WaveStatus.CLOSED: frozenset(),
}


def can_transition(current: WaveStatus, target: WaveStatus) -> bool:
    return target in _ALLOWED[current]


def transition(current: WaveStatus, target: WaveStatus) -> WaveStatus:
    """Сменить состояние волны.

    Закрытая волна не открывается обратно. Это не строгость ради строгости:
    по закрытой волне уже сдан оперативный пакет (§13.1), и дописанное
    в неё интервью сделает выгрузку Заказчика неверной задним числом.
    """
    if current is target:
        return current
    if not can_transition(current, target):
        raise WaveError(
            f"волна в состоянии «{current.value}» не может перейти в «{target.value}»"
        )
    return target


# ------------------------------------------------------------- готовность


@dataclass(frozen=True, slots=True)
class Check:
    """Один пункт проверки готовности."""

    name: str
    passed: bool
    detail: str = ""
    blocking: bool = True
    """Незакрывающие пункты не мешают запуску, но должны быть видны."""


@dataclass(frozen=True, slots=True)
class Readiness:
    checks: tuple[Check, ...]

    @property
    def ready(self) -> bool:
        return all(c.passed for c in self.checks if c.blocking)

    @property
    def blockers(self) -> tuple[Check, ...]:
        return tuple(c for c in self.checks if c.blocking and not c.passed)

    @property
    def warnings(self) -> tuple[Check, ...]:
        return tuple(c for c in self.checks if not c.blocking and not c.passed)


@dataclass(frozen=True, slots=True)
class WaveSetup:
    """Что известно о волне на момент проверки."""

    has_questionnaire: bool
    questionnaire_valid: bool
    voice_missing: int
    voice_stale: int
    has_reference_distributions: bool
    caller_ids_active: int
    contacts_available: int
    target_min: int
    calling_window_configured: bool


def readiness(setup: WaveSetup) -> Readiness:
    """Проверить готовность волны к запуску."""
    return Readiness(checks=(
        Check(
            "Анкета привязана", setup.has_questionnaire,
            "" if setup.has_questionnaire else "к волне не привязана версия анкеты",
        ),
        Check(
            "Анкета проходит проверку", setup.questionnaire_valid,
            "" if setup.questionnaire_valid
            else "граф анкеты не связен или не все пути приводят к завершению",
        ),
        Check(
            "Озвучены все реплики", setup.voice_missing == 0,
            "" if setup.voice_missing == 0
            else f"не озвучено реплик: {setup.voice_missing}",
        ),
        Check(
            # Самый неочевидный и самый опасный пункт: сценарий выглядит
            # озвученным, робот читает прошлую редакцию вопроса, а в массив
            # пишется код по новой редакции.
            "Озвучка не устарела", setup.voice_stale == 0,
            "" if setup.voice_stale == 0
            else f"формулировка изменилась после озвучки у реплик: {setup.voice_stale}",
        ),
        Check(
            "Заданы эталонные распределения", setup.has_reference_distributions,
            "" if setup.has_reference_distributions
            else "без эталонов Заказчика взвешивание по ТЗ §5.4 невозможно",
        ),
        Check(
            "В карусели есть живые номера", setup.caller_ids_active > 0,
            "" if setup.caller_ids_active
            else "нет активных исходящих номеров",
        ),
        Check(
            "Настроено окно обзвона", setup.calling_window_configured,
            "" if setup.calling_window_configured
            else "не задано разрешённое время звонков (ТЗ §6.1)",
        ),
        Check(
            "Хватает контактов на целевой объём",
            setup.contacts_available >= setup.target_min * CONTACTS_PER_INTERVIEW,
            f"контактов {setup.contacts_available} при потребности около "
            f"{setup.target_min * CONTACTS_PER_INTERVIEW}: столько номеров "
            f"нужно, чтобы набрать {setup.target_min} валидных интервью "
            f"при достижимости робо-обзвона",
            blocking=False,
        ),
        Check(
            "Несколько номеров в карусели", setup.caller_ids_active > 1,
            "с одного номера операторы связи блокируют быстро",
            blocking=False,
        ),
    ))
