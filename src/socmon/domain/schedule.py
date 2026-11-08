"""Регламент дозвона: окна звонков и повторные попытки.

ТЗ §6.1 задаёт окна (будни 18:00-20:00, выходные 12:00-20:00), §6.2 —
регламент повторов. Это самый жёсткий по последствиям модуль: ошибка здесь —
звонок ночью, то есть жалоба, стоп-лист и репутационный ущерб Заказчику.

Окна считаются в местном времени территории, а не в UTC: 20:00 в Москве и
20:00 в UTC — разные вещи, и путаница даёт ровно ночные звонки.

Разброс времени попыток внутри окна детерминирован: смещение выводится из
идентификатора контакта и номера попытки. Требование §6.2 «в различные
временные точки» выполняется, при этом план дозвона воспроизводим — что важно
и для разбора инцидентов, и для §12 (воспроизводимость расчётов).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from hashlib import sha256
from zoneinfo import ZoneInfo


@dataclass(frozen=True, slots=True)
class DailyWindow:
    """Разрешённый интервал звонков в течение суток."""

    start: time
    end: time

    def __post_init__(self) -> None:
        if self.start >= self.end:
            raise ValueError("начало окна должно быть раньше конца")

    @property
    def minutes(self) -> int:
        return (self.end.hour - self.start.hour) * 60 + (self.end.minute - self.start.minute)


@dataclass(frozen=True, slots=True)
class CallPolicy:
    """Регламент обзвона (ТЗ §6.1, §6.2)."""

    timezone: str = "Europe/Moscow"
    weekday: DailyWindow = DailyWindow(time(18, 0), time(20, 0))
    weekend: DailyWindow = DailyWindow(time(12, 0), time(20, 0))
    max_attempts: int = 5
    """ТЗ §6.2 допускает 3-5. Берём верх диапазона: у ВЦИОМ-Спутник — не менее
    пяти попыток, и каждая заметно поднимает достижимость."""

    min_interval_hours: int = 48
    """«Интервал между попытками — не менее 48 часов» (ТЗ §6.2)."""

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)

    def window_for(self, day: date) -> DailyWindow:
        return self.weekend if day.weekday() >= 5 else self.weekday

    def is_open(self, moment: datetime) -> bool:
        """Открыто ли окно звонков в этот момент."""
        local = self._local(moment)
        window = self.window_for(local.date())
        return window.start <= local.time() < window.end

    def next_open(self, moment: datetime) -> datetime:
        """Ближайший момент, когда звонить можно.

        Если окно уже открыто — возвращает сам момент; звонок, готовый прямо
        сейчас, не нужно откладывать до завтра.
        """
        local = self._local(moment)
        for offset in range(0, 8):
            day = local.date() + timedelta(days=offset)
            window = self.window_for(day)
            opening = datetime.combine(day, window.start, tzinfo=self.tz)
            if offset == 0 and self.is_open(moment):
                return moment
            if opening > local:
                return opening
        raise AssertionError("за неделю не нашлось ни одного открытого окна")

    def _local(self, moment: datetime) -> datetime:
        if moment.tzinfo is None:
            raise ValueError("момент времени обязан быть с зоной: наивное время даёт ночные звонки")
        return moment.astimezone(self.tz)


#: Регламент по умолчанию. Отдельная константа, а не вызов в аргументе:
#: политика неизменяема, и один общий экземпляр честнее пересоздания.
DEFAULT_POLICY = CallPolicy()


#: Причины недозвона, по которым повтор осмыслен.
#:
#: Технические сбои сюда намеренно не входят — урок блюпринта VOIF: повторяя
#: их, система маскирует собственную поломку вместо того, чтобы о ней
#: сигнализировать. Явный отказ и просьба не звонить не повторяются никогда
#: (ТЗ §6.2).
RETRYABLE: frozenset[str] = frozenset({"no_answer", "busy", "answering_machine", "dropped"})


@dataclass(frozen=True, slots=True)
class AttemptPlan:
    """Когда делать следующую попытку — или почему её не будет."""

    at: datetime | None
    reason: str | None = None

    @property
    def scheduled(self) -> bool:
        return self.at is not None


def plan_next_attempt(
    *,
    contact_key: str,
    attempts_made: int,
    last_attempt_at: datetime,
    last_disposition: str | None,
    policy: CallPolicy = DEFAULT_POLICY,
    now: datetime | None = None,
) -> AttemptPlan:
    """Запланировать следующую попытку по номеру.

    `contact_key` нужен только как источник детерминированного разброса времени;
    расшифровывать номер для планирования не требуется — годится слепой индекс
    в hex.
    """
    if last_disposition is not None and last_disposition not in RETRYABLE:
        return AttemptPlan(None, f"причина «{last_disposition}» не подлежит повтору")
    if attempts_made >= policy.max_attempts:
        return AttemptPlan(None, f"исчерпан лимит попыток ({policy.max_attempts})")

    earliest = last_attempt_at + timedelta(hours=policy.min_interval_hours)
    if now is not None and now > earliest:
        earliest = now

    candidate = policy.next_open(earliest)
    shifted = _shift_within_window(candidate, contact_key, attempts_made, policy)
    return AttemptPlan(shifted)


def _shift_within_window(
    opening: datetime, contact_key: str, attempt_no: int, policy: CallPolicy
) -> datetime:
    """Развести попытки по разным точкам внутри окна (ТЗ §6.2).

    Смещение выводится из хеша пары (контакт, номер попытки): у одного абонента
    попытки приходятся на разное время суток, а весь план при этом
    воспроизводим по тем же входным данным.

    Разброс считается от самого раннего допустимого момента до конца окна, а не
    от его открытия. Считать от открытия и потом обрезать снизу нельзя: когда
    интервал в 48 часов приходится на середину окна, обрезка съедает весь
    разброс и все попытки сходятся в одну и ту же минуту.
    """
    local = opening.astimezone(policy.tz)
    window = policy.window_for(local.date())
    closes_at = datetime.combine(local.date(), window.end, tzinfo=policy.tz)

    remaining_min = int((closes_at - local).total_seconds() // 60)
    if remaining_min <= 1:
        return local

    seed = sha256(f"{contact_key}:{attempt_no}".encode()).digest()
    return local + timedelta(minutes=int.from_bytes(seed[:4], "big") % remaining_min)
