"""Диалер: цикл выбора контактов и набора.

Работает тиками. Каждый тик атомарно забирает у источника готовые к набору
контакты и отдаёт их движку телефонии. Именно этот цикл первым упирается
в CPU под нагрузкой, поэтому в блюпринте VOIF он вынесен из общих воркеров
в отдельный движок — и здесь он тоже сам по себе.

Три защиты работают одновременно и закрывают разные беды:

* **темп набора** не даёт захлебнуться собственным обзвоном;
* **карусель номеров** не даёт оператору связи заблокировать единственный
  номер — с одного номера при нашем расчёте это вопрос часов
  (CONCEPT.md §8.1);
* **предохранитель** останавливает волну при массовых технических обрывах,
  чтобы не прожечь попытки дозвона впустую, пока лежит транк.
"""

from __future__ import annotations

import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum

from socmon.core.codes import Disposition
from socmon.runtime.ports import CallerIdPool, Claim, ContactSource
from socmon.telephony.provider import (
    CallHandle,
    OriginateRequest,
    TelephonyError,
    TelephonyProvider,
)

Clock = Callable[[], datetime]


def _utcnow() -> datetime:
    return datetime.now(UTC)


class BreakerState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    PROBING = "probing"
    """Пробный звонок после паузы: прошёл — возобновляем, нет — ждём дальше."""


@dataclass(frozen=True, slots=True)
class DialerPolicy:
    max_concurrent: int = 60
    """Потолок одновременных звонков. Расчёт по ТЗ §5.2 и §6.1 даёт 60-100
    линий; конкретное число упирается в ёмкость транка."""

    calls_per_second: float = 2.0
    burst: float = 5
    originate_timeout_sec: int = 30

    failure_window: int = 30
    """По скольким последним звонкам считается доля обрывов."""

    failure_threshold: float = 0.5
    min_samples: int = 10
    """Раньше этого числа звонков предохранитель не срабатывает: на трёх
    неудачах подряд останавливать волну — значит останавливать её постоянно."""

    cooldown_sec: float = 120.0


DEFAULT_DIALER = DialerPolicy()

#: Причины, которые считаются отказом инфраструктуры, а не поведением абонента.
#:
#: Занято и «не берут трубку» сюда не входят намеренно: это нормальная жизнь
#: обзвона, и останавливать по ним волну нельзя.
INFRASTRUCTURE_FAILURES = frozenset({
    Disposition.TECH_ERROR,
    Disposition.NO_OPERATOR,
    Disposition.TIMEOUT,
})


@dataclass
class _TokenBucket:
    """Ограничитель темпа с допуском на короткие всплески."""

    rate: float
    capacity: float
    tokens: float = field(init=False)
    updated_at: datetime | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.tokens = self.capacity

    def take(self, now: datetime, wanted: int) -> int:
        if self.updated_at is not None:
            elapsed = (now - self.updated_at).total_seconds()
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.updated_at = now
        granted = min(wanted, int(self.tokens))
        self.tokens -= granted
        return granted


@dataclass
class CircuitBreaker:
    """Предохранитель на массовые технические обрывы."""

    policy: DialerPolicy = DEFAULT_DIALER
    state: BreakerState = BreakerState.CLOSED
    opened_at: datetime | None = None
    _recent: deque[bool] = field(default_factory=deque)

    def record(self, failed: bool, now: datetime) -> None:
        self._recent.append(failed)
        while len(self._recent) > self.policy.failure_window:
            self._recent.popleft()

        if self.state is BreakerState.PROBING:
            # Пробный звонок решает судьбу паузы.
            if failed:
                self.opened_at = now
                self.state = BreakerState.OPEN
            else:
                self.state = BreakerState.CLOSED
                self._recent.clear()
            return

        if self.state is BreakerState.CLOSED and self._tripped():
            self.state = BreakerState.OPEN
            self.opened_at = now

    def _tripped(self) -> bool:
        if len(self._recent) < self.policy.min_samples:
            return False
        return sum(self._recent) / len(self._recent) > self.policy.failure_threshold

    def allowance(self, now: datetime, slots: int) -> int:
        """Сколько звонков разрешено начать прямо сейчас."""
        if self.state is BreakerState.CLOSED:
            return slots
        if self.state is BreakerState.PROBING:
            return 0
        assert self.opened_at is not None
        if now - self.opened_at < timedelta(seconds=self.policy.cooldown_sec):
            return 0
        # Пауза вышла: пробуем ровно один звонок.
        self.state = BreakerState.PROBING
        return min(1, slots)

    @property
    def is_open(self) -> bool:
        return self.state is not BreakerState.CLOSED

    @property
    def failure_rate(self) -> float:
        if not self._recent:
            return 0.0
        return sum(self._recent) / len(self._recent)


@dataclass(frozen=True, slots=True)
class TickReport:
    claimed: int
    originated: int
    failed: int
    breaker: BreakerState
    active: int
    reason: str | None = None


class Dialer:
    """Цикл набора по одной волне."""

    def __init__(
        self,
        provider: TelephonyProvider,
        source: ContactSource,
        caller_ids: CallerIdPool,
        *,
        wave_id: uuid.UUID,
        on_originated: Callable[[uuid.UUID, CallHandle], None],
        policy: DialerPolicy = DEFAULT_DIALER,
        clock: Clock = _utcnow,
    ) -> None:
        self._provider = provider
        self._source = source
        self._caller_ids = caller_ids
        self._wave_id = wave_id
        self._on_originated = on_originated
        self._policy = policy
        self._clock = clock
        self._bucket = _TokenBucket(policy.calls_per_second, policy.burst)
        self.breaker = CircuitBreaker(policy)
        self._active: set[uuid.UUID] = set()

    @property
    def active(self) -> int:
        return len(self._active)

    def release(self, attempt_id: uuid.UUID, disposition: Disposition | None) -> None:
        """Сообщить диалеру, что звонок закончился.

        Предохранитель считает только инфраструктурные отказы: занято
        и недозвон — нормальная жизнь обзвона, останавливать по ним волну
        нельзя.
        """
        self._active.discard(attempt_id)
        self.breaker.record(disposition in INFRASTRUCTURE_FAILURES, self._clock())

    async def tick(self) -> TickReport:
        """Один цикл набора."""
        now = self._clock()
        free = self._policy.max_concurrent - len(self._active)
        if free <= 0:
            return TickReport(0, 0, 0, self.breaker.state, self.active,
                              "потолок одновременных звонков")

        allowed = self.breaker.allowance(now, free)
        if allowed <= 0:
            return TickReport(0, 0, 0, self.breaker.state, self.active,
                              f"предохранитель: доля обрывов {self.breaker.failure_rate:.0%}")

        if self._caller_ids.available == 0:
            return TickReport(0, 0, 0, self.breaker.state, self.active,
                              "в карусели не осталось живых номеров")

        granted = self._bucket.take(now, allowed)
        if granted <= 0:
            return TickReport(0, 0, 0, self.breaker.state, self.active, "темп набора")

        claims = list(self._source.claim(self._wave_id, granted, now))
        originated = failed = 0
        for claim in claims:
            if await self._dial(claim, now):
                originated += 1
            else:
                failed += 1

        return TickReport(
            claimed=len(claims),
            originated=originated,
            failed=failed,
            breaker=self.breaker.state,
            active=self.active,
        )

    async def _dial(self, claim: Claim, now: datetime) -> bool:
        caller_id = self._caller_ids.take()
        try:
            handle = await self._provider.originate(OriginateRequest(
                attempt_id=str(claim.attempt_id),
                destination=claim.phone,
                caller_id=caller_id,
                timeout_sec=self._policy.originate_timeout_sec,
            ))
        except TelephonyError:
            # Набрать не удалось — контакт возвращается в очередь, попытка
            # не засчитывается. Иначе сбой транка съедал бы лимит попыток
            # на номер, отведённый регламентом (ТЗ §6.2).
            self._source.release(claim.attempt_id, now)
            self.breaker.record(True, now)
            return False

        self._active.add(claim.attempt_id)
        self._on_originated(claim.attempt_id, handle)
        return True


class RoundRobinCallerIds:
    """Простая карусель: номера по кругу, выгоревшие выбывают.

    Выгоревший номер не удаляется, а помечается: история звонков с него
    должна остаться (ТЗ §10.1, и то же правило в блюпринте VOIF).
    """

    def __init__(self, numbers: list[str]) -> None:
        if not numbers:
            raise ValueError("пустой пул исходящих номеров")
        self._numbers = list(numbers)
        self._burned: set[str] = set()
        self._cursor = 0

    def take(self) -> str:
        alive = [n for n in self._numbers if n not in self._burned]
        if not alive:
            raise TelephonyError("все исходящие номера выведены из ротации")
        number = alive[self._cursor % len(alive)]
        self._cursor += 1
        return number

    def burn(self, caller_id: str) -> None:
        self._burned.add(caller_id)

    @property
    def available(self) -> int:
        return len(self._numbers) - len(self._burned)
