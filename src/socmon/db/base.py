"""Базовые определения схемы.

Контуры разведены по трём схемам PostgreSQL, и это не косметика, а исполнение
ТЗ §14.2 («раздельное хранение телефонных номеров и массива ответов»):

* ``contact`` — номера, попытки дозвона, стоп-лист. Доступ минимальный.
* ``survey``  — обезличенный массив: интервью, ответы, кодирование. Здесь
  работает аналитик, и телефонов тут нет ни в одном столбце.
* ``ops``     — эксплуатация: пул исходящих номеров, журнал действий.

Единственная точка соединения контуров — таблица ``contact.identity_link``.
Права на неё выдаются отдельно от прав на остальные таблицы, поэтому
разделение — граница доступа, а не соглашение.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, ClassVar

from sqlalchemy import DateTime, MetaData, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

CONTACT_SCHEMA = "contact"
SURVEY_SCHEMA = "survey"
OPS_SCHEMA = "ops"
ALL_SCHEMAS = (CONTACT_SCHEMA, SURVEY_SCHEMA, OPS_SCHEMA)

#: Единые соглашения об именах: без них autogenerate плодит имена вида
#: `ck_…_1`, и миграции перестают быть читаемыми и переносимыми.
NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_N_name)s",
    "pk": "pk_%(table_name)s",
}

TZDateTime = DateTime(timezone=True)
"""Время только с зоной. Наивное время в этой системе — это ночные звонки
и разъезжающиеся волны."""


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class TimestampedMixin:
    created_at: Mapped[datetime] = mapped_column(
        TZDateTime, nullable=False, server_default=func.now()
    )


class VersionedMixin:
    """Оптимистичная блокировка.

    Прямо из блюпринта VOIF: без счётчика версии конкурентные обновления
    затирают друг друга, звонки начинают «мерцать» между статусами, а один
    контакт уходит в набор дважды. Чинить это задним числом дороже, чем
    заложить сразу.
    """

    version: Mapped[int] = mapped_column(nullable=False, default=1)

    __mapper_args__: ClassVar[dict[str, Any]] = {"version_id_col": version}
