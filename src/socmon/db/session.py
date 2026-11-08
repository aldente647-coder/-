"""Подключение к базе.

Строка подключения — только из окружения (ТЗ §14.3: секреты не в коде).
"""

from __future__ import annotations

import os

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

DATABASE_URL_VAR = "DATABASE_URL"
DEFAULT_URL = "postgresql+psycopg://localhost/socmon_dev"


def database_url() -> str:
    """Строка подключения из окружения.

    Одна переменная на всё: и миграции, и команды, и кабинет. Раньше их было
    две — alembic читал свою, команды свою, — и миграция бодро сообщала
    об успехе, применившись к другой базе. На сервере это тихая потеря
    данных, а не опечатка.
    """
    return os.environ.get(DATABASE_URL_VAR, DEFAULT_URL)


def make_engine(url: str | None = None, *, echo: bool = False) -> Engine:
    return create_engine(url or database_url(), echo=echo, future=True)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(engine, expire_on_commit=False)
