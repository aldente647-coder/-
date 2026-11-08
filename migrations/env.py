"""Окружение миграций.

Схемы (contact, survey, ops) создаются до применения миграций: Alembic сам
их не заводит, а без них первая же CREATE TABLE упадёт.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import text

from socmon.db.base import ALL_SCHEMAS
from socmon.db.models import Base
from socmon.db.session import database_url, make_engine

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def include_object(obj, name, type_, reflected, compare_to):  # type: ignore[no-untyped-def]
    """Не трогаем чужие схемы, если база делится с другими сервисами."""
    return not (type_ == "table" and obj.schema not in ALL_SCHEMAS)


def run_migrations_offline() -> None:
    context.configure(
        url=database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        include_schemas=True,
        include_object=include_object,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = make_engine()
    with engine.connect() as connection:
        for schema in ALL_SCHEMAS:
            connection.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))
        connection.commit()

        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_schemas=True,
            include_object=include_object,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
