from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import Engine, create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session
from sqlalchemy.schema import CreateSchema, DropSchema

from socmon.core.questionnaire import Questionnaire
from socmon.db.base import ALL_SCHEMAS, Base
from socmon.db.models import *  # noqa: F403  регистрация всех таблиц в metadata

REFERENCE = Path(__file__).resolve().parents[1] / "src/socmon/questionnaires/core-v1.yaml"


@pytest.fixture(scope="session")
def core() -> Questionnaire:
    return Questionnaire.from_yaml(REFERENCE)


class FakeClock:
    """Управляемое время: длительность интервью — измеряемая величина (ТЗ §3.3),
    её нельзя оставлять на усмотрение реальных часов в тестах."""

    def __init__(self, step_sec: float = 1.0) -> None:
        self._now = datetime(2026, 8, 20, 18, 0, 0, tzinfo=UTC)
        self._step = timedelta(seconds=step_sec)

    def __call__(self) -> datetime:
        self._now += self._step
        return self._now


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


# ---------------------------------------------------------------- база данных

TEST_DATABASE_URL = os.environ.get(
    "SOCMON_TEST_DATABASE_URL", "postgresql+psycopg://localhost/socmon_test"
)


def _postgres_available() -> bool:
    try:
        engine = create_engine(TEST_DATABASE_URL)
        with engine.connect():
            return True
    except OperationalError:
        return False


needs_db = pytest.mark.skipif(
    not _postgres_available(),
    reason=f"нет PostgreSQL по {TEST_DATABASE_URL}",
)


@pytest.fixture(scope="session")
def engine() -> Iterator[Engine]:
    """Схема поднимается один раз на прогон и сносится в конце."""
    eng = create_engine(TEST_DATABASE_URL)
    with eng.begin() as conn:
        for schema in ALL_SCHEMAS:
            conn.execute(DropSchema(schema, cascade=True, if_exists=True))
            conn.execute(CreateSchema(schema))
    Base.metadata.create_all(eng)
    yield eng
    with eng.begin() as conn:
        for schema in ALL_SCHEMAS:
            conn.execute(DropSchema(schema, cascade=True, if_exists=True))
    eng.dispose()


@pytest.fixture
def session(engine: Engine) -> Iterator[Session]:
    """Каждый тест — во вложенной транзакции с откатом: тесты про ограничения
    обязаны видеть чистую базу, иначе падают друг об друга."""
    connection = engine.connect()
    transaction = connection.begin()
    sess = Session(bind=connection, join_transaction_mode="create_savepoint")
    try:
        yield sess
    finally:
        sess.close()
        transaction.rollback()
        connection.close()
