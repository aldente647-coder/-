"""Структурные инварианты схемы. Базы не требуют — проверяют саму модель."""

from __future__ import annotations

import re

import pytest
from sqlalchemy import Table
from sqlalchemy.schema import CreateTable

from socmon.db import models  # noqa: F401  регистрация таблиц в metadata
from socmon.db.base import CONTACT_SCHEMA, OPS_SCHEMA, SURVEY_SCHEMA, Base

#: Всё, что может оказаться персональными контактными данными.
PII_HINT = re.compile(r"phone|msisdn|номер|contact_id|blind", re.IGNORECASE)

SURVEY_TABLES = [t for t in Base.metadata.tables.values() if t.schema == SURVEY_SCHEMA]
CONTACT_TABLES = [t for t in Base.metadata.tables.values() if t.schema == CONTACT_SCHEMA]


@pytest.mark.parametrize("table", SURVEY_TABLES, ids=lambda t: t.name)
def test_analytic_contour_holds_no_contact_data(table: Table) -> None:
    """Главный инвариант ТЗ §14.2: в контуре ответов нет ни одного столбца,
    по которому можно выйти на абонента."""
    leaks = [c.name for c in table.columns if PII_HINT.search(c.name)]
    assert not leaks, f"{table.name}: {leaks}"


@pytest.mark.parametrize("table", SURVEY_TABLES, ids=lambda t: t.name)
def test_analytic_contour_has_no_foreign_keys_into_contacts(table: Table) -> None:
    """Соединить контуры можно только через identity_link. Прямой внешний ключ
    из survey в contact сделал бы разделение декоративным."""
    for fk in table.foreign_keys:
        assert fk.column.table.schema != CONTACT_SCHEMA, f"{table.name} → {fk.target_fullname}"


@pytest.mark.parametrize("table", CONTACT_TABLES, ids=lambda t: t.name)
def test_contact_contour_holds_no_answers(table: Table) -> None:
    """И симметрично: в контактном контуре не должно быть ответов."""
    forbidden = {"code", "answer", "transcript", "themes", "sentiment"}
    assert not (forbidden & {c.name for c in table.columns}), table.name


def test_only_identity_link_bridges_the_contours() -> None:
    bridges = {
        t.name
        for t in CONTACT_TABLES
        for c in t.columns
        if c.name == "interview_id"
    }
    assert bridges == {"identity_link"}


def test_phone_is_never_stored_in_the_clear() -> None:
    from socmon.db.models import Contact

    columns = {c.name for c in Contact.__table__.columns}
    assert "phone" not in columns
    assert {"blind_index", "phone_ciphertext"} <= columns


def test_every_table_lives_in_a_known_schema() -> None:
    assert {t.schema for t in Base.metadata.tables.values()} == {
        CONTACT_SCHEMA, SURVEY_SCHEMA, OPS_SCHEMA
    }


def test_all_timestamps_carry_a_timezone() -> None:
    """Наивное время в этой системе — это ночные звонки и разъезжающиеся волны."""
    for table in Base.metadata.tables.values():
        for column in table.columns:
            if column.type.__class__.__name__ == "DateTime":
                assert getattr(column.type, "timezone", False), f"{table.name}.{column.name}"


def test_ddl_renders_for_postgresql() -> None:
    from sqlalchemy.dialects.postgresql import dialect

    for table in Base.metadata.tables.values():
        rendered = str(CreateTable(table).compile(dialect=dialect()))  # type: ignore[no-untyped-call]
        assert "CREATE TABLE" in rendered


def test_section_10_1_fields_are_all_present() -> None:
    """ТЗ §10.1 перечисляет состав данных по контакту и попытке.
    Проверяем, что ничего не потерялось."""
    from socmon.db.models import Answer, CallAttempt, Interview

    attempt = {c.name for c in CallAttempt.__table__.columns}
    assert {
        "id", "contact_id", "wave_id", "attempt_no", "started_at",
        "duration_sec", "stage", "disposition", "provider_call_id",
    } <= attempt

    interview = {c.name for c in Interview.__table__.columns}
    assert {
        "id", "wave_id", "questionnaire_version_id", "started_at", "timezone",
        "duration_sec", "consent", "outcome", "transferred_to_operator",
        "recording_uri", "transcript_uri", "validity", "validity_reasons",
    } <= interview

    answer = {c.name for c in Answer.__table__.columns}
    assert {"node_id", "core_key", "code", "text", "attempts", "answered_at"} <= answer
