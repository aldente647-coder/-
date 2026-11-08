"""Загрузка базы контактов (ТЗ §4.2)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from socmon.core.codes import CallStage, Disposition
from socmon.db.contacts import (
    ImportReport,
    base_summary,
    deactivate_never_answering,
    dial_order,
    dialable,
    import_numbers,
    new_seed,
    read_column,
)
from socmon.db.models import CallAttempt, Contact, ContactBatch
from socmon.domain.identity import PhoneVault
from tests.conftest import needs_db
from tests.test_db_constraints import make_wave

pytestmark = [needs_db, pytest.mark.usefixtures("engine")]


@pytest.fixture
def vault() -> PhoneVault:
    return PhoneVault(index_key=b"i" * 32, encryption_key=b"e" * 32)


def numbers(count: int, start: int = 1_000_000) -> list[str]:
    return [f"+7949{start + i:07d}" for i in range(count)]


def load(
    session: Session,
    vault: PhoneVault,
    rows: list[str],
    *,
    name: str = "поставка",
    seed: str | None = None,
    chunk_size: int = 5_000,
) -> ImportReport:
    return import_numbers(
        session, vault, rows,
        name=name, source="Заказчик", seed=seed, chunk_size=chunk_size,
    )


# ------------------------------------------------------------------ загрузка


def test_numbers_are_loaded_and_counted(session: Session, vault: PhoneVault) -> None:
    report = load(session, vault, numbers(50))
    assert report.total == 50
    assert report.loaded == 50
    assert report.rejected == 0
    assert report.usable_share == 1.0


def test_phone_is_never_stored_in_the_clear(
    session: Session, vault: PhoneVault
) -> None:
    """ТЗ §14.2: номер лежит зашифрованным, ищется по слепому индексу."""
    load(session, vault, ["+7 949 123-45-67"])
    contact = session.scalars(select(Contact)).first()
    assert contact is not None
    assert b"9491234567" not in contact.phone_ciphertext
    from socmon.domain.identity import SealedPhone

    restored = vault.open(SealedPhone(
        blind_index=contact.blind_index, ciphertext=contact.phone_ciphertext
    ))
    assert restored == "+79491234567"


def test_different_spellings_are_one_contact(
    session: Session, vault: PhoneVault
) -> None:
    """«+7 949…», «8949…» и «7(949)…» — один абонент, и в выборку
    он не должен попасть трижды (ТЗ §4.2)."""
    report = load(session, vault, ["+79491234567", "8949 123 45 67", "7(949)1234567"])
    assert report.loaded == 1
    assert report.duplicate_in_file == 2


def test_duplicates_inside_the_file_and_in_the_base_are_counted_apart(
    session: Session, vault: PhoneVault
) -> None:
    """Первое говорит о качестве файла, второе — о пересечении поставок.
    Свести в одно число значит потерять и то и другое."""
    load(session, vault, numbers(10))
    report = load(session, vault, [*numbers(10), *numbers(5, start=2_000_000)] * 1)
    assert report.duplicate_in_base == 10
    assert report.duplicate_in_file == 0
    assert report.loaded == 5


def test_malformed_numbers_are_rejected_with_a_reason(
    session: Session, vault: PhoneVault
) -> None:
    report = load(session, vault, ["+79491234567", "не номер", "12", "", "abc"])
    assert report.loaded == 1
    assert report.rejected == 4
    assert all(r.reason for r in report.samples)


def test_rejected_values_are_masked_in_the_report(
    session: Session, vault: PhoneVault
) -> None:
    """Отчёт о загрузке читают люди, которым номера видеть незачем."""
    report = load(session, vault, ["12345678901234567890"])
    assert report.samples[0].value == "123…90"


def test_reject_samples_are_capped(session: Session, vault: PhoneVault) -> None:
    """Полный список миллиона ошибок никто не читает."""
    report = load(session, vault, ["мусор"] * 500)
    assert report.rejected == 500
    assert len(report.samples) == 50


def test_dirty_file_is_flagged(session: Session, vault: PhoneVault) -> None:
    report = load(session, vault, [*numbers(4), *(["мусор"] * 6)])
    assert any("вернуть файл Заказчику" in n for n in report.notes)


def test_batch_records_what_happened(session: Session, vault: PhoneVault) -> None:
    report = load(session, vault, [*numbers(8), "мусор"], name="Волна 17")
    batch = session.get(ContactBatch, report.batch_id)
    assert batch is not None
    assert batch.name == "Волна 17"
    assert batch.rows_total == 9
    assert batch.rows_loaded == 8
    assert batch.rows_rejected == 1


def test_large_file_is_processed_in_chunks(
    session: Session, vault: PhoneVault
) -> None:
    """Миллион строк стоит столько же памяти, сколько тысяча."""
    report = load(session, vault, numbers(3000), chunk_size=100)
    assert report.loaded == 3000


# ------------------------------------------------------- случайный порядок


def test_order_is_random_relative_to_the_number(
    session: Session, vault: PhoneVault
) -> None:
    """Порядок без случайности даёт систематическое смещение по тому
    признаку, по которому база отсортирована у Заказчика (ТЗ §4.2)."""
    load(session, vault, numbers(200))
    order = [c.dial_order for c in dialable(session, 200)]
    assert order == sorted(order)

    sealed = [c.phone_ciphertext for c in dialable(session, 200)]
    assert len(sealed) == 200
    # Порядок набора не совпадает с порядком загрузки.
    loaded_order = [
        c.dial_order for c in session.scalars(
            select(Contact).order_by(Contact.created_at, Contact.id)
        ).all()
    ]
    assert loaded_order != sorted(loaded_order)


def test_order_is_reproducible_from_the_seed() -> None:
    """§5.1: воспроизводимый отбор с фиксацией алгоритма и seed."""
    seed, blind = "abc123", b"x" * 32
    assert dial_order(seed, blind) == dial_order(seed, blind)
    assert dial_order("другое", blind) != dial_order(seed, blind)


def test_order_key_fits_a_signed_bigint() -> None:
    for _ in range(200):
        assert 0 <= dial_order(new_seed(), uuid.uuid4().bytes * 2) < 2**63


def test_seed_is_stored_with_the_batch(session: Session, vault: PhoneVault) -> None:
    report = load(session, vault, numbers(5))
    batch = session.get(ContactBatch, report.batch_id)
    assert batch is not None
    assert batch.seed == report.seed
    assert len(batch.seed) == 32


def test_explicit_seed_reproduces_the_same_order(
    session: Session, vault: PhoneVault
) -> None:
    load(session, vault, numbers(20), seed="фиксированное-зерно")
    first = [c.dial_order for c in dialable(session, 20)]

    session.rollback()
    load(session, vault, numbers(20), seed="фиксированное-зерно")
    assert [c.dial_order for c in dialable(session, 20)] == first


# ------------------------------------------------------------ кого не звоним


def test_stop_list_survives_a_new_delivery(
    session: Session, vault: PhoneVault
) -> None:
    """Затереть отметку новой поставкой значило бы позвонить человеку,
    который просил больше не звонить (ТЗ §6.2)."""
    load(session, vault, ["+79491234567"])
    contact = session.scalars(select(Contact)).one()
    contact.do_not_call = True
    contact.do_not_call_at = datetime.now(UTC)
    session.flush()

    load(session, vault, ["+79491234567"])
    refreshed = session.scalars(select(Contact)).one()
    assert refreshed.do_not_call is True


def test_excluded_contacts_are_not_dialled(
    session: Session, vault: PhoneVault
) -> None:
    load(session, vault, numbers(4))
    rows = session.scalars(select(Contact).order_by(Contact.dial_order)).all()
    rows[0].do_not_call = True
    rows[1].invalid_number = True
    rows[2].never_answers = True
    session.flush()

    assert len(dialable(session, 10)) == 1


def test_persistent_non_answers_are_marked_not_deleted(
    session: Session, vault: PhoneVault
) -> None:
    """История звонков — часть расчёта достижимости, удалять её нельзя
    (ТЗ §4.2)."""
    load(session, vault, ["+79491234567"])
    contact = session.scalars(select(Contact)).one()
    wave = make_wave(session, f"w-{uuid.uuid4().hex[:6]}")
    for attempt_no in range(1, 4):
        session.add(CallAttempt(
            contact_id=contact.id, wave_id=wave.id, attempt_no=attempt_no,
            scheduled_at=datetime.now(UTC),
            ended_at=datetime.now(UTC) + timedelta(seconds=30),
            stage=CallStage.FINISHED, disposition=Disposition.NO_ANSWER,
        ))
    session.flush()

    assert deactivate_never_answering(session, min_attempts=3) == 1
    refreshed = session.scalars(select(Contact)).one()
    assert refreshed.never_answers is True
    assert refreshed.deactivated_at is not None
    assert session.scalars(select(CallAttempt)).all()  # история цела


def test_someone_who_once_answered_is_not_excluded(
    session: Session, vault: PhoneVault
) -> None:
    load(session, vault, ["+79491234567"])
    contact = session.scalars(select(Contact)).one()
    wave = make_wave(session, f"w-{uuid.uuid4().hex[:6]}")
    now = datetime.now(UTC)
    session.add_all([
        CallAttempt(contact_id=contact.id, wave_id=wave.id, attempt_no=1,
                    scheduled_at=now, ended_at=now, stage=CallStage.FINISHED,
                    disposition=Disposition.NO_ANSWER),
        CallAttempt(contact_id=contact.id, wave_id=wave.id, attempt_no=2,
                    scheduled_at=now, started_at=now, ended_at=now,
                    stage=CallStage.FINISHED),
        CallAttempt(contact_id=contact.id, wave_id=wave.id, attempt_no=3,
                    scheduled_at=now, ended_at=now, stage=CallStage.FINISHED,
                    disposition=Disposition.NO_ANSWER),
    ])
    session.flush()

    assert deactivate_never_answering(session, min_attempts=3) == 0


def test_summary_counts_the_base(session: Session, vault: PhoneVault) -> None:
    load(session, vault, numbers(10))
    session.scalars(select(Contact)).first().do_not_call = True  # type: ignore[union-attr]
    session.flush()

    summary = base_summary(session)
    assert summary["total"] == 10
    assert summary["do_not_call"] == 1
    assert summary["dialable"] == 9


# ---------------------------------------------------------------- чтение файла


@pytest.mark.parametrize(
    "text",
    [
        "+79491234567\n+79491234568\n",
        "phone;name\n+79491234567;Иван\n+79491234568;Пётр\n",
        "телефон,ФИО\n+79491234567,Иван\n+79491234568,Пётр\n",
        "+79491234567\t Иван\n+79491234568\t Пётр\n",
    ],
)
def test_common_file_shapes_are_understood(text: str) -> None:
    """Заказчик присылает и точку с запятой, и запятую, и просто список.
    Спорить об этом в переписке дольше, чем определить."""
    assert list(read_column(text)) == ["+79491234567", "+79491234568"]


def test_header_row_is_not_taken_for_a_number() -> None:
    assert "phone" not in list(read_column("phone\n+79491234567\n"))


def test_first_row_of_data_is_not_lost() -> None:
    """Файл без заголовка — обычное дело, и терять первую строку нельзя."""
    assert list(read_column("+79491234567\n+79491234568\n")) == [
        "+79491234567", "+79491234568",
    ]


def test_explicit_column_wins_over_guessing() -> None:
    text = "89991112233;+79491234567\n89991112244;+79491234568\n"
    assert list(read_column(text, column=1)) == ["+79491234567", "+79491234568"]
