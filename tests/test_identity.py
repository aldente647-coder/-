from __future__ import annotations

import pytest

from socmon.domain.identity import (
    PhoneError,
    PhoneVault,
    new_pseudonym,
    normalize_phone,
)

INDEX_KEY = bytes(range(32))
ENC_KEY = bytes(range(32, 64))


@pytest.fixture
def vault() -> PhoneVault:
    return PhoneVault(INDEX_KEY, ENC_KEY)


@pytest.mark.parametrize(
    "raw",
    ["+7 949 123-45-67", "8(949)1234567", "79491234567", "+7-949-123-45-67", "9491234567"],
)
def test_same_subscriber_normalizes_to_one_number(raw: str) -> None:
    """Дедупликация базы (ТЗ §4.2) держится на канонической форме."""
    assert normalize_phone(raw) == "+79491234567"


@pytest.mark.parametrize("raw", ["", "не номер", "12", "1" * 20])
def test_garbage_is_rejected(raw: str) -> None:
    with pytest.raises(PhoneError):
        normalize_phone(raw)


def test_blind_index_is_deterministic(vault: PhoneVault) -> None:
    """Поиск «найди контакт по телефону» работает точным сравнением
    по индексу — та причина, по которой блюпринт отказался от шифрования,
    здесь снята."""
    assert vault.lookup_index("8 949 123 45 67") == vault.lookup_index("+79491234567")


def test_blind_index_separates_different_numbers(vault: PhoneVault) -> None:
    assert vault.lookup_index("+79491234567") != vault.lookup_index("+79491234568")


def test_index_key_change_breaks_lookup() -> None:
    """Ключ индекса менять нельзя без пересчёта базы — фиксируем это тестом,
    чтобы свойство не потерялось при рефакторинге."""
    other = PhoneVault(bytes(range(1, 33)), ENC_KEY)
    assert PhoneVault(INDEX_KEY, ENC_KEY).lookup_index("+79491234567") != other.lookup_index(
        "+79491234567"
    )


def test_roundtrip(vault: PhoneVault) -> None:
    sealed = vault.seal("8 949 123 45 67")
    assert vault.open(sealed) == "+79491234567"


def test_ciphertext_is_not_deterministic(vault: PhoneVault) -> None:
    """Разные nonce на каждый вызов: иначе шифротекст сам стал бы индексом
    и раскрыл бы совпадения номеров."""
    first, second = vault.seal("+79491234567"), vault.seal("+79491234567")
    assert first.ciphertext != second.ciphertext
    assert first.blind_index == second.blind_index


def test_ciphertext_is_tamper_evident(vault: PhoneVault) -> None:
    from cryptography.exceptions import InvalidTag

    from socmon.domain.identity import SealedPhone

    sealed = vault.seal("+79491234567")
    flipped = bytes([sealed.ciphertext[-1] ^ 0xFF])
    broken = SealedPhone(sealed.blind_index, sealed.ciphertext[:-1] + flipped)
    with pytest.raises(InvalidTag):
        vault.open(broken)


def test_short_keys_are_refused() -> None:
    with pytest.raises(ValueError, match="слепого индекса"):
        PhoneVault(b"short", ENC_KEY)
    with pytest.raises(ValueError, match="AES"):
        PhoneVault(INDEX_KEY, b"short")


def test_keys_come_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOCMON_PHONE_INDEX_KEY", INDEX_KEY.hex())
    monkeypatch.setenv("SOCMON_PHONE_ENC_KEY", ENC_KEY.hex())
    assert PhoneVault.from_env().lookup_index("+79491234567") is not None


def test_missing_key_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SOCMON_PHONE_INDEX_KEY", raising=False)
    monkeypatch.delenv("SOCMON_PHONE_ENC_KEY", raising=False)
    with pytest.raises(RuntimeError, match="SOCMON_PHONE_INDEX_KEY"):
        PhoneVault.from_env()


def test_pseudonym_is_not_derived_from_the_number(vault: PhoneVault) -> None:
    """Псевдоним выдаётся заново на каждое интервью и не выводится из номера:
    из аналитического массива один человек в разных волнах не сшивается."""
    assert new_pseudonym() != new_pseudonym()
