"""Идентичность контакта и разделение контуров.

ТЗ §14.2 требует раздельного хранения телефонных номеров и массива ответов
плюс псевдонимизации идентификаторов. Блюпринт VOIF решает обратную задачу —
там телефоны лежат открытым текстом, а шифрование столбца отвергнуто, потому
что «сломает поиск по номеру».

Слепой индекс снимает это противоречие: рядом с зашифрованным номером хранится
`HMAC(нормализованный номер, ключ)`. Поиск и дедупликация работают точным
сравнением по индексу, а сам номер без ключа не восстановить.

Два ключа и оба — из окружения, никогда не из базы:
* ключ слепого индекса — детерминированный, менять его нельзя без пересчёта
  всей базы (индексы разойдутся);
* ключ шифрования — для AES-GCM, допускает ротацию.
"""

from __future__ import annotations

import hmac
import os
import re
import uuid
from dataclasses import dataclass
from hashlib import sha256

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

_NON_DIGIT = re.compile(r"\D+")
_NONCE_BYTES = 12


class PhoneError(ValueError):
    """Номер не приводится к каноническому виду."""


def normalize_phone(raw: str) -> str:
    """Привести номер к E.164.

    Дедупликация (ТЗ §4.2) работает только по каноническому виду: «+7 949…»,
    «8949…» и «7(949)…» — это один и тот же абонент, и попасть в выборку
    трижды он не должен.
    """
    digits = _NON_DIGIT.sub("", raw)
    if not digits:
        raise PhoneError(f"в строке {raw!r} нет цифр")
    if len(digits) == 11 and digits[0] == "8":
        digits = "7" + digits[1:]
    if len(digits) == 10:
        digits = "7" + digits
    if not 11 <= len(digits) <= 15:
        raise PhoneError(f"{raw!r}: непохожая на номер длина ({len(digits)} цифр)")
    return "+" + digits


@dataclass(frozen=True, slots=True)
class SealedPhone:
    """Номер в том виде, в каком он ложится в контактный контур."""

    blind_index: bytes
    """Детерминированный HMAC. По нему ищут и дедуплицируют."""

    ciphertext: bytes
    """AES-GCM: nonce ‖ шифротекст ‖ тег. Недетерминирован — искать нельзя."""


class PhoneVault:
    """Шифрование номеров и построение слепого индекса."""

    def __init__(self, index_key: bytes, encryption_key: bytes) -> None:
        if len(index_key) < 32:
            raise ValueError("ключ слепого индекса короче 32 байт")
        if len(encryption_key) not in (16, 24, 32):
            raise ValueError("ключ AES должен быть 16, 24 или 32 байта")
        self._index_key = index_key
        self._aead = AESGCM(encryption_key)

    @classmethod
    def from_env(cls) -> PhoneVault:
        """Ключи только из окружения (ТЗ §14.2, §14.3)."""
        try:
            index_key = bytes.fromhex(os.environ["SOCMON_PHONE_INDEX_KEY"])
            enc_key = bytes.fromhex(os.environ["SOCMON_PHONE_ENC_KEY"])
        except KeyError as exc:
            raise RuntimeError(f"не задана переменная окружения {exc.args[0]}") from exc
        return cls(index_key, enc_key)

    def blind_index(self, normalized_phone: str) -> bytes:
        return hmac.new(self._index_key, normalized_phone.encode("utf-8"), sha256).digest()

    def seal(self, raw_phone: str) -> SealedPhone:
        normalized = normalize_phone(raw_phone)
        nonce = os.urandom(_NONCE_BYTES)
        blob = self._aead.encrypt(nonce, normalized.encode("utf-8"), None)
        return SealedPhone(blind_index=self.blind_index(normalized), ciphertext=nonce + blob)

    def open(self, sealed: SealedPhone) -> str:
        """Раскрыть номер. Вызов допустим только из контактного контура —
        аналитик работает с массивом, где номера нет вовсе."""
        nonce, blob = sealed.ciphertext[:_NONCE_BYTES], sealed.ciphertext[_NONCE_BYTES:]
        return self._aead.decrypt(nonce, blob, None).decode("utf-8")

    def lookup_index(self, raw_phone: str) -> bytes:
        """Индекс для поиска: «найди контакт по номеру» без расшифровки базы."""
        return self.blind_index(normalize_phone(raw_phone))


def new_pseudonym() -> uuid.UUID:
    """Обезличенный идентификатор интервью для аналитического контура.

    Выдаётся на каждое интервью заново и никак не выводится из номера. Связь
    «псевдоним ↔ контакт» живёт единственной таблицей в контактном контуре под
    отдельной ролью доступа; из аналитического массива один и тот же человек
    в разных волнах не сшивается.
    """
    return uuid.uuid4()
