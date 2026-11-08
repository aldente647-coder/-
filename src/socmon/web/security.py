"""Роли, пароли и границы доступа.

Роль здесь — не пункт меню, а граница доступа к персональным данным.
ТЗ §14.2 требует разграничения прав, раздельного хранения номеров
и ответов и журнала доступа; §13.4 — просмотра записи «при наличии
соответствующего уровня доступа».

Отсюда главный инвариант модуля: **право видеть номер и право слушать
запись — отдельные права**, и по умолчанию их нет ни у кого. Аналитик,
которому нужны цифры, номера не видит вовсе; это не ограничение
интерфейса, а исполнение §14.2.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from dataclasses import dataclass
from enum import StrEnum

SCRYPT_N = 2**14
SCRYPT_R = 8
SCRYPT_P = 1
SALT_BYTES = 16


class Permission(StrEnum):
    """Отдельные права, а не «уровни доступа».

    Уровни соблазняют считать, что больший включает меньший. С
    персональными данными это неверно: дежурному оператору номер нужен,
    а аналитику — нет, хотя аналитик «главнее».
    """

    VIEW_DASHBOARD = "view_dashboard"
    VIEW_ANALYTICS = "view_analytics"
    VIEW_INTERVIEWS = "view_interviews"
    VIEW_PHONE = "view_phone"
    """Видеть телефонный номер. ТЗ §14.2: номера отделены от ответов."""

    LISTEN_RECORDING = "listen_recording"
    """Слушать запись разговора. Каждое прослушивание журналируется."""

    MANAGE_WAVE = "manage_wave"
    MANAGE_QUESTIONNAIRE = "manage_questionnaire"
    MANAGE_DIALLING = "manage_dialling"
    MANAGE_CONTACTS = "manage_contacts"
    """Загружать базу номеров. Отдельно от права видеть номер: загрузить
    файл и разглядывать чужие телефоны на экране — разные действия."""
    REVIEW_CODING = "review_coding"
    EXPORT_DATA = "export_data"
    MANAGE_USERS = "manage_users"
    VIEW_AUDIT = "view_audit"


class Role(StrEnum):
    ADMIN = "admin"
    """Администратор платформы со стороны Исполнителя."""

    CUSTOMER = "customer"
    """Руководитель со стороны Заказчика: цифры, выгрузки, отчёты."""

    ANALYST = "analyst"
    """Аналитик. Номеров не видит — работает с обезличенным массивом."""

    CODER = "coder"
    """Кодировщик: ручная проверка неоднозначной разметки (ТЗ §11.3)."""

    OPERATOR = "operator"
    """Дежурный оператор (ТЗ §9.2)."""

    AUDITOR = "auditor"
    """Проверяющий: журнал и качество, без права что-либо менять."""


ROLE_PERMISSIONS: dict[Role, frozenset[Permission]] = {
    Role.ADMIN: frozenset(Permission),
    Role.CUSTOMER: frozenset({
        Permission.VIEW_DASHBOARD,
        Permission.VIEW_ANALYTICS,
        Permission.VIEW_INTERVIEWS,
        Permission.LISTEN_RECORDING,
        Permission.MANAGE_WAVE,
        Permission.MANAGE_QUESTIONNAIRE,
        Permission.MANAGE_CONTACTS,
        Permission.EXPORT_DATA,
        Permission.VIEW_AUDIT,
    }),
    Role.ANALYST: frozenset({
        Permission.VIEW_DASHBOARD,
        Permission.VIEW_ANALYTICS,
        Permission.VIEW_INTERVIEWS,
        Permission.EXPORT_DATA,
    }),
    Role.CODER: frozenset({
        Permission.VIEW_INTERVIEWS,
        Permission.REVIEW_CODING,
        Permission.LISTEN_RECORDING,
    }),
    Role.OPERATOR: frozenset({
        Permission.VIEW_DASHBOARD,
        Permission.VIEW_PHONE,
    }),
    Role.AUDITOR: frozenset({
        Permission.VIEW_DASHBOARD,
        Permission.VIEW_INTERVIEWS,
        Permission.VIEW_ANALYTICS,
        Permission.VIEW_AUDIT,
    }),
}

ROLE_TITLES: dict[Role, str] = {
    Role.ADMIN: "Администратор платформы",
    Role.CUSTOMER: "Руководитель Заказчика",
    Role.ANALYST: "Аналитик",
    Role.CODER: "Кодировщик",
    Role.OPERATOR: "Дежурный оператор",
    Role.AUDITOR: "Проверяющий",
}


@dataclass(frozen=True, slots=True)
class Principal:
    """Кто действует."""

    user_id: str
    login: str
    full_name: str
    role: Role

    @property
    def permissions(self) -> frozenset[Permission]:
        return ROLE_PERMISSIONS[self.role]

    def can(self, permission: Permission) -> bool:
        return permission in self.permissions

    @property
    def sees_phone_numbers(self) -> bool:
        return self.can(Permission.VIEW_PHONE)


def mask_phone(phone: str) -> str:
    """Скрыть номер, оставив хвост для сверки.

    Приём из блюпринта VOIF: операторы часто работают в общем помещении,
    и полные номера на экране видны всем вокруг. Хвоста хватает, чтобы
    свериться с абонентом, и не хватает, чтобы унести базу.
    """
    digits = [c for c in phone if c.isdigit()]
    if len(digits) < 4:
        return "•" * len(digits)
    return "•" * (len(digits) - 4) + "".join(digits[-4:])


# ------------------------------------------------------------------- пароли


def hash_password(password: str) -> str:
    """Захешировать пароль.

    scrypt из стандартной библиотеки: специально медленный и требовательный
    к памяти, чего и надо от хеша пароля. Лишней зависимости не берём —
    криптографию проще не тащить, чем сопровождать.
    """
    if len(password) < 12:
        raise ValueError("пароль короче 12 символов")
    if len(password) > 1024:
        raise ValueError("пароль длиннее 1024 символов")
    salt = os.urandom(SALT_BYTES)
    digest = hashlib.scrypt(
        password.encode("utf-8"), salt=salt, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P
    )
    return f"scrypt${SCRYPT_N}${SCRYPT_R}${SCRYPT_P}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """Проверить пароль. Сравнение постоянного времени."""
    try:
        algorithm, n, r, p, salt_hex, digest_hex = stored.split("$")
        if (algorithm != "scrypt" or len(password) > 1024
                or (int(n), int(r), int(p)) != (SCRYPT_N, SCRYPT_R, SCRYPT_P)
                or len(salt_hex) != SALT_BYTES * 2 or len(digest_hex) != 128):
            return False
        digest = hashlib.scrypt(
            password.encode("utf-8"), salt=bytes.fromhex(salt_hex),
            n=int(n), r=int(r), p=int(p),
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(digest.hex(), digest_hex)


def new_session_token() -> str:
    return secrets.token_urlsafe(32)
