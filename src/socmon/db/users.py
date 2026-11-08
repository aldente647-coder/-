"""Изменение пароля с немедленным отзывом браузерных сессий."""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from socmon.db.models import AuditLog, User
from socmon.web.security import hash_password


def change_password(db: Session, login: str, password: str) -> None:
    digest = hash_password(password)
    row = db.scalar(select(User).where(User.login == login).with_for_update())
    if row is None:
        raise ValueError("учётная запись не найдена")
    row.password_hash = digest
    row.session_version += 1
    db.add(AuditLog(
        actor="local-cli", action="change_password", entity_type="user",
        entity_id=str(row.id), diff={"sessions_revoked": True},
    ))
    db.flush()
