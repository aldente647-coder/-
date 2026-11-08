"""Точка входа для локального запуска кабинета.

Нужна, чтобы поднимать приложение обычными средствами `uvicorn --factory`
без написания скрипта на каждый запуск: фабрика принимает параметры,
а сюда они приходят из окружения.
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from socmon.web.app import create_app


def app() -> FastAPI:
    url = os.environ.get("DATABASE_URL")
    secret = os.environ.get("SOCMON_SECRET_KEY")
    if not url or not secret:
        raise RuntimeError("нужны DATABASE_URL и SOCMON_SECRET_KEY")
    return create_app(
        sessionmaker(bind=create_engine(url)), secret_key=secret,
        https_only=os.environ.get("SOCMON_ALLOW_HTTP") != "1",
    )
