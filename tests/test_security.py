"""Проверки атакующих запросов без доступа к данным стенда."""
from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from socmon.export.writers import csv_value, validate_wave_code
from socmon.telephony.asterisk import AsteriskConfig
from socmon.web.app import create_app
from socmon.web.artifacts import existing_archive
from socmon.web.protection import LoginLimiter, local_redirect


def client() -> TestClient:
    return TestClient(
        create_app(sessionmaker(create_engine("sqlite://")), secret_key="x" * 32),
        base_url="https://testserver",
    )


def token(browser: TestClient) -> str:
    response = browser.get("/login")
    match = re.search(r'name="csrf_token" value="([^"]+)"', response.text)
    assert match
    return match[1]


def test_session_cookie_is_secure_and_pages_cannot_be_framed() -> None:
    with client() as browser:
        response = browser.get("/login")
    cookie = response.headers["set-cookie"].lower()
    assert "secure" in cookie and "httponly" in cookie and "samesite=lax" in cookie
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["referrer-policy"] == "same-origin"
    assert response.headers["x-frame-options"] == "DENY"
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
    nonce = re.search(r'<script nonce="([^"]+)"', response.text)
    assert nonce and f"'nonce-{nonce[1]}'" in response.headers["content-security-policy"]


def test_csrf_is_required_even_for_login() -> None:
    with client() as browser:
        response = browser.post("/login", data={"login": "admin", "password": "x" * 12})
        assert response.status_code == 403


def test_csrf_token_from_another_session_is_refused() -> None:
    with client() as alice, client() as bob:
        foreign = token(alice)
        token(bob)
        response = bob.post("/logout", data={"csrf_token": foreign})
        assert response.status_code == 403


def test_valid_form_token_is_accepted() -> None:
    with client() as browser:
        response = browser.post("/logout", data={"csrf_token": token(browser)},
                                follow_redirects=False)
        assert response.status_code == 303


def test_cross_site_origin_is_refused_even_with_token() -> None:
    with client() as browser:
        response = browser.post("/logout", data={"csrf_token": token(browser)},
                                headers={"Origin": "https://attacker.example"})
        assert response.status_code == 403


def test_oversized_stream_without_content_length_is_rejected() -> None:
    with client() as browser:
        response = browser.post("/login", content=iter([b"x" * 40000, b"x" * 40000]))
        assert response.status_code == 413


@pytest.mark.parametrize("value", ["//evil.test", "/\\evil.test", "/\nevil.test",
                                   "https://evil.test", "\\evil.test"])
def test_login_redirect_cannot_escape_origin(value: str) -> None:
    assert local_redirect(value) == "/"


def test_login_redirect_keeps_local_query() -> None:
    assert local_redirect("/waves?page=2") == "/waves?page=2"


@pytest.mark.parametrize("code", ["../../secret", "x/../../../secret", "/tmp/x", "x\\y"])
def test_export_code_cannot_become_a_path(code: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        validate_wave_code(code)
    with pytest.raises(ValueError):
        existing_archive(code, root=tmp_path)


@pytest.mark.parametrize("value", ["=1+1", "+1+1", "-1+1", "@SUM(A1)", "\t=1", "  =1"])
def test_csv_cells_cannot_be_interpreted_as_formulas(value: str) -> None:
    assert csv_value(value) == "'" + value
    assert csv_value(-2.0) == -2.0
    assert csv_value("обычный текст") == "обычный текст"


def test_limiter_bounds_attempts_and_memory() -> None:
    limiter = LoginLimiter(limit=2, capacity=3)
    assert limiter.allow("one")
    assert limiter.allow("one")
    assert not limiter.allow("one")
    for index in range(10):
        limiter.allow(str(index))
    assert len(limiter._buckets) == 3


def test_ari_password_is_absent_from_url_and_repr() -> None:
    config = AsteriskConfig(password="s&cret?password")
    assert "s&cret" not in config.websocket_url
    assert "s&cret" not in repr(config)
    assert "api_key" not in config.websocket_url
    assert config.websocket_headers["Authorization"].startswith("Basic ")


def test_short_session_secret_is_refused() -> None:
    with pytest.raises(ValueError):
        create_app(sessionmaker(create_engine("sqlite://")), secret_key="unsafe")


async def test_audio_identifier_is_single_use_and_server_closes_connections() -> None:
    import asyncio
    import uuid

    from socmon.asr.audiosocket import AudioSocketServer, Frame, FrameType
    from tests.test_live import Recognizer

    async def speech(value: object) -> None:
        pass

    server = AudioSocketServer(Recognizer(), speech, port=0)
    stream = server.expect(str(uuid.uuid4()))
    await server.start()
    reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
    writer.write(Frame(FrameType.IDENTIFIER, uuid.UUID(stream).bytes).encode())
    await writer.drain()
    # Первый идентификатор будет принят до второго благодаря порядку TCP,
    # а второй поток с тем же токеном должен быть закрыт без ASR.
    for _ in range(100):
        if server.active:
            break
        await asyncio.sleep(0.001)
    assert server.active
    other, duplicate = await asyncio.open_connection("127.0.0.1", server.port)
    duplicate.write(Frame(FrameType.IDENTIFIER, uuid.UUID(stream).bytes).encode())
    await duplicate.drain()
    assert await asyncio.wait_for(other.read(), 1) == b""
    await server.stop()
    assert await asyncio.wait_for(reader.read(), 1) == b""
    assert not server.active
    writer.close()
    duplicate.close()
    await writer.wait_closed()
    await duplicate.wait_closed()


def test_same_origin_browser_form_is_accepted() -> None:
    with client() as browser:
        response = browser.post("/logout", data={"csrf_token": token(browser)},
                                headers={"Origin": "https://testserver",
                                         "Sec-Fetch-Site": "same-origin"},
                                follow_redirects=False)
        assert response.status_code == 303
