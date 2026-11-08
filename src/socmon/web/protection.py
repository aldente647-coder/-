"""Защита браузерных запросов до выполнения операций кабинета."""
from __future__ import annotations

import asyncio
import hmac
import secrets
import threading
import time
from collections import OrderedDict
from urllib.parse import urlsplit

from fastapi import HTTPException, Request
from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send


def csrf_token(request: Request) -> str:
    token = request.session.get("csrf_token")
    if not isinstance(token, str):
        token = secrets.token_urlsafe(32)
        request.session["csrf_token"] = token
    return token


async def verify_csrf(request: Request) -> None:
    if request.method in {"GET", "HEAD", "OPTIONS"}:
        return
    expected = request.session.get("csrf_token")
    supplied = request.headers.get("x-csrf-token")
    if supplied is None:
        supplied = str((await request.form()).get("csrf_token", ""))
    if (not isinstance(expected, str) or not supplied
            or not hmac.compare_digest(expected.encode(), supplied.encode())):
        raise HTTPException(403, "Форма устарела. Обновите страницу и повторите действие.")
    if request.headers.get("sec-fetch-site") == "cross-site":
        raise HTTPException(403, "Запрос с другого сайта запрещён")
    origin = request.headers.get("origin")
    if origin and origin != str(request.base_url).rstrip("/"):
        raise HTTPException(403, "Запрос с другого сайта запрещён")


def local_redirect(target: str) -> str:
    if (not target.startswith("/") or target.startswith("//") or "\\" in target
            or any(ord(char) < 32 or ord(char) == 127 for char in target)):
        return "/"
    parsed = urlsplit(target)
    return target if not parsed.scheme and not parsed.netloc else "/"


class LoginLimiter:
    """Ограниченная по памяти защита одного процесса; без доверия X-Forwarded-For."""

    def __init__(self, limit: int = 10, window: float = 300, capacity: int = 4096) -> None:
        self.limit, self.window, self.capacity = limit, window, capacity
        self._buckets: OrderedDict[str, tuple[float, int]] = OrderedDict()
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            since, count = self._buckets.get(key, (now, 0))
            if now - since >= self.window:
                since, count = now, 0
            self._buckets[key] = (since, count + 1)
            self._buckets.move_to_end(key)
            while len(self._buckets) > self.capacity:
                self._buckets.popitem(last=False)
            return count < self.limit


class RequestProtection:
    """Лимит тела до multipart-разбора, заголовки и nonce для скриптов."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        nonce = secrets.token_urlsafe(24)
        scope.setdefault("state", {})["csp_nonce"] = nonce

        async def secure_send(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers["Cache-Control"] = "no-store"
                headers["X-Content-Type-Options"] = "nosniff"
                headers["X-Frame-Options"] = "DENY"
                headers["Referrer-Policy"] = "same-origin"
                headers["Content-Security-Policy"] = (
                    "default-src 'self'; script-src 'nonce-" + nonce + "'; "
                    "style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
                    "media-src 'self'; object-src 'none'; base-uri 'none'; "
                    "frame-ancestors 'none'; form-action 'self'"
                )
            await send(message)

        if scope["method"] in {"POST", "PUT", "PATCH", "DELETE"}:
            path = scope["path"]
            limit = 64 * 1024
            if path == "/contacts/upload":
                limit = 9 * 1024 * 1024
            elif path.startswith("/questionnaires/") and path.endswith("/upload"):
                limit = 600 * 1024
            try:
                length = int(Headers(scope=scope).get("content-length", "0"))
                if length < 0:
                    raise ValueError
            except ValueError:
                await JSONResponse({"detail": "Неверная длина запроса"}, 400)(
                    scope, receive, secure_send,
                )
                return
            body = bytearray()
            if length <= limit:
                try:
                    async with asyncio.timeout(30):
                        while True:
                            message = await receive()
                            if message["type"] == "http.disconnect":
                                return
                            body.extend(message.get("body", b""))
                            if len(body) > limit or not message.get("more_body", False):
                                break
                except TimeoutError:
                    await JSONResponse({"detail": "Истекло время загрузки"}, 408)(
                        scope, receive, secure_send,
                    )
                    return
            if length > limit or len(body) > limit:
                await JSONResponse({"detail": "Запрос больше допустимого размера"}, 413)(
                    scope, receive, secure_send,
                )
                return
            delivered = False

            async def replay() -> Message:
                nonlocal delivered
                if delivered:
                    return await receive()
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}

            await self.app(scope, replay, secure_send)
        else:
            await self.app(scope, receive, secure_send)
