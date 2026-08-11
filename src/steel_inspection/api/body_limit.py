"""Dependency-free ASGI request-body limiting."""

from collections.abc import Awaitable, Callable
from typing import Any

from starlette.exceptions import HTTPException

Scope = dict[str, Any]
Message = dict[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]


class _RequestBodyTooLarge(HTTPException):
    """Interrupt downstream request parsing once the byte limit is exceeded."""

    def __init__(self) -> None:
        super().__init__(status_code=413, detail="Content Too Large")


class RequestBodyLimitMiddleware:
    """Reject HTTP bodies over a byte limit before FastAPI parses multipart uploads."""

    def __init__(self, app: ASGIApp, max_body_size: int) -> None:
        self.app = app
        self.max_body_size = max_body_size

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        if _content_length(scope) > self.max_body_size:
            await _send_too_large(send)
            return

        total_size = 0
        response_started = False

        async def receive_with_limit() -> Message:
            nonlocal total_size
            message = await receive()
            if message["type"] == "http.request":
                total_size += len(message.get("body", b""))
                if total_size > self.max_body_size:
                    raise _RequestBodyTooLarge
            return message

        async def send_with_tracking(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, receive_with_limit, send_with_tracking)
        except _RequestBodyTooLarge:
            if response_started:
                raise
            await _send_too_large(send)


def _content_length(scope: Scope) -> int:
    """Return a valid Content-Length header or zero when it is absent or malformed."""
    for name, value in scope.get("headers", []):
        if name.lower() == b"content-length":
            try:
                return int(value)
            except (TypeError, ValueError):
                return 0
    return 0


async def _send_too_large(send: Send) -> None:
    """Send a minimal ASGI 413 response without handing the body to FastAPI."""
    body = b"Content Too Large"
    await send(
        {
            "type": "http.response.start",
            "status": 413,
            "headers": [(b"content-type", b"text/plain; charset=utf-8"), (b"content-length", str(len(body)).encode())],
        }
    )
    await send({"type": "http.response.body", "body": body, "more_body": False})
