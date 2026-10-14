"""Rate limiting and API key authentication middleware.

Uses slowapi for token-bucket rate limiting (1200 req/min, burst 100, IP-keyed)
matching the Go backend's ApiRateLimitMiddleware pattern.

API key validation reads X-API-Key header and compares against settings.api_key.
"""


from collections.abc import Awaitable, Callable
from typing import Any

from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app import codes
from app.config import settings

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["1200/minute"],
    headers_enabled=True,
)


async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> dict[str, Any]:
    """Return unified {code, message} JSON envelope for rate-limit errors."""
    return {"code": codes.CODE_AI_RATE_LIMITED, "message": "Too many requests"}


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """Validate X-API-Key header on all routes. Health is exempt."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if request.url.path == "/health" or not settings.api_key:
            return await call_next(request)

        api_key = request.headers.get("x-api-key", "")
        if api_key != settings.api_key:
            return JSONResponse(
                status_code=401,
                content={
                    "code": codes.CODE_AI_UNAUTHORIZED,
                    "message": "Invalid or missing API key",
                },
            )

        return await call_next(request)
