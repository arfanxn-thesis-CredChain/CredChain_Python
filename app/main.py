"""CredChain Python AI Service — FastAPI application factory.

Single uvicorn worker, EmbeddingGemma loaded once via lifespan, Gemini
client initialized at startup. i18n middleware mirrors Go backend pattern.
Structured JSON logging, unified error envelope.
Reachable only inside the Docker backend network — never exposed publicly.
"""

import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from google import genai
from google.genai import types
from huggingface_hub import login as hf_login
from sentence_transformers import SentenceTransformer
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIASGIMiddleware

from app import codes
from app.config import settings
from app.errors import AppError, http_status_for
from app.gemini import GeminiClient
from app.logger import get_logger
from app.middleware import limiter, rate_limit_exceeded_handler
from app.routes import router

log = get_logger("main")

SUPPORTED_LANGS = {"id", "en"}
DEFAULT_LANG = "id"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load EmbeddingGemma + initialize Gemini client at startup."""
    log.info("loading models", extra={"extra_fields": {"phase": "startup"}})
    app.state.models_loaded = False

    hf_login(token=settings.hf_token)
    app.state.gemini_raw_client = genai.Client(
        api_key=settings.gemini_api_key,
        http_options=types.HttpOptions(timeout=settings.gemini_timeout_ms),
    )
    app.state.gemini_client = GeminiClient(
        app.state.gemini_raw_client,
        extraction_model=settings.extraction_model,
        retry_wait_seconds=settings.retry_wait_seconds,
        max_retries=settings.max_retries,
        file_poll_timeout_seconds=settings.file_poll_timeout_seconds,
    )

    log.info(
        "loading embedding model",
        extra={"extra_fields": {"model": settings.embedding_model_id}},
    )
    app.state.embedding_model = SentenceTransformer(settings.embedding_model_id, device="cpu")

    app.state.models_loaded = True
    log.info("models ready", extra={"extra_fields": {"phase": "ready"}})

    yield

    log.info("shutting down", extra={"extra_fields": {"phase": "shutdown"}})


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=http_status_for(exc.code),
            content={"code": exc.code, "message": exc.message, "errors": exc.errors},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError,
    ) -> JSONResponse:
        field_errors: dict[str, list[str]] = {}
        for err in exc.errors():
            loc = ".".join(str(p) for p in err["loc"][1:]) or "body"
            field_errors.setdefault(loc, []).append(err["msg"])
        return JSONResponse(
            status_code=400,
            content={
                "code": codes.CODE_AI_VALIDATION,
                "message": "Validation failed",
                "errors": field_errors,
            },
        )

    @app.exception_handler(RateLimitExceeded)
    async def _rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
        body = await rate_limit_exceeded_handler(request, exc)
        return JSONResponse(status_code=429, content=body)

    @app.exception_handler(Exception)
    async def unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error", extra={"extra_fields": {"path": request.url.path}})
        return JSONResponse(
            status_code=500,
            content={"code": codes.CODE_AI_INTERNAL, "message": "Internal server error"},
        )


def create_app() -> FastAPI:
    app = FastAPI(title="CredChain Python AI Service", version="0.2.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=False,
    )

    app.state.limiter = limiter
    app.add_middleware(SlowAPIASGIMiddleware)

    @app.middleware("http")
    async def api_key_middleware(
        request: Request, call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        if request.url.path == "/health" or not settings.api_key:
            return await call_next(request)
        if request.headers.get("x-api-key", "") != settings.api_key:
            return JSONResponse(
                status_code=401,
                content={
                    "code": codes.CODE_AI_UNAUTHORIZED,
                    "message": "Invalid or missing API key",
                },
            )
        return await call_next(request)

    @app.middleware("http")
    async def i18n_middleware(
        request: Request, call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        accept_lang = request.headers.get("Accept-Language", "").strip()
        lang = accept_lang.split(",")[0].split(";")[0].strip().lower()
        request.state.lang = lang if lang in SUPPORTED_LANGS else DEFAULT_LANG
        return await call_next(request)

    @app.middleware("http")
    async def request_logging_middleware(
        request: Request, call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request.state.request_id = str(uuid.uuid4())
        start = time.perf_counter()
        response = await call_next(request)
        latency_ms = (time.perf_counter() - start) * 1000
        status_code = response.status_code
        outcome = (
            "success" if 200 <= status_code < 300
            else "client_error" if 400 <= status_code < 500
            else "server_error"
        )
        log.info(
            "request completed",
            extra={"extra_fields": {
                "request_id": request.state.request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": status_code,
                "latency_ms": round(latency_ms, 2),
                "outcome": outcome,
            }},
        )
        return response

    register_error_handlers(app)
    app.include_router(router)
    return app


app = create_app()
