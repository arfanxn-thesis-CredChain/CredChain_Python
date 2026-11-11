# CredChain Python - Agent Instructions

Python AI service called by the Go backend over HTTP. Uses Google Gemini (Files API + direct upload) for document extraction and EmbeddingGemma (sentence-transformers) for semantic similarity embeddings. Requires a Gemini API key; no longer fully offline. When `API_KEY` is configured, authentication covers every path except `/health`. Rate-limited at 1200 req/min per IP via slowapi. Compose attaches to the external `credchain` network and publishes `127.0.0.1:8081:8081` for host access; **never expose to public internet**.

This file is the authoritative reference for AI assistants and engineers working in `CredChain_Python/`.

## Repo Position

Sibling to `CredChain_Golang/` (backend, sole HTTP caller), `CredChain_Solidity/` (contracts), and `CredChain_React/` (frontend).

- **Consumer:** the Go backend is the only intended caller. Requests flow `React → Go API → Python AI`. The frontend never talks to this service directly.
- **Locales:** `locales/{en,id}.json` contain four `verdict.*` description templates per language. These are separate from Go's API/mail catalogs; they do not share a synchronized key set.
- **Network isolation:** Compose attaches the service to `credchain` and exposes port 8081 on host loopback only. Configure `API_KEY` to enforce X-API-Key authentication on all paths except `/health`; an empty value disables it. Rate-limited at 1200 req/min per IP via slowapi ASGI middleware. CORS is configured via `CORS_ALLOW_ORIGINS`; exposing it publicly would allow arbitrary Gemini execution against arbitrary input.

## Critical Commands

```bash
# The Python service ALWAYS runs in Docker, started by the orchestrator:
#   cd ../CredChain_Golang && make local-up        # full stack (prod)
#   cd ../CredChain_Golang && make dev-up    # local hybrid

# Service-local tasks (run inside the container, no host venv needed):
make check              # ruff check + mypy app/ tests/ + pytest tests/ -v  (pre-push gate)
make format             # ruff format inside disposable container; host files are not mounted
make generate-api-key   # print key; container-local .env.docker is discarded
```

`.github/workflows/ci.yml` runs on pushes to `master`, but invokes nonexistent `make lint`, `make typecheck`, and `make test` targets. The canonical gate is `make check`; the workflow needs a separate configuration fix. There are no host run targets (`serve`/`dev`/`install`); the service is developed and run in the container.

## Environment Setup

Copy `.env.example` → `.env.docker` and fill in the keys below. Deps are installed when the container image is built; there is no host venv step. The service is started by the orchestrator (`cd ../CredChain_Golang && make local-up` or `make dev-up`) and binds `:8081`.

**Required API keys:**
- `GEMINI_API_KEY` — Google Gemini API key (fatal if empty; extraction calls will fail)
- `HF_TOKEN` — HuggingFace token for gated model access (required for EmbeddingGemma download)

EmbeddingGemma is downloaded at startup from HuggingFace Hub via `sentence-transformers`. No baked-in model files.

## Tech Stack

Pinned in `pyproject.toml`:

| Layer | Tool | Version |
|---|---|---|
| Language | Python | ≥3.11, <3.13 |
| Web framework | FastAPI | 0.115.5 |
| ASGI server | uvicorn[standard] | 0.32.1 |
| Gemini SDK | google-genai | 2.8.0 |
| Embeddings | sentence-transformers (EmbeddingGemma) | ≥3.3.1 |
| HuggingFace | huggingface-hub | 1.18.0 |
| Math | numpy | — |
| Image | Pillow | 11.0.0 |
| ML framework | torch (CPU-only, installed in Dockerfile) | unpinned |
| Rate limiting | slowapi | 0.1.9 |
| CLI | typer | (transitive dep via huggingface_hub — not declared in pyproject.toml) |
| Test | pytest | 8.3.4 |
| Async test | pytest-asyncio | 0.24.0 |
| HTTP test | httpx | 0.28.1 |
| Lint | ruff | 0.8.4 |
| Type check | mypy | 1.13.0 |

## Project Architecture

Flat layout under `app/`. 15 source modules + `tests/`:

```
CredChain_Python/
  app/
    main.py             → FastAPI app + lifespan (loads EmbeddingGemma + initializes Gemini client) + i18n middleware + error handlers
    routes.py           → 4 endpoints: /extract /verify /extract-ids (Gemini), /health (readiness)
    schemas.py          → Pydantic Response[T] envelope + per-endpoint payloads
    config.py           → pydantic-settings .env loader + custom CSV env source
    codes.py            → 50xxxx response codes (System/Extract/Verify/ExtractIds/Health)
    errors.py           → AppError + http_status_for
    logger.py           → structured JSON logger (mirrors Go's zap shape)
    gemini.py           → GeminiClient — Files API (batch) + direct upload (single) + retry logic
    embeddings.py       → EmbeddingGemma encode + cosine_similarity helper
    description.py      → bilingual description from locales/ templates (no LLM call)
    i18n.py             → locale loader + localize(key, lang, **vars)
    verdict.py          → similarity → verdict mapping (configurable thresholds)
    prompts.py          → Gemini prompt constants (PROMPT_EXTRACT_DOCUMENT, PROMPT_EXTRACT_IDS)
    middleware.py       → slowapi rate limiter (1200/min, IP-keyed) + API key authentication middleware
    cli.py              → Typer CLI — `generate-api-key` command for API key generation
  tests/                → conftest.py + test files (fully mocked)
  locales/              → tracked, JSON locale files (id, en) for description templates
  pyproject.toml        → pinned deps + ruff + mypy + pytest config
  Makefile              → all critical commands
  Dockerfile            → single-stage Python 3.11-slim build
  docker-compose.yml    → AI service + backend network attach
  .env / .env.docker / .env.example
  README.md
  AGENTS.md             → this file
```

## Key Patterns & Conventions

### Endpoints

All POST endpoints accept `files: list[UploadFile]` (multi-file batch). `MAX_FILES_PER_REQUEST` configures the batch limit, defaulting to 100.

| Method | Path | Purpose | Code |
|---|---|---|---|
| POST | `/extract` | Gemini Files API extraction, returns `{text, ids, embedding}` | 500100 |
| POST | `/verify` | Gemini direct upload + EmbeddingGemma similarity, returns `{similarity_score, similarity_percent, verdict, descriptions}` | 500200 |
| POST | `/extract-ids` | Gemini ID extraction, returns `{ids}` only | 500300 |
| GET | `/health` | Liveness, returns `{"code": 500900, "message": "healthy"}` (200) or `{"code": 500950, "message": "model loading"}` (503) | 500900 / 500950 |

Upload limit: 10 MB per file. Allowed MIME: `application/pdf`, `image/{jpeg,png,webp,tiff}`. `validate_files` first checks that the batch is nonempty and within `MAX_FILES_PER_REQUEST`. Each endpoint then calls `validate_file` inside its per-file loop to reject unsupported MIME types, empty files, and oversized files before processing that file.

### Multi-file Response Envelope

```json
{
  "code": 500100,
  "message": "Document(s) extracted successfully",
  "data": [
    { "text": "...", "ids": [...], "embedding": [...] },
    null,
    { "text": "...", "ids": [...], "embedding": [...] }
  ],
  "errors": {
    "files.1": ["Gemini extraction failed"]
  }
}
```

- `data` is always a list of length `len(files)`. Failed files = `null`.
- `errors` keys use `files.<index>` notation (mirrors Go's field-error convention).
- Top-level `code` is the success code when ≥1 file succeeded; an error code when all failed.
- Shape mirrors the Go backend's `{code, message, data, errors}` envelope — same wire contract.

### `/verify` Request Format

`/verify` accepts `files` (multipart) and `embeddings` (JSON string of float arrays). Each file pairs positionally with one embedding array:

```
POST /verify
files: <file0.pdf>
files: <file1.pdf>
embeddings: [[0.1, ...], [0.3, ...]]
```

`len(files) == len(embeddings)` is required; mismatch returns HTTP 400 code `500241`. `parse_embeddings` in `routes.py` validates the JSON shape and raises `AppError` on malformed input.

### Gemini Pipeline (Files API vs Direct Upload)

Two extraction paths in `app/gemini.py`:

- **Files API** (used by `/extract`): Upload files → poll until ACTIVE → generate content. Better for batch, supports larger files via Gemini's server-side processing.
- **Direct upload** (used by `/verify` and `/extract-ids`): Bytes inlined in the prompt. Simpler per-file flow, no polling overhead.

`GeminiClient` wraps both with retry logic for 429/RESOURCE_EXHAUSTED errors. `MAX_RETRIES` defaults to 2 total attempts (including the first), with a `RETRY_WAIT_SECONDS` code default of 2 seconds. `.env.example` overrides the delay to 60 seconds.

### Models Loaded via Lifespan

EmbeddingGemma (`google/embeddinggemma-300M`) is loaded once via FastAPI `lifespan` and accessed in handlers via `Depends(get_embedding_model)`. The Gemini client is initialized at startup and injected via `Depends(get_gemini_client)`.

The lifespan handler sets `app.state.models_loaded = True` after both initialize successfully. `/health` returns code `500900` (loaded, HTTP 200) or `500950` (not yet ready, HTTP 503).

### Single-Worker Concurrency

Uvicorn runs with a single worker (`--workers 1`), loading one model instance per process. Handlers offload blocking work with `asyncio.to_thread`, so one worker does not guarantee serialized model calls or thread safety.

The Go Python client serializes requests with a per-instance mutex. Other client instances or backend replicas do not share that lock. Operators should size connection pools and request timeouts accordingly; service-wide serialization is not established by this setup.

### Gemini Prompts

Extraction prompts are module-level constants in `app/prompts.py`:

- `PROMPT_EXTRACT_DOCUMENT` — extracts raw text + IDs, returns `{raw_text: str, ids: [{type: str, value: str}]}`
- `PROMPT_EXTRACT_IDS` — extracts IDs only, returns `{ids: [{type: str, value: str}]}`

Gemini is configured with `response_mime_type="application/json"` for structured JSON output. Empty, malformed JSON, or non-object responses raise `AppError` with `CODE_AI_GEMINI_FAILED`; malformed JSON also prints a warning.

### Verdict Thresholds

Verdict thresholds are configurable via env vars (defined in `app/verdict.py` and `app/config.py`):

| Verdict | Default Threshold | Description |
|---|---|---|
| `tampered` | ≥ 0.95 | Suspiciously near-perfect — likely copy with minor edits |
| `suspicious` | ≥ 0.75 | High similarity suggesting a derivative |
| `low_similarity` | ≥ 0.55 | Some overlap but uncertain |
| `not_similar` | < 0.55 | Unrelated documents |

Env vars: `VERDICT_TAMPERED_THRESHOLD`, `VERDICT_SUSPICIOUS_THRESHOLD`, `VERDICT_LOW_SIMILARITY_THRESHOLD`.

**Important:** These labels describe the implemented similarity heuristic, not proof of tampering or authenticity. Cosine similarity alone does not establish whether a document was altered. Go performs its own exact-hash and lifecycle checks before using Python similarity results.

The verdict is computed from embedding cosine similarity alone (EmbeddingGemma). Similarity is formatted as `"XX.X%"` via `verdict.format_percent()`.

### Locale-Based Description Generation

`/verify` returns **bilingual** human-readable descriptions for each verdict (always both English and Indonesian). Descriptions are rendered from `locales/{en,id}.json` templates — **no LLM call** for descriptions. This keeps `/verify` fast and deterministic.

`app/i18n.localize(key, lang, **vars)` resolves the template and substitutes variables. `app/description.build_description(verdict, similarity_percent)` returns `{"en": "...", "id": "..."}` — always both languages.

### i18n Middleware

`main.py` registers an i18n middleware (mirrors Go backend pattern):

```python
@app.middleware("http")
async def i18n_middleware(request, call_next):
    accept_lang = request.headers.get("Accept-Language", "").strip()
    lang = accept_lang.split(",")[0].split(";")[0].strip().lower()
    request.state.lang = lang if lang in {"id", "en"} else "id"
    return await call_next(request)
```

### API Key Authentication

The active middleware is the inline `api_key_middleware` in `main.py`, not the unused `ApiKeyMiddleware` class in `middleware.py`.

Every path except `/health`, including `/extract`, `/verify`, and `/extract-ids`, requires an `X-API-Key` header matching `settings.api_key` when configured. If `API_KEY` is empty, authentication is disabled.

The `api_key_middleware` function in `main.py` validates the header as an HTTP middleware (registered via `@app.middleware("http")`):

```python
@app.middleware("http")
async def api_key_middleware(request, call_next):
    if request.url.path == "/health" or not settings.api_key:
        return await call_next(request)
    if request.headers.get("x-api-key", "") != settings.api_key:
        return JSONResponse(status_code=401, content={
            "code": 500010, "message": "Invalid or missing API key"})
    return await call_next(request)
```

### Rate Limiting

Global rate limiter via slowapi (ASGI middleware), keyed by IP address — mirrors the Go backend's `ApiRateLimitMiddleware`:

| Limit | Burst | Key | Scope |
|---|---|---|---|
| 1200 req/min | — | `get_remote_address` (IP) | All routes |

Configuration in `app/middleware.py`:

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["1200/minute"],
    headers_enabled=True,
)
```

Registered as `SlowAPIASGIMiddleware` in `main.py`. Rate-limit exceeded returns `{code: 500051, message: "Too many requests"}` with HTTP 429.

### CLI — API Key Generation

Generate a 64-char hex API key:

```bash
make generate-api-key       # prints key; writes .env.docker inside disposable container only
```

Implemented via Typer in `app/cli.py` using `secrets.token_hex(32)`. The `--env`/`-e` flag specifies the target file. Finds `API_KEY=` line and replaces; appends if not found. Prints the generated key to stdout. The Make target does not mount the host env file, so manually save the printed key to the host `.env.docker` and keep it out of logs and source control. `make format` has the same persistence limitation for source edits and assumes Ruff is installed in the image.

### Custom Env CSV Source

`pydantic-settings 2.6.1` does not export `NoDecode`, which breaks list-of-string env parsing. `app/config.py` defines `_CsvEnvSettingsSource` and `_CsvDotEnvSettingsSource` that pre-split comma-separated values for `cors_allow_origins` before pydantic-settings sees them.

## Configuration / Env Vars

| Var | Default | Purpose |
|---|---|---|
| `FASTAPI_PORT` | `8081` | Settings field; Docker CMD and Compose use fixed port 8081 rather than this value |
| `LOG_LEVEL` | `info` | `debug` / `info` / `warn` / `error` |
| `LOG_OUTPUT` | `stdout` | `stdout` or file path |
| `GEMINI_API_KEY` | — | Google Gemini API key (required) |
| `EXTRACTION_MODEL` | `gemini-3.1-flash-lite` | Gemini model for extraction |
| `RETRY_WAIT_SECONDS` | `2` | Delay between Gemini rate-limit retries; `.env.example` sets 60 |
| `MAX_RETRIES` | `2` | Total Gemini generation attempts, including the first |
| `GEMINI_TIMEOUT_MS` | `20000` | Gemini client request timeout in milliseconds |
| `FILE_POLL_TIMEOUT_SECONDS` | `20` | Files API processing poll timeout |
| `HF_TOKEN` | — | HuggingFace token for gated model access |
| `EMBEDDING_MODEL_ID` | `google/embeddinggemma-300M` | HuggingFace model ID for embeddings |
| `API_KEY` | — | Shared secret for X-API-Key authentication (empty = disabled) |
| `VERDICT_TAMPERED_THRESHOLD` | `0.95` | Cosine similarity ≥ this → tampered |
| `VERDICT_SUSPICIOUS_THRESHOLD` | `0.75` | Cosine similarity ≥ this → suspicious |
| `VERDICT_LOW_SIMILARITY_THRESHOLD` | `0.55` | Cosine similarity ≥ this → low_similarity |
| `LOCALES_DIR` | `./locales` | Parent dir for locale JSON files |
| `MAX_FILES_PER_REQUEST` | `100` | Configurable multi-file batch limit |
| `CORS_ALLOW_ORIGINS` | `*` | Comma-separated origins |

## Testing

- **Framework:** pytest 8.3.4 + pytest-asyncio 0.24.0 + httpx 0.28.1 (for FastAPI TestClient)
- **Fixtures:** `tests/conftest.py` provides mocked `mock_embedding_model` and `mock_gemini_client` dependencies.
- **Coverage layout:** 7 of 15 source modules have test files: `test_cli.py`, `test_description.py`, `test_embeddings.py`, `test_gemini.py`, `test_routes.py`, `test_schemas.py`, `test_verdict.py`.

Modules without tests: `codes.py`, `config.py`, `errors.py`, `i18n.py`, `logger.py`, `main.py`, `middleware.py`, `prompts.py`.
- **mypy:** strict mode on `app/`; relaxed on `tests/` (allows missing return type annotations on test functions).
- **ruff config:** select `E,F,I,N,UP,B,SIM,RET,PT`; line-length 100; `B008` ignored in `app/routes.py` (FastAPI `Depends()` / `File()` defaults are idiomatic).
- **pytest config:** `testpaths=["tests"]`, asyncio mode auto.

When adding a new endpoint or module, add at least: one happy-path test, one validation-failure test, one downstream-failure test (mock the model raising), and one mypy-clean signature.

## Cross-Repo Integration

- **`../CredChain_Golang/AGENTS.md`**: intended sole HTTP caller. Each Go Python-client instance serializes its requests. Both services use `{code, message, data, errors}` envelopes, but Go translates downstream errors rather than forwarding the Python response unchanged.
- **`../CredChain_Solidity/AGENTS.md`** — no integration; this service never touches the chain.

Response codes use a 6-digit `AABBCC` format. This service owns category `50` (AI). The Go backend uses categories `10` (system), `20` (auth), `30` (user), `40` (credential). The Go client converts Python failures to Go errors; credential verification maps them to backend codes such as `400444` and `400446`, rather than preserving the original `50xxxx` code.

Locale files in `locales/{en,id}.json` provide the four `verdict.*` description templates rendered by `/verify`. Keep the English and Indonesian Python catalogs aligned; Go's API/mail catalogs have a separate key set.

## Deployment

**Push to master branch only when build succeeds. Do not create feature branches, bugfix branches, or any other branch types — commit directly to master.**

Before pushing, run the repo's canonical verification command and confirm it passes:

```bash
make check
```

### API Key Sync

After running `make generate-api-key`, save the printed key as `API_KEY` in the host
`.env.docker` and as `PYTHON_AI_API_KEY` in the active Go env file (`.env.docker` for Docker,
`.env` for host development). The Make target does not update the host file.
If Python's nonempty key and Go's value mismatch, AI requests fail with 401.

## See Also

- `README.md` — quick-start for human contributors
- `Makefile` — canonical commands
- `pyproject.toml` — frozen dependency pins + tool config
- `../AGENTS.md` (workspace root): multi-repo reference
- `../CredChain_Golang/AGENTS.md` — backend contract reference
