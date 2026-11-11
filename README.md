# CredChain — Python AI Service

Internal FastAPI service for document OCR, field extraction (Gemini), semantic similarity (EmbeddingGemma), and bilingual description rendering.

Never expose this service to the public internet. Compose attaches it to the external `credchain` network at `http://python:8081` and publishes port 8081 on host loopback (`127.0.0.1`) only.

## Stack

Python 3.11 · FastAPI 0.115.5 · Uvicorn 0.32.1 · Google Gemini (genai 2.8.0) · EmbeddingGemma via sentence-transformers · torch (CPU-only, unpinned Docker install) · slowapi 0.1.9

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/extract` | Batch OCR + Gemini extraction + EmbeddingGemma embedding |
| POST | `/verify` | Batch similarity + verdict + bilingual description |
| POST | `/extract-ids` | Batch ID extraction (Gemini-native) |
| GET | `/health` | Liveness + model readiness |

All POST endpoints accept multiple files. Response shape: `{code, message, data, errors}`.
`/verify` accepts an `embeddings` JSON form field (array of float arrays), one per file.
When `API_KEY` is configured, middleware requires `X-API-Key` on every path except `/health`, including the POST endpoints. An empty `API_KEY` disables authentication.

### Verdicts

| Verdict | Threshold |
|---|---|
| `tampered` | similarity ≥ 0.95 |
| `suspicious` | similarity ≥ 0.75 |
| `low_similarity` | similarity ≥ 0.55 |
| `not_similar` | similarity < 0.55 |

## Quick Start

The service always runs in Docker, started by the orchestrator.

```bash
cp .env.example .env.docker
# Set GEMINI_API_KEY and HF_TOKEN
# Set API_KEY in this host .env.docker and the same value as Go's PYTHON_AI_API_KEY.
# make generate-api-key prints a key, but its container-local file is discarded.

# Start it (from the orchestrator repo):
cd ../CredChain_Golang && make local-up     # full stack, or `make dev-up` for local hybrid
```

The health check waits up to 600s for the embedding model to download on first start.

## Project Structure

```
app/
├── main.py           # FastAPI app, lifespan (model loading), middleware
├── routes.py         # 4 endpoints
├── schemas.py        # Pydantic response models
├── config.py         # Pydantic-settings env loader
├── codes.py          # 50xxxx response codes
├── errors.py         # AppError + HTTP status mapping
├── gemini.py         # Gemini client (Files API + direct upload + retry)
├── embeddings.py     # EmbeddingGemma encode + cosine_similarity
├── verdict.py        # Similarity → verdict mapping
├── description.py    # Bilingual description from locale templates
├── i18n.py           # Locale loader + localize()
├── prompts.py        # Gemini prompt constants
├── middleware.py      # slowapi rate limiter
├── cli.py            # Typer CLI (generate-api-key)
└── logger.py         # Structured JSON logger
```

## Key Commands

| Command | Purpose |
|---|---|
| `make check` | Ruff lint + Mypy + pytest (pre-push gate, runs in container) |
| `make format` | Ruff format in a disposable container; no host-source mount, so edits do not persist to the checkout |
| `make generate-api-key` | Print a 64-char hex API key; writes only the disposable container's `.env.docker` |

Starting/stopping the service is done from the orchestrator (`CredChain_Golang`: `make local-up` / `make down` / `make dev-up`).

## Related Docs

- [AGENTS.md](AGENTS.md) — Full architecture, middleware details, cross-repo integration
