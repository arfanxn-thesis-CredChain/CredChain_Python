# CredChain Python (AI service) — always runs in Docker.
# Orchestration (up/down/logs) lives in CredChain_Golang. These are the
# service-local tasks only.

.PHONY: check format generate-api-key

# Pre-push gate: lint + typecheck + tests, run inside the container (no host venv).
check:
	docker compose run --build --rm -v "$(CURDIR)/tests:/app/tests:ro" python sh -c "pip install --no-cache-dir -e '.[dev]' && ruff check && mypy app/ tests/ && pytest tests/ -v"

format:
	docker compose run --rm python ruff format .

generate-api-key:
	docker compose run --rm python python -m app.cli --env .env.docker
