PYTHON ?= .venv/Scripts/python.exe
LANGFUSE_COMPOSE ?= docker compose -f docker-compose.langfuse.yml

.PHONY: langfuse-up langfuse-down cost-report test

langfuse-up:
	$(LANGFUSE_COMPOSE) up -d

langfuse-down:
	$(LANGFUSE_COMPOSE) down

cost-report:
	$(PYTHON) -m app.jobs.cost_by_intent --days 7

test:
	$(PYTHON) -m pytest -q
