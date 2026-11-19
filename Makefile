PYTHON ?= .venv/Scripts/python.exe
LANGFUSE_COMPOSE ?= docker compose -f docker-compose.langfuse.yml

.PHONY: langfuse-up langfuse-down test

langfuse-up:
	$(LANGFUSE_COMPOSE) up -d

langfuse-down:
	$(LANGFUSE_COMPOSE) down

test:
	$(PYTHON) -m pytest -q
