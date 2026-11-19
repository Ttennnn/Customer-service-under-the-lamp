PYTHON ?= .venv/Scripts/python.exe
LANGFUSE_COMPOSE ?= docker compose -f docker-compose.langfuse.yml

.PHONY: langfuse-up langfuse-down cost-report calibrate-confidence flywheel eval-flywheel test

langfuse-up:
	$(LANGFUSE_COMPOSE) up -d

langfuse-down:
	$(LANGFUSE_COMPOSE) down

cost-report:
	$(PYTHON) -m app.jobs.cost_by_intent --days 7

calibrate-confidence:
	$(PYTHON) -m app.jobs.calibrate_confidence

flywheel:
	$(PYTHON) -m app.jobs.flywheel_pipeline

eval-flywheel:
	$(PYTHON) -m app.jobs.eval_flywheel --triggered-by 手动

test:
	$(PYTHON) -m pytest -q
