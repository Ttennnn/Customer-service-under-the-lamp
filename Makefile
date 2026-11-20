PYTHON ?= .venv/Scripts/python.exe
LANGFUSE_COMPOSE ?= docker compose -f docker-compose.langfuse.yml

.PHONY: langfuse-up langfuse-down cost-report calibrate-confidence flywheel eval-flywheel eval-flywheel-prompts test

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

# Real-LLM prompt quality gate (>= 80% on tests/eval/flywheel_samples.json).
# Importing the module is offline; this target is the live run.
eval-flywheel-prompts:
	$(PYTHON) -m app.jobs.eval_flywheel_prompts

test:
	$(PYTHON) -m pytest -q
