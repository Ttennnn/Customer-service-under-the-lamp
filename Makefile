PYTHON ?= .venv/Scripts/python.exe
ML_PYTHON ?= .venv-ml/Scripts/python.exe
CH10_RUN = PYTHONPATH=. $(ML_PYTHON) -m
LANGFUSE_COMPOSE ?= docker compose -f docker-compose.langfuse.yml

.PHONY: langfuse-up langfuse-down cost-report calibrate-confidence flywheel eval-flywheel eval-flywheel-prompts test ch10-env ch10-golden ch10-corpus ch10-dataset ch10-train ch10-export classifier-up classifier-down ch10-eval ch10-threshold-scan classify-pool

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

# --- ch10 topic classifier -------------------------------------------------

ch10-env:
	uv venv --python 3.12 --allow-existing .venv-ml
	uv pip install --python .venv-ml --extra-index-url https://download.pytorch.org/whl/cu124 -r requirements-ch10.txt

ch10-golden:
	$(CH10_RUN) scripts.ch10.validate_golden --golden data/ch10/golden_samples.jsonl --out data/ch10/reports/golden_report.json

ch10-corpus:
	$(CH10_RUN) scripts.ch10.build_corpus --stage all
	$(CH10_RUN) scripts.ch10.prelabel --input data/ch10/corpus_clean.jsonl --out data/ch10/corpus_real_labeled.jsonl
	$(CH10_RUN) scripts.ch10.synthetic --real-labeled data/ch10/corpus_real_labeled.jsonl --out-dir data/ch10/generated --labeled-out data/ch10/corpus_labeled.jsonl --max-llm-calls 400
	$(CH10_RUN) scripts.ch10.audit --labeled data/ch10/corpus_labeled.jsonl --out data/ch10/sample_review.md

ch10-dataset:
	$(CH10_RUN) scripts.ch10.build_dataset --labeled data/ch10/corpus_labeled.jsonl --out data/ch10/dataset
	$(CH10_RUN) scripts.ch10.augment_train

ch10-train:
	$(CH10_RUN) scripts.ch10.train --train data/ch10/dataset/train_augmented.jsonl --val data/ch10/dataset/val.jsonl --out data/ch10/model

ch10-export:
	$(CH10_RUN) scripts.ch10.export_onnx --model data/ch10/model --out data/ch10/onnx --val data/ch10/dataset/val.jsonl --reports data/ch10/reports

classifier-up:
	$(CH10_RUN) scripts.ch10.serve --daemon --port 8110

classifier-down:
	$(CH10_RUN) scripts.ch10.serve --stop

ch10-eval: classifier-up
	$(CH10_RUN) scripts.ch10.evaluate --test data/ch10/dataset/test.jsonl --threshold data/ch10/model/threshold.json --base-url http://127.0.0.1:8110 --out data/ch10/reports

ch10-threshold-scan: classifier-up
	$(CH10_RUN) scripts.ch10.scan_threshold_replay --val data/ch10/dataset/val.jsonl --base-url http://127.0.0.1:8110 --threshold data/ch10/model/threshold.json --out data/ch10/reports

classify-pool:
	$(CH10_RUN) scripts.ch10.classify_pool $(if $(FORCE),--force,)
