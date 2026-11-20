# ch10 多标签主题分类器 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 训练一个 RoBERTa-wwm-ext 多标签主题分类器，把飞轮低置信度问题按 17 类批量归类，落 `topic_classifications`，并在飞轮后台展示主题分布与九项验收证据。

**Architecture:** 分阶段离线 JSONL 流水线（捞取/清洗/预标/模拟补足/切分/增强）→ transformers 全参微调 → ONNX 导出 → FastAPI + onnxruntime 独立推理服务（:8110）→ 旁路 `classify_pool` 批量写库 → 主应用只读 API + 原生静态页。主应用与 ML 依赖完全隔离。

**Tech Stack:** Python 3.12 `.venv-ml`（torch、transformers、accelerate、scikit-learn、onnx、onnxruntime、tokenizers、numpy）、Python 3.13 主 `.venv`（FastAPI、SQLAlchemy、LangChain）、MySQL 8、原生 HTML/CSS/JS、GNU Make（Windows Git Bash）。

**Spec:** `docs/superpowers/specs/2026-11-20-ecommerce-cs-ch10-topic-classifier-design.md`

## Global Constraints

- **17 类与顺序固定**：退换货、物流、尺码、发票、质量问题、运费、优惠活动、价保、支付、订单修改、库存补货、商品信息、保修维修、账号、会员积分、评价、其他。仅 `app/core/taxonomy.py` 定义一份，其他模块 import。
- **标注铁律**：字面提到几个诉求打几个标签，一个不多一个不少；「其他」互斥兜底（命中任何具体类就不打「其他」；全不过线取最高分）。
- **数据规模**：每类约 100 个正例、总量约 1700（以标签计）；方言约 15%、错别字约 10%、多诉求句约 15%；真实问题按用户选的 A 混入 80/10/10。
- **切分**：80/10/10，标签组合分层，组合 <10 按第一标签归层；切分前全局近重复合并（字符 3-gram Jaccard ≥0.85）；跨集 exact 文本重叠必须为 0。
- **增强只扩训练集**：每条训练样本 1 个变体；验证/测试原样。
- **训练**：`hfl/chinese-roberta-wwm-ext`，全参微调（禁用 LoRA），lr=2e-5、batch=16、max_length=128、epochs≤8、weight_decay=0.01、seed=42；早停盯验证集 micro-F1、patience=2。
- **阈值**：候选 0.30–0.70 步进 0.05；torch 扫描只作对照，导出后用 ONNX 在验证集重扫并覆盖 `threshold.json`；评测、服务、重演一律以 ONNX 为准。
- **红线是诊断展示**：严档（退换货、物流、尺码、发票、质量问题）F1≥0.9；中档（运费、优惠活动、价保、支付、订单修改、库存补货、商品信息、保修维修）≥0.8；宽档不设线。不达标标红，不阻塞交付；严档过半不达标最多补一轮。
- **部署**：ONNX + onnxruntime + tokenizers，服务 :8110，进程不 import torch/transformers；`/classify` 批量上限 64；`/healthz` 探活。
- **旁路**：`classify_pool` 只处理已归并（`matched_review_id IS NOT NULL`）且未归类的行；默认 `--min-batch 10`；幂等；主链路 `/chat` 不调用分类器。
- **DDL**：`sql/ch10-ddl.sql` 原样使用 spec §10.1，不改列类型/唯一键/外键。
- **依赖隔离**：主 `.venv` 零 ML 依赖；所有 `scripts/ch10/*` 用 `.venv-ml`（CPython 3.12）；主应用与测试用 `.venv`（Python 3.13）。
- **库/框架 API**：transformers、torch.onnx、onnxruntime、FastAPI、SQLAlchemy 动手前先用 Context7 MCP 的 `resolve-library-id` + `query-docs` 查当前官方文档；不凭记忆写。
- **留痕**：每个任务完成后在 `dev-notes/ch10.md` 追加一段，记「用户关键原话 / 关键产出 / 拒绝或纠偏 / 翻车与返工」；不许收尾一次性补。
- **前端例外**：主题页与验收页走 Vibe Coding，用户描述效果、直接改；不套 brainstorm/TDD/code review。
- **提交**：每个任务一个 commit，测试与 dev-notes 一起提交；提交信息用 `feat:` / `test:` / `docs:` / `fix:`。

## Review Focus

1. **多诉求句与近邻类目**：如「买大了想退」必须同时命中尺码+退换货；「开胶了能换吗」质量问题+退换货；「退货运费谁出」只命中运费，不命中物流。测试在 Task 6（黄金样例的 5 个多诉求/近邻断言）与 Task 21（验收单句现场打分）。
2. **空标签与「其他」**：任何输入都不能产出空 labels；具体类命中时不得同时出现「其他」；全不过线取最高分。测试在 Task 1、Task 13、Task 15。
3. **跨集泄漏**：训练/验证/测试不能有 exact 重复文本；近重复改写不能绕过。测试在 Task 9（split/manifest）与 Task 21（验收 data 页）。
4. **幂等与部分失败**：`classify_pool` 重跑不重复写；服务中途失败不留半批；唯一键冲突可恢复。测试在 Task 15。
5. **产物缺失与服务离线**：验收 API 回 `present=false` 而不是 500；服务没起时页面显示离线而不是白屏；非法类目回 400。测试在 Task 13、Task 16、Task 18。

---

## File Structure

**共享/主应用**
- `app/core/taxonomy.py`：17 类、边界、示例、label id、阈值应用与兜底。
- `app/core/paths.py`：`data/ch10` 与 `log/acceptance` 路径常量。
- `app/core/jobs.py`：make 子进程白名单作业运行器。
- `app/models/topic.py`：`TopicClassification` ORM。
- `app/db/init_ch10.py`：幂等建表。
- `app/api/topics.py`：分布/问题列表 API + 两个页面路由。
- `app/api/acceptance.py`：九项验收只读 API + 单句试分类代理。
- `app/api/jobs.py`：作业发起/状态/停止 API。
- `app/static/topics.html`、`topic-questions.html`、`acceptance*.html`、`acceptance.css`、`acceptance.js`；修改 `admin.js`、`admin.html`。

**ML 脚本（`.venv-ml`）**
- `scripts/ch10/corpus_lib.py`：Sample、稳定 id、脱敏、去重、近重复、JSONL。
- `scripts/ch10/prompts.py`、`schemas.py`、`llm_io.py`：Prompt、结构化 schema、调用预算。
- `scripts/ch10/build_corpus.py`：捞真实 + 清洗。
- `scripts/ch10/prelabel.py`：预标。
- `scripts/ch10/validate_golden.py`：黄金样例闸。
- `scripts/ch10/synthetic.py`：模拟补足、冻结、run-id。
- `scripts/ch10/audit.py`：抽审导出与纠正应用。
- `scripts/ch10/build_dataset.py`：合并、近重复、分层切分、manifest。
- `scripts/ch10/augment_train.py`：只扩训练集。
- `scripts/ch10/train_lib.py` / `train.py`：纯函数 + Trainer。
- `scripts/ch10/eval_lib.py` / `evaluate.py` / `scan_threshold_replay.py`：评测、报告、阈值重演。
- `scripts/ch10/export_onnx.py`：ONNX 导出与一致性校验。
- `scripts/ch10/inference_lib.py` / `serve.py`：轻运行时推理服务。
- `scripts/ch10/classify_lib.py` / `classify_pool.py`：旁路批量归类。

**产物**
- `data/ch10/corpus_raw.jsonl`、`corpus_clean.jsonl`、`corpus_real_labeled.jsonl`、`corpus_labeled.jsonl`、`generated/<run-id>/synthetic.jsonl`、`dataset/{train,val,test}.jsonl`、`dataset/train_augmented.jsonl`、`golden_samples.jsonl`、`audit_corrections.json`、`sample_review.md`。
- `data/ch10/model/`、`data/ch10/onnx/`、`data/ch10/reports/*.json|md`。
- `log/acceptance/<job>.log`、`log/classifier.log`、`log/classifier.pid`。

---

### Task 1: 共享类目表与路径

**Files:**
- Create: `app/core/__init__.py`
- Create: `app/core/taxonomy.py`
- Create: `app/core/paths.py`
- Test: `tests/test_ch10_taxonomy.py`

**Interfaces:**
- Produces: `TOPIC_CLASSES: tuple[str, ...]`、`TOPIC_BOUNDARIES: dict[str, str]`、`TOPIC_EXAMPLES: dict[str, tuple[str, ...]]`、`LABEL2ID: dict[str,int]`、`ID2LABEL: dict[int,str]`、`FALLBACK_LABEL: str`、`TAXONOMY_VERSION: str`、`normalize_labels(labels: Iterable[str]) -> list[str]`、`labels_from_scores(scores: Mapping[str,float], threshold: float) -> list[str]`。
- Produces: `ROOT: Path`、`DATA_DIR`、`REPORT_DIR`、`MODEL_DIR`、`ONNX_DIR`、`GENERATED_DIR`、`GOLDEN_PATH`、`CORRECTIONS_PATH`、`LOG_DIR`。

- [ ] **Step 1: Write the failing tests**

```python
import pytest
from app.core import paths, taxonomy


def test_taxonomy_has_exactly_17_unique_classes_in_order():
    assert taxonomy.TOPIC_CLASSES == (
        "退换货", "物流", "尺码", "发票", "质量问题", "运费", "优惠活动", "价保",
        "支付", "订单修改", "库存补货", "商品信息", "保修维修", "账号",
        "会员积分", "评价", "其他",
    )
    assert len(set(taxonomy.TOPIC_CLASSES)) == 17


def test_boundaries_and_examples_cover_every_class():
    assert set(taxonomy.TOPIC_BOUNDARIES) == set(taxonomy.TOPIC_CLASSES)
    assert set(taxonomy.TOPIC_EXAMPLES) == set(taxonomy.TOPIC_CLASSES)
    assert all(taxonomy.TOPIC_BOUNDARIES[label].strip() for label in taxonomy.TOPIC_CLASSES)
    assert all(taxonomy.TOPIC_EXAMPLES[label] for label in taxonomy.TOPIC_CLASSES)


def test_normalize_rejects_unknown_and_orders_by_taxonomy():
    assert taxonomy.normalize_labels(["退换货", "尺码", "退换货"]) == ["尺码", "退换货"]
    with pytest.raises(ValueError, match="unknown topic label"):
        taxonomy.normalize_labels(["售后"])


def test_labels_from_scores_threshold_and_fallback():
    scores = {label: 0.1 for label in taxonomy.TOPIC_CLASSES}
    scores["尺码"] = 0.91
    scores["退换货"] = 0.88
    scores["其他"] = 0.95
    assert taxonomy.labels_from_scores(scores, 0.5) == ["尺码", "退换货"]

    low = {label: 0.2 for label in taxonomy.TOPIC_CLASSES}
    low["物流"] = 0.49
    assert taxonomy.labels_from_scores(low, 0.5) == ["物流"]


def test_paths_are_under_repo_root():
    assert paths.DATA_DIR == paths.ROOT / "data" / "ch10"
    assert paths.REPORT_DIR == paths.DATA_DIR / "reports"
    assert paths.GOLDEN_PATH == paths.DATA_DIR / "golden_samples.jsonl"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ch10_taxonomy.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.core'`.

- [ ] **Step 3: Implement the modules**

Implement `app/core/taxonomy.py` with the 17 names, boundary text and examples copied from spec §5; `LABEL2ID/ID2LABEL` built from the tuple; `normalize_labels` dedupes, rejects unknown with `ValueError("unknown topic label: ...")`, and sorts by `LABEL2ID`. `labels_from_scores` selects `score >= threshold`, drops `其他` when any specific label is selected, and if the selection is empty returns `[max(scores, key=scores.get)]`. `TAXONOMY_VERSION = hashlib.sha1("|".join(TOPIC_CLASSES).encode()).hexdigest()[:8]`.

Implement `app/core/paths.py` with `ROOT = Path(__file__).resolve().parents[2]` and the constants listed in Interfaces.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ch10_taxonomy.py -v`
Expected: PASS.

- [ ] **Step 5: Update `dev-notes/ch10.md` and commit**

Append a `## 阶段：Task 1 完成` section with the four required lines.
```bash
git add app/core tests/test_ch10_taxonomy.py dev-notes/ch10.md
git commit -m "feat: add ch10 taxonomy and shared paths"
```

---

### Task 2: ML 环境与 Make 变量

**Files:**
- Create: `requirements-ch10.txt`
- Modify: `Makefile` (add `ML_PYTHON`, `ch10-env`, all ch10 targets, phony)
- Modify: `.gitignore` (ignore `.venv-ml/`, `data/ch10/model/`, `data/ch10/onnx/`, `data/ch10/dataset/`, `data/ch10/generated/`, `data/ch10/corpus_*.jsonl`, `data/ch10/sample_review.md`, `log/classifier.*`)

**Interfaces:**
- Produces: `.venv-ml/Scripts/python.exe` with importable `torch`、`transformers`、`sklearn`、`onnx`、`onnxruntime`、`tokenizers`、`numpy`。
- Produces: `make ch10-env` 可重复执行。

- [ ] **Step 1: Check current library versions with Context7**

Use `mcp__context7__resolve-library-id` for `PyTorch`, `Transformers`, `ONNX Runtime`, then `query-docs` for Python 3.12 installation / runtime compatibility. Record chosen versions in the commit body.

- [ ] **Step 2: Create `requirements-ch10.txt`**

First line `-r requirements.txt`, then pinned ML packages. Include `fastapi`, `uvicorn`, `httpx` only if not already in main requirements (they are, so no duplicates).

- [ ] **Step 3: Add Makefile variables and all ch10 targets**

Add `ML_PYTHON ?= .venv-ml/Scripts/python.exe` and `CH10_RUN = PYTHONPATH=. $(ML_PYTHON) -m`. Add `ch10-env` (uv venv + uv pip install), plus these recipes (they may point at scripts that do not exist yet; that is expected until later tasks):
```make
ch10-env:
	uv venv --python 3.12 .venv-ml
	uv pip install --python .venv-ml -r requirements-ch10.txt
ch10-golden:
	$(CH10_RUN) scripts.ch10.validate_golden --golden data/ch10/golden_samples.jsonl --out data/ch10/reports/golden_report.json
ch10-corpus:
	$(CH10_RUN) scripts.ch10.build_corpus --stage all
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
```
Add all names to `.PHONY`. Do not duplicate the existing `test`/ch09 targets.

- [ ] **Step 4: Create the env and verify imports**

Run:
```bash
make ch10-env
.venv-ml/Scripts/python.exe -c "import torch, transformers, sklearn, onnx, onnxruntime, tokenizers, numpy; print(torch.__version__, transformers.__version__, onnxruntime.__version__)"
```
Expected: prints three version strings; exit 0.

- [ ] **Step 5: Run the existing suite to prove no pollution**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: same pass count as ch09 baseline (994 passed, 1 warning), no import errors.

- [ ] **Step 6: Update `dev-notes/ch10.md` and commit**

```bash
git add requirements-ch10.txt Makefile .gitignore dev-notes/ch10.md
git commit -m "chore: add isolated ch10 ML environment"
```

---

### Task 3: DDL、ORM 与幂等建表

**Files:**
- Create: `sql/ch10-ddl.sql`
- Create: `app/models/topic.py`
- Modify: `app/models/__init__.py`
- Create: `app/db/init_ch10.py`
- Test: `tests/test_ch10_schema.py`

**Interfaces:**
- Produces: `TopicClassification` ORM（`id`、`question_id`、`labels`、`classified_at`；`question_id` 唯一）。
- Produces: `apply_ch10(engine: Engine) -> None`，幂等。
- Consumes: `app.db.sql_utils.load_statements`。

- [ ] **Step 1: Write the failing tests**

```python
from pathlib import Path
import pytest
from app.models import LowConfidenceQuestion, TopicClassification


def test_ch10_ddl_is_verbatim_user_ddl():
    sql = Path("sql/ch10-ddl.sql").read_text(encoding="utf-8")
    assert "SET NAMES utf8mb4;" in sql
    assert "CREATE TABLE topic_classifications" in sql
    assert "BIGINT UNSIGNED" in sql
    assert "labels        JSON            NOT NULL" in sql
    assert "UNIQUE KEY uk_question_id (question_id)" in sql
    assert "CONSTRAINT fk_topic_question FOREIGN KEY (question_id)" in sql
    assert "COMMENT='主题分类结果'" in sql


def test_orm_roundtrip_and_unique_question(session):
    q = LowConfidenceQuestion(raw_question="买大了想退", source="self_check")
    session.add(q); session.commit(); session.refresh(q)
    row = TopicClassification(question_id=q.id, labels=["尺码", "退换货"])
    session.add(row); session.commit(); session.refresh(row)
    assert row.id > 0 and row.labels == ["尺码", "退换货"] and row.classified_at
    session.add(TopicClassification(question_id=q.id, labels=["物流"]))
    with pytest.raises(Exception):
        session.commit()


def test_apply_ch10_creates_then_noops(monkeypatch):
    import app.db.init_ch10 as init_ch10
    # Fake conn/engine pattern copied from tests/test_ch09_schema.py.
    monkeypatch.setattr(init_ch10, "_existing_table", lambda conn: False)
    conn = _FakeConn()
    init_ch10.apply_ch10(_FakeEngine(conn))
    assert len(conn.driver_executed) == len(init_ch10._load_statements())

    monkeypatch.setattr(init_ch10, "_existing_table", lambda conn: True)
    conn2 = _FakeConn()
    init_ch10.apply_ch10(_FakeEngine(conn2))
    assert conn2.executed == []
```

Use the `_FakeConn`/`_FakeEngine` helpers from `tests/test_ch09_schema.py` (copy them into this file).

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ch10_schema.py -v`
Expected: FAIL with missing `sql/ch10-ddl.sql` / `TopicClassification`.

- [ ] **Step 3: Implement**

Write `sql/ch10-ddl.sql` exactly as spec §10.1. Add `TopicClassification` with `BigInteger().with_variant(Integer, "sqlite")`, `ForeignKey("low_confidence_questions.id")`, `JSON` labels, `server_default=func.now()` for `classified_at`, and `UniqueConstraint("question_id", name="uk_question_id")`. Export it in `app/models/__init__.py`.

`app/db/init_ch10.py`: `SQL_PATH`/`TABLE`, `_load_statements()` via `load_statements`, `_existing_table(conn)` via `information_schema.tables`, `apply_ch10(engine)` no-op if present else `conn.exec_driver_sql` every statement; `main()`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ch10_schema.py -v`
Expected: PASS.

- [ ] **Step 5: Apply to the live MySQL DB**

Run: `.venv/Scripts/python.exe -m app.db.init_ch10`
Expected: `ch10 topic classification schema created...`; rerun prints already initialized.

- [ ] **Step 6: Update `dev-notes/ch10.md` and commit**

```bash
git add sql/ch10-ddl.sql app/models/topic.py app/models/__init__.py app/db/init_ch10.py tests/test_ch10_schema.py dev-notes/ch10.md
git commit -m "feat: add topic_classifications schema and ORM"
```

---

### Task 4: 语料工具库（TDD）

**Files:**
- Create: `scripts/__init__.py`
- Create: `scripts/ch10/__init__.py`
- Create: `scripts/ch10/corpus_lib.py`
- Test: `tests/test_ch10_corpus_lib.py`

**Interfaces:**
- Produces: `Sample` dataclass（`id: str`、`text: str`、`labels: tuple[str,...] = ()`、`origin: str = "real"`、`source_refs: tuple[dict,...] = ()`、`meta: dict = field(default_factory=dict)`）。
- Produces: `stable_id(origin: str, text: str) -> str`、`normalize_text(text: str) -> str`、`redact_pii(text: str) -> tuple[str, list[str]]`、`dedupe_exact(samples) -> list[Sample]`、`jaccard_3gram(a: str, b: str) -> float`、`dedupe_near(samples, threshold=0.85) -> tuple[list[Sample], int]`、`read_jsonl(path)`、`write_jsonl(path, rows)`、`label_counts(samples) -> dict[str,int]`、`apply_corrections(samples, corrections: dict[str, list[str]]) -> list[Sample]`。

- [ ] **Step 1: Write the failing tests**

```python
from scripts.ch10.corpus_lib import (
    Sample, apply_corrections, dedupe_exact, dedupe_near,
    jaccard_3gram, redact_pii, stable_id,
)


def test_redact_phone_order_and_social_ids():
    text, flags = redact_pii("手机号 13800000000，订单 E20241111001，微信 wx_abc123")
    assert "13800000000" not in text and "E20241111001" not in text and "wx_abc123" not in text
    assert "[手机号]" in text and "[订单号]" in text and "[微信号]" in text
    assert set(flags) >= {"phone", "order", "wechat"}
    # product model numbers are not order ids
    clean, flags2 = redact_pii("MH-LP100 废砂盒还有吗")
    assert clean == "MH-LP100 废砂盒还有吗" and flags2 == []


def test_stable_id_is_content_addressed_and_origin_scoped():
    assert stable_id("real", "买大了想退") == stable_id("real", "买大了想退")
    assert stable_id("real", "买大了想退") != stable_id("synthetic", "买大了想退")


def test_dedupe_exact_merges_source_refs():
    a = Sample(id="a", text="开发票", origin="real", source_refs=({"question_id": 1},))
    b = Sample(id="b", text="开发票", origin="real", source_refs=({"question_id": 2},))
    out = dedupe_exact([a, b])
    assert len(out) == 1 and out[0].source_refs == ({"question_id": 1}, {"question_id": 2})


def test_jaccard_and_near_dedupe():
    assert jaccard_3gram("买大了想退", "买大了想退货") > 0.5
    a = Sample(id="a", text="买大了想退货", origin="synthetic")
    b = Sample(id="b", text="买大了想退货", origin="synthetic")
    out, merged = dedupe_near([a, b])
    assert len(out) == 1 and merged == 1


def test_apply_corrections_unknown_id_raises():
    s = Sample(id="x", text="退活", origin="real")
    out = apply_corrections([s], {"x": ["退换货"]})
    assert out[0].labels == ("退换货",) and out[0].meta["audited"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ch10_corpus_lib.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts'`.

- [ ] **Step 3: Implement**

Regexes: Chinese mobile `(?<!\d)1[3-9]\d{9}(?!\d)`; order `\b[A-Za-z]{1,4}\d{8,}\b`; WeChat/QQ only when followed by an identifier (`(?:微信|WeChat|wechat|wx|QQ|qq)\s*[:：]?\s*[A-Za-z][A-Za-z0-9_-]{5,19}`). Replace with the placeholders and return flag names. `normalize_text` strips, collapses whitespace, normalizes full-width punctuation to ASCII only where harmless. `stable_id` = `sha1(f"{origin}|{text}".encode()).hexdigest()[:12]`. `dedupe_exact` keys on `normalize_text(text)`. `jaccard_3gram` uses character 3-grams of normalized text. `dedupe_near` greedily keeps the first, merges `source_refs`, returns `(kept, merged_count)`. JSONL helpers use UTF-8 and `ensure_ascii=False`. `apply_corrections` validates ids and labels via `normalize_labels`, raises `KeyError` for unknown id.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ch10_corpus_lib.py -v`
Expected: PASS.

- [ ] **Step 5: Update `dev-notes/ch10.md` and commit**

```bash
git add scripts/__init__.py scripts/ch10 tests/test_ch10_corpus_lib.py dev-notes/ch10.md
git commit -m "feat: add ch10 corpus utilities"
```

---

### Task 5: Prompt、Schema 与预标（TDD，fake LLM）

**Files:**
- Create: `scripts/ch10/prompts.py`
- Create: `scripts/ch10/schemas.py`
- Create: `scripts/ch10/llm_io.py`
- Create: `scripts/ch10/prelabel.py`
- Test: `tests/test_ch10_prelabel.py`

**Interfaces:**
- Produces: `PRELABEL_SYSTEM`、`CLEAN_SYSTEM`、`SYNTHETIC_SYSTEM`、`AUGMENT_SYSTEM`、`render_taxonomy() -> str`。
- Produces: `PrelabelItem`/`PrelabelBatch`、`CleanItem`/`CleanBatch`、`SyntheticItem`/`SyntheticBatch`、`AugmentItem`/`AugmentBatch`（pydantic）。
- Produces: `BudgetExceeded`、`CallBudget(max_calls: int | None)`、`invoke_structured(llm, system, human, schema, budget) -> dict`。
- Produces: `prelabel_samples(llm, samples, budget, batch_size=10) -> list[Sample]`、`main(argv)`。

- [ ] **Step 1: Write the failing tests**

```python
import pytest
from app.core.taxonomy import TOPIC_CLASSES
from scripts.ch10.prompts import PRELABEL_SYSTEM, render_taxonomy
from scripts.ch10.schemas import PrelabelBatch, SyntheticBatch
from scripts.ch10.llm_io import BudgetExceeded, CallBudget
from scripts.ch10.prelabel import prelabel_samples
from scripts.ch10.corpus_lib import Sample


def test_taxonomy_rendered_with_boundaries_and_literal_rule():
    text = render_taxonomy()
    for label in TOPIC_CLASSES:
        assert label in text
    assert "字面" in PRELABEL_SYSTEM and "其他" in PRELABEL_SYSTEM


def test_schema_rejects_unknown_label_and_dedupes_via_normalize():
    with pytest.raises(ValueError):
        PrelabelBatch(items=[{"id": "x", "labels": ["售后"]}])


def test_budget_raises_after_limit():
    budget = CallBudget(1)
    budget.spend()
    with pytest.raises(BudgetExceeded):
        budget.spend()


class FakeStructured:
    def __init__(self, outputs): self.outputs = outputs; self.calls = 0
    def invoke(self, messages):
        out = self.outputs[self.calls]; self.calls += 1; return out


class FakeLLM:
    def __init__(self, outputs): self.outputs = outputs
    def with_structured_output(self, schema, method="json_mode"):
        return FakeStructured(self.outputs)


def test_prelabel_batches_preserve_ids_and_normalize_order():
    samples = [Sample(id="a", text="买大了想退", origin="real"),
               Sample(id="b", text="开发票", origin="real")]
    llm = FakeLLM([{"items": [{"id": "a", "labels": ["退换货", "尺码"]},
                              {"id": "b", "labels": ["发票"]}]}])
    out = prelabel_samples(llm, samples, CallBudget(5), batch_size=10)
    assert [s.id for s in out] == ["a", "b"]
    assert out[0].labels == ("尺码", "退换货")
    assert out[1].labels == ("发票",)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ch10_prelabel.py -v`
Expected: FAIL with missing modules.

- [ ] **Step 3: Implement**

`prompts.py`: `render_taxonomy()` emits a 17-row table with `#`、类目、边界、示例 from `app.core.taxonomy`; `PRELABEL_SYSTEM` includes the literal rule、近邻边界（修/退、运费/物流、价保/优惠）、「其他」互斥兜底, and instructs JSON `{"items":[{"id","labels"}]}`. `CLEAN_SYSTEM` says only fix typos/format, never semantics or tone. `SYNTHETIC_SYSTEM` asks for questions with exact labels and the 15%/10%/15% style mix. `AUGMENT_SYSTEM` asks for one paraphrase preserving the exact label set.

`schemas.py`: pydantic models; `field_validator` on `labels` calls `normalize_labels`, so unknown labels raise `ValueError`.

`llm_io.py`: `CallBudget.spend()` raises `BudgetExceeded` when `max_calls` reached; `invoke_structured` calls `llm.with_structured_output(schema, method="json_mode")`, spends one call, and returns a dict (accept pydantic or dict like `app/services/flywheel.py::_coerce_output`).

`prelabel.py`: chunk samples, build a human message with `id\ttext` lines, invoke, map results by id, reject missing/extra ids, preserve source sample order, set `origin` and `source_refs`. CLI reads `corpus_clean.jsonl`, writes the real-only intermediate `corpus_real_labeled.jsonl`, `--max-llm-calls`, `--batch-size`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ch10_prelabel.py -v`
Expected: PASS.

- [ ] **Step 5: Update `dev-notes/ch10.md` and commit**

```bash
git add scripts/ch10/prompts.py scripts/ch10/schemas.py scripts/ch10/llm_io.py scripts/ch10/prelabel.py tests/test_ch10_prelabel.py dev-notes/ch10.md
git commit -m "feat: add ch10 prelabel prompt and pipeline"
```

---

### Task 6: 黄金样例闸（验证跑，替代 TDD）

**Files:**
- Create: `data/ch10/golden_samples.jsonl`（30 条，进 git）
- Create: `scripts/ch10/validate_golden.py`
- Test: `tests/test_ch10_golden.py`（纯比较逻辑，用 fake prelabel）
- Output: `data/ch10/reports/golden_report.json`

**Interfaces:**
- Produces: `compare_labels(golden: list[dict], predicted: dict[str, list[str]]) -> dict`（`pass_rate`、`passed`、`total`、`mismatches`）。
- Produces: `main(argv) -> int`，pass_rate < 0.8 时退出码 1。

- [ ] **Step 1: Write the 30 golden samples for user review**

Draft `golden_samples.jsonl` with `{"id","text","labels","note"}` covering: 4 个近邻对（退换货/保修维修、运费/物流、价保/优惠活动、尺码/退换货）、5 个多诉求句、4 个方言句、3 个错别字句、4 个「其他」互斥句、4 个单类目边界句、2 个账号/会员积分句。必须包含：
- `买大了想退` → `["尺码","退换货"]`
- `开胶了还能换吗` → `["质量问题","退换货"]`
- `退货运费谁出` → `["运费"]`
- `这鞋买大了一码` → `["尺码"]`
- `刚买就降价了能补差价吗` → `["价保"]`
Pause and let the user review/correct this file before Step 2.

- [ ] **Step 2: Write the failing comparison test**

```python
from scripts.ch10.validate_golden import compare_labels


def test_compare_labels_exact_set_match_and_rate():
    golden = [{"id": "a", "text": "买大了想退", "labels": ["尺码", "退换货"]},
              {"id": "b", "text": "开发票", "labels": ["发票"]}]
    predicted = {"a": ["退换货", "尺码"], "b": ["支付"]}
    out = compare_labels(golden, predicted)
    assert out["pass_rate"] == 0.5
    assert out["mismatches"] == [{"id": "b", "gold": ["发票"], "predicted": ["支付"]}]
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ch10_golden.py -v`
Expected: FAIL with missing module.

- [ ] **Step 4: Implement `compare_labels` and the CLI**

`compare_labels` compares label sets order-insensitively (`set`), returns `pass_rate = passed/total` (0 if empty), and mismatches. CLI loads golden, calls `prelabel_samples` with the real LLM from `build_llm(get_settings(), temperature=0.0)`, writes `golden_report.json` with `pass_rate`、`passed`、`total`、`mismatches`、`model`、`generated_at`, and returns 1 below `--min-pass-rate 0.8`.

- [ ] **Step 5: Run the live golden gate**

Run: `PYTHONPATH=. .venv-ml/Scripts/python.exe -m scripts.ch10.validate_golden --golden data/ch10/golden_samples.jsonl --out data/ch10/reports/golden_report.json`
Expected: `pass_rate >= 0.8`; if not, revise `PRELABEL_SYSTEM` using the mismatches and rerun. Do not proceed to bulk prelabel until it passes.

- [ ] **Step 6: User reviews mismatches, update notes, commit**

Show the mismatch list to the user; apply corrections to `golden_samples.jsonl` or the prompt. Append the dev-notes entry with the pass rate and what changed.
```bash
git add data/ch10/golden_samples.jsonl scripts/ch10/validate_golden.py tests/test_ch10_golden.py dev-notes/ch10.md
git commit -m "feat: add ch10 golden-sample gate"
```

---

### Task 7: 真实语料捞取与清洗（TDD + 验证跑）

**Files:**
- Create: `scripts/ch10/build_corpus.py`
- Test: `tests/test_ch10_build_corpus.py`
- Output: `data/ch10/corpus_raw.jsonl`、`corpus_clean.jsonl`

**Interfaces:**
- Produces: `fetch_real_questions(session) -> list[dict]`（只取 `matched_review_id IS NOT NULL`；字段 `question_id`、`review_id`、`raw_question`、`normalized_question`、`source`、`created_at`）。
- Produces: `choose_text(row) -> tuple[str, bool]`（优先 `normalized_question`，空则退回 `raw_question`，第二项表示 `normalized`）。
- Produces: `build_real_samples(rows) -> list[Sample]`（脱敏、`dedupe_exact`、稳定 id、`source_refs`）。
- Produces: `clean_with_llm(llm, samples, budget) -> tuple[list[Sample], list[dict]]`（before/after 日志）。
- Produces: `main(argv) -> int`。

- [ ] **Step 1: Write the failing tests**

```python
from scripts.ch10.build_corpus import build_real_samples, choose_text, fetch_real_questions


def test_fetch_only_matched_and_join_normalized(session):
    from app.models import LowConfidenceQuestion, ReviewQueue
    r = ReviewQueue(normalized_question="支持以旧换新吗"); session.add(r); session.commit(); session.refresh(r)
    matched = LowConfidenceQuestion(raw_question="支持以旧换新不", source="self_check", matched_review_id=r.id)
    unmatched = LowConfidenceQuestion(raw_question="没人管", source="self_check")
    session.add_all([matched, unmatched]); session.commit()
    rows = fetch_real_questions(session)
    assert [row["question_id"] for row in rows] == [matched.id]
    assert rows[0]["normalized_question"] == "支持以旧换新吗"


def test_choose_text_prefers_normalized_and_falls_back():
    assert choose_text({"normalized_question": "开发票", "raw_question": "能开票不"}) == ("开发票", True)
    assert choose_text({"normalized_question": None, "raw_question": "能开票不"}) == ("能开票不", False)


def test_build_real_samples_redacts_dedupes_and_keeps_refs():
    rows = [
        {"question_id": 1, "review_id": 1, "normalized_question": "手机号13800000000能改吗", "raw_question": "x", "source": "self_check", "created_at": None},
        {"question_id": 2, "review_id": 1, "normalized_question": "手机号13800000000能改吗", "raw_question": "y", "source": "user_feedback", "created_at": None},
    ]
    out = build_real_samples(rows)
    assert len(out) == 1
    assert "13800000000" not in out[0].text
    assert out[0].source_refs == ({"question_id": 1, "review_id": 1}, {"question_id": 2, "review_id": 1})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ch10_build_corpus.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement fetch/clean and the CLI**

`fetch_real_questions` joins `ReviewQueue` on `matched_review_id`, orders by `id`. `build_real_samples` calls `choose_text`, `redact_pii`, dedupes by normalized text, and sets `origin="real"`. `clean_with_llm` batches 10 samples, asks for `{"items":[{"id","text"}]}`, validates ids and non-empty text, keeps a per-sample `{"id","before","after"}` change log; on a failed batch, leave those samples unchanged and record the error (do not silently drop). CLI stages: `--stage fetch|clean|all`, `--max-llm-calls`, writes `corpus_raw.jsonl` and `corpus_clean.jsonl`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ch10_build_corpus.py -v`
Expected: PASS.

- [ ] **Step 5: Run the live fetch/clean**

Run: `PYTHONPATH=. .venv-ml/Scripts/python.exe -m scripts.ch10.build_corpus --stage all`
Expected: `corpus_raw.jsonl` ~12 real normalized questions; `corpus_clean.jsonl` same count; before/after log present. Show the user a sample of the changes.

- [ ] **Step 6: Update `dev-notes/ch10.md` and commit**

```bash
git add scripts/ch10/build_corpus.py tests/test_ch10_build_corpus.py dev-notes/ch10.md
git commit -m "feat: fetch and clean ch10 real questions"
```

---

### Task 8: 模拟语料补足、冻结与抽审（验证跑）

**Files:**
- Create: `scripts/ch10/synthetic.py`
- Create: `scripts/ch10/audit.py`
- Test: `tests/test_ch10_synthetic.py`、`tests/test_ch10_audit.py`（fake LLM/纯逻辑）
- Output: `data/ch10/generated/<run-id>/synthetic.jsonl`、`data/ch10/sample_review.md`、`audit_corrections.json`

**Interfaces:**
- Produces: `target_gap(label_counts: dict[str,int], target=100) -> dict[str,int]`。
- Produces: `run_id_for(params: dict, *, force: bool = False) -> str`、`load_or_generate(llm, gap, out_dir, *, ratios, budget, force=False) -> list[Sample]`。
- Produces: `validate_synthetic(samples) -> list[Sample]`（17 类子集、无重复、正例计数）。
- Produces: `cross_check_labels(samples, prelabel_fn) -> dict`（生成标签与盲标标签的一致率 + 不一致样本）。
- Produces: `merge_labeled(real_labeled: list[Sample], synthetic: list[Sample], corrections: dict[str, list[str]]) -> list[Sample]`（应用纠正、去重、写 `corpus_labeled.jsonl` 前的内存结果）。
- Produces: `export_review(samples, corrections, path) -> str`（全量 real + 每类 3 条 synthetic + 预标/生成不一致项）。
- Produces: `apply_corrections`（复用 `corpus_lib`）。

- [ ] **Step 1: Write the failing tests**

```python
from scripts.ch10.synthetic import target_gap, run_id_for, validate_synthetic
from scripts.ch10.audit import review_rows


def test_target_gap_fills_missing_only():
    assert target_gap({"退换货": 88, "物流": 3}, target=100)["退换货"] == 12
    assert target_gap({"退换货": 120}, target=100)["退换货"] == 0


def test_run_id_is_stable_and_force_changes_it():
    p = {"target": 100, "multi": 0.15}
    assert run_id_for(p) == run_id_for(p)
    assert run_id_for(p, force=True) != run_id_for(p)


def test_validate_synthetic_rejects_unknown_and_empty():
    from scripts.ch10.corpus_lib import Sample
    good = Sample(id="a", text="买大了想退", labels=("尺码", "退换货"), origin="synthetic")
    bad = Sample(id="b", text="售后", labels=("售后",), origin="synthetic")
    assert validate_synthetic([good, bad]) == [good]


def test_review_rows_include_all_real_and_three_per_class():
    rows = review_rows(real=[], synthetic=[])
    assert rows == []  # empty input is safe; with data, counts are asserted in Task 21


def test_cross_check_flags_label_disagreement():
    from scripts.ch10.corpus_lib import Sample
    from scripts.ch10.synthetic import cross_check_labels
    samples = [Sample(id="a", text="买大了想退", labels=("尺码", "退换货"), origin="synthetic")]
    report = cross_check_labels(samples, lambda text: ["尺码"])
    assert report["agreement_rate"] == 0.0
    assert report["disagreements"][0]["id"] == "a"


def test_merge_labeled_applies_corrections_and_dedupes():
    from scripts.ch10.corpus_lib import Sample
    from scripts.ch10.synthetic import merge_labeled
    real = [Sample(id="r1", text="买大了想退", labels=("尺码",), origin="real")]
    merged = merge_labeled(real, [], {"r1": ["尺码", "退换货"]})
    assert merged[0].labels == ("尺码", "退换货") and merged[0].meta["audited"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ch10_synthetic.py tests/test_ch10_audit.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

`synthetic.py`: `target_gap` computes `max(0, target - count)`. `run_id_for` hashes `taxonomy.TAXONOMY_VERSION` + sorted params; `force=True` appends a UTC timestamp. `load_or_generate` reuses `generated/<run-id>/synthetic.jsonl` if present; otherwise generates in batches of 10, asks for `{"items":[{"text","labels"}]}`, injects the dialect/typo/multi ratios through the prompt, validates labels, assigns `stable_id("synthetic", text)`, stops when each gap is filled. `cross_check_labels` runs the prelabel prompt on every synthetic text, compares label sets, returns `agreement_rate` and the disagreements. `merge_labeled` applies `audit_corrections.json`, sets `audited=True` on corrected samples, dedupes exact text across real/synthetic, and preserves `source_refs`. On `--force`, archive old `audit_corrections.json` to `audit_corrections.<old-run>.json` and print how many corrections will be invalidated. Enforce `CallBudget`.

`audit.py`: `review_rows(real, synthetic)` returns rows for `sample_review.md`; include every distinct real sample and 3 synthetic per class (dedup by text), plus any sample whose generation labels disagree with the blind prelabel. `export_review` renders a Markdown table with `id`、`text`、`labels`、`origin`、`note`. Applying corrections sets `audited=True` and writes the corrected labels.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ch10_synthetic.py tests/test_ch10_audit.py -v`
Expected: PASS.

- [ ] **Step 5: Run blind prelabel + generation live**

Run the prelabel on `corpus_clean.jsonl` (writes `corpus_real_labeled.jsonl`), then generation with target 100/class. Expected: per-class positive counts ~100, total label assignments ~1700, dialect/typo/multi ratios within ±3%, no unknown labels. Run `cross_check_labels` on the synthetic set; record the agreement rate and the disagreement list. Merge real labeled + synthetic + corrections into `corpus_labeled.jsonl`. Then export `sample_review.md`.

- [ ] **Step 6: User audit checkpoint**

Show `sample_review.md` to the user. Apply their corrections to `audit_corrections.json`; rerun the merge so `corpus_labeled.jsonl` reflects them. Update dev-notes with the audit agreement rate and corrections count. Commit:
```bash
git add scripts/ch10/synthetic.py scripts/ch10/audit.py tests/test_ch10_synthetic.py tests/test_ch10_audit.py dev-notes/ch10.md
git commit -m "feat: generate and audit ch10 synthetic corpus"
```

---

### Task 9: 分层划分与泄漏闸（TDD）

**Files:**
- Create: `scripts/ch10/build_dataset.py`
- Test: `tests/test_ch10_split.py`
- Output: `data/ch10/dataset/{train,val,test}.jsonl`、`dataset/manifest.json`

**Interfaces:**
- Produces: `split_samples(samples, seed=42) -> dict[str, list[Sample]]`（keys `train`/`val`/`test`）。
- Produces: `check_no_leak(splits) -> dict`（跨集 exact 重叠数）。
- Produces: `class_coverage(splits) -> dict`（每类三份是否都有）。
- Produces: `write_dataset(splits, out_dir, manifest_extra) -> dict`。

- [ ] **Step 1: Write the failing tests**

```python
from scripts.ch10.build_dataset import check_no_leak, class_coverage, split_samples
from scripts.ch10.corpus_lib import Sample


def _sample(i, labels=("尺码",)):
    return Sample(id=f"s{i}", text=f"问题{i}", labels=labels, origin="synthetic")


def test_split_is_deterministic_and_roughly_80_10_10():
    data = [_sample(i, ("尺码",)) for i in range(100)]
    a = split_samples(data, seed=42)
    b = split_samples(data, seed=42)
    assert [s.id for s in a["train"]] == [s.id for s in b["train"]]
    assert abs(len(a["train"]) / len(data) - 0.8) <= 0.03


def test_no_leak_and_every_class_in_all_splits():
    data = [_sample(i, ("尺码",)) for i in range(60)] + [_sample(i + 60, ("物流",)) for i in range(60)]
    splits = split_samples(data, seed=42)
    assert check_no_leak(splits) == {"train_val": 0, "train_test": 0, "val_test": 0}
    assert all(class_coverage(splits)[label] == [True, True, True] for label in ("尺码", "物流"))


def test_rare_combo_uses_first_label_layer():
    data = [_sample(i, ("尺码", "退换货")) for i in range(12)] + [_sample(i + 12, ("尺码",)) for i in range(12)]
    splits = split_samples(data, seed=42)
    assert len(splits["val"]) + len(splits["test"]) > 0


def test_near_duplicates_are_merged_before_split():
    data = [Sample(id="a", text="买大了想退货", labels=("尺码", "退换货"), origin="synthetic"),
            Sample(id="b", text="买大了想退货", labels=("尺码", "退换货"), origin="synthetic")]
    splits = split_samples(data, seed=42)
    assert sum(len(v) for v in splits.values()) == 1
    assert check_no_leak(splits) == {"train_val": 0, "train_test": 0, "val_test": 0}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ch10_split.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

Merge near-duplicates first (call `dedupe_near` before split). Group by label-combination key; if a group has `<10` samples, use the first label in `TOPIC_CLASSES` order as the key. Shuffle each group with `random.Random(seed)`. Allocate `n_train=floor(n*0.8)`, `n_val=floor(n*0.1)`, `n_test=n-n_train-n_val`; groups with `n<3` go to train (record them in `manifest["undersized_groups"]`); groups with `n>=3` get at least one val and one test by moving from train. After allocation, for any class missing from val or test, move one train sample containing it (prefer one whose other labels are already covered); if no train sample can be moved without emptying train, leave the coverage gap and record it rather than crashing. Assert exact-text overlap is zero. Manifest fields: `counts`、`origin_counts`、`per_class`（train/val/test counts and ratios）、`coverage`、`undersized_groups`、`leak`、`near_duplicates_merged`、`seed`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ch10_split.py -v`
Expected: PASS.

- [ ] **Step 5: Run the live split**

Run: `PYTHONPATH=. .venv-ml/Scripts/python.exe -m scripts.ch10.build_dataset --labeled data/ch10/corpus_labeled.jsonl --out data/ch10/dataset`
Expected: `manifest.json` shows leak counts all 0, every class present in all three splits, ratios within ±5%; exit 0.

- [ ] **Step 6: Update `dev-notes/ch10.md` and commit**

```bash
git add scripts/ch10/build_dataset.py tests/test_ch10_split.py dev-notes/ch10.md
git commit -m "feat: split ch10 dataset with leak gate"
```

---

### Task 10: 训练集增强（验证跑）

**Files:**
- Create: `scripts/ch10/augment_train.py`
- Test: `tests/test_ch10_augment.py`
- Output: `data/ch10/dataset/train_augmented.jsonl`

**Interfaces:**
- Produces: `accept_variant(original: Sample, variant_text: str, predicted_labels: list[str]) -> bool`（标签集合完全一致才接受）。
- Produces: `augment_train(llm, train, budget, prelabel_fn) -> tuple[list[Sample], dict]`（`kept`、`dropped`、`drop_rate`、`failures`）。
- Produces: CLI.

- [ ] **Step 1: Write the failing tests**

```python
from scripts.ch10.augment_train import accept_variant, augment_train
from scripts.ch10.corpus_lib import Sample


def test_accept_only_when_label_set_unchanged():
    original = Sample(id="a", text="买大了想退", labels=("尺码", "退换货"), origin="real")
    assert accept_variant(original, "买大了一码想退货", ["退换货", "尺码"]) is True
    assert accept_variant(original, "买大了想退", ["尺码"]) is False


def test_augment_keeps_original_and_reports_drop_rate():
    train = [Sample(id="a", text="开发票", labels=("发票",), origin="real")]
    def prelabel(text): return ["发票"]
    out, report = augment_train(FakeLLM(["开票"], prelabel), train, budget=None)
    assert len(out) == 2 and report["kept"] == 1 and report["dropped"] == 0
```

Use a `FakeLLM` that returns one paraphrase per sample; `augment_train` accepts an injected `prelabel_fn` so no network is needed in tests.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ch10_augment.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

For each train sample, call the LLM with `AUGMENT_SYSTEM` once, require a non-empty paraphrase different from the original, then run the prelabel prompt on the variant and accept only if the label set matches exactly. Output includes the original and accepted variant; rejected variants are counted, not silently dropped. `FAILURES` on LLM/parse errors leave the original unchanged. CLI reads `dataset/train.jsonl`, writes `dataset/train_augmented.jsonl` + `dataset/augment_report.json`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ch10_augment.py -v`
Expected: PASS.

- [ ] **Step 5: Run the live augmentation**

Run: `PYTHONPATH=. .venv-ml/Scripts/python.exe -m scripts.ch10.augment_train`
Expected: `train_augmented.jsonl` is roughly 2× train; drop rate reported; val/test untouched. If drop rate > 20%, stop and inspect the prompt/labels before training.

- [ ] **Step 6: Update `dev-notes/ch10.md` and commit**

```bash
git add scripts/ch10/augment_train.py tests/test_ch10_augment.py dev-notes/ch10.md
git commit -m "feat: augment ch10 training set only"
```

---

### Task 11: 训练（TDD 纯逻辑 + 验证跑）

**Files:**
- Create: `scripts/ch10/train_lib.py`
- Create: `scripts/ch10/train.py`
- Test: `tests/test_ch10_train_lib.py`
- Output: `data/ch10/model/`、`data/ch10/reports/train_report.json`

**Interfaces:**
- Produces: `multi_hot(labels, label2id) -> list[float]`、`compute_metrics(gold: list[set[str]], pred: list[set[str]]) -> dict`（micro/macro P/R/F1）、`scan_threshold(probs: list[list[float]], gold, candidates=(0.30,...,0.70)) -> dict`（`best_threshold`、`best_micro_f1`、`scan`）。
- Produces: `train_model(train_path: Path, val_path: Path, out_dir: Path, *, model_name: str = "hfl/chinese-roberta-wwm-ext", max_length: int = 128, epochs: int = 8, batch_size: int = 16, lr: float = 2e-5) -> dict` in `train.py`.

- [ ] **Step 1: Write the failing tests**

```python
from scripts.ch10.train_lib import compute_metrics, multi_hot, scan_threshold


def test_multi_hot_uses_taxonomy_order():
    from app.core.taxonomy import LABEL2ID
    vec = multi_hot(["尺码", "退换货"], LABEL2ID)
    assert len(vec) == 17 and vec[LABEL2ID["尺码"]] == 1.0 and vec[LABEL2ID["退换货"]] == 1.0
    assert sum(vec) == 2.0


def test_compute_metrics_known_values():
    gold = [{"尺码", "退换货"}, {"物流"}]
    pred = [{"尺码"}, {"物流"}]
    out = compute_metrics(gold, pred)
    assert round(out["micro_f1"], 4) == 0.8
    assert 0 <= out["macro_f1"] <= 1


def test_scan_threshold_returns_best_candidate():
    probs = [[0.9] + [0.0] * 16, [0.4] + [0.0] * 16]
    gold = [{"退换货"}, set()]
    out = scan_threshold(probs, gold, candidates=(0.3, 0.5, 0.7))
    assert out["best_threshold"] in (0.3, 0.5, 0.7)
    assert len(out["scan"]) == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ch10_train_lib.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `train_lib.py`**（pure Python/typing only; no torch import）

Use the same micro/macro formulas as `eval_lib` will use; centralize them in `train_lib` and have `eval_lib` import them to avoid two truths. `scan_threshold` computes micro-F1 for each candidate and picks the highest; ties choose the highest threshold. Add the fallback-aware prediction helper from `taxonomy.labels_from_scores` when comparing.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ch10_train_lib.py -v`
Expected: PASS.

- [ ] **Step 5: Implement `train.py` and run it live**

Before writing, use Context7 `query-docs` for `transformers` `Trainer` / `AutoModelForSequenceClassification` multi-label classification and `EarlyStoppingCallback`. Implement:
- `Dataset` returning `input_ids`、`attention_mask`、`token_type_ids`（if tokenizer provides）、`labels` float multi-hot.
- `TrainingArguments` with `problem_type` handled by model config: `AutoConfig.from_pretrained(..., num_labels=17, problem_type="multi_label_classification")`; `metric_for_best_model="micro_f1"`, `greater_is_better=True`, `load_best_model_at_end=True`, `evaluation_strategy="epoch"`, `save_strategy="epoch"`, `save_total_limit=2`, `logging_steps=20`, `report_to=[]`, `seed=42`, `fp16=True` on CUDA.
- `train_report.json`: epoch metrics, best checkpoint, early-stop flag, duration, dataset sizes, and `torch_threshold_scan` (the full nine-candidate table).
- Save model + tokenizer to `data/ch10/model/`.
- After training, scan the nine thresholds on the validation set with torch and write a **provisional** `threshold.json` with `source: "torch"`; export will overwrite it with the ONNX version.
- CLI `--train data/ch10/dataset/train_augmented.jsonl --val data/ch10/dataset/val.jsonl --out data/ch10/model`.

Run: `PYTHONPATH=. .venv-ml/Scripts/python.exe -m scripts.ch10.train --train data/ch10/dataset/train_augmented.jsonl --val data/ch10/dataset/val.jsonl --out data/ch10/model`
Expected: training completes, early stopping/epoch curve in `train_report.json`, model files present. If CUDA OOM, reduce batch to 8 and rerun; record the change.

- [ ] **Step 6: Update `dev-notes/ch10.md` and commit**

```bash
git add scripts/ch10/train_lib.py scripts/ch10/train.py tests/test_ch10_train_lib.py dev-notes/ch10.md
git commit -m "feat: train ch10 multi-label classifier"
```

---

### Task 12: ONNX 导出与阈值重扫（验证跑）

**Files:**
- Create: `scripts/ch10/export_onnx.py`
- Test: `tests/test_ch10_export_lib.py`（纯一致性比较）
- Output: `data/ch10/onnx/model.onnx`、tokenizer 文件、`export_report.json`、更新 `threshold.json`、`data/ch10/reports/val_scores.json`

**Interfaces:**
- Produces: `compare_predictions(torch_probs, onnx_probs, threshold, atol=1e-4) -> dict`（max diff、label equality rate、ok）。
- Produces: CLI.

- [ ] **Step 1: Write the failing test**

```python
from scripts.ch10.export_onnx import compare_predictions


def test_compare_predictions_requires_identical_labels():
    torch_probs = [[0.9, 0.1], [0.2, 0.8]]
    onnx_probs = [[0.9, 0.1], [0.2, 0.8]]
    assert compare_predictions(torch_probs, onnx_probs, 0.5, atol=1e-4)["ok"] is True
    bad = [[0.9, 0.1], [0.6, 0.4]]
    assert compare_predictions(torch_probs, bad, 0.5, atol=1e-4)["ok"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ch10_export_lib.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement the comparator and export CLI**

Before writing, use Context7 for `torch.onnx.export` dynamic axes and `onnxruntime` inference. Export with dynamic batch/sequence axes, opset 17 (record actual), save tokenizer files (`save_pretrained`). Validate 32 samples: torch logits vs ONNX logits max diff, and labels with the same threshold must match exactly. Then run ONNX on the validation set, reuse `scripts.ch10.train_lib.scan_threshold` to scan the nine thresholds, overwrite `threshold.json` with `source: "onnx"` and the ONNX scan, write the fixed validation scores to `data/ch10/reports/val_scores.json`, and write `export_report.json` (`opset`、`samples`、`max_logit_diff`、`label_match_rate`、`ok`).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ch10_export_lib.py -v`
Expected: PASS.

- [ ] **Step 5: Run the live export**

Run: `PYTHONPATH=. .venv-ml/Scripts/python.exe -m scripts.ch10.export_onnx --model data/ch10/model --out data/ch10/onnx --val data/ch10/dataset/val.jsonl --reports data/ch10/reports`
Expected: `ok: true`, `label_match_rate: 1.0`, tokenizer.json present, `threshold.json` now `source: "onnx"`, `val_scores.json` present. If labels differ, fix tokenizer/input names/opset and rerun; do not proceed.

- [ ] **Step 6: Update `dev-notes/ch10.md` and commit**

```bash
git add scripts/ch10/export_onnx.py tests/test_ch10_export_lib.py dev-notes/ch10.md
git commit -m "feat: export ch10 classifier to ONNX"
```

---

### Task 13: 推理服务（TDD + live 探活）

**Files:**
- Create: `scripts/ch10/inference_lib.py`
- Create: `scripts/ch10/serve.py`
- Test: `tests/test_ch10_serve.py`
- Output: service on `127.0.0.1:8110`、`log/classifier.log`、`log/classifier.pid`

**Interfaces:**
- Produces: `Classifier(model_path, tokenizer_path, threshold_path)` with `classify(texts: list[str]) -> list[dict]`（`{"labels": [...], "scores": {...}}`）。
- Produces: `create_app(classifier=None) -> FastAPI`、`main(argv) -> int`（`--daemon`/`--stop`）。

- [ ] **Step 1: Write the failing API tests**

```python
from fastapi.testclient import TestClient
from scripts.ch10.serve import create_app


class FakeClassifier:
    def classify(self, texts):
        assert texts == ["买大了想退"]
        return [{"labels": ["尺码", "退换货"], "scores": {"尺码": 0.9, "退换货": 0.8}}]


def test_healthz_and_classify():
    client = TestClient(create_app(FakeClassifier()))
    assert client.get("/healthz").json()["status"] == "ok"
    body = client.post("/classify", json={"texts": ["买大了想退"]}).json()
    assert body["items"][0]["labels"] == ["尺码", "退换货"]


def test_classify_rejects_empty_and_too_large():
    client = TestClient(create_app(FakeClassifier()))
    assert client.post("/classify", json={"texts": []}).status_code == 400
    assert client.post("/classify", json={"texts": ["x"] * 65}).status_code == 400
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ch10_serve.py -v`
Expected: FAIL with missing module (or import skip if onnxruntime missing; the module must import onnxruntime lazily inside `Classifier.__init__` so the FastAPI tests run in the main `.venv`).

- [ ] **Step 3: Implement**

`inference_lib.py`: load `tokenizer.json` with `tokenizers.Tokenizer`, enable truncation to 128 and padding to the batch max; build numpy inputs from the ONNX session's actual input names; run `session.run(None, feed)`; sigmoid; `labels_from_scores` from taxonomy; return scores for all 17. `serve.py`: FastAPI app with `POST /classify` (`texts` 1–64, else 400), `GET /healthz`; `--daemon` spawns a detached process with Windows `creationflags=CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS`, writes pid, polls health up to 30s; `--stop` kills the pid tree with `taskkill /PID /T /F` on Windows and `os.killpg` on POSIX. The service process imports torch/transformers nowhere.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ch10_serve.py -v`
Expected: PASS.

- [ ] **Step 5: Start the live service and verify**

Run: `make classifier-up` then `curl -s http://127.0.0.1:8110/healthz` and `curl -s -X POST http://127.0.0.1:8110/classify -H "Content-Type: application/json" -d '{"texts":["买大了想退"]}'`
Expected: health `{"status":"ok",...}`; classify returns `尺码` + `退换货` (plus scores). `make classifier-down` stops it cleanly.

- [ ] **Step 6: Update `dev-notes/ch10.md` and commit**

```bash
git add scripts/ch10/inference_lib.py scripts/ch10/serve.py tests/test_ch10_serve.py dev-notes/ch10.md
git commit -m "feat: add onnx classifier inference service"
```

---

### Task 14: 评测、混淆矩阵与阈值重演（TDD + 验证跑）

**Files:**
- Create: `scripts/ch10/eval_lib.py`
- Create: `scripts/ch10/evaluate.py`
- Create: `scripts/ch10/scan_threshold_replay.py`
- Test: `tests/test_ch10_eval_lib.py`
- Output: `data/ch10/reports/eval_report.{md,json}`、`error_samples.md`、`threshold_scan.json`、`val_scores.json`

**Interfaces:**
- Produces: `tier_for(label) -> str`、`red_line_for(label) -> float | None`、`error_directions(gold, pred) -> dict`、`reconcile_errors(report) -> bool`。
- Produces: `replay_threshold(val_scores, candidates, expected) -> dict`。

- [ ] **Step 1: Write the failing tests**

```python
from scripts.ch10.eval_lib import error_directions, red_line_for, tier_for


def test_tiers_and_red_lines():
    assert tier_for("退换货") == "严" and red_line_for("退换货") == 0.9
    assert tier_for("运费") == "中" and red_line_for("运费") == 0.8
    assert tier_for("评价") == "宽" and red_line_for("评价") is None


def test_error_directions_miss_extra_misplaced():
    gold = [{"尺码", "退换货"}]
    pred = [{"尺码", "物流"}]
    out = error_directions(gold, pred)
    assert out["miss"] == {"退换货": 1}
    assert out["extra"] == {"物流": 1}
    assert out["misplaced_pairs"] == [{"gold": "退换货", "pred": "物流"}]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ch10_eval_lib.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

`eval_lib`: tier/red-line tables from spec §8; `error_directions` counts misses/extras and pairs a miss with an extra as a misplaced pair; `reconcile_errors` asserts `miss_total + extra_total == total_fp + total_fn` (each misplaced contributes one to each side). `evaluate.py` loads the test set, scores it through the **live service** (`--base-url http://127.0.0.1:8110` by default; `--onnx` in-process is allowed only for offline debugging), uses sklearn `multilabel_confusion_matrix` for the 17 matrices, writes the `.md` and `.json` from the same in-memory dict, and includes `corpus_composition`、`threshold`（`source: "onnx"`）、`overall`、`per_class`、`confusion_matrices`、`error_counts`、`errors`. `scan_threshold_replay.py` re-scores the validation set through the service **once**, writes the fixed `val_scores.json`, applies the nine candidates, writes `threshold_scan.json`, and exits non-zero if the winner != `threshold.json["threshold"]`. If the service is not up, both CLIs exit with a clear "start classifier first" error instead of silently using torch.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ch10_eval_lib.py -v`
Expected: PASS.

- [ ] **Step 5: Run the live evaluation and replay**

Ensure the service is up: `make classifier-up`.
Run: `PYTHONPATH=. .venv-ml/Scripts/python.exe -m scripts.ch10.evaluate --test data/ch10/dataset/test.jsonl --threshold data/ch10/model/threshold.json --base-url http://127.0.0.1:8110 --out data/ch10/reports`
Expected: 17 classes, per-class P/R/F1/support, 17 confusion matrices, error direction totals reconcile, `eval_report.md/json` present, source notes ONNX.
Then run: `PYTHONPATH=. .venv-ml/Scripts/python.exe -m scripts.ch10.scan_threshold_replay --val data/ch10/dataset/val.jsonl --base-url http://127.0.0.1:8110 --threshold data/ch10/model/threshold.json --out data/ch10/reports`
Expected: `val_scores.json` written from the service; winner matches `threshold.json`; exit 0.

- [ ] **Step 6: Update `dev-notes/ch10.md` and commit**

```bash
git add scripts/ch10/eval_lib.py scripts/ch10/evaluate.py scripts/ch10/scan_threshold_replay.py tests/test_ch10_eval_lib.py dev-notes/ch10.md
git commit -m "feat: evaluate ch10 classifier with red lines and errors"
```

---

### Task 15: 旁路批量归类（TDD + live）

**Files:**
- Create: `scripts/ch10/classify_lib.py`
- Create: `scripts/ch10/classify_pool.py`
- Test: `tests/test_ch10_classify.py`
- Output: `topic_classifications` rows、`data/ch10/reports/classify_run.json`

**Interfaces:**
- Produces: `select_unclassified(session, limit=None, reclassify=False) -> list[dict]`（含 `question_id`、`text`、`normalized`）。
- Produces: `write_classifications(session, items) -> int`（幂等，唯一键冲突跳过）。
- Produces: `call_service(base_url, texts, batch_size=32, timeout=10) -> list[list[str]]`。
- Produces: `run(session, base_url, min_batch=10, force=False, limit=None, service=None) -> dict`（`service` 可注入，测试用）。

- [ ] **Step 1: Write the failing tests**

```python
from scripts.ch10.classify_lib import select_unclassified, write_classifications


def _seed(session):
    from app.models import LowConfidenceQuestion, ReviewQueue
    r = ReviewQueue(normalized_question="买大了想退"); session.add(r); session.commit(); session.refresh(r)
    q = LowConfidenceQuestion(raw_question="买大了想退", source="self_check", matched_review_id=r.id)
    session.add(q); session.commit(); session.refresh(q)
    return q, r


def test_select_only_unclassified_matched(session):
    q, r = _seed(session)
    rows = select_unclassified(session)
    assert [row["question_id"] for row in rows] == [q.id]
    assert rows[0]["text"] == "买大了想退" and rows[0]["normalized"] is True


def test_write_is_idempotent(session):
    q, _ = _seed(session)
    items = [{"question_id": q.id, "labels": ["尺码", "退换货"]}]
    assert write_classifications(session, items) == 1
    assert write_classifications(session, items) == 0
    assert select_unclassified(session) == []


def test_min_batch_and_force(session):
    from scripts.ch10.classify_lib import run
    _seed(session)
    calls = []
    out = run(session, "http://fake", min_batch=10, force=False,
              service=lambda texts, **k: calls.append(texts))
    assert out["skipped"] is True and calls == []

    out2 = run(session, "http://fake", min_batch=10, force=True,
               service=lambda texts, **k: [["尺码"]] * len(texts))
    assert out2["classified"] == 1


def test_service_failure_writes_nothing(session):
    from app.models import TopicClassification
    from scripts.ch10.classify_lib import run
    _seed(session)
    def boom(texts, **kwargs):
        raise RuntimeError("service down")
    out = run(session, "http://fake", min_batch=1, force=True, service=boom)
    assert out["ok"] is False
    assert session.query(TopicClassification).count() == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ch10_classify.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

`select_unclassified` uses a `LEFT JOIN topic_classifications` filtered to `matched_review_id IS NOT NULL`; text is `review_queue.normalized_question`, falling back to `raw_question` and setting `normalized=False`. `write_classifications` inserts only missing question ids in one transaction; on `IntegrityError` roll back and re-query/skip (idempotent). `call_service` posts `{"texts": batch}` and validates the response shape/labels. `run` checks `len(pending) < min_batch and not force` → returns `{"skipped": True, "reason": "not enough pending"}`; otherwise batches and writes; on any service error, writes nothing from the failed batch and returns non-zero. `classify_run.json` records pending, classified, skipped, label counts, duration.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ch10_classify.py -v`
Expected: PASS.

- [ ] **Step 5: Run the live batch**

With the service up: `make classify-pool FORCE=1`
Expected: 16 rows classified, `topic_classifications` count 16; rerun says `0 条新问题`. Show the label distribution.

- [ ] **Step 6: Update `dev-notes/ch10.md` and commit**

```bash
git add scripts/ch10/classify_lib.py scripts/ch10/classify_pool.py tests/test_ch10_classify.py dev-notes/ch10.md
git commit -m "feat: batch-classify low-confidence questions"
```

---

### Task 16: 主题分布与问题列表 API（TDD）

**Files:**
- Create: `app/api/topics.py`
- Modify: `app/main.py` (include page/API routers)
- Test: `tests/test_ch10_topics_api.py`

**Interfaces:**
- Produces: `GET /api/topics/distribution` → `{total, pending, last_classified_at, labels:[{label,count,samples:[{text,question_id}]}]}`。
- Produces: `GET /api/topics/questions?label=&page=&size=` → `{label,total,page,pages,size,items:[...]}`；非法 label 400。
- Produces: page routes `/topics`、`/topics/questions` (serve static HTML; HTML added in Task 17).

- [ ] **Step 1: Write the failing tests**

```python
from fastapi.testclient import TestClient
from app.main import app
from app.models import LowConfidenceQuestion, ReviewQueue, TopicClassification


def test_distribution_counts_and_dedupes_samples(session):
    r = ReviewQueue(normalized_question="买大了想退", occurrence_count=3); session.add(r); session.commit(); session.refresh(r)
    qs = [LowConfidenceQuestion(raw_question=f"买大了想退{i}", source="self_check", matched_review_id=r.id) for i in range(2)]
    session.add_all(qs); session.commit()
    for q in qs: session.add(TopicClassification(question_id=q.id, labels=["尺码", "退换货"]))
    session.commit()
    body = TestClient(app).get("/api/topics/distribution").json()
    assert body["total"] == 2
    size = next(x for x in body["labels"] if x["label"] == "尺码")
    assert size["count"] == 2 and len(size["samples"]) == 1
    # multi-label questions appear under every hit label
    assert next(x for x in body["labels"] if x["label"] == "退换货")["count"] == 2


def test_questions_invalid_label_is_400_and_pagination():
    client = TestClient(app)
    assert client.get("/api/topics/questions", params={"label": "售后"}).status_code == 400
    assert client.get("/api/topics/questions", params={"label": "尺码", "page": 1, "size": 2}).status_code == 200
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ch10_topics_api.py -v`
Expected: FAIL with 404.

- [ ] **Step 3: Implement**

Query `TopicClassification` joined to `LowConfidenceQuestion` and `ReviewQueue` once in Python; aggregate counts per label; dedupe sample texts per label; `pending` counts matched low-confidence rows with no classification. `questions` validates `label in LABEL2ID`, filters in Python, paginates (`page>=1`, `size` 1–100), and returns `normalized=false` when the text came from `raw_question`. Include `source`、`occurrence_count`、`review_status`、`created_at`、`classified_at`、`labels`. Add routers to `app/main.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ch10_topics_api.py -v`
Expected: PASS.

- [ ] **Step 5: Update `dev-notes/ch10.md` and commit**

```bash
git add app/api/topics.py app/main.py tests/test_ch10_topics_api.py dev-notes/ch10.md
git commit -m "feat: add topic distribution and question APIs"
```

---

### Task 17: 主题页（Vibe Coding）

**Files:**
- Create: `app/static/topics.html`
- Create: `app/static/topic-questions.html`
- Modify: `app/static/admin.js` (add nav items)
- Test: manual browser + `curl`

**Description to implement:**
- `/topics`：标题「主题分布」；顶部显示已归类总数、未归类数、最近归类时间；17 类横向条形图（纯 HTML/CSS，宽度按 count/max），每行显示类目名、计数、占比；点击类目名进 `/topics/questions?label=...`；每类下方最多 3 条去重样例。
- `/topics/questions`：地址栏读 `label`、`page`、`size`；表格列：标准化问法、原话、命中类目、落池入口、同义合并条数、审核状态、落池时间、归类时间；分页上一页/下一页；非法类目显示 API 的 400 文案；刷新保持同一页。
- 样式复用 `review.html` 的 CSS 变量与像素风；不引外部图表库。

- [ ] **Step 1: Implement the pages against the API**
- [ ] **Step 2: Start the app and verify visually**

Run: `python run.py`, open `http://localhost:8000/topics` and click through to a category. Verify bars, counts, samples, pagination, invalid label message, and that the shared nav highlights the current page. Ask the user for visual corrections and apply them.

- [ ] **Step 3: Update `dev-notes/ch10.md` and commit**

```bash
git add app/static/topics.html app/static/topic-questions.html app/static/admin.js dev-notes/ch10.md
git commit -m "feat: add topic distribution pages"
```

---

### Task 18: 验收 API（TDD）

**Files:**
- Create: `app/api/acceptance.py`
- Modify: `app/main.py` (include page/API routers)
- Test: `tests/test_ch10_acceptance_api.py`

**Interfaces:**
- Produces: `GET /api/acceptance/overview|eval|data|errors|service`、`POST /api/acceptance/classify`。
- Produces: page routes `/acceptance`、`/acceptance/eval`、`/acceptance/data`、`/acceptance/errors` (HTML added in Task 20).
- `overview` gate ids: `corpus`、`golden`、`train`、`export`、`eval`、`threshold`、`confusion`、`errors`、`classify`.

- [ ] **Step 1: Write the failing tests**

```python
from fastapi.testclient import TestClient
from app.main import app
from app.api import acceptance


def test_missing_artifacts_are_present_false_not_500(tmp_path, monkeypatch):
    monkeypatch.setattr(acceptance, "REPORT_DIR", tmp_path)
    body = TestClient(app).get("/api/acceptance/eval").json()
    assert body["present"] is False and "ch10-eval" in body["hint"]


def test_overview_counts_nine_gates_and_service_offline(monkeypatch, tmp_path):
    monkeypatch.setattr(acceptance, "REPORT_DIR", tmp_path)
    monkeypatch.setattr(acceptance, "_service_health", lambda: False)
    body = TestClient(app).get("/api/acceptance/overview").json()
    assert len(body["gates"]) == 9 and body["total"] == 9
    assert body["classifier_online"] is False
    assert any(g["present"] is False and g["state"] == "missing" for g in body["gates"])


def test_invalid_topic_label_is_rejected_by_classify_proxy(monkeypatch):
    monkeypatch.setattr(acceptance, "_service_health", lambda: True)
    resp = TestClient(app).post("/api/acceptance/classify", json={"texts": []})
    assert resp.status_code == 400
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ch10_acceptance_api.py -v`
Expected: FAIL with 404.

- [ ] **Step 3: Implement**

Read `REPORT_DIR` files with `json.loads` in try/except; missing/corrupt → `{"present": false, "hint": "make <target>"}`. `overview` builds the nine gates with `present`、`state` (`pass`/`fail`/`missing`)、`job`、`log`、`detail`. The `eval` gate passes when `eval_report.json` has 17 per-class rows; red lines are included as diagnostics, not gate failure. `service` probes `http://127.0.0.1:8110/healthz` with `httpx` timeout 2. `classify` validates 1–64 non-empty texts, proxies to the service, and returns 502 on connection errors; invalid input is 400. Do not recompute metrics in the API.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ch10_acceptance_api.py -v`
Expected: PASS.

- [ ] **Step 5: Update `dev-notes/ch10.md` and commit**

```bash
git add app/api/acceptance.py app/main.py tests/test_ch10_acceptance_api.py dev-notes/ch10.md
git commit -m "feat: add ch10 acceptance API"
```

---

### Task 19: 作业运行器（TDD）

**Files:**
- Create: `app/core/jobs.py`
- Create: `app/api/jobs.py`
- Modify: `app/main.py` (include router)
- Modify: `Makefile` (ch10 targets)
- Test: `tests/test_ch10_jobs.py`

**Interfaces:**
- Produces: `JobSpec(name, argv, heavy, description)`、`JOBS: dict[str, JobSpec]`。
- Produces: `start_job(name) -> dict | None`、`job_status(name) -> dict | None`、`stop_job(name) -> bool`、`_platform_is_windows() -> bool`。
- Produces: `POST /api/jobs/{name}/run` (202/404/409)、`GET /api/jobs/{name}/status`、`POST /api/jobs/{name}/stop`。

- [ ] **Step 1: Write the failing tests**

```python
from app.core import jobs


class FakeProc:
    def __init__(self):
        self.pid = 1234
        self.returncode = None
    def poll(self):
        return self.returncode
    def finish(self, code):
        self.returncode = code


def test_unknown_job_rejected_and_reentry_rejected(monkeypatch):
    assert jobs.start_job("not-a-job") is None
    fake = FakeProc()
    monkeypatch.setattr(jobs.subprocess, "Popen", lambda *a, **k: fake)
    assert jobs.start_job("ch10-eval") is not None
    assert jobs.start_job("ch10-eval") is None  # same name still running
    fake.finish(0)
    assert jobs.job_status("ch10-eval")["exit_code"] == 0


def test_stop_windows_kills_tree(monkeypatch):
    called = []
    fake = FakeProc()
    monkeypatch.setattr(jobs, "_platform_is_windows", lambda: True)
    monkeypatch.setattr(jobs.subprocess, "Popen", lambda *a, **k: fake)
    monkeypatch.setattr(jobs.subprocess, "run", lambda argv, **k: called.append(argv))
    jobs.start_job("classify-pool")
    jobs.stop_job("classify-pool")
    assert called and "taskkill" in called[0][0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ch10_jobs.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

`JOBS` maps each name to a fixed `["make", target]` argv (plus `env={"FORCE": "1"}` for `classify-pool-force`). `start_job` acquires a per-name lock, refuses if running, opens `log/acceptance/<name>.log`, starts `Popen(cwd=ROOT, stdout=log, stderr=STDOUT, start_new_session=True)` on POSIX or `creationflags=CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS` on Windows, and stores state in memory. `job_status` returns running/started/finished/exit_code/output_tail (last 4000 chars of the log). `stop_job` uses `taskkill /PID <pid> /T /F` on Windows, `os.killpg(os.getpgid(pid), SIGTERM)` on POSIX. Heavy jobs are marked in `JobSpec`; the API does not enforce confirmation (the page does).

The Task 2 Makefile already defines `ch10-golden`、`ch10-corpus`、`ch10-dataset`、`ch10-train`、`ch10-export`、`classifier-up`、`classifier-down`、`ch10-eval`、`ch10-threshold-scan`、`classify-pool`; register exactly those names in `JOBS`. Do not invent a command string in the runner: the registry only stores the target name and `make` reads the recipe.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ch10_jobs.py -v`
Expected: PASS.

- [ ] **Step 5: Verify a real make target through the API**

Start the app, `POST /api/jobs/ch10-golden/run`, poll status, verify the log tail and exit code. (This runs the real golden gate; it may take a minute.)

- [ ] **Step 6: Update `dev-notes/ch10.md` and commit**

```bash
git add app/core/jobs.py app/api/jobs.py app/main.py Makefile tests/test_ch10_jobs.py dev-notes/ch10.md
git commit -m "feat: add make-backed acceptance job runner"
```

---

### Task 20: 验收页与后台首页卡（Vibe Coding）

**Files:**
- Create: `app/static/acceptance.html`
- Create: `app/static/acceptance-eval.html`
- Create: `app/static/acceptance-data.html`
- Create: `app/static/acceptance-errors.html`
- Create: `app/static/acceptance.css`
- Create: `app/static/acceptance.js`
- Modify: `app/static/admin.html` (add card)
- Modify: `app/static/admin.js` (nav item already added in Task 17; ensure「验收」present)

**Description to implement:**
- `/acceptance`：九张闸门卡 + 顶部总闸 N/9 + 分类器在线状态；每卡显示 pass/fail/missing 三态、重跑按钮、日志窗口；heavy 作业二次确认。
- `/acceptance/eval`：micro vs macro、17 类 P/R/F1/support/红线表、九档阈值扫描、17 张 2×2 混淆矩阵、单句试分类输入框（调用 `/api/acceptance/classify`）。
- `/acceptance/data`：语料血缘四段、文件盘点、三份考卷分布与泄漏自检、训练产物、ONNX 产物。
- `/acceptance/errors`：逐条标准/预测对照、错误方向记账、边界摩擦配对。
- `/admin`：新增一张卡显示过闸数 N/9 与分类器在线状态。
- 共享 `acceptance.css/js`；不引外部图表库；missing 不显示失败文案。

- [ ] **Step 1: Implement the pages against the APIs**
- [ ] **Step 2: Start the app and verify visually**

Open all four pages plus `/admin`; verify gate states, rerun buttons, log tail, confusion matrices, single-sentence classification (`买大了想退` → 尺码 + 退换货), and missing-state copy. Ask the user for visual corrections and apply them.

- [ ] **Step 3: Update `dev-notes/ch10.md` and commit**

```bash
git add app/static/acceptance* app/static/admin.html app/static/admin.js dev-notes/ch10.md
git commit -m "feat: add ch10 acceptance pages"
```

---

### Task 21: 端到端验收与收尾

**Files:**
- Modify: `dev-notes/ch10.md`
- Output: acceptance evidence, screenshots if useful

- [ ] **Step 1: Run the full pipeline from a clean state**

```bash
make ch10-env
make ch10-golden
make ch10-corpus
make ch10-dataset
make ch10-train
make ch10-export
make classifier-up
make ch10-eval
make ch10-threshold-scan
make classify-pool FORCE=1
```
Expected: each target exits 0; reports under `data/ch10/reports/`; `topic_classifications` 16 rows.

- [ ] **Step 2: Verify acceptance criteria**

1. `/acceptance/eval` shows per-class F1, red-line diagnostics, 17 confusion matrices.
2. `/topics` shows 17 category counts; click a category → paginated list with source, merge count, review status.
3. `POST /api/acceptance/classify {"texts":["买大了想退"]}` returns both `尺码` and `退换货` with per-class scores.
4. `/acceptance` shows all nine gates present, N/9, classifier online; rerun one lightweight job and confirm log tail updates.

- [ ] **Step 3: Run both test suites**

```bash
.venv/Scripts/python.exe -m pytest -q
```
Expected: main suite passes with no regressions; all ch10 pure-logic tests are part of this suite. The ML paths were already verified by the live runs in Tasks 6–15.

- [ ] **Step 4: Whole-branch review**

Invoke `superpowers:requesting-code-review` on the backend/ML changes, then `superpowers:receiving-code-review` for findings. Exclude the Vibe-Coded static HTML/CSS/JS pages from the code-review gate (their acceptance is the user's visual check). Fix verified issues. Append the review conclusion to `dev-notes/ch10.md`.

- [ ] **Step 5: Final dev-notes and commit**

Append the finish section with: acceptance commands, test results, dev-notes path, known residual risks (synthetic-heavy corpus, diagnostic red lines, Windows-specific service lifecycle). Commit:
```bash
git add dev-notes/ch10.md
git commit -m "docs: record ch10 acceptance evidence"
```
