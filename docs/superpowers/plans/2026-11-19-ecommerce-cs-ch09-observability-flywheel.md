# Ch09 可观测性与数据飞轮 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给客服系统接上 Langfuse v3 全链路观测与意图 token 账，并把置信度低、selfcheck 不够答、用户 👎 三类问题经批处理飞轮送入 `review_queue`，人工审核通过后写回知识库形成闭环。

**Architecture:** 观测采用“编译期挂一次 CallbackHandler、脚本为辅”——`compile().with_config()` 后 `stream` 走带回调的 binding，`get_state` 通过 `_state_graph` 取 `bound`；意图写入 trace metadata/tag，成本与评估由 `app/jobs` 脚本产出报表。飞轮是串行批处理：扫 `matched_review_id IS NULL`，一次 LLM 输出标准化+查重+示例答案，命中 `review_queue` 全部行只累加次数、状态不变。

**Tech Stack:** Python 3.13、FastAPI、SQLAlchemy/MySQL、LangGraph 1.2.20、LangChain 1.5、Langfuse v3（`langfuse==3.15.0`）、Milvus、现有 ch04 评估集。

**Spec:** `docs/superpowers/specs/2026-11-19-ch09-observability-flywheel-design.md`

## Global Constraints

- 当前分支从 `feature/ch08-tool-system` 切出 `feature/ch09-observability-flywheel`；不要在旧分支直接改。
- 基线测试：772 passed（ch08）。每个任务结束必须跑聚焦测试 + 全量 `pytest -q`，不得把回归留给最后。
- 只新增依赖 `langfuse==3.15.0`；不引入 APScheduler、pandas、前端依赖或第二个调度框架。
- Langfuse 是 v3：镜像 `langfuse/langfuse:3` / `langfuse-worker:3`，SDK `3.15.0`；不按 v4 API 写。
- `compile().with_config()` 返回 `RunnableBinding`，没有 `get_state`/`get_state_history`；所有 state 读取必须走 `_state_graph(graph)`，漏一处会在 interrupt/resume/feedback 处炸。
- 飞轮只批处理，不在聊天请求里同步调用；验收用 `make flywheel`。
- 查重比对 `review_queue` **全部行**；命中已驳回/已通过只 `occurrence_count += 1`，`review_status` 不变，不新建行。
- 落池 `source`：置信度低 `retrieval_low_conf`，selfcheck 不够答 `self_check`，用户反馈 `user_feedback`。infra/rerank 失败不落池。
- `retrieved_chunks`/`retrieved_snapshot` 只存 Top 3，answer 截断 500 字符；没走检索存 NULL。
- 阈值必须由 `tests/eval/rag_cases.json` live 校准（Youden J）；服务不可用就阻塞，不写占位值。
- 前端走 Vibe Coding：不 TDD、不 code review；浏览器人工验收。
- Windows：Python 用 `.venv/Scripts/python.exe`；中文输出加 `PYTHONUTF8=1`。
- 每个阶段完成后立即追记 `dev-notes/ch09.md`，不许收尾一次性补。

## Review Focus

1. **`with_config` 后的 state 读取**：编译期挂回调解包成 `RunnableBinding` 后，interrupt/resume、`has_pending_interrupt`、feedback 的快照回捞仍必须读到同一 thread 的 state；错用 binding 会 500 或永远 409。
2. **编译期共享 CallbackHandler 的并发隔离**：两个会话并发聊天时 trace 不能串；用两个并发请求验证 sessionId/intent 各归各。
3. **飞轮命中已驳回/已通过行**：必须只累加 `occurrence_count`、状态不变、不新建；命中后仍要回写 `low_confidence_questions.matched_review_id`。
4. **selfcheck 基础设施失败**：selfcheck 抛错/超时只能保守兜底，绝不能落池；只有 `useful=false` 才 `source=self_check`。
5. **审核写回的顺序与幂等**：只有 `vectorize_pending` 成功后才能把 `review_status` 置通过；失败保持待审、重试复用同一 chunk，不重复建行。

---

## File Structure

**新增/修改的核心文件**

- `sql/ch09-ddl.sql`：用户 DDL 原文。
- `app/db/sql_utils.py`：引号感知的 SQL 分句；ch04 兼容导入。
- `app/db/init_ch09.py`：幂等建表/加列。
- `app/models/review.py`：`ReviewQueue`、`EvalRun`。
- `app/models/knowledge.py`：`LowConfidenceQuestion` 新增两列。
- `app/models/__init__.py`：导出新模型。
- `app/config.py`：Langfuse/置信度/飞轮/报表配置；`build_llm(stream_usage=True)`。
- `app/observability.py`：Langfuse 客户端、CallbackHandler、intent trace 更新、`attach_tracing`、`_state_graph`、flush。
- `app/api/ch06.py`：`_build_and_cache_graph` 挂回调；state 读取适配。
- `app/agent/service.py`：`_thread_config` session metadata；`_finish_turn` 落池 source/快照；state 读取适配。
- `app/agent/state.py`：`evidence_confidence`/`evidence_signals`/`fallback_source`/`retrieved_snapshot`。
- `app/agent/confidence.py`：`compute_evidence_confidence`。
- `app/agent/graph.py`：置信度闸、selfcheck 闸、快照、pool source。
- `app/rag/retrieval.py`：`RetrievedChunk.is_key_clause`。
- `app/services/low_confidence.py`：`retrieved_chunks` 参数。
- `app/services/flywheel.py`：标准化+查重一次 LLM、批处理。
- `app/services/review_repo.py`：review queue 查询/更新。
- `app/services/knowledge_approval.py`：approve/reject、写回 + 向量化。
- `app/services/admin_jobs.py`：白名单作业运行器。
- `app/api/feedback.py`、`app/api/review.py`、`app/api/observability.py`、`app/api/admin.py`、`app/api/admin_jobs.py`。
- `app/jobs/calibrate_confidence.py`、`app/jobs/cost_by_intent.py`、`app/jobs/flywheel_pipeline.py`、`app/jobs/eval_flywheel.py`。
- `scripts/calibrate_confidence.py`、`scripts/cost_by_intent.py`、`scripts/flywheel_pipeline.py`、`scripts/eval_flywheel.py`：薄包装。
- `app/static/ch07.html`、`app/static/review.html`、`app/static/observability.html`、`app/static/admin.html`、`app/static/admin.js`。
- `docker-compose.langfuse.yml`、`Makefile`、`data/ch09/reports/.gitkeep`。
- `tests/test_ch09_schema.py`、`test_observability.py`、`test_ch09_confidence.py`、`test_ch09_calibration.py`、`test_flywheel.py`、`test_review_api.py`、`test_feedback_api.py`、`test_eval_flywheel.py`、`test_admin_jobs.py`、`test_observability_api.py`、`tests/eval/flywheel_samples.json`。

---

## Task 1: 分支、依赖、配置、Langfuse compose、DDL/ORM/初始化

**Files:**
- Create: `sql/ch09-ddl.sql`
- Create: `app/db/sql_utils.py`
- Modify: `app/db/init_ch04.py`（改为复用 `sql_utils`，保留 `_split_statements` 兼容导出）
- Create: `app/db/init_ch09.py`
- Create: `app/models/review.py`
- Modify: `app/models/knowledge.py`
- Modify: `app/models/__init__.py`
- Modify: `app/config.py`
- Modify: `requirements.txt`
- Modify: `.env.example`
- Create: `docker-compose.langfuse.yml`
- Create: `Makefile`
- Create: `data/ch09/reports/.gitkeep`
- Test: `tests/test_ch09_schema.py`、`tests/test_config.py`、`tests/test_ch04_schema.py`（兼容）

**Interfaces:**
- Produces: `ReviewQueue`/`EvalRun` ORM；`LowConfidenceQuestion.retrieved_chunks`、`.matched_review_id`；`Settings.langfuse_public_key/langfuse_secret_key/langfuse_base_url/langfuse_environment/langfuse_trace_name/stream_usage/evidence_confidence_threshold/evidence_valid_floor/evidence_valid_cap/evidence_margin_scale/low_confidence_snapshot_top_k/flywheel_candidate_limit/flywheel_llm_max_tokens/flywheel_timeout_s/ch09_report_dir`。
- Produces: `app.db.init_ch09.apply_ch09(engine) -> None`；`app.db.sql_utils.split_statements(sql) -> list[str]`。

- [ ] **Step 1: 建分支**

```bash
git switch -c feature/ch09-observability-flywheel
```

- [ ] **Step 2: 写 schema 失败测试**

```python
# tests/test_ch09_schema.py
from pathlib import Path
from app.db.sql_utils import split_statements
from app.models import EvalRun, LowConfidenceQuestion, ReviewQueue

def test_ch09_ddl_has_required_tables_columns_and_fk():
    sql = Path("sql/ch09-ddl.sql").read_text(encoding="utf-8")
    assert "CREATE TABLE review_queue" in sql
    assert "CREATE TABLE eval_runs" in sql
    assert "retrieved_chunks" in sql
    assert "matched_review_id" in sql
    assert "fk_lcq_review" in sql

def test_split_statements_handles_comment_semicolons():
    sql = "CREATE TABLE t (c INT COMMENT 'a;b'); ALTER TABLE t ADD COLUMN d INT;"
    assert len(split_statements(sql)) == 2

def test_orm_tables_can_be_created(session):
    assert session.query(ReviewQueue).count() == 0
    assert session.query(EvalRun).count() == 0
    assert LowConfidenceQuestion.__table__.c.retrieved_chunks is not None
    assert LowConfidenceQuestion.__table__.c.matched_review_id is not None
```

- [ ] **Step 3: 运行测试确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ch09_schema.py -q`
Expected: FAIL（模块/表不存在）。

- [ ] **Step 4: 落 DDL、ORM、sql_utils、init_ch09**

- `sql/ch09-ddl.sql` 使用用户 2026-11-19 提供的 DDL 原文（`SET NAMES utf8mb4;`、`CREATE review_queue`、`CREATE eval_runs`、`ALTER low_confidence_questions ...`）。
- `app/db/sql_utils.py`：把 `init_ch04.py` 的 `_split_statements` 移到这里并公开为 `split_statements`；`init_ch04.py` 改为 `from app.db.sql_utils import split_statements as _split_statements`。
- `app/db/init_ch09.py`：
  - `SQL_PATH`、`TABLES = ("review_queue","eval_runs")`。
  - 检查表存在 + `low_confidence_questions` 两列存在；全在 no-op、全缺执行、部分抛 `RuntimeError`。
  - `def apply_ch09(engine) -> None`，`main()` 用 `get_engine()`。
- `app/models/review.py`：
  - `ReviewQueue`：`id`（BigInteger SQLite variant）、`normalized_question` String(512)、`ai_suggested_answer` Text、`occurrence_count` Integer default 1、`review_status` Enum("待审","通过","驳回") default "待审"、`approved_answer` Text、`created_at`/`updated_at` server default/onupdate。
  - `EvalRun`：`id`、`triggered_by` Enum("定时","手动") default "定时"、`dataset_size` Integer、`metrics` JSON、`created_at`。
- `LowConfidenceQuestion` 加 `retrieved_chunks: Optional[list] = mapped_column(JSON)`、`matched_review_id: Optional[int] = mapped_column(BigInteger, ForeignKey("review_queue.id", ondelete="SET NULL"), index=True)`。
- `app/models/__init__.py` 导出。
- `app/config.py` 新增配置；`build_llm` 增加 `stream_usage=True`。移除/弃用 `confidence_min_score`（若保留字段，注释 deprecated 且不再被闸门使用）。
- `requirements.txt` 加 `langfuse==3.15.0`；`.env.example` 加 Langfuse/置信度/飞轮/报表键。
- `docker-compose.langfuse.yml`：v3 官方栈六服务；只暴露 3000；MinIO 不映射宿主端口；web/worker 共用 DB/CH/Redis/S3 env；web 额外 NEXTAUTH/SALT/ENCRYPTION_KEY；可选 headless init；`TELEMETRY_ENABLED=false`。
- `Makefile`：`PYTHON ?= .venv/Scripts/python.exe`；`langfuse-up`、`langfuse-down`、`test`；后续任务追加 `flywheel`/`cost-report`/`eval-flywheel`/`calibrate-confidence`。
- `data/ch09/reports/.gitkeep`。

- [ ] **Step 5: 运行测试确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ch09_schema.py tests/test_ch04_schema.py tests/test_config.py -q`
Expected: PASS（更新 config 测试以匹配新默认值）。

- [ ] **Step 6: 提交**

```bash
git add sql/ch09-ddl.sql app/db app/models app/config.py requirements.txt .env.example docker-compose.langfuse.yml Makefile data/ch09/reports/.gitkeep tests/test_ch09_schema.py tests/test_config.py tests/test_ch04_schema.py
git commit -m "feat: add ch09 schema, config, and langfuse v3 stack"
```

---

## Task 2: Langfuse 观测核心与编译期回调

**Files:**
- Create: `app/observability.py`
- Modify: `app/api/ch06.py`
- Modify: `app/agent/service.py`
- Modify: `app/agent/graph.py`
- Test: `tests/test_observability.py`、`tests/test_agent_service.py`、`tests/test_ch06_api.py`

**Interfaces:**
- Produces:
  - `tracing_enabled(settings) -> bool`
  - `get_langfuse_client(settings)`
  - `get_callback_handler(settings)`
  - `attach_tracing(graph, settings)`
  - `_state_graph(graph)`
  - `update_intent_trace(settings, intent: str, confidence: float) -> None`
  - `flush_tracing()`
- Consumes: `Settings.langfuse_*`、`build_llm(stream_usage=True)`。

- [ ] **Step 1: Context7 核对 Langfuse v3**

Run a Context7 query for `langfuse==3.15.0`:
- `CallbackHandler` construction and `LANGFUSE_BASE_URL`/host.
- `update_current_trace(metadata=..., tags=...)` exact signature.
- `LangChain` callback config keys for `langfuse_session_id`/`langfuse_user_id`.
Record the exact signatures in the task notes before coding.

- [ ] **Step 2: 写失败测试**

```python
# tests/test_observability.py
from types import SimpleNamespace
from app.observability import tracing_enabled, attach_tracing, _state_graph

def test_tracing_disabled_without_keys():
    s = SimpleNamespace(langfuse_public_key="", langfuse_secret_key="", langfuse_base_url="")
    assert tracing_enabled(s) is False

def test_attach_tracing_noop_when_disabled():
    graph = object()
    assert attach_tracing(graph, _settings()) is graph

def test_state_graph_unwraps_binding():
    class Binding:
        bound = "graph"
    assert _state_graph(Binding()) == "graph"
    assert _state_graph("graph") == "graph"
```

Add a service test asserting `_thread_config(1, user_id="u1", settings=...)` includes `langfuse_session_id: "conv:1"` and `langfuse_user_id: "u1"`; no `callbacks` key when tracing disabled.

Add a graph test that `classify_intent_node` calls `update_intent_trace` when tracing enabled (monkeypatch the function) and does not raise when it raises.

- [ ] **Step 3: 运行测试确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_observability.py -q`
Expected: FAIL（模块不存在）。

- [ ] **Step 4: 实现**

- `app/observability.py`：
  - 惰性 import langfuse；进程级 `_client`/`_handler` 单例 + `threading.Lock`。
  - `tracing_enabled`: public/secret/base_url 全部非空。
  - `get_langfuse_client`: `Langfuse(public_key=..., secret_key=..., host=settings.langfuse_base_url)`。
  - `get_callback_handler`: `CallbackHandler()`（默认客户端）。
  - `attach_tracing`: enabled 时 `graph.with_config({"callbacks":[handler]})`；否则原样返回；任何异常记 warning 并返回原图。
  - `_state_graph`: `getattr(graph, "bound", graph)`。
  - `update_intent_trace`: enabled 时 `client.update_current_trace(metadata={"intent":..., "intent_confidence":...}, tags=[f"intent:{intent}"])`；异常 warning，不抛。
  - `flush_tracing`: client.flush()（异常吞掉）。
- `app/api/ch06.py::_build_and_cache_graph`：build 后 `graph = attach_tracing(graph, settings)`；存 `app.state.ch06_graph`。
- `app/api/ch06.py` 和 `app/agent/service.py` 所有 `graph.get_state/get_state_history` 改为 `_state_graph(graph)`。
- `app/agent/service.py::_thread_config(conversation_id, *, user_id=None, settings=None)`：`user_id` 与 `settings` 都提供且 tracing enabled 时才加 `metadata={"langfuse_session_id": f"conv:{cid}", "langfuse_user_id": user_id}`；`has_pending_interrupt` 继续可只传 conversation_id，不破坏现有签名。
- `app/agent/graph.py::classify_intent_node`：分类后调用 `update_intent_trace(settings, result.intent, result.confidence)`。
- `app/config.py::build_llm`：`ChatOpenAI(..., stream_usage=settings.stream_usage)`。
- `app/main.py` lifespan shutdown 调 `flush_tracing()`。

- [ ] **Step 5: 运行测试**

Run: `.venv/Scripts/python.exe -m pytest tests/test_observability.py tests/test_agent_service.py tests/test_ch06_api.py -q`
Expected: PASS；现有断言如需更新（config metadata 只在 tracing enabled 时出现）同步修。

- [ ] **Step 6: 提交**

```bash
git add app/observability.py app/api/ch06.py app/agent/service.py app/agent/graph.py app/config.py app/main.py tests/test_observability.py tests/test_agent_service.py tests/test_ch06_api.py
git commit -m "feat: attach langfuse v3 tracing to the compiled graph"
```

---

## Task 3: 意图成本脚本

**Files:**
- Create: `app/jobs/cost_by_intent.py`
- Create: `scripts/cost_by_intent.py`
- Modify: `Makefile`
- Test: `tests/test_cost_by_intent.py`

**Interfaces:**
- Produces: `app.jobs.cost_by_intent.main(argv=None) -> int`；`build_report(rows, window_days) -> dict`；`aggregate_by_intent(observations) -> list[dict]`。
- Consumes: Langfuse client/`Settings`；`ch09_report_dir`。

- [ ] **Step 1: Context7 核对 metrics API**

Query Langfuse v3 docs for:
- `api.metrics_v2.metrics` availability.
- `api.legacy.metrics_v1` observations view + tags/traceId grouping.
- measures `inputTokens/outputTokens/totalTokens`.
Record exact call names.

- [ ] **Step 2: 写失败测试**

```python
# tests/test_cost_by_intent.py
from app.jobs.cost_by_intent import aggregate_by_intent, build_report

def test_aggregate_by_intent_sums_tokens_and_computes_share():
    rows = [
        {"tags": ["intent:商品咨询"], "traceId": "t1", "totalTokens": 100},
        {"tags": ["intent:商品咨询"], "traceId": "t1", "totalTokens": 50},
        {"tags": ["intent:退款退货"], "traceId": "t2", "totalTokens": 50},
    ]
    out = aggregate_by_intent(rows)
    by = {r["intent"]: r for r in out}
    assert by["商品咨询"]["total_tokens"] == 150
    assert by["商品咨询"]["requests"] == 1  # unique traceId
    assert by["商品咨询"]["share"] == 0.75
    assert by["退款退货"]["avg_tokens"] == 50

def test_build_report_has_rows_and_window():
    report = build_report([{"intent": "闲聊", "requests": 1, "total_tokens": 10,
                            "avg_tokens": 10, "share": 1.0}], 7)
    assert report["window_days"] == 7
    assert report["rows"][0]["intent"] == "闲聊"
```

- [ ] **Step 3: 运行测试确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cost_by_intent.py -q`
Expected: FAIL（模块不存在）。

- [ ] **Step 4: 实现**

- `app/jobs/cost_by_intent.py`：
  - `--days`（默认 7）、`--json`。
  - `fetch_observations(client, settings, days)`：优先 `client.api.metrics_v2.metrics(query=...)`；捕获 404/400 后走 `client.api.legacy.metrics_v1`；再失败则拉 observations 列表。
  - 查询按 `tags` 过滤/分组 + `traceId`；客户端按 `intent:` tag 聚合；同 trace 去重计请求数。
  - 输出 `data/ch09/reports/cost_by_intent.txt` 和 `.json`；JSON 形状见 spec §5。
  - 打印表格：意图 / 请求数 / 总 tokens / 平均 tokens / 占比。
  - 不重算页面上已有的统计；脚本算完写产物。
- `scripts/cost_by_intent.py`：`from app.jobs.cost_by_intent import main; raise SystemExit(main())`。
- `Makefile` 加 `cost-report`。

- [ ] **Step 5: 运行测试 + 导入检查**

Run:
```bash
.venv/Scripts/python.exe -m pytest tests/test_cost_by_intent.py -q
.venv/Scripts/python.exe -c "import app.jobs.cost_by_intent"
```
Expected: PASS；导入不建客户端、不连网。

- [ ] **Step 6: 提交**

```bash
git add app/jobs/cost_by_intent.py scripts/cost_by_intent.py Makefile tests/test_cost_by_intent.py
git commit -m "feat: add intent token cost report"
```

---

## Task 4: 证据置信度纯函数、key clause、校准脚本

**Files:**
- Modify: `app/rag/retrieval.py`
- Modify: `app/agent/confidence.py`
- Create: `app/jobs/calibrate_confidence.py`
- Create: `scripts/calibrate_confidence.py`
- Modify: `Makefile`
- Test: `tests/test_agent_confidence.py`、`tests/test_retrieval.py`、`tests/test_ch09_calibration.py`

**Interfaces:**
- Produces:
  - `RetrievedChunk.is_key_clause: bool = False`
  - `EvidenceConfidence(score: float, signals: dict)`
  - `compute_evidence_confidence(hits, *, valid_floor=0.1, valid_cap=3, margin_scale=0.2) -> EvidenceConfidence`
  - `app.jobs.calibrate_confidence.select_threshold(scored: list[tuple[float,bool]]) -> dict`
  - `app.jobs.calibrate_confidence.main(argv=None) -> int`
- Consumes: `retrieve_hybrid`、`analyze`、`Settings.evidence_*`。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_agent_confidence.py
from app.agent.confidence import compute_evidence_confidence

def hit(score, key=False):
    return {"score": score, "is_key_clause": key}

def test_empty_is_zero():
    out = compute_evidence_confidence([])
    assert out.score == 0.0
    assert out.signals["valid_count"] == 0

def test_key_clause_adds_weight():
    plain = compute_evidence_confidence([hit(0.8), hit(0.5)])
    keyed = compute_evidence_confidence([hit(0.8, True), hit(0.5)])
    assert keyed.score > plain.score
    assert keyed.signals["key_clause_hit"] is True

def test_single_hit_margin_is_top1():
    out = compute_evidence_confidence([hit(0.7)])
    assert out.signals["margin"] == 0.7
```

```python
# tests/test_ch09_calibration.py
from app.jobs.calibrate_confidence import select_threshold

def test_youden_picks_separating_threshold():
    scored = [(0.9, True), (0.8, True), (0.2, False), (0.1, False)]
    out = select_threshold(scored)
    assert 0.2 < out["threshold"] <= 0.8
    assert out["youden_j"] == 1.0
```

Add retrieval test: `_as_chunk`/`retrieve_hybrid` backfills `is_key_clause` from a fake KnowledgeChunk row.

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_confidence.py tests/test_ch09_calibration.py -q`
Expected: FAIL（函数不存在）。

- [ ] **Step 3: 实现纯函数**

- `RetrievedChunk` 加 `is_key_clause: bool = False`；`_as_chunk` 传 `bool(row.is_key_clause)`；`CitationCollector.add` payload 加 `is_key_clause`（同时更新精确键断言）。
- `app/agent/confidence.py`：
  - 归一化 input（支持 `RetrievedChunk` 与 dict）。
  - `top1`/`top2`/`margin`（单条时 margin=top1）/`valid_count`/`key_clause_hit`。
  - 权重 `0.50/0.20/0.20/0.10`；score 归一到 0–1。
  - `signals` 含全部原始值与阈值常量。
  - 保留旧 `ConfidenceVerdict`/`assess_confidence` 兼容，或替换为包装；新闸门走 `compute_evidence_confidence`。

- [ ] **Step 4: 实现校准脚本**

- 读 `tests/eval/rag_cases.json`，跳过 `excluded`；正例 `answerable=true`，负例 `bucket=="D_unanswerable"`。
- 每条 `analyze(query)` 一次，`retrieve_hybrid(strategy=hybrid_rerank, analysis=..., top_k=10)`；排除 retrieval_error/rerank_failed。
- `select_threshold`: 扫 0.00–1.00 步长 0.01，最大化 Youden J；并列更高 precision，再并列更接近 0.5。
- 写 `data/ch09/reports/confidence_calibration.txt/.json`（分布、扫描曲线、推荐阈值、TPR/FPR/J、样本/排除数）。
- 打印推荐阈值；不自动改 config（由 live 验收任务回填）。
- `scripts/calibrate_confidence.py` 薄包装；`Makefile` 加 `calibrate-confidence`。

- [ ] **Step 5: 运行测试 + 导入**

Run:
```bash
.venv/Scripts/python.exe -m pytest tests/test_agent_confidence.py tests/test_ch09_calibration.py tests/test_retrieval.py -q
.venv/Scripts/python.exe -c "import app.jobs.calibrate_confidence"
```
Expected: PASS；导入不连网/连库。

- [ ] **Step 6: 提交**

```bash
git add app/rag/retrieval.py app/agent/confidence.py app/jobs/calibrate_confidence.py scripts/calibrate_confidence.py Makefile tests/test_agent_confidence.py tests/test_ch09_calibration.py tests/test_retrieval.py
git commit -m "feat: add evidence confidence signals and calibration"
```

---

## Task 5: 置信度闸、selfcheck 闸、快照与落池 source

**Files:**
- Modify: `app/agent/state.py`
- Modify: `app/agent/graph.py`
- Modify: `app/agent/service.py`
- Modify: `app/services/low_confidence.py`
- Modify: `app/services/chat.py`
- Test: `tests/test_agent_graph.py`、`tests/test_agent_service.py`、`tests/test_agent_refund_flow.py`、`tests/test_low_confidence.py`、`tests/test_ch07_service.py`

**Interfaces:**
- Produces:
  - State fields `evidence_confidence: float | None`、`evidence_signals: dict`、`fallback_source: str`、`retrieved_snapshot: list[dict]`。
  - `record_low_confidence(..., retrieved_chunks: list[dict] | None = None)`。
  - `self_check_gate` graph node.
- Consumes: `compute_evidence_confidence`、`assess_evidence`、`Settings.evidence_confidence_threshold`。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_agent_service.py
def test_gate_gap_records_retrieval_low_conf_with_snapshot(session):
    _run(session, final={
        "route": "knowledge", "gate_passed": False,
        "gate_reason": "low_confidence", "fallback_source": "retrieval_low_conf",
        "evidence_confidence": 0.21,
        "retrieved_snapshot": [{"question": "q", "answer": "a",
                                "rerank_score": 0.2, "section_path": "s",
                                "is_key_clause": False}],
        "answer": "兜底",
    })
    row = session.query(LowConfidenceQuestion).one()
    assert row.source == "retrieval_low_conf"
    assert row.retrieved_chunks[0]["rerank_score"] == 0.2
    assert "evidence_confidence" in row.reason

def test_selfcheck_gap_records_self_check(session):
    _run(session, final={
        "route": "knowledge", "gate_passed": False,
        "fallback_source": "self_check", "answer": "兜底",
    })
    row = session.query(LowConfidenceQuestion).one()
    assert row.source == "self_check"

@pytest.mark.parametrize("source", ["", "infra_error", "rerank_failed"])
def test_infra_does_not_pool(session, source):
    _run(session, final={"gate_passed": False, "fallback_source": source, "answer": "兜底"})
    assert session.query(LowConfidenceQuestion).count() == 0
```

Add a graph test:
- confidence fail → `fallback_source == "retrieval_low_conf"`.
- confidence pass + selfcheck `useful=False` → `fallback_source == "self_check"`, no Agent.
- confidence pass + selfcheck raises → `fallback_source == ""`, no pool, fallback text.
- normal pass stores `retrieved_snapshot` top3.

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_graph.py tests/test_agent_service.py tests/test_low_confidence.py -q`
Expected: FAIL（字段/节点/source 不存在）。

- [ ] **Step 3: 实现 state/置信度闸/快照**

- `app/agent/state.py` 加四个字段。
- `retrieve_knowledge_node` / `retrieve_policy_node`：
  - 调 `compute_evidence_confidence(result.chunks)`；
  - `retrieved_snapshot = top3`（question/answer/rerank_score/section_path/is_key_clause），无论闸是否通过；
  - 过阈值：`evidence_strong=True`，`fallback_source=""`；
  - 不过：`fallback_source="retrieval_low_conf"`，`gate_passed=False`；
  - rerank/infra 仍短路，`fallback_source=""`，不落池。
- `confidence_gate_node`：保留兜底话术；pass 时发 sources 帧并进入 selfcheck。

- [ ] **Step 4: 实现 selfcheck 节点**

- 新增 `self_check_gate_node(state, config)`：
  - 仅当 `gate_passed=True` 时调用 `assess_evidence(self_check_llm, state.user_message, state.sources)`。
  - `useful=False`：`fallback_source="self_check"`，`gate_passed=False`，答案 `KNOWLEDGE_GAP_TEXT`，emit 文本帧。
  - 异常：`fallback_source=""`，`gate_passed=False`，答案 `SELF_CHECK_FAILURE_TEXT`，不落池。
  - `useful=True`：保持 `gate_passed=True`，进入 `main_agent`。
- 图边：`confidence_gate` pass → `self_check_gate`；selfcheck pass → `main_agent`；fail → `record_log`。
- `build_graph` 加 `self_check_llm=None`，默认 `intent_llm`；API 构建时传 `intent_llm`。
- 更新 `record_log_node` payload：`evidence_confidence/evidence_signals/fallback_source/snapshot_count`。

- [ ] **Step 5: 落池**

- `record_low_confidence` 加 `retrieved_chunks=None`，写 ORM。
- `_finish_turn`：
  - 读 `fallback_source`；若在 `{"retrieval_low_conf","self_check"}`，`record_low_confidence(..., source=fallback_source, reason=..., retrieved_chunks=state.retrieved_snapshot)`。
  - 其它 source 不落池。
- `app/services/chat.py` 旧 `/chat` selfcheck：`record_low_confidence(..., source="self_check", retrieved_chunks=collector.payload()[:3])`。
- `_GATE_GAP_REASONS` 更新为基于 `fallback_source`，不再用 `gate_reason` 白名单。

- [ ] **Step 6: 运行测试**

Run:
```bash
.venv/Scripts/python.exe -m pytest tests/test_agent_graph.py tests/test_agent_service.py tests/test_agent_refund_flow.py tests/test_low_confidence.py tests/test_ch07_service.py -q
```
Expected: PASS；更新旧 `below_threshold`/`confidence_gate` 断言。

- [ ] **Step 7: 提交**

```bash
git add app/agent/state.py app/agent/graph.py app/agent/service.py app/services/low_confidence.py app/services/chat.py tests/test_agent_graph.py tests/test_agent_service.py tests/test_agent_refund_flow.py tests/test_low_confidence.py tests/test_ch07_service.py
git commit -m "feat: add evidence confidence and selfcheck gates with snapshots"
```

---

## Task 6: 飞轮批处理（标准化+查重+待审）

**Files:**
- Create: `app/services/review_repo.py`
- Create: `app/services/flywheel.py`
- Create: `app/prompts/flywheel.py`
- Create: `app/jobs/flywheel_pipeline.py`
- Create: `scripts/flywheel_pipeline.py`
- Create: `app/jobs/eval_flywheel_prompts.py`
- Create: `tests/eval/flywheel_samples.json`
- Modify: `Makefile`
- Test: `tests/test_flywheel.py`、`tests/test_eval_flywheel_prompts.py`

**Interfaces:**
- Produces:
  - `FlywheelOutput(normalized_question: str, matched_question_id: int | None, ai_suggested_answer: str)`
  - `normalize_and_match(llm, raw_question: str, candidates: list[ReviewQueue]) -> FlywheelOutput`
  - `process_pending(session, llm, limit: int = 50) -> FlywheelReport`
  - `FlywheelReport(scanned: int, created: int, merged: int, failed: int)`
  - `fetch_unmatched_low_conf(session, limit) -> list[LowConfidenceQuestion]`
  - `list_review_candidates(session, limit=200) -> list[ReviewQueue]`
  - `insert_review_item(session, output) -> ReviewQueue`
  - `increment_occurrence(session, review_id) -> ReviewQueue`
  - `set_matched_review(session, lcq_id, review_id) -> None`
- Consumes: `LowConfidenceQuestion`、`ReviewQueue`、`build_llm(temperature=0)`。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_flywheel.py
import pytest
from app.models import LowConfidenceQuestion, ReviewQueue
from app.services.flywheel import FlywheelOutput, process_pending

class FakeLLM:
    def __init__(self, output): self.output = output
    def with_structured_output(self, schema, method=None): return self
    def invoke(self, messages): return self.output

@pytest.fixture
def low_conf_row(session):
    row = LowConfidenceQuestion(raw_question="退款多久到账", source="self_check")
    session.add(row); session.commit(); session.refresh(row)
    return row

def test_new_question_creates_pending_review(session, low_conf_row):
    out = FlywheelOutput("退款多久到账", None, "1-3 个工作日")
    report = process_pending(session, FakeLLM(out))
    assert report.created == 1
    row = session.query(ReviewQueue).one()
    assert row.review_status == "待审"
    assert row.occurrence_count == 1
    assert low_conf_row.matched_review_id == row.id

def test_match_rejected_row_increments_without_revival(session, low_conf_row):
    review = ReviewQueue(normalized_question="退款多久到账",
                         review_status="驳回", occurrence_count=2)
    session.add(review); session.commit()
    out = FlywheelOutput("退款多久到账", review.id, "")
    report = process_pending(session, FakeLLM(out))
    assert report.merged == 1
    assert review.review_status == "驳回"
    assert review.occurrence_count == 3
    assert low_conf_row.matched_review_id == review.id

def test_hallucinated_id_is_skipped(session, low_conf_row):
    out = FlywheelOutput("新问题", 999999, "")
    report = process_pending(session, FakeLLM(out))
    assert report.failed == 1
    assert low_conf_row.matched_review_id is None

def test_serial_same_batch_second_matches_first(session, monkeypatch):
    for q in ("q1", "q2"):
        session.add(LowConfidenceQuestion(raw_question=q, source="self_check"))
    session.commit()
    calls = []

    def fake_normalize(llm, raw_question, candidates):
        calls.append([(c.id, c.normalized_question) for c in candidates])
        if len(calls) == 1:
            return FlywheelOutput("标准问题1", None, "")
        return FlywheelOutput("标准问题2", candidates[0].id, "")

    monkeypatch.setattr("app.services.flywheel.normalize_and_match", fake_normalize)
    report = process_pending(session, object())
    assert (report.created, report.merged) == (1, 1)
    assert calls[1]  # 第二条看到了第一条刚建的行
```

Add `tests/eval/flywheel_samples.json` 10–15 cases; `app/jobs/eval_flywheel_prompts.py` runs real LLM and exits 1 below 80%.

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_flywheel.py -q`
Expected: FAIL（模块不存在）。

- [ ] **Step 3: 实现 repo 与 prompt**

- `review_repo.py`：
  - `list_unmatched_low_confidence`、`list_review_candidates`（全部状态，`updated_at DESC`，limit 200）、`insert_review_item`（`normalized_question` 截断 512、`ai_suggested_answer` 可空）、`increment_occurrence`（同 transaction，`occurrence_count += 1`，状态不动）、`set_matched_review`。
- `app/prompts/flywheel.py`：
  - `FLYWHEEL_NORMALIZE_PROMPT`：输入原话 + 候选 `id/问题/次数/状态`；输出 JSON `{normalized_question, matched_question_id, ai_suggested_answer}`；保留实体；示例答案只备查；JSON only。
- `flywheel.py`：
  - `normalize_and_match` 用 `with_structured_output(FlywheelOutput, method="json_mode")`。
  - 校验 `matched_question_id` 在候选 id 集合内；越界/坏 JSON/异常 → 抛 `FlywheelError` 或返回 None；`process_pending` 捕获后 `failed += 1`，不写 matched。
  - `process_pending` 串行；每条形如：normalize → 命中/新建 → 回写 matched → commit；失败 rollback 当前条目。
  - 候选每次重新查询（同批新建行能被后续命中）。

- [ ] **Step 4: 实现 job/script/samples**

- `app/jobs/flywheel_pipeline.py`：`--limit`（默认 50）；建 LLM `build_llm(temperature=0.0, max_tokens=settings.flywheel_llm_max_tokens)`；打印报告；退出码 0/1。
- `scripts/flywheel_pipeline.py` 薄包装；`Makefile` 加 `flywheel`。
- `tests/eval/flywheel_samples.json`：10–15 条；`app/jobs/eval_flywheel_prompts.py` 真实 LLM 跑样例，命中/不命中/坏 JSON 统计，阈值 ≥80%；导入不建 LLM。

- [ ] **Step 5: 运行测试**

Run:
```bash
.venv/Scripts/python.exe -m pytest tests/test_flywheel.py tests/test_eval_flywheel_prompts.py -q
.venv/Scripts/python.exe -c "import app.jobs.flywheel_pipeline"
```
Expected: PASS；导入不连网。

- [ ] **Step 6: 提交**

```bash
git add app/services/review_repo.py app/services/flywheel.py app/prompts/flywheel.py app/jobs/flywheel_pipeline.py app/jobs/eval_flywheel_prompts.py scripts/flywheel_pipeline.py tests/eval/flywheel_samples.json Makefile tests/test_flywheel.py tests/test_eval_flywheel_prompts.py
git commit -m "feat: add batch flywheel normalization and dedup"
```

---

## Task 7: 审核 API 与知识库写回

**Files:**
- Create: `app/services/knowledge_approval.py`
- Create: `app/api/review.py`
- Modify: `app/main.py`
- Test: `tests/test_review_api.py`

**Interfaces:**
- Produces:
  - `approve_review(session, review_id, approved_answer, *, embedder=None, store=None, settings=None) -> ReviewQueue`
  - `reject_review(session, review_id) -> ReviewQueue`
  - REST: `GET /api/review/queue`, `GET /api/review/{id}`, `POST /api/review/{id}/approve`, `POST /api/review/{id}/reject`; page `/review`.
- Consumes: `insert_mined_qa`、`vectorize_pending`、`ReviewQueue`。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_review_api.py
def test_approve_writes_chunk_then_marks_approved(session, monkeypatch):
    review = ReviewQueue(normalized_question="退款多久到账", review_status="待审")
    session.add(review); session.commit()
    calls = []
    monkeypatch.setattr("app.services.knowledge_approval._write_and_vectorize",
                        lambda *a, **k: calls.append("write"))
    out = approve_review(session, review.id, "1-3 个工作日")
    assert calls == ["write"]
    assert out.review_status == "通过"
    assert out.approved_answer == "1-3 个工作日"

def test_approve_vectorize_failure_keeps_pending(session, monkeypatch):
    review = ReviewQueue(normalized_question="退款多久到账", review_status="待审")
    session.add(review); session.commit()

    def fail(*args, **kwargs):
        raise RuntimeError("milvus down")

    monkeypatch.setattr("app.services.knowledge_approval._write_and_vectorize", fail)
    with pytest.raises(RuntimeError):
        approve_review(session, review.id, "答案")
    assert session.get(ReviewQueue, review.id).review_status == "待审"

def test_approve_non_pending_returns_409(client, session):
    review = ReviewQueue(normalized_question="已处理", review_status="通过")
    session.add(review); session.commit()
    resp = client.post(f"/api/review/{review.id}/approve",
                       json={"approved_answer": "答案"})
    assert resp.status_code == 409
```

Add API tests for list/detail/reject/404.

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_review_api.py -q`
Expected: FAIL。

- [ ] **Step 3: 实现写回**

- `knowledge_approval.py`：
  - `_write_and_vectorize(session, review, approved_answer, embedder, store)`：`insert_mined_qa(session, [MinedQA(question=..., answer=..., category="通用")])`；`vectorize_pending(session, embedder, store)`；返回 chunk。
  - `approve_review`：仅待审；空答案 422；调用写回；成功置 `通过`/`approved_answer`；失败 rollback 状态、抛错。
  - `reject_review`：仅待审；置驳回。
  - 模块锁 + `kb_build.is_running()` 防并发。
- `app/api/review.py`：四个 JSON 接口 + `/review` 页面路由。
- `app/main.py` include routers。

- [ ] **Step 4: 运行测试**

Run: `.venv/Scripts/python.exe -m pytest tests/test_review_api.py tests/test_kb_api.py -q`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add app/services/knowledge_approval.py app/api/review.py app/main.py tests/test_review_api.py
git commit -m "feat: add review queue API and knowledge writeback"
```

---

## Task 8: 用户反馈 API

**Files:**
- Create: `app/api/feedback.py`
- Modify: `app/main.py`
- Test: `tests/test_feedback_api.py`

**Interfaces:**
- Produces:
  - `FeedbackRequest(session_id, conversation_id, rating: Literal["up","down"], question: str)`
  - `POST /api/feedback`
  - `record_feedback(session, graph, req) -> dict`
- Consumes: `graph.get_state` via `_state_graph`；`record_low_confidence`。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_feedback_api.py
from types import SimpleNamespace
from app.services.history import get_or_create_conversation

def _graph(user_message="q", snapshot=None):
    return SimpleNamespace(get_state=lambda config: SimpleNamespace(values={
        "user_message": user_message,
        "retrieved_snapshot": snapshot or [],
    }))

def test_up_only_logs_and_does_not_write(session, caplog):
    cid = get_or_create_conversation(session, "u1").id
    out = record_feedback(session, _graph(), FeedbackRequest(
        session_id="u1", conversation_id=cid, rating="up", question="q"))
    assert out["status"] == "ignored"
    assert session.query(LowConfidenceQuestion).count() == 0
    assert "feedback up" in caplog.text

def test_down_writes_user_feedback_with_latest_snapshot(session):
    cid = get_or_create_conversation(session, "u1").id
    snap = [{"question": "支持以旧换新吗", "answer": "不支持", "rerank_score": 0.4,
             "section_path": "售后 > 换货", "is_key_clause": False}]
    out = record_feedback(session, _graph("支持以旧换新吗", snap), FeedbackRequest(
        session_id="u1", conversation_id=cid, rating="down",
        question="支持以旧换新吗"))
    row = session.query(LowConfidenceQuestion).one()
    assert out["status"] == "recorded"
    assert row.source == "user_feedback"
    assert row.retrieved_chunks[0]["section_path"] == "售后 > 换货"

def test_down_question_mismatch_stores_null_snapshot(session):
    cid = get_or_create_conversation(session, "u1").id
    snap = [{"question": "别的", "answer": "a", "rerank_score": 0.9,
             "section_path": "s", "is_key_clause": False}]
    record_feedback(session, _graph("另一轮", snap), FeedbackRequest(
        session_id="u1", conversation_id=cid, rating="down", question="本轮问题"))
    row = session.query(LowConfidenceQuestion).one()
    assert row.retrieved_chunks is None
```

API test for ownership:
```python
def test_wrong_owner_404(client, session):
    cid = get_or_create_conversation(session, "owner").id
    resp = client.post("/api/feedback", json={
        "session_id": "intruder", "conversation_id": cid,
        "rating": "down", "question": "q"})
    assert resp.status_code == 404
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_feedback_api.py -q`
Expected: FAIL。

- [ ] **Step 3: 实现**

- Pydantic request；校验 conversation 属于 session_id；rating down 落池，up 只 `logger.info("feedback up ...")`。
- 新增 `get_feedback_graph_dep`：尝试复用 `get_ch06_graph_dep`，构建失败返回 `None`；反馈不能因图不可用而 500。
- `_state_graph(graph).get_state({"configurable":{"thread_id":f"conv:{cid}"}})`；`values.user_message == question` 且 `retrieved_snapshot` 非空 → Top3；否则 None。
- 读 state 异常/图为 None → None，仍落池。
- `app/main.py` include router。

- [ ] **Step 4: 运行测试**

Run: `.venv/Scripts/python.exe -m pytest tests/test_feedback_api.py tests/test_ch07_service.py -q`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add app/api/feedback.py app/main.py tests/test_feedback_api.py
git commit -m "feat: record user down feedback into the low-confidence pool"
```

---

## Task 9: 评估流水线与 eval_runs 趋势

**Files:**
- Create: `app/jobs/eval_flywheel.py`
- Create: `scripts/eval_flywheel.py`
- Modify: `Makefile`
- Test: `tests/test_eval_flywheel.py`

**Interfaces:**
- Produces:
  - `run_once(triggered_by: Literal["手动","定时"], limit: int | None = None, *, runner=None) -> dict`
  - `main(argv=None) -> int`
  - `load_metrics_from_report(path: Path) -> dict`
  - `format_trend(runs: list[EvalRun]) -> str`
- Consumes: `eval_rag.main`；`EvalRun`。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_eval_flywheel.py
def test_load_metrics_from_report(tmp_path):
    report = {"cases_loaded": 300, "overall": {"hybrid_rerank": {
        "retrieval": {"recall_at_10": 0.9, "mrr_at_10": 0.8},
        "generation": {"faithfulness": 0.95},
        "d_accuracy": {"accuracy": 0.97}}}}
    path = tmp_path / "r.json"; path.write_text(json.dumps(report))
    m = load_metrics_from_report(path)
    assert m["recall_at_10"] == 0.9
    assert m["mrr_at_10"] == 0.8
    assert m["faithfulness"] == 0.95
    assert m["d_refusal_accuracy"] == 0.97

def test_run_once_inserts_eval_run(session, tmp_path):
    report = {"cases_loaded": 300, "overall": {"hybrid_rerank": {
        "retrieval": {"recall_at_10": 0.9, "mrr_at_10": 0.8},
        "generation": {"faithfulness": 0.95},
        "d_accuracy": {"accuracy": 0.97}}}}

    def fake_runner(argv):
        out = Path(argv[argv.index("--out") + 1])
        out.write_text("md", encoding="utf-8")
        out.with_suffix(".json").write_text(json.dumps(report), encoding="utf-8")
        return 0

    out = run_once("手动", runner=fake_runner)
    assert out["ok"] is True
    row = session.query(EvalRun).one()
    assert row.triggered_by == "手动"
    assert row.metrics["recall_at_10"] == 0.9

def test_failed_runner_does_not_insert(session):
    out = run_once("手动", runner=lambda argv: 2)
    assert out["ok"] is False
    assert session.query(EvalRun).count() == 0
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_eval_flywheel.py -q`
Expected: FAIL。

- [ ] **Step 3: 实现**

- `run_once` 调 `eval_rag.main(["--cases","tests/eval/rag_cases.json","--strategies","hybrid_rerank","--out", str(report_path), *limit])`；runner 可注入。
- 读 sidecar；提取 `overall.hybrid_rerank` 的 Recall@10/MRR/Faithfulness/D accuracy；`dataset_size=cases_loaded`。
- 插入 `EvalRun`；commit；返回 `{"run_id":..., "metrics":...}`。
- `format_trend` 读最近 10 轮，比较相邻；上涨 ↑、下跌 ↓（⚠）、持平 →。
- 写 `data/ch09/reports/eval_trend.txt`；打印同样内容。
- `scripts/eval_flywheel.py` 薄包装；`Makefile` 加 `eval-flywheel`。
- `--triggered-by` 默认 `手动`。

- [ ] **Step 4: 运行测试**

Run: `.venv/Scripts/python.exe -m pytest tests/test_eval_flywheel.py tests/test_eval_rag.py -q`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add app/jobs/eval_flywheel.py scripts/eval_flywheel.py Makefile tests/test_eval_flywheel.py
git commit -m "feat: persist eval runs and print metric trends"
```

---

## Task 10: 观测/后台 API 与白名单作业运行器

**Files:**
- Create: `app/services/admin_jobs.py`
- Create: `app/api/admin_jobs.py`
- Create: `app/api/observability.py`
- Create: `app/api/admin.py`
- Modify: `app/main.py`
- Test: `tests/test_admin_jobs.py`、`tests/test_observability_api.py`

**Interfaces:**
- Produces:
  - `ALLOWED_JOBS = {"cost-report","eval-flywheel","calibrate-confidence"}`
  - `start_job(name) -> dict | None`
  - `job_status(name) -> dict`
  - `POST /api/admin/jobs/{name}/run`、`GET /api/admin/jobs/{name}/status`
  - `GET /api/observability/overview`
  - `GET /api/admin/summary`
  - Pages `/observability`、`/admin`
- Consumes: cost JSON、calibration JSON、`EvalRun`、`ReviewQueue`、job mains。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_admin_jobs.py
def test_unknown_job_is_rejected():
    assert start_job("rm -rf /") is None

def test_same_job_is_mutex(monkeypatch):
    gate = threading.Event()
    monkeypatch.setitem(admin_jobs.ALLOWED_JOBS, "cost-report",
                        lambda: gate.wait(1))
    first = start_job("cost-report")
    second = start_job("cost-report")
    assert first is not None
    assert second is None
    gate.set()

def test_job_output_tail_is_captured(monkeypatch):
    monkeypatch.setitem(admin_jobs.ALLOWED_JOBS, "cost-report",
                        lambda: print("cost report done"))
    start_job("cost-report")
    deadline = time.time() + 2
    while time.time() < deadline:
        state = job_status("cost-report")
        if not state["running"]:
            break
        time.sleep(0.01)
    assert "cost report done" in job_status("cost-report")["output_tail"]
```

```python
# tests/test_observability_api.py
def test_overview_blocks_are_independent(client, session, tmp_path):
    # cost file missing, calibration file missing, eval_runs present
    data = client.get("/api/observability/overview").json()
    assert data["cost"]["present"] is False
    assert data["eval"]["present"] is True
    assert "error" in data["calibration"] or data["calibration"]["present"] is False
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_admin_jobs.py tests/test_observability_api.py -q`
Expected: FAIL。

- [ ] **Step 3: 实现作业运行器**

- `admin_jobs.py`：白名单映射到 callable；`start_job` 线程 + 锁；捕获 stdout/stderr 到 `output_tail`（截 4000 字符）；状态 `running/started_at/finished_at/exit_code/output_tail/last_error`；未知作业返回 None。
- `api/admin_jobs.py`：404 未知、409 运行中、202 启动、status 查询。
- 作业函数调用 `app.jobs.*.main()`，不拼 shell。

- [ ] **Step 4: 实现 overview/admin API**

- `overview`：
  - `cost`: 读 `data/ch09/reports/cost_by_intent.json`；缺失 `{present:false, hint:...}`；读取失败 `{present:false, error:...}`。
  - `eval`: 读 `EvalRun` 最近 10 轮；DB 失败只影响本块。
  - `calibration`: 读 `confidence_calibration.json` + `settings.evidence_confidence_threshold`；若推荐值 != 在用值，返回 `needs_update=true`。
  - 三块独立 try/except。
- `admin/summary`：counts by `review_status` + highest occurrence preview + nav links.
- 页面路由 `/observability`、`/admin`；`/review` 页面已在 Task 7。
- main include routers.

- [ ] **Step 5: 运行测试**

Run: `.venv/Scripts/python.exe -m pytest tests/test_admin_jobs.py tests/test_observability_api.py tests/test_review_api.py -q`
Expected: PASS。

- [ ] **Step 6: 提交**

```bash
git add app/services/admin_jobs.py app/api/admin_jobs.py app/api/observability.py app/api/admin.py app/main.py tests/test_admin_jobs.py tests/test_observability_api.py
git commit -m "feat: add observability/admin APIs and whitelisted job runner"
```

---

## Task 11: 前端（Vibe Coding）

**Files:**
- Modify: `app/static/ch07.html`
- Create: `app/static/review.html`
- Create: `app/static/observability.html`
- Create: `app/static/admin.html`
- Create: `app/static/admin.js`
- Modify: `app/static/kb.html`、`app/static/rag_eval.html`（加共用导航）
- Modify: `app/api/review.py`/`app/api/observability.py`/`app/api/admin.py`（页面路由如未加）

**Interfaces:**
- Consumes: `/api/feedback`、`/api/review/*`、`/api/observability/overview`、`/api/admin/summary`、`/api/admin/jobs/*/status`。
- Produces: 四个后台页 + ch07 反馈条。

- [ ] **Step 1: ch07 反馈条**

- 发送时记录 `ctx.userQuestion`；resume 轮用 `lastUserMessage`。
- `finishAssistantMessage` 在非空、非 error、非 interrupted 回答下加 👍/👎。
- 点击锁定、显示“已反馈”；👍/👎 都 POST `/api/feedback`（带 `session_id/conversation_id/rating/question`）；失败提示重试。
- 复用现有 CSS 变量；不引依赖。

- [ ] **Step 2: `/review` 页**

- 列表默认待审、按 occurrence_count 降序；状态筛选。
- 通过弹框预填 `ai_suggested_answer`；驳回按钮。
- 行详情展开原话 + 快照卡（section_path/rerank_score/answer 摘要）。
- 页顶三道闸提示。
- 所有文本用 `textContent`，不用 innerHTML。

- [ ] **Step 3: `/observability` 页**

- 三块：成本表、评估趋势（↑↓/⚠）、校准推荐 vs 在用阈值。
- 重跑按钮：成本可直接跑；评估/校准二次确认；轮询 status，显示 output_tail。
- 缺产物显示提示，不假装有数。

- [ ] **Step 4: `/admin` 页与共用导航**

- `admin.js` `mountAdminNav()`：导航到 `/admin`、`/review`、`/observability`、`/kb`、`/rag-eval`。
- `/admin` 三张计数卡 + 结论 + 入口。
- 给 `/kb`、`/rag-eval` 加导航挂载。

- [ ] **Step 5: 静态验证**

Run:
```bash
.venv/Scripts/python.exe -m pytest tests/test_review_api.py tests/test_observability_api.py -q
.venv/Scripts/python.exe -c "from app.main import app; print([r.path for r in app.routes if r.path in {'/admin','/review','/observability'}])"
```
Expected: 页面路由存在；API 测试绿。
Then run `python run.py` and manually check in browser; record screenshots/notes in dev-notes.

- [ ] **Step 6: 提交**

```bash
git add app/static/ch07.html app/static/review.html app/static/observability.html app/static/admin.html app/static/admin.js app/static/kb.html app/static/rag_eval.html
git commit -m "feat: add ch09 admin, review, observability, and feedback UI"
```

---

## Task 12: Makefile/scripts 补齐与本地文档

**Files:**
- Modify: `Makefile`
- Create: `scripts/calibrate_confidence.py`、`scripts/cost_by_intent.py`、`scripts/flywheel_pipeline.py`、`scripts/eval_flywheel.py`（如任务中未建）
- Modify: `README.md`（若存在）或新增 `docs/ch09-operations.md`
- Test: none（命令级检查）

- [ ] **Step 1: 补齐 Makefile 目标**

```make
PYTHON ?= .venv/Scripts/python.exe
langfuse-up:
	docker compose -f docker-compose.langfuse.yml up -d
langfuse-down:
	docker compose -f docker-compose.langfuse.yml down
flywheel:
	$(PYTHON) -m app.jobs.flywheel_pipeline
cost-report:
	$(PYTHON) -m app.jobs.cost_by_intent --days 7
eval-flywheel:
	$(PYTHON) -m app.jobs.eval_flywheel --triggered-by 手动
calibrate-confidence:
	$(PYTHON) -m app.jobs.calibrate_confidence
test:
	$(PYTHON) -m pytest -q
```

- [ ] **Step 2: 补脚本包装**

每个 `scripts/*.py` 只做：
```python
from app.jobs.<module> import main
raise SystemExit(main())
```

- [ ] **Step 3: 写操作文档**

- Langfuse 首次启动/建 key/环境变量。
- `make flywheel` cron 示例。
- 校准后回填 `app/config.py`。
- 报表路径。
- 单进程限制。

- [ ] **Step 4: 检查命令**

Run:
```bash
make -n langfuse-up flywheel cost-report eval-flywheel calibrate-confidence test
.venv/Scripts/python.exe -c "import scripts.cost_by_intent" 2>/dev/null || true
```
Expected: 目标展开正确；脚本可导入（不执行 main）。

- [ ] **Step 5: 提交**

```bash
git add Makefile scripts docs/ch09-operations.md
git commit -m "chore: add ch09 make targets, script wrappers, and operations notes"
```

---

## Task 13: live 校准与七条验收

**Files:**
- Modify: `app/config.py`（回填校准阈值默认值）
- Modify: `.env.example`（同步阈值）
- Modify: `dev-notes/ch09.md`（阶段记录）
- Test: live commands + `pytest -q`

**Interfaces:**
- Consumes: 全部前序任务。

- [ ] **Step 1: 启动基础设施**

```bash
docker compose up -d
make langfuse-up
# 等 Langfuse web healthy；确认 3000/3306/19530 可达
.venv/Scripts/python.exe -m app.db.init_ch09
```

- [ ] **Step 2: live 校准**

```bash
make calibrate-confidence
```
Expected: `data/ch09/reports/confidence_calibration.json` 有推荐阈值与 Youden J；把推荐值写进 `app/config.py::evidence_confidence_threshold` 默认值与 `.env.example`，再跑一次聚焦测试确认配置断言。

- [ ] **Step 3: 启动服务并验收 1–7**

```bash
python run.py
```
按 spec §17 逐条执行：
1. Langfuse UI 点任意 trace（记录 trace id/截图）。
2. 问知识库没有的问题 → 兜底 → `make flywheel` → `/review` 详情有原话+快照。
3. `/review` 通过 → 再问 → 答对；确认 `knowledge_chunks` 新行 + Milvus 向量。
4. ch07 点 👎 → `make flywheel` → `/review` 出现；再点 👎 不应强重复。
5. `make cost-report` → 表格；`/observability` 同源。
6. `make eval-flywheel` 两轮 → `eval_runs` 两行；`/observability` 趋势 ↑↓/⚠。
7. `/observability` 三块可看；点“重跑 意图成本账”现场跑一轮，页面看到 output_tail。

- [ ] **Step 4: 全量回归**

```bash
.venv/Scripts/python.exe -m pytest -q
```
Expected: 全绿（基线 772 + 新增）。

- [ ] **Step 5: 记录 evidence**

把命令、输出摘要、截图路径、MySQL 行、Langfuse trace id 写进 `dev-notes/ch09.md` 的验收阶段。

- [ ] **Step 6: 提交**

```bash
git add app/config.py .env.example dev-notes/ch09.md data/ch09/reports docs/superpowers/reports
git commit -m "test: record ch09 live calibration and acceptance"
```

---

## Task 14: Code Review、修复与 Finish

**Files:**
- Modify: 按 review 发现
- Modify: `dev-notes/ch09.md`

- [ ] **Step 1: 请求 code review**

Use `superpowers:requesting-code-review` on the branch diff. Focus:
- `with_config` + `_state_graph` 是否覆盖所有 state 读取。
- flywheel 全量查重/状态不变/事务边界。
- selfcheck infra 不落池。
- approve 向量化顺序/幂等。
- feedback 快照 turn 匹配/归属校验。
- 现有 772 测试回归。

- [ ] **Step 2: 处理 review 发现**

- 逐条验证；修复后跑聚焦 + 全量。
- 不接受“为了过评审放宽安全约束”。

- [ ] **Step 3: 定点复评**

Review 修复后请同一 reviewer 复评 Important/Critical；记录结论。

- [ ] **Step 4: Finish**

Use `superpowers:finishing-a-development-branch`；按用户选择 PR 或本地合并。
- 交付：功能演示命令、测试结果、dev-notes 路径、PR 链接（若选 PR）。
- dev-notes 追加 finish 阶段。

- [ ] **Step 5: 最终提交**

```bash
git add -A
git status --short
git commit -m "chore: finish ch09 observability and data flywheel"
```

---

## Self-Review Notes

- Spec §4/§5/§6/§7/§8/§9/§10/§11/§12/§17 均有任务：Task 2/3/4/5/6/7/8/9/10/11/13。
- 路径已映射到当前仓库；没有新建 `app/core`/`app/kb`/`app/graph` 平行结构。
- 类型一致性：`RetrievedChunk.is_key_clause`、`EvidenceConfidence.signals`、`FlywheelOutput`、`ReviewQueue`、`EvalRun`、`_state_graph` 在后续任务中保持同名。
- Review Focus 5 条均落到 owning task 的测试：Task 2（state 读取）、Task 2/13（并发）、Task 6（全量查重）、Task 5（selfcheck infra）、Task 7（approve 顺序）。
- 前端任务按用户要求走 Vibe Coding，不设 TDD；其余任务均有失败测试步骤。
