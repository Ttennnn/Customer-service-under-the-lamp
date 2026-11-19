# Ch09 设计（对齐课程参考 spec）：可观测性与数据飞轮

- 状态：待批准
- 日期：2026-11-19
- 分支：`feature/ch09-observability-flywheel`（从 `feature/ch08-tool-system` 切出）
- 参考：用户 2026-11-19 提供的《Ch09 设计：可观测性与数据飞轮》课程 spec
- 技术栈定死：Langfuse 开源自部署 **v3**、FastAPI、SQLAlchemy/MySQL、LangGraph 1.2.20、Milvus、ch04 评估集

> 这份文档是对参考 spec 的“当前仓库对齐版”。参考 spec 里的 `app/core`、`app/graph`、`app/kb`、`scripts/eval_ch04.py`、`tests/data/eval_ch04.jsonl` 等路径在本仓库不存在，统一映射到现有结构（见 §0.2）。参考 spec 中与上一轮口头选择冲突的地方，以参考 spec 为准：**Langfuse v3、编译期挂回调、飞轮批处理、全量查重、selfcheck 进图、后台报表页**。

---

## 0. 对齐说明

### 0.1 本版覆盖的上一轮选择

| 主题 | 上一轮口头选择 | 本版（参考 spec） |
|---|---|---|
| Langfuse | v4 + 统一调用入口 | **v3 + 编译后 `with_config` 挂一次 CallbackHandler** |
| 飞轮触发 | 落池后同步处理 | **定时批处理脚本；验收手动 `make flywheel`** |
| 查重范围 | 只比对待审 | **和 `review_queue` 全部行比对；命中已驳回/已通过只累加次数，状态不变** |
| 反馈入池 | SSE 发 assistant message id | **前端把该轮用户原话 `question` 发给后端；后端用 checkpointer 最近 state 回捞快照** |
| selfcheck | 只接旧 `/chat` 的现有落池 | **主图知识/退款路径在置信度闸后增加 selfcheck 闸** |
| 前端 | 只做 `/review` | **`/review` + `/observability` + `/admin` + 共用后台导航；ch07 聊天页 👎 接后端** |

### 0.2 路径映射

| 参考 spec | 当前仓库 |
|---|---|
| `app/core/confidence.py` | `app/agent/confidence.py` |
| `app/core/llm.py` / `get_chat_model()` | `app/config.py::build_llm` |
| `app/graph/state.py` | `app/agent/state.py` |
| `app/graph/nodes.py` | `app/agent/graph.py`、`app/agent/subgraph.py` |
| `app/db/models.py` | `app/models/` |
| `app/db/repository.py` | 现有 `app/services/` + 新增 `app/services/review_repo.py`/`flywheel.py` |
| `app/kb/dualwrite.py` | `app/rag/ingest.py::insert_mined_qa` + `app/rag/vectorize.py::vectorize_pending` |
| `app/core/jobs.py` | 新增 `app/services/admin_jobs.py`（白名单作业运行器） |
| `app/api/review.py` | 新增 `app/api/review.py` |
| `app/api/feedback.py` | 新增 `app/api/feedback.py` |
| `app/api/observability.py` | 新增 `app/api/observability.py` |
| `app/static/index.html` | `app/static/ch07.html`（当前 `/` 实际服务的聊天页） |
| `scripts/*.py` | `app/jobs/*.py` + 薄 `scripts/*.py` 包装 |
| `tests/data/eval_ch04.jsonl` | `tests/eval/rag_cases.json`（300 条） |
| `data/ch09/reports/` | 同路径，新建 |

---

## 1. 目标与边界

### 1.1 目标（需求 6 条）

1. **接 Langfuse**：环境变量齐全时，图编译后挂一次 `CallbackHandler`；每条请求的完整 trace（节点 prompt、工具调用、检索结果、token、耗时）在 Langfuse 界面可看。
2. **Cost Control**：意图写进 trace metadata/tag；脚本按 `intent:xxx` 聚合 token 账。
3. **数据飞轮三入口**：置信度低、selfcheck 不够答、用户 👎，都进 `low_confidence_questions`，各标 `source`；落池带 `retrieved_chunks` 快照。
4. **飞轮流水线**：批处理扫未归并行 → 一次 LLM 输出标准化问题 + 匹配 id + 示例答案 → 命中累加 `occurrence_count`，未命中新建 `review_queue`；人工审核通过后写回知识库。
5. **自动化评估**：复用 ch04 评估集与指标，每轮写 `eval_runs`，脚本/页面看趋势。
6. **前端配套**：`/review` 待审后台、`/observability` 三张报表页、`/admin` 后台首页、ch07 聊天页 👎 接后端。

### 1.2 非目标

- 不做低置信度问题主题分类微调（下一章）。
- 不接 LangSmith。
- 不在 Langfuse 里配自定义模型单价；统计以 token 数为准，成本字段有则展示。
- 不引 APScheduler 等进程内调度框架；飞轮/评估是脚本 + Makefile + cron 示例。
- 👍 不落库，只记应用日志。
- 不改变 MCP 工具内部行为。

### 1.3 反目标

- 伪造阈值校准或 live 验收数字。
- 飞轮失败丢问题；聊天因为 Langfuse/飞轮失败而报错。
- 把 infra/rerank 失败当知识缺口落池。
- 自动通过审核或自动改知识库。

---

## 2. 现状与硬约束

- 当前工作树在 `feature/ch08-tool-system`，测试基线 **772 passed**。
- 主聊天页：`app/static/ch07.html` → `POST /ch07/chat` → `app/agent/service.py::stream_chat_agent` → `graph.stream(...)`。
- 旧聊天：`POST /chat` → `app/services/chat.py::stream_chat`，不走图，已有 ch04 selfcheck。
- 图懒加载：`app/api/ch06.py::_build_and_cache_graph` → `app/agent/graph.py::build_graph` → `graph.compile(checkpointer=...)`。
- 当前置信度闸：`app/agent/confidence.py::assess_confidence`，只判 Top1 >= 0.1；位置在 `retrieve_knowledge/retrieve_policy → confidence_gate → main_agent`。
- 当前主图**没有** selfcheck 节点；selfcheck 只在旧 `/chat`。
- 当前 `low_confidence_questions` 无快照/归并列；当前 `review_queue`/`eval_runs` 不存在。
- 当前 ch07 页没有 👍/👎。
- Docker/MySQL/Milvus/Langfuse 当前未启动；阈值校准和 live 验收必须等服务恢复。

### 2.1 关键技术约束

1. **`compile()` 没有 callbacks 参数**。参考 spec 的写法是 `compile().with_config({"callbacks":[CallbackHandler()]})`，可行，但返回 `RunnableBinding`，**没有 `get_state`/`get_state_history`**。本仓库服务层需要这两个方法，所以增加一个极薄的 `_state_graph(graph)`：有 `bound` 就取 `graph.bound`，否则取 `graph`，仅用于读 state/history；`stream/invoke` 走带 callback 的 binding。
2. **Langfuse v3**：SDK 钉 `langfuse==3.15.0`；服务端镜像 `langfuse/langfuse:3` / `langfuse-worker:3`。v4 API 不适用。
3. **CallbackHandler 编译期共享**：一个 handler 服务所有请求。必须用真实并发/多轮请求验证 trace 不串；若实测串 trace，退回每次 `graph.stream` config 注入（记录为返工，不静默改）。
4. **selfcheck 需要模型**：主图新增 selfcheck 闸，用 temperature-0 的 `intent_llm`（或独立 `self_check_llm`），只在此前置信度闸已通过时调用，保持“先机械后语义”的顺序。
5. **single-process 假设**：批处理/作业运行器/审核写回按单进程实现；多 worker 需外部队列，本章不做。

---

## 3. 总体架构

```text
可观测（前半）
  docker-compose.langfuse.yml: web:3000 + worker + postgres + clickhouse + redis + minio
        ▲ trace 上报（全在自家服务器）
  _build_and_cache_graph(): compile().with_config({"callbacks":[CallbackHandler()]})
        │ 缺 key 不挂，系统照常跑
  _thread_config(): metadata={langfuse_session_id/langfuse_user_id/...}
  classify_intent 节点 → update_current_trace(metadata={"intent":...}, tags=["intent:xxx"])
  build_llm(stream_usage=True) → 所有流式/非流式调用带 usage
  app/jobs/cost_by_intent.py → Langfuse legacy metrics 按 intent tag 聚合 → data/ch09/reports
  app/jobs/eval_flywheel.py → ch04 评估集/指标 → eval_runs → 趋势

数据飞轮（后半）
  入口1 evidence_confidence 低 → fallback_source=retrieval_low_conf
  入口2 selfcheck 不够答      → fallback_source=self_check
  入口3 👎 POST /api/feedback → source=user_feedback
        ↓ _finish_turn 落 low_confidence_questions(+retrieved_chunks)
  make flywheel → app/jobs/flywheel_pipeline.py
        │ 扫 matched_review_id IS NULL
        │ 一次 LLM：标准化 + 查重 + 示例答案
        ↓ 命中 review_queue 任意状态 → occurrence_count+1，状态不变
        │ 未命中 → 新建待审行
        ↓ 回写 matched_review_id
  /review 审核页 → approve → ch03 的 insert_mined_qa + vectorize_pending
        ↓ 知识库（MySQL + Milvus）→ 再问能检到
```

---

## 4. Langfuse v3 部署与接入

### 4.1 部署

新增 `docker-compose.langfuse.yml`，独立于现有 compose：

- `langfuse-web`：`langfuse/langfuse:3`，宿主 `3000:3000`
- `langfuse-worker`：`langfuse/langfuse-worker:3`
- `langfuse-postgres`：`postgres:16-alpine`
- `langfuse-clickhouse`：`clickhouse/clickhouse-server:24.8`
- `langfuse-redis`：`redis:7-alpine`
- `langfuse-minio`：`minio/minio` + `minio/mc` init 建 `langfuse` bucket

不映射 ClickHouse/Redis/Postgres/MinIO 到宿主端口，避免和现有 Milvus 的 9000/9001 冲突；只暴露 3000。

关键 env（web/worker 共用）：

```ini
DATABASE_URL=postgresql://langfuse:langfuse@langfuse-postgres:5432/langfuse
CLICKHOUSE_URL=http://langfuse-clickhouse:8123
CLICKHOUSE_MIGRATION_URL=clickhouse://langfuse-clickhouse:9000
CLICKHOUSE_USER=clickhouse
CLICKHOUSE_PASSWORD=clickhouse
CLICKHOUSE_CLUSTER_ENABLED=false
REDIS_HOST=langfuse-redis
REDIS_PORT=6379
REDIS_AUTH=langfuse
LANGFUSE_S3_EVENT_UPLOAD_BUCKET=langfuse
LANGFUSE_S3_EVENT_UPLOAD_REGION=us-east-1
LANGFUSE_S3_EVENT_UPLOAD_ACCESS_KEY_ID=minio
LANGFUSE_S3_EVENT_UPLOAD_SECRET_ACCESS_KEY=miniosecret
LANGFUSE_S3_EVENT_UPLOAD_ENDPOINT=http://langfuse-minio:9000
LANGFUSE_S3_EVENT_UPLOAD_FORCE_PATH_STYLE=true
LANGFUSE_S3_EVENT_UPLOAD_PREFIX=events/
TELEMETRY_ENABLED=false
```

web 额外：`NEXTAUTH_URL=http://localhost:3000`、`NEXTAUTH_SECRET`、`SALT`、`ENCRYPTION_KEY`（64 hex）。

初始化：参考 spec 要求首次启动后在 UI 建组织/项目拿 key。为减少演示步骤，compose 同时提供 headless init（`LANGFUSE_INIT_ORG_ID/PROJECT_ID/PUBLIC_KEY/SECRET_KEY/USER_*`），并在 README 说明两选一。开发密钥不是生产配置。

Makefile：`make langfuse-up` / `make langfuse-down`。

### 4.2 应用配置

`app/config.py` 新增：

```ini
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_BASE_URL=http://localhost:3000
LANGFUSE_ENVIRONMENT=development
```

- `langfuse_base_url` 用 `LANGFUSE_BASE_URL`（参考 spec 的命名）；构造 `Langfuse(host=...)` 时传它，不依赖 SDK 自身的 `LANGFUSE_HOST`。
- 三个值（public/secret/base_url）齐全才启用 tracing；否则 `CallbackHandler` 不挂，测试零网络。
- `langfuse` 依赖钉 `3.15.0`。

### 4.3 编译期挂回调

`app/api/ch06.py::_build_and_cache_graph`：

```python
graph = build_graph(llm, intent_llm, checkpointer=handle.saver, settings=settings)
if tracing_enabled(settings):
    graph = graph.with_config({"callbacks": [get_callback_handler(settings)]})
app.state.ch06_graph = graph
```

- `get_callback_handler` 进程级单例；`Langfuse(...)` 客户端也进程级单例。
- 缺 key / import 失败：记 warning，图原样返回。
- **state 读取适配**：`app/agent/service.py` 与 `app/api/ch06.py` 所有 `graph.get_state(...)`/`get_state_history(...)` 改为 `_state_graph(graph).get_state(...)`；`_state_graph` 返回 `getattr(graph, "bound", graph)`。`graph.stream` 仍用 binding，回调生效。
- 会话归属：`_thread_config()` 在原有 `thread_id` 基础上加：
  ```python
  config["metadata"] = {
      "langfuse_session_id": f"conv:{conversation_id}",
      "langfuse_user_id": user_id,
      "langfuse_trace_name": settings.langfuse_trace_name,
  }
  ```
  具体键名以 Context7/SDK v3 实测为准；若 v3 只认 `session_id`，按实测改。
- 并发风险：编译期 handler 共享；实现后用两个并发会话验证 trace 不串。若串，退回每次 `graph.stream` 注入 handler（记录为返工）。

### 4.4 token usage 收口

`app/config.py::build_llm` 增加 `stream_usage=True`（等价于 OpenAI `stream_options={"include_usage": True}`，Context7 已核实）。这样：

- `main_agent` 的 `.stream`、意图/指代/改写/selfcheck 的 `.invoke`、评估生成/裁判的 `.invoke` 都带 usage；
- 不在各调用点重复 bind usage 参数；
- 若 DeepSeek 实测拒绝 `stream_options`，停下来问用户，不改非流式、不估算 token。

### 4.5 意图写进 trace

`app/agent/graph.py::classify_intent_node` 得到 `intent/intent_confidence` 后：

```python
if tracing_enabled(settings):
    try:
        get_langfuse_client(settings).update_current_trace(
            metadata={"intent": intent, "intent_confidence": confidence},
            tags=[f"intent:{intent}"],
        )
    except Exception:
        logger.warning("langfuse update_current_trace failed", exc_info=True)
```

- 必须静默跳过/降级，不影响主流程。
- v3 的准确方法名/参数（`update_current_trace`、`tags`）在实现前用 Context7 和安装包核对；若与课程 README 冲突，停下来问用户。
- 意图 tag 是后面成本脚本的聚合键。

---

## 5. Cost Control：按意图算 token 账

新增 `app/jobs/cost_by_intent.py` + `scripts/cost_by_intent.py` 薄包装 + `make cost-report`。

- 参数：`--days N`（默认 7）、`--json`。
- 数据源按参考 spec 的实装勘误：
  1. 优先尝试 Langfuse v2 Metrics API（若自部署 server 支持）；
  2. 不支持则走 `client.api.legacy.metrics_v1` 的 observations 视图，按 `tags + traceId` 分组，客户端按 `intent:xxx` 聚合；
  3. 两者都失败：拉 trace 列表/observations 自行聚合，报告里写明数据源。
- 输出 `data/ch09/reports/cost_by_intent.txt` 与 `.json`：
  ```json
  {
    "generated_at": "...",
    "window_days": 7,
    "rows": [
      {"intent": "商品咨询", "requests": 12, "total_tokens": 58320,
       "avg_tokens": 4860, "share": 0.41}
    ]
  }
  ```
- 请求数、总量、均值、占比全部脚本算好写进产物；页面不重算。
- 验收 5 直接看表格；`/observability` 读同一份 JSON。

---

## 6. 证据置信度闸（入口 1）

### 6.1 纯函数 `app/agent/confidence.py`

```python
@dataclass(frozen=True)
class EvidenceConfidence:
    score: float
    signals: dict  # top1_score/valid_count/margin/key_clause_hit/...

def compute_evidence_confidence(
    hits: Sequence[RetrievedChunk | dict],
    *,
    valid_floor: float = 0.1,
    valid_cap: int = 3,
    margin_scale: float = 0.2,
) -> EvidenceConfidence:
    ...
```

信号：

- `top1_score`：精排 Top1 相关性分（0–1）。
- `valid_count`：`rerank_score >= valid_floor` 的条数，归一化封顶 `valid_cap=3`。
- `margin`：Top1 − Top2；只有一条时取 `top1_score` 自身；归一化 `min(margin/margin_scale,1)`。
- `key_clause_hit`：Top 命中里是否存在 `is_key_clause=True`（ch03 splitter 已给 policy/manual 打标）。

固定权重（模块常量，不进 settings）：

```text
score = 0.50*top1 + 0.20*min(valid_count/3,1)
      + 0.20*min(margin/0.2,1) + 0.10*key_clause_hit
```

`RetrievedChunk` 新增 `is_key_clause: bool = False`，`retrieve_hybrid` 从 MySQL 回填时带上；`sources`/快照也带上。测试构造保持向后兼容（默认 False）。

### 6.2 阈值校准

新增 `app/jobs/calibrate_confidence.py` + `scripts/calibrate_confidence.py` + `make calibrate-confidence`。

- 数据：`tests/eval/rag_cases.json`（300 条，跳过 `excluded`）。
- 标签：`answerable=true`（A/B/C/E）为正例，`D_unanswerable` 为负例；`retrieval_error/rerank_failed` 排除并计数。
- 每条用 `hybrid_rerank` 检索；每个 case 只调一次 `analyze()` 并复用。
- 扫阈值 0.00–1.00，步长 0.01；最大化 **Youden J = TPR − FPR**；并列时取更高 precision，再并列取更接近 0.5 的点。
- 输出：
  - `data/ch09/reports/confidence_calibration.txt`
  - `.json`：正负分布、完整扫描曲线、推荐阈值、TPR/FPR/J、样本/排除数、公式常量。
- 校准得到的值写入 `Settings.evidence_confidence_threshold` 默认值与 `.env.example`；报告写明来源与日期。
- live 服务不可用：任务阻塞，不写占位阈值。

### 6.3 闸位与状态

顺序不变：`检索 → 置信度闸 → selfcheck 闸 → Agent 或兜底`。

- 置信度：`score < threshold` → `fallback_source="retrieval_low_conf"`，`evidence_strong=False`，reason 带信号值。
- selfcheck：置信度通过后才调 `assess_evidence(intent_llm, question, sources)`。
  - `useful=false` → `fallback_source="self_check"`，落池。
  - selfcheck 异常/超时 → 保守兜底，**不落池**（infra）。
- `State` 新增：
  - `evidence_confidence: float | None`
  - `evidence_signals: dict`
  - `fallback_source: str`（`retrieval_low_conf` / `self_check` / 空）
  - `retrieved_snapshot: list[dict]`
- **只要走了检索就存快照**，不只闸不过时：`retrieved_snapshot` 取精排后前 3 条 `{question, answer, rerank_score, section_path, is_key_clause}`。闸不过时随落池写入；正常轮次留在 state 供 👎 回捞。
- 兜底话术与转人工按钮行为不变；闸门仍在 Agent 之前。

### 6.4 落池

`app/agent/service.py::_finish_turn`：

- `fallback_source == "retrieval_low_conf"` → `source="retrieval_low_conf"`。
- `fallback_source == "self_check"` → `source="self_check"`。
- 其它（infra/rerank/order）不落池。
- `reason` 带 `evidence_confidence` 与 signals；`retrieved_chunks=retrieved_snapshot`。
- 旧 `/chat` 的 selfcheck 落池保持 `source="self_check"`，快照取 collector 前 3。

---

## 7. 三个入口与快照

| 入口 | 触发 | source | 快照 |
|---|---|---|---|
| 置信度低 | 主图/退款图 `confidence_gate` | `retrieval_low_conf` | `state.retrieved_snapshot`（前 3） |
| selfcheck 不够答 | 主图 `self_check_gate`；旧 `/chat` | `self_check` | state 快照 / collector 前 3 |
| 用户 👎 | `POST /api/feedback` | `user_feedback` | checkpointer 最近 state 的 `retrieved_snapshot` |

- `low_confidence_questions.retrieved_chunks` JSON；`matched_review_id` FK；由 ch09 DDL 增列。
- 快照字段：`question/answer/rerank_score/section_path/is_key_clause`；answer 可截断到 500 字符。
- 没走检索（闲聊/业务数据类）→ `retrieved_chunks=NULL`。
- 落池入口不触发飞轮；飞轮由批处理脚本统一跑。

---

## 8. 用户反馈入口

### 8.1 后端 `POST /api/feedback`

请求体（在当前仓库加 `session_id` 做归属校验）：

```json
{"session_id": "...", "conversation_id": 5, "rating": "up"|"down", "question": "用户原话"}
```

- `up`：记一行应用日志，返回 200，不落库。
- `down`：
  - 校验 conversation 属于 session_id，否则 404。
  - 落 `low_confidence_questions(source="user_feedback", reason="用户反馈未解决")`。
  - 快照：`graph.get_state({"configurable":{"thread_id":f"conv:{cid}"}})`；若 `values.user_message == question` 且有 `retrieved_snapshot`，存前 3；对不上/没走检索存 NULL。
  - 幂等：前端已禁用重复点击；后端不做强幂等，重复落池交给飞轮查重。
  - 返回 `{status:"recorded", low_confidence_id}`。
- 不调用飞轮；等 `make flywheel`。
- graph 不可用/读 state 失败：快照 NULL，仍然落池。

### 8.2 前端 ch07

Vibe Coding：

- 发送一轮时把 `ctx.userQuestion = text` 记住；resume 轮沿用 `lastUserMessage`。
- `finishAssistantMessage` 在非空、非 error、非 interrupted 的回答下显示 👍/👎。
- 点击：
  - 锁定两键，显示“已反馈”。
  - 👍 发 `/api/feedback`（后端只记日志）；👎 发 `/api/feedback`（后端落池）。
  - 失败静默/提示重试，不伪装成功。
- 历史消息不回填反馈按钮。

---

## 9. 飞轮流水线（定时批处理）

### 9.1 数据表/ORM

- `review_queue`、`eval_runs` 使用用户 2026-11-19 提供的 DDL 原文，落 `sql/ch09-ddl.sql`。
- `low_confidence_questions` 增 `retrieved_chunks` JSON、`matched_review_id` BIGINT UNSIGNED FK `ON DELETE SET NULL`。
- ORM：
  - `app/models/review.py`：`ReviewQueue`、`EvalRun`（中文 ENUM `待审/通过/驳回`、`定时/手动`）。
  - `LowConfidenceQuestion` 加两列映射。
  - `id` 用 `BigInteger().with_variant(Integer, "sqlite")`。
- 初始化：`app/db/init_ch09.py`，幂等；复用 ch04 的引号感知 `_split_statements`（抽 `app/db/sql_utils.py`）。
- 测试库用 `Base.metadata.create_all`；MySQL 用 `python -m app.db.init_ch09`。

### 9.2 一次 LLM 输出标准化 + 查重

新增 `app/services/flywheel.py`：

```python
class FlywheelOutput(BaseModel):
    normalized_question: str
    matched_question_id: int | None = None
    ai_suggested_answer: str = ""
```

Prompt（`app/prompts/flywheel.py` 或 `app/prompts/customer_service.py` 同级新增）：

- 输入：用户原话 + 候选列表（`review_queue` **全部行**，按 `updated_at DESC` 截 200 条：`id + normalized_question + occurrence_count + status`）。
- 一次输出 JSON：标准化问题、示例答案、命中 id（没有同类为 null）。
- 要求：保留关键实体；示例答案只备查；JSON only。

`process_pending(limit)`：

1. 扫 `low_confidence_questions` where `matched_review_id IS NULL`，按 id 升序，取 `limit`。
2. 逐条**串行**调用模型（不并发；同批第二条能命中第一条刚建的行）。
3. `matched_question_id` 非空且确实存在于 `review_queue`：
   - `occurrence_count += 1`；
   - **状态不变**（命中已驳回/已通过也只累加；驳回即终审，不复活、不新建）。
4. 未命中：新建 `review_queue(status='待审', occurrence_count=1, normalized_question, ai_suggested_answer)`。
5. 回写 `low_confidence_questions.matched_review_id`。
6. 坏 JSON / 幻觉 id / 模型异常：该条跳过并告警，游标留在原地，下轮重试；不新建行。
7. 整个条目的“查重命中/新建 + 回写”在同一事务里；失败 rollback，low_confidence 行保留、matched_review_id 仍 NULL。

### 9.3 驱动

- `app/jobs/flywheel_pipeline.py`：`python -m app.jobs.flywheel_pipeline --limit N`。
- `scripts/flywheel_pipeline.py` 薄包装。
- `make flywheel`。
- 打印：扫描数、新建缺口数、归并数、失败数。
- 交付 cron 示例：`*/30 * * * * cd /path && make flywheel`；不引进程内调度器。
- 单进程串行；跨进程并发需外部锁，本章不做。

---

## 10. 审核 API 与写回知识库

### 10.1 REST

新增 `app/api/review.py`：

| 接口 | 行为 |
|---|---|
| `GET /api/review/queue?status=待审` | 列表：标准化问题、出现次数、示例答案、状态；`occurrence_count DESC` |
| `GET /api/review/{id}` | 详情：队列行 + 归并原话列表（raw_question/source/reason/created_at/retrieved_chunks） |
| `POST /api/review/{id}/approve` | body `{"approved_answer": str}`；仅待审可操作；写回知识库成功后置通过 |
| `POST /api/review/{id}/reject` | 置驳回；仅待审可操作 |

- 非待审 → 409；不存在 → 404；核准答案空 → 422。
- 不做登录鉴权（与现有后台页一致；文档注明生产要补）。

### 10.2 写回

新增 `app/services/knowledge_approval.py`：

- 用 `app/rag/ingest.py::insert_mined_qa` 写入一条 QA chunk：
  - `content_type="mined_qa"`、`category="通用"`、`questions=normalized_question`、`answer=approved_answer`、`vectorize_status="pending"`。
  - 指纹去重：重复审批/重试复用同一行。
- 调 `app/rag/vectorize.py::vectorize_pending(session, embedder, store)` 同步向量化。
- 向量化成功后才把 `review_status="通过"`、`approved_answer` 写回并 commit。
- 失败：状态保持待审、返回 5xx；chunk 可能已 pending，重试会复用并继续向量化（可重试，不重复建行）。
- 用模块锁 + `kb_build.is_running()` 防止并发构建。
- 可注入 `embedder/store` 便于测试，不触达 live Milvus。

---

## 11. 自动化评估流水线

### 11.1 `eval_runs`

- DDL 用户原文；`triggered_by` `手动/定时`，`dataset_size`，`metrics` JSON。
- metrics 每轮：
  ```json
  {
    "strategy": "hybrid_rerank",
    "cases_evaluated": 269,
    "excluded": 31,
    "recall_at_10": 0.99,
    "mrr_at_10": 0.95,
    "faithfulness": 0.99,
    "d_refusal_accuracy": 0.96
  }
  ```
  （可同时存 Recall@5 等辅助值；趋势只看上表。）
- `dataset_size` 用 JSON 的 `cases_loaded`（本轮评估集条数）。

### 11.2 脚本

新增 `app/jobs/eval_flywheel.py` + `scripts/eval_flywheel.py` + `make eval-flywheel`：

- 复用 `app/jobs/eval_rag.py` 的检索/生成/裁判函数，抽公共函数，不复制粘贴。
- 只跑生产策略 `hybrid_rerank`。
- 取 `tests/eval/rag_cases.json`；可选 `--limit`。
- 生成报告与 JSON sidecar；读 sidecar 提取 `overall.hybrid_rerank` 的 Recall@10/MRR/Faithfulness/D 拒答准确率。
- 插入 `eval_runs(triggered_by, dataset_size, metrics)`。
- 打印最近 N 轮趋势：每项相对上一轮 `↑/↓/→`，下滑标 ⚠；趋势文本落 `data/ch09/reports/eval_trend.txt`。
- **权威源是 `eval_runs` 表**；页面直接读表，不读趋势文本。
- 失败不插行；不写假指标。
- 无进程内定时器；cron 示例写入 README。

---

## 12. 前端

前端统一 Vibe Coding，不套 TDD/code review；最终浏览器人工验收。

### 12.1 ch07 聊天页

- 👎/👍 接 `/api/feedback`（见 §8.2）。
- 反馈条 UI 行为与 ch04 一致：点亮所选、显示“已反馈”、一次性锁定。
- 不引入前端依赖；复用现有 CSS 变量。

### 12.2 `/review` 待审后台

- 列表：标准化问题、出现次数（降序）、示例答案摘要、状态筛选（默认待审）。
- 行操作：通过（弹框填核准答案，预填 `ai_suggested_answer`）/ 驳回。
- 行详情：归并进来的原话列表（source 标签、时间、reason）+ 每条 `retrieved_chunks` 快照卡（section_path、rerank_score、原文摘要）。
- 页顶审核三道闸提示：垃圾/时效/频次。
- 通过后提示“已写回知识库”；失败提示重试。

### 12.3 `/observability` 报表页

三块，各自状态与重跑按钮：

| 块 | 权威源 | 缺产物 |
|---|---|---|
| 意图成本账 | `data/ch09/reports/cost_by_intent.json` | `present=false` + 提示先聊几句再重跑 |
| 评估趋势 | `eval_runs` 最近 10 轮 | `present=false` + 提示先跑一轮 |
| 置信度校准 | `data/ch09/reports/confidence_calibration.json` + `settings.evidence_confidence_threshold` | `present=false` + 提示先扫一遍 |

- **不许第二个真相**：占比/单均/Youden J/推荐阈值都由脚本算好写入产物；API 不重算。涨跌箭头由页面拿同一批 `eval_runs` 两两相减。
- 三块互不连坐：Langfuse 没起只影响成本块；MySQL 挂了只影响趋势块。
- 重跑按钮走白名单作业运行器（§12.5）；eval/calibrate 二次确认。
- 推荐阈值 ≠ 在用阈值时，页面明确提示“该回填 `app/config.py`”。

### 12.4 `/admin` 后台首页与共用导航

- `app/static/admin.js` 提供 `mountAdminNav`；`/admin`、`/review`、`/observability`、`/kb`、`/rag-eval` 共用一套导航。
- `/admin` 卡片：
  - 待审/通过/驳回三个数；
  - 一句结论（如“3 条待审，最高频问题出现 8 次”）；
  - 指向 `/review`、`/observability` 的入口。
- 不改各页原地址。

### 12.5 白名单作业运行器

新增 `app/services/admin_jobs.py` + `app/api/admin_jobs.py`：

- 允许的作业名：`cost-report`、`eval-flywheel`、`calibrate-confidence`。
- 前端只传作业名，传不进命令片段。
- 后台线程执行，返回 `running/started_at/finished_at/exit_code/output_tail`；页面轮询。
- 同一作业互斥；不同作业可并行（但评估/校准重活建议二次确认）。
- 运行器只调用 `app/jobs/*.main()`，不拼 shell。

---

## 13. 数据库与初始化

- `sql/ch09-ddl.sql`：用户 2026-11-19 提供的 DDL 原文（见 §13.1）。
- `app/db/init_ch09.py`：
  - 表/列全在 → no-op；
  - 全缺 → 执行全部；
  - 部分存在 → 抛清晰错误，不半应用。
- 复用/抽出 `_split_statements` 到 `app/db/sql_utils.py`；ch04 保持兼容。
- MySQL 中文 ENUM/JSON 按 utf8mb4；SQLite 测试用 ORM `create_all`。

### 13.1 DDL 原文（逐字落 `sql/ch09-ddl.sql`）

```sql
-- =============================================================
-- ch09 · 可观测性与数据飞轮 · 建表 DDL
-- 本章新建:review_queue(待审队列,标准化去重后的知识缺口)
--          eval_runs(自动化评估流水线每轮结果,连起来看趋势)
-- 并给 ch04 的 low_confidence_questions 加两列:
--   matched_review_id 记这条原话查重后归并到哪个缺口
--   retrieved_chunks  存落池时的召回片段快照,审核页给人看
-- 建表顺序:先 CREATE review_queue,再 ALTER low_confidence_questions 加外键指向它
-- =============================================================

SET NAMES utf8mb4;

CREATE TABLE review_queue (
  id                  BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '缺口主键,也是查重命中要返回的 matched_question_id',
  normalized_question VARCHAR(512)    NOT NULL                COMMENT '标准化后的 FAQ 式问题',
  ai_suggested_answer TEXT            NULL                    COMMENT '模型生成的示例答案,备查',
  occurrence_count    INT UNSIGNED    NOT NULL DEFAULT 1      COMMENT '出现次数,查重命中累加,越高越该优先补',
  review_status       ENUM('待审','通过','驳回') NOT NULL DEFAULT '待审' COMMENT '人工审核状态',
  approved_answer     TEXT            NULL                    COMMENT '审核通过时补的核准答案,走 ch03 落库流程写回知识库',
  created_at          DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '首次入队时间',
  updated_at          DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (id),
  KEY idx_review_status (review_status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='飞轮待审队列';

CREATE TABLE eval_runs (
  id           BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '评估轮次主键',
  triggered_by ENUM('定时','手动') NOT NULL DEFAULT '定时' COMMENT '这轮怎么起的:定时任务,或某次改动后手动跑',
  dataset_size INT UNSIGNED    NOT NULL                COMMENT '这轮跑的评估集条数',
  metrics      JSON            NOT NULL                COMMENT '各指标分数,如 {"recall_at_k":0.82,"mrr":0.71,"faithfulness":0.90}',
  created_at   DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '跑完落表时间',
  PRIMARY KEY (id),
  KEY idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='自动化评估流水线轮次结果';

ALTER TABLE low_confidence_questions
  ADD COLUMN retrieved_chunks  JSON            NULL COMMENT '落池时的召回片段快照:Top 几条的原文与得分,审核页展示用;没走检索为 NULL' AFTER reason,
  ADD COLUMN matched_review_id BIGINT UNSIGNED NULL COMMENT '查重后归并到的缺口,指向 review_queue.id' AFTER retrieved_chunks,
  ADD KEY idx_matched_review_id (matched_review_id),
  ADD CONSTRAINT fk_lcq_review FOREIGN KEY (matched_review_id) REFERENCES review_queue (id) ON DELETE SET NULL;
```

---

## 14. 配置汇总

```ini
# Langfuse v3
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_BASE_URL=http://localhost:3000
LANGFUSE_ENVIRONMENT=development
LANGFUSE_TRACE_NAME=cs-chat
STREAM_USAGE=true

# Evidence confidence
EVIDENCE_CONFIDENCE_THRESHOLD=0.0          # 校准后回填
EVIDENCE_VALID_FLOOR=0.1
EVIDENCE_VALID_CAP=3
EVIDENCE_MARGIN_SCALE=0.2
LOW_CONFIDENCE_SNAPSHOT_TOP_K=3

# Flywheel
FLYWHEEL_CANDIDATE_LIMIT=200
FLYWHEEL_LLM_MAX_TOKENS=800
FLYWHEEL_TIMEOUT_S=20

# Reports
CH09_REPORT_DIR=data/ch09/reports
```

- `EVIDENCE_CONFIDENCE_THRESHOLD` 校准前不写死；测试显式注入。
- 移除/弃用 `CONFIDENCE_MIN_SCORE`：新闸门不使用；保留字段也必须标 deprecated 且不参与判定。
- `.env.example` 同步；`.env` 由实现阶段补本地开发 key（不提交）。

---

## 15. 错误处理与安全

- Langfuse 缺 key/不可用：不挂回调，聊天照常；`update_current_trace` 静默降级。
- 飞轮模型坏 JSON/幻觉 id：跳过、游标停留、下轮重试；不丢问题。
- 飞轮 DB 失败：rollback，low_confidence 行保留。
- 审核写回失败：状态保持待审，5xx；chunk pending 可重试，不重复建行。
- 反馈 conversation 不属于 session：404；question 对不上最新 state：快照 NULL，仍落池。
- 只落 Top 3 快照，answer 截断 500 字符。
- 单进程锁/串行；多 worker 不保证。
- 审核页无登录；本地演示，生产需补权限。
- trace 包含用户问题/回答，但全在自部署 Langfuse。

---

## 16. 测试与验证

### 16.1 TDD 单测

- `compute_evidence_confidence`：空召回、单条、同分、key clause、有效条数封顶、margin 归一化。
- `confidence` 阈值：Youden J 选择函数（合成标签）。
- 主图：置信度 fail → `retrieval_low_conf`；selfcheck fail → `self_check`；infra/selfcheck 异常不落池；快照 top3。
- 旧 `/chat` selfcheck source/快照。
- `flywheel`：命中/未命中/幻觉 id/坏 JSON/已驳回命中状态不变/串行/幂等。
- `/api/feedback`：up 不落库、down 落库、快照对得上/对不上、归属校验。
- `/api/review`：列表/详情/approve/reject/409/写回失败回滚。
- `insert_mined_qa` + `vectorize_pending` mock；审批幂等。
- `admin_jobs`：白名单、互斥、未知作业拒绝。
- `init_graph` 缺 key 不挂回调；`with_config` 后用 `_state_graph` 能读 state。
- `cost_by_intent`：fake Langfuse client、空数据、聚合占比。
- `eval_flywheel`：指标提取、eval_runs 插入、失败不插行、趋势箭头。
- 现有 772 测试零回归。

### 16.2 标注样例 / 数据验证

- `tests/eval/flywheel_samples.json`：10–15 条口语原话 × 候选队列 × 期望命中/不命中；真实 LLM 跑通过率 ≥ 80% 视为可用；不达标改 prompt，不放宽标注。
- 置信度校准：评估集分布 + Youden J 报告即验证。
- 评估流水线：至少两轮真实评估，趋势出得来。

### 16.3 live 验收

```bash
make langfuse-up
docker compose up -d
python -m app.db.init_ch09
make calibrate-confidence
python run.py

# 验收 1：Langfuse UI 点任意 trace
# 验收 2：问知识库没有的问题 → 兜底 → make flywheel → /review 详情
# 验收 3：/review 通过 → 再问 → 答对
# 验收 4：聊天页 👎 → make flywheel → /review
# 验收 5：make cost-report → /observability 同源
# 验收 6：make eval-flywheel 两轮 → 趋势
# 验收 7：/observability 三块可看、可重跑
```

全量测试：`.venv/Scripts/python.exe -m pytest -q`。

---

## 17. 验收映射

| # | 验收 | 落点 |
|---|---|---|
| 1 | Langfuse 完整链路 | §4 |
| 2 | 兜底 + 待审队列 + 详情快照 | §6/§7/§9/§12.2 |
| 3 | 审核通过后答对 | §10.2 |
| 4 | 👎 落池 → 飞轮 → 待审 | §8/§9 |
| 5 | 按意图 token 统计 | §5/§12.3 |
| 6 | 至少两轮评估趋势 | §11/§12.3 |
| 7 | 三张报表页可看可重跑 | §12.3/§12.5 |

---

## 18. 风险与开放问题

1. **Docker/Milvus/MySQL/Langfuse 当前不可用**：校准、审核写回、live 验收必须等服务恢复；不伪造。
2. **Langfuse v3 SDK API 形态**：`CallbackHandler` 构造、session/metadata 键名、`update_current_trace`、legacy metrics 方法名，实现前用 Context7/安装包核对；若与课程 README 冲突，停下来问用户。
3. **编译期共享 handler 并发串 trace**：用两个并发会话验证；串则退回每次 stream 注入 handler（返工记录）。
4. **`with_config` 丢 `get_state`**：用 `_state_graph` 取 `bound`；所有 state 读取点必须替换，漏一处会在 interrupt/resume 处炸。
5. **DeepSeek streaming usage**：`stream_usage=True` 若被上游拒绝，停下来问用户，不改非流式/估算。
6. **成本 API 版本差异**：v2 metrics 可能不被自部署 v3 支持；按参考 spec 的 legacy v1 方案 + 客户端聚合，报告写明数据源。
7. **selfcheck 新增调用**：主图知识/退款轮次多一次 LLM 调用，成本/延迟上升；这是参考 spec 要求的顺序，验收时观察。
8. **查重候选截断 200**：教学规模够用；截断时日志说明。
9. **审核写回同步向量化依赖嵌入上游在线**：失败保持待审、可重试；不做异步补偿队列。
10. **审核页/报表页无鉴权**：本地演示；生产需补。

---

## 19. 留给构建者的决定

- trace metadata 精确键名、成本 API 的具体 SDK 调用（Context7 核实后定）。
- 飞轮 prompt 措辞与样例。
- 审核页/报表页/后台首页的布局与交互（Vibe Coding）。
- `init_ch09` 的语句切分复用方式。
- Makefile 与 `scripts/` 包装的细节。
- 是否在 ch07 页加入口链接；不改变验收路径。
