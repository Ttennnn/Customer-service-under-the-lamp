# ch04 RAG 进阶 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把现有 BGE-M3 dense 单路检索升级为 Milvus 原生 BM25 + dense 混合检索 + bge-reranker-v2-m3 精排,补齐引用、生成前自评/拒答、低置信度池,并用四策略分桶评估集量化效果;前端支持引用可点与一次性满意度反馈。

**Architecture:** 新集合 `knowledge_hybrid` 同时存 dense 向量、BM25 `text/sparse` 与过滤元数据;`retrieve_hybrid()` 统一 dense / bm25 / hybrid / hybrid_rerank 四策略;`query_faq` 输出带 `[n]` 的证据并通过可变 `CitationCollector` 把来源透出到 SSE;`stream_chat` 在最终生成前用独立 temperature=0 LLM 自评证据,不足则拒答并落 `low_confidence_questions`。评估任务对 35 条分桶样本跑四策略 Recall/MRR/Faithfulness 并落 `faith_cases`。

**Tech Stack:** Python 3.13, FastAPI 0.162.x, SQLAlchemy 2.x, pymilvus 3.0.3, Milvus 3.0.3(原生 BM25 + chinese analyzer + hybrid_search RRF), 硅基流动 BGE-M3 / bge-reranker-v2-m3, MySQL 8, LangChain 1.x, pytest, Vite + React + Tailwind。

**Spec:** `docs/superpowers/specs/2026-11-16-ecommerce-cs-ch04-rag-advanced-design.md`

## Global Constraints

- 稠密向量固定 BGE-M3(硅基流动云端,1024 维);重排固定 `BAAI/bge-reranker-v2-m3`,走 `POST https://api.siliconflow.cn/v1/rerank`。
- Milvus `knowledge_hybrid` 的 `text` 字段:`enable_analyzer=True`, `analyzer_params={"type": "chinese"}`;BM25 Function + `SPARSE_INVERTED_INDEX`/`metric_type="BM25"`/`k1=1.2,b=0.75`。
- 召回深度 `hybrid_recall_top_k=50`;RRF `k=60`;最终 `rerank_top_n=10`;`query_faq` 超时 25 秒且 `retries=0`;rerank 超时 10 秒。
- 过滤只来自显式 `category` / `content_type` 参数或评估集 `filters`,精确匹配;不让 query 改写 LLM 猜过滤值。
- `section_path` 是引用与评估的定位键;Recall/MRR 按 `section_path` 去重,同一节多个 chunk 不重复计分。
- prompt 证据顺序固定「最强在首尾」:`[1,3,5,...,6,4,2]`;引用编号、sources 帧、前端来源列表都跟随该顺序。
- 自评 `useful=false` 才写 `low_confidence_questions(source='self_check')`;检索/自评 infra 失败只拒答 + 日志,不写池。
- `faith_cases` 一题一行(`eval_id` 唯一),`citations` 存本轮 Top-10 证据全集 JSON。
- 前端反馈纯本地 `localStorage`,不新增后端 API;前端不走 TDD,以 `npm run build` + 手工验收为准。
- 每个任务完成即追加 `dev-notes/ch04.md` 阶段记录(用户原话/产出/拒绝纠偏/翻车返工),不得收尾一次性补记。
- 不删除旧 `knowledge` 集合、不改 ch01–ch03 表结构、不新增认证/队列/调度器。
- 分支:`feature/ch04-rag-advanced`,从当前 `feature/ch02-tools` 起;本地提交,不推送、不建 PR。

## Review Focus

1. **Milvus 3.0.3 的 BM25 + 内置 chinese analyzer 是否真的可用**:Context7 文档对 v3.0.x,但假 client 测不出服务端兼容性。Task 1 必须跑真实临时集合 smoke;失败则停下问用户,不自行换方案。
2. **引用编号跨工具线程/多次 query_faq 是否稳定**:`execute_tool` 在子线程执行;collector 必须是 contextvar 里的可变对象,靠 chunk_id 去重递增。Task 6 的线程可见性测试。
3. **自评闸门不能误伤或漏放**:infra 失败不能写池;真实无命中必须写池;混合意图不能把订单/物流答案一起拒掉。Task 7/8 测试 + Task 9 标注样例。
4. **四策略对比不能被指标口径污染**:同节多 chunk 不能重复 Recall、rerank 失败不能冒充 hybrid_rerank、四策略必须复用同一个 QueryAnalysis。Task 12 测试。
5. **前端未知引用与反馈锁定**:未知 `[n]` 必须按普通文本显示,不能点;👍/👎 只能写一次并显示「已反馈」。Task 13 手工验收。

---

## File Structure

| 文件 | 责任 |
|---|---|
| `sql/ch04_rag_advanced.sql` | 本章两表权威 DDL(用户原文) |
| `app/db/init_ch04.py` | 幂等 apply ch04 DDL;半初始化报错 |
| `app/models/knowledge.py` | 新增 `LowConfidenceQuestion` / `FaithCase` ORM |
| `app/config.py` | 混合/重排/超时配置;`build_llm(temperature=...)` |
| `app/rag/milvus_store.py` | `knowledge_hybrid` schema、BM25 Function、dense/bm25/hybrid 检索、recreate |
| `app/rag/vectorize.py` | 写入 BM25 文本与元数据;`vectorize_all` |
| `app/jobs/reindex_hybrid.py` | 全量迁移到 `knowledge_hybrid` |
| `app/rag/rerank.py` | 硅基流动 rerank 客户端与降级错误 |
| `app/rag/query_understanding.py` | 改写 + 同义词扩展(不改入库文本) |
| `app/rag/splitter.py` | frontmatter category 对所有 content_type 优先 |
| `app/rag/retrieval.py` | 四策略统一入口、`RetrievalResult` |
| `app/tools/context.py` | `CitationCollector` contextvar |
| `app/tools/faq.py` | 带 `[n]` 的 query_faq、过滤参数、MISS/ERROR 常量 |
| `app/tools/registry.py` | `query_faq` 超时 25 秒 |
| `app/rag/self_check.py` | 证据充分性自评 |
| `app/services/low_confidence.py` | 低置信度问题落库 |
| `app/services/chat.py` | sources 帧、自评、拒答、混合意图、引用有效性 |
| `app/main.py` | temperature=0 自评 LLM 依赖 |
| `app/prompts/customer_service.py` | 引用/拒答/负面知识/型号路由约束 |
| `app/services/faithfulness.py` | Faithfulness 裁判 |
| `app/services/faith_cases.py` | `faith_cases` upsert 台账 |
| `app/jobs/eval_prompts.py` | 改写/自评标注样例验证 |
| `app/jobs/eval_rag.py` | 四策略检索/生成评估与报告 |
| `knowledge/商品型号与参数.md` | 型号样例知识(BM25 命中用) |
| `tests/eval/rag_cases.json` | 35 条分桶、难度、ground-truth 评估集 |
| `tests/eval/query_rewrite_samples.json` | 10 条改写标注样例 |
| `tests/eval/self_check_samples.json` | 10 条自评标注样例(5 可答 + 5 不可答) |
| `frontend/src/types.ts` | `CitationSource` / feedback 类型 |
| `frontend/src/api/chat.ts` | 解析 sources SSE 帧 |
| `frontend/src/hooks/useChat.ts` | pending sources、feedback reducer |
| `frontend/src/components/MessageItem.tsx` | 引用 chip、来源卡片、👍/👎 |
| `docs/superpowers/reports/2026-11-16-ch04-rag-eval.md` | 四策略报告 |
| `dev-notes/ch04.md` | 阶段追记 |

---

### Task 0: 分支、ch04 建表与 ORM、文档归位

**Files:**
- Create: `sql/ch04_rag_advanced.sql`, `app/db/init_ch04.py`, `tests/test_ch04_schema.py`
- Modify: `app/models/knowledge.py`, `app/models/__init__.py`
- Verify/commit: `dev-notes/ch03.md`, `dev-notes/ch04.md`, renamed ch03 knowledge-ui spec/plan

**Interfaces:**
- Produces: `apply_ch04(engine: Engine) -> None`; ORM `LowConfidenceQuestion`, `FaithCase`
- Produces: `app.models.LowConfidenceQuestion`, `app.models.FaithCase` exported from `app.models`

- [ ] **Step 1: Create the feature branch and verify the doc renumbering already in the working tree**

Run:
```bash
git switch -c feature/ch04-rag-advanced
rg -n "补记:知识录入页|RAG 进阶" dev-notes/ch03.md dev-notes/ch04.md
```
Expected: branch created; ch03 has the 补记 heading; ch04 starts with `# 第四章 · RAG 进阶`.

- [ ] **Step 2: Write the failing schema/ORM tests**

Create `tests/test_ch04_schema.py` with:
```python
from app.db.init_ch04 import TABLES, _load_statements
from app.models import FaithCase, LowConfidenceQuestion

def test_ddl_contains_both_tables_and_utf8_statement():
    sql = "\n".join(_load_statements())
    assert "CREATE TABLE low_confidence_questions" in sql
    assert "CREATE TABLE faith_cases" in sql
    assert "SET NAMES utf8mb4" in sql
    assert TABLES == ("low_confidence_questions", "faith_cases")

def test_low_confidence_question_roundtrip(session):
    row = LowConfidenceQuestion(
        conversation_id=None, raw_question="支持以旧换新吗", source="self_check",
        reason="证据不足",
    )
    session.add(row); session.commit(); session.refresh(row)
    assert row.id > 0 and row.source == "self_check" and row.created_at is not None

def test_faith_case_json_and_enum(session):
    row = FaithCase(
        eval_id="A01", bucket="A_policy", query="退款多久", strategy="hybrid_rerank",
        answer="答案", reason="编造", citations=[{"n": 1, "chunk_id": 3}],
        judge_model="deepseek-chat",
    )
    session.add(row); session.commit(); session.refresh(row)
    assert row.status == "未解决" and row.seen_count == 1
    assert row.citations == [{"n": 1, "chunk_id": 3}]
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ch04_schema.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.db.init_ch04'` / missing model names.

- [ ] **Step 4: Add the DDL, ORM models, and init script**

Create `sql/ch04_rag_advanced.sql` with the exact two-table DDL from the spec (including `SET NAMES utf8mb4;`).

In `app/models/knowledge.py` add:
```python
class LowConfidenceQuestion(Base):
    __tablename__ = "low_confidence_questions"
    id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    conversation_id: Mapped[Optional[int]] = mapped_column(BigInteger, ForeignKey("conversations.id"))
    raw_question: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(Enum("retrieval_low_conf", "self_check", "user_feedback", name="low_confidence_source"), index=True)
    reason: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), index=True)

class FaithCase(Base):
    __tablename__ = "faith_cases"
    id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    eval_id: Mapped[str] = mapped_column(String(16), unique=True)
    bucket: Mapped[str] = mapped_column(String(24))
    query: Mapped[str] = mapped_column(String(512))
    strategy: Mapped[str] = mapped_column(String(24), default="hybrid_rerank", server_default="hybrid_rerank")
    answer: Mapped[str] = mapped_column(Text)
    reason: Mapped[str] = mapped_column(Text)
    citations: Mapped[Optional[list]] = mapped_column(JSON)
    judge_model: Mapped[Optional[str]] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(Enum("未解决", "已解决", "无需解决", name="faith_case_status"), default="未解决", server_default="未解决", index=True)
    seen_count: Mapped[int] = mapped_column(Integer, default=1, server_default="1")
    first_seen_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    last_seen_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), index=True)
    resolution: Mapped[Optional[str]] = mapped_column(String(300))
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
```

Export both from `app/models/__init__.py`.

Create `app/db/init_ch04.py` modeled on `app/db/init_ch03.py`:
```python
SQL_PATH = Path(__file__).resolve().parents[2] / "sql" / "ch04_rag_advanced.sql"
TABLES = ("low_confidence_questions", "faith_cases")
def apply_ch04(engine: Engine) -> None: ...
def main() -> None: ...
```
`_load_statements` drops `--` lines and splits on `;`. `apply_ch04` creates both when absent, no-ops when both exist, raises on partial schema.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ch04_schema.py -q`
Expected: all pass.

- [ ] **Step 6: Append the dev-notes stage and commit**

Append `## 阶段:Task 0 完成(分支/ch04 建表/文档归位)` to `dev-notes/ch04.md` with: user's exact approval words (`确定`), outputs (DDL/ORM/init; ch03 merge), rejections (none), rework (the shell backtick fix), evidence (test command and pass count).

Commit:
```bash
git add sql/ch04_rag_advanced.sql app/db/init_ch04.py app/models/knowledge.py app/models/__init__.py tests/test_ch04_schema.py dev-notes/ch03.md dev-notes/ch04.md docs/superpowers/specs docs/superpowers/plans
git commit -m "feat: add ch04 schema and merge knowledge UI docs into ch03"
```

---

### Task 1: 配置与 Milvus `knowledge_hybrid` 集合(含真实 BM25 smoke)

**Files:**
- Modify: `app/config.py`, `.env.example`, `app/rag/milvus_store.py`
- Test: `tests/test_milvus_store.py`, `tests/test_config.py`

**Interfaces:**
- Produces: `VectorItem(vector_id, chunk_id, vector, text, category, content_type, section_path)`
- Produces: `KnowledgeVectorStore.dense_search(vector, top_k, expr=None) -> list[SearchHit]`
- Produces: `KnowledgeVectorStore.bm25_search(query_text, top_k, expr=None) -> list[SearchHit]`
- Produces: `KnowledgeVectorStore.hybrid_search(vector, query_text, top_k, expr=None, rrf_k=60) -> list[SearchHit]`
- Produces: `KnowledgeVectorStore.recreate_collection() -> None`
- Produces: config fields `milvus_hybrid_collection`, `hybrid_recall_top_k`, `hybrid_rrf_k`, `rerank_top_n`, `rerank_model`, `rerank_base_url`, `rerank_api_key`, `rerank_timeout_s`, `query_rewrite_enabled`
- Produces: `build_llm(settings, temperature=0.7)`

- [ ] **Step 1: Write the failing config/store tests**

Add to `tests/test_milvus_store.py`:
```python
def test_hybrid_schema_has_text_bm25_function_and_scalar_fields():
    c = FakeClient()
    KnowledgeVectorStore(c, "knowledge_hybrid", 1024).ensure_collection()
    names = {f["name"] for f in c.fields}
    assert {"id", "chunk_id", "vector", "text", "sparse", "category", "content_type", "section_path"} <= names
    text = next(f for f in c.fields if f["name"] == "text")
    assert text["enable_analyzer"] is True
    assert text["analyzer_params"] == {"type": "chinese"}
    assert c.functions and c.functions[0].name == "bm25_fn"
    assert any(i["field_name"] == "sparse" and i["metric_type"] == "BM25" for i in c.indexes)

def test_upsert_includes_bm25_text_and_metadata():
    c = FakeClient(); s = KnowledgeVectorStore(c, "knowledge_hybrid", 1024)
    s.upsert([VectorItem("chunk_1", 1, [0.0]*1024, "商品\n型号\nLX-A100", "商品", "manual", "商品型号与参数 > LX-A100 落地灯")])
    assert c.upserted[1][0]["text"] == "商品\n型号\nLX-A100"
    assert c.upserted[1][0]["category"] == "商品"

def test_bm25_search_passes_raw_text_and_filter():
    c = FakeClient(); s = KnowledgeVectorStore(c, "knowledge_hybrid", 1024)
    s.bm25_search("LX-A100", 50, expr='category == "商品"')
    call = c.search_calls[-1]
    assert call["data"] == ["LX-A100"]
    assert call["anns_field"] == "sparse"
    assert call["search_params"] == {"metric_type": "BM25"}
    assert call["filter"] == 'category == "商品"'

def test_hybrid_search_builds_two_requests_and_rrf():
    c = FakeClient(); s = KnowledgeVectorStore(c, "knowledge_hybrid", 1024)
    s.hybrid_search([0.1]*1024, "LX-A100", 50, rrf_k=60)
    reqs = c.hybrid_calls[-1]["reqs"]
    assert [r.anns_field for r in reqs] == ["vector", "sparse"]
    assert c.hybrid_calls[-1]["ranker"].k == 60
```

Add to `tests/test_config.py`:
```python
def test_ch04_defaults_exist():
    s = Settings(openai_api_key="x")
    assert s.milvus_hybrid_collection == "knowledge_hybrid"
    assert s.hybrid_recall_top_k == 50 and s.hybrid_rrf_k == 60
    assert s.rerank_model == "BAAI/bge-reranker-v2-m3"
    assert s.rerank_timeout_s == 10.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_milvus_store.py tests/test_config.py -q`
Expected: FAIL — missing fields/methods, `FakeClient` lacks `hybrid_search`/`add_function`.

- [ ] **Step 3: Implement config and store changes**

In `app/config.py` add the fields from the spec and:
```python
def build_llm(settings: Settings, temperature: float = 0.7) -> ChatOpenAI:
    return ChatOpenAI(..., temperature=temperature)
```
Update `.env.example` with the new keys (no secrets).

In `app/rag/milvus_store.py`:
- Extend `VectorItem`.
- `ensure_collection` adds `text`, `sparse`, `category`, `content_type`, `section_path`; `schema.add_function(Function(name="bm25_fn", function_type=FunctionType.BM25, input_field_names=["text"], output_field_names=["sparse"]))`; indexes: vector FLAT/COSINE, sparse SPARSE_INVERTED_INDEX/BM25 `{"bm25_k1":1.2,"bm25_b":0.75}`, category/content_type INVERTED.
- `dense_search`: `client.search(..., filter=expr or "", output_fields=["chunk_id"])`.
- `bm25_search`: `data=[query_text]`, `anns_field="sparse"`, `search_params={"metric_type":"BM25"}`.
- `hybrid_search`: two `AnnSearchRequest` + `RRFRanker(k=rrf_k)`, `client.hybrid_search(collection, reqs, ranker=..., limit=top_k, output_fields=["chunk_id"])`.
- `recreate_collection`: drop then `ensure_collection`; reset `_ensured`.
- `get_vector_store()` uses `settings.milvus_hybrid_collection`.

Update `FakeClient` in tests with `add_function`, `hybrid_search`, `drop_collection`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_milvus_store.py tests/test_config.py -q`
Expected: all pass.

- [ ] **Step 5: Live smoke the real Milvus BM25 + chinese analyzer**

Start Milvus if needed (`docker compose up -d`). Then run:
```bash
.venv/Scripts/python.exe - <<'PY'
from pymilvus import MilvusClient
from app.rag.milvus_store import KnowledgeVectorStore, VectorItem
from app.config import get_settings

name = "ch04_bm25_smoke"
client = MilvusClient(uri=get_settings().milvus_uri)
if client.has_collection(name):
    client.drop_collection(name)
store = KnowledgeVectorStore(client, name, 1024)
store.upsert([
    VectorItem("chunk_1", 1, [0.0]*1024, "商品\nLX-A100 落地灯\n色温 3000K-5000K", "商品", "manual", "商品型号与参数 > LX-A100 落地灯"),
    VectorItem("chunk_2", 2, [0.0]*1024, "运费\n满 99 元包邮", "运费", "policy", "运费与配送说明 > 运费标准"),
])
hits = store.bm25_search("LX-A100 色温", 5)
print([(h.chunk_id, round(h.score, 4)) for h in hits])
assert hits and hits[0].chunk_id == 1
client.drop_collection(name)
print("BM25_SMOKE_OK")
PY
```
Expected: first line includes `(1, ...)`; final line `BM25_SMOKE_OK`. If the collection/function/analyzer fails, stop and report to the user; do not replace Milvus BM25 with an app-level BM25.

- [ ] **Step 6: Append dev-notes and commit**

Append `## 阶段:Task 1 完成(配置 + knowledge_hybrid 集合 + BM25 smoke)` with the smoke output, any version quirks, rejections, rework, and test counts.

```bash
git add app/config.py .env.example app/rag/milvus_store.py tests/test_milvus_store.py tests/test_config.py dev-notes/ch04.md
git commit -m "feat: add hybrid Milvus collection with native BM25"
```

---

### Task 2: 向量化元数据与全量迁移 job

**Files:**
- Modify: `app/rag/vectorize.py`
- Create: `app/jobs/reindex_hybrid.py`
- Test: `tests/test_vectorize.py`, `tests/test_reindex_hybrid.py`

**Interfaces:**
- Consumes: `VectorItem` from Task 1
- Produces: `vectorize_all(session, embedder, store, batch_size=16) -> VectorizeReport`
- Produces: `python -m app.jobs.reindex_hybrid [--recreate] [--yes]`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_vectorize.py`:
```python
def test_vectorize_pending_writes_bm25_text_and_metadata(session):
    c = _chunk(session)
    s = FakeStore(); vectorize_pending(session, FakeEmbedder(), s)
    item = s.items[0]
    assert item.text == "售后\n问\n答"
    assert item.category == "售后" and item.content_type == "faq"
    assert item.section_path == "文档 > 节"

def test_vectorize_all_reprocesses_done_rows(session):
    c = _chunk(session, status="done")
    e = FakeEmbedder(); s = FakeStore()
    r = vectorize_all(session, e, s)
    assert r.processed == 1 and s.items[0].chunk_id == c.id
```
Create `tests/test_reindex_hybrid.py`:
```python
def test_recreate_requires_yes(monkeypatch, capsys):
    from app.jobs import reindex_hybrid
    assert reindex_hybrid.main(["--recreate"]) == 2
    assert "--yes" in capsys.readouterr().out

def test_recreate_calls_store_recreate(monkeypatch):
    called = []
    monkeypatch.setattr(reindex_hybrid, "get_vector_store", lambda: FakeStore(recreate=called))
    assert reindex_hybrid.main(["--recreate", "--yes", "--limit", "0"]) == 0
    assert called == ["recreate"]
```
(Use a fake session/embedder/store as the existing test style; `--limit 0` means no rows.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_vectorize.py tests/test_reindex_hybrid.py -q`
Expected: FAIL — `vectorize_all` missing; reindex module missing.

- [ ] **Step 3: Implement vectorize + reindex**

In `app/rag/vectorize.py`:
- `vectorize_pending` builds `VectorItem(..., text=vectorization_text(row), category=row.category, content_type=row.content_type, section_path=row.section_path)`.
- Factor `_vectorize_rows(rows, ...)`; `vectorize_all` selects all rows in id order (no status filter), calls the same batching, returns `VectorizeReport`.
- `vectorize_all` does not change `vectorize_status` semantics beyond marking rows `done` (they already are); it must be idempotent by `chunk_{id}` upsert.

Create `app/jobs/reindex_hybrid.py`:
- `main(argv=None) -> int`; args `--recreate`, `--yes`, `--limit N` (optional, for smoke).
- If `--recreate` without `--yes`: print `refusing to recreate without --yes` and return 2.
- If `--recreate --yes`: `store.recreate_collection()`.
- Open a session; call `vectorize_all`; print `{"chunks": ..., "embedded": ..., "upserted": ...}` JSON; return 0.
- Module-level names `get_vector_store`, `get_embedder`, `get_session_factory` for monkeypatch.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_vectorize.py tests/test_reindex_hybrid.py -q`
Expected: all pass.

- [ ] **Step 5: Append dev-notes and commit**

Append `## 阶段:Task 2 完成(全量迁移到 knowledge_hybrid)` with outputs, rejections (did not copy old vectors; re-embed chosen), rework, test counts.

```bash
git add app/rag/vectorize.py app/jobs/reindex_hybrid.py tests/test_vectorize.py tests/test_reindex_hybrid.py dev-notes/ch04.md
git commit -m "feat: reindex knowledge chunks into hybrid collection"
```

---

### Task 3: 硅基流动 reranker 客户端

**Files:**
- Create: `app/rag/rerank.py`
- Test: `tests/test_rerank.py`

**Interfaces:**
- Produces: `RerankHit(index: int, score: float)`
- Produces: `RerankError`
- Produces: `SiliconFlowReranker(base_url, api_key, model, timeout_s=10.0, max_retries=2, client=None, sleep=time.sleep).rerank(query, documents, top_n=None) -> list[RerankHit]`
- Produces: `get_reranker() -> SiliconFlowReranker` (falls back from `RERANK_API_KEY`/`RERANK_BASE_URL` to embedding settings)

- [ ] **Step 1: Write the failing tests**

```python
def test_rerank_posts_expected_payload_and_sorts():
    client = FakeClient(json={"results": [
        {"index": 1, "relevance_score": 0.9},
        {"index": 0, "relevance_score": 0.2},
    ]})
    r = SiliconFlowReranker("https://api.siliconflow.cn/v1", "k", "BAAI/bge-reranker-v2-m3", client=client)
    hits = r.rerank("q", ["a", "b"], top_n=2)
    assert client.last["json"] == {"model": "BAAI/bge-reranker-v2-m3", "query": "q", "documents": ["a", "b"], "top_n": 2, "return_documents": False}
    assert [(h.index, h.score) for h in hits] == [(1, 0.9), (0, 0.2)]

def test_rerank_retries_429_then_raises():
    client = FakeClient(statuses=[429, 200], json={"results": [{"index": 0, "relevance_score": 0.8}]})
    sleeps = []
    r = SiliconFlowReranker("u", "k", "m", client=client, sleep=sleeps.append)
    assert r.rerank("q", ["a"])[0].score == 0.8
    assert sleeps == [0.5]

def test_rerank_rejects_out_of_range_index():
    client = FakeClient(json={"results": [{"index": 9, "relevance_score": 0.5}]})
    with pytest.raises(RerankError):
        SiliconFlowReranker("u", "k", "m", client=client).rerank("q", ["a"])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_rerank.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `app/rag/rerank.py`**

Mirror `app/rag/embedding.py` retry/timeout patterns. Use `httpx.Client(timeout=timeout_s)`. Retry only 429/5xx/`httpx.TransportError`; other statuses raise `RerankError` immediately. `get_reranker()` resolves `rerank_base_url or embedding_base_url`, `rerank_api_key or embedding_api_key`, and raises `RerankError` when no key.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_rerank.py -q`
Expected: all pass.

- [ ] **Step 5: Append dev-notes and commit**

Append `## 阶段:Task 3 完成(reranker 客户端)` with outputs, rejections (no local model), rework, test counts.

```bash
git add app/rag/rerank.py tests/test_rerank.py dev-notes/ch04.md
git commit -m "feat: add SiliconFlow bge reranker client"
```

---

### Task 4: Query 理解与 splitter category 优先

**Files:**
- Create: `app/rag/query_understanding.py`
- Modify: `app/rag/splitter.py`
- Test: `tests/test_query_understanding.py`, `tests/test_splitter_category.py`

**Interfaces:**
- Produces: `QueryAnalysis(original, rewritten, keywords, rewritten_by)`
- Produces: `SYNONYMS: dict[str, tuple[str, ...]]`
- Produces: `expand_synonyms(text: str) -> list[str]`
- Produces: `analyze(query: str, llm=None) -> QueryAnalysis`
- Modifies: `_build_chunk` uses `doc.category` when present for all content types

- [ ] **Step 1: Write the failing tests**

`tests/test_query_understanding.py`:
```python
class FakeLLM:
    def __init__(self, payload): self.payload = payload
    def with_structured_output(self, schema, method=None): return self
    def invoke(self, messages): return self.payload

def test_analyze_returns_rewrite_and_keywords():
    a = analyze("邮费咋算", llm=FakeLLM(QueryRewrite(rewritten="运费怎么计算", keywords=["运费", "配送费"])))
    assert a.rewritten == "运费怎么计算" and a.keywords == ["运费", "配送费"] and a.rewritten_by == "llm"

def test_analyze_falls_back_to_original_and_static_synonyms():
    class Boom:
        def with_structured_output(self, *a, **k): raise RuntimeError("down")
    a = analyze("邮费咋算", llm=Boom())
    assert a.rewritten == "邮费咋算" and "运费" in a.keywords and a.rewritten_by == "fallback"

def test_expand_synonyms_skips_terms_already_present():
    assert expand_synonyms("邮费 运费") == ["配送费"]
```

`tests/test_splitter_category.py`:
```python
def test_manual_frontmatter_category_wins(tmp_path):
    p = tmp_path / "m.md"
    p.write_text("---\ncontent_type: manual\ncategory: 商品\n---\n# 商品型号\n\n## LX-A100 落地灯\n\n色温 3000K\n", encoding="utf-8")
    chunk = split_document(load_document(p))[0]
    assert chunk.category == "商品"
    assert chunk.section_path == "商品型号 > LX-A100 落地灯"

def test_manual_without_frontmatter_category_keeps_heading_path(tmp_path):
    # same file without category -> category == "商品型号"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_query_understanding.py tests/test_splitter_category.py -q`
Expected: FAIL — module missing / category still `商品型号`.

- [ ] **Step 3: Implement query understanding and splitter change**

`app/rag/query_understanding.py`:
- Pydantic `QueryRewrite(rewritten: str, keywords: list[str])`.
- `SYNONYMS` covers at least 邮费→运费/配送费, 快递费→运费, 到账→退款/原路退回, 灯不亮→故障/维修, 换货→换新.
- `expand_synonyms` scans original text for keys, returns deduped expansion terms not already present.
- `analyze` calls `llm.with_structured_output(QueryRewrite, method="json_mode")`; on any exception returns `QueryAnalysis(original=query, rewritten=query, keywords=expand_synonyms(query), rewritten_by="fallback")`.
- Default `llm`: `build_llm(get_settings(), temperature=0.0)` resolved lazily inside `analyze` to avoid import cycles.

`app/rag/splitter.py`:
```python
category = doc.category or (doc.title if doc.content_type == "faq" else (_PATH_SEP.join(path[:-1]) if len(path) > 1 else doc.title))
```
Keep `questions=leaf`, `section_path` unchanged.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_query_understanding.py tests/test_splitter_category.py tests/test_splitter_sections.py tests/test_splitter_long.py -q`
Expected: all pass.

- [ ] **Step 5: Append dev-notes and commit**

Append `## 阶段:Task 4 完成(Query 理解 + category 优先)` with outputs, rejections (no LLM-inferred filters; no storage-side synonym fan-out), rework, test counts.

```bash
git add app/rag/query_understanding.py app/rag/splitter.py tests/test_query_understanding.py tests/test_splitter_category.py dev-notes/ch04.md
git commit -m "feat: add query rewrite and frontmatter category priority"
```

---

### Task 5: 四策略检索入口

**Files:**
- Modify: `app/rag/retrieval.py`
- Test: `tests/test_retrieval.py`

**Interfaces:**
- Consumes: `QueryAnalysis`, `analyze` (Task 4); `SiliconFlowReranker` (Task 3); store methods (Task 1)
- Produces: `RetrievalStrategy` (`dense`, `bm25`, `hybrid`, `hybrid_rerank`)
- Produces: `RetrievedChunk(chunk_id, score, questions, answer, section_path, category, content_type)`
- Produces: `RetrievalResult(chunks, strategy, analysis, rerank_failed=False)`
- Produces: `retrieve_hybrid(query, *, strategy=..., session=None, embedder=None, store=None, reranker=None, analyzer=None, analysis=None, category=None, content_type=None, recall_top_k=None, top_k=None) -> RetrievalResult`
- Produces: `retrieve(keyword, *, session=None, embedder=None, store=None, top_k=None, threshold=None) -> list[RetrievedChunk]` (wrapper)

- [ ] **Step 1: Write the failing tests**

Keep the existing `retrieve()` threshold/session tests (they pin the compatibility wrapper) and add the new strategy tests around fakes:
```python
def test_dense_strategy_only_calls_dense_search(session):
    row = _row(session, "问", "答")
    store = FakeHybridStore(dense=[SearchHit(row.id, 0.8)])
    result = retrieve_hybrid("q", strategy=RetrievalStrategy.DENSE, session=session,
                             embedder=FakeEmbedder(), store=store, analysis=FIXED_ANALYSIS)
    assert [c.chunk_id for c in result.chunks] == [row.id]
    assert store.dense_calls and not store.bm25_calls and not store.hybrid_calls

def test_bm25_strategy_uses_expanded_query(session):
    row = _row(session, "问", "答")
    store = FakeHybridStore(bm25=[SearchHit(row.id, 1.2)])
    retrieve_hybrid("邮费", strategy=RetrievalStrategy.BM25, session=session,
                    embedder=FakeEmbedder(), store=store,
                    analysis=QueryAnalysis("邮费", "运费怎么算", ["运费", "配送费"], "llm"))
    assert "运费" in store.bm25_calls[0][0]

def test_hybrid_rerank_sorts_by_rerank_score_and_flags_failure(session):
    a, b = _row(session, "q1", "a1"), _row(session, "q2", "a2")
    store = FakeHybridStore(hybrid=[SearchHit(a.id, 0.9), SearchHit(b.id, 0.8)])
    reranker = FakeReranker([RerankHit(1, 0.99), RerankHit(0, 0.1)])
    result = retrieve_hybrid("q", strategy=RetrievalStrategy.HYBRID_RERANK, session=session,
                             embedder=FakeEmbedder(), store=store, reranker=reranker,
                             analysis=FIXED_ANALYSIS)
    assert [c.chunk_id for c in result.chunks] == [b.id, a.id]
    assert result.rerank_failed is False

def test_hybrid_rerank_failure_falls_back_and_sets_flag(session):
    row = _row(session, "q", "a")
    store = FakeHybridStore(hybrid=[SearchHit(row.id, 0.9)])
    result = retrieve_hybrid("q", strategy=RetrievalStrategy.HYBRID_RERANK, session=session,
                             embedder=FakeEmbedder(), store=store, reranker=BoomReranker(),
                             analysis=FIXED_ANALYSIS)
    assert result.rerank_failed is True and [c.chunk_id for c in result.chunks] == [row.id]

def test_filter_expr_is_passed_to_both_branches(session):
    row = _row(session, "q", "a")
    store = FakeHybridStore(hybrid=[SearchHit(row.id, 0.9)])
    retrieve_hybrid("q", strategy=RetrievalStrategy.HYBRID, session=session,
                    embedder=FakeEmbedder(), store=store, category="商品",
                    content_type="manual", analysis=FIXED_ANALYSIS)
    expr = store.hybrid_calls[0][1]
    assert 'category == "商品"' in expr and 'content_type == "manual"' in expr

def test_injected_analysis_skips_analyzer(session):
    row = _row(session, "q", "a")
    def boom(_): raise AssertionError("analyzer must not be called")
    retrieve_hybrid("q", session=session, embedder=FakeEmbedder(),
                    store=FakeHybridStore(dense=[SearchHit(row.id, 0.9)]),
                    analyzer=boom, analysis=FIXED_ANALYSIS)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_retrieval.py -q`
Expected: FAIL — missing `RetrievalResult` / `retrieve_hybrid` / strategy.

- [ ] **Step 3: Implement `retrieve_hybrid`**

Rules:
- `analysis = analysis or (analyzer or analyze)(query)`.
- Build expr with exact quoted values; use `expr_params` if the client supports it, otherwise escape `"` and `\`.
- `dense`: `store.dense_search(embedder.embed_one(analysis.rewritten), recall_k, expr)`; cut `top_k`.
- `bm25`: query text = `" ".join([analysis.rewritten, *analysis.keywords])`; `store.bm25_search(...)`; cut `top_k`.
- `hybrid`: `store.hybrid_search(vector, bm25_text, recall_k, expr, rrf_k)`; cut `top_k`.
- `hybrid_rerank`: hybrid candidates → MySQL rows → `reranker.rerank(analysis.rewritten, [f"{section_path}\n{questions}\n{answer}"...], top_n=top_k)`; on `RerankError` use hybrid order and `rerank_failed=True`.
- MySQL backfill with one `in_` query preserving order; skip missing ids.
- `retrieve()` wrapper: if `threshold is not None`, run the legacy dense path (embed → `store.dense_search` → filter `score >= threshold` → MySQL backfill) so existing ch03 callers/tests keep their contract; otherwise return `retrieve_hybrid(..., strategy=HYBRID_RERANK).chunks`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_retrieval.py -q`
Expected: all pass.

- [ ] **Step 5: Append dev-notes and commit**

Append `## 阶段:Task 5 完成(四策略检索)` with outputs, rejections (no cross-strategy score comparison; rerank failure not hidden), rework, test counts.

```bash
git add app/rag/retrieval.py tests/test_retrieval.py dev-notes/ch04.md
git commit -m "feat: add four retrieval strategies with hybrid and rerank"
```

---

### Task 6: 引用收集器、首尾排序与 query_faq 改造

**Files:**
- Modify: `app/tools/context.py`, `app/tools/faq.py`, `app/tools/registry.py`
- Test: `tests/test_citation_collector.py`, `tests/test_tools_faq.py`
- Modify existing assertions: `tests/test_chat.py` (`query_faq` timeout 25.0), `tests/test_tools_business.py` (`tool_timeout("query_faq") == 25.0`)

**Interfaces:**
- Consumes: `retrieve_hybrid`, `RetrievedChunk` (Task 5)
- Produces: `CitationCollector.add(chunks) -> list[int]`, `.payload() -> list[dict]`
- Produces: `current_citation_collector: ContextVar[CitationCollector | None]`
- Produces: `order_for_prompt(items: Sequence[RetrievedChunk]) -> list[RetrievedChunk]`
- Produces: `query_faq(keyword, category=None, content_type=None) -> str`
- Produces: `MISS_TEXT`, `ERROR_TEXT`
- Modifies: `TOOL_TIMEOUTS["query_faq"] = 25.0`

- [ ] **Step 1: Write the failing tests**

`tests/test_citation_collector.py`:
```python
def test_order_for_prompt_puts_strongest_first_and_last():
    items = [make_chunk(i) for i in range(5)]
    assert [c.chunk_id for c in order_for_prompt(items)] == [0, 2, 4, 3, 1]

def test_collector_dedupes_and_numbers_globally():
    c = CitationCollector()
    assert c.add([make_chunk(1), make_chunk(2)]) == [1, 2]
    assert c.add([make_chunk(2), make_chunk(3)]) == [2, 3]
    assert [s["n"] for s in c.payload()] == [1, 2, 3]

def test_collector_mutation_is_visible_from_execute_tool():
    collector = CitationCollector()
    current_citation_collector.set(collector)
    @tool
    def fake_tool() -> str:
        collector.add([make_chunk(7)])
        return "ok"
    assert execute_tool(fake_tool, {}) == "ok"
    assert collector.payload()[0]["chunk_id"] == 7
```

`tests/test_tools_faq.py`:
```python
def test_query_faq_returns_cited_evidence(monkeypatch):
    monkeypatch.setattr("app.tools.faq.retrieve_hybrid", lambda *a, **k: RetrievalResult([make_chunk(1), make_chunk(2)], RetrievalStrategy.HYBRID_RERANK, FIXED_ANALYSIS))
    out = query_faq.func("完全不相关的问题")
    assert "[1] 章节:" in out and "[2] 章节:" in out
    assert "完全不相关的问题" not in out  # evidence, not the raw query echo

def test_query_faq_passes_filters(monkeypatch):
    calls = []
    monkeypatch.setattr("app.tools.faq.retrieve_hybrid", lambda q, **k: calls.append(k) or RetrievalResult([], RetrievalStrategy.HYBRID_RERANK, FIXED_ANALYSIS))
    query_faq.func("色温", category="商品", content_type="manual")
    assert calls[0] == {"category": "商品", "content_type": "manual"}

def test_query_faq_miss_sets_faq_called(monkeypatch):
    collector = CitationCollector(); current_citation_collector.set(collector)
    monkeypatch.setattr("app.tools.faq.retrieve_hybrid", lambda *a, **k: RetrievalResult([], RetrievalStrategy.HYBRID_RERANK, FIXED_ANALYSIS))
    assert query_faq.func("不存在") == MISS_TEXT
    assert collector.faq_called is True and collector.payload() == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_citation_collector.py tests/test_tools_faq.py -q`
Expected: FAIL — missing collector/order; `query_faq` still uses `retrieve`.

- [ ] **Step 3: Implement collector, order, and tool**

- `CitationCollector` stores dicts with keys `n, chunk_id, section_path, question, answer, category, content_type, score`.
- `order_for_prompt` alternates first/next-best to start and end.
- `query_faq` uses `retrieve_hybrid`, `order_for_prompt`, collector; returns `_MISS_TEXT`/`_ERROR_TEXT` constants. If no collector is set, create a local one.
- Update tool docstring to explain `[n]`, filtering, and "不确定就不要过滤".
- Registry timeout 25.0; update `tests/test_chat.py` and `tests/test_tools_business.py` expectations from 8.0 to 25.0.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_citation_collector.py tests/test_tools_faq.py tests/test_chat.py tests/test_tools_business.py -q`
Expected: all pass.

- [ ] **Step 5: Append dev-notes and commit**

Append `## 阶段:Task 6 完成(引用收集 + query_faq)` with outputs, rejections (no per-call numbering reset), rework, test counts.

```bash
git add app/tools/context.py app/tools/faq.py app/tools/registry.py tests/test_citation_collector.py tests/test_tools_faq.py dev-notes/ch04.md
git commit -m "feat: emit cited evidence from query_faq"
```

---

### Task 7: 自评与低置信度池

**Files:**
- Create: `app/rag/self_check.py`, `app/services/low_confidence.py`
- Test: `tests/test_self_check.py`, `tests/test_low_confidence.py`

**Interfaces:**
- Produces: `EvidenceAssessment(useful: bool, reason: str)`
- Produces: `assess_evidence(llm, question, sources, retries=1) -> EvidenceAssessment`
- Produces: `record_low_confidence(session, conversation_id, raw_question, reason, source="self_check") -> LowConfidenceQuestion`

- [ ] **Step 1: Write the failing tests**

```python
def test_assess_evidence_returns_useful_false():
    llm = FakeStructuredLLM(EvidenceAssessment(useful=False, reason="只沾边"))
    a = assess_evidence(llm, "支持以旧换新吗", [{"n": 1, "answer": "退货政策"}])
    assert a.useful is False and a.reason == "只沾边"

def test_assess_evidence_retries_once_then_raises():
    llm = FlakyLLM(fail_times=2)
    with pytest.raises(RuntimeError):
        assess_evidence(llm, "q", [])

def test_record_low_confidence_inserts_self_check_row(session):
    row = record_low_confidence(session, None, "支持以旧换新吗", "证据不足")
    session.commit()
    assert row.source == "self_check" and row.raw_question.endswith("吗")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_self_check.py tests/test_low_confidence.py -q`
Expected: FAIL — modules missing.

- [ ] **Step 3: Implement self-check and low-confidence service**

`assess_evidence`:
- Build a JSON-mode structured output with `EvidenceAssessment`.
- Prompt lists question and each source (`n`, `section_path`, `answer`, `score`), and says "证据不足/只沾边/互相矛盾/需要库外知识 → useful=false".
- Retry `retries` times; re-raise the last error.
`record_low_confidence`: add row, flush, return; caller commits.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_self_check.py tests/test_low_confidence.py -q`
Expected: all pass.

- [ ] **Step 5: Append dev-notes and commit**

Append `## 阶段:Task 7 完成(自评 + 低置信度池)` with outputs, rejections (infra failures not pooled), rework, test counts.

```bash
git add app/rag/self_check.py app/services/low_confidence.py tests/test_self_check.py tests/test_low_confidence.py dev-notes/ch04.md
git commit -m "feat: add evidence self-check and low-confidence pool"
```

---

### Task 8: Chat 编排(sources 帧、拒答、混合意图、引用校验)

**Files:**
- Modify: `app/services/chat.py`, `app/main.py`
- Test: `tests/test_chat_rag.py` (new), `tests/test_chat.py` (keep green)

**Interfaces:**
- Consumes: collector (Task 6), `assess_evidence` / `record_low_confidence` (Task 7)
- Produces: `stream_chat(..., self_check_llm=None)`
- Produces: SSE frame `{"type":"sources","sources":[...]}`
- Produces: `get_self_check_llm_dep` in `app/main.py`

- [ ] **Step 1: Write the failing tests**

Use fake LLM/tool/session; assert frames:
```python
def test_query_faq_emits_sources_frame_and_refuses_when_useful_false(...):
    frames = list(stream_chat(..., self_check_llm=FakeAssessment(False, "证据不足")))
    assert any(json.loads(f)["type"] == "sources" for f in frames if f.startswith("{"))
    assert any("知识库暂未收录" in f for f in frames)
    assert session.query(LowConfidenceQuestion).count() == 1

def test_query_order_without_faq_skips_self_check(...):
    # tool call is query_order only; no sources frame; final answer streams
    assert not any('"type":"sources"' in f for f in frames)

def test_infra_error_refuses_without_pool(...):
    # query_faq returns ERROR_TEXT; assert "知识库暂时不可用" and pool count 0

def test_mixed_intent_keeps_other_tool_answer(...):
    # query_order + query_faq; assessment false; assert final generation called
    # and a SystemMessage containing "无法确认" was passed

def test_unknown_citation_is_logged_not_blocking(caplog, ...):
    # final answer contains [9] but sources only [1]; assert warning + normal completion
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_chat_rag.py -q`
Expected: FAIL — no sources frame/refusal.

- [ ] **Step 3: Implement chat orchestration**

In `stream_chat`:
- Create `CitationCollector`, set contextvar before tool execution.
- For `query_faq`, call `execute_tool(..., timeout_s=25, retries=0)`.
- After tools, snapshot sources; yield sources frame if `faq_called`.
- Distinguish `ERROR_TEXT`/`工具执行失败` (infra) from `MISS_TEXT`.
- Run `assess_evidence(self_check_llm or llm, message, sources)`.
- `useful=false`: record pool; if other non-faq tool results exist, append SystemMessage constraint and continue; else yield refusal and return.
- Self-check failure: conservative refusal, no pool; same mixed-intent path.
- After final generation, regex `\[(\d+)\]`; log unknown numbers.
- Persist the assistant message on every refusal path (pool-backed refusal, infra refusal, self-check-failure refusal) before returning; persist final answers as before.
In `main.py`: add `get_self_check_llm_dep` and pass to `stream_chat`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_chat_rag.py tests/test_chat.py -q`
Expected: all pass.

- [ ] **Step 5: Append dev-notes and commit**

Append `## 阶段:Task 8 完成(chat sources/自评/拒答/混合意图)` with outputs, rejections (no whole-turn refusal for mixed intent; no pool on infra), rework, test counts.

```bash
git add app/services/chat.py app/main.py tests/test_chat.py tests/test_chat_rag.py dev-notes/ch04.md
git commit -m "feat: gate chat answers on evidence self-check"
```

---

### Task 9: System Prompt 与改写/自评标注样例验证

**Files:**
- Modify: `app/prompts/customer_service.py`
- Create: `app/jobs/eval_prompts.py`, `tests/eval/query_rewrite_samples.json`, `tests/eval/self_check_samples.json`
- Test: `tests/test_prompt.py`, `tests/test_eval_prompts.py`

**Interfaces:**
- Produces: updated `render_system_prompt(business_name="本店") -> str`
- Produces: `render_answer_system_prompt(business_name="本店") -> str` — only the knowledge/citation/negative constraints, no tool instructions, for eval generation
- Produces: `python -m app.jobs.eval_prompts --suite rewrite|self_check --samples PATH`
- Sample schemas:
  - rewrite: `[{"query": "...", "expect_rewritten_any": ["..."], "expect_keywords_any": ["..."]}]`
  - self_check: `[{"query": "...", "sources": [...], "expect_useful": true/false}]`

- [ ] **Step 1: Write the failing prompt/data tests**

```python
def test_prompt_contains_citation_and_negative_constraints():
    p = render_system_prompt()
    assert "[n]" in p and "不得编造" in p
    assert "不承诺具体到账时间" in p
    assert "query_product 只用于" in p

def test_eval_prompt_samples_are_valid_json():
    for path in (REWRITE_SAMPLES, SELF_CHECK_SAMPLES):
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["samples"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_prompt.py tests/test_eval_prompts.py -q`
Expected: FAIL — missing constraints/samples.

- [ ] **Step 3: Update prompt, samples, and eval_prompts job**

- Add the spec's 【知识库与引用】 and 【禁止承诺的负面知识】 blocks; route model/spec questions to `query_faq`, not `query_product`.
- Factor those two blocks into `render_answer_system_prompt()` so eval generation can reuse the exact same constraints without the tool list.
- Create 10 rewrite samples and 10 self-check samples (5 answerable + 5 unanswerable, including "支持以旧换新吗" and "LX-Z900 的色温").
- `eval_prompts.py` runs the real `analyze()` / `assess_evidence()` and prints PASS/FAIL per sample; exits non-zero if any fail. It is a manual validation job, no import-time network.

- [ ] **Step 4: Run the prompt tests and the real sample validation**

Run:
```bash
.venv/Scripts/python.exe -m pytest tests/test_prompt.py tests/test_eval_prompts.py -q
.venv/Scripts/python.exe -m app.jobs.eval_prompts --suite rewrite --samples tests/eval/query_rewrite_samples.json
.venv/Scripts/python.exe -m app.jobs.eval_prompts --suite self_check --samples tests/eval/self_check_samples.json
```
Expected: pytest pass; both suites report `10 passed, 0 failed`. Record the actual outputs in dev-notes. If a sample fails, adjust the prompt (not the sample) and rerun.

- [ ] **Step 5: Append dev-notes and commit**

Append `## 阶段:Task 9 完成(Prompt + 标注样例验证)` with the user's prompt-quality requirement, outputs, rejections (did not weaken samples to make them pass), rework, evidence.

```bash
git add app/prompts/customer_service.py app/jobs/eval_prompts.py tests/eval/query_rewrite_samples.json tests/eval/self_check_samples.json tests/test_prompt.py tests/test_eval_prompts.py dev-notes/ch04.md
git commit -m "feat: add citation-aware customer service prompt"
```

---

### Task 10: 型号知识文档与分桶评估集

**Files:**
- Create: `knowledge/商品型号与参数.md`, `tests/eval/rag_cases.json`
- Test: `tests/test_eval_cases.py`

**Interfaces:**
- Produces: 35-case eval JSON with `id/bucket/difficulty/query/answerable/ground_truth/filters/notes`
- Produces: model doc chunks with `section_path` values used by ground truth

- [ ] **Step 1: Write the failing data-validation test**

```python
def test_eval_cases_reference_real_section_paths():
    docs = load_documents(Path("knowledge"))
    chunks = [c for d in docs for c in split_document(d)]
    paths = {c.section_path for c in chunks}
    data = json.loads(Path("tests/eval/rag_cases.json").read_text(encoding="utf-8"))
    cases = data["cases"]
    assert len(cases) == 35
    assert len({c["id"] for c in cases}) == 35
    assert {c["bucket"] for c in cases} == {"A_policy", "B_model", "C_colloquial", "D_unanswerable", "E_multi"}
    for c in cases:
        assert c["difficulty"] in {"easy", "medium", "hard"}
        assert c["answerable"] is (c["bucket"] != "D_unanswerable")
        for p in c.get("ground_truth", {}).get("section_paths", []):
            assert p in paths, (c["id"], p)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_eval_cases.py -q`
Expected: FAIL — missing `rag_cases.json` / paths.

- [ ] **Step 3: Create the model doc and 35-case eval set**

`knowledge/商品型号与参数.md`:
- frontmatter `content_type: manual`, `category: 商品`
- H1 `商品型号与参数`
- Sections `LX-A100 落地灯`, `LX-B200 台灯`, `LX-C300 吸顶灯`, `型号对照`
- Each includes 色温、功率、适用面积、保修期、调光、包装清单; exact model tokens in headings and body.

`tests/eval/rag_cases.json`:
- 7 per bucket, each with easy/medium/hard spread.
- A_policy: 退货时限、退款说明、运费标准、退换货运费、发货时效、投诉处理、上门取件.
- B_model: LX-A100/B200/C300 的色温、功率、保修、调光、包装清单.
- C_colloquial: 邮费咋算、钱啥时候退、灯不亮了、想换个大点的、快递一直不动、发票怎么搞、能不能货到付款.
- D_unanswerable: 支持以旧换新吗、LX-Z900 色温、增值税专票、旧机回收、企业月结、延保购买、会员积分抵现.
- E_multi: 型号 + 保修 + 运费组合;退货时限 + 退款到账;换货 + 维修时限等.
- Include 3 filter cases with `"filters": {"category": "商品"}` or `{"content_type": "manual"}`.

- [ ] **Step 4: Run the data validation test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_eval_cases.py -q`
Expected: PASS. If a ground-truth path fails, fix the case to the actual splitter path (do not change the doc to fit a wrong path unless the doc genuinely needs the section).

- [ ] **Step 5: Append dev-notes and commit**

Append `## 阶段:Task 10 完成(型号文档 + 35 条评估集)` with the bucket/difficulty distribution, any path corrections, rejections (no vague "looks relevant" ground truth), evidence.

```bash
git add knowledge/商品型号与参数.md tests/eval/rag_cases.json tests/test_eval_cases.py dev-notes/ch04.md
git commit -m "test: add model knowledge and 35-case retrieval eval set"
```

---

### Task 11: Faithfulness 裁判与编造个案台账

**Files:**
- Create: `app/services/faithfulness.py`, `app/services/faith_cases.py`
- Test: `tests/test_faithfulness.py`, `tests/test_faith_cases.py`

**Interfaces:**
- Produces: `FaithfulnessVerdict(faithful: bool, unsupported_claims: list[str], reason: str)`
- Produces: `judge_faithfulness(llm, question, answer, citations, retries=1) -> FaithfulnessVerdict`
- Produces: `FaithCasePayload(eval_id, bucket, query, strategy, answer, reason, citations, judge_model)`
- Produces: `upsert_faith_case(session, payload) -> FaithCase`

- [ ] **Step 1: Write the failing tests**

```python
def test_judge_returns_unfaithful_verdict():
    llm = FakeStructuredLLM(FaithfulnessVerdict(faithful=False, unsupported_claims=["承诺当天到账"], reason="证据没有"))
    v = judge_faithfulness(llm, "退款多久", "当天到账", [{"n": 1, "answer": "1-3 个工作日"}])
    assert v.faithful is False and v.unsupported_claims == ["承诺当天到账"]

def test_upsert_inserts_then_increments_and_reopens(session):
    p = FaithCasePayload("A01", "A_policy", "退款多久", "hybrid_rerank", "答案", "编造", [{"n": 1}], "deepseek-chat")
    first = upsert_faith_case(session, p); session.commit()
    assert first.seen_count == 1 and first.status == "未解决"
    first.status = "已解决"; first.resolved_at = datetime(2026, 11, 1); session.commit()
    again = upsert_faith_case(session, p); session.commit()
    assert again.id == first.id and again.seen_count == 2
    assert again.status == "未解决" and again.resolved_at == datetime(2026, 11, 1)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_faithfulness.py tests/test_faith_cases.py -q`
Expected: FAIL — modules missing.

- [ ] **Step 3: Implement judge and upsert**

- Judge prompt: "只依据证据逐句判断事实性陈述;列出无证据支持的句子;JSON only"; use `temperature=0.0`.
- Upsert: select by `eval_id`; insert or update; `seen_count += 1`; `last_seen_at = datetime.now()`; `status = "未解决"`; keep `resolved_at`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_faithfulness.py tests/test_faith_cases.py -q`
Expected: all pass.

- [ ] **Step 5: Append dev-notes and commit**

Append `## 阶段:Task 11 完成(Faithfulness + faith_cases)` with outputs, rejections (one row per eval_id, no per-strategy duplicate rows), rework, test counts.

```bash
git add app/services/faithfulness.py app/services/faith_cases.py tests/test_faithfulness.py tests/test_faith_cases.py dev-notes/ch04.md
git commit -m "feat: add faithfulness judge and fabrication ledger"
```

---

### Task 12: 四策略评估 job 与报告

**Files:**
- Create: `app/jobs/eval_rag.py`
- Modify: `tests/test_eval_scoring.py`
- Test: `tests/test_eval_scoring.py`, `tests/test_eval_rag.py`

**Interfaces:**
- Consumes: `retrieve_hybrid` (Task 5), `assess_evidence` (Task 7), `judge_faithfulness` / `upsert_faith_case` (Task 11), eval set (Task 10)
- Produces: `section_paths_in_order(chunks) -> list[str]`
- Produces: `score_retrieval_case(case, chunks) -> RetrievalScore`
- Produces: `citation_validity(answer, sources) -> float | None`
- Produces: `build_eval_answer_messages(question: str, evidence: list[dict]) -> list[BaseMessage]` (uses `render_answer_system_prompt()`, no tools bound)
- Produces: `python -m app.jobs.eval_rag --cases ... --strategies ... --out ... [--limit N] [--retrieval-only] [--no-faith-cases]`

- [ ] **Step 1: Write the failing scoring tests**

```python
def test_recall_dedupes_same_section_and_handles_multi_hop():
    chunks = [make_chunk(1, "A > X"), make_chunk(2, "A > X"), make_chunk(3, "A > Y")]
    case = {"answerable": True, "ground_truth": {"section_paths": ["A > X", "A > Y"]}}
    s = score_retrieval_case(case, chunks[:2])
    assert s.recall_at_5 == 0.5 and s.mrr_at_10 == 1.0

def test_citation_validity_counts_unknown_numbers():
    assert citation_validity("见 [1] 和 [9]", [{"n": 1}]) == 0.5
    assert citation_validity("没有编号", [{"n": 1}]) is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_eval_scoring.py tests/test_eval_rag.py -q`
Expected: FAIL — missing functions/CLI.

- [ ] **Step 3: Implement `eval_rag.py`**

- Build one `judge_llm = build_llm(get_settings(), temperature=0.0)` and reuse it for both `assess_evidence` and `judge_faithfulness`.
- Precompute one `analyze(case.query)` per case; pass `analysis=` to all four strategies.
- Retrieval: top-10; section-level Recall@5/@10 and MRR@10; rerank_failed exclusion; D bucket separate.
- Generation: `assess_evidence` first (all cases); refusal on false; otherwise generate with `build_eval_answer_messages(question, evidence)` (same system constraints, no tools), judge faithfulness, upsert `faith_cases` when unfaithful.
- Report markdown tables per strategy and bucket; include refusal/faithfulness/citation validity/fallback counts/run parameters.
- `--limit`, `--retrieval-only`, `--no-faith-cases`, `--out`; write report to the path.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_eval_scoring.py tests/test_eval_rag.py -q`
Expected: all pass.

- [ ] **Step 5: Run a limited live evaluation**

Prerequisites: Task 1 live smoke passed, Task 2 reindex ran, Task 10 docs/eval set present, Task 9 prompts validated.
```bash
.venv/Scripts/python.exe -m app.jobs.eval_rag --limit 3 --out /tmp/ch04-eval-smoke.md
```
Expected: exit 0; report has four strategy rows; no tracebacks. Record the numbers in dev-notes. Then run the full 35-case report only when the user is ready for the long run.

- [ ] **Step 6: Append dev-notes and commit**

Append `## 阶段:Task 12 完成(四策略评估 job)` with the smoke numbers, rejections (did not hide rerank fallback or chunk double-counting), rework, evidence.

```bash
git add app/jobs/eval_rag.py tests/test_eval_scoring.py tests/test_eval_rag.py dev-notes/ch04.md
git commit -m "feat: add four-strategy RAG evaluation job"
```

---

### Task 13: 前端引用与满意度反馈(Vibe Coding)

**Files:**
- Modify: `frontend/src/types.ts`, `frontend/src/api/chat.ts`, `frontend/src/hooks/useChat.ts`, `frontend/src/components/MessageItem.tsx`, `frontend/src/components/MessageList.tsx`, `frontend/src/index.css`

**Interfaces:**
- Consumes: SSE `{"type":"sources","sources":[...]}` (Task 8)
- Produces: `CitationSource`, `FeedbackValue`; assistant message `sources?: CitationSource[]`, `feedback?: FeedbackValue`
- Produces: localStorage record `{sessionId, messageId, value, content, createdAt}`

- [ ] **Step 1: Add types and SSE source parsing**

In `types.ts` add `CitationSource` and `FeedbackValue`; extend assistant `Message`.
In `api/chat.ts` add `onSources` to `StreamHandlers`; in the JSON-frame branch, if `type === "sources"`, call `handlers.onSources(sources)`.

- [ ] **Step 2: Wire pending sources and feedback into `useChat`**

- `pendingSourcesRef` holds the latest sources; on `stream_start`, attach to the new assistant message.
- Add reducer action `feedback` that sets `feedback` on the message and writes localStorage; ignore if already set.
- Expose `onFeedback(messageId, value)`.

- [ ] **Step 3: Render citation chips, source card, and feedback in `MessageItem`**

- Extend `renderInline` to recognize `[n]` after Markdown links; render valid `n` as a `<button>` chip, invalid as plain text.
- Add state `openSourceN`; clicking a chip toggles a card below the answer with `section_path` and `answer`.
- Add bottom-left 👍/👎 buttons for assistant messages after streaming; selected button highlighted, other disabled, label `已反馈` after click.
- Pass `onFeedback` from `MessageList`/`Widget` as needed.

- [ ] **Step 4: Build and manually verify**

Run:
```bash
cd frontend && npm run build
```
Expected: `tsc` + `vite build` exit 0, no TypeScript errors.
Manual: send a knowledge question; click `[1]`; see the source card with section path and chunk text; click 👍; button highlights, shows `已反馈`, both buttons lock; click 👎 afterward does nothing. Check localStorage has the record. Also verify an unknown `[9]` renders as plain text.

- [ ] **Step 5: Append dev-notes and commit**

Append `## 阶段:Task 13 完成(前端引用 + 反馈)` with the user's UI requirements, outputs, rejections (no backend feedback API), rework, manual verification evidence.

```bash
git add frontend/src/types.ts frontend/src/api/chat.ts frontend/src/hooks/useChat.ts frontend/src/components/MessageItem.tsx frontend/src/components/MessageList.tsx frontend/src/index.css dev-notes/ch04.md
git commit -m "feat: add clickable citations and answer feedback"
```

---

### Task 14: 集成验收、全量报告、Code Review 与 Finish

**Files:**
- Modify: `dev-notes/ch04.md`
- Create: `docs/superpowers/reports/2026-11-16-ch04-rag-eval.md`

**Interfaces:**
- Consumes: all previous tasks
- Produces: full four-strategy report; acceptance evidence; review conclusion; finish decision

- [ ] **Step 1: Run the full test suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: all tests pass; record the exact count. If failures, fix before proceeding.

- [ ] **Step 2: Apply schema, reindex, and run the full evaluation**

```bash
.venv/Scripts/python.exe -m app.db.init_ch04
.venv/Scripts/python.exe -m app.jobs.reindex_hybrid
.venv/Scripts/python.exe -m app.jobs.eval_rag --cases tests/eval/rag_cases.json --strategies dense,bm25,hybrid,hybrid_rerank --out docs/superpowers/reports/2026-11-16-ch04-rag-eval.md
```
Expected: report exists with four strategies, per-bucket tables, D-bucket refusal accuracy, faithfulness, citation validity, faith_cases counts. Record the numbers in dev-notes.

- [ ] **Step 3: Live acceptance**

Run uvicorn (`app.main:app`, port 8000) and verify:
0. Prove the BM25 branch directly:
```bash
.venv/Scripts/python.exe - <<'PY'
from app.rag.retrieval import RetrievalStrategy, retrieve_hybrid
r = retrieve_hybrid("LX-A100 的色温是多少", strategy=RetrievalStrategy.BM25)
print([(c.chunk_id, c.section_path) for c in r.chunks])
assert any("LX-A100" in (c.section_path or "") for c in r.chunks)
PY
```
Expected: prints a hit whose `section_path` contains `LX-A100 落地灯`.
1. Ask `LX-A100 的色温是多少` in the chat: SSE includes a `query_faq` tool frame and a sources frame; answer cites `[n]`; click shows `商品型号与参数 > LX-A100 落地灯` and the original text.
2. Ask `你们支持以旧换新吗`: response is a clear refusal; `low_confidence_questions` has a new `source='self_check'` row with the original text and correct `conversation_id`.
3. Ask `我的订单 E20241111001 到哪了，顺便问下退款多久到账`: order/logistics answer is not discarded if the knowledge part is insufficient; the knowledge part says it cannot confirm.
4. Frontend feedback: 👍 lights up, shows `已反馈`, locks; localStorage has the record.
5. `npm run build` passes.

- [ ] **Step 4: Request code review**

Use `superpowers:requesting-code-review` (or `/code-review`) on the ch04 branch. Append the review conclusion and any fix wave to `dev-notes/ch04.md`; rerun affected tests and the full suite.

- [ ] **Step 5: Finish the branch**

Use `superpowers:finishing-a-development-branch`. The user's prior chapters chose local merge to `feature/ch02-tools`; ask for the merge target/method at finish. Append the finish decision and merge result to `dev-notes/ch04.md`.

- [ ] **Step 6: Final delivery note**

Ensure `dev-notes/ch04.md` contains: demo commands, test results, report path, spec/plan paths, dev-notes path, and any deferred minors. Commit the final docs.

```bash
git add docs/superpowers/reports/2026-11-16-ch04-rag-eval.md dev-notes/ch04.md
git commit -m "docs: record ch04 evaluation and acceptance"
```

---

## Self-Review Notes

- Spec coverage: schema/ORM (Task 0), Milvus BM25/hybrid (1), migration (2), reranker (3), query understanding + category (4), four strategies (5), citations (6), self-check/pool (7), chat gate (8), prompt (9), eval set (10), faithfulness/ledger (11), eval job/report (12), frontend (13), acceptance/review/finish (14).
- The plan deliberately does not paste full implementations; test assertions and interfaces pin the decisions, bodies are left to the implementer.
- All metrics and thresholds are copied from the spec; no TBDs.
