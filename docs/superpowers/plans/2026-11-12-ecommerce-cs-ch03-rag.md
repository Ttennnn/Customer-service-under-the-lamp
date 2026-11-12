# ch03 RAG 知识库 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `query_faq` 从 SQL `LIKE` 关键词查表升级为 BGE-M3 + Milvus 的 dense 向量语义检索,并交付文档建库、历史对话挖知识两条可重跑的离线流水线。

**Architecture:** 离线侧把 Markdown 知识文档结构感知切块,先写 MySQL `knowledge_chunks(pending)`,再调硅基流动 BGE-M3 向量化,Milvus 以 `chunk_{id}` 为幂等主键 upsert,回填 `vector_id/status=done`;对话挖矿走 staging + 整体去重后复用同一条向量化流水线。在线侧 `query_faq` 问题向量化 → Milvus COSINE Top-K → MySQL 取原文,工具入参/出参契约不变。

**Tech Stack:** Python 3.13, FastAPI, SQLAlchemy 2.x, MySQL 8, LangChain, pymilvus 3.0.3, Milvus 3.0.3, SiliconFlow `BAAI/bge-m3` Embeddings, httpx, pytest.

**Spec:** `docs/superpowers/specs/2026-11-12-ecommerce-cs-ch03-rag-design.md`

## Global Constraints

- 嵌入模型固定 BGE-M3,走硅基流动云端 `POST https://api.siliconflow.cn/v1/embeddings`,不在本机下载模型;`BAAI/bge-m3` 单条上限 8192 token,向量维度 1024。
- 向量库固定 Milvus 3.0.x Docker standalone;PyMilvus 固定 `3.0.3`。
- MySQL 是原文权威源;`knowledge_chunks` / `qa_extraction_staging` 严格按用户 DDL,**不增删列**。
- 向量化文本固定为 `f"{category}\n{questions}\n{answer}"`;`section_path` / `content_type` / `is_key_clause` / `prev/next` 只存不进向量。
- `query_faq(keyword: str) -> str` 的名称、入参、返回类型、未命中话术不变;只做 dense 单路,不做关键词召回/混合/重排。
- 默认参数:`CHUNK_MAX_CHARS=500`、`CHUNK_OVERLAP_CHARS=80`、`EMBEDDING_BATCH_SIZE=16`、`FAQ_TOP_K=3`、`FAQ_SCORE_THRESHOLD=0.35`、`QA_DEDUP_THRESHOLD=0.92`。
- Milvus 主键固定 `chunk_{MySQL id}`;写入用 `upsert`;集合只存 `id/chunk_id/vector`,索引 `FLAT` + `COSINE`。
- 文档入库幂等键:`(content_type, section_path, category, questions, answer)`;`--reset` 只清文档域,`--reset-all` 才清全部且必须 `--yes`。
- 所有密钥/连接串只走 `.env`;`.env.example` 只留空值;不得把真实 key 写进代码、测试、计划或 dev-notes。
- 每个任务完成后立即追加 `dev-notes/ch03.md`(四件事:用户关键原话/我的产出/拒绝或纠偏/翻车返工),不许最后统一补。
- 分支基线:`feature/ch02-tools`(ch02 尚未合并,ch03 依赖其工具/执行器代码);执行前新建 `feature/ch03-rag`。

## Review Focus

1. **中断后重跑收敛**:向量化中途失败/被杀,`pending` 行必须能被下一次 `build_knowledge` 或 `vectorize` 捡起;重跑不能让 MySQL 行或 Milvus 实体翻倍。—— Task 8 / 10 / 15 的测试与验收步骤。
2. **Markdown 边界**:代码围栏里的 `#` 不是标题;大表格每块复制表头;超长无标点文本不丢内容;重叠不从半句话开始。—— Task 4 / 5 的测试。
3. **云端嵌入失败**:429/503/504/超时要重试,重试耗尽保持 `pending` 并报错;维度不是 1024 要立刻失败;`query_faq` 基础设施异常返回安全话术而不是把 `/chat` 打挂。—— Task 6 / 11 的测试。
4. **挖矿去重与中断**:跨批次精确重复、近似重复、与既有知识重复都要丢;中断留下的 `extracted` 半成品重跑要先清再抽,且已入库知识不重复。—— Task 12 / 13 的测试。
5. **重建数据安全**:`--reset` 不能误删 `mined_qa`;Milvus 删除失败时不能先删 MySQL 留下孤儿向量;`--reset-all` 必须显式确认。—— Task 9 / 10 的测试。

---

### Before Task 1: 计划评审留痕

- [ ] **Append the plan-review entry to `dev-notes/ch03.md`** under `## 阶段:计划评审通过`, with the user's approval words, the plan path, the self-review conclusion (spec coverage / step scan / type consistency / review focus / proportion), what was rejected or corrected, and any rework. Do this **before** starting Task 1, not at the end.

---

### Task 1: 依赖、配置与 Milvus Compose

**Files:**
- Modify: `requirements.txt`, `app/config.py`, `.env.example`, `docker-compose.yml`, local `.env`(不提交)
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: existing `Settings` / `get_settings()` from `app/config.py`.
- Produces: `Settings` 新字段 `embedding_base_url`, `embedding_api_key`, `embedding_model`, `embedding_dim`, `embedding_batch_size`, `milvus_uri`, `milvus_collection`, `faq_top_k`, `faq_score_threshold`, `knowledge_docs_dir`, `chunk_max_chars`, `chunk_overlap_chars`, `qa_mining_batch_size`, `qa_dedup_threshold`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py — append
def test_ch03_settings_defaults():
    s = Settings(_env_file=None, openai_api_key="x")
    assert s.embedding_model == "BAAI/bge-m3"
    assert s.embedding_dim == 1024
    assert s.embedding_batch_size == 16
    assert s.milvus_uri == "http://localhost:19530"
    assert s.milvus_collection == "knowledge"
    assert s.faq_top_k == 3
    assert s.faq_score_threshold == 0.35
    assert s.chunk_max_chars == 500
    assert s.chunk_overlap_chars == 80
    assert s.qa_dedup_threshold == 0.92
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_config.py::test_ch03_settings_defaults -v`
Expected: FAIL because `Settings` has no `embedding_model` etc.

- [ ] **Step 3: Implement**

Add the fields to `Settings` with the defaults above (`embedding_api_key: str = ""`, `embedding_base_url: str = "https://api.siliconflow.cn/v1"`). Add to `requirements.txt`:

```text
pymilvus==3.0.3
```

Append the `.env.example` block exactly (values blank where secret):

```dotenv
EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1
EMBEDDING_API_KEY=
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DIM=1024
EMBEDDING_BATCH_SIZE=16
MILVUS_URI=http://localhost:19530
MILVUS_COLLECTION=knowledge
FAQ_TOP_K=3
FAQ_SCORE_THRESHOLD=0.35
KNOWLEDGE_DOCS_DIR=knowledge
CHUNK_MAX_CHARS=500
CHUNK_OVERLAP_CHARS=80
QA_MINING_BATCH_SIZE=10
QA_DEDUP_THRESHOLD=0.92
```

Add `EMBEDDING_API_KEY=<provided key>` to the local, gitignored `.env` only.

Extend `docker-compose.yml` with the official Milvus v3.0.3 standalone three-service layout (`etcd`, `minio`, `standalone`), pinning the images from the official `milvus-standalone-docker-compose.yaml` for `v3.0.3`, exposing `19530`/`9091`, adding named volumes, `depends_on`, and healthchecks. Do not change the existing MySQL service.

- [ ] **Step 4: Run tests and start infrastructure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_config.py -v`
Expected: PASS.

Run: `docker compose up -d`
Expected: `docker compose ps` shows MySQL, etcd, minio, standalone; `curl -f http://localhost:9091/healthz` returns success; MySQL still reachable.

- [ ] **Step 5: Append dev-notes + commit**

Append `## 阶段:Task 1 完成` to `dev-notes/ch03.md` with the four required items (user words: none for this task; outputs: config fields/compose; rejections: none; rework: any image/version issue actually hit).

```bash
git add requirements.txt app/config.py .env.example docker-compose.yml tests/test_config.py dev-notes/ch03.md
git commit -m "feat: add ch03 config and Milvus standalone compose"
```

---

### Task 2: Knowledge ORM models + DDL + init script

**Files:**
- Create: `app/models/knowledge.py`, `sql/ch03_knowledge.sql`, `app/db/init_ch03.py`, `tests/test_knowledge_models.py`
- Modify: `app/models/__init__.py`

**Interfaces:**
- Produces: `KnowledgeChunk` and `QaExtractionStaging` SQLAlchemy models; `apply_ch03(engine) -> None`, `main() -> None` in `app/db/init_ch03.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_knowledge_models.py
from app.models import KnowledgeChunk, QaExtractionStaging

def _cols(model):
    return {c.name: c for c in model.__table__.columns}

def test_knowledge_chunk_columns_and_enums():
    c = _cols(KnowledgeChunk)
    assert list(c) == [
        "id", "category", "questions", "answer", "section_path", "content_type",
        "is_key_clause", "prev_chunk_id", "next_chunk_id", "vector_id",
        "vectorize_status", "created_at", "updated_at",
    ]
    assert c["section_path"].type.length == 512
    assert c["vector_id"].type.length == 64
    assert c["vectorize_status"].type.enums == ["pending", "done"]
    assert c["is_key_clause"].nullable is False
    assert c["section_path"].nullable is True
    assert any(str(fk.target_fullname) == "knowledge_chunks.id"
               for fk in KnowledgeChunk.__table__.foreign_keys)

def test_staging_columns_and_enums():
    c = _cols(QaExtractionStaging)
    assert list(c) == ["id", "batch_no", "source_ref", "question", "answer", "status", "created_at"]
    assert c["status"].type.enums == ["extracted", "kept", "discarded"]
    assert c["batch_no"].index is True
    assert c["status"].index is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_knowledge_models.py -v`
Expected: FAIL/import error because models do not exist.

- [ ] **Step 3: Implement**

Commit the user DDL verbatim as `sql/ch03_knowledge.sql` (copy from spec, including `SET NAMES utf8mb4;`). Add the ORM models in `app/models/knowledge.py`:

- `KnowledgeChunk`: `id` BIGINT/Integer variant autoincrement; `category` String(255) index; `questions` Text; `answer` Text; `section_path` String(512) nullable; `content_type` String(32) nullable; `is_key_clause` Boolean default False server_default `text("0")`; `prev_chunk_id`/`next_chunk_id` BigInteger FK `knowledge_chunks.id` with `ondelete="SET NULL"`; `vector_id` String(64) nullable; `vectorize_status` Enum("pending","done") default pending server_default pending, index; timestamps with `server_default=func.now()` and `onupdate=func.now()`.
- `QaExtractionStaging`: exact columns from DDL, `batch_no`/`status` indexed, status Enum default `extracted`, timestamps.
- Export both from `app/models/__init__.py`.

`app/db/init_ch03.py` reads `sql/ch03_knowledge.sql`, strips `--` comment lines, splits on `;`, checks `information_schema.tables` for both tables; if both present it is a no-op, if neither present it executes each statement, if only one exists it raises a clear error. Expose `apply_ch03(engine)` and `main()`.

- [ ] **Step 4: Run tests and apply DDL**

Run: `.venv/Scripts/python.exe -m pytest tests/test_knowledge_models.py tests/test_models.py -v`
Expected: PASS.

Run: `.venv/Scripts/python.exe -m app.db.init_ch03`
Expected: prints created tables; rerun prints already initialized; `SHOW TABLES` contains both.

- [ ] **Step 5: Append dev-notes + commit**

```bash
git add app/models/knowledge.py app/models/__init__.py sql/ch03_knowledge.sql app/db/init_ch03.py tests/test_knowledge_models.py dev-notes/ch03.md
git commit -m "feat: add knowledge chunk models and ch03 DDL"
```

---

### Task 3: Sample knowledge documents + loader

**Files:**
- Create: `knowledge/退货政策.md`, `knowledge/商品FAQ.md`, `knowledge/售后手册.md`, `knowledge/运费与配送说明.md`, `app/rag/__init__.py`, `app/rag/docs.py`, `tests/test_docs.py`

**Interfaces:**
- Produces:

```python
@dataclass(frozen=True)
class Document:
    path: Path
    title: str
    content_type: str       # policy | faq | manual
    category: str | None
    body: str               # full markdown body including H1

def load_document(path: Path) -> Document
def load_documents(docs_dir: Path) -> list[Document]
```

- [ ] **Step 1: Write the failing test**

```python
# tests/test_docs.py
from pathlib import Path
from app.rag.docs import load_document, load_documents

def test_load_document_frontmatter(tmp_path):
    p = tmp_path / "a.md"
    p.write_text("---\ncontent_type: policy\ncategory: 售后\n---\n# 标题\n\n正文。\n", encoding="utf-8")
    d = load_document(p)
    assert d.title == "标题"
    assert d.content_type == "policy" and d.category == "售后"
    assert "正文。" in d.body and "content_type" not in d.body

def test_invalid_content_type_raises(tmp_path):
    p = tmp_path / "a.md"; p.write_text("---\ncontent_type: blog\n---\n# T\n", encoding="utf-8")
    try:
        load_document(p)
    except ValueError as e:
        assert "content_type" in str(e)
    else:
        raise AssertionError("expected ValueError")

def test_sample_docs_cover_shipping_fee():
    docs = load_documents(Path("knowledge"))
    assert len(docs) >= 4
    text = "\n".join(d.body for d in docs)
    for needle in ["包邮", "99", "8 元", "邮费", "运费", "7 天"]:
        assert needle in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_docs.py -v`
Expected: FAIL because `app.rag.docs` and `knowledge/` do not exist.

- [ ] **Step 3: Implement**

Create the four sample docs. Required facts (write natural Chinese prose around them):

- `退货政策.md` (`content_type: policy`): 签收后 7 天内可申请退货;商品完好、吊牌齐全;退款 1-3 个工作日原路退回;质量问题商家承担运费、非质量问题买家承担;含一个 `【关键条款】` 标记。
- `商品FAQ.md` (`content_type: faq`, `category: 商品`): H2 真实问法「尺码怎么选?」「发票怎么开?」「商品有质量问题怎么办?」「支持货到付款吗?」,每节给答案。
- `售后手册.md` (`content_type: manual`): 换货流程、维修、投诉、上门取件;含时限和「不得/必须」类关键条款。
- `运费与配送说明.md` (`content_type: policy`): 单笔订单实付满 99 元包邮(偏远地区除外);未满 99 元收取 8 元运费;偏远地区加收 10 元;退换货质量问题商家承担、非质量问题买家承担;现货 48 小时内发货。明确同时出现「邮费」「运费」两种说法。

Implement `load_document`: parse `---` frontmatter as simple `key: value` lines; require `content_type` in `{"policy","faq","manual"}`; parse `category` optional; H1 title required; body is everything after frontmatter (including H1). `load_documents` globs `*.md` sorted by filename and returns documents.

- [ ] **Step 4: Run tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_docs.py -v`
Expected: PASS; print shows 4 docs.

- [ ] **Step 5: Append dev-notes + commit**

```bash
git add knowledge app/rag/__init__.py app/rag/docs.py tests/test_docs.py dev-notes/ch03.md
git commit -m "feat: add sample knowledge docs and markdown loader"
```

---

### Task 4: Section extraction and chunk field derivation

**Files:**
- Create: `app/rag/splitter.py`, `tests/test_splitter_sections.py`

**Interfaces:**
- Produces:

```python
@dataclass(frozen=True)
class Chunk:
    category: str
    questions: str
    answer: str
    section_path: str | None
    content_type: str
    is_key_clause: bool

def split_document(doc: Document, max_chars: int = 500, overlap_chars: int = 80) -> list[Chunk]
```

- [ ] **Step 1: Write the failing test**

```python
# tests/test_splitter_sections.py
from pathlib import Path
from app.rag.docs import load_document
from app.rag.splitter import split_document

def _doc(tmp_path, text, content_type="policy", category=None):
    fm = f"---\ncontent_type: {content_type}\n" + (f"category: {category}\n" if category else "") + "---\n"
    p = tmp_path / "d.md"; p.write_text(fm + text, encoding="utf-8")
    return load_document(p)

def test_policy_heading_paths(tmp_path):
    d = _doc(tmp_path, "# 退货政策\n\n## 退货时限\n\n签收后 7 天内。\n\n### 七天无理由\n\n商品完好。\n")
    chunks = split_document(d)
    assert [(c.questions, c.category, c.section_path) for c in chunks] == [
        ("退货时限", "退货政策", "退货政策 > 退货时限"),
        ("七天无理由", "退货政策 > 退货时限", "退货政策 > 退货时限 > 七天无理由"),
    ]
    assert all(c.content_type == "policy" for c in chunks)

def test_faq_uses_frontmatter_category(tmp_path):
    d = _doc(tmp_path, "# 商品FAQ\n\n## 尺码怎么选?\n\n按脚长选。\n", "faq", "商品")
    (c,) = split_document(d)
    assert c.questions == "尺码怎么选?"
    assert c.category == "商品"
    assert c.section_path == "商品FAQ > 尺码怎么选?"

def test_code_fence_heading_ignored(tmp_path):
    d = _doc(tmp_path, "# 手册\n\n## 示例\n\n```\n# 不是标题\n```\n\n正文。\n")
    chunks = split_document(d)
    assert len(chunks) == 1 and "# 不是标题" in chunks[0].answer

def test_empty_section_skipped_and_key_clause(tmp_path):
    d = _doc(tmp_path, "# 政策\n\n## 空节\n\n## 关键\n\n【关键条款】必须在 7 天内申请。\n")
    (c,) = split_document(d)
    assert c.questions == "关键" and c.is_key_clause is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_splitter_sections.py -v`
Expected: FAIL because `app.rag.splitter` does not exist.

- [ ] **Step 3: Implement**

Parse headings with a stack while tracking triple-backtick code fences. Each heading opens a section with `path` tuple; body is lines until the next heading. Emit a chunk for every section with non-empty body (parent intro text included). Derive fields exactly as the tests show. `is_key_clause` is true if body contains `【关键条款】`, or (for `policy`/`manual`) body matches `不予|不得|必须|免责|期限内|时效`. For this task, `answer = body.strip()` regardless of `max_chars`; Task 5 adds splitting.

- [ ] **Step 4: Run tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_splitter_sections.py -v`
Expected: PASS.

- [ ] **Step 5: Append dev-notes + commit**

```bash
git add app/rag/splitter.py tests/test_splitter_sections.py dev-notes/ch03.md
git commit -m "feat: add structure-aware markdown section chunking"
```

---

### Task 5: Long-text recursive split, overlap, and table row splitting

**Files:**
- Modify: `app/rag/splitter.py`
- Test: `tests/test_splitter_long.py`

**Interfaces:**
- Consumes `Chunk`, `split_document` from Task 4.
- Produces: same public interface; long bodies may yield multiple `Chunk`s in order.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_splitter_long.py
from app.rag.splitter import split_document

def _doc(tmp_path, text, content_type="policy"):
    p = tmp_path / "d.md"
    p.write_text(f"---\ncontent_type: {content_type}\n---\n{text}", encoding="utf-8")
    from app.rag.docs import load_document
    return load_document(p)

def test_long_text_splits_on_sentence_boundaries(tmp_path):
    body = "".join(f"第{i}句话说完整。" for i in range(40))
    d = _doc(tmp_path, f"# 政策\n\n## 长节\n\n{body}\n")
    chunks = split_document(d, max_chars=120, overlap_chars=30)
    assert len(chunks) > 1
    for c in chunks:
        assert c.answer.endswith(("。", "!", "?", ";", "!"))
        assert len(c.answer) <= 120 + 30
    assert chunks[0].answer != chunks[1].answer

def test_table_repeats_header_in_every_chunk(tmp_path):
    rows = "\n".join(f"| 商品{i} | {i} 元 |" for i in range(40))
    table = "| 商品 | 价格 |\n|---|---|\n" + rows
    d = _doc(tmp_path, f"# 价目\n\n## 价格表\n\n{table}\n")
    chunks = split_document(d, max_chars=160, overlap_chars=0)
    assert len(chunks) > 1
    for c in chunks:
        assert c.answer.startswith("| 商品 | 价格 |")
        assert "|---|" in c.answer
    joined = "\n".join(c.answer for c in chunks)
    assert "商品39" in joined

def test_no_punctuation_still_splits(tmp_path):
    d = _doc(tmp_path, "# 政策\n\n## 长节\n\n" + "无标点" * 200)
    chunks = split_document(d, max_chars=100, overlap_chars=0)
    assert len(chunks) > 1 and all(c.answer for c in chunks)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_splitter_long.py -v`
Expected: FAIL because long bodies are returned as one chunk.

- [ ] **Step 3: Implement**

Add recursive splitting with this priority: blank-line paragraphs → list items / table blocks → sentence boundaries (`。!?;!?` and `…`); hard split only when a single sentence exceeds `max_chars`. Detect a Markdown table as a header line + separator line (`|---|` style) + contiguous `|` rows; when a table block exceeds `max_chars`, group rows so each group includes the header + separator, preserving every row. For non-table chunks, prepend up to `overlap_chars` from the previous chunk's tail, but first find a sentence boundary inside that tail and start after it; if no boundary exists, use no overlap. Ensure each chunk's `answer` ends at a sentence boundary when one exists; hard-split chunks may end mid-text by design. Recompute `is_key_clause` per emitted chunk.

- [ ] **Step 4: Run tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_splitter_sections.py tests/test_splitter_long.py -v`
Expected: PASS.

- [ ] **Step 5: Append dev-notes + commit**

```bash
git add app/rag/splitter.py tests/test_splitter_long.py dev-notes/ch03.md
git commit -m "feat: split long sections with overlap and table headers"
```

---

### Task 6: SiliconFlow BGE-M3 embedding client

**Files:**
- Create: `app/rag/embedding.py`, `tests/test_embedding.py`

**Interfaces:**
- Produces:

```python
class EmbeddingError(RuntimeError): ...

class SiliconFlowEmbedder:
    def __init__(self, base_url: str, api_key: str, model: str = "BAAI/bge-m3",
                 dim: int = 1024, batch_size: int = 16,
                 client: httpx.Client | None = None, max_retries: int = 3,
                 sleep=time.sleep) -> None: ...
    def embed(self, texts: list[str]) -> list[list[float]]
    def embed_one(self, text: str) -> list[float]
    def close(self) -> None

def get_embedder() -> SiliconFlowEmbedder
```

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_embedding.py
import httpx
from app.rag.embedding import SiliconFlowEmbedder, EmbeddingError

def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))

def test_batches_and_orders_by_index():
    calls = []
    def handler(req):
        payload = __import__("json").loads(req.content)
        calls.append(payload)
        data = [{"index": i, "embedding": [float(i)] * 1024} for i in range(len(payload["input"]))]
        return httpx.Response(200, json={"data": list(reversed(data))})
    e = SiliconFlowEmbedder("https://x/v1", "k", batch_size=2, client=_client(handler))
    out = e.embed(["a", "b", "c"])
    assert len(calls) == 2 and calls[0]["input"] == ["a", "b"]
    assert [v[0] for v in out] == [0.0, 1.0, 2.0]

def test_dimension_mismatch_raises():
    def handler(req):
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.0] * 3}]})
    e = SiliconFlowEmbedder("https://x/v1", "k", client=_client(handler))
    try:
        e.embed_one("a")
    except EmbeddingError as exc:
        assert "1024" in str(exc)
    else:
        raise AssertionError("expected EmbeddingError")

def test_retries_503_then_succeeds():
    n = {"i": 0}
    def handler(req):
        n["i"] += 1
        if n["i"] == 1:
            return httpx.Response(503, json={"message": "busy"})
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.0] * 1024}]})
    e = SiliconFlowEmbedder("https://x/v1", "k", client=_client(handler), sleep=lambda _: None)
    assert len(e.embed_one("a")) == 1024 and n["i"] == 2

def test_401_raises_without_retry():
    n = {"i": 0}
    def handler(req):
        n["i"] += 1
        return httpx.Response(401, json={"message": "bad key"})
    e = SiliconFlowEmbedder("https://x/v1", "k", client=_client(handler), sleep=lambda _: None)
    try:
        e.embed_one("a")
    except EmbeddingError as exc:
        assert "401" in str(exc) and n["i"] == 1
    else:
        raise AssertionError("expected EmbeddingError")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_embedding.py -v`
Expected: FAIL because module does not exist.

- [ ] **Step 3: Implement**

Use `httpx.Client.post(f"{base_url}/embeddings", headers={"Authorization": f"Bearer {api_key}"}, json={"model": model, "input": batch, "encoding_format": "float"})`. Split `texts` into `batch_size` chunks; for each response sort `data` by `index` and validate every vector length equals `dim`; raise `EmbeddingError` with status/body on non-200; retry only 429, 5xx, and `httpx.TransportError` up to `max_retries` with sleep `0.5 * 2**attempt`; empty input returns `[]`. `embed_one` returns `embed(texts=[text])[0]`. `get_embedder()` reads `get_settings()`, raises a clear `EmbeddingError` if `embedding_api_key` is empty, and caches a module-level singleton under a lock.

- [ ] **Step 4: Run tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_embedding.py -v`
Expected: PASS.

- [ ] **Step 5: Append dev-notes + commit**

```bash
git add app/rag/embedding.py tests/test_embedding.py dev-notes/ch03.md
git commit -m "feat: add SiliconFlow BGE-M3 embedding client"
```

---

### Task 7: Milvus knowledge collection wrapper

**Files:**
- Create: `app/rag/milvus_store.py`, `tests/test_milvus_store.py`

**Interfaces:**
- Produces:

```python
@dataclass(frozen=True)
class VectorItem:
    vector_id: str
    chunk_id: int
    vector: list[float]

@dataclass(frozen=True)
class SearchHit:
    chunk_id: int
    score: float

class KnowledgeVectorStore:
    def __init__(self, client, collection: str, dim: int) -> None: ...
    def ensure_collection(self) -> None
    def upsert(self, items: list[VectorItem]) -> None
    def search(self, vector: list[float], top_k: int) -> list[SearchHit]
    def delete_ids(self, ids: list[str]) -> None
    def count(self) -> int

def get_vector_store() -> KnowledgeVectorStore
```

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_milvus_store.py
from app.rag.milvus_store import KnowledgeVectorStore, VectorItem

class FakeClient:
    def __init__(self):
        self.created = None; self.upserted = None; self.deleted = None
        self.search_result = [[{"id": "chunk_1", "distance": 0.8, "entity": {"chunk_id": 1}}]]
    def has_collection(self, name): return False
    def create_schema(self, **kw): 
        class S:
            def add_field(self, *a, **k): pass
        return S()
    def prepare_index_params(self):
        class I:
            def add_index(self, **k): pass
        return I()
    def create_collection(self, name, schema=None, index_params=None, **kw): self.created = (name, kw)
    def upsert(self, name, data): self.upserted = (name, data)
    def delete(self, name, ids=None): self.deleted = (name, ids)
    def search(self, **kw): return self.search_result
    def get_collection_stats(self, name): return {"row_count": 7}

def test_ensure_upsert_search_delete():
    c = FakeClient(); s = KnowledgeVectorStore(c, "knowledge", 1024)
    s.ensure_collection(); assert c.created[0] == "knowledge"
    s.upsert([VectorItem("chunk_1", 1, [0.0] * 1024)])
    assert c.upserted[1][0] == {"id": "chunk_1", "chunk_id": 1, "vector": [0.0] * 1024}
    assert s.search([0.0] * 1024, 3)[0].chunk_id == 1
    s.delete_ids(["chunk_1"]); assert c.deleted[1] == ["chunk_1"]
    assert s.count() == 7

def test_search_accepts_top_level_chunk_id():
    c = FakeClient(); c.search_result = [[{"id": "chunk_2", "distance": 0.5, "chunk_id": 2}]]
    assert KnowledgeVectorStore(c, "knowledge", 1024).search([0.0] * 1024, 3)[0].chunk_id == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_milvus_store.py -v`
Expected: FAIL because module does not exist.

- [ ] **Step 3: Implement**

`ensure_collection` calls `has_collection`; if absent, build a schema with `VARCHAR(64)` primary key `id`, `INT64 chunk_id`, `FLOAT_VECTOR(1024) vector`; index params `FLAT` + `COSINE`; create with `consistency_level="Strong"`; load if the client exposes `load_collection`. `upsert` maps `VectorItem`s to dicts. `search` calls `client.search(collection_name=..., data=[vector], anns_field="vector", search_params={"metric_type": "COSINE"}, limit=top_k, output_fields=["chunk_id"])`, then reads `chunk_id` from `hit["entity"]["chunk_id"]` or `hit["chunk_id"]`; skip hits without it; return `SearchHit(chunk_id, float(hit["distance"]))`. `delete_ids` no-ops on empty. `count` reads `get_collection_stats(...).get("row_count", 0)`. Each public method (`upsert`, `search`, `delete_ids`, `count`) calls `ensure_collection()` first (idempotent, so the first caller creates/loads the collection). `get_vector_store()` reads settings, creates `MilvusClient(uri=settings.milvus_uri)`, caches under a lock.

- [ ] **Step 4: Run tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_milvus_store.py -v`
Expected: PASS.

- [ ] **Step 5: Append dev-notes + commit**

```bash
git add app/rag/milvus_store.py tests/test_milvus_store.py dev-notes/ch03.md
git commit -m "feat: add Milvus knowledge collection wrapper"
```

---

### Task 8: Vectorize pending chunks idempotently

**Files:**
- Create: `app/rag/vectorize.py`, `tests/test_vectorize.py`

**Interfaces:**
- Consumes: `KnowledgeChunk` (Task 2), `SiliconFlowEmbedder` (Task 6), `KnowledgeVectorStore` / `VectorItem` (Task 7).
- Produces:

```python
@dataclass(frozen=True)
class VectorizeReport:
    processed: int
    failed: int

def vectorization_text(chunk: KnowledgeChunk) -> str
def vector_id_for(chunk_id: int) -> str          # f"chunk_{chunk_id}"
def vectorize_pending(session, embedder, store, batch_size: int = 16,
                      max_chunks: int | None = None) -> VectorizeReport
```

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_vectorize.py
from app.models import KnowledgeChunk
from app.rag.vectorize import vectorize_pending, vectorization_text, vector_id_for
from app.rag.milvus_store import VectorItem

class FakeEmbedder:
    def __init__(self): self.calls = []
    def embed(self, texts):
        self.calls.append(texts)
        return [[0.1] * 1024 for _ in texts]

class FakeStore:
    def __init__(self): self.items = []
    def upsert(self, items): self.items.extend(items)

def _chunk(session, q="问", a="答", cat="售后", status="pending"):
    c = KnowledgeChunk(category=cat, questions=q, answer=a, content_type="faq",
                       section_path="文档 > 节", vectorize_status=status)
    session.add(c); session.commit(); session.refresh(c); return c

def test_vectorize_pending_sets_done_and_vector_id(session):
    c = _chunk(session); e = FakeEmbedder(); s = FakeStore()
    r = vectorize_pending(session, e, s)
    session.expire_all(); row = session.get(KnowledgeChunk, c.id)
    assert r.processed == 1 and row.vectorize_status == "done"
    assert row.vector_id == vector_id_for(c.id)
    assert e.calls == [["售后\n问\n答"]]
    assert s.items[0].chunk_id == c.id and len(s.items[0].vector) == 1024

def test_second_run_does_nothing(session):
    _chunk(session); e = FakeEmbedder(); s = FakeStore()
    vectorize_pending(session, e, s)
    vectorize_pending(session, e, s)
    assert len(e.calls) == 1 and len(s.items) == 1

def test_embed_failure_leaves_pending_and_retry_works(session):
    c = _chunk(session)
    class Boom:
        def embed(self, texts): raise RuntimeError("api down")
    try:
        vectorize_pending(session, Boom(), FakeStore())
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected failure")
    session.expire_all(); assert session.get(KnowledgeChunk, c.id).vectorize_status == "pending"
    vectorize_pending(session, FakeEmbedder(), FakeStore())

def test_max_chunks_leaves_rest_pending(session):
    c1 = _chunk(session, q="q1"); c2 = _chunk(session, q="q2")
    r = vectorize_pending(session, FakeEmbedder(), FakeStore(), max_chunks=1)
    session.expire_all()
    assert r.processed == 1
    assert session.get(KnowledgeChunk, c1.id).vectorize_status == "done"
    assert session.get(KnowledgeChunk, c2.id).vectorize_status == "pending"

def test_text_and_id_helpers():
    c = KnowledgeChunk(category="C", questions="Q", answer="A")
    assert vectorization_text(c) == "C\nQ\nA"
    assert vector_id_for(42) == "chunk_42"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_vectorize.py -v`
Expected: FAIL because module does not exist.

- [ ] **Step 3: Implement**

Select `pending` rows ordered by `id`, limited by `max_chunks`. For each embedding batch (up to `batch_size`): build texts, call `embedder.embed`, validate each vector length is 1024, upsert `VectorItem(vector_id_for(row.id), row.id, vector)`, then set `row.vector_id` and `row.vectorize_status='done'`, commit once per batch. On any exception, rollback the current batch, leave rows `pending`, and re-raise. Count processed. Do not mark done before Milvus upsert returns.

- [ ] **Step 4: Run tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_vectorize.py -v`
Expected: PASS.

- [ ] **Step 5: Append dev-notes + commit**

```bash
git add app/rag/vectorize.py tests/test_vectorize.py dev-notes/ch03.md
git commit -m "feat: vectorize pending knowledge chunks idempotently"
```

---

### Task 9: Document ingestion, neighbor links, and reset semantics

**Files:**
- Create: `app/rag/ingest.py`, `tests/test_ingest.py`

**Interfaces:**
- Consumes: `Chunk` (Task 4/5), `KnowledgeChunk` (Task 2), `KnowledgeVectorStore` (Task 7).
- Produces:

```python
@dataclass(frozen=True)
class IngestReport:
    inserted: int
    reused: int

@dataclass(frozen=True)
class MinedQA:
    question: str
    answer: str
    category: str

def chunk_fingerprint(chunk: Chunk) -> tuple[str, str, str, str, str]
def ingest_document_chunks(session, chunks: list[Chunk]) -> tuple[list[KnowledgeChunk], IngestReport]
def insert_mined_qa(session, items: list[MinedQA]) -> list[KnowledgeChunk]
def reset_document_domain(session, store) -> int
def reset_all(session, store) -> int
```

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_ingest.py
from app.models import KnowledgeChunk
from app.rag.splitter import Chunk
from app.rag.ingest import (ingest_document_chunks, insert_mined_qa, MinedQA,
                            reset_document_domain, reset_all)

def _chunk(q="问", a="答", cat="售后", path="文档 > 节"):
    return Chunk(category=cat, questions=q, answer=a, section_path=path,
                 content_type="faq", is_key_clause=False)

class FakeStore:
    def __init__(self): self.deleted = []
    def delete_ids(self, ids): self.deleted.append(ids)

def test_ingest_twice_is_idempotent_and_links(session):
    chunks = [_chunk(q="q1"), _chunk(q="q2"), _chunk(q="q3")]
    rows1, r1 = ingest_document_chunks(session, chunks)
    rows2, r2 = ingest_document_chunks(session, chunks)
    assert (r1.inserted, r1.reused) == (3, 0) and (r2.inserted, r2.reused) == (0, 3)
    assert [r.id for r in rows1] == [r.id for r in rows2]
    session.expire_all()
    assert rows1[0].prev_chunk_id is None and rows1[0].next_chunk_id == rows1[1].id
    assert rows1[1].prev_chunk_id == rows1[0].id and rows1[1].next_chunk_id == rows1[2].id
    assert rows1[2].next_chunk_id is None

def test_reset_document_keeps_mined(session):
    doc_rows, _ = ingest_document_chunks(session, [_chunk(q="doc")])
    doc_rows[0].vector_id = "chunk_doc"; session.commit()
    mined = insert_mined_qa(session, [MinedQA("挖矿问", "挖矿答", "售后")])
    mined[0].vector_id = "chunk_mined"; session.commit()
    store = FakeStore(); n = reset_document_domain(session, store)
    assert n == 1 and store.deleted == [["chunk_doc"]]
    session.expire_all()
    ids = [r.id for r in session.query(KnowledgeChunk).all()]
    assert ids == [mined[0].id]

def test_reset_all_clears_everything(session):
    ingest_document_chunks(session, [_chunk(q="doc")])
    insert_mined_qa(session, [MinedQA("挖矿问", "挖矿答", "售后")])
    assert reset_all(session, FakeStore()) == 2
    assert session.query(KnowledgeChunk).count() == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ingest.py -v`
Expected: FAIL because module does not exist.

- [ ] **Step 3: Implement**

Fingerprint is the identity tuple `(content_type or "", section_path or "", category, questions, answer)`. Before insert, protect the DDL lengths: truncate `section_path` to 512 chars (keep the final heading segment visible when truncating) and `category` to 255. For one document's chunks, load existing document-domain rows (`content_type != 'mined_qa' or content_type is null`) into a dict keyed by fingerprint; reuse matching rows, insert the rest with `vectorize_status='pending'`; flush, then set `prev_chunk_id`/`next_chunk_id` by list order and commit. Return rows in input order plus the report. `insert_mined_qa` uses the same fingerprint rule, writes `content_type='mined_qa'`, `section_path=None`, `is_key_clause=False`, `pending`, no neighbor links. `reset_document_domain` first collects non-null `vector_id`s, calls `store.delete_ids` (in batches of 500) before deleting any MySQL rows; then deletes document-domain rows. `reset_all` does the same for every row. If `store.delete_ids` raises, do not touch MySQL.

- [ ] **Step 4: Run tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ingest.py -v`
Expected: PASS.

- [ ] **Step 5: Append dev-notes + commit**

```bash
git add app/rag/ingest.py tests/test_ingest.py dev-notes/ch03.md
git commit -m "feat: ingest chunks with idempotent fingerprints and safe reset"
```

---

### Task 10: `build_knowledge` and `vectorize` CLI jobs

**Files:**
- Create: `app/jobs/__init__.py`, `app/jobs/build_knowledge.py`, `app/jobs/vectorize.py`, `tests/test_build_knowledge.py`

**Interfaces:**
- Consumes: `load_documents` (Task 3), `split_document` (Task 4/5), `ingest_document_chunks` / `reset_document_domain` / `reset_all` (Task 9), `vectorize_pending` (Task 8), `get_embedder` (Task 6), `get_vector_store` (Task 7).
- Produces:

```python
@dataclass(frozen=True)
class BuildReport:
    documents: int
    chunks: int
    inserted: int
    reused: int
    vectorized: int

def build_knowledge(session, docs_dir, embedder, store, *,
                    max_chars: int = 500, overlap_chars: int = 80,
                    reset: bool = False, reset_all: bool = False,
                    do_vectorize: bool = True,
                    max_chunks: int | None = None) -> BuildReport

def main(argv: list[str] | None = None) -> int
```

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_build_knowledge.py
from pathlib import Path
from app.models import KnowledgeChunk
from app.jobs.build_knowledge import build_knowledge

class FakeEmbedder:
    def embed(self, texts): return [[0.0] * 1024 for _ in texts]
class FakeStore:
    def __init__(self): self.items = []
    def upsert(self, items): self.items.extend(items)
    def delete_ids(self, ids): pass

def _docs(tmp_path):
    d = tmp_path / "knowledge"; d.mkdir()
    (d / "a.md").write_text("---\ncontent_type: policy\n---\n# 退货政策\n\n## 退货时限\n\n签收后 7 天内。\n", encoding="utf-8")
    (d / "b.md").write_text("---\ncontent_type: policy\n---\n# 运费\n\n## 邮费\n\n满 99 元包邮。\n", encoding="utf-8")
    return d

def test_build_twice_is_idempotent(session, tmp_path):
    d = _docs(tmp_path); e = FakeEmbedder(); s = FakeStore()
    r1 = build_knowledge(session, d, e, s)
    r2 = build_knowledge(session, d, e, s)
    assert r1.inserted == 2 and r1.vectorized == 2
    assert r2.inserted == 0 and r2.reused == 2 and r2.vectorized == 0
    assert session.query(KnowledgeChunk).count() == 2
    assert len(s.items) == 2

def test_no_vectorize_leaves_pending(session, tmp_path):
    d = _docs(tmp_path); r = build_knowledge(session, d, FakeEmbedder(), FakeStore(), do_vectorize=False)
    assert r.vectorized == 0
    assert session.query(KnowledgeChunk).filter_by(vectorize_status="pending").count() == 2

def test_reset_preserves_mined(session, tmp_path):
    d = _docs(tmp_path); build_knowledge(session, d, FakeEmbedder(), FakeStore())
    from app.rag.ingest import insert_mined_qa, MinedQA
    insert_mined_qa(session, [MinedQA("挖矿问", "挖矿答", "售后")]); session.commit()
    r = build_knowledge(session, d, FakeEmbedder(), FakeStore(), reset=True)
    assert r.inserted == 2
    assert session.query(KnowledgeChunk).filter_by(content_type="mined_qa").count() == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_build_knowledge.py -v`
Expected: FAIL because jobs do not exist.

- [ ] **Step 3: Implement**

`build_knowledge` optionally calls `reset_all` or `reset_document_domain`, loops over `load_documents(docs_dir)`, calls `split_document` per document, then `ingest_document_chunks` per document (so neighbor links stay document-local), accumulates the report, then calls `vectorize_pending(..., max_chunks=max_chunks)` if `do_vectorize`. CLI `main` parses `--docs-dir`, `--reset`, `--reset-all`, `--yes`, `--no-vectorize`, `--max-chunks`, `--max-chars`, `--overlap-chars`; `--reset-all` without `--yes` prints the blast radius and returns 2; prints a one-line summary. `app/jobs/vectorize.py` exposes `main` with `--batch-size`/`--max-chunks` and calls `vectorize_pending`.

- [ ] **Step 4: Run tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_build_knowledge.py tests/test_vectorize.py -v`
Expected: PASS.

- [ ] **Step 5: Append dev-notes + commit**

```bash
git add app/jobs/__init__.py app/jobs/build_knowledge.py app/jobs/vectorize.py tests/test_build_knowledge.py dev-notes/ch03.md
git commit -m "feat: add build_knowledge and vectorize CLI jobs"
```

---

### Task 11: Retrieval service, vector `query_faq`, and per-tool timeout

**Files:**
- Create: `app/rag/retrieval.py`, `tests/test_retrieval.py`
- Modify: `app/tools/faq.py`, `app/tools/registry.py`, `app/services/chat.py`, `tests/test_tools_faq.py`, `tests/test_tools_business.py`, `tests/test_chat.py`

**Interfaces:**
- Consumes: `SiliconFlowEmbedder` (Task 6), `KnowledgeVectorStore` (Task 7), `KnowledgeChunk` (Task 2).
- Produces:

```python
@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: int
    score: float
    questions: str
    answer: str

def retrieve(keyword: str, *, session=None, embedder=None, store=None,
             top_k: int | None = None, threshold: float | None = None) -> list[RetrievedChunk]

def tool_timeout(name: str) -> float   # registry.py
```

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_retrieval.py
from app.models import KnowledgeChunk
from app.rag.retrieval import retrieve
from app.rag.milvus_store import SearchHit

class FakeEmbedder:
    def embed_one(self, text): return [0.1] * 1024
class FakeStore:
    def __init__(self, hits): self.hits = hits
    def search(self, vector, top_k): return self.hits

def _row(session, q, a):
    c = KnowledgeChunk(category="售后", questions=q, answer=a, content_type="faq"); session.add(c); session.commit(); session.refresh(c); return c

def test_retrieve_orders_by_score_and_filters(session):
    a = _row(session, "问A", "答A"); b = _row(session, "问B", "答B")
    hits = [SearchHit(b.id, 0.9), SearchHit(a.id, 0.2)]
    out = retrieve("邮费", session=session, embedder=FakeEmbedder(), store=FakeStore(hits), threshold=0.5)
    assert [(r.chunk_id, r.answer) for r in out] == [(b.id, "答B")]

def test_retrieve_empty(session):
    assert retrieve("x", session=session, embedder=FakeEmbedder(), store=FakeStore([])) == []
```

```python
# tests/test_tools_faq.py — replace the old LIKE tests
from app.models import KnowledgeChunk
from app.rag.milvus_store import SearchHit
from app.tools.faq import query_faq

def test_query_faq_returns_vector_hit_text(session, monkeypatch):
    c = KnowledgeChunk(category="物流", questions="邮费怎么算?", answer="满 99 元包邮,未满收 8 元。", content_type="faq")
    session.add(c); session.commit(); session.refresh(c)
    monkeypatch.setattr("app.tools.faq.get_embedder", lambda: type("E", (), {"embed_one": lambda self, t: [0.1] * 1024})())
    monkeypatch.setattr("app.tools.faq.get_vector_store", lambda: type("S", (), {"search": lambda self, v, k: [SearchHit(c.id, 0.7)]})())
    out = query_faq.invoke({"keyword": "邮费是多少"})
    assert "问:邮费怎么算?" in out and "答:满 99 元包邮" in out

def test_query_faq_miss_text(session, monkeypatch):
    monkeypatch.setattr("app.tools.faq.get_embedder", lambda: type("E", (), {"embed_one": lambda self, t: [0.1] * 1024})())
    monkeypatch.setattr("app.tools.faq.get_vector_store", lambda: type("S", (), {"search": lambda self, v, k: []})())
    assert query_faq.invoke({"keyword": "不存在"}) == "未找到相关问题,请换个关键词再试试。"

def test_query_faq_infrastructure_error_is_safe(session, monkeypatch):
    monkeypatch.setattr("app.tools.faq.get_embedder", lambda: (_ for _ in ()).throw(RuntimeError("down")))
    out = query_faq.invoke({"keyword": "邮费"})
    assert "知识库暂时不可用" in out
```

```python
# tests/test_tools_business.py — add
def test_tool_timeout_override():
    from app.tools.registry import tool_timeout
    assert tool_timeout("query_faq") == 8.0
    assert tool_timeout("query_order") == 5.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_retrieval.py tests/test_tools_faq.py tests/test_tools_business.py -v`
Expected: FAIL because `retrieve` / `tool_timeout` do not exist and old FAQ tests still expect LIKE.

- [ ] **Step 3: Implement**

`retrieve` embeds the query, calls `store.search(vector, top_k)`, drops hits with `score < threshold`, fetches `KnowledgeChunk` rows by id preserving hit order (skipping ids no longer present in MySQL, which can happen between a reset and a Milvus cleanup), returns `RetrievedChunk`s. Resolve defaults from settings. If `session` is None, open one via `db_session.get_session_factory()()` (import `from app.db import session as db_session` so the test fixture monkeypatch reaches it) and close it; injected session is not closed. `retrieve` resolves missing `embedder`/`store` via `get_embedder()` / `get_vector_store()` imported into `app.rag.retrieval`.

Rewrite `app/tools/faq.py` to keep the same `@tool` name and `keyword: str` signature; update the docstring to say it does semantic retrieval in the knowledge base (the old “模糊匹配” wording is no longer true). Import `get_embedder`, `get_vector_store`, and `retrieve` into `app.tools.faq`, and call `retrieve(keyword, embedder=get_embedder(), store=get_vector_store())` so tests can monkeypatch `app.tools.faq.get_embedder` / `app.tools.faq.get_vector_store`. Format as `"\n\n".join(f"问:{r.questions}\n答:{r.answer}")`, return the exact miss text when empty, and catch `Exception` (log it) returning `"知识库暂时不可用,请稍后重试或联系人工客服。"`.

In `registry.py`, add `TOOL_TIMEOUTS = {"query_faq": 8.0}` and `tool_timeout(name)`. In `app/services/chat.py`, call `execute_tool(tool, tc["args"], timeout_s=tool_timeout(tc["name"]))`. Add a chat test that a `query_faq` tool call passes `timeout_s=8.0` by monkeypatching `app.services.chat.execute_tool` and using a fake LLM that emits `{"name": "query_faq", "args": {"keyword": "邮费"}, "id": "c1"}`.

- [ ] **Step 4: Run tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_retrieval.py tests/test_tools_faq.py tests/test_tools_business.py tests/test_chat.py -v`
Expected: PASS; old LIKE tests removed.

- [ ] **Step 5: Append dev-notes + commit**

```bash
git add app/rag/retrieval.py app/tools/faq.py app/tools/registry.py app/services/chat.py tests/test_retrieval.py tests/test_tools_faq.py tests/test_tools_business.py tests/test_chat.py dev-notes/ch03.md
git commit -m "feat: replace query_faq with Milvus vector retrieval"
```

---

### Task 12: QA extraction prompt, schema, and dedup

**Files:**
- Create: `app/services/qa_mining.py`, `tests/test_qa_mining.py`

**Interfaces:**
- Produces:

```python
class ExtractedQA(BaseModel):
    question: str
    answer: str
    category: str

class QaExtractionBatch(BaseModel):
    items: list[ExtractedQA]

@dataclass(frozen=True)
class ConversationTranscript:
    source_ref: str
    messages: list[tuple[str, str]]   # (role, content) only user/assistant

QA_EXTRACTION_SYSTEM_PROMPT: str

def extract_qa_batch(llm, transcripts: list[ConversationTranscript]) -> list[ExtractedQA]
def normalize_category(value: str) -> str   # unknown -> "其他"
def normalize_question(text: str) -> str
def dedup_questions(candidates: list[str], existing: list[str],
                    embedder, threshold: float = 0.92) -> list[bool]
```

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_qa_mining.py
from app.services.qa_mining import (ConversationTranscript, extract_qa_batch,
                                    normalize_question, dedup_questions)

class FakeLLM:
    def with_structured_output(self, schema, method=None):
        class S:
            def invoke(self, messages):
                return schema(items=[{"question": "邮费是多少", "answer": "满99包邮", "category": "物流"},
                                     {"question": "邮费是多少?", "answer": "满99包邮", "category": "乱七八糟"}])
        return S()

def test_extract_normalizes_category():
    out = extract_qa_batch(FakeLLM(), [ConversationTranscript("c1", [("user", "邮费")])])
    assert [x.question for x in out] == ["邮费是多少", "邮费是多少?"]
    assert out[1].category == "其他"

def test_normalize_question():
    assert normalize_question(" 邮费，是多少？ ") == "邮费是多少"

class FakeEmbedder:
    def embed(self, texts):
        # two clearly separated directions: 邮费 vs 尺码
        out = []
        for t in texts:
            v = [0.0] * 1024
            v[0 if "邮费" in t else 1] = 1.0
            out.append(v)
        return out

def test_dedup_exact_and_existing():
    keep = dedup_questions(["邮费是多少", "邮费是多少?", "尺码怎么选"], ["邮费是多少"], FakeEmbedder())
    assert keep == [False, False, True]

def test_dedup_near_duplicate(monkeypatch):
    class E:
        def embed(self, texts):
            return [[1.0] + [0.0] * 1023 if "邮费" in t else [0.0, 1.0] + [0.0] * 1022 for t in texts]
    assert dedup_questions(["邮费是多少", "邮费怎么算"], [], E(), threshold=0.9) == [True, False]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_qa_mining.py -v`
Expected: FAIL because module does not exist.

- [ ] **Step 3: Implement**

`QA_EXTRACTION_SYSTEM_PROMPT` must instruct: extract only reusable, non-order-specific Q&A; keep the user's real question; compress the agent's answer; category must be one of `售后/物流/支付/商品/其他`; output a JSON object `{"items": [{"question": "...", "answer": "...", "category": "..."}]}`; return `{"items": []}` if nothing reusable. `extract_qa_batch` serializes transcripts as JSON in a `HumanMessage`, calls `llm.with_structured_output(QaExtractionBatch, method="json_mode").invoke(...)`, drops empty question/answer, and maps unknown categories through `normalize_category`.

`normalize_question` uses `unicodedata.normalize("NFKC", text).lower()`, removes whitespace and punctuation (`re.sub(r"[\s\W_]+", "", text)`), preserving Chinese. `dedup_questions` keeps first occurrence by normalized exact match; then embeds candidates and existing questions, computes cosine, and drops a candidate if cosine ≥ threshold against any already-kept candidate or existing question. Return one bool per candidate, in order.

- [ ] **Step 4: Run tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_qa_mining.py -v`
Expected: PASS.

- [ ] **Step 5: Append dev-notes + commit**

```bash
git add app/services/qa_mining.py tests/test_qa_mining.py dev-notes/ch03.md
git commit -m "feat: add QA extraction prompt and dedup helpers"
```

---

### Task 13: `mine_qa` job with staging and global dedup

**Files:**
- Create: `app/jobs/mine_qa.py`, `tests/test_mine_qa.py`

**Interfaces:**
- Consumes: `ConversationTranscript`, `extract_qa_batch`, `dedup_questions` (Task 12); `insert_mined_qa`, `MinedQA` (Task 9); `vectorize_pending` (Task 8).
- Produces:

```python
@dataclass(frozen=True)
class MiningReport:
    batches: int
    extracted: int
    kept: int
    discarded: int
    inserted: int
    vectorized: int

def load_transcripts(session, *, limit=None, since=None, force=False) -> list[ConversationTranscript]
def mine_qa(session, llm, embedder, store, *, limit=None, batch_size=10,
            since=None, force=False, dry_run=False, do_vectorize=True) -> MiningReport
def main(argv: list[str] | None = None) -> int
```

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_mine_qa.py
from datetime import datetime
from app.models import Conversation, Message, QaExtractionStaging, KnowledgeChunk
from app.jobs.mine_qa import mine_qa

class FakeLLM:
    def with_structured_output(self, schema, method=None):
        class S:
            def invoke(self, messages):
                return schema(items=[{"question": "邮费是多少", "answer": "满99包邮", "category": "物流"},
                                     {"question": "邮费是多少?", "answer": "满99包邮", "category": "物流"}])
        return S()

class FakeEmbedder:
    def embed(self, texts): return [[1.0] + [0.0] * 1023 for _ in texts]

class FakeStore:
    def __init__(self): self.items = []
    def upsert(self, items): self.items.extend(items)

def _conv(session, uid):
    c = Conversation(user_id=uid, status="已结束"); session.add(c); session.flush()
    session.add_all([Message(conversation_id=c.id, role="user", content="邮费是多少"),
                     Message(conversation_id=c.id, role="assistant", content="满99包邮")])
    session.commit(); return c

def test_mine_dedups_and_inserts(session):
    _conv(session, "u1")
    r = mine_qa(session, FakeLLM(), FakeEmbedder(), FakeStore())
    assert r.extracted == 2 and r.kept == 1 and r.discarded == 1 and r.inserted == 1
    assert session.query(QaExtractionStaging).count() == 2
    assert session.query(QaExtractionStaging).filter_by(status="discarded").count() == 1
    assert session.query(KnowledgeChunk).filter_by(content_type="mined_qa").count() == 1

def test_rerun_skips_already_mined(session):
    _conv(session, "u1")
    mine_qa(session, FakeLLM(), FakeEmbedder(), FakeStore())
    r = mine_qa(session, FakeLLM(), FakeEmbedder(), FakeStore())
    assert r.extracted == 0 and r.inserted == 0

def test_dry_run_writes_nothing(session):
    _conv(session, "u1")
    r = mine_qa(session, FakeLLM(), FakeEmbedder(), FakeStore(), dry_run=True)
    assert r.kept == 1
    assert session.query(QaExtractionStaging).count() == 0
    assert session.query(KnowledgeChunk).count() == 0

def test_stale_extracted_rows_are_cleared_before_re_extract(session):
    _conv(session, "u1")
    session.add(QaExtractionStaging(batch_no="old", source_ref="1", question="旧", answer="旧"))
    session.commit()
    mine_qa(session, FakeLLM(), FakeEmbedder(), FakeStore())
    assert session.query(QaExtractionStaging).filter_by(batch_no="old").count() == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_mine_qa.py -v`
Expected: FAIL because job does not exist.

- [ ] **Step 3: Implement**

`load_transcripts` joins conversations/messages, keeps only user/assistant content, groups by conversation, and applies `since`. Then **first** deletes `extracted` rows whose `source_ref` is in the candidate set (interrupted-run cleanup), and **only after that** skips `source_ref`s that still have `kept`/`discarded` staging rows unless `force`; finally applies `limit` and orders by `updated_at`. This order is required so a stale `extracted` row cannot make its conversation look “already mined”. Batch transcripts by `batch_size`; call `extract_qa_batch`; insert staging rows with a fresh `batch_no` (`qa-YYYYmmddHHMMSS-<4 hex>`) and `source_ref`; commit per batch. After all batches, gather all `extracted` rows, run `dedup_questions` against existing `knowledge_chunks.questions` (split on newlines), set kept/discarded, then `insert_mined_qa` for kept rows. If `do_vectorize`, call `vectorize_pending`. `dry_run` performs extraction and in-memory dedup but performs **no** stale-row deletion or staging/knowledge writes; skip the cleanup step in that mode. CLI parses `--limit`, `--batch-size`, `--since`, `--force`, `--dry-run`, `--no-vectorize`.

- [ ] **Step 4: Run tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_mine_qa.py -v`
Expected: PASS.

- [ ] **Step 5: Append dev-notes + commit**

```bash
git add app/jobs/mine_qa.py tests/test_mine_qa.py dev-notes/ch03.md
git commit -m "feat: add conversation QA mining job"
```

---

### Task 14: Evaluation harness and annotated samples

**Files:**
- Create: `app/jobs/eval_retrieval.py`, `app/jobs/eval_qa_mining.py`, `tests/eval/faq_cases.json`, `tests/eval/qa_mining_samples.json`, `tests/test_eval_scoring.py`

**Interfaces:**
- Produces:

```python
def score_faq_case(case: dict, hits: list) -> bool
def score_mining_case(case: dict, got: list) -> bool
def main(argv: list[str] | None = None) -> int   # one per script
```

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_scoring.py
from app.jobs.eval_retrieval import score_faq_case
from app.rag.retrieval import RetrievedChunk

def test_score_hit_by_keyword():
    hits = [RetrievedChunk(1, 0.7, "邮费怎么算?", "满 99 元包邮,未满收 8 元。")]
    assert score_faq_case({"query": "邮费是多少", "expect_hit": True, "expect_any": ["包邮", "运费"]}, hits)
    assert not score_faq_case({"query": "x", "expect_hit": True, "expect_any": ["发票"]}, hits)

def test_score_negative_case():
    assert score_faq_case({"query": "老板是谁", "expect_hit": False}, [])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_eval_scoring.py -v`
Expected: FAIL because eval script does not exist.

- [ ] **Step 3: Implement eval data and scripts**

`tests/eval/faq_cases.json` must include at least these cases:

```json
{"cases": [
  {"query": "邮费是多少", "expect_hit": true, "expect_any": ["包邮", "运费", "8 元"]},
  {"query": "退货运费谁承担", "expect_hit": true, "expect_any": ["商家承担", "买家承担"]},
  {"query": "几天能退款", "expect_hit": true, "expect_any": ["1-3", "1 到 3", "工作日"]},
  {"query": "怎么开发票", "expect_hit": true, "expect_any": ["发票"]},
  {"query": "尺码怎么选", "expect_hit": true, "expect_any": ["尺码", "脚长"]},
  {"query": "你们老板是谁", "expect_hit": false}
]}
```

`eval_retrieval` calls `retrieve(query)` with settings defaults, prints each hit's score/questions/answer snippet, prints PASS/FAIL and a summary, exits non-zero if any case fails. `score_faq_case` returns true for `expect_hit=false` when hits are empty; for hit cases, any hit answer containing any `expect_any` keyword passes.

`tests/eval/qa_mining_samples.json` contains 2 transcripts (one return-policy conversation with duplicate questions, one shipping-fee conversation) plus expected question/answer keywords. `eval_qa_mining` calls `extract_qa_batch` with the real configured LLM, checks each expected item has a candidate whose question contains a question keyword and answer contains an answer keyword, prints PASS/FAIL, and exits non-zero on failure. `score_mining_case` is the pure scoring helper used by the test.

- [ ] **Step 4: Run unit tests, then live eval**

Run: `.venv/Scripts/python.exe -m pytest tests/test_eval_scoring.py -v`
Expected: PASS.

After Task 10/13 are built and Milvus is up, run:
`.venv/Scripts/python.exe -m app.jobs.eval_retrieval`
Expected: all cases PASS at the chosen threshold; record the actual scores.

Run:
`.venv/Scripts/python.exe -m app.jobs.eval_qa_mining`
Expected: key QA pairs found; record the output.

If retrieval cases fail on threshold, adjust `FAQ_SCORE_THRESHOLD` in `.env` (and the default in `app/config.py` if the calibration should be permanent) and rerun until pass; record the final value in dev-notes.

- [ ] **Step 5: Append dev-notes + commit**

```bash
git add app/jobs/eval_retrieval.py app/jobs/eval_qa_mining.py tests/eval tests/test_eval_scoring.py app/config.py .env.example dev-notes/ch03.md
git commit -m "test: add retrieval and QA mining evaluation harness"
```

---

### Task 15: Acceptance, full test run, and delivery notes

**Files:**
- Modify: `dev-notes/ch03.md`

**Interfaces:**
- Consumes: all prior tasks.

- [ ] **Step 1: Run the full test suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: all tests pass; record the count. If any fail, fix before proceeding.

- [ ] **Step 2: Build the knowledge base and verify counts**

Run:

```bash
docker compose up -d
.venv/Scripts/python.exe -m app.db.init_ch03
.venv/Scripts/python.exe -m app.jobs.build_knowledge
```

Then verify:

```bash
docker exec cs_customer_mysql mysql -uroot -proot --default-character-set=utf8mb4 cs_customer \
  -e "SELECT vectorize_status, COUNT(*) FROM knowledge_chunks GROUP BY vectorize_status;"
```

Expected: `done` count > 0, `pending` count = 0. Verify Milvus entity count equals the `done` count via a short Python check using `get_vector_store().count()`.

- [ ] **Step 3: Acceptance 1 — semantic recall**

Run:

```bash
.venv/Scripts/python.exe -c "from app.tools.faq import query_faq; print(query_faq.invoke({'keyword':'邮费是多少'}))"
```

Expected: output contains `包邮` / `运费` / `8 元`, i.e., the shipping-fee section, not the miss text. Then start the API and send a `/chat` request with `邮费是多少`; record whether the final answer uses the shipping-fee facts. (The tool result is deterministic; the LLM wording may vary.)

- [ ] **Step 4: Acceptance 2 — interrupt and resume**

Run a controlled interruption:

```bash
# leave some chunks pending
.venv/Scripts/python.exe -m app.jobs.build_knowledge --reset --no-vectorize
.venv/Scripts/python.exe -m app.jobs.vectorize --max-chunks 2
# expect pending > 0 here
.venv/Scripts/python.exe -m app.jobs.build_knowledge
```

Expected: after the final build, `pending = 0`, Milvus entity count equals `done` count, and running `build_knowledge` again does not increase either count. Record the exact before/after counts.

- [ ] **Step 5: Acceptance 3 — evaluation**

Run both eval commands from Task 14 and record pass/fail plus the calibrated `FAQ_SCORE_THRESHOLD`.

- [ ] **Step 6: Append final dev-notes and commit**

Append `## 阶段:验收与交付` to `dev-notes/ch03.md` with:

- user's key acceptance words;
- demo commands and test counts;
- acceptance 1/2/3 results with actual numbers;
- any rejections/corrections and any rework;
- spec/plan paths.

```bash
git add dev-notes/ch03.md
git commit -m "docs: record ch03 acceptance results and delivery notes"
```

- [ ] **Step 7: Request code review, then finish**

Use `superpowers:requesting-code-review` (or `/code-review` at the level the user chooses) on the full branch. Append the review conclusion and any fixes to `dev-notes/ch03.md`, then use `superpowers:finishing-a-development-branch` to integrate/merge; append the finish decision to dev-notes as well.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-11-12-ecommerce-cs-ch03-rag.md`. Please review the plan. Which execution approach would you prefer?

- **Subagent-driven** — a fresh subagent implements each task and a fresh reviewer checks it before the next one starts, then a whole-branch review at the end. Most thorough; costs a fresh context per task and per review.
- **Native** — I implement every task myself in this session, the way this harness runs work, then one fresh reviewer on the most capable model checks the whole branch. Cheapest and fastest; no independent review until the end. Runs well with a mid-tier session model, since the plan carries the design.

**For this plan I recommend Subagent-driven**, because there are 15 tightly sequenced tasks with shared interfaces (chunk fields, vector_id, staging semantics) and two data-loss-prone areas (`--reset`, mining reruns); a fresh reviewer per task catches interface drift before it compounds. Does the plan capture what you want, and which approach should we use?
