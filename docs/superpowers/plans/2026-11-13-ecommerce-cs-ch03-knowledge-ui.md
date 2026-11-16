# ch03 知识录入页(`/kb`) Implementation Plan(原 ch04)

> 2026-11-16 章节归位:本计划原编号 ch04,已并入 ch03;原 ch04 让位给「RAG 进阶」。分支名 `feature/ch04-knowledge-ui` 为历史记录,保留不改。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有 FastAPI 8000 上提供 `http://localhost:8000/kb`:上传 Markdown(页面代填元数据)→ 触发文档域重建 → 查看 pending/done 状态。

**Architecture:** FastAPI 直接返回自包含静态页 `/kb`;`/kb/api/upload` 校验并写入 `knowledge/uploads/`,`/kb/api/build` 在后台线程调用 ch03 的 `build_knowledge(reset=True)`,`/kb/api/status` 查询文档域 pending/done 与内存构建状态,`/kb/api/files` 列出 root+uploads 的有效语料。loader 改为「root seed + uploads,同名 uploads shadow seed」,不新增表、不改 Milvus schema。

**Tech Stack:** Python 3.13, FastAPI 0.162.4, Starlette 1.10.0, SQLAlchemy 2.x, MySQL, python-multipart, pytest, 原生 HTML/CSS/JS.

**Spec:** `docs/superpowers/specs/2026-11-12-ecommerce-cs-ch03-knowledge-ui-design.md`

## Global Constraints

- 页面地址固定 `http://localhost:8000/kb`;由 FastAPI 提供,不依赖 Vite/React。
- 建库复用 `app.jobs.build_knowledge.build_knowledge`,参数 `reset=True`(只重建文档域,保留 `mined_qa`)。
- 上传目录固定 `knowledge/uploads/`;只收 `.md`;单文件 ≤ 2 MiB;单次 ≤ 20 个;UTF-8(允许 BOM)。
- 页面选 `content_type`(policy/faq/manual)+ 可选 `category`;后端仅在文件缺 frontmatter 时注入;已有 frontmatter 以文件内为准。
- 建库语料 = `knowledge/` 根目录 seed + `knowledge/uploads/`;**同名 uploads 优先(shadow seed)**;页面只列有效文档并标 `source`。
- 状态主计数只统计文档域(`content_type != 'mined_qa' OR content_type IS NULL`);`mined_qa` 单独计数,不参与进度条。
- 不新增数据库表;不改 `sql/ch03_knowledge.sql`;不改 Milvus 集合 schema。
- 无认证,本地 demo;页面与交付说明明确不要暴露公网。
- 构建中上传/建库均返回 409;构建状态只存单进程内存。
- 后端 TDD;每个任务完成即追加 `dev-notes/ch03.md`(四件事:用户原话/产出/拒绝纠偏/翻车返工)。
- 分支:`feature/ch04-knowledge-ui`,从 `feature/ch02-tools` 起(ch03 已合并)。
- 新增依赖仅 `python-multipart`;不引入前端框架/队列/新库。

## Review Focus

1. **上传文件不能破坏建库**:路径穿越、超大、非法 UTF-8、缺 H1、非法/缺失 frontmatter 都要被拒;合法文件落盘后必须能被 `load_document` 解析。—— Task 4 的上传测试。
2. **同名覆盖语义**:root seed 与 uploads 同名时,loader/建库/文件列表只看到 upload;不同名时两者都在;不得出现新旧 chunk 共存。—— Task 1 / Task 3 / Task 6 的测试与验收。
3. **构建并发与失败**:重复触发返回 409;后台异常写 `last_error` 且进程不挂;线程自开自关 SQLAlchemy session。—— Task 2 / Task 3 的测试。
4. **状态口径**:文档域 pending/done/total 与 `mined_qa` 分开;`total` 不含 mined;`last_report.vectorized` 只标注为本次向量化。—— Task 2 / Task 3 的测试。
5. **页面可用性与安全**:`/kb` 200、上传/建库/状态控件齐全;动态文本用 `textContent`;构建中禁用按钮;页面不引外部 CDN。—— Task 5 的测试与手工冒烟。

---

### Before Task 1: 计划评审留痕

- [ ] **Append the plan-review entry to `dev-notes/ch03.md`** under `## 阶段:计划评审通过`: user's approval words, plan path, self-review conclusion (spec coverage / step scan / type consistency / review focus / proportion), rejections/corrections, rework. Do this before Task 1.

---

### Task 1: Effective corpus (root + uploads, uploads shadow seed)

**Files:**
- Modify: `app/rag/docs.py`
- Modify: `app/jobs/build_knowledge.py`
- Test: `tests/test_docs.py`, `tests/test_build_knowledge.py`

**Interfaces:**
- Consumes: existing `Document`, `load_document`.
- Produces:

```python
# app/rag/docs.py
def effective_doc_paths(docs_dir: Path) -> list[Path]
def load_documents(docs_dir: Path) -> list[Document]  # now uses effective_doc_paths
```

- `effective_doc_paths` 规则:`docs_dir/*.md`(root seed)+ `docs_dir/uploads/*.md`;按文件名建 map,upload 覆盖同名 root;返回按文件名排序的路径。只读这两层,不递归其他子目录。
- `build_knowledge._validate_docs_dir` 改用 `effective_doc_paths` 判断非空。

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_docs.py — append
from app.rag.docs import effective_doc_paths

def _write_doc(path, title, body="正文。"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\ncontent_type: policy\n---\n# {title}\n\n{body}\n", encoding="utf-8"
    )

def test_uploads_shadow_root_same_name(tmp_path):
    root = tmp_path / "knowledge"; root.mkdir()
    _write_doc(root / "退货政策.md", "seed 版")
    _write_doc(root / "uploads" / "退货政策.md", "upload 版")
    _write_doc(root / "运费.md", "运费 seed")
    paths = effective_doc_paths(root)
    assert [p.name for p in paths] == ["退货政策.md", "运费.md"]
    docs = load_documents(root)
    assert {d.title for d in docs} == {"upload 版", "运费 seed"}

def test_uploads_additive_when_names_differ(tmp_path):
    root = tmp_path / "knowledge"; root.mkdir()
    _write_doc(root / "a.md", "A")
    _write_doc(root / "uploads" / "b.md", "B")
    assert {d.title for d in load_documents(root)} == {"A", "B"}
```

```python
# tests/test_build_knowledge.py — append
def test_build_accepts_uploads_only_corpus(session, tmp_path):
    d = tmp_path / "knowledge"; (d / "uploads").mkdir(parents=True)
    (d / "uploads" / "only.md").write_text(
        "---\ncontent_type: policy\n---\n# 只有上传\n\n正文。\n", encoding="utf-8"
    )
    r = build_knowledge(session, d, FakeEmbedder(), FakeStore(), do_vectorize=False)
    assert r.documents == 1 and r.chunks == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_docs.py tests/test_build_knowledge.py -q`
Expected: FAIL because `effective_doc_paths` does not exist / uploads-only corpus is rejected.

- [ ] **Step 3: Implement**

In `app/rag/docs.py`, add `effective_doc_paths` with the root+uploads merge/shadow rule; change `load_documents` to iterate it. In `app/jobs/build_knowledge.py`, `_validate_docs_dir` calls `effective_doc_paths(docs_dir)` and raises `CorpusError` if empty. Keep `load_document` unchanged.

- [ ] **Step 4: Run tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_docs.py tests/test_build_knowledge.py -q`
Expected: PASS; existing ch03 loader/build tests unchanged.

- [ ] **Step 5: Append dev-notes + commit**

```bash
git add app/rag/docs.py app/jobs/build_knowledge.py tests/test_docs.py tests/test_build_knowledge.py dev-notes/ch03.md
git commit -m "feat: load root and uploads with upload shadowing"
```

---

### Task 2: KB build state machine and status counts

**Files:**
- Create: `app/services/kb_build.py`, `tests/test_kb_build.py`

**Interfaces:**
- Consumes: `build_knowledge`, `get_settings`, `get_embedder`, `get_vector_store`, `db_session`.
- Produces:

```python
@dataclass
class BuildState:
    running: bool = False
    started_at: datetime | None = None
    finished_at: datetime | None = None
    last_report: BuildReport | None = None
    last_error: str | None = None

def is_running() -> bool
def start_build() -> bool                 # False = already running
def wait_for_build(timeout: float | None = None) -> bool
def snapshot() -> dict                    # state only
def status(session) -> dict               # counts + state
def reset_state_for_tests() -> None
```

`status(session)` returns exactly:

```python
{
  "pending": int, "done": int, "total": int,
  "mined_total": int, "mined_pending": int, "mined_done": int,
  "running": bool,
  "started_at": str | None, "finished_at": str | None,
  "last_report": dict | None, "last_error": str | None,
}
```

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_kb_build.py
import threading
from app.models import KnowledgeChunk
from app.jobs.build_knowledge import BuildReport
from app.services import kb_build

def _row(session, status="pending", content_type="policy"):
    c = KnowledgeChunk(category="c", questions="q", answer="a",
                       content_type=content_type, vectorize_status=status)
    session.add(c); session.commit(); return c

def test_status_splits_document_and_mined(session):
    _row(session, "pending"); _row(session, "done")
    _row(session, "pending", "mined_qa"); _row(session, "done", "mined_qa")
    s = kb_build.status(session)
    assert (s["pending"], s["done"], s["total"]) == (1, 1, 2)
    assert (s["mined_pending"], s["mined_done"], s["mined_total"]) == (1, 1, 2)

def test_start_build_success(monkeypatch, session):
    kb_build.reset_state_for_tests()
    monkeypatch.setattr(kb_build, "get_settings", lambda: type("S", (), {"knowledge_docs_dir": "knowledge"})())
    monkeypatch.setattr(kb_build, "get_embedder", lambda: object())
    monkeypatch.setattr(kb_build, "get_vector_store", lambda: object())
    def fake_build(session, docs_dir, embedder, store, *, reset=False):
        return BuildReport(documents=1, chunks=2, inserted=2, reused=0, vectorized=2)
    monkeypatch.setattr(kb_build, "build_knowledge", fake_build)
    assert kb_build.start_build() is True
    assert kb_build.wait_for_build(2) is True
    s = kb_build.status(session)
    assert s["running"] is False and s["last_report"]["documents"] == 1

def test_start_build_rejects_concurrent(monkeypatch, session):
    kb_build.reset_state_for_tests()
    gate = threading.Event()
    monkeypatch.setattr(kb_build, "get_settings", lambda: type("S", (), {"knowledge_docs_dir": "knowledge"})())
    monkeypatch.setattr(kb_build, "get_embedder", lambda: object())
    monkeypatch.setattr(kb_build, "get_vector_store", lambda: object())
    def blocking(*a, **k):
        gate.wait(5); return BuildReport(0, 0, 0, 0, 0)
    monkeypatch.setattr(kb_build, "build_knowledge", blocking)
    assert kb_build.start_build() is True
    assert kb_build.start_build() is False
    gate.set(); assert kb_build.wait_for_build(2) is True

def test_build_failure_is_visible(monkeypatch, session):
    kb_build.reset_state_for_tests()
    monkeypatch.setattr(kb_build, "get_settings", lambda: type("S", (), {"knowledge_docs_dir": "knowledge"})())
    monkeypatch.setattr(kb_build, "get_embedder", lambda: object())
    monkeypatch.setattr(kb_build, "get_vector_store", lambda: object())
    def boom(*a, **k): raise RuntimeError("boom")
    monkeypatch.setattr(kb_build, "build_knowledge", boom)
    assert kb_build.start_build() is True
    assert kb_build.wait_for_build(2) is True
    assert "boom" in kb_build.status(session)["last_error"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_kb_build.py -q`
Expected: FAIL because module does not exist.

- [ ] **Step 3: Implement**

Module-level `_state`, `_lock`, `_done = threading.Event()`. `start_build` checks/sets `running` under the lock and starts a daemon thread. The thread opens `db_session.get_session_factory()()`, resolves `get_settings()`, calls `build_knowledge(session, Path(settings.knowledge_docs_dir), get_embedder(), get_vector_store(), reset=True)`, records the report, sets `running=False`/`finished_at`/`_done`, catches exceptions into `last_error`, and closes the session in `finally`. `status(session)` queries document-domain and `mined_qa` counts separately with `or_(content_type != 'mined_qa', content_type.is_(None))`; returns the exact keys above with ISO timestamps and `asdict(last_report)`.

- [ ] **Step 4: Run tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_kb_build.py -q`
Expected: PASS.

- [ ] **Step 5: Append dev-notes + commit**

```bash
git add app/services/kb_build.py tests/test_kb_build.py dev-notes/ch03.md
git commit -m "feat: add KB build state machine and status counts"
```

---

### Task 3: KB API router (files/status/build) + dependency

**Files:**
- Create: `app/api/__init__.py`, `app/api/kb.py`
- Modify: `app/main.py`, `requirements.txt`
- Test: `tests/test_kb_api.py`

**Interfaces:**
- Consumes: `effective_doc_paths`, `kb_build` service.
- Produces:

```python
# app/api/kb.py
api_router = APIRouter(prefix="/kb/api")

@api_router.get("/files") -> dict
@api_router.get("/status") -> dict
@api_router.post("/build", status_code=202) -> dict
```

- `app/main.py` calls `app.include_router(api_router)`.
- `requirements.txt` adds `python-multipart`.
- `GET /kb` page route is **Task 5**, not here.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_kb_api.py
from pathlib import Path
from fastapi.testclient import TestClient
from app.main import app

def _settings(tmp_path):
    from app.config import Settings
    return Settings(_env_file=None, openai_api_key="x", knowledge_docs_dir=str(tmp_path / "knowledge"))

def test_files_lists_effective_docs_with_sources(monkeypatch, tmp_path):
    import app.api.kb as kb
    from app.rag.docs import effective_doc_paths
    root = tmp_path / "knowledge"; (root / "uploads").mkdir(parents=True)
    (root / "seed.md").write_text("---\ncontent_type: policy\n---\n# Seed\n\nS\n", encoding="utf-8")
    (root / "uploads" / "seed.md").write_text("---\ncontent_type: policy\n---\n# Upload\n\nU\n", encoding="utf-8")
    monkeypatch.setattr(kb, "get_settings", lambda: _settings(tmp_path))
    r = TestClient(app).get("/kb/api/files")
    assert r.status_code == 200
    assert r.json()["files"] == [{"name": "seed.md", "source": "upload", "size": (root/"uploads"/"seed.md").stat().st_size, "modified_at": r.json()["files"][0]["modified_at"]}]

def test_status_returns_counts_and_state(monkeypatch, session):
    import app.api.kb as kb
    from app.models import KnowledgeChunk
    session.add(KnowledgeChunk(category="c", questions="q", answer="a", content_type="policy", vectorize_status="done"))
    session.commit()
    monkeypatch.setattr(kb, "get_settings", lambda: _settings(__import__("pathlib").Path("/tmp")))
    r = TestClient(app).get("/kb/api/status")
    assert r.status_code == 200
    body = r.json()
    assert body["done"] == 1 and body["total"] == 1 and body["mined_total"] == 0

def test_build_returns_202_and_409(monkeypatch):
    import app.api.kb as kb
    monkeypatch.setattr(kb.kb_build, "start_build", lambda: True)
    assert TestClient(app).post("/kb/api/build").status_code == 202
    monkeypatch.setattr(kb.kb_build, "start_build", lambda: False)
    assert TestClient(app).post("/kb/api/build").status_code == 409
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_kb_api.py -q`
Expected: FAIL because router/module does not exist.

- [ ] **Step 3: Implement**

`app/api/kb.py`: `api_router` with the three endpoints. `files` uses `effective_doc_paths(Path(settings.knowledge_docs_dir))`, returns `name/source/size/modified_at` (`source` = `"upload"` when `path.parent.name == "uploads"` else `"seed"`). `status` opens `db_session.get_session_factory()()`, calls `kb_build.status(session)`, closes in `finally`; DB errors become 500. `build` calls `kb_build.start_build()`; 202 `{"status":"running"}` or 409 `{"detail":"a build is already running"}`. `app/main.py` includes the router. Add `python-multipart` to `requirements.txt` and install it.

- [ ] **Step 4: Run tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_kb_api.py tests/test_kb_build.py -q`
Expected: PASS.

- [ ] **Step 5: Append dev-notes + commit**

```bash
git add app/api/__init__.py app/api/kb.py app/main.py requirements.txt tests/test_kb_api.py dev-notes/ch03.md
git commit -m "feat: add KB files/status/build API"
```

---

### Task 4: Upload endpoint with metadata injection and validation

**Files:**
- Modify: `app/api/kb.py`
- Test: `tests/test_kb_api.py`

**Interfaces:**
- Produces: `POST /kb/api/upload` (multipart: `files`, `content_type`, optional `category`) returning `{"saved": [...], "rejected": [...]}`.
- Helpers in `app/api/kb.py`:

```python
MAX_UPLOAD_BYTES = 2 * 1024 * 1024
MAX_FILES = 20
ALLOWED_CONTENT_TYPES = {"policy", "faq", "manual"}

def has_frontmatter(text: str) -> bool
def inject_frontmatter(text: str, content_type: str, category: str | None) -> str
def uploads_dir(settings) -> Path
```

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_kb_api.py — append
def _client(monkeypatch, tmp_path):
    import app.api.kb as kb
    monkeypatch.setattr(kb, "get_settings", lambda: _settings(tmp_path))
    return TestClient(app)

def test_upload_plain_md_injects_frontmatter(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    r = c.post("/kb/api/upload",
               files={"files": ("new.md", b"# New\n\nBody\n", "text/markdown")},
               data={"content_type": "policy", "category": "售后"})
    assert r.status_code == 200
    saved = r.json()["saved"][0]
    assert saved["content_type"] == "policy" and saved["name"] == "new.md"
    p = tmp_path / "knowledge" / "uploads" / "new.md"
    text = p.read_text(encoding="utf-8")
    assert text.startswith("---\ncontent_type: policy\ncategory: 售后\n---")
    from app.rag.docs import load_document
    assert load_document(p).title == "New"

def test_upload_existing_frontmatter_wins(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    body = "---\ncontent_type: faq\ncategory: 商品\n---\n# FAQ\n\n## 尺码怎么选?\n\n按脚长。\n"
    r = c.post("/kb/api/upload",
               files={"files": ("faq.md", body.encode(), "text/markdown")},
               data={"content_type": "policy"})
    assert r.json()["saved"][0]["content_type"] == "faq"

def test_upload_rejects_bad_files(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    for name, data in [("a.txt", b"# A\n"), ("bad.md", b"\xff\xfe"), ("noh1.md", b"no heading\n")]:
        r = c.post("/kb/api/upload", files={"files": (name, data, "text/markdown")}, data={"content_type": "policy"})
        assert r.status_code == 200 and r.json()["saved"] == []
    assert not any((tmp_path / "knowledge" / "uploads").glob("*.md"))

def test_upload_rejects_path_traversal_and_invalid_type(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    r = c.post("/kb/api/upload", files={"files": ("../x.md", b"# X\n", "text/markdown")}, data={"content_type": "policy"})
    assert r.json()["saved"] == []
    r = c.post("/kb/api/upload", files={"files": ("x.md", b"# X\n", "text/markdown")}, data={"content_type": "blog"})
    assert r.status_code == 400

def test_upload_rejects_oversized_file(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    data = b"# X\n" + b"x" * (2 * 1024 * 1024)
    r = c.post("/kb/api/upload", files={"files": ("big.md", data, "text/markdown")}, data={"content_type": "policy"})
    assert r.json()["saved"] == []
    assert "2 MiB" in r.json()["rejected"][0]["reason"] or "size" in r.json()["rejected"][0]["reason"].lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_kb_api.py -q`
Expected: FAIL because `/kb/api/upload` does not exist.

- [ ] **Step 3: Implement**

Add the route with `files: list[UploadFile] = File(...)`, `content_type: str = Form(...)`, `category: str | None = Form(None)`. Validate `content_type` in `ALLOWED_CONTENT_TYPES` (400), `len(files) <= MAX_FILES` (400), `kb_build.is_running()` (409), and `category` contains no newline (400). For each file: reject raw names containing `/` or `\`, non-`.md`, empty; read bytes, reject > 2 MiB, decode `utf-8-sig`; if `has_frontmatter` use text as-is, else `inject_frontmatter`; write to a `.tmp-<uuid>` in uploads dir, call `load_document` for validation, then `os.replace` to final name; append saved or rejected with reason. Always clean the temp file. Return 200 with saved/rejected.

- [ ] **Step 4: Run tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_kb_api.py -q`
Expected: PASS.

- [ ] **Step 5: Append dev-notes + commit**

```bash
git add app/api/kb.py tests/test_kb_api.py dev-notes/ch03.md
git commit -m "feat: add validated markdown upload with metadata injection"
```

---

### Task 5: `/kb` static page + page route

**Files:**
- Create: `app/static/kb.html`
- Modify: `app/api/kb.py` (page router), `app/main.py`
- Test: `tests/test_kb_api.py`

**Interfaces:**
- Produces: `GET /kb` returning `FileResponse(app/static/kb.html, media_type="text/html")`.
- The page calls the Task 3/4 endpoints.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_kb_api.py — append
def test_kb_page_is_served():
    r = TestClient(app).get("/kb")
    assert r.status_code == 200
    body = r.text
    for needle in ["知识库录入", 'id="upload"', 'id="build"', 'id="status"', "/kb/api/upload", "/kb/api/build", "/kb/api/status"]:
        assert needle in body
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_kb_api.py::test_kb_page_is_served -q`
Expected: FAIL because `/kb` is not routed.

- [ ] **Step 3: Implement**

Create `app/static/kb.html`: self-contained HTML/CSS/JS with:
- header「知识库录入」+ warning「不要暴露公网」;
- upload section: file input (`accept=".md"`, multiple), `content_type` select, `category` input, upload button; result list;
- files section: effective docs with `source` badge;
- build section: start button, status cards (`pending`/`done`/`total`), progress bar, mined line, last report, last error, timestamps;
- JS: `fetch` to `/kb/api/files|upload|build|status`; use `textContent` for all dynamic text; disable upload/build while running; poll status every 2s while running or pending > 0; show rejected reasons/errors.
Add `page_router = APIRouter()` with `@page_router.get("/kb", response_class=FileResponse)` in `app/api/kb.py`; include it in `app/main.py`. Path resolution via `Path(__file__).resolve().parents[1] / "static" / "kb.html"`.

- [ ] **Step 4: Run tests and smoke**

Run: `.venv/Scripts/python.exe -m pytest tests/test_kb_api.py -q`
Expected: PASS.

Run uvicorn and `curl -fsS http://localhost:8000/kb | head`; expected 200/HTML. Manually check in a browser: upload a `.md`, see it in the list, click build, watch status.

- [ ] **Step 5: Append dev-notes + commit**

```bash
git add app/static/kb.html app/api/kb.py app/main.py tests/test_kb_api.py dev-notes/ch03.md
git commit -m "feat: add /kb knowledge entry page"
```

---

### Task 6: Integration acceptance and delivery notes

**Files:**
- Modify: `dev-notes/ch03.md`

**Interfaces:**
- Consumes all prior tasks.

- [ ] **Step 1: Run the full test suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: all tests pass; record the count. ch03's 185 tests must remain included.

- [ ] **Step 2: Live smoke**

Start the API:

```bash
.venv/Scripts/python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Then:

```bash
curl -fsS http://localhost:8000/kb | head -5
curl -fsS http://localhost:8000/kb/api/files
```

Expected: `/kb` HTML 200; files list includes the four seed docs.

- [ ] **Step 3: Upload + build acceptance**

Create a temporary Markdown without frontmatter:

```bash
printf '# 知识库验收文档\n\n这是一份用于验收的上传文档。\n' > /tmp/kb-accept.md
curl -fsS -X POST http://localhost:8000/kb/api/upload \
  -F 'files=@/tmp/kb-accept.md;type=text/markdown' \
  -F 'content_type=policy' \
  -F 'category=其他'
```

Expected: `saved` contains `kb-accept.md`, `content_type=policy`; the file at `knowledge/uploads/kb-accept.md` starts with injected frontmatter.

Trigger build and poll:

```bash
curl -fsS -X POST http://localhost:8000/kb/api/build
for i in $(seq 1 60); do
  curl -fsS http://localhost:8000/kb/api/status
  sleep 2
done
```

Expected: status transitions to `running` then `pending=0`; `last_report.documents` includes the uploaded doc; page cards show pending/done.

- [ ] **Step 4: Shadowing + regression**

Optionally upload a same-name file (e.g. `退货政策.md`) and confirm `/kb/api/files` shows only the upload source; restore by deleting the uploaded file and rebuilding. Confirm `query_faq("邮费是多少")` still returns shipping facts and `pytest -q` remains green.

- [ ] **Step 5: Append dev-notes + commit**

Append `## 阶段:验收与交付` to `dev-notes/ch03.md` with:
- user's range words (`上传 Markdown → 触发建库 → 看 pending/done`; `/kb` at 8000);
- demo commands and test counts;
- acceptance results (upload metadata injection, build status, shadowing, invalid file rejection);
- rejections/corrections and rework;
- spec/plan paths.

```bash
git add dev-notes/ch03.md
git commit -m "docs: record ch04 knowledge UI acceptance"
```

- [ ] **Step 6: Request code review, then finish**

Use `superpowers:requesting-code-review` (or `/code-review`) on the ch04 branch; append the review conclusion and any fixes to `dev-notes/ch03.md`. Then use `superpowers:finishing-a-development-branch` to merge/retain the branch, and append the finish decision to dev-notes.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-11-13-ecommerce-cs-ch03-knowledge-ui.md`. Please review the plan. Which execution approach would you prefer?

- **Subagent-driven** — a fresh subagent implements each task and a fresh reviewer checks it before the next one starts, then a whole-branch review at the end. Most thorough; costs a fresh context per task and per review.
- **Native** — I implement every task myself in this session, then one fresh reviewer on the most capable model checks the whole branch. Cheapest and fastest; no independent review until the end.

**For this plan I recommend Subagent-driven**, because the upload endpoint and shadowing semantics are security/data-consistency sensitive, and six tasks share the same router/service interfaces; a fresh reviewer per task catches drift early. Does the plan capture what you want, and which approach should we use?
