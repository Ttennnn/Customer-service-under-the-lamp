# 电商智能客服系统 · 第四章:知识录入页(`/kb`)— 设计文档

日期:2026-11-12
状态:待批准(待 builder 预审 / 待用户批准)

## Intent & Why

ch03 已经把知识库后端流水线做完了:Markdown 结构感知切分、MySQL + Milvus 双写、历史对话挖矿、`query_faq` 向量检索,验收通过。但知识录入目前只有 CLI:`build_knowledge`、`mine_qa`、`vectorize`。知识运营人员没有页面可用。

第四章在**现有 FastAPI 8000 端口**上提供一个知识录入页 `http://localhost:8000/kb`:

- 上传 Markdown;
- 一键触发建库;
- 看到 `pending` / `done` 状态与最近一次建库结果。

**第一版只做这三件事**:不做单条知识 CRUD、不做挖矿 staging 审核、不做删除、不新增表、不改你的 DDL、不改 Milvus schema。建库完全复用 ch03 的切分/向量化流水线。

## Goals

1. **`/kb` 页面**:FastAPI 直接返回一个自包含的 HTML/CSS/JS 页面(不经过 Vite/React 构建),同源调用 8000 的 API。
2. **上传 Markdown + 页面代填元数据**:多文件上传,只收 `.md`,限制大小与文件数,文件名做路径穿越清洗,落到 `knowledge/uploads/`;页面选择 `content_type`(policy/faq/manual)与可选 `category`,后端仅在文件**缺少 frontmatter** 时自动补;已有 frontmatter 的文件以文件内为准。同名上传覆盖(作为文档更新)。
3. **触发建库**:后台线程跑 `build_knowledge(reset=True)`——只重建文档域、保留 `mined_qa`;构建中再次触发返回 409。
4. **状态可观测**:`GET /kb/api/status` 返回 `pending/done/total`、是否正在构建、最近一次 `BuildReport`、最近错误、起止时间;页面轮询展示。
5. **文件可见**:`GET /kb/api/files` 列出 `knowledge/uploads/*.md` 的名称、大小、修改时间。
6. **全语料 + uploads 同名覆盖**:建库读取 `knowledge/` 根目录的 seed 文档 + `knowledge/uploads/` 的上传文档;**同名时 uploads 优先(shadow seed)**;页面文件列表展示全部**有效文档**并标注来源(`seed` / `upload`)。
7. **状态口径**:页面主计数只统计**文档域** `content_type != 'mined_qa' OR content_type IS NULL`;另外单独显示被保留的**挖矿知识(`mined_qa`)条数**,避免 `reset=True` 保留的挖矿行把进度条撑歪。
8. **验证**:后端 TDD;`/kb` 做页面冒烟;ch03 的 185 条测试与「邮费是多少」验收保持通过。

## Non-Goals(本章不做)

- 单条知识的新增/编辑/删除/chunk 浏览。
- `mine_qa` 的页面触发、`qa_extraction_staging` 的 kept/discarded 审核。
- 上传文件的删除、版本历史、回收站;seed 文档的删除/重命名。同名上传只 shadow seed,不物理删除 seed 文件。
- 登录、权限、多用户、审计日志。
- 新建表、改用户 DDL、改 Milvus 集合 schema。
- 前端 React/Vite 集成;`/kb` 是独立的原生静态页。
- 后台队列/Redis/Celery/多 worker;构建状态只存在单进程内存里。
- 增量文档更新算法;更新文档靠 `reset=True` 重建文档域。
- 构建进度百分比/流式日志;只做状态轮询与最近结果。
- 文件格式除 `.md` 外的支持。
- 生产级并发控制/限流/鉴权。

## Anti-Goals(技术上能跑通也算失败)

- `/kb` 页面必须能通过 `http://localhost:8000/kb` 直接打开,不能要求另起 5173 或构建前端。
- 上传不能造成路径穿越、覆盖 `knowledge/` 之外的任意文件。
- 构建失败不能把进程打挂;页面必须能看到错误。
- 构建中重复点击不能并发跑两个建库,导致 reset/向量化互相踩踏。
- 新页面不能复制一套切分/入库逻辑;必须调用 ch03 的 `build_knowledge`。
- 不能改动 `sql/ch03_knowledge.sql` 或用户已有的两张表结构。

## Constraints(定死,不可自行更改)

- 页面地址固定:`http://localhost:8000/kb`。
- 页面由 FastAPI 提供,端口 8000;不依赖 Vite/React。
- 建库复用 `app.jobs.build_knowledge.build_knowledge`;参数 `reset=True`(只清文档域、保留 `mined_qa`)。
- 上传目录固定 `knowledge/uploads/`;只收 `.md`。
- 上传页必须提供 `content_type` 选择;后端仅在文件缺 frontmatter 时补;已有 frontmatter 以文件内为准。
- 建库语料 = `knowledge/` 根目录 + `knowledge/uploads/`;同名 uploads 优先(shadow seed)。
- 不新增数据库表;`knowledge_chunks` / `qa_extraction_staging` 以用户 DDL 为准。
- 只做文档建库;不触发 `mine_qa`。
- 所有密钥/连接串继续只走 `.env`;页面不展示、不接收 API key。
- 后端 TDD;每个任务完成补 `dev-notes/ch04.md`。

## Context7 已核实的接口事实

- **FastAPI 文件上传**需要安装 `python-multipart`;端点参数用 `files: list[UploadFile] = File(...)`(或 `Annotated[list[UploadFile], File()]`)。`UploadFile` 提供 `filename`、`size`、`content_type`,读取用 `await file.read()`;同步 `def` 端点可用 `file.file`。
- **返回单个静态 HTML** 用 `fastapi.responses.FileResponse(path, media_type="text/html")`;不需要为单文件挂 `StaticFiles`。
- 当前环境:`fastapi 0.162.4`、`starlette 1.10.0`;`python-multipart` 尚未安装,需要加入 `requirements.txt`。

## 架构与数据流

```
浏览器 http://localhost:8000/kb
  GET  /kb                    → FileResponse(app/static/kb.html)
  GET  /kb/api/files          → knowledge/uploads/*.md 列表
  POST /kb/api/upload         → 校验 .md/大小/UTF-8/文件名 → 写入 knowledge/uploads/
  POST /kb/api/build          → 后台线程 build_knowledge(reset=True)
  GET  /kb/api/status         → MySQL pending/done/total + 内存构建状态

建库线程:
  get_settings().knowledge_docs_dir
  → load_documents(root *.md + uploads/*.md, uploads 同名 shadow root)
  → split_document
  → ingest_document_chunks(reset 文档域)
  → vectorize_pending
  → 记录 BuildReport / error
```

### 后端文件布局

```
app/api/__init__.py             # 新包
app/api/kb.py                   # /kb 与 /kb/api/* 路由、上传校验
app/services/kb_build.py        # 构建状态机 + 后台线程 + 状态计数
app/static/kb.html              # 自包含页面(HTML/CSS/JS)
app/rag/docs.py                 # effective_doc_paths: root *.md + uploads/*.md (uploads 同名 shadow root)
app/jobs/build_knowledge.py     # 目录校验用 effective_doc_paths
app/main.py                     # include_router(kb_router)
requirements.txt                # + python-multipart
```

## API 契约

### `GET /kb`

返回 `FileResponse("app/static/kb.html", media_type="text/html")`,200。页面标题包含「知识库录入」。

### `GET /kb/api/files`

200:

```json
{
  "files": [
    {"name": "运费与配送说明.md", "source": "seed", "size": 1234, "modified_at": "2026-11-12T20:00:00"},
    {"name": "运费补充.md", "source": "upload", "size": 456, "modified_at": "2026-11-12T20:05:00"}
  ]
}
```

按**有效语料**返回:`knowledge/` 根目录的 seed 文档 + `knowledge/uploads/` 的上传文档;同名时只返回 upload(seed 被 shadow,不重复列出)。`source` 为 `seed`/`upload`。不返回文件内容。

### `POST /kb/api/upload`

`multipart/form-data`:

- `files`: 多文件,字段名 `files`;
- `content_type`: 必填,`policy` / `faq` / `manual`;
- `category`: 可选,FAQ 文档建议填(如 `商品`)。

校验与元数据代填:

- 扩展名必须 `.md`(大小写不敏感);文件名取 `Path(filename).name`;拒绝空名、含 `/` 或 `\` 的原始名(路径穿越)。
- 单文件 ≤ **2 MiB**;单次 ≤ **20** 个文件;内容必须是合法 UTF-8(允许 BOM)。
- **文件已带 frontmatter**:以文件内为准,不注入;再用 `load_document` 校验(必须能解析、`content_type` 合法、有 H1),不合格拒绝。
- **文件没有 frontmatter**:后端按页面参数在正文前注入:
  ```markdown
  ---
  content_type: <页面选择>
  category: <页面填写,可省略>
  ---

  <原始正文>
  ```
  然后写入临时文件、用 `load_document` 校验,合格才 `os.replace` 到最终文件;不合格删除临时文件并拒绝。
- 临时文件放在 `knowledge/uploads/` 下、扩展名不是 `.md`(如 `.tmp-<uuid>`),不会被建库扫描到;最终写入用原子替换。
- 同名文件覆盖(视为文档更新);构建时 `reset=True` 会清掉旧 chunk。
- 构建中上传返回 **409**(避免构建快照漏文件)。

返回 200:

```json
{
  "saved": [{"name": "运费补充.md", "size": 456, "content_type": "policy"}],
  "rejected": [{"name": "a.txt", "reason": "only .md files are accepted"}]
}
```

全部被拒时仍返回 200 + `saved=[]` + `rejected`(页面展示原因)。

### `POST /kb/api/build`

- 无请求体。
- 若已有构建在跑:409 `{"detail": "a build is already running"}`。
- 否则启动后台 daemon 线程,立即 202:

```json
{"status": "running", "started_at": "2026-11-12T20:00:00"}
```

线程内:

```python
session = get_session_factory()()
try:
    report = build_knowledge(
        session,
        Path(settings.knowledge_docs_dir),
        get_embedder(),
        get_vector_store(),
        reset=True,
    )
    state.success(report)
except Exception as exc:
    logger.exception("kb build failed")
    state.fail(str(exc))
finally:
    session.close()
```

### `GET /kb/api/status`

200:

```json
{
  "pending": 0,
  "done": 20,
  "total": 20,
  "mined_total": 3,
  "mined_pending": 0,
  "mined_done": 3,
  "running": false,
  "started_at": "2026-11-12T20:00:00",
  "finished_at": "2026-11-12T20:00:12",
  "last_report": {"documents": 4, "chunks": 20, "inserted": 20, "reused": 0, "vectorized": 20},
  "last_error": null
}
```

- `pending/done/total` 只统计**文档域**:`content_type != 'mined_qa' OR content_type IS NULL`;`total = pending + done`。
- `mined_total/mined_pending/mined_done` 单独统计 `content_type = 'mined_qa'`,只展示、不参与进度条。
- 构建状态是单进程内存;进程重启后回到 `running=false`、`last_report=null`。
- `last_report.vectorized` 是 `vectorize_pending` 的实际处理数;如果当时还有 pending 的挖矿行,它可能包含这些行,页面文案标注「本次向量化」。进度条只按文档域 `done/total` 计算。

## 页面行为(`app/static/kb.html`)

- 标题「知识库录入」,说明:上传 Markdown 后点击「开始建库」;建库会重建文档域,保留挖矿知识。
- 上传区:
  - `<input type="file" accept=".md" multiple>` + 「上传」按钮;
  - `content_type` 下拉(policy/faq/manual,必选)+ `category` 输入(可选,FAQ 建议填);
  - 说明:文件已有 frontmatter 时以文件内为准;没有时由页面选择的类型自动补;
  - 上传中禁用;显示 saved/rejected 结果与拒绝原因;
  - 构建中禁用上传(与后端 409 一致)。
- 文件列表:名称、大小、修改时间、来源徽标(`seed` / `upload`);同名上传会 shadow 同名 seed,列表只显示有效文档;上传成功后自动刷新。
- 建库区:
  - 「开始建库」按钮;构建中禁用并显示「构建中…」;
  - 状态卡:文档域 `pending` / `done` / `total`,进度条 `done / total`;
  - 单独一行「挖矿知识保留:N 条(pending X / done Y)」,不参与进度;
  - 最近一次 `BuildReport`(documents/chunks/inserted/reused/vectorized)、`last_error`;
  - 起止时间。
- 轮询:页面加载时拉一次;构建中或 `pending > 0` 时每 2 秒轮询 `status`;空闲时停止轮询(或 10 秒慢轮询)。
- 所有动态文本用 `textContent`,不拼 `innerHTML`;文件名来自服务端清单,仍需转义。
- 无外部 CDN/字体/JS 依赖;内联 CSS,深色/浅色跟随系统即可(简单配色)。

## 上传校验与安全

- 路径穿越:只取 `Path(filename).name`;原始名含 `/` 或 `\` 直接拒绝;保存路径固定为 `uploads_dir / name`。
- 扩展名:`.md` 后缀;`content_type` 不作为唯一依据。
- 大小:读取时累计字节数,超过 2 MiB 立即拒绝(不写盘)。
- 编码:尝试 `bytes.decode("utf-8-sig")`;失败拒绝。
- 元数据:无 frontmatter 时按页面参数注入;写临时文件后用 `load_document` 校验(H1、`content_type`);已有 frontmatter 以文件为准;不合格拒绝。
- 并发:构建中上传/建库均 409;上传写文件用唯一临时名再 `os.replace` 原子替换,避免半文件;临时文件扩展名不是 `.md`,不会被建库扫描。
- 不做认证:本地 demo;在页面与文档中明确「不要暴露到公网」。

## 建库状态机

`app/services/kb_build.py` 提供:

```python
@dataclass
class BuildState:
    running: bool = False
    started_at: datetime | None = None
    finished_at: datetime | None = None
    last_report: BuildReport | None = None
    last_error: str | None = None

def status(session) -> dict          # 计数 + 状态快照
def start_build() -> datetime | None # None = 已有构建在跑;否则返回 started_at
def reset_state_for_tests() -> None  # 测试用
```

- 模块级 `threading.Lock` 保护 `running` 的检查与置位;`start_build` 返回 None 时路由返回 409。
- 线程 daemon=True;异常写 `last_error` 并 `logger.exception`,不向外抛。
- `status(session)` 的 DB 查询分两域:文档域(`content_type != 'mined_qa' OR content_type IS NULL`)按 `vectorize_status` 分组得到 `pending/done/total`;`mined_qa` 单独得到 `mined_pending/mined_done/mined_total`。
- 状态不持久化;进程重启后空状态。

## 测试策略

### 后端 TDD

- `tests/test_kb_api.py`:
  - `GET /kb` 200,HTML 含「知识库录入」;
  - 无 frontmatter 的纯 Markdown + `content_type=policy` 上传后,落盘文件带注入的 frontmatter,`load_document` 能解析;
  - 已有合法 frontmatter 的文件不被页面参数覆盖;已有非法 frontmatter / 缺 H1 被拒;
  - `.txt`、超大、非法 UTF-8、路径穿越文件名被拒;
  - 同名上传覆盖,且 `GET /kb/api/files` 只返回 upload,seed 被 shadow;
  - `GET /kb/api/files` 返回 `source` 为 `seed`/`upload`,按名字排序;
  - `GET /kb/api/status` 用 SQLite 写入文档域 pending/done 与 mined_qa 行,断言文档域计数与 mined 计数分开;
  - `POST /kb/api/build` 用 monkeypatch 的 fake `build_knowledge` + fake embedder/store:202 → 状态 running → 完成后 last_report;构建中再次 POST 409;异常时 last_error 可见。
  - `python-multipart` 安装后 TestClient 才能发 multipart。
- `tests/test_docs.py`:新增「root seed + uploads 同名 → 只返回 upload」「不同名 → 两者都返回」测试;现有 4 份样本文档仍全部加载。
- `tests/test_build_knowledge.py`:目录校验对「只有 uploads 子目录、根目录无 md」通过;同名 shadow 后建库只看到 upload 版本。
- 全量回归:ch03 的 185 条测试必须继续通过。

### 页面冒烟

- 启动 uvicorn 8000,`curl -fsS http://localhost:8000/kb` 返回 200 且包含页面标题/上传/建库元素。
- 用 API 上传一份临时 `.md`、触发 build、轮询 status 到 pending=0;完成后删除临时文件(或保留在 uploads 供演示)。
- 浏览器手工检查:文件选择、上传结果、建库按钮、状态卡、错误展示。

### 验收(来自本章范围)

1. 打开 `http://localhost:8000/kb`,选择一份**没有 frontmatter** 的纯 Markdown + `content_type=policy`,上传成功并出现在文件列表;落盘文件带注入的 frontmatter。
2. 上传一份与 seed 同名的文档(如 `退货政策.md`),文件列表只显示 upload(seed 被 shadow);建库后检索使用 upload 版本。
3. 点击「开始建库」,状态从 running → pending → done;页面显示文档域 pending/done 与最近一次 BuildReport。
4. 上传 `.txt`/超大/缺 H1 的文件被拒并显示原因。
5. 构建中重复触发返回 409/按钮禁用,不会并发建库。
6. `query_faq("邮费是多少")` 与 ch03 全量测试不受影响。

## 我替用户做的决定(标记为 my call,用户可在审阅时推翻)

1. **页面形态**:FastAPI 直接返回自包含 `app/static/kb.html`(原生 HTML/CSS/JS),不集成 Vite/React;地址固定 `/kb`。
2. **构建方式**:后台 daemon 线程 + 单进程内存状态 + 2 秒轮询;不引入队列/worker/新表。
3. **构建语义**:`build_knowledge(reset=True)`,只重建文档域,保留 `mined_qa`;上传同名文件覆盖并靠重建消除旧 chunk。
4. **语料范围**:建库读 `knowledge/` 根目录 + `knowledge/uploads/`;**同名 uploads 优先(shadow seed)**;`load_documents` 与 `build_knowledge` 目录校验按这个有效语料实现;页面文件列表也按有效语料展示。
5. **上传限制与元数据**:`.md` only、单文件 2 MiB、单次 20 个、UTF-8(允许 BOM);页面选 `content_type`(policy/faq/manual)与可选 `category`,后端仅在文件缺 frontmatter 时注入,已有 frontmatter 以文件为准;临时文件 + `load_document` 校验 + 原子替换;构建中上传/建库 409。
6. **无认证**:本地 demo;页面与交付说明明确不要暴露公网。
7. **状态口径**:文档域 `pending/done/total` 主计数 + 单独的 `mined_qa` 条数;不做 chunk 级进度、不做历史构建记录;`last_report.vectorized` 标注为「本次向量化」。
8. **依赖**:新增 `python-multipart`;不新增其他库。
9. **同名覆盖语义**:同名上传 shadow seed,不物理删除 seed 文件;页面只列有效文档并标 `source`。

## 留给 builder 的

- 路由/服务具体函数签名、FastAPI `UploadFile` 的异步读取细节、`FileResponse` 路径解析、原子写入实现、线程状态机的测试同步方式、页面 CSS/JS 具体结构、TestClient 与 SQLite 夹具组织、冒烟脚本。
- 上述 my call 的具体落地,除非用户推翻。
