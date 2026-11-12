# 电商智能客服 · 第二章(工具调用)实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给 ch01 的纯对话闭环装上「查数据 + 建工单」的手:MySQL 持久化(四张表)+ 5 个 LangChain `@tool` + 单轮 Function Calling 接进现有 `/chat` SSE 入口,聊天页气泡显示工具徽章。

**Architecture:** FastAPI 暴露 `POST /chat`(SSE)。会话/消息以 MySQL 为唯一真相源(取代内存 `SessionStore`)。模型统一走 LangChain `ChatOpenAI`(OpenAI 协议直连 DeepSeek)。`/chat` 每轮:落库用户消息 → 读历史组上下文 → `bind_tools` 让模型定工具 → 单轮执行 → `ToolMessage` 回灌 → 最终答案仍逐 token 流式。工具统一注册 + 统一执行入口(参数校验 / 超时重试 / 错误兜底)。

**Tech Stack:** Python 3.11+ · FastAPI · SQLAlchemy 2.0(`pymysql` 驱动)· LangChain(`langchain` + `langchain-openai` + `@tool`)· MySQL 8.0(Docker)· DeepSeek(OpenAI 协议)。

**Spec:** `docs/superpowers/specs/2026-11-12-ecommerce-cs-ch02-tools-design.md`

## Global Constraints

- 技术栈定死:Python + FastAPI + SQLAlchemy + MySQL(Docker)+ LangChain `@tool`;不引入 LangGraph / Alembic / 向量库。
- 模型接入:沿用 ch01 的 OpenAI 协议直连 DeepSeek,`OPENAI_BASE_URL`/`OPENAI_API_KEY`/`MODEL_NAME`/`DATABASE_URL` 全在 `.env`。
- **四张表结构以用户 DDL 为准**(见 spec):`conversations.status` ENUM(`进行中`/`已转人工`/`已结束`);`messages.role` ENUM(`user`/`assistant`/`tool`);`tickets.ticket_type` ENUM(`售后`/`投诉`/`咨询`);`tickets.status` ENUM(`待处理`/`已处理`);`tickets.ticket_no` 为业务主键。
- 五个工具名称与用途固定:`query_order`/`query_product`/`query_logistics`/`query_faq`/`create_ticket`。
- **单轮调用**:模型调一次工具就收敛,不回环;最终回答仍 SSE 逐 token 流式,工具执行段先推状态帧。
- 客户端继续发 `session_id`,后端映射为 `conversations.user_id`(spec 已定)。
- 每个涉及库 API 的任务,实现前先用 Context7 查最新文档(`ChatOpenAI.bind_tools` 对 DeepSeek 的行为、`@tool` 参数 Schema 生成、SQLAlchemy 2.0 `Enum`/`JSON`/`BigInteger` 映射)。

## Review Focus

以下是 spec 未明说、但真实用户会撞上、且无任务自带测试最容易翻车的输入/失败模式,按最可能踩中排序:

1. **`messages.content` 为 NULL**(assistant 纯工具调用行)→ 上下文重建与 token 裁剪不能对 NULL 调 `len()` 崩掉,按 0 计。→ Task 8 测试。
2. **工具参数非法/缺失**(模型传错字段)→ 执行器返回错误串回灌,不抛异常挂流。→ Task 7 测试。
3. **工具超时或抛异常** → 执行器超时重试后仍失败则返回错误串,SSE 不挂起。→ Task 7 测试。
4. **模型不调工具直接答** → 无工具路径仍逐 token 流式 + 落库,不崩。→ Task 10 测试。
5. **同一 user 在「已转人工」后再发消息** → 找不到「进行中」会话则新建,不报主键/外键错。→ Task 8 测试(`get_or_create_conversation`)。
6. **tool 消息必须紧跟 assistant 的 tool_calls**(OpenAI 协议)→ 重建与裁剪都按原子块,不拆散 tool 对。→ Task 8 测试。

---

## File Structure

```
app/
  config.py                     # 扩:新增 database_url(读 .env)
  main.py                       # 扩:/chat 走工具化流、注入 get_db + ALL_TOOLS
  schemas.py                    # 不变
  prompts/customer_service.py   # 扩:system prompt 加工具使用约束
  db/
    __init__.py
    base.py                     # Base(DeclarativeBase)
    session.py                  # get_engine / get_session_factory / get_db(懒加载)
    seed.py                     # init_db(create_all)+ seed_faq + CLI main
  models/
    __init__.py                 # 导入四模型,使 Base.metadata 完整
    conversation.py             # Conversation
    message.py                  # Message
    faq.py                      # Faq
    ticket.py                   # Ticket
  tools/
    __init__.py
    context.py                  # current_conversation_id: ContextVar[int | None]
    registry.py                 # ALL_TOOLS / TOOLS_BY_NAME / TOOL_LABELS
    business.py                 # query_order / query_product / query_logistics(mock)
    faq.py                      # query_faq(SQL LIKE)
    ticket.py                   # create_ticket(写 tickets + 置会话已转人工)
    executor.py                 # execute_tool(校验/超时/重试/兜底)
  services/
    history.py                  # get_or_create_conversation / append_message / load_history
    context.py                  # 改:estimate_tokens(None→0)、build_messages 原子裁剪
    chat.py                     # 改:单轮工具编排 stream_chat
docker-compose.yml              # mysql:8.0
tests/
  conftest.py                   # SQLite 内存 fixture(StaticPool)+ monkeypatch get_session_factory
  test_db_config.py
  test_models.py
  test_seed.py
  test_tools_business.py
  test_tools_faq.py
  test_tools_ticket.py
  test_executor.py
  test_history.py
  test_chat.py                  # 改:工具化流编排(假 LLM + SQLite)
  test_api.py                   # 改:注入假 LLM + SQLite session
```

---

## Task 1: 数据库配置 + Docker Compose + 分层骨架

**Files:**
- Modify: `app/config.py`
- Create: `docker-compose.yml`, `app/db/__init__.py`, `app/db/base.py`, `app/db/session.py`
- Test: `tests/test_db_config.py`

**Interfaces:**
- Produces:
  - `app.config.Settings.database_url: str`(新增字段,读 `DATABASE_URL`,默认 `mysql+pymysql://root:root@localhost:3306/cs_customer?charset=utf8mb4`,可被 `DATABASE_URL` 覆盖;给默认值以免破坏 ch01 的 `Settings()` 测试)
  - `app.db.base.Base(DeclarativeBase)`
  - `app.db.session.get_engine() -> Engine`(懒加载,`create_engine(get_settings().database_url, pool_pre_ping=True)`)
  - `app.db.session.get_session_factory() -> sessionmaker`(懒加载,`sessionmaker(bind=get_engine(), autoflush=False, expire_on_commit=False)`)
  - `app.db.session.get_db() -> Iterator[Session]`(FastAPI 依赖,`yield` 一个 Session 并在 finally close)

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_db_config.py
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker
from app.config import Settings
from app.db.base import Base
from app.db import session as db_session

def test_database_url_from_env(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "mysql+pymysql://u:p@localhost:3306/x")
    assert Settings().database_url == "mysql+pymysql://u:p@localhost:3306/x"

def test_engine_and_factory_lazy(monkeypatch):
    monkeypatch.setattr(db_session, "get_settings", lambda: Settings(
        openai_api_key="sk-test", database_url="sqlite://"
    ))
    db_session.get_engine.cache_clear() if hasattr(db_session.get_engine, "cache_clear") else None
    assert isinstance(db_session.get_engine(), Engine)
    assert isinstance(db_session.get_session_factory(), sessionmaker)
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_db_config.py -v`
Expected: FAIL(`database_url` 未定义 / `app.db` 模块缺失)。

- [ ] **Step 3: 实现配置与骨架**

`app/config.py` 加 `database_url: str` 字段。`app/db/base.py` 定义 `class Base(DeclarativeBase): pass`。`app/db/session.py` 用 `functools.lru_cache` 或模块级缓存实现 `get_engine`/`get_session_factory` 懒加载;`get_db` 为生成器依赖。`docker-compose.yml`:mysql:8.0、`MYSQL_ROOT_PASSWORD`/`MYSQL_DATABASE=cs_customer`、端口 `3306:3306`、named volume、字符集 utf8mb4。

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/test_db_config.py -v`
Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add app/config.py app/db/ docker-compose.yml tests/test_db_config.py .env.example
git commit -m "feat: add database config, SQLAlchemy session skeleton, and MySQL compose"
```

---

## Task 2: 四张表 SQLAlchemy 模型

**Files:**
- Create: `app/models/__init__.py`, `app/models/conversation.py`, `app/models/message.py`, `app/models/faq.py`, `app/models/ticket.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: `Base`(Task 1)
- Produces(SQLAlchemy 2.0 `Mapped`/`mapped_column`,tablename 用复数对齐 DDL):
  - `Conversation`(表 `conversations`):`id`(BigInteger PK auto)、`user_id`(String(64) 非空 index)、`status`(Enum `进行中`/`已转人工`/`已结束`,默认 `进行中`)、`created_at`(DateTime `server_default=func.now()`)、`updated_at`(DateTime `server_default=func.now()` + `onupdate=func.now()`)。
  - `Message`(表 `messages`):`id`(BigInteger PK auto)、`conversation_id`(BigInteger `ForeignKey("conversations.id")` index)、`role`(Enum `user`/`assistant`/`tool`)、`content`(Text 可空)、`tool_calls`(JSON 可空)、`tool_call_id`(String(64) 可空)、`created_at`。
  - `Faq`(表 `faq`):`id`(BigInteger PK auto)、`question`(String(512))、`answer`(Text)、`category`(String(64) index)、`created_at`、`updated_at`。
  - `Ticket`(表 `tickets`):`ticket_no`(String(32) PK)、`conversation_id`(BigInteger `ForeignKey("conversations.id")` index)、`description`(Text)、`ticket_type`(Enum `售后`/`投诉`/`咨询`)、`status`(Enum `待处理`/`已处理`,默认 `待处理`)、`created_at`。

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_models.py
from sqlalchemy import JSON
from app.models import Conversation, Message, Faq, Ticket

def _cols(model):
    return {c.name: c for c in model.__table__.columns}

def test_conversation_status_enum():
    assert _cols(Conversation)["status"].type.enums == ["进行中", "已转人工", "已结束"]

def test_message_role_and_tool_calls():
    c = _cols(Message)
    assert c["role"].type.enums == ["user", "assistant", "tool"]
    assert isinstance(c["tool_calls"].type, JSON)
    assert c["content"].nullable is True

def test_message_fk_to_conversation():
    assert any(str(fk.target_fullname) == "conversations.id"
               for fk in Message.__table__.foreign_keys)

def test_ticket_enums_and_pk():
    c = _cols(Ticket)
    assert c["ticket_no"].primary_key
    assert c["ticket_type"].type.enums == ["售后", "投诉", "咨询"]
    assert c["status"].type.enums == ["待处理", "已处理"]

def test_faq_fields():
    c = _cols(Faq)
    assert c["question"].type.length == 512
    assert c["category"].index is True
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_models.py -v`
Expected: FAIL(`app.models` 模块缺失)。

- [ ] **Step 3: 实现四个模型**

按 Interfaces 定义,统一从 `app.db.base.Base` 继承;`app/models/__init__.py` 导入四模型使 `Base.metadata` 完整。Enum 用 SQLAlchemy `Enum(...)`(MySQL 上生成原生 ENUM,SQLite 测试自动降级为 VARCHAR)。

- [ ] **Step 4: 运行确认通过** → `pytest tests/test_models.py -v` → PASS

- [ ] **Step 5: Commit**

```bash
git add app/models/ tests/test_models.py
git commit -m "feat: add faq/conversation/message/ticket ORM models"
```

---

## Task 3: 建表 + FAQ 种子数据(数据类 · 标注样例验证)

**Files:**
- Create: `app/db/seed.py`
- Test: `tests/test_seed.py`

**Interfaces:**
- Produces:
  - `app.db.seed.FAQ_SEED: list[dict]`(含 `question`/`answer`/`category` 三键)
  - `app.db.seed.seed_faq(session: Session) -> int`(批量插入,返回插入行数)
  - `app.db.seed.init_db(engine: Engine) -> None`(`Base.metadata.create_all(engine)`)
  - `app.db.seed.main() -> None`(CLI:`get_engine()` → `init_db` → `get_session_factory()()` → `seed_faq` → commit)

- [ ] **Step 1: 写标注期望的验证(本任务是数据类,不走红绿,直接对标注期望断言)**

FAQ 种子必须:**含**「退货政策」(命中验收 2 的 `LIKE %退货%`),**含**「物流时效」「换货」「退款」等条目,**全文任何 question/answer 都不含「邮费」**(保证验收 3 漏召回)。

```python
# tests/test_seed.py
from app.db.seed import FAQ_SEED, seed_faq

def test_faq_seed_has_return_policy_and_no_shipping():
    qs = " ".join(r["question"] for r in FAQ_SEED)
    answers = " ".join(r["answer"] for r in FAQ_SEED)
    assert "退货" in qs or "退货" in answers
    assert "邮费" not in qs and "邮费" not in answers

def test_seed_faq_inserts_rows(session):
    n = seed_faq(session)
    assert n == len(FAQ_SEED)
    session.commit()
```

- [ ] **Step 2: 运行验证**

Run: `pytest tests/test_seed.py -v`
Expected: 第一次 FAIL(`FAQ_SEED`/`seed_faq` 未定义,或 `session` fixture 未提供)。

- [ ] **Step 3: 实现 `app/db/seed.py`**

`FAQ_SEED` 提供 4~6 条真实问答(退货政策、换货、退款到账、物流时效、发货时间等),刻意不含「邮费」。`seed_faq` 用 `session.add_all([Faq(**r) for r in FAQ_SEED])` 后 `flush` 返回条数。`tests/conftest.py` 提供 `session` fixture:SQLite 文件库(`create_engine(f"sqlite:///{tmp_path}/test.db", connect_args={"check_same_thread": False})`,用文件库避免内存库 `StaticPool` 单连接在工具子线程/跨会话时的共享连接冲突)+ `Base.metadata.create_all` + 一个 `Session`,并把 `app.db.session.get_session_factory` monkeypatch 成该引擎的 sessionmaker(供工具在其自己的会话/子线程里访问同一库)。

- [ ] **Step 4: 运行确认通过** → `pytest tests/test_seed.py -v` → PASS

- [ ] **Step 5: Commit**

```bash
git add app/db/seed.py tests/test_seed.py tests/conftest.py
git commit -m "feat: add table init and FAQ seed data"
```

---

## Task 4: 工具注册表 + 三个 mock 工具

**Files:**
- Create: `app/tools/__init__.py`, `app/tools/registry.py`, `app/tools/business.py`
- Test: `tests/test_tools_business.py`

**Interfaces:**
- Produces:
  - `app.tools.business.query_order(order_id: str) -> str`(返回 JSON 串:`order_id` 回显 + `items`/`amount`/`status`)
  - `app.tools.business.query_product(product_name: str) -> str`(返回 JSON 串:`product_name` 回显 + `price`/`stock`/`specs`)
  - `app.tools.business.query_logistics(order_id: str) -> str`(返回 JSON 串:`order_id` 回显 + `logistics_no`/`carrier`/`current_node`/`traces`)
  - `app.tools.registry.ALL_TOOLS: list[BaseTool]`、`TOOLS_BY_NAME: dict[str, BaseTool]`、`TOOL_LABELS: dict[str, str]`(`query_order`查订单、`query_product`查商品、`query_logistics`查物流、`query_faq`查常见问题、`create_ticket`创建工单)
  - 三个函数都用 `@tool` 装饰,docstring 写明用途与参数含义。

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_tools_business.py
import json
from app.tools.business import query_order, query_product, query_logistics

def test_query_order_echoes_id():
    data = json.loads(query_order.invoke({"order_id": "1001"}).content)
    assert data["order_id"] == "1001"
    assert set(data) >= {"order_id", "items", "amount", "status"}

def test_query_logistics_has_node():
    data = json.loads(query_logistics.invoke({"order_id": "1001"}).content)
    assert data["order_id"] == "1001"
    assert set(data) >= {"logistics_no", "carrier", "current_node", "traces"}

def test_registry_has_three_business_tools_and_five_labels():
    from app.tools.registry import ALL_TOOLS, TOOLS_BY_NAME, TOOL_LABELS
    assert [t.name for t in ALL_TOOLS] == ["query_order", "query_product", "query_logistics"]
    assert set(TOOLS_BY_NAME) == {t.name for t in ALL_TOOLS}
    assert set(TOOL_LABELS) == {"query_order", "query_product", "query_logistics", "query_faq", "create_ticket"}
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_tools_business.py -v`
Expected: FAIL(模块缺失)。

- [ ] **Step 3: 实现工具与注册表**

实现前 Context7 查 `langchain.tools` 的 `@tool` 与 `.invoke(dict)` 返回(`ToolMessage`,取 `.content`)。mock 工具内部用 `random` 生成价格/库存/物流节点等字段,`order_id`/`product_name` 原样回显;`json.dumps(..., ensure_ascii=False)`。`registry.py` 汇总五个工具(name 固定为上述五个)。注意 Task 5/6 的工具尚未实现时,先让 `registry.py` 只 import 已存在的 business 三工具并留出占位,等 Task 5/6 落地后补齐 import(或本任务先以 `# noqa` 容忍缺失)。**为让测试可跑**,本任务 `registry.py` 先只导出 business 三工具 + 预留 `TOOL_LABELS` 五键;Task 5/6 完成后各自把新工具挂进 `ALL_TOOLS` 并补测试。

- [ ] **Step 4: 运行确认通过** → `pytest tests/test_tools_business.py -v` → PASS

- [ ] **Step 5: Commit** → `git add app/tools/ tests/test_tools_business.py && git commit -m "feat: add tool registry and mock order/product/logistics tools"`

---

## Task 5: query_faq 工具(SQL LIKE 查 faq 表)

**Files:**
- Create: `app/tools/faq.py`
- Test: `tests/test_tools_faq.py`

**Interfaces:**
- Consumes: `Faq`(Task 2)、`get_session_factory`(Task 1)、`session` fixture(Task 3)
- Produces:
  - `app.tools.faq.query_faq(keyword: str) -> str`(`@tool`):用 `session.query(Faq).filter(or_(Faq.question.like(f"%{keyword}%"), Faq.answer.like(f"%{keyword}%")))` 查;命中返回格式化问答文本;未命中返回「未找到相关问题」。

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_tools_faq.py
from app.db.seed import seed_faq
from app.models import Faq
from app.tools.faq import query_faq

def test_query_faq_hits_return_policy(session):
    seed_faq(session); session.commit()
    out = query_faq.invoke({"keyword": "退货"}).content
    assert "退货" in out

def test_query_faq_misses_shipping(session):
    seed_faq(session); session.commit()
    out = query_faq.invoke({"keyword": "邮费"}).content
    assert "未找到" in out
```

- [ ] **Step 2: 运行确认失败** → `pytest tests/test_tools_faq.py -v` → FAIL

- [ ] **Step 3: 实现 `app/tools/faq.py`**

`query_faq` 内部 `s = get_session_factory()()` 开新会话(工具在子线程执行,不能复用外部会话),查询后 `s.close()`。把 `query_faq` 挂进 `registry.ALL_TOOLS`/`TOOLS_BY_NAME`。

- [ ] **Step 4: 运行确认通过** → `pytest tests/test_tools_faq.py -v` → PASS

- [ ] **Step 5: Commit** → `git add app/tools/faq.py app/tools/registry.py tests/test_tools_faq.py && git commit -m "feat: add query_faq tool (SQL LIKE over faq table)"`

---

## Task 6: create_ticket 工具 + 会话上下文 contextvar

**Files:**
- Create: `app/tools/context.py`, `app/tools/ticket.py`
- Test: `tests/test_tools_ticket.py`

**Interfaces:**
- Consumes: `Ticket`/`Conversation`(Task 2)、`get_session_factory`(Task 1)、`session` fixture
- Produces:
  - `app.tools.context.current_conversation_id: ContextVar[int | None]`(默认 None)
  - `app.tools.ticket.create_ticket(description: str, ticket_type: Literal["售后","投诉","咨询"]) -> str`(`@tool`):生成 `ticket_no`(`T`+`YYYYMMDD`+当日三位递增序号),写 `Ticket`(status 默认 `待处理`,conversation_id 从 contextvar 取),把关联 `Conversation.status` 置为 `已转人工`,返回工单号;contextvar 为 None 时返回错误串。

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_tools_ticket.py
import re
from app.db.seed import seed_faq
from app.models import Conversation, Ticket
from app.tools.context import current_conversation_id
from app.tools.ticket import create_ticket

def _make_conversation(session):
    c = Conversation(user_id="u1", status="进行中")
    session.add(c); session.commit(); session.refresh(c)
    return c.id

def test_create_ticket_writes_and_archives(session):
    cid = _make_conversation(session)
    current_conversation_id.set(cid)
    out = create_ticket.invoke({"description": "屏幕碎了", "ticket_type": "售后"}).content
    session.expire_all()
    assert re.match(r"T\d{8}\d{3}", out)
    t = session.query(Ticket).one()
    assert t.conversation_id == cid and t.status == "待处理"
    assert session.query(Conversation).get(cid).status == "已转人工"

def test_create_ticket_without_context_errors(session):
    current_conversation_id.set(None)
    out = create_ticket.invoke({"description": "x", "ticket_type": "咨询"}).content
    assert "失败" in out or "错误" in out

def test_registry_has_five_tools():
    from app.tools.registry import ALL_TOOLS
    assert [t.name for t in ALL_TOOLS] == ["query_order", "query_product", "query_logistics", "query_faq", "create_ticket"]
```

- [ ] **Step 2: 运行确认失败** → `pytest tests/test_tools_ticket.py -v` → FAIL

- [ ] **Step 3: 实现 `context.py` 与 `ticket.py`**

`ticket_type` 用 `typing.Literal` 约束;`@tool` 由类型注解 + docstring 生成 Schema(Literal → enum)。`ticket_no` 生成:统计当日 `Ticket` 数 + 1,拼 `T{date}{seq:03d}`,加锁或捕获主键冲突重试防撞。`create_ticket` 内部自开会话(`get_session_factory()()`),提交后关闭;挂进 `registry.ALL_TOOLS`。

- [ ] **Step 4: 运行确认通过** → `pytest tests/test_tools_ticket.py -v` → PASS

- [ ] **Step 5: Commit** → `git add app/tools/context.py app/tools/ticket.py app/tools/registry.py tests/test_tools_ticket.py && git commit -m "feat: add create_ticket tool with conversation contextvar"`

---

## Task 7: 工具执行器(校验 / 超时 / 重试 / 兜底)

**Files:**
- Create: `app/tools/executor.py`
- Test: `tests/test_executor.py`

**Interfaces:**
- Consumes: 任意 `BaseTool`(含 Task 4/5/6 工具)
- Produces:
  - `app.tools.executor.execute_tool(tool: BaseTool, args: dict, timeout_s: float = 5, retries: int = 1) -> str`:先用 `tool.args_schema` 校验参数(非法返回错误串)→ 在 daemon 子线程执行 `tool.invoke(args)` 并取 `.content` → 超时/异常重试 `retries` 次 → 仍失败返回「工具执行失败」串。任何路径都返回 `str`,不抛异常。

- [ ] **Step 1: 写失败的测试(用假工具)**

```python
# tests/test_executor.py
import time
from langchain.tools import tool
from app.tools.executor import execute_tool

@tool
def fast_tool(x: int) -> str:
    """double the input"""
    return str(x * 2)

@tool
def slow_tool(x: int) -> str:
    """sleep forever"""
    time.sleep(10)
    return str(x)

@tool
def boom_tool(x: int) -> str:
    """always raise"""
    raise RuntimeError("boom")

def test_execute_success():
    assert execute_tool(fast_tool, {"x": 3}, timeout_s=5) == "6"

def test_execute_bad_args_returns_error():
    out = execute_tool(fast_tool, {"x": "not-an-int"}, timeout_s=5)
    assert "失败" in out or "错误" in out

def test_execute_timeout_returns_error():
    assert "失败" in execute_tool(slow_tool, {"x": 1}, timeout_s=0.2, retries=1)

def test_execute_exception_returns_error():
    assert "失败" in execute_tool(boom_tool, {"x": 1}, timeout_s=5)
```

- [ ] **Step 2: 运行确认失败** → `pytest tests/test_executor.py -v` → FAIL

- [ ] **Step 3: 实现 `execute_tool`**

参数校验用 `tool.args_schema.model_validate(args)`,捕获 `ValidationError`。执行用 daemon 线程(`threading.Thread(daemon=True)` + `queue.Queue`,`queue.get(timeout=timeout_s)`),避免超时后非 daemon 线程池阻塞进程退出;捕获 `queue.Empty`(超时)/`Exception`;重试逻辑包在外层循环。实现前 Context7 确认 `tool.invoke(dict)` 的返回类型(`ToolMessage`,取 `.content`)。

- [ ] **Step 4: 运行确认通过** → `pytest tests/test_executor.py -v` → PASS

- [ ] **Step 5: Commit** → `git add app/tools/executor.py tests/test_executor.py && git commit -m "feat: add tool executor with validation, timeout, and retry"`

---

## Task 8: 上下文重建 + 原子裁剪 + 会话/消息读写

**Files:**
- Create: `app/services/history.py`
- Modify: `app/services/context.py`
- Test: `tests/test_history.py`, `tests/test_context.py`(扩)

**Interfaces:**
- Consumes: `Message`/`Conversation`(Task 2)、`BaseMessage`(langchain_core)
- Produces:
  - `app.services.history.get_or_create_conversation(session, user_id: str) -> Conversation`(找最近一条 `status="进行中"` 且 `user_id` 匹配的会话,无则新建并 commit)
  - `app.services.history.append_message(session, conversation_id: int, role: str, content: str | None = None, tool_calls: list | None = None, tool_call_id: str | None = None) -> Message`
  - `app.services.history.load_history(session, conversation_id: int) -> list[BaseMessage]`(按 id 升序重建:`user`→HumanMessage、`assistant`(无 tool_calls)→AIMessage(content)、`assistant`(有 tool_calls)→AIMessage(content="", tool_calls=json.loads(...))、`tool`→ToolMessage(content, tool_call_id))
  - `app.services.context.estimate_tokens(text: str | None) -> int`(`None` 返回 0)
  - `app.services.context.build_messages(system, history: list[BaseMessage], current: HumanMessage, max_tokens: int) -> list[BaseMessage]`(原子裁剪:assistant-tool_calls 与其后的 tool 消息视为一块,整块保留/丢弃)

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_history.py
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from app.models import Conversation
from app.services.history import get_or_create_conversation, append_message, load_history

def test_get_or_create_reuses_active(session):
    c1 = get_or_create_conversation(session, "u1")
    c2 = get_or_create_conversation(session, "u1")
    assert c1.id == c2.id and c1.status == "进行中"

def test_get_or_create_new_after_archived(session):
    c1 = get_or_create_conversation(session, "u1")
    c1.status = "已转人工"; session.commit()
    c2 = get_or_create_conversation(session, "u1")
    assert c2.id != c1.id

def test_load_history_rebuilds_tool_round(session):
    cid = get_or_create_conversation(session, "u1").id
    append_message(session, cid, "user", "物流到哪")
    append_message(session, cid, "assistant", tool_calls=[{"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c1"}])
    append_message(session, cid, "tool", "物流结果", tool_call_id="c1")
    append_message(session, cid, "assistant", "已为您查到")
    session.commit()
    msgs = load_history(session, cid)
    assert isinstance(msgs[0], HumanMessage)
    assert isinstance(msgs[1], AIMessage) and msgs[1].tool_calls[0]["name"] == "query_logistics"
    assert isinstance(msgs[2], ToolMessage) and msgs[2].tool_call_id == "c1"
    assert isinstance(msgs[3], AIMessage) and msgs[3].content == "已为您查到"
```

```python
# tests/test_context.py(扩)
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from app.services.context import build_messages, estimate_tokens

def test_estimate_tokens_none_is_zero():
    assert estimate_tokens(None) == 0

def test_build_messages_keeps_tool_round_atomic():
    system = SystemMessage(content="S")
    hist = [
        HumanMessage(content="old"),
        AIMessage(content="", tool_calls=[{"name": "query_order", "args": {}, "id": "c1"}]),
        ToolMessage(content="X" * 20, tool_call_id="c1"),
        AIMessage(content="answer"),
    ]
    msgs = build_messages(system, hist, HumanMessage(content="now"), max_tokens=30)
    # 只裁掉最旧的 HumanMessage("old"),tool_calls+tool 对作为一块保留
    names = [type(m).__name__ for m in msgs]
    assert names == ["SystemMessage", "AIMessage", "ToolMessage", "AIMessage", "HumanMessage"]

def test_build_messages_drops_tool_round_atomically():
    system = SystemMessage(content="S")
    hist = [
        AIMessage(content="", tool_calls=[{"name": "query_order", "args": {}, "id": "c1"}]),
        ToolMessage(content="X" * 100, tool_call_id="c1"),
        AIMessage(content="answer"),
    ]
    msgs = build_messages(system, hist, HumanMessage(content="now"), max_tokens=30)
    # tool 结果太大 → tool_calls 与其 tool 结果整块丢弃,不能只丢一半留下孤儿 ToolMessage
    names = [type(m).__name__ for m in msgs]
    assert names == ["SystemMessage", "AIMessage", "HumanMessage"]
```

- [ ] **Step 2: 运行确认失败** → `pytest tests/test_history.py tests/test_context.py -v` → FAIL

- [ ] **Step 3: 实现 `history.py` 与改造 `context.py`**

`build_messages` 算法:把 `history` 切成原子块(普通消息一块;`AIMessage` 带 tool_calls 时吞并紧随其后的 `ToolMessage` 为一块),按块累计 `estimate_tokens`,超出预算从最旧块丢弃;`system` 与 `current` 永不丢。实现前 Context7 确认 `AIMessage(tool_calls=...)` 与 `ToolMessage(content, tool_call_id)` 的构造参数。

- [ ] **Step 4: 运行确认通过** → `pytest tests/test_history.py tests/test_context.py -v` → PASS

- [ ] **Step 5: Commit** → `git add app/services/history.py app/services/context.py tests/test_history.py tests/test_context.py && git commit -m "feat: add DB-backed history reconstruction with atomic tool-round trimming"`

---

## Task 9: System Prompt 工具使用约束(纯 Prompt · 标注样例验证)

**Files:**
- Modify: `app/prompts/customer_service.py`
- Test: `tests/test_prompt.py`(扩)

**Interfaces:**
- Produces: `render_system_prompt(business_name="本店") -> str` 在 ch01 基础上追加:可用能力(查订单/查商品/查物流/查常见问题/创建工单)、何时该调、查到结果礼貌转述、查不到如实告知并引导转人工。

- [ ] **Step 1: 写冒烟测试(本任务是纯 Prompt,行为正确性放到 Task 11 端到端用标注样例验证)**

```python
# tests/test_prompt.py(扩)
def test_render_mentions_tools():
    p = render_system_prompt()
    assert "订单" in p and "物流" in p and "工单" in p
    assert "转人工" in p
```

- [ ] **Step 2: 运行确认失败** → `pytest tests/test_prompt.py -v` → FAIL

- [ ] **Step 3: 改 `render_system_prompt`**

在模板里追加一段「工具能力与使用规范」,措辞由实现者起草;保留 ch01 的角色/约束。

- [ ] **Step 4: 运行确认通过** → `pytest tests/test_prompt.py -v` → PASS

- [ ] **Step 5: Commit** → `git add app/prompts/customer_service.py tests/test_prompt.py && git commit -m "feat: extend system prompt with tool-use guidance"`

---

## Task 10: 工具化 chat 编排 + main.py 接线

**Files:**
- Modify: `app/services/chat.py`, `app/main.py`
- Test: `tests/test_chat.py`(改), `tests/test_api.py`(改)

**Interfaces:**
- Consumes: `history`/`context`(Task 8)、`registry`/`executor`(Task 4/7)、`Settings`(Task 1)
- Produces:
  - `app.services.chat.stream_chat(llm, session: Session, tools: list, system_prompt: str, user_id: str, message: str, max_tokens: int) -> Iterator[str]`(yield 裸 token 串,或 JSON 状态帧 `{"type":"tool","tool":<name>,"label":<中文>}`)
  - `app.main.py`:`/chat` 增 `session=Depends(get_db)`、传入 `ALL_TOOLS` 与 `req.session_id`;`stream_chat` 的每个 yield 仍 `ServerSentEvent(raw_data=...)` 原样发。

- [ ] **Step 1: 写失败的测试(假 LLM + SQLite)**

```python
# tests/test_chat.py(改)
import json
from langchain_core.messages import AIMessage
from app.tools.registry import ALL_TOOLS
from app.services.chat import stream_chat

class FakeChunk:
    def __init__(self, content): self.content = content

class FakeBoundLLM:
    def __init__(self, tool_call=False):
        self.tool_call = tool_call
    def invoke(self, messages):
        if self.tool_call:
            return AIMessage(content="", tool_calls=[{"name": "query_order", "args": {"order_id": "1001"}, "id": "c1"}])
        return AIMessage(content="")
    def stream(self, messages):
        yield FakeChunk("已"); yield FakeChunk("查到")

class FakeLLM:
    def bind_tools(self, tools):
        return FakeBoundLLM(tool_call=self.tool_call)
    def stream(self, messages):
        yield FakeChunk("直接"); yield FakeChunk("回答")

def test_stream_chat_tool_round_emits_frame_and_persists(session):
    events = list(stream_chat(FakeLLM(tool_call=True), session, ALL_TOOLS, "SYS", "u1", "订单1001物流", 1000))
    assert any(e.startswith('{"type":"tool"') for e in events)
    assert events[-1] == "查到"  # 最终答案 token
    rows = session.query(Message).all()
    roles = [r.role for r in rows]
    assert roles == ["user", "assistant", "tool", "assistant"]

def test_stream_chat_no_tool_still_streams(session):
    events = list(stream_chat(FakeLLM(tool_call=False), session, ALL_TOOLS, "SYS", "u1", "你好", 1000))
    assert events == ["直接", "回答"]
```

- [ ] **Step 2: 运行确认失败** → `pytest tests/test_chat.py -v` → FAIL

- [ ] **Step 3: 实现 `stream_chat` 并改 `main.py`**

流程:`get_or_create_conversation` → 设 contextvar → `load_history`(旧历史)→ `build_messages` 组上下文 → `append_message(user)` 落库并 commit → `bound = llm.bind_tools(tools)` → `ai_msg = bound.invoke(messages)`:
- 无 `tool_calls`:直接 `for chunk in llm.stream(messages): yield chunk.content`,累积后落 `assistant` 消息。
- 有 `tool_calls`:落 `assistant`(tool_calls)→ 对每个 tool_call:yield 状态帧 → `execute_tool(TOOLS_BY_NAME[name], args)` → 落 `tool` 消息 + 构造 `ToolMessage` → 回灌后 `for chunk in bound.stream(final_messages): yield chunk.content` 累积,若 content 空且再出 tool_calls 则 yield 兜底文案;落最终 `assistant` 消息。
实现前 Context7 复核 `ChatOpenAI.bind_tools` 对 DeepSeek 的行为(是否需 `extra_body={"thinking":{"type":"disabled"}}`)。

- [ ] **Step 4: 运行确认通过** → `pytest tests/test_chat.py -v` → PASS

- [ ] **Step 5: 改 `tests/test_api.py` 用 SQLite session 注入,验证 `/chat` 仍 SSE + `[DONE]`**

- [ ] **Step 6: Commit** → `git add app/services/chat.py app/main.py tests/test_chat.py tests/test_api.py && git commit -m "feat: wire single-round tool calling into the chat SSE flow"`

---

## Task 11: Docker 起 MySQL + 端到端验收 + 全量测试 + dev-notes

**Files:**
- Create: `dev-notes/ch02.md`
- Modify: `.env`(填 `DATABASE_URL`)

**说明:** 本任务不走 TDD,是真实 DeepSeek + 真实 MySQL 跑三条验收,并留痕。

- [ ] **Step 1: `docker compose up -d` 起 MySQL,`.env` 填 `DATABASE_URL=mysql+pymysql://root:<pass>@localhost:3306/cs_customer?charset=utf8mb4`,`pip install pymysql`,`python -m app.db.seed` 建表灌数据**

- [ ] **Step 2: `uvicorn app.main:app --reload` 起服务**

- [ ] **Step 3: 验收 1 —— 订单物流(工具徽章)**:浏览器打开聊天页问「订单 1001 的物流到哪了」,应看到 `query_logistics` 徽章 + 按返回结果作答。

- [ ] **Step 4: 验收 2 —— FAQ 命中**:问「退货政策是什么」,`query_faq` 查得到并作答。

- [ ] **Step 5: 验收 3 —— 漏召回**:问「邮费是多少」,`query_faq` 查不到(预期),记下留给下一步升级。

- [ ] **Step 6: 全量单测复跑** `pytest -q`,全部通过。

- [ ] **Step 7: `dev-notes/ch02.md` 补记验收结果(三条各 pass/fail + 关键输出片段 + 翻车/返工),commit**

---

## Task 12: 聊天页工具徽章(Vibe Coding · 例外流程)

**Files:**
- Modify: `frontend/src/api/chat.ts`, `frontend/src/types.ts`, `frontend/src/hooks/useChat.ts`, `frontend/src/components/MessageItem.tsx`(必要时)

**说明:** 用户指定聊天页改造走 Vibe Coding,不套 brainstorm / TDD / code review。效果由用户描述、直接实现并演示确认。

**效果:** 后端推 `{"type":"tool","tool":...,"label":...}` 帧时,`parseSSE` 识别并回调 `onTool`;`useChat` 把该帧的工具名暂存,在随后的 assistant 气泡上渲染一个工具徽章(如「🔧 查询物流」);无工具帧则无徽章。保留现有手动 SSE 解析与状态机。

- [ ] **Step 1: 改 `api/chat.ts` + `types.ts` + `useChat.ts` + `MessageItem.tsx` 实现徽章渲染**
- [ ] **Step 2: `npm run dev` 起前端,配合后端演示三条验收,确认徽章显示与文字回答**
- [ ] **Step 3: Commit**

```bash
git add frontend/src/
git commit -m "feat: render tool-call badge in chat bubbles"
```

---

## Self-Review 结论

- **Spec 覆盖**:持久化基座(Task 1/2/3)、五个工具(Task 4/5/6)、工具基础设施(Task 7)、接入 `/chat`(Task 10)、单轮约束(Task 10)、状态帧 + 工具徽章(Task 10/12)、会话/消息落库(Task 8/10)、三条验收(Task 11)—— 全部有对应任务。
- **Step 可执行性**:每个测试步给了测试名与断言代码,每个实现步给了签名与边界。
- **类型一致**:`get_engine`/`get_session_factory`/`get_db`(Task 1)→ `Base`/模型(Task 2)→ `get_or_create_conversation`/`append_message`/`load_history`(Task 8)→ `stream_chat`(Task 10)的签名与返回类型在各自 Interfaces 中一致;`execute_tool(tool, args, ...)` 在 Task 7 定义、Task 10 调用一致。
- **Review Focus**:6 条分别归属 Task 8 / 7 / 7 / 10 / 8 / 8 的测试。
- **比例**:计划长度与 spec 相当,未写成代码转录;纯 Prompt/数据任务(Task 3/9)已按要求改用标注样例/冒烟验证。
