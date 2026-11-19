# Ch08 工具系统 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把客服系统静态的内置工具层升级为「统一注册中心 + 单一执行引擎 + 两个 MCP Server + 建工单 interrupt 确认流 + tool_audit_logs 审计」。

**Architecture:** `app/tools/builtin/` 启动扫描登记内置工具；`get_all_specs()` 每次现拉 MCP 工具并与内置合并；所有模型调用走 `app/tools/engine.py` 的校验→权限→执行→分诊→格式化→审计管道；`tools_node` 在执行任何工具之前对 `create_ticket` 做纯校验并 `interrupt` 确认；两台 MCP Server 独立进程走 Streamable HTTP。

**Tech Stack:** Python 3.13、LangGraph 1.2.20、LangChain 1.5、FastAPI、SQLAlchemy 2.1、MCP 官方 Python SDK、`langchain-mcp-adapters`、`jsonschema`。

**Spec:** `docs/superpowers/specs/2026-11-19-ecommerce-cs-ch08-tool-system-design.md`

## Global Constraints

- MCP Server 用官方 Python SDK，Streamable HTTP；MCP Client 用 `langchain-mcp-adapters.MultiServerMCPClient`。不得换方案。
- 写操作只有 `create_ticket`；写操作 0 重试；未经确认的写调用不得落库。
- MCP 工具在本侧一律只读，忽略 Server 的 `destructiveHint` / 描述 / 自定义字段。
- 审计表不挂外键；审计写失败只 log，绝不阻断工具执行。
- `tool_audit_logs` 的 status 只能是 `成功 / 失败 / 超时 / 校验拦下 / 权限拒绝`。
- `MCP_ENABLED` 的代码默认值为 `False`；测试默认零网络。
- 内置工具扫描目录是 `app/tools/builtin/`；新增内置工具只丢文件，不改核心注册表。
- 全量测试命令：`.venv/Scripts/python.exe -m pytest -q`。
- `.env` 当前是 ch07 演示配置；跑全量前用 ch07 记录里的默认值环境变量覆盖：`MAX_AGENT_STEPS=6 TOOL_RESULT_MAX_TOKENS=2000 RERANK_TOP_K= MODEL_CONTEXT_WINDOW=65536 MAX_OUTPUT_TOKENS=2000 MAX_USER_INPUT_TOKENS=2000`。

## Review Focus

1. **确认被绕过**：pending 工单时用户发新消息、前端发非法 resume、同批多个 `create_ticket`——都不能写 tickets；Task 9/10 的测试必须覆盖。
2. **resume 重跑读工具**：同一 AIMessage 里 `query_order` + `create_ticket`，确认/取消后 `query_order` 只能执行一次；Task 9 用计数工具验证。
3. **写操作超时后重复执行**：`create_ticket` 超时必须 `retry_count=0` 且只落一张工单；Task 6/14 验证。
4. **审计失败反拦执行**：审计 insert 抛异常时工具照常返回结果；Task 6 用 monkeypatch 验证。
5. **回复漏工单号**：模型确认后不写工单号时，服务层必须确定性追加并流式发给前端；Task 10 验证。

---

### Task 1: 分支、依赖、配置与审计表

**Files:**
- Create: `sql/ch08-ddl.sql`
- Create: `app/models/tool_audit.py`
- Modify: `app/models/__init__.py`
- Modify: `app/config.py`
- Modify: `requirements.txt`
- Modify: `.env.example`
- Test: `tests/test_tool_audit_model.py`

**Interfaces:**
- Produces: `ToolAuditLog` ORM class with table `tool_audit_logs`.
- Produces: `Settings.mcp_enabled: bool = False`, `mcp_logistics_url`, `mcp_aftersales_url`, `mcp_discovery_timeout_s: float = 2`, `mcp_tool_timeout_s: float = 10`, `tool_default_timeout_s: float = 5`, `tool_max_retries: int = 2`, `tool_audit_result_max_chars: int = 500`, `demo_ticket_delay_seconds: float = 0`.

- [ ] **Step 1: Create the chapter branch**

```bash
git checkout -b feature/ch08-tool-system
```

- [ ] **Step 2: Install and pin dependencies**

Run:
```bash
.venv/Scripts/python.exe -m pip install mcp langchain-mcp-adapters jsonschema
.venv/Scripts/python.exe -m pip show mcp langchain-mcp-adapters jsonschema
```
Expected: all three importable. Record exact versions as `==` pins in `requirements.txt`. Verify the installed MCP SDK import path:
```bash
.venv/Scripts/python.exe -c "from mcp.server import MCPServer; print(MCPServer)"
```
If that import fails, use the version's own documented path (`from mcp.server.fastmcp import FastMCP`) and record it in dev-notes. Do not change the chosen libraries.

- [ ] **Step 3: Write the failing model test**

Create `tests/test_tool_audit_model.py`:
```python
from app.models import ToolAuditLog


def test_tool_audit_log_roundtrip(session):
    row = ToolAuditLog(
        conversation_id=7,
        tool_call_id="call-1",
        tool_name="query_order",
        tool_source="builtin",
        mcp_server=None,
        arguments={"order_id": "1001"},
        result_summary="订单 1001 已签收",
        status="成功",
        error_message=None,
        retry_count=0,
        duration_ms=12,
    )
    session.add(row)
    session.commit()

    loaded = session.query(ToolAuditLog).one()
    assert loaded.status == "成功"
    assert loaded.tool_source == "builtin"
    assert loaded.arguments == {"order_id": "1001"}
    assert loaded.conversation_id == 7
```

- [ ] **Step 4: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tool_audit_model.py -v`
Expected: FAIL with `ImportError: cannot import name 'ToolAuditLog'`.

- [ ] **Step 5: Implement the ORM model and DDL**

`app/models/tool_audit.py`:
```python
from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Enum, Integer, JSON, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class ToolAuditLog(Base):
    __tablename__ = "tool_audit_logs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int | None] = mapped_column(BigInteger, index=True)
    tool_call_id: Mapped[str | None] = mapped_column(String(64))
    tool_name: Mapped[str] = mapped_column(String(128), index=True)
    tool_source: Mapped[str] = mapped_column(Enum("builtin", "mcp", name="tool_source"))
    mcp_server: Mapped[str | None] = mapped_column(String(64))
    arguments: Mapped[dict | None] = mapped_column(JSON)
    result_summary: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(
        Enum("成功", "失败", "超时", "校验拦下", "权限拒绝", name="tool_audit_status"),
        index=True,
    )
    error_message: Mapped[str | None] = mapped_column(String(512))
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
```

Add `ToolAuditLog` to `app/models/__init__.py` imports and `__all__`.

Create `sql/ch08-ddl.sql` with the exact DDL from spec Appendix A (copy verbatim, including `SET NAMES utf8mb4;`).

- [ ] **Step 6: Add settings**

Add the fields from Interfaces to `app/config.py` with the exact defaults. Add the same keys to `.env.example`, with `MCP_ENABLED=false` and the two `MOCK_DELAY_SECONDS` / `DEMO_TICKET_DELAY_SECONDS` keys.

- [ ] **Step 7: Run the test**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tool_audit_model.py -v`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add sql/ch08-ddl.sql app/models/tool_audit.py app/models/__init__.py app/config.py requirements.txt .env.example tests/test_tool_audit_model.py
git commit -m "feat: add tool audit schema and ch08 settings"
```

---

### Task 2: ToolSpec 与注册中心（含验收 1 的插件）

**Files:**
- Create: `app/tools/spec.py`
- Create: `app/tools/registry.py` (rewrite existing)
- Create: `app/tools/builtin/__init__.py`
- Create: `app/tools/builtin/promotions.py`
- Test: `tests/test_tool_registry.py`

**Interfaces:**
- Produces:
```python
Permission = Literal["read", "write"]
ToolSource = Literal["builtin", "mcp"]

@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    json_schema: dict
    tool: BaseTool
    permission: Permission
    source: ToolSource
    mcp_server: str | None = None
    timeout: float | None = None
    inject_conversation: bool = False
    format_result: Callable | None = None
    classify: Callable | None = None
    exposed_to_agent: bool = True
    routes: frozenset[str] | None = None
    label: str | None = None

def spec_from_tool(
    tool: BaseTool, *,
    permission: Permission = "read",
    source: ToolSource = "builtin",
    mcp_server: str | None = None,
    timeout: float | None = None,
    inject_conversation: bool = False,
    exposed_to_agent: bool = True,
    routes: frozenset[str] | None = None,
    label: str | None = None,
    classify: Callable | None = None,
    format_result: Callable | None = None,
) -> ToolSpec

def register(spec: ToolSpec) -> None
def load_builtin_tools() -> None
def builtin_specs() -> dict[str, ToolSpec]
def get_all_specs(settings: Settings | None = None) -> dict[str, ToolSpec]
def reset_registry() -> None
```

`spec_from_tool` fills `name=tool.name`, `description=tool.description`, `json_schema=tool.args_schema.model_json_schema()`.

- [ ] **Step 1: Write the failing registry tests**

Create `tests/test_tool_registry.py` with tests that:
```python
from langchain.tools import tool

from app.tools.registry import (
    builtin_specs, get_all_specs, load_builtin_tools, register, reset_registry,
)
from app.tools.spec import ToolSpec


@tool
def hello(name: str) -> str:
    """Say hello."""
    return f"hi {name}"


def _spec(name="hello", routes=None):
    return ToolSpec(
        name=name,
        description="Say hello.",
        json_schema=hello.args_schema.model_json_schema(),
        tool=hello,
        permission="read",
        source="builtin",
        routes=routes,
    )


def test_builtin_scan_registers_promotions_tool():
    reset_registry()
    load_builtin_tools()
    specs = builtin_specs()
    assert "query_promotions" in specs
    spec = specs["query_promotions"]
    assert spec.source == "builtin"
    assert spec.permission == "read"
    assert spec.description
    assert spec.json_schema["type"] == "object"


def test_duplicate_builtin_registration_raises():
    reset_registry()
    register(_spec())
    with pytest.raises(ValueError):
        register(_spec())


def test_get_all_specs_without_mcp_has_only_builtins(monkeypatch):
    reset_registry()
    load_builtin_tools()
    settings = Settings(_env_file=None, openai_api_key="test", mcp_enabled=False)
    monkeypatch.setattr(
        "app.tools.registry.fetch_mcp_specs",
        lambda settings: (_ for _ in ()).throw(AssertionError("no network")),
    )
    specs = get_all_specs(settings)
    assert "query_promotions" in specs
    assert all(spec.source == "builtin" for spec in specs.values())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tool_registry.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement spec.py and registry.py**

`spec.py` contains the dataclass above. `registry.py`:
- module dict `_BUILTIN`, bool `_LOADED`.
- `register` raises `ValueError(f"duplicate tool: {spec.name}")`.
- `load_builtin_tools()` uses `pkgutil.iter_modules(app.tools.builtin.__path__)` and `importlib.import_module(f"app.tools.builtin.{name}")`, then sets `_LOADED = True`. Importing a module with `register(...)` at bottom does the registration.
- `get_all_specs(settings=None)`:
  - `load_builtin_tools()`.
  - start with `dict(_BUILTIN)`.
  - if `settings and settings.mcp_enabled`: `for spec in fetch_mcp_specs(settings): if spec.name in merged: logger.warning(...); continue; merged[spec.name] = spec`.
  - return merged.
- `reset_registry()` clears `_BUILTIN`, `_LOADED`, and the `app.tools.builtin` modules from `sys.modules`? Do not clear sys.modules; instead keep a `_REGISTERED_NAMES` and let re-import be safe. For tests, modules already imported won't re-run `register`. Implement `load_builtin_tools()` to iterate modules and call `importlib.reload(module)` when `_LOADED` is False? That can duplicate. Simpler: every builtin module exposes a module-level `SPECS: list[ToolSpec]`; `load_builtin_tools()` imports modules and registers `module.SPECS`, not decorator side effects. `register` is still public. Then reset+load works even if modules cached. Use this pattern.
  - `_spec_from_tool(tool, ...)` helper in `spec.py` to build `ToolSpec` with schema.
- `builtin_specs()` calls `load_builtin_tools()` and returns a copy.

`builtin/promotions.py`:
```python
from langchain.tools import tool
from app.tools.spec import ToolSpec, spec_from_tool

@tool
def query_promotions(category: str = "全部") -> str:
    """查询当前店铺的优惠活动。"""
    return json.dumps({...}, ensure_ascii=False)

SPECS = [spec_from_tool(query_promotions, permission="read", label="查优惠")]
```

- [ ] **Step 4: Run tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tool_registry.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/tools/spec.py app/tools/registry.py app/tools/builtin/ tests/test_tool_registry.py
git commit -m "feat: add ToolSpec registry with plugin scan"
```

---

### Task 3: 迁移订单/商品工具并下线 query_logistics

**Files:**
- Create: `app/tools/builtin/orders.py`
- Modify: `app/tools/business.py` (keep mock data pure functions; remove `query_logistics`)
- Modify: `app/tools/registry.py` if the old static registry imports need removal
- Modify: `app/agent/graph.py:205-220` (`fetch_order_node` uses `order_snapshot`)
- Modify: `tests/test_tools_business.py`
- Modify: `tests/test_agent_tools.py`, `tests/test_agent_subgraph.py`, `tests/test_agent_raw_loop.py`, `tests/test_agent_service.py` where `query_logistics` is referenced

**Interfaces:**
- Produces: `query_order`, `query_product` as `BaseTool`; `DEMO_ORDERS`, `list_user_orders`, `order_snapshot` pure functions in `app/tools/business.py`.
- Produces: builtin specs contain `query_order`, `query_product`, not `query_logistics`.

- [ ] **Step 1: Update the failing tool-list tests**

In `tests/test_tools_business.py`, change the expected builtin names to:
```python
names = set(builtin_specs())
assert {"query_order", "query_product", "query_promotions"} <= names
assert "query_logistics" not in names
```
(Task 4 adds the remaining builtins; do not assert them here yet.)

Remove or rewrite `test_query_logistics_*` to assert the function is gone:
```python
def test_query_logistics_is_removed_from_builtin():
    import app.tools.business as business
    assert not hasattr(business, "query_logistics")
```

- [ ] **Step 2: Run tests to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tools_business.py -v`
Expected: FAIL because `builtin_specs` has no order tools / `query_logistics` still exists.

- [ ] **Step 3: Move implementations**

- Move `query_order`, `query_product`, `_order_rng`, `_days_ago` into `app/tools/builtin/orders.py`.
- Keep `DEMO_ORDERS`, `list_user_orders`, and a new `order_snapshot(order_id)` in `app/tools/business.py`; `order_snapshot` returns `copy.deepcopy(DEMO_ORDERS[order_id])` or `None`.
- `builtin/orders.py` imports `DEMO_ORDERS`, `_order_rng` from `app.tools.business`; avoid circular import by keeping RNG helper in `business.py` and importing it.
- Define `SPECS = [spec_from_tool(query_order, permission="read", label="查订单"), spec_from_tool(query_product, permission="read", label="查商品")]`.
- Delete `query_logistics` entirely. Do not leave a re-export.
- Update `app/tools/registry.py` to no longer import the old static tools.
- Modify `app/agent/graph.py` `fetch_order_node`: replace the `execute_tool(query_order, ...)` call with `order_data = order_snapshot(order_id)` and drop the corresponding imports. `order_snapshot` must return the same dict shape `query_order` produced, including generated data for unknown ids.

- [ ] **Step 4: Update agent/prompt references**

- `app/agent/tools.py`: remove `query_logistics` from static lists (this file will be fully rewritten in Task 8; for now remove the symbol so imports don't break).
- `app/prompts/customer_service.py`: remove the `query_logistics` bullet from `_AGENT_TOOL_RULES` and add a line that logistics is queried via the MCP `query_logistics` tool with `tracking_no` from `query_order`.
- Update tests that reference the removed tool by using `query_order` or a fake name.

- [ ] **Step 5: Run focused tests**

Run:
```bash
.venv/Scripts/python.exe -m pytest tests/test_tools_business.py tests/test_agent_tools.py tests/test_agent_subgraph.py -q
```
Expected: PASS, with no `query_logistics` import errors.

- [ ] **Step 6: Commit**

```bash
git add app/tools/builtin/orders.py app/tools/business.py app/agent/tools.py app/prompts/customer_service.py tests
git commit -m "refactor: move order tools to builtin and retire query_logistics"
```

---

### Task 4: 迁移 faq / create_ticket / 控制工具并支持演示延迟

**Files:**
- Create: `app/tools/builtin/faq.py`
- Create: `app/tools/builtin/tickets.py`
- Create: `app/tools/builtin/refunds.py`
- Modify: `app/tools/faq.py`, `app/tools/ticket.py` to re-export for compatibility
- Modify: `app/agent/tools.py` to re-export control tools
- Modify: `tests/test_tools_faq.py`, `tests/test_tools_ticket.py`, `tests/test_tools_business.py`
- Test: `tests/test_tools_ticket.py::test_create_ticket_sleeps_after_commit`

**Interfaces:**
- Produces builtin specs: `query_faq` (`exposed_to_agent=False`, timeout 30), `create_ticket` (`permission="write"`, `inject_conversation=True`), `submit_refund` (`routes={"refund_flow"}`), `suggest_handoff` (`routes=None`).
- `create_ticket` reads `get_settings().demo_ticket_delay_seconds` and sleeps **after** `s.commit()`, before returning the ticket number.

- [ ] **Step 1: Write the delay test**

```python
def test_create_ticket_sleeps_after_commit(session, monkeypatch):
    import time as time_module
    import app.tools.builtin.tickets as tickets

    calls = []
    monkeypatch.setattr(tickets.time, "sleep", lambda seconds: calls.append(seconds))
    monkeypatch.setattr(
        tickets, "get_settings",
        lambda: SimpleNamespace(demo_ticket_delay_seconds=0.5),
    )
    token = current_conversation_id.set(1)
    try:
        out = tickets.create_ticket.invoke(
            {"description": "屏幕碎了", "ticket_type": "售后"}
        )
    finally:
        current_conversation_id.reset(token)

    assert TICKET_NO_PATTERN.match(out)
    assert calls == [0.5]
    assert session.query(Ticket).count() == 1
```

- [ ] **Step 2: Run test to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tools_ticket.py::test_create_ticket_sleeps_after_commit -v`
Expected: FAIL because `app.tools.builtin.tickets` does not exist.

- [ ] **Step 3: Move implementations and register**

- Move `query_faq` + `order_for_prompt` + `MISS_TEXT`/`ERROR_TEXT` + `CitationCollector` usage into `builtin/faq.py`; keep imports from `app.tools.context`.
  - `SPECS = [spec_from_tool(query_faq, permission="read", timeout=30.0, exposed_to_agent=False, label="查常见问题")]`.
- Move `create_ticket` + `TICKET_NO_PATTERN` into `builtin/tickets.py`; add:
```python
settings = get_settings()
...
s.commit()
if settings.demo_ticket_delay_seconds > 0:
    time.sleep(settings.demo_ticket_delay_seconds)
return ticket_no
```
  - `SPECS = [spec_from_tool(create_ticket, permission="write", inject_conversation=True, label="创建工单")]`.
- Move `submit_refund` from `app/agent/tools.py` into `builtin/refunds.py`; register with `routes=frozenset({"refund_flow"})`.
- Move `suggest_handoff` into `builtin/refunds.py` (or `builtin/control.py`); register with no route restriction.
- `app/tools/faq.py` and `app/tools/ticket.py` become re-export shims: `from app.tools.builtin.faq import *` etc., so existing imports keep working.
- `app/agent/tools.py` imports `suggest_handoff`/`submit_refund` from the builtin module and re-exports them.

- [ ] **Step 4: Update tests**

- `tests/test_tools_business.py` expected builtin names now include `query_faq`, `create_ticket`, `submit_refund`, `suggest_handoff`.
- `tests/test_tools_ticket.py` keeps direct `create_ticket.invoke` tests; they pass because the tool reads contextvar and writes DB.
- `tests/test_tools_faq.py` imports still work through shims.

- [ ] **Step 5: Run focused tests**

Run:
```bash
.venv/Scripts/python.exe -m pytest tests/test_tools_faq.py tests/test_tools_ticket.py tests/test_tools_business.py tests/test_agent_tools.py -q
```
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add app/tools/builtin app/tools/faq.py app/tools/ticket.py app/agent/tools.py tests
git commit -m "refactor: move faq ticket and control tools into builtin registry"
```

---

### Task 5: JSON Schema 校验、结果格式化与分类

**Files:**
- Create: `app/tools/validation.py`
- Create: `app/tools/formatting.py`
- Test: `tests/test_tool_validation.py`
- Test: `tests/test_tool_formatting.py`

**Interfaces:**
- Produces: `validate_arguments(schema: dict, arguments: dict) -> list[str]` (空列表表示通过，错误为中文)。
- Produces: `classify_result(spec: ToolSpec, raw: object) -> Literal["ok","empty","error"]`.
- Produces: `format_result(spec: ToolSpec, raw: object, max_chars: int) -> tuple[str, object]` returning `(model_content, data)`.

- [ ] **Step 1: Write failing validation tests**

```python
def test_missing_required_is_chinese():
    schema = {"type": "object", "properties": {"description": {"type": "string"}},
              "required": ["description"]}
    errors = validate_arguments(schema, {})
    assert errors and "缺少必填参数 description" in errors[0]


def test_type_enum_and_minimum():
    schema = {
        "type": "object",
        "properties": {
            "ticket_type": {"type": "string", "enum": ["售后", "投诉"]},
            "count": {"type": "integer", "minimum": 1},
        },
        "required": ["ticket_type", "count"],
    }
    assert "类型应为 string" in validate_arguments(schema, {"ticket_type": 1, "count": 1})[0]
    assert "只能取" in validate_arguments(schema, {"ticket_type": "x", "count": 1})[0]
    assert "不能小于 1" in validate_arguments(schema, {"ticket_type": "售后", "count": 0})[0]
```

- [ ] **Step 2: Write failing formatting/classification tests**

```python
def test_query_faq_content_is_not_json_wrapped():
    spec = builtin_specs()["query_faq"]
    content, data = format_result(spec, "[1] 章节:x\n问:y\n答:z", 2000)
    assert content.startswith("[1]")


def test_create_ticket_ok_and_error_classification():
    spec = builtin_specs()["create_ticket"]
    assert classify_result(spec, "T20261119001") == "ok"
    assert classify_result(spec, "创建工单失败:缺少会话上下文。") == "error"


def test_mcp_found_false_is_empty():
    spec = builtin_specs()["query_order"]
    assert classify_result(spec, '{"found": false}') == "empty"


def test_logistics_status_code_is_translated():
    spec = ToolSpec(name="query_logistics", description="", json_schema={},
                    tool=query_order, permission="read", source="mcp",
                    mcp_server="logistics")
    content, data = format_result(
        spec, '{"tracking_no":"SF1","status_code":"TRANSIT","traces":[]}', 2000
    )
    assert "运输中" in content
    assert "TRANSIT" not in content
```

- [ ] **Step 3: Run tests to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tool_validation.py tests/test_tool_formatting.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 4: Implement validation.py**

Use `jsonschema.Draft202012Validator`. For each error:
- `validator == "required"`: `参数校验未通过：缺少必填参数 {name}`
- `validator == "type"`: `参数校验未通过：参数 {path} 类型应为 {expected}，实际为 {actual type}`
- `validator == "enum"`: `参数校验未通过：参数 {path} 只能取 {values}`
- `validator == "minimum"` / `maximum`: `参数校验未通过：参数 {path} 不能小于/大于 {limit}`
- otherwise: `参数校验未通过：参数 {path} {message}`
Path is `".".join(str(p) for p in error.absolute_path) or "(根)"`.

- [ ] **Step 5: Implement formatting.py**

- `FIELD_ALLOWLISTS` and `ENUM_MAPS` from spec §5.
- `classify_result`: default, `query_faq` sentinels, `create_ticket` regex, MCP `found:false`.
- `format_result`:
  - `query_faq`: return raw unchanged.
  - `create_ticket`: content `json.dumps({"ticket_no": raw}, ensure_ascii=False)`, data=raw.
  - known JSON: parse, filter allowlist, translate enum fields, serialize compact JSON.
  - unknown: `json.dumps(raw, ensure_ascii=False, default=str)` if not str, else raw.
  - truncate to `max_chars`, append `…（结果已截断）`.
- Empty/error content is produced by engine, not formatter.

- [ ] **Step 6: Run tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tool_validation.py tests/test_tool_formatting.py -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add app/tools/validation.py app/tools/formatting.py tests/test_tool_validation.py tests/test_tool_formatting.py
git commit -m "feat: add JSON schema validation and result formatting"
```

---

### Task 6: 审计写入、权限策略与统一执行引擎

**Files:**
- Create: `app/tools/audit.py`
- Create: `app/tools/policy.py`
- Create: `app/tools/engine.py`
- Create: `app/tools/spec.py` additions: `ToolRun`
- Test: `tests/test_tool_engine.py`
- Test: `tests/test_tool_audit.py`

**Interfaces:**
- Produces:
```python
WRITE_TOOLS = frozenset({"create_ticket"})

@dataclass
class ToolRun:
    status: Literal["ok","empty","error","timeout","validation_error","permission_denied"]
    content: str
    data: object = None
    duration_ms: int = 0
    retry_count: int = 0
    error_message: str | None = None

def record_tool_audit(...) -> None
def execute(spec, arguments, *, conversation_id, tool_call_id=None,
            confirmed=False, settings=None, retries=None) -> ToolRun
def execute_tool_call(tool_call, conversation_id, *, confirmed=False,
                      specs=None, settings=None) -> ToolRun
def deny_validation(spec, arguments, *, conversation_id, tool_call_id,
                    error_message) -> ToolRun
def deny_permission(spec, arguments, *, conversation_id, tool_call_id,
                    error_message) -> ToolRun
```

- [ ] **Step 1: Write failing engine tests**

Use a local `@tool` that records calls:
```python
@tool
def echo(value: str) -> str:
    """Echo value."""
    calls.append(value)
    return value


def _spec(tool=echo, name="echo", permission="read", timeout=0.2,
          classify=None):
    return ToolSpec(
        name=name, description="Echo.", json_schema=tool.args_schema.model_json_schema(),
        tool=tool, permission=permission, source="builtin", timeout=timeout,
        classify=classify,
    )
```

Tests:
```python
def test_validation_error_audits_and_does_not_execute(session):
    spec = _spec()
    run = execute(spec, {}, conversation_id=1)
    assert run.status == "validation_error"
    assert "缺少必填参数 value" in run.content
    row = session.query(ToolAuditLog).one()
    assert row.status == "校验拦下"
    assert calls == []


def test_unconfirmed_write_is_denied_and_audited(session):
    spec = _spec(name="create_ticket", permission="write")
    run = execute(spec, {"value": "x"}, conversation_id=1)
    assert run.status == "permission_denied"
    row = session.query(ToolAuditLog).one()
    assert row.status == "权限拒绝"


def test_read_timeout_retries_once(session):
    def slow(value):
        time.sleep(0.2)
        return value
    spec = _spec(tool=tool(slow), timeout=0.05)
    run = execute(spec, {"value": "x"}, conversation_id=1, retries=1)
    assert run.status == "timeout"
    assert run.retry_count == 1
    row = session.query(ToolAuditLog).one()
    assert row.status == "超时" and row.retry_count == 1


def test_write_timeout_never_retries(session):
    ...
    run = execute(spec, ..., confirmed=True, retries=2)
    assert run.retry_count == 0
    assert session.query(ToolAuditLog).one().retry_count == 0


def test_empty_result_is_success_and_not_retried(session):
    spec = _spec(tool=tool(lambda value: ""))
    run = execute(spec, {"value": "x"}, conversation_id=1)
    assert run.status == "empty"
    assert session.query(ToolAuditLog).one().status == "成功"


def test_audit_failure_does_not_block_execution(session, monkeypatch):
    monkeypatch.setattr("app.tools.engine.record_tool_audit",
                        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("db down")))
    run = execute(_spec(), {"value": "ok"}, conversation_id=1)
    assert run.status == "ok"
    assert run.content == "ok"


def test_conversation_id_reaches_tool_thread(session):
    @tool
    def read_cid(value: str) -> str:
        """Read cid."""
        return str(current_conversation_id.get())
    spec = _spec(tool=read_cid)
    run = execute(spec, {"value": "x"}, conversation_id=42)
    assert run.content == "42"
```

- [ ] **Step 2: Run tests to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tool_engine.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement audit.py**

`record_tool_audit` takes the exact fields, opens `get_session_factory()()`, adds `ToolAuditLog`, commits, closes in `finally`; wraps everything in `try/except Exception: logger.exception(...)`. Truncate `result_summary` to `settings.tool_audit_result_max_chars`; serialize arguments with `json.dumps(default=str, ensure_ascii=False)`.

- [ ] **Step 4: Implement policy.py**

```python
WRITE_TOOLS = frozenset({"create_ticket"})

def effective_permission(spec: ToolSpec) -> str:
    if spec.source == "mcp":
        return "read"
    return "write" if spec.name in WRITE_TOOLS else "read"
```
`register` must not allow a builtin spec to set `permission="write"` for a name outside `WRITE_TOOLS`; raise at registration (add to registry Task 2 if not already).

- [ ] **Step 5: Implement engine.py**

Pipeline as spec §5:
- `execute`:
  1. `validate_arguments`; on errors, call audit (status 校验拦下), return `validation_error` with Chinese content.
  2. `permission = effective_permission(spec)`; if `write` and not `confirmed`: audit 权限拒绝, return.
  3. Execute with attempts: `0` if write else `max(0, retries if retries is not None else settings.tool_max_retries)`.
     - Set `current_conversation_id` to `conversation_id` before copying context.
     - Timeout: `spec.timeout or (settings.mcp_tool_timeout_s if spec.source=="mcp" else settings.tool_default_timeout_s)`.
     - Sync tools: daemon thread + `queue.Queue.get(timeout=...)`.
     - MCP tools: detect `inspect.iscoroutinefunction(spec.tool.func)` or `spec.source=="mcp"`; call `run_async(spec.tool.ainvoke(arguments), timeout=...)`.
     - Retry only on timeout or transient exception (`TimeoutError`, `ConnectionError`, `OSError`, `httpx.TransportError`, `asyncio.TimeoutError`); backoff `0.2 * attempt` via `time.sleep`.
     - On final timeout: status timeout, audit, content `工具执行超时：{name}（已重试 {n} 次）`.
     - On exception: status error, audit, content `工具暂时不可用：{name}（{type}）`.
  4. Classify raw: `classify_result`. If empty: audit 成功, content `未查询到相关结果：{name}`. If error: audit 失败, content `工具执行失败：{raw}`. Else format, audit 成功.
  5. Always include `retry_count`, `duration_ms`, `data`.
- `execute_tool_call`: find spec in `specs or get_all_specs(settings)`; unknown → audit with `tool_source="builtin"` and return `error`.
- `deny_validation` / `deny_permission`: build a `ToolRun`, audit with the right status, return content without executing.
- Hook semantics: if `spec.classify` is not None, call it as `spec.classify(raw)` instead of the default classifier; if `spec.format_result` is not None, apply it as a raw pre-transform before the shared formatter. Both default to the module functions in `app/tools/formatting.py`.

- [ ] **Step 6: Run tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tool_engine.py tests/test_tool_audit.py -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add app/tools/spec.py app/tools/audit.py app/tools/policy.py app/tools/engine.py tests/test_tool_engine.py tests/test_tool_audit.py
git commit -m "feat: add unified tool execution engine with audit"
```

---

### Task 7: 异步桥、MCP Client 与两台 MCP Server

**Files:**
- Create: `app/tools/async_bridge.py`
- Create: `app/tools/mcp_client.py`
- Create: `mcp_servers/__init__.py`
- Create: `mcp_servers/logistics_server.py`
- Create: `mcp_servers/aftersales_server.py`
- Test: `tests/test_mcp_servers.py`
- Test: `tests/test_mcp_client.py`

**Interfaces:**
- Produces: `run_async(coro, timeout: float | None = None)` and `shutdown_async_bridge()`.
- Produces: `fetch_mcp_specs(settings: Settings) -> list[ToolSpec]`.
- Produces MCP tools: `query_logistics(tracking_no)`, `query_warranty(order_id)`, `query_return_status(order_id)`.

- [ ] **Step 1: Write failing server-function tests**

Import the module-level tool functions directly (expose them as plain functions plus register them on the server):
```python
def test_logistics_mock_has_internal_status_code(monkeypatch):
    monkeypatch.setenv("MOCK_DELAY_SECONDS", "0")
    data = json.loads(query_logistics("SF1234567890"))
    assert data["tracking_no"] == "SF1234567890"
    assert data["status_code"] in {"PICKED_UP","TRANSIT","ARRIVED","DELIVERING","DELIVERED"}
    assert data["traces"]


def test_aftersales_tools_are_stable_by_order(monkeypatch):
    monkeypatch.setenv("MOCK_DELAY_SECONDS", "0")
    a = json.loads(query_warranty("1001"))
    b = json.loads(query_warranty("1001"))
    assert a == b
    assert a["warranty_status"] in {"IN_WARRANTY","OUT_OF_WARRANTY","UNKNOWN"}
    assert json.loads(query_return_status("1001"))["status"] in {
        "APPLIED","PICKING","IN_TRANSIT","RECEIVED","REFUNDING","DONE"
    }
```

- [ ] **Step 2: Write failing client tests**

`tests/test_mcp_client.py`:
```python
def test_disabled_mcp_returns_no_specs(monkeypatch):
    settings = Settings(_env_file=None, openai_api_key="test", mcp_enabled=False)
    monkeypatch.setattr("app.tools.mcp_client._fetch_all", ...)
    assert fetch_mcp_specs(settings) == []


def test_merge_drops_mcp_duplicate(monkeypatch):
    reset_registry()
    load_builtin_tools()
    fake = ToolSpec(name="query_order", description="",
                    json_schema={}, tool=query_order, permission="read",
                    source="mcp", mcp_server="logistics")
    monkeypatch.setattr("app.tools.registry.fetch_mcp_specs", lambda settings: [fake])
    specs = get_all_specs(Settings(_env_file=None, openai_api_key="test", mcp_enabled=True))
    assert specs["query_order"].source == "builtin"
```

- [ ] **Step 3: Run tests to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_mcp_servers.py tests/test_mcp_client.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 4: Implement async_bridge.py**

- Module-level `_loop`, `_thread`, `_lock`.
- `_ensure_loop()` starts `asyncio.new_event_loop()` + `loop.run_forever()` in a daemon thread.
- `run_async(coro, timeout=None)`:
  - ensure loop;
  - `future = asyncio.run_coroutine_threadsafe(coro, loop)`;
  - `return future.result(timeout)`;
  - on `concurrent.futures.TimeoutError`, `future.cancel()` and raise `TimeoutError`.
- `shutdown_async_bridge()` stops the loop and joins.

- [ ] **Step 5: Implement mcp_client.py**

- Build `MultiServerMCPClient` config only from settings.
- `fetch_mcp_specs` returns `[]` if not enabled.
- Async `_fetch_all`:
  - for each server, `await asyncio.wait_for(client.get_tools(server_name=name), settings.mcp_discovery_timeout_s)`;
  - catch exceptions per server, log warning, continue;
  - build `ToolSpec` for each tool:
    - `json_schema = getattr(tool, "args", None) or tool.args_schema.model_json_schema()`
    - `permission="read"`, `source="mcp"`, `mcp_server=name`, `timeout=settings.mcp_tool_timeout_s`.
    - Leave `classify` / `format_result` as `None`: the engine's default classifier/formatter already handles MCP `found:false` and enum translation by tool name.
- Use `run_async(_fetch_all(), timeout=None)`; per-server `wait_for` bounds each call. The discovery timeout in spec is 2s per server; use 2× for total.

- [ ] **Step 6: Implement the two servers**

- Use the installed SDK's current API (verify in Task 1). Each file exposes plain tool functions for tests, and registers them on the server object.
- `logistics_server.py`:
  - `query_logistics(tracking_no: str) -> str` with stable RNG seeded by SHA-256 of tracking_no; returns `tracking_no, carrier, status_code, traces`.
  - `time.sleep(float(os.environ.get("MOCK_DELAY_SECONDS", "0")))` at the start.
  - `if __name__ == "__main__": server.run(transport="streamable-http", host="127.0.0.1", port=8101)`.
- `aftersales_server.py`:
  - `query_warranty(order_id)` and `query_return_status(order_id)` with stable RNG.
  - same delay env and port 8102.
- JSON `ensure_ascii=False`.

- [ ] **Step 7: Run tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_mcp_servers.py tests/test_mcp_client.py -q`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add app/tools/async_bridge.py app/tools/mcp_client.py mcp_servers tests/test_mcp_servers.py tests/test_mcp_client.py
git commit -m "feat: add MCP client bridge and two business MCP servers"
```

---

### Task 8: Agent 从注册中心绑定工具并走统一引擎

**Files:**
- Modify: `app/agent/tools.py`
- Modify: `app/agent/subgraph.py`
- Modify: `app/agent/state.py`
- Test: `tests/test_agent_tools.py`
- Test: `tests/test_agent_subgraph.py`

**Interfaces:**
- Produces: `tools_for_route(route: str, specs: dict[str, ToolSpec]) -> list[BaseTool]`.
- Produces: `legacy_tools(specs: dict[str, ToolSpec]) -> list[BaseTool]` (three read tools).
- `agent_node` calls `get_all_specs(settings)` before binding.
- `tools_node` calls `get_all_specs(settings)` before lookup and executes through `engine.execute_tool_call`.

- [ ] **Step 1: Update failing tests**

```python
def test_business_route_binds_registry_tools(monkeypatch):
    llm = ScriptedLLM([text_chunk("hi")])
    monkeypatch.setattr(
        "app.agent.subgraph.get_all_specs",
        lambda settings: {
            "query_order": builtin_specs()["query_order"],
            "create_ticket": builtin_specs()["create_ticket"],
            "query_faq": builtin_specs()["query_faq"],
            "suggest_handoff": builtin_specs()["suggest_handoff"],
        },
    )
    build_agent_subgraph(llm).invoke(_initial_state())
    assert llm.bound_tool_names == ["query_order", "create_ticket", "suggest_handoff"]


def test_tools_node_uses_engine(monkeypatch):
    seen = {}

    def fake_execute_tool_call(tool_call, conversation_id, *, confirmed=False, specs=None, settings=None):
        seen["name"] = tool_call["name"]
        return ToolRun(status="ok", content="订单已签收")

    monkeypatch.setattr("app.agent.subgraph.execute_tool_call", fake_execute_tool_call)
    llm = ScriptedLLM([
        tool_chunk("query_order", '{"order_id":"1001"}'),
        text_chunk("已查"),
    ])
    result = build_agent_subgraph(llm).invoke(_initial_state())
    assert seen["name"] == "query_order"
    assert any(isinstance(m, ToolMessage) and m.content == "订单已签收"
               for m in result["messages"])
```

- [ ] **Step 2: Run tests to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_tools.py tests/test_agent_subgraph.py -v`
Expected: FAIL because the static tool lists no longer match.

- [ ] **Step 3: Implement `app/agent/tools.py`**

- Keep `HANDOFF_LABELS`, `suggested_actions_from_handoff`, `suggested_actions_from_refund`.
- Import `suggest_handoff` and `submit_refund` from the builtin module.
- `tools_for_route(route, specs)`:
  - route `"refund_flow"`: `submit_refund` + `suggest_handoff` + `create_ticket` + aftersales MCP tools.
  - other routes: all `exposed_to_agent` specs whose `routes` is None or contains the route.
  - exclude `query_faq` via `exposed_to_agent=False`.
  - deterministic order: registry insertion order.
- `legacy_tools(specs)` returns exactly `query_order`, `query_product`, `query_faq` in that order.
- Remove `AGENT_TOOLS`, `REFUND_AGENT_TOOLS`, `AGENT_TOOLS_BY_NAME`, `BUSINESS_TOOL_NAMES` static constants. Update imports/tests.

- [ ] **Step 4: Implement subgraph binding/execution**

- `build_agent_subgraph` gains `settings` (already) and uses `get_all_specs(settings)` inside `agent_node` unless `explicit_tools` is provided.
- `tools_node`:
  - fetch `specs = get_all_specs(settings)`.
  - for each tool call:
    - `suggest_handoff` / `submit_refund`: run the existing guard, then `execute(spec, args, ...)`; merge actions on success.
    - other: `run = execute_tool_call(tool_call, state["conversation_id"], specs=specs, settings=settings)`.
  - `ToolMessage(content=run.content, tool_call_id=...)`.
  - use `run` status for token delta.
- Do not add the create_ticket interrupt yet; `create_ticket` calls will be denied by the permission gate in this intermediate task. Task 9 adds the interrupt.

- [ ] **Step 5: Run focused tests**

Run:
```bash
.venv/Scripts/python.exe -m pytest tests/test_agent_tools.py tests/test_agent_subgraph.py tests/test_agent_context.py -q
```
Expected: PASS after updating static-list assertions to registry-driven expectations.

- [ ] **Step 6: Commit**

```bash
git add app/agent/tools.py app/agent/subgraph.py app/agent/state.py tests/test_agent_tools.py tests/test_agent_subgraph.py tests/test_agent_context.py
git commit -m "refactor: bind and execute agent tools through the registry"
```

---

### Task 9: 建工单 interrupt 确认流与 resume

**Files:**
- Modify: `app/agent/subgraph.py`
- Modify: `app/agent/state.py` (add `created_ticket_no`)
- Modify: `app/agent/service.py` (resume value union)
- Modify: `app/schemas.py` (`ResumeRequest`)
- Modify: `app/api/ch06.py`, `app/api/ch07.py`
- Test: `tests/test_agent_ticket_confirm.py`
- Test: `tests/test_agent_service.py`
- Test: `tests/test_schemas.py`

**Interfaces:**
- `build_agent_subgraph(..., checkpointer=None)` passes `checkpointer` to `graph.compile(...)` for tests.
- `resume_chat_agent(graph, session, user_id, resume_value: str | dict, *, settings, conversation_id=None)`.
- `ResumeRequest` fields: `session_id`, `order_id: str | None`, `confirmed: bool | None`, `conversation_id`.
- `AgentState`/`CustomerServiceState` gain `created_ticket_no: str`.

- [ ] **Step 1: Write failing graph-level confirmation tests**

Use `InMemorySaver`, a real SQLite session fixture, and `build_agent_subgraph(llm, settings=..., checkpointer=saver)`:
```python
def _config():
    return {"configurable": {"thread_id": "ticket-1"}}


def test_confirm_creates_ticket_and_emits_number(session):
    llm = ScriptedLLM([
        tool_chunk("create_ticket",
                   '{"description":"猫砂盆漏电，需要售后跟进","ticket_type":"售后"}'),
        text_chunk("已为您建单"),
    ])
    graph = build_agent_subgraph(llm, settings=_settings(), checkpointer=InMemorySaver())
    first = graph.invoke(_initial_state(), _config())
    assert "__interrupt__" in first
    assert first["__interrupt__"][0].value["kind"] == "confirm_ticket"

    final = graph.invoke(Command(resume={"confirmed": True}), _config())
    assert TICKET_NO_PATTERN.match(final["created_ticket_no"])
    assert session.query(Ticket).count() == 1
    assert session.query(ToolAuditLog).one().status == "成功"


def test_cancel_creates_no_ticket_and_audits_permission_denied(session):
    ...
    graph.invoke(Command(resume={"confirmed": False}), _config())
    assert session.query(Ticket).count() == 0
    row = session.query(ToolAuditLog).one()
    assert row.status == "权限拒绝"


def test_invalid_resume_fails_closed(session):
    ...
    graph.invoke(Command(resume="yes"), _config())
    assert session.query(Ticket).count() == 0
    assert session.query(ToolAuditLog).one().status == "权限拒绝"


def _dual_tool_chunk():
    return AIMessageChunk(
        content="",
        tool_call_chunks=[
            {"name": "query_order", "args": '{"order_id":"1001"}',
             "id": "c1", "index": 0},
            {"name": "create_ticket",
             "args": '{"description":"需要建单跟进","ticket_type":"售后"}',
             "id": "c2", "index": 1},
        ],
    )


def test_read_tool_runs_once_across_interrupt(session, monkeypatch):
    counter = {"n": 0}

    def counting_execute(tool_call, conversation_id, **kwargs):
        if tool_call["name"] == "query_order":
            counter["n"] += 1
        return ToolRun(status="ok", content="订单已签收")

    monkeypatch.setattr("app.agent.subgraph.execute_tool_call", counting_execute)
    llm = ScriptedLLM([_dual_tool_chunk(), text_chunk("已处理")])
    graph = build_agent_subgraph(
        llm, settings=_settings(), checkpointer=InMemorySaver()
    )
    first = graph.invoke(_initial_state(), _config())
    assert "__interrupt__" in first
    assert counter["n"] == 0       # nothing executes before confirmation
    graph.invoke(Command(resume={"confirmed": True}), _config())
    assert counter["n"] == 1
```

- [ ] **Step 2: Write failing service/schema tests**

```python
def test_resume_request_requires_exactly_one_of_order_or_ticket():
    with pytest.raises(ValidationError):
        ResumeRequest(session_id="s", order_id=None, confirmed=None)
    with pytest.raises(ValidationError):
        ResumeRequest(session_id="s", order_id="1001", confirmed=True)
    assert ResumeRequest(session_id="s", confirmed=True).confirmed is True


def test_resume_chat_agent_passes_dict_to_command(session):
    graph = FakeGraph([], {})
    list(resume_chat_agent(graph, session, "u1", {"confirmed": True}, settings=_settings()))
    assert graph.commands[-1].resume == {"confirmed": True}
```

- [ ] **Step 3: Run tests to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_ticket_confirm.py tests/test_schemas.py -v`
Expected: FAIL.

- [ ] **Step 4: Implement `tools_node` confirmation flow**

Order inside `tools_node`:
1. `specs = get_all_specs(settings)`.
2. Identify `ticket_calls = [tc for tc in last.tool_calls if tc["name"] == "create_ticket"]`.
3. If `ticket_calls`:
   - `first = ticket_calls[0]`; `spec = specs["create_ticket"]`.
   - `errors = validate_arguments(spec.json_schema, first["args"])`.
   - If no errors: `decision = interrupt({"kind": "confirm_ticket", "preview": {"ticket_type": first["args"].get("ticket_type"), "description": first["args"].get("description")}})`.
   - `confirmed = isinstance(decision, dict) and decision.get("confirmed") is True`.
4. Execute tool calls in order:
   - `create_ticket`:
     - If it is `first` and no validation errors: `run = execute(spec, args, conversation_id=..., tool_call_id=..., confirmed=confirmed, settings=settings)`.
       - If `run.status == "ok"`: `created_ticket_no = run.data`.
     - Otherwise: `run = deny_validation(spec, args, ..., error_message="一轮只能处理一个建工单请求")`.
   - Validation failure for first: `run = deny_validation(spec, args, ..., error_message="; ".join(errors))`.
   - Control tools: existing guard; on failure `deny_validation`; on success `execute`.
   - Others: `execute_tool_call(tc, cid, specs=specs, settings=settings)`.
5. Return `created_ticket_no` in the update when set.

Critical: `interrupt` must be called before any tool execution, including read tools. The tests above enforce this.

- [ ] **Step 5: Implement state, schema, service, APIs**

- `AgentState`/`CustomerServiceState`: `created_ticket_no: str`.
- `ResumeRequest`: `order_id: str | None = None`, `confirmed: bool | None = None`; `@model_validator(mode="after")` checks exactly one is not None; strip validator still applies to provided strings.
- `resume_chat_agent`:
  - rename `order_id` param to `resume_value: str | dict`;
  - audit text: string → `[订单选择] 订单 {value}`; dict with `confirmed` → `[工单确认] 确认提交` / `[工单确认] 取消`;
  - `_iter_forwarded_frames(graph, Command(resume=resume_value), config)`.
- `ch06_chat_resume` and `ch07_chat_resume` build `resume_value`.
- Remove the old required `order_id` assumption in tests.

- [ ] **Step 6: Run tests**

Run:
```bash
.venv/Scripts/python.exe -m pytest tests/test_agent_ticket_confirm.py tests/test_agent_service.py tests/test_schemas.py tests/test_ch07_service.py -q
```
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add app/agent/subgraph.py app/agent/state.py app/agent/service.py app/schemas.py app/api/ch06.py app/api/ch07.py tests
git commit -m "feat: add create_ticket interrupt confirmation flow"
```

---

### Task 10: 工单号兜底与 ch05/ch06 写路径迁移

**Files:**
- Modify: `app/agent/service.py` (`_finish_turn`)
- Modify: `app/api/ch05.py`
- Modify: `app/api/ch06.py`
- Test: `tests/test_agent_service.py`
- Test: `tests/test_api.py`
- Test: `tests/test_ch05_ticket.py` (or existing ch05 tests)

**Interfaces:**
- `_finish_turn` appends `\n\n工单已创建，工单号：{no}` when `created_ticket_no` is not in `answer`, yields it as a token frame, and persists the augmented answer.
- ch05/ch06 endpoints call `engine.execute(spec, args, confirmed=True)` and use `run.data` for the ticket number; `/ch06/refunds` uses `retries=0`.

- [ ] **Step 1: Write failing tests**

```python
def test_finish_turn_appends_ticket_number(session):
    graph = FakeGraph(
        [{"type": "token", "text": "已为您处理"}],
        {"answer": "已为您处理", "created_ticket_no": "T20261119001"},
    )
    events = list(stream_chat_agent(graph, session, "u1", "建工单", settings=_settings()))
    assert any("T20261119001" in event for event in events)
    row = session.query(Message).filter(Message.role == "assistant").one()
    assert "T20261119001" in row.content


def test_ch05_ticket_uses_confirmed_engine(session, monkeypatch):
    seen = {}
    def fake_execute(spec, args, **kwargs):
        seen["confirmed"] = kwargs.get("confirmed")
        return ToolRun(status="ok", content='{"ticket_no":"T20261119001"}',
                       data="T20261119001")
    monkeypatch.setattr("app.api.ch05.execute", fake_execute)
    ...
    assert seen["confirmed"] is True
```

- [ ] **Step 2: Run tests to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_service.py tests/test_api.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement service fallback**

In `_finish_turn`, before `append_message`:
```python
ticket_no = state_values.get("created_ticket_no") or ""
answer = state_values.get("answer") or ""
if ticket_no and ticket_no not in answer:
    suffix = f"\n\n工单已创建，工单号：{ticket_no}"
    answer += suffix
    yield suffix
```
Then persist `answer`. Keep options/yield order: suffix first, then options. (The API wrapper treats a plain string as a token.)

- [ ] **Step 4: Migrate ch05/ch06 writes**

- `app/api/ch05.py`: import `execute` and `builtin_specs`; call
```python
run = execute(builtin_specs()["create_ticket"], {...}, conversation_id=conversation.id,
              confirmed=True)
if run.status != "ok":
    raise HTTPException(502, TICKET_CREATE_FAILED)
return {"ticket_no": run.data, "message": "工单已创建"}
```
- `app/api/ch06.py` refunds: same, with `retries=0`; failures 502.
- Set `current_conversation_id` is no longer needed for create_ticket (engine sets it), but keep if other code relies.

- [ ] **Step 5: Run tests**

Run:
```bash
.venv/Scripts/python.exe -m pytest tests/test_agent_service.py tests/test_api.py tests/test_ch05_ticket.py -q
```
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add app/agent/service.py app/api/ch05.py app/api/ch06.py tests
git commit -m "feat: guarantee ticket number in replies and migrate write endpoints"
```

---

### Task 11: legacy `/chat` 迁移到注册中心与统一引擎

**Files:**
- Modify: `app/services/chat.py`
- Modify: `app/main.py`
- Modify: `app/api/conversations.py` if it passes tools
- Test: `tests/test_chat.py`
- Test: `tests/test_chat_rag.py`
- Test: `tests/test_api.py`

**Interfaces:**
- `stream_chat(..., tools, ...)` keeps its signature; tool calls are executed through `engine.execute_tool_call`.
- `main.chat` passes `legacy_tools(get_all_specs(settings))`.

- [ ] **Step 1: Write failing tests**

```python
def test_legacy_chat_binds_only_three_read_tools(monkeypatch):
    seen = {}
    def fake_stream_chat(llm, session, tools, prompt, *args, **kwargs):
        seen["names"] = [t.name for t in tools]
        yield "ok"
    monkeypatch.setattr("app.main.stream_chat", fake_stream_chat)
    ...
    assert seen["names"] == ["query_order", "query_product", "query_faq"]


def test_legacy_tool_call_goes_through_engine(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "app.services.chat.execute_tool_call",
        lambda tc, cid, **kwargs: (
            calls.append(tc["name"]),
            ToolRun(status="ok", content="订单已签收"),
        )[1],
    )
    ...  # fake LLM emits query_order, then text
    assert calls == ["query_order"]
```

- [ ] **Step 2: Run tests to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_chat.py -v`
Expected: FAIL because legacy still imports `execute_tool`/`TOOLS_BY_NAME` and `ALL_TOOLS`.

- [ ] **Step 3: Implement**

- `stream_chat` receives `specs: dict[str, ToolSpec] | None = None` (optional); resolve the spec by name, call `execute_tool_call(tc, cid, specs=specs)`.
- Unknown tool → ToolRun error content; no exception.
- `main.chat` computes `specs = get_all_specs(settings)` and `tools = legacy_tools(specs)`; pass both.
- Remove `ALL_TOOLS` usage from `main.py`; keep a compatibility alias only if tests need it.
- Ensure `query_faq` content is preserved so `_faq_failed` and `sources` logic still works.

- [ ] **Step 4: Run tests**

Run:
```bash
.venv/Scripts/python.exe -m pytest tests/test_chat.py tests/test_chat_rag.py tests/test_api.py -q
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/services/chat.py app/main.py app/api/conversations.py tests
git commit -m "refactor: route legacy chat through the tool engine"
```

---

### Task 12: 第九类意图「人工」与路由

**Files:**
- Modify: `app/agent/state.py`
- Modify: `app/agent/intent.py`
- Modify: `app/agent/routing.py`
- Modify: `app/jobs/eval_intent.py`
- Modify: `tests/eval/ch06_intent_cases.json`
- Test: `tests/test_router.py`
- Test: `tests/test_intent_cases.py`

**Interfaces:**
- `Intent` union includes `"人工"`.
- `INTENT_LABELS` includes `"人工"` after `"投诉"`.
- `INTENT_ROUTES["人工"] = "business"`.

This task is prompt/data work: replace TDD with the annotated eval set.

- [ ] **Step 1: Update the intent prompt and labels**

- `app/agent/state.py`: add `"人工"` to `Intent`.
- `app/agent/intent.py`:
  - `INTENT_LABELS` includes `"人工"`.
  - Prompt says "九类"; add category definition: 「人工：明确要求创建工单、转人工客服、人工跟进；不包括仅表达不满的投诉。」
  - Add examples: 「帮我建个工单，让售后跟进」→ 人工；「转人工客服」→ 人工。
  - Output format says 九类之一.
- `app/agent/routing.py`: `INTENT_ROUTES["人工"] = "business"`.
- `app/jobs/eval_intent.py`: `_LABEL_ORDER` includes `"人工"`; `_REPORT_PATH` points to `docs/superpowers/reports/2026-11-19-ch08-intent-eval.md`; `PROMPT_NOTE` records the ninth-class prompt change.

- [ ] **Step 2: Add two annotated cases**

Append to `tests/eval/ch06_intent_cases.json`:
```json
{"id": "M1", "text": "帮我建个工单，让售后跟进一下", "intent": "人工"},
{"id": "M2", "text": "别让机器人回复了，转人工客服", "intent": "人工"}
```
Keep the file valid JSON and do not reuse prompt few-shot text.

- [ ] **Step 3: Add routing/unit tests**

```python
def test_manual_intent_routes_to_business():
    assert route_for_intent("人工") == "business"
```

- [ ] **Step 4: Run the eval**

Run:
```bash
.venv/Scripts/python.exe -m app.jobs.eval_intent
```
Expected: 64 cases, no `error`, overall accuracy ≥ 0.90, per-class `人工` accuracy ≥ 0.80. Record the actual numbers in `dev-notes/ch08.md` and the report path printed by the job. If the classifier mislabels the two new cases, adjust the prompt examples (not the test labels) and rerun.

- [ ] **Step 5: Run intent tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_router.py tests/test_intent_cases.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add app/agent/state.py app/agent/intent.py app/agent/routing.py app/jobs/eval_intent.py tests/eval/ch06_intent_cases.json tests/test_router.py tests/test_intent_cases.py
git commit -m "feat: add manual-handoff intent routed to the main agent"
```

---

### Task 13: 前端工单预览卡（Vibe Coding）

**Files:**
- Modify: `app/static/ch07.html`
- No automated test.

This task is explicitly outside brainstorm/TDD/code review. Implement the described behavior directly and verify in the browser.

- [ ] **Step 1: Generalize the interrupt handler**

- `handleInterrupt(ctx, payload)`:
  - `payload.kind === "select_order"` → existing order picker.
  - `payload.kind === "confirm_ticket"` → new preview card.
- `confirm_ticket` preview fields: `payload.preview.ticket_type`, `payload.preview.description`.

- [ ] **Step 2: Render the preview card**

Add a card under the assistant bubble:
```text
┌ 工单预览 ─────────────────────┐
│ 工单类型：售后                 │
│ 问题描述：猫砂盆漏电，需要…    │
│ [确认提交]  [取消]             │
└───────────────────────────────┘
```
Use the existing `.order-card` visual language; add a `.ticket-preview` block. Disable both buttons after one click.

- [ ] **Step 3: Wire resume**

- 确认: `POST /ch07/chat/resume` with `{session_id, conversation_id, confirmed: true}`; stream via the existing SSE reader/watchdog/error handler.
- 取消: same with `confirmed: false`.
- While the preview is pending, keep the composer locked (`pickerPending` behavior) and clear it when the stream ends or another interrupt arrives.
- Do not touch the order picker or `renderOptions` ticket button / `/ch05/tickets` path.

- [ ] **Step 4: Manual verification**

```bash
python run.py
```
In the browser at `http://localhost:8000/`:
1. Send 「帮我建个工单」; Agent asks for the problem description.
2. Reply with a description; preview card appears.
3. Click 「确认提交」; the reply shows a ticket number.
4. Repeat and click 「取消」; the reply says canceled and no ticket is added.
5. Send a normal order question and use the order picker; it still works.

- [ ] **Step 5: Commit**

```bash
git add app/static/ch07.html
git commit -m "feat: add ticket confirmation preview card"
```

---

### Task 14: 全量验收、MCP 进程演示与交付记录

**Files:**
- Modify: `dev-notes/ch08.md`
- Optionally create: `docs/superpowers/reports/2026-11-19-ch08-acceptance.md`

- [ ] **Step 1: Run the full suite**

Run:
```bash
MAX_AGENT_STEPS=6 TOOL_RESULT_MAX_TOKENS=2000 RERANK_TOP_K= MODEL_CONTEXT_WINDOW=65536 \
MAX_OUTPUT_TOKENS=2000 MAX_USER_INPUT_TOKENS=2000 \
.venv/Scripts/python.exe -m pytest -q
```
Expected: all tests pass. If any fail, fix before proceeding.

- [ ] **Step 2: Apply DDL**

```bash
mysql -uroot -proot cs_customer < sql/ch08-ddl.sql
```
Verify:
```bash
mysql -uroot -proot -e "SHOW CREATE TABLE cs_customer.tool_audit_logs\G"
```

- [ ] **Step 3: Start the two MCP servers**

```bash
python -m mcp_servers.logistics_server
python -m mcp_servers.aftersales_server
```
Confirm ports 8101/8102 are listening.

- [ ] **Step 4: Run acceptance 1–6**

- 验收 1: `app/tools/builtin/promotions.py` exists; restart客服; ask 「现在有什么优惠活动」; Agent calls `query_promotions`.
- 验收 2: ask 「订单 1001 的物流到哪了」; logs/tool frames show `query_order` then MCP `query_logistics`; answer includes tracking data.
- 验收 3: add `query_delivery_eta` to `logistics_server.py`, restart only 8101, ask 「这单预计什么时候送达」; Agent uses the new tool without restarting客服.
- 验收 4: 「帮我建个工单」→ follow-up missing description → preview → confirm → `tickets` row + ticket number in reply.
- 验收 5: repeat → cancel → no new ticket; `tool_audit_logs` row for `create_ticket` status `权限拒绝`.
- 验收 6:
  - set `MOCK_DELAY_SECONDS=6` and `MCP_TOOL_TIMEOUT_S=1`; ask物流; audit shows `超时`, `retry_count>0`, `duration_ms`.
  - set `DEMO_TICKET_DELAY_SECONDS=6` and `TOOL_DEFAULT_TIMEOUT_S=1` (or spec timeout); confirm a ticket; audit shows `超时`, `retry_count=0`; exactly one ticket row.

- [ ] **Step 5: Record dev-notes and finish**

Append to `dev-notes/ch08.md`:
- 用户关键原话 for this stage.
- Test counts and eval numbers.
- Demo evidence (audit rows, ticket numbers).
- Any deviations/known regressions (legacy `/chat` no logistics/create ticket).
- Finish section: branch, spec/plan paths, DDL path, commands.

- [ ] **Step 6: Commit**

```bash
git add dev-notes/ch08.md docs/superpowers/reports/2026-11-19-ch08-acceptance.md
git commit -m "docs: record ch08 acceptance and delivery"
```

---

## Self-Review Notes

- Spec coverage: registry (Task 2), validation (5), permission (6/9), engine (6), audit (1/6), MCP servers/client (7), ticket confirm (9/10/13), intent (12), legacy (11), acceptance (14).
- The plan deliberately keeps `create_ticket` bound but permission-denied between Task 8 and Task 9; Task 9 replaces that intermediate behavior with the interrupt flow.
- `query_logistics` removal is intentional; legacy `/chat` loses it and `create_ticket` (documented in spec §12 and Task 14).
- Exact Chinese status values are global constraints and repeated in audit tests.
