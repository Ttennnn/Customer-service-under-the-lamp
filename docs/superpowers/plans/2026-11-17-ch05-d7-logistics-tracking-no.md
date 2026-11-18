# Ch05 D7 物流链式依赖 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `query_logistics` 的入参从 `order_id` 改为 `tracking_no`，并让 `query_order` 返回 `tracking_no`，使「先查订单再查物流」成为真实数据依赖。

**Architecture:** 只改 ch02 现有 mock 业务工具的契约与 ch05 Agent 的 prompt/假 args；不新增工具、不改图结构、不动 `/ch05` 接口与旧 React `/chat`。`query_order` 每次生成并返回 `tracking_no`；`query_logistics` 只接受该单号并回显。

**Tech Stack:** Python 3.13、LangChain `@tool`、pytest。

**Spec:** `docs/superpowers/specs/2026-11-17-ecommerce-cs-ch05-agent-workflow-design.md`（§5.3 D7 行、§6.5.0、§9.3 验收 5、§11 风险 2）

## Global Constraints

- 只修改现有工具 `query_order` / `query_logistics`；不新增业务工具。
- `query_logistics(tracking_no: str)` 不再接受 `order_id`；参数 schema 必填。
- `query_order` 返回体必须含 `tracking_no`，格式 `^SF\d{10}$`。
- `query_logistics` 返回体为 `{"tracking_no","carrier","current_node","traces"}`，回显入参单号。
- 旧 React `/chat`、`/ch05/chat`、`/ch05/tickets` 的实现不改。
- 全量基线 `465 passed`；改后必须全绿。
- 测试命令统一用 `.venv/Scripts/python.exe -m pytest`（worktree 内）。

## Review Focus

1. 旧调用方仍传 `{"order_id": ...}` 给 `query_logistics`：必须抛参数校验错误，不能静默按订单号返回物流（Task 1）。
2. `query_order` 无条件带合法 `tracking_no`：否则链式调用会断（Task 1）。
3. 物流结果里的 `tracking_no` 必须与订单返回的单号逐字一致：防止物流工具自己重新生成单号（Task 1）。
4. Agent prompt 必须写明单号来源与串行规则：否则模型可能猜单号（Task 2）。
5. 历史消息里残留旧 `{"order_id": ...}` 工具参数只能回放、不能被再次执行或校验（Task 2 验证 `test_history.py` 保持不动）。

---

### Task 1: 工具契约与单元测试

**Files:**
- Modify: `app/tools/business.py`（模块 docstring、`query_order`、`query_logistics`）
- Test: `tests/test_tools_business.py`

**Interfaces:**
- Produces:
  - `query_order(order_id: str) -> str`：JSON `{"order_id","items","amount","status","tracking_no"}`。
  - `query_logistics(tracking_no: str) -> str`：JSON `{"tracking_no","carrier","current_node","traces"}`。

- [ ] **Step 1: 写失败测试**

在 `tests/test_tools_business.py` 顶部补 `import re`、`import pytest`，把 `test_query_order_echoes_id` / `test_query_logistics_has_node` 替换为：

```python
def test_query_order_returns_tracking_no():
    data = json.loads(query_order.invoke({"order_id": "1001"}))
    assert data["order_id"] == "1001"
    assert set(data) >= {"order_id", "items", "amount", "status", "tracking_no"}
    assert re.fullmatch(r"SF\d{10}", data["tracking_no"])


def test_query_logistics_uses_tracking_no():
    data = json.loads(query_logistics.invoke({"tracking_no": "SF1234567890"}))
    assert data["tracking_no"] == "SF1234567890"
    assert set(data) >= {"tracking_no", "carrier", "current_node", "traces"}


def test_order_tracking_no_flows_into_logistics():
    order = json.loads(query_order.invoke({"order_id": "1001"}))
    logistics = json.loads(query_logistics.invoke({"tracking_no": order["tracking_no"]}))
    assert logistics["tracking_no"] == order["tracking_no"]


def test_query_logistics_rejects_order_id_argument():
    with pytest.raises(Exception) as excinfo:
        query_logistics.invoke({"order_id": "1001"})
    assert "tracking_no" in str(excinfo.value)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tools_business.py -v`
Expected: FAIL；`test_query_order_returns_tracking_no` 缺 `tracking_no`，物流用例因参数 `tracking_no` 不存在而失败。

- [ ] **Step 3: 实现契约**

在 `app/tools/business.py`：

- 模块 docstring 末句改为「并原样回显入参中的 `order_id` / `product_name` / `tracking_no`」。
- `query_order`：在返回 dict 中增加 `"tracking_no": tracking_no`；`tracking_no` 用 `"SF" + "".join(str(random.randint(0, 9)) for _ in range(10))` 生成。docstring 补一句：`查询物流时请使用本工具返回的 tracking_no。`
- `query_logistics`：签名改为 `def query_logistics(tracking_no: str) -> str:`；docstring 改为「根据物流单号查询物流单号、承运商、当前物流节点与完整物流轨迹。物流单号必须来自 query_order 返回的 tracking_no，不要用订单号或猜测值代替。」；删除随机生成 `logistics_no` 的行；返回 `json.dumps({"tracking_no": tracking_no, "carrier": carrier, "current_node": current_node, "traces": traces}, ensure_ascii=False)`。

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tools_business.py -v`
Expected: PASS（4 passed）。

- [ ] **Step 5: 提交**

```bash
git add app/tools/business.py tests/test_tools_business.py
git commit -m "feat: chain logistics lookup on order tracking number"
```

---

### Task 2: Prompt、假 args 与历史兼容

**Files:**
- Modify: `app/prompts/customer_service.py:113-122`（`_AGENT_TOOL_RULES`）
- Modify: `tests/test_agent_raw_loop.py:71`
- Test: `tests/test_prompt.py`
- Keep: `tests/test_history.py:19` 不改（旧参数回放兼容）

**Interfaces:**
- Consumes: Task 1 的工具契约。
- Produces: `render_agent_system_prompt()` 文本含 `tracking_no` 来源规则。

- [ ] **Step 1: 写失败测试**

在 `tests/test_prompt.py` 增加：

```python
def test_agent_prompt_requires_tracking_no_from_query_order():
    from app.prompts.customer_service import render_agent_system_prompt

    p = render_agent_system_prompt()
    assert "tracking_no 只能来自 query_order 的返回" in p
    assert "不要用订单号或猜测值代替" in p
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_prompt.py -v`
Expected: FAIL，断言文本尚不存在。

- [ ] **Step 3: 实现 prompt 与假 args**

`_AGENT_TOOL_RULES` 中两行改为：

```text
- query_order:查询订单状态、商品明细、金额与物流单号(tracking_no)。
- query_logistics:按 tracking_no 查询承运商、当前节点与物流轨迹;tracking_no 只能来自 query_order 的返回,不要用订单号或猜测值代替。
```

`tests/test_agent_raw_loop.py:71` 的 `tool_chunk("query_logistics", '{"order_id":"1001"}', "c2")` 改为 `tool_chunk("query_logistics", '{"tracking_no":"SF1234567890"}', "c2")`。

`tests/test_history.py` 保持旧参数不变，并在测试内加一行注释：`# 旧历史参数只回放、不执行，用于验证兼容性`。

- [ ] **Step 4: 运行相关测试确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_prompt.py tests/test_agent_raw_loop.py tests/test_history.py -v`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add app/prompts/customer_service.py tests/test_prompt.py tests/test_agent_raw_loop.py tests/test_history.py
git commit -m "docs: tell agent to fetch tracking number before logistics"
```

---

### Task 3: 全量回归与开发记录

**Files:**
- Modify: `dev-notes/ch05.md`（追加「D7 补做」阶段）
- Commit: `docs/superpowers/specs/2026-11-17-ecommerce-cs-ch05-agent-workflow-design.md`、本计划文件

- [ ] **Step 1: 跑全量测试**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: 全绿；数量 ≥ 468（基线 465，替换 2 条并新增 3 条）。若失败，按 systematic-debugging 修复后再继续。

- [ ] **Step 2: 追记 dev-notes**

在 `dev-notes/ch05.md` 末尾追加一节，包含四样：

- 用户这一步发的关键原话：「只落实 D7 物流链式依赖」与设计批准。
- 我的关键产出：工具契约、prompt 规则、测试结果（写明实际 passed 数）。
- 我拒绝或纠偏了什么：不做格式闸、不改图/接口/页面；`test_history.py` 刻意保留旧参数。
- 翻车与返工：如实记录 RED/GREEN；若全量有失败，写明修复过程。

- [ ] **Step 3: 提交文档**

```bash
git add dev-notes/ch05.md docs/superpowers/specs/2026-11-17-ecommerce-cs-ch05-agent-workflow-design.md docs/superpowers/plans/2026-11-17-ch05-d7-logistics-tracking-no.md
git commit -m "docs: record ch05 d7 tracking-number dependency"
```

- [ ] **Step 4: 推送更新 PR**

Run: `git push origin feature/ch05-agent-workflow`
Expected: origin 接受，PR #3 自动包含新 commits。
