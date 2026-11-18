# Ch06 意图识别与对话管理 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 ch05 的占位分流器升级为正式版：指代消解 + 口语归一、八类意图 + 置信度、五出口分流、退款/售后确定性子流程、订单选择器 interrupt/resume、退款表单。

**Architecture:** 在现有 LangGraph 父图上扩展普通节点（不新增嵌套子图）；`resolve_reference → classify_intent → route_by_intent` 后按五出口分流；退款/售后走 `fetch_order → retrieve_policy → confidence_gate → main_agent`；`main_agent` 仍是 ch05 的 `agent ↔ tools` ReAct 子图，按 route 选 prompt/工具。订单选择器用 LangGraph `interrupt`/`Command(resume)`；退款单复用 `tickets` 表。

**Tech Stack:** Python 3.13、FastAPI、LangGraph 1.2.20、langgraph-checkpoint-sqlite 3.1.3（同步 `SqliteSaver`）、LangChain/DeepSeek OpenAI 协议、SQLAlchemy、Milvus/BGE（ch04 检索）、原生 HTML/CSS/JS。

**Spec:** `docs/superpowers/specs/2026-11-18-ecommerce-cs-ch06-dispatcher-design.md`

## Global Constraints

- 不新增依赖/组件；不新建业务表；不修改知识入库/切分/向量化路径。
- 编排固定 LangGraph 1.2.20 + 同步 `SqliteSaver`；不换 async / 不换框架。
- 上游固定 DeepSeek OpenAI 协议；意图/核心指代/扩写用 temperature=0.0。
- 八类意图：物流、订单、商品咨询、退款退货、售后、投诉、闲聊、其他。
- 五出口：`escalate`、`fallback_script`、`knowledge`、`refund_flow`、`business`。
- `其他` 与 `闲聊` 走 `fallback_script` 固定文案；`需求澄清留在主力 Agent` 仅适用于业务/知识/退款流程内的缺信息追问。
- Query 扩写严格 3 条，只在 `refund_flow` 检索侧；商品咨询走单 query + 同义词。
- 订单目录固定 1001/1002/1003；退款流程只接受目录内订单；不建 orders 表。
- 退款原因固定五项：`七天无理由`、`质量问题`、`发错货`、`不想要了`、`其他`。
- 退款单写 `tickets`，`ticket_type='退款'`；不自动建单、不自动提交退款。
- 置信度闸在 Agent 之前；知识类和退款类都过闸；`infra_error`/`rerank_failed` 不写低置信度池。
- 前端 Vibe Coding：不套 brainstorm/TDD/code review；按 spec §11 的交互效果直接做，浏览器验收。
- 纯 Prompt/数据任务用标注集/评估集验证，不用 TDD；解析失败/越界必须安全兜底。
- 涉及 LangGraph/FastAPI/SQLAlchemy 的具体 API，动手前用 Context7 复核；版本以本机安装为准。
- 每个任务结束前更新 `dev-notes/ch06.md`（关键原话、产出、纠偏、翻车），再提交；不许收尾一次性补。
- 旧 `POST /chat`（legacy 单轮）保持可用；旧 `/ch05` 链接 307 到 `/ch06`。

## Review Focus

1. **订单选择器 pending 时又发新消息**：`/ch06/chat` 必须重新发 `interrupt` 帧、不写用户消息、不跑新回合；否则线程卡死或重复分类。测试落在 Task 15/16。
2. **无 pending 的 `/ch06/chat/resume`**：必须 409、不调用 `Command(resume=...)`、不写审计；否则会恢复一个不存在的中断。测试落在 Task 16。
3. **退款表单非法原因/非本人目录订单**：必须 422、不写 tickets；否则会落脏工单。测试落在 Task 16。
4. **coref/intent/expand 的 LLM 返回空/非法 JSON**：coref 回退原文、intent 回退「其他」、expand 回退单 query；不能 500。测试落在 Task 4/6/8。
5. **退款多 query 中任一 rerank 失败**：整趟按 `rerank_failed` 保守兜底，不进 Agent、不出退款表单、不写低置信度池。测试落在 Task 13。

---

### Task 1: Interrupt/resume 红线冒烟

**Files:**
- Create: `tests/test_interrupt_smoke.py`

**Interfaces:**
- Consumes: `langgraph.checkpoint.memory.InMemorySaver`、`langgraph.types.Command`、`langgraph.types.interrupt`
- Produces: 确认 1.2.20 的实测行为：`interrupt` 后 `get_state(config).interrupts` 可读；`Command(resume=...)` 续跑；被中断节点从头重跑。

- [ ] **Step 1: 写冒烟测试**

```python
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class CounterState(TypedDict, total=False):
    selected: str


def test_interrupt_surfaces_and_resumes_from_node_start():
    calls: list[str] = []

    def gate(state: CounterState) -> dict:
        calls.append("pre")
        selected = interrupt({"kind": "select_order", "orders": ["1001"]})
        calls.append("post")
        return {"selected": selected}

    graph = StateGraph(CounterState)
    graph.add_node("gate", gate)
    graph.add_edge(START, "gate")
    graph.add_edge("gate", END)
    compiled = graph.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "smoke"}}

    list(compiled.stream({}, config, stream_mode="custom"))
    snap = compiled.get_state(config)
    assert snap.interrupts
    assert snap.interrupts[0].value["kind"] == "select_order"

    list(compiled.stream(Command(resume="1001"), config, stream_mode="custom"))
    assert compiled.get_state(config).values["selected"] == "1001"
    assert calls == ["pre", "pre", "post"]
```

- [ ] **Step 2: 运行冒烟**

Run: `.venv/Scripts/python.exe -m pytest tests/test_interrupt_smoke.py -q`
Expected: PASS。若失败，**停**，把实际 `stream`/`get_state` 形状写进 dev-notes 并问用户，不继续。

- [ ] **Step 3: 更新 dev-notes 并提交**

```bash
git add tests/test_interrupt_smoke.py dev-notes/ch06.md
git commit -m "test: pin langgraph interrupt/resume surface"
```

---

### Task 2: State、五出口、配置种子

**Files:**
- Modify: `app/agent/state.py`
- Modify: `app/agent/routing.py`
- Modify: `app/config.py`
- Test: `tests/test_agent_routing.py`、`tests/test_ch06_config.py`

**Interfaces:**
- Produces:
  - `Intent = Literal["物流","订单","商品咨询","退款退货","售后","投诉","闲聊","其他"]`
  - `Route = Literal["escalate","fallback_script","knowledge","refund_flow","business"]`
  - `CustomerServiceState` 新字段：`resolved_query`、`intent_confidence`、`session_id`、`order_id`、`order_data`、`expanded_queries`；移除 `resolved_message`。
  - `AgentState` 新字段：`order_data`。
  - `INTENT_ROUTES: dict[Intent, Route]`；`route_for_intent(intent: str) -> Route`。
  - `Settings.intent_model=""`、`intent_small_model=""`、`intent_mode: Literal["accuracy","cost"]="accuracy"`、`intent_conf_threshold=0.7`、`ch06_checkpoint_db=".data/ch06-checkpoints.sqlite"`。
  - `build_llm(settings, temperature=0.7, model: str | None = None)`。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_agent_routing.py
def test_intent_routes_cover_eight_intents():
    assert INTENT_ROUTES == {
        "物流": "business",
        "订单": "business",
        "商品咨询": "knowledge",
        "退款退货": "refund_flow",
        "售后": "refund_flow",
        "投诉": "escalate",
        "闲聊": "fallback_script",
        "其他": "fallback_script",
    }


def test_unknown_intent_falls_back_to_business():
    assert route_for_intent("未知") == "business"
```

```python
# tests/test_ch06_config.py
from app.config import Settings, build_llm


def test_ch06_settings_defaults():
    s = Settings(_env_file=None, openai_api_key="x")
    assert s.intent_model == ""
    assert s.intent_small_model == ""
    assert s.intent_mode == "accuracy"
    assert s.intent_conf_threshold == 0.7
    assert s.ch06_checkpoint_db == ".data/ch06-checkpoints.sqlite"


def test_build_llm_model_override():
    s = Settings(_env_file=None, openai_api_key="x", model_name="base")
    llm = build_llm(s, model="intent-model")
    assert llm.model_name == "intent-model"
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_routing.py tests/test_ch06_config.py -q`
Expected: FAIL（旧映射仍是 7 类/4 出口；`Settings` 无新字段；`build_llm` 不接受 `model`）。

- [ ] **Step 3: 实现**

按 **Interfaces** 改 `state.py`、`routing.py`、`config.py`。`build_llm` 的 `model` 为空时用 `settings.model_name`。

- [ ] **Step 4: 运行确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_routing.py tests/test_ch06_config.py -q`
Expected: PASS。

- [ ] **Step 5: 更新 dev-notes 并提交**

```bash
git add app/agent/state.py app/agent/routing.py app/config.py tests/test_agent_routing.py tests/test_ch06_config.py dev-notes/ch06.md
git commit -m "feat: add ch06 state, five routes and intent config seeds"
```

---

### Task 3: 演示订单目录与退款工单枚举

**Files:**
- Modify: `app/tools/business.py`
- Modify: `app/tools/ticket.py`
- Modify: `app/models/ticket.py`
- Modify: `app/schemas.py`（只改 `TicketRequest`）
- Create: `sql/ch06_ticket_type.sql`
- Test: `tests/test_tools_business.py`、`tests/test_tools_ticket.py`、`tests/test_models.py`、`tests/test_schemas.py`

**Interfaces:**
- Produces:
  - `DEMO_ORDERS: dict[str, dict]`，键 `1001`/`1002`/`1003`；每单含 `order_id, items, amount, status, tracking_no, created_at, signed_at`。
  - `list_user_orders(session_id: str) -> list[dict]`，返回深拷贝的三张卡。
  - `query_order` 对 1001/1002/1003 返回 `DEMO_ORDERS` 同一份数据；其他 ID 保持随机，但输出也含 `created_at`/`signed_at`。
  - `create_ticket(... ticket_type: Literal["售后","投诉","咨询","退款"])`。
  - `Ticket.ticket_type` 枚举加 `退款`；`TicketRequest.ticket_type` 加 `退款`。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_tools_business.py
import json
from app.tools.business import list_user_orders, query_order


def test_list_user_orders_returns_three_demo_orders():
    orders = list_user_orders("s1")
    assert [o["order_id"] for o in orders] == ["1001", "1002", "1003"]
    assert orders[0]["status"] == "已签收"
    assert orders[0]["signed_at"]


def test_query_order_matches_demo_order():
    payload = json.loads(query_order.invoke({"order_id": "1001"}))
    demo = list_user_orders("s1")[0]
    assert payload["order_id"] == "1001"
    assert payload["status"] == demo["status"]
    assert payload["signed_at"] == demo["signed_at"]


def test_query_order_unknown_id_keeps_contract():
    payload = json.loads(query_order.invoke({"order_id": "9999"}))
    assert payload["order_id"] == "9999"
    for key in ("items", "amount", "status", "tracking_no", "created_at", "signed_at"):
        assert key in payload


def test_list_user_orders_returns_copies():
    first = list_user_orders("s1")
    first[0]["status"] = "mutated"
    assert list_user_orders("s1")[0]["status"] != "mutated"
```

```python
# tests/test_models.py（追加）
def test_ticket_enum_includes_refund():
    from app.models import Ticket
    assert Ticket.__table__.c.ticket_type.type.enums == ["售后", "投诉", "咨询", "退款"]
```

```python
# tests/test_schemas.py（追加）
def test_ticket_request_accepts_refund():
    from app.schemas import TicketRequest
    req = TicketRequest(session_id="s", description="x", ticket_type="退款")
    assert req.ticket_type == "退款"
```

`tests/test_tools_ticket.py` 追加一条参数化 refund 用例，复用现有 `session` + `current_conversation_id` 的测试写法，断言返回 `T...` 且 `Ticket.ticket_type == "退款"`。

- [ ] **Step 2: 运行确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tools_business.py tests/test_tools_ticket.py tests/test_models.py tests/test_schemas.py -q`
Expected: FAIL（无 `list_user_orders`、无日期字段、枚举无 `退款`）。

- [ ] **Step 3: 实现**

- `business.py`：`DEMO_ORDERS` 用 `datetime.now()` 相对日期；1001 `已签收/签收 3 天前`，1002 `运输中/signed_at=None`，1003 `待发货/signed_at=None`。`query_order` 先查目录；随机分支补 `created_at`/`signed_at`。`list_user_orders` 用 `copy.deepcopy`。
- `ticket.py` / `models/ticket.py` / `schemas.py`：加 `退款`。
- `sql/ch06_ticket_type.sql`：

```sql
ALTER TABLE tickets
  MODIFY COLUMN ticket_type ENUM('售后','投诉','咨询','退款') NOT NULL;
```

- [ ] **Step 4: 运行确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tools_business.py tests/test_tools_ticket.py tests/test_models.py tests/test_schemas.py -q`
Expected: PASS。

- [ ] **Step 5: 更新 dev-notes 并提交**

```bash
git add app/tools/business.py app/tools/ticket.py app/models/ticket.py app/schemas.py sql/ch06_ticket_type.sql tests/test_tools_business.py tests/test_tools_ticket.py tests/test_models.py tests/test_schemas.py dev-notes/ch06.md
git commit -m "feat: add demo orders and refund ticket type"
```

---

### Task 4: 指代消解模块与历史渲染

**Files:**
- Create: `app/agent/coref.py`
- Modify: `app/services/context.py`
- Test: `tests/test_agent_coref.py`、`tests/test_context.py`

**Interfaces:**
- Produces:
  - `class CorefRewrite(BaseModel): resolved_query: str`
  - `COREF_SYSTEM_PROMPT: str`
  - `def resolve_query(llm, message: str, history: Sequence[BaseMessage] = ()) -> str`
  - `def render_recent_dialogue(messages: Sequence[BaseMessage], max_chars: int = 1200) -> str`
- `resolve_query`：结构化输出；异常/空串返回 `message`；历史渲染进 HumanMessage。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_agent_coref.py
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from app.agent.coref import COREF_SYSTEM_PROMPT, CorefRewrite, resolve_query
from app.services.context import render_recent_dialogue


class FakeStructured:
    def __init__(self, result=None, error=None):
        self.result, self.error, self.messages = result, error, []

    def with_structured_output(self, schema, method=None):
        return self

    def invoke(self, messages):
        self.messages.extend(messages)
        if self.error:
            raise self.error
        return self.result


def test_render_recent_dialogue_roles_and_order_marker():
    text = render_recent_dialogue([
        HumanMessage(content="订单 1001 的物流到哪了"),
        AIMessage(content="已签收"),
        SystemMessage(content="[订单选择器] 用户选择了订单 1001"),
        ToolMessage(content="TOOL-SHOULD-NOT-APPEAR", tool_call_id="c1"),
    ])
    assert "订单 1001 的物流到哪了" in text
    assert "已签收" in text
    assert "[订单选择器]" in text
    assert "TOOL-SHOULD-NOT-APPEAR" not in text


def test_resolve_query_returns_llm_output():
    llm = FakeStructured(result=CorefRewrite(resolved_query="订单 1001 的商品能退吗"))
    assert resolve_query(llm, "它能退吗", [HumanMessage(content="订单 1001")]) == "订单 1001 的商品能退吗"


def test_resolve_query_falls_back_on_error_and_blank():
    assert resolve_query(FakeStructured(error=ValueError("x")), "它能退吗") == "它能退吗"
    assert resolve_query(FakeStructured(result=CorefRewrite(resolved_query="  ")), "它能退吗") == "它能退吗"


def test_coref_prompt_pins_passthrough_and_entities():
    assert "原样" in COREF_SYSTEM_PROMPT
    assert "订单号" in COREF_SYSTEM_PROMPT
    assert "不要引入" in COREF_SYSTEM_PROMPT


def test_resolve_query_includes_history():
    llm = FakeStructured(result=CorefRewrite(resolved_query="x"))
    resolve_query(llm, "它能退吗", [HumanMessage(content="订单 1001")])
    assert "订单 1001" in llm.messages[-1].content
```

`tests/test_context.py` 追加预算裁剪用例：超长历史时旧消息被丢弃、最新消息保留。

- [ ] **Step 2: 运行确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_coref.py tests/test_context.py -q`
Expected: FAIL（模块/函数不存在）。

- [ ] **Step 3: 实现**

`coref.py` 写 prompt（规则见 spec §7.1）；`render_recent_dialogue` 从新到旧取 Human/AI/System 文本，跳过 Tool，按字符预算，返回正序 `用户：…\n客服：…`。`resolve_query` 用 `with_structured_output(CorefRewrite, method="json_mode")`；异常/空白回退原文。

- [ ] **Step 4: 运行确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_coref.py tests/test_context.py -q`
Expected: PASS。

- [ ] **Step 5: 更新 dev-notes 并提交**

```bash
git add app/agent/coref.py app/services/context.py tests/test_agent_coref.py tests/test_context.py dev-notes/ch06.md
git commit -m "feat: add coreference and query rewrite module"
```

---

### Task 5: Coref 评估集与真实 LLM 验证

**Files:**
- Create: `tests/eval/ch06_coref_cases.json`
- Create: `app/jobs/eval_coref.py`
- Test: `tests/test_eval_coref.py`
- Report: `docs/superpowers/reports/2026-11-18-ch06-coref-eval.md`（脚本生成）

**Interfaces:**
- Produces:
  - `evaluate_case(case: dict, resolved: str) -> dict` → `{"id","passed","reason"}`
  - `summarize(results: list[dict]) -> dict` → `{"total","passed","accuracy","failures"}`
  - `load_cases(path: Path) -> list[dict]`
  - `run_cases(cases, llm) -> list[dict]`
  - `main() -> None`

- [ ] **Step 1: 写数据集（≥10 条）**

每条：`id`、`history`（`[{"role":"user|assistant","content":...}]`）、`message`、`contains`/`not_contains` 或 `passthrough: true`。至少覆盖：
- 多轮「它能退吗」→ 含订单号 1001、不含「它」；
- 商品指代「这个还有别的颜色吗」→ 含商品名；
- 已完整透传「订单 1001 什么时候发货」→ `passthrough: true`；
- 无历史口语归一「邮费怎么算」→ 含「运费」；
- 订单选择器标记后的「这个能退吗」→ 含 1001。

- [ ] **Step 2: 写纯函数测试**

```python
# tests/test_eval_coref.py
from app.jobs.eval_coref import evaluate_case, summarize


def test_evaluate_case_contains_and_passthrough():
    assert evaluate_case({"id": "C1", "contains": ["1001"], "not_contains": ["它"]}, "订单 1001 能退吗")["passed"]
    assert not evaluate_case({"id": "C2", "contains": ["1001"]}, "能退吗")["passed"]
    assert evaluate_case({"id": "C3", "passthrough": True, "message": "运费怎么算"}, "运费怎么算")["passed"]
    assert not evaluate_case({"id": "C4", "passthrough": True, "message": "运费怎么算"}, "运费如何计算")["passed"]


def test_summarize_counts():
    out = summarize([{"passed": True}, {"passed": False}])
    assert out["total"] == 2 and out["passed"] == 1 and out["accuracy"] == 0.5
```

- [ ] **Step 3: 实现脚本并跑真实评估**

`eval_coref.py` 的 `main()` 构建 `build_llm(settings, temperature=0.0, model=settings.intent_model or None)`，加载数据集，把 history 转成 `HumanMessage`/`AIMessage`，调用 `resolve_query`，写报告。

Run: `.venv/Scripts/python.exe -m app.jobs.eval_coref`
Expected: 报告生成；通过率 ≥90%，透传用例 100%。低于则调整 `COREF_SYSTEM_PROMPT` 后重跑，并把调整记录进报告/dev-notes。

- [ ] **Step 4: 运行纯函数测试**

Run: `.venv/Scripts/python.exe -m pytest tests/test_eval_coref.py -q`
Expected: PASS。

- [ ] **Step 5: 更新 dev-notes 并提交**

```bash
git add tests/eval/ch06_coref_cases.json app/jobs/eval_coref.py tests/test_eval_coref.py docs/superpowers/reports/2026-11-18-ch06-coref-eval.md dev-notes/ch06.md
git commit -m "test: add ch06 coref evaluation set and report"
```

---

### Task 6: 八类意图四件套

**Files:**
- Modify: `app/agent/intent.py`
- Test: `tests/test_agent_intent.py`

**Interfaces:**
- Produces:
  - `class IntentResult(BaseModel): intent: Intent; confidence: float = 0.0`
  - `INTENT_SYSTEM_PROMPT: str`
  - `def classify_intent(llm, message: str, history: Sequence[BaseMessage] = (), *, raise_on_error: bool = False) -> IntentResult`
- 异常/越界/空输出 → `IntentResult(intent="其他", confidence=0.0)`；confidence 夹到 `[0,1]`。`raise_on_error=True` 时改为抛出，供 eval 区分“真实其他”和“网络/解析失败”。

- [ ] **Step 1: 写失败测试**

```python
from app.agent.intent import INTENT_SYSTEM_PROMPT, IntentResult, classify_intent


class FakeStructured:
    def __init__(self, result=None, error=None):
        self.result, self.error, self.messages = result, error, []

    def with_structured_output(self, schema, method=None):
        return self

    def invoke(self, messages):
        self.messages.extend(messages)
        if self.error:
            raise self.error
        return self.result


def test_prompt_has_eight_labels_and_json_fields():
    for label in ["物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "闲聊", "其他"]:
        assert label in INTENT_SYSTEM_PROMPT
    assert "confidence" in INTENT_SYSTEM_PROMPT
    assert "只输出" in INTENT_SYSTEM_PROMPT and "JSON" in INTENT_SYSTEM_PROMPT
    assert "reason" not in INTENT_SYSTEM_PROMPT


def test_prompt_has_boundary_examples():
    for text in ["这个还能退吗", "刚收到就坏了", "我要投诉", "你好，在吗", "帮我写一首诗"]:
        assert text in INTENT_SYSTEM_PROMPT


def test_classify_returns_intent_and_clamps_confidence():
    assert classify_intent(FakeStructured(IntentResult(intent="退款退货", confidence=0.91)), "x").confidence == 0.91
    assert classify_intent(FakeStructured(IntentResult(intent="物流", confidence=1.5)), "x").confidence == 1.0
    assert classify_intent(FakeStructured(IntentResult(intent="物流", confidence=-0.2)), "x").confidence == 0.0


def test_classify_falls_back_to_other():
    out = classify_intent(FakeStructured(error=ValueError("bad json")), "x")
    assert out.intent == "其他" and out.confidence == 0.0


def test_classify_can_raise_for_eval():
    import pytest
    with pytest.raises(ValueError):
        classify_intent(FakeStructured(error=ValueError("bad json")), "x", raise_on_error=True)


def test_classify_passes_history():
    llm = FakeStructured(IntentResult(intent="物流", confidence=0.8))
    classify_intent(llm, "那它到哪了", [("user", "订单 1001 的物流")])
    assert "订单 1001 的物流" in llm.messages[-1].content
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_intent.py -q`
Expected: FAIL（旧七类、无 confidence、回退售后）。

- [ ] **Step 3: 实现**

按 spec §7.2 写八类选择题、边界 few-shot、JSON 字段、其他兜底；`classify_intent` 用 `with_structured_output(IntentResult, method="json_mode")`；校验 intent 在八类内，confidence 夹取；异常/空/越界回退「其他」。

- [ ] **Step 4: 运行确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_intent.py -q`
Expected: PASS。

- [ ] **Step 5: 更新 dev-notes 并提交**

```bash
git add app/agent/intent.py tests/test_agent_intent.py dev-notes/ch06.md
git commit -m "feat: rewrite intent classifier with eight classes and confidence"
```

---

### Task 7: 意图评估集与真实 LLM 验证

**Files:**
- Create: `tests/eval/ch06_intent_cases.json`
- Modify: `app/jobs/eval_intent.py`（默认路径、8 类、history、报告路径）
- Modify: `tests/test_intent_cases.py`
- Report: `docs/superpowers/reports/2026-11-18-ch06-intent-eval.md`

**Interfaces:**
- Consumes: `classify_intent(llm, message, history)`、`IntentResult.intent/confidence`
- Produces: `_LABEL_ORDER` 8 类；`run_cases` 支持 `row.get("history", [])`；报告含总体/每类/混淆矩阵/JSON fallback 列表。

- [ ] **Step 1: 写数据集（≥56 条）**

八类各 6 条 + 8 条边界/其他 + 6 条多轮漂移。`history` 可选。不得复用 prompt 里的「」few-shot 原文。边界至少覆盖：
- 「这个还能退吗」→ 退款退货；
- 「刚收到就坏了，怎么保修」→ 售后；
- 「我要投诉，顺便把钱退了」→ 投诉；
- 「你好，在吗」→ 闲聊；
- 「帮我写一首诗」→ 其他。

- [ ] **Step 2: 更新数据集校验测试**

```python
def test_intent_cases_cover_eight_classes():
    rows = json.loads(CASES.read_text(encoding="utf-8"))
    by_intent = {}
    for row in rows:
        by_intent[row["intent"]] = by_intent.get(row["intent"], 0) + 1
    assert set(by_intent) == {"物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "闲聊", "其他"}
    assert all(count >= 6 for count in by_intent.values())


def test_intent_cases_do_not_reuse_prompt_examples():
    rows = json.loads(CASES.read_text(encoding="utf-8"))
    for row in rows:
        for example in PROMPT_EXAMPLES:
            assert example not in row["text"]
```

- [ ] **Step 3: 更新 eval 脚本并跑真实评估**

`app/jobs/eval_intent.py`：`_DEFAULT_CASES` 指向 `ch06_intent_cases.json`；`_LABEL_ORDER` 8 类；`run_cases` 调用 `classify_intent(llm, row["text"], row.get("history", []), raise_on_error=True)`；捕获异常记为 `error`，不计入正确率并在报告单列；`intent="其他"` 是合法预测，不自动当 fallback。

Run: `.venv/Scripts/python.exe -m app.jobs.eval_intent`
Expected: 总体 ≥90%、每类 ≥80%、无 `error`；低于则调整 prompt 后重跑，记录。

- [ ] **Step 4: 运行校验测试**

Run: `.venv/Scripts/python.exe -m pytest tests/test_intent_cases.py -q`
Expected: PASS。

- [ ] **Step 5: 更新 dev-notes 并提交**

```bash
git add tests/eval/ch06_intent_cases.json app/jobs/eval_intent.py tests/test_intent_cases.py docs/superpowers/reports/2026-11-18-ch06-intent-eval.md dev-notes/ch06.md
git commit -m "test: add ch06 intent evaluation set and report"
```

---

### Task 8: Query 扩写模块

**Files:**
- Modify: `app/rag/query_understanding.py`
- Test: `tests/test_agent_expand.py`

**Interfaces:**
- Produces:
  - `class QueryExpansion(BaseModel): queries: list[str]`
  - `QUERY_EXPANSION_SYSTEM_PROMPT: str`
  - `def expand_queries(llm, query: str, order_data: dict | None = None) -> list[str]`
- 合法 3 条去重返回；数量≠3/异常/空 → `[query]`。

- [ ] **Step 1: 写失败测试**

```python
from app.rag.query_understanding import QueryExpansion, expand_queries


class FakeStructured:
    def __init__(self, result=None, error=None):
        self.result, self.error, self.messages = result, error, []

    def with_structured_output(self, schema, method=None):
        return self

    def invoke(self, messages):
        self.messages.extend(messages)
        if self.error:
            raise self.error
        return self.result


def test_expand_returns_three_unique_queries():
    llm = FakeStructured(QueryExpansion(queries=["a", "b", "c"]))
    assert expand_queries(llm, "订单 1001 能退吗", {"order_id": "1001"}) == ["a", "b", "c"]


def test_expand_falls_back_when_not_three():
    for queries in (["a", "b"], ["a", "b", "c", "d"], ["a", "a", "b"]):
        assert expand_queries(FakeStructured(QueryExpansion(queries=queries)), "q") == ["q"]


def test_expand_falls_back_on_error():
    assert expand_queries(FakeStructured(error=ValueError("x")), "q") == ["q"]


def test_expand_prompt_mentions_queries_and_three():
    from app.rag.query_understanding import QUERY_EXPANSION_SYSTEM_PROMPT
    assert "queries" in QUERY_EXPANSION_SYSTEM_PROMPT
    assert "3" in QUERY_EXPANSION_SYSTEM_PROMPT
    assert "JSON" in QUERY_EXPANSION_SYSTEM_PROMPT


def test_expand_includes_order_data():
    llm = FakeStructured(QueryExpansion(queries=["a", "b", "c"]))
    expand_queries(llm, "能退吗", {"order_id": "1001", "status": "已签收"})
    assert "1001" in llm.messages[-1].content and "已签收" in llm.messages[-1].content
```

（`QueryExpansion`/`expand_queries` 放 `app/rag/query_understanding.py`；测试从该模块导入。）

- [ ] **Step 2: 运行确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_expand.py -q`
Expected: FAIL（函数不存在）。

- [ ] **Step 3: 实现**

写扩写 prompt（严格 3 条、侧重点不同、保留订单号/商品/型号、不引入新参数、只输出 `{"queries":[...]}`）；用 `with_structured_output(QueryExpansion, method="json_mode")`；清洗/去重；数量≠3 或异常回退 `[query]`。

- [ ] **Step 4: 运行确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_expand.py -q`
Expected: PASS。

- [ ] **Step 5: 更新 dev-notes 并提交**

```bash
git add app/rag/query_understanding.py tests/test_agent_expand.py dev-notes/ch06.md
git commit -m "feat: add three-query refund expansion"
```

---

### Task 9: 扩写评估集与真实 LLM 验证

**Files:**
- Create: `tests/eval/ch06_expand_cases.json`
- Create: `app/jobs/eval_expand.py`
- Test: `tests/test_eval_expand.py`
- Report: `docs/superpowers/reports/2026-11-18-ch06-expand-eval.md`

**Interfaces:**
- Produces: `evaluate_case(case, queries) -> dict`、`summarize(results) -> dict`、`load_cases`、`run_cases`、`main()`。
- 检查：长度 3、去重、非空、`must_include` 至少出现在一条 query。

- [ ] **Step 1: 写数据集（≥6 条）**

每条：`id`、`query`、`order_data`、`must_include`（订单号/商品名）。覆盖「已签收能退吗」「质量有问题想退」「运输中想取消」「售后维修」等。

- [ ] **Step 2: 写纯函数测试**

```python
# tests/test_eval_expand.py
from app.jobs.eval_expand import evaluate_case, summarize


def test_evaluate_case_checks_three_and_entities():
    case = {"id": "E1", "must_include": ["1001"]}
    assert evaluate_case(case, ["订单 1001 退货政策", "退货条件", "退款流程"])["passed"]
    assert not evaluate_case(case, ["只有一条"])["passed"]
    assert not evaluate_case(case, ["退货条件", "退款流程", "其他"])["passed"]


def test_summarize_counts():
    out = summarize([{"passed": True}, {"passed": False}])
    assert out["total"] == 2 and out["passed"] == 1
```

- [ ] **Step 3: 实现脚本并跑真实评估**

Run: `.venv/Scripts/python.exe -m app.jobs.eval_expand`
Expected: 3 条率 ≥90%，关键实体保留；低于则调 prompt 重跑并记录。

- [ ] **Step 4: 运行纯函数测试**

Run: `.venv/Scripts/python.exe -m pytest tests/test_eval_expand.py -q`
Expected: PASS。

- [ ] **Step 5: 更新 dev-notes 并提交**

```bash
git add tests/eval/ch06_expand_cases.json app/jobs/eval_expand.py tests/test_eval_expand.py docs/superpowers/reports/2026-11-18-ch06-expand-eval.md dev-notes/ch06.md
git commit -m "test: add ch06 expansion evaluation set and report"
```

---

### Task 10: Agent 工具、退款拦截、售后 prompt

**Files:**
- Modify: `app/agent/tools.py`
- Modify: `app/agent/subgraph.py`
- Modify: `app/prompts/customer_service.py`
- Test: `tests/test_agent_tools.py`、`tests/test_agent_subgraph.py`、`tests/test_prompt.py`

**Interfaces:**
- Produces:
  - `@tool def submit_refund(order_id: str) -> str`
  - `GENERAL_AGENT_TOOLS`（业务三件 + `suggest_handoff`）、`REFUND_AGENT_TOOLS`（`submit_refund` + `suggest_handoff`）
  - `AGENT_TOOLS_BY_NAME: dict[str, BaseTool]`
  - `def tools_for_route(route: str) -> list[BaseTool]`
  - `def suggested_actions_from_refund(order_id: str) -> list[dict]`
  - `def render_after_sales_agent_system_prompt(business_name="本店", order_data=None, evidence=None, intent="") -> str`
- `tools_node` 拦截 `submit_refund`：校验 `order_data["order_id"]`；写 `{"action":"refund_form","label":"提交退款工单","order_id":...}`；不调 `execute_tool`。
- `agent_node` 按 `state["route"]` 选 prompt/工具：`refund_flow` 用售后 prompt + 退款工具；其他用通用 prompt + 通用工具。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_agent_tools.py（追加）
def test_tools_for_route_splits_refund_and_business():
    from app.agent.tools import tools_for_route
    assert [t.name for t in tools_for_route("refund_flow")] == ["submit_refund", "suggest_handoff"]
    assert [t.name for t in tools_for_route("business")] == [
        "query_order", "query_product", "query_logistics", "suggest_handoff"
    ]
```

```python
# tests/test_agent_subgraph.py（追加）
def test_submit_refund_is_intercepted_without_execute_tool(monkeypatch):
    import app.agent.subgraph as sg
    monkeypatch.setattr(
        sg, "execute_tool",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not execute")),
    )
    llm = ScriptedLLM([
        tool_chunk("submit_refund", '{"order_id": "1001"}'),
        text_chunk("可以退，请填写退款原因。"),
    ])
    graph = build_agent_subgraph(llm)
    result = graph.invoke({
        "messages": [HumanMessage(content="能退吗")],
        "route": "refund_flow",
        "order_data": {"order_id": "1001", "status": "已签收"},
        "suggested_actions": [],
        "token_estimate": 0,
        "conversation_id": 1,
    })
    assert result["suggested_actions"] == [
        {"action": "refund_form", "label": "提交退款工单", "order_id": "1001"}
    ]


def test_refund_route_binds_after_sales_tools_and_prompt():
    llm = ScriptedLLM([text_chunk("可以退")])
    graph = build_agent_subgraph(llm)
    graph.invoke({
        "messages": [HumanMessage(content="能退吗")],
        "route": "refund_flow",
        "order_data": {"order_id": "1001", "status": "已签收"},
        "evidence": [],
        "suggested_actions": [],
        "token_estimate": 0,
        "conversation_id": 1,
    })
    assert llm.bound_tool_names == ["submit_refund", "suggest_handoff"]


def test_after_sales_prompt_has_order_and_repair_rules():
    from app.prompts.customer_service import render_after_sales_agent_system_prompt
    prompt = render_after_sales_agent_system_prompt(
        order_data={"order_id": "1001", "status": "已签收"},
        evidence=[],
        intent="售后",
    )
    assert "1001" in prompt
    assert "不调用 submit_refund" in prompt
    assert "退款表单" in prompt
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_tools.py tests/test_agent_subgraph.py tests/test_prompt.py -q`
Expected: FAIL（无 `submit_refund`、无用例级工具选择、无售后 prompt）。

- [ ] **Step 3: 实现**

- `tools.py`：`submit_refund` 只返回 JSON 确认；`tools_for_route` 返回工具子集；`suggested_actions_from_refund` 生成动作。
- `subgraph.py`：`agent_node` 按 route 选 prompt/工具；`tools_node` 识别 `submit_refund`（在 `suggest_handoff` 之后、业务工具之前），校验订单号，合并 `suggested_actions`，不调 `execute_tool`；`_CONTROL_TOOL_LABELS["submit_refund"]="准备退款表单"`。
- `prompts/customer_service.py`：`render_after_sales_agent_system_prompt`，含订单块、证据块、退款/保修分支规则、禁止编造/禁止自动提交。

- [ ] **Step 4: 运行确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_tools.py tests/test_agent_subgraph.py tests/test_prompt.py -q`
Expected: PASS。

- [ ] **Step 5: 更新 dev-notes 并提交**

```bash
git add app/agent/tools.py app/agent/subgraph.py app/prompts/customer_service.py tests/test_agent_tools.py tests/test_agent_subgraph.py tests/test_prompt.py dev-notes/ch06.md
git commit -m "feat: add refund control tool and after-sales agent prompt"
```

---

### Task 11: 父图核心路由（不含退款细节）

**Files:**
- Modify: `app/agent/graph.py`
- Modify: `tests/agent_fakes.py`
- Test: `tests/test_agent_graph.py`

**Interfaces:**
- `build_graph(llm, intent_llm, *, checkpointer=None, settings=None)`。
- 节点：`resolve_reference`、`classify_intent`、`route_by_intent`、`retrieve_knowledge`、`confidence_gate`、`complaint_reply`、`script_reply`、`main_agent`、`record_log`；`fetch_order`/`retrieve_policy` 先放 `raise NotImplementedError` 的占位节点（Task 12/13 替换），边已连通。
- `route_by_intent` 写 `route`；条件边映射五出口。
- `resolve_reference` 调 `resolve_query(intent_llm, user_message, messages[:-1])`。
- `classify_intent` 调 `classify_intent(intent_llm, resolved_query, history)`。
- `retrieve_knowledge` 调 `retrieve_hybrid(resolved_query, analysis=QueryAnalysis(...))`。
- `record_log` 增加 `resolved_query/intent_confidence/order_id/expanded_queries/refund_form`。

- [ ] **Step 1: 更新 fakes 与写失败测试**

`tests/agent_fakes.py` 增加：

```python
class ScriptedStructuredLLM:
    def __init__(self, resolved: str, intent, queries=None):
        self.resolved = resolved
        self.intent = intent
        self.queries = queries or ["q1", "q2", "q3"]
        self.schema = None
        self.messages = []

    def with_structured_output(self, schema, method=None):
        self.schema = schema
        return self

    def invoke(self, messages):
        from app.agent.coref import CorefRewrite
        from app.rag.query_understanding import QueryExpansion
        self.messages.extend(messages)
        if self.schema is CorefRewrite:
            return CorefRewrite(resolved_query=self.resolved)
        if self.schema is QueryExpansion:
            return QueryExpansion(queries=self.queries)
        return self.intent
```

`tests/test_agent_graph.py` 更新 `_state`（`resolved_query`）和 `_build`：

```python
def _build(llm, intent, resolved=None, queries=None, **kwargs):
    intent_llm = ScriptedStructuredLLM(
        resolved or intent,
        IntentResult(intent=intent, confidence=0.9),
        queries=queries,
    )
    return build_graph(llm, intent_llm, checkpointer=InMemorySaver(), **kwargs)
```

新增/更新：

```python
def test_business_route_writes_resolved_query_and_intent(monkeypatch):
    ...
    assert result["resolved_query"] == "订单 1001 什么时候发货"
    assert result["intent"] == "物流"
    assert result["intent_confidence"] == 0.9


@pytest.mark.parametrize("intent,text", [("闲聊", CHITCHAT_TEXT), ("其他", OTHER_TEXT)])
def test_script_reply_is_fixed_and_does_not_call_llm(intent, text):
    graph = _build(ExplodingLLM(), intent)
    result = graph.invoke(_state("测试"), CONFIG)
    assert result["answer"] == text
    assert result["suggested_actions"] == []


def test_complaint_route_uses_escalate_buttons():
    ...


def test_knowledge_route_passes_analysis_and_gates(monkeypatch):
    calls = []
    def fake_retrieve(query, **kwargs):
        calls.append((query, kwargs.get("analysis")))
        return make_retrieval_result([make_chunk(0.9)])
    monkeypatch.setattr("app.agent.graph.retrieve_hybrid", fake_retrieve)
    graph = _build(
        ScriptedLLM([text_chunk("支持降噪")]),
        "商品咨询",
        resolved="这款耳机支持降噪吗",
    )
    result = graph.invoke(_state("这款耳机支持降噪吗"), CONFIG)
    assert calls[0][0] == "这款耳机支持降噪吗"
    assert calls[0][1].rewritten == "这款耳机支持降噪吗"
    assert result["gate_passed"] is True


def test_record_log_has_ch06_fields(caplog):
    ...
    assert "resolved_query" in payload
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_graph.py -q`
Expected: FAIL（旧节点/旧 state/旧 fakes）。

- [ ] **Step 3: 实现**

按 **Interfaces** 改 `graph.py`：节点、边、五出口、script/complaint、main_agent 包装；`fetch_order`/`retrieve_policy` 先写占位节点（`raise NotImplementedError`，只有退款路由会走到）。`retrieve_knowledge` 用 `analysis` 跳过二次改写。`record_log` 加字段。

- [ ] **Step 4: 运行确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_graph.py -q`
Expected: PASS（退款路径的占位节点不被核心测试触发）。

- [ ] **Step 5: 更新 dev-notes 并提交**

```bash
git add app/agent/graph.py tests/agent_fakes.py tests/test_agent_graph.py dev-notes/ch06.md
git commit -m "feat: wire ch06 core routes into parent graph"
```

---

### Task 12: `fetch_order` 槽位与 interrupt/resume

**Files:**
- Modify: `app/agent/graph.py`
- Test: `tests/test_agent_refund_flow.py`

**Interfaces:**
- `def extract_order_id(text: str) -> str | None`
- `def fetch_order_node(state) -> dict`
- 行为：只从 `resolved_query` 抽号；目录内才调用 `query_order`；目录外/缺失 → `interrupt({"kind":"select_order","slot":"order_id","orders":[...]})`；resume 合法 → 写 `order_id`、`order_data`、`SystemMessage("[订单选择器] 用户选择了订单 <id>")`；非法 → `ORDER_INVALID_TEXT` + `AIMessage` + `gate_reason="order_slot_failed"`；工具失败 → `ORDER_FAILED_TEXT` + `gate_reason="order_fetch_failed"`。
- 本任务的边先接 `refund_flow → fetch_order → record_log`（成功/失败都到 `record_log`），这样 `fetch_order` 可以独立测试；Task 13 再把成功边改到 `retrieve_policy`。

- [ ] **Step 1: 写失败测试**

```python
def test_extract_order_id_patterns():
    assert extract_order_id("订单 1001 能退吗") == "1001"
    assert extract_order_id("订单号：1002") == "1002"
    assert extract_order_id("1003号订单") == "1003"
    assert extract_order_id("金额 299 元") is None
    assert extract_order_id("2026年能退吗") is None


def test_fetch_order_rejects_unknown_id_and_interrupts(monkeypatch):
    calls = []
    monkeypatch.setattr("app.agent.graph.execute_tool", lambda *a, **k: calls.append(a) or "{}")
    # 用图跑到 interrupt，断言 interrupts payload 含 1001/1002/1003，calls == []
    ...


def test_fetch_order_resume_fills_slot_and_adds_system_message(monkeypatch):
    # 先跑 "能退吗" → interrupt；再 Command(resume="1001")
    # 断言 order_id=="1001"、query_order 只一次、messages 含 SystemMessage 标记
    ...


def test_fetch_order_invalid_resume_falls_back(monkeypatch):
    # resume "9999" → answer ORDER_INVALID_TEXT，无 order_data
    ...
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_refund_flow.py -q`
Expected: FAIL（`NotImplementedError`/无函数）。

- [ ] **Step 3: 实现**

按 **Interfaces** 写 `extract_order_id` 与 `fetch_order_node`；边：`refund_flow → fetch_order → record_log`。`interrupt()` 前只做正则/`list_user_orders`/日志。

- [ ] **Step 4: 运行确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_refund_flow.py -q`
Expected: PASS。

- [ ] **Step 5: 更新 dev-notes 并提交**

```bash
git add app/agent/graph.py tests/test_agent_refund_flow.py dev-notes/ch06.md
git commit -m "feat: add fetch_order slot with interrupt/resume"
```

---

### Task 13: `retrieve_policy` 多 query 检索与合并

**Files:**
- Modify: `app/agent/graph.py`
- Test: `tests/test_agent_refund_flow.py`

**Interfaces:**
- `def retrieve_policy_node(state) -> dict`
- 行为：`expand_queries(intent_llm, resolved_query, order_data)`；每条 query 调 `retrieve_hybrid(query, analysis=QueryAnalysis(...))`；任一异常 → `infra_error`；任一 `rerank_failed` → `rerank_failed`；按 `chunk_id` 去重保留最高分，降序取 `settings.rerank_top_n`；`order_for_prompt` + `CitationCollector` + `assess_confidence`；写 `expanded_queries`。

- [ ] **Step 1: 写失败测试**

测试状态用 `resolved_query="订单 1001 能退吗"`，并 monkeypatch `execute_tool` 返回演示订单，避免进入 interrupt；`intent_llm` 的 `queries` 用 `["q1","q2","q3"]`。

```python
def test_retrieve_policy_runs_three_queries_and_merges(monkeypatch):
    ...
    assert [c[0] for c in calls] == ["q1", "q2", "q3"]
    assert all(c[1] is not None for c in calls)  # analysis 已传，跳过二次改写


def test_retrieve_policy_dedupes_and_keeps_max_score(monkeypatch):
    # chunk_id 1 在 q1=0.4、q2=0.8 → 合并后一条 0.8
    ...


def test_retrieve_policy_infra_error(monkeypatch):
    ...


def test_retrieve_policy_rerank_failure(monkeypatch):
    ...


def test_retrieve_policy_low_score_fails_gate(monkeypatch):
    ...
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_refund_flow.py -q`
Expected: FAIL（占位节点）。

- [ ] **Step 3: 实现**

按 **Interfaces** 写 `retrieve_policy_node`；边 `fetch_order` 成功 → `retrieve_policy → confidence_gate`；gate 通过边先保留到 `record_log`（Task 14 改为 `main_agent`）。日志 `ch06 retrieve_policy order_id=... queries=... hits=... top_score=...`。

- [ ] **Step 4: 运行确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_refund_flow.py -q`
Expected: PASS。

- [ ] **Step 5: 更新 dev-notes 并提交**

```bash
git add app/agent/graph.py tests/test_agent_refund_flow.py dev-notes/ch06.md
git commit -m "feat: add refund policy multi-query retrieval"
```

---

### Task 14: 退款/售后完整路径

**Files:**
- Modify: `app/agent/graph.py`（去掉占位、连通完整路径）
- Modify: `app/agent/subgraph.py`（如 Task 10 已做则只补集成测试）
- Test: `tests/test_agent_refund_flow.py`、`tests/test_agent_graph.py`

**Interfaces:**
- `route=refund_flow`：`fetch_order → retrieve_policy → confidence_gate → main_agent → record_log`；闸失败 → `record_log`，不进 Agent。
- `main_agent` 的退款 route 绑定 `REFUND_AGENT_TOOLS`，prompt 含 order_data/evidence；退款诉求可出 `refund_form`；保修/维修不出。

- [ ] **Step 1: 写失败测试**

测试状态用 `resolved_query="订单 1001 能退吗"`，`execute_tool` 返回演示订单，`intent_llm.queries=["q1","q2","q3"]`，避免 interrupt。

```python
def test_refund_flow_order_then_policy_then_agent(monkeypatch):
    # 共享 calls 列表，断言 query_order → retrieve_hybrid ×3 → agent LLM 的顺序
    ...


def test_refund_gate_failure_skips_agent_and_form(monkeypatch):
    ...


def test_refund_question_can_emit_refund_form(monkeypatch):
    # fake agent LLM: submit_refund 调用 + 文本；断言 options 含 refund_form
    ...


def test_after_sales_repair_does_not_emit_refund_form(monkeypatch):
    # fake agent LLM: 只文本或 suggest_handoff；断言无 refund_form
    ...
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_refund_flow.py tests/test_agent_graph.py -q`
Expected: FAIL（占位/未连通）。

- [ ] **Step 3: 实现**

连通完整路径；确保 `confidence_gate` 通过边到 `main_agent`；`main_agent` 按 route 选 prompt/工具；退款失败兜底不进 Agent。

- [ ] **Step 4: 运行确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_refund_flow.py tests/test_agent_graph.py -q`
Expected: PASS。

- [ ] **Step 5: 更新 dev-notes 并提交**

```bash
git add app/agent/graph.py app/agent/subgraph.py tests/test_agent_refund_flow.py tests/test_agent_graph.py dev-notes/ch06.md
git commit -m "feat: complete refund and after-sales subflow"
```

---

### Task 15: Service interrupt/resume

**Files:**
- Modify: `app/agent/service.py`
- Modify: `tests/agent_fakes.py`
- Test: `tests/test_agent_service.py`

**Interfaces:**
- `class NoPendingInterrupt(Exception)`
- `def has_pending_interrupt(graph, user_id: str) -> bool`
- `stream_chat_agent(graph, session, user_id, message, *, settings)`：启动前检测 pending；有则重发 `interrupt` 帧、不写用户消息、不跑图；流后检测 pending，有则发 `interrupt` 帧、不写 assistant 答案。
- `resume_chat_agent(graph, session, user_id, order_id, *, settings) -> Iterator[str]`：无 pending 抛 `NoPendingInterrupt`；有则写审计 user 消息、`Command(resume=order_id)`、转发帧、写 assistant 答案/options/低置信度池。
- `_reset_inputs` 增加 `resolved_query/intent_confidence/session_id/order_id/order_data/expanded_queries`。

- [ ] **Step 1: 写失败测试**

```python
def test_stream_chat_reemits_pending_interrupt_without_new_turn(session):
    graph = FakeGraph(frames=[], final={}, snapshots=[snapshot_with_interrupt()])
    frames = list(stream_chat_agent(graph, session, "s1", "我要退款", settings=...))
    assert any('"type":"interrupt"' in frame for frame in frames)
    assert graph.stream_calls == []


def test_stream_chat_emits_interrupt_and_skips_answer(session):
    # stream 正常结束，流后 snapshot 有 interrupt → 发 interrupt 帧，不写 assistant 消息
    ...


def test_resume_chat_uses_command_and_writes_answer(session):
    graph = FakeGraph(... snapshots=[snapshot_with_interrupt(), snapshot_without_interrupt(final=...)]
    frames = list(resume_chat_agent(graph, session, "s1", "1001", settings=...))
    assert graph.inputs[-1].resume == "1001"
    assert assistant message written


def test_resume_chat_raises_without_pending(session):
    with pytest.raises(NoPendingInterrupt):
        list(resume_chat_agent(graph, session, "s1", "1001", settings=...))
```

`FakeGraph` 增加 `snapshots` 队列与 `stream_calls`；`get_state` 依次返回。

- [ ] **Step 2: 运行确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_service.py -q`
Expected: FAIL（无 pending 检测/无 resume）。

- [ ] **Step 3: 实现**

按 **Interfaces** 改 `service.py`；`Command` 从 `langgraph.types` 导入；审计 user 消息 `[订单选择] 订单 <id>`；不把审计消息送进图。

- [ ] **Step 4: 运行确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_service.py -q`
Expected: PASS。

- [ ] **Step 5: 更新 dev-notes 并提交**

```bash
git add app/agent/service.py tests/agent_fakes.py tests/test_agent_service.py dev-notes/ch06.md
git commit -m "feat: add ch06 interrupt and resume service"
```

---

### Task 16: `/ch06` API、退款端点与兼容层

**Files:**
- Create: `app/api/ch06.py`
- Create: `app/static/ch06.html`（最小页面壳；Task 17 补交互）
- Modify: `app/api/ch05.py`
- Modify: `app/main.py`
- Modify: `app/schemas.py`
- Test: `tests/test_ch06_api.py`、`tests/test_ch05_api.py`、`tests/test_ch05_page.py`

**Interfaces:**
- `CH06_PAGE`；`get_ch06_graph_dep`；`close_ch06_graph`。
- `ResumeRequest(session_id, order_id)`；`RefundRequest(session_id, order_id, reason: Literal[...])`；`REFUND_REASONS: tuple[str,...]`。
- `GET /`、`GET /ch06` → `ch06.html`；`GET /ch05` → 307 `/ch06`；`POST /ch05/chat` → 307 `/ch06/chat`；`POST /ch05/tickets` 保留。
- `POST /ch06/chat` SSE；`POST /ch06/chat/resume` SSE（无 pending 409）；`POST /ch06/refunds` JSON。
- 懒加载使用 `settings.ch06_checkpoint_db`；`build_llm(settings, temperature=0.0, model=settings.intent_model or None)`。
- `/ch06/refunds`：校验 reason/order 在目录内；`execute_tool(create_ticket, {...}, timeout_s=5, retries=0)`；`TICKET_NO_PATTERN`；成功 `{"ticket_no","message":"退款单已创建"}`。

- [ ] **Step 1: 写失败测试**

```python
def test_ch06_page_and_root(client):
    assert client.get("/ch06").status_code == 200
    assert client.get("/").status_code == 200
    assert "lamp-cs-ch06-session-id" in client.get("/ch06").text


def test_ch05_redirects(client):
    assert client.get("/ch05", follow_redirects=False).status_code == 307
    assert client.post("/ch05/chat", json={"session_id": "s", "message": "x"}, follow_redirects=False).status_code == 307


def test_resume_without_pending_409(client):
    app.dependency_overrides[get_ch06_graph_dep] = lambda: FakeGraph(... interrupt free ...)
    r = client.post("/ch06/chat/resume", json={"session_id": "s", "order_id": "1001"})
    assert r.status_code == 409


def test_ch06_refund_writes_ticket(client, session):
    r = client.post("/ch06/refunds", json={"session_id": "s", "order_id": "1001", "reason": "质量问题"})
    assert r.status_code == 200
    ticket = session.query(Ticket).one()
    assert ticket.ticket_type == "退款"
    assert "订单 1001" in ticket.description and "质量问题" in ticket.description


def test_ch06_refund_invalid_reason_422(client):
    assert client.post("/ch06/refunds", json={"session_id": "s", "order_id": "1001", "reason": "违规"}).status_code == 422


def test_ch06_refund_unknown_order_422(client):
    assert client.post("/ch06/refunds", json={"session_id": "s", "order_id": "9999", "reason": "质量问题"}).status_code == 422
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ch06_api.py tests/test_ch05_api.py tests/test_ch05_page.py -q`
Expected: FAIL（无 ch06 模块/路由/页面）。

- [ ] **Step 3: 实现**

- `app/api/ch06.py`：懒加载、页面、chat/resume/refunds；resume 用同步 `has_pending_interrupt` 检查后返回 `EventSourceResponse`，生成器自建 DB session（避免依赖 session 提前关闭）。
- `app/api/ch05.py`：改为兼容层（`/ch05`、`/ch05/chat` 307；`/ch05/tickets` 保留）；删除 ch05 graph 懒加载/关闭。
- `app/main.py`：根路由 `CH06_PAGE`；注册 ch06；lifespan `close_ch06_graph`；移除 ch05 graph 状态。
- `app/schemas.py`：`ResumeRequest`、`RefundRequest`、`REFUND_REASONS`。
- `app/static/ch06.html`：最小壳（title/ch06、session key、输入框/按钮、`/ch06/chat` 占位脚本）。

- [ ] **Step 4: 运行确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ch06_api.py tests/test_ch05_api.py tests/test_ch05_page.py -q`
Expected: PASS。

- [ ] **Step 5: 更新 dev-notes 并提交**

```bash
git add app/api/ch06.py app/api/ch05.py app/main.py app/schemas.py app/static/ch06.html tests/test_ch06_api.py tests/test_ch05_api.py tests/test_ch05_page.py dev-notes/ch06.md
git commit -m "feat: add ch06 api, refund endpoint and ch05 compatibility"
```

---

### Task 17: 前端（Vibe Coding）

**Files:**
- Modify: `app/static/ch06.html`

**说明：** 前端不走 TDD/code review。按 spec §11 直接实现，用户描述效果后迭代。

- [ ] **Step 1: 实现交互**

- 会话 key `lamp-cs-ch06-session-id`；发送走 `/ch06/chat`。
- 解析 `interrupt` 帧：`kind=select_order` → 聊天流渲染订单卡片（订单号/商品摘要/金额/状态/签收时间）。
- 点选卡片：禁用卡片、追加「已选择订单 X」、`POST /ch06/chat/resume`，续流渲染到新助手气泡；等待期间禁用输入框；失败恢复卡片。
- `options` 帧：保留转人工/建工单；`refund_form` → 「提交退款工单」按钮 → 表单（订单号只读 + 五项原因下拉）→ `POST /ch06/refunds` → 显示退款单号；成功禁用，失败可重试。
- 保留 90 秒无数据超时、错误帧、tool/sources 状态行。

- [ ] **Step 2: 浏览器验收**

启动 `.venv/Scripts/python.exe run.py`，用 Playwright/Edge headless 打开 `http://localhost:8000/`：
1. 发「我要退款」→ 订单卡片出现；
2. 点 1001 → 续流 → 政策/判断 → 「提交退款工单」；
3. 选「质量问题」→ 提交 → 显示退款单号；
4. 发「我要投诉」→ 转人工/建工单按钮仍在；
5. 发「你好」→ 固定闲聊文案。
截图保存到 `docs/superpowers/reports/ch06-browser-*.png`，结果写 dev-notes。

- [ ] **Step 3: 更新 dev-notes 并提交**

```bash
git add app/static/ch06.html docs/superpowers/reports/ch06-browser-*.png dev-notes/ch06.md
git commit -m "feat: add ch06 chat page with order picker and refund form"
```

---

### Task 18: 多轮验收、全量测试、code review、finish

**Files:**
- Create: `app/jobs/eval_ch06.py`
- Test: `tests/test_eval_ch06.py`
- Report: `docs/superpowers/reports/2026-11-18-ch06-acceptance.md`
- Update: `dev-notes/ch06.md`

**Interfaces:**
- `run_dialogue(llm, turns) -> list[dict]`：逐轮 `resolve_query` + `classify_intent`，累积 history；返回 `resolved_query/intent/confidence/expected`。
- `summarize(results) -> dict`：每轮 intent 对、resolved 含必须实体。

- [ ] **Step 1: 写多轮验收脚本**

对话：
1. 用户「订单 1001 的物流到哪了」→ 期望 `物流`，resolved 含 `1001`；
2. 用户「它能退吗」→ 期望 `退款退货`，resolved 含 `1001`、不含 `它`；
3. 用户「那它现在到哪了」→ 期望 `物流`，resolved 含 `1001`。

- [ ] **Step 2: 写纯函数测试**

```python
def test_summarize_dialogue_pass_fail():
    results = [
        {"expected": "物流", "intent": "物流", "resolved_query": "订单 1001 物流", "must_include": ["1001"]},
        {"expected": "退款退货", "intent": "物流", "resolved_query": "它能退吗", "must_include": ["1001"]},
    ]
    out = summarize(results)
    assert out["passed"] == 1 and out["failed"] == 1
```

- [ ] **Step 3: 跑真实多轮验收**

Run: `.venv/Scripts/python.exe -m app.jobs.eval_ch06`
Expected: 三轮全过；报告写 `2026-11-18-ch06-acceptance.md`。失败则调 coref/intent prompt 后重跑。

- [ ] **Step 4: 全量 pytest**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: 全部 PASS；记录数量。

- [ ] **Step 5: Code review**

调用 `superpowers:requesting-code-review` 审查 `feature/ch06-dispatcher` 相对 ch05 的 diff；修复发现；做 scoped re-review。把结论写进 dev-notes。

- [ ] **Step 6: Finish**

调用 `superpowers:finishing-a-development-branch`；按用户选择合并/PR。把 finish 结论写进 dev-notes。

- [ ] **Step 7: 提交**

```bash
git add app/jobs/eval_ch06.py tests/test_eval_ch06.py docs/superpowers/reports/2026-11-18-ch06-acceptance.md dev-notes/ch06.md
git commit -m "test: add ch06 multi-turn acceptance and finish notes"
```

---

## Self-Review

**1. Spec coverage**

- 指代消解 + 改写：Task 4/5；意图四件套：Task 6/7；扩写：Task 8/9；五出口：Task 2/11；退款子流程：Task 12/13/14；槽位/interrupt：Task 12/15/16；退款表单/tickets：Task 3/16/17；前端：Task 17；配置种子：Task 2；验收：Task 18。
- 无遗漏。

**2. Step scan**

- 每个任务都有可执行动作、可检查结果、提交步骤；Prompt/数据任务用评估集替代 TDD；前端明确 Vibe Coding。

**3. Type consistency**

- `resolved_query`、`intent_confidence`、`order_id`、`order_data`、`expanded_queries` 在 State、graph、service、tests 中同名。
- `QueryExpansion.queries`、`CorefRewrite.resolved_query`、`IntentResult.intent/confidence` 一致。
- `tools_for_route("refund_flow") == [submit_refund, suggest_handoff]`；`general == [query_order, query_product, query_logistics, suggest_handoff]`。

**4. Review Focus**

- 5 条均已落到 Task 16/15/13/6/8 的测试；无空项。

**5. Proportion**

- 计划以签名、测试名/断言、文件路径为主；未写完整实现体；Prompt 只列规则，不贴全文。
