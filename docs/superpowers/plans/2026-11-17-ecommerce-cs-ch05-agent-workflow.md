# Chapter 5 Agent Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `/ch05` customer-service workflow built from a deterministic LangGraph skeleton plus a nested ReAct Agent subgraph, with forced RAG + confidence gate for knowledge intents, fixed complaint/chitchat replies, independent transfer/ticket frontend buttons, and SQLite-checkpointed state.

**Architecture:** Parent `StateGraph` routes seven intents into four exits; knowledge exits force `retrieve_hybrid` then a threshold gate before entering a nested `agent ↔ tools` ReAct subgraph. The subgraph streams tokens/tool frames through `get_stream_writer`; the parent graph shares `messages` and persists state with `SqliteSaver`. `/ch05/chat` adapts the graph stream to SSE; `/ch05/tickets` is the only backend action for tickets. The existing React `/chat` stays untouched.

**Tech Stack:** Python 3.13, FastAPI, LangGraph 1.2.20, langgraph-checkpoint-sqlite 3.1.3, LangChain 1.4.10, SQLAlchemy, existing ch02 tools + ch04 retrieval, vanilla HTML/CSS/JS.

**Spec:** `docs/superpowers/specs/2026-11-17-ecommerce-cs-ch05-agent-workflow-design.md`

## Global Constraints

- LangGraph is `1.2.20`; add `langgraph-checkpoint-sqlite==3.1.3` only. Do not replace the fixed stack.
- Existing `POST /chat` and React `/` must keep working; new work goes under `/ch05` and `app/agent/`.
- Agent tools are exactly `query_order`, `query_product`, `query_logistics`, and the control tool `suggest_handoff`. Never bind `query_faq` or `create_ticket` to the Agent.
- `create_ticket` writes only the `tickets` table. It must not change `conversations.status`.
- `thread_id` is `f"user:{session_id}"`; never use the auto-increment `conversation.id`.
- Subgraph custom stream requires `graph.stream(..., stream_mode="custom", subgraphs=True, version="v2")`.
- The confidence gate uses the Top-1 reranker score and never treats `rerank_failed` / `infra_error` as a knowledge gap or a low-confidence-pool row.
- Fixed-route nodes (`complaint_reply`, `chitchat_reply`, gate fallback, `finalize`) emit one token frame and append an `AIMessage` to state.
- Chinese copy values are fixed by the spec: `转人工`, `建工单`, `已转接人工客服`, `您好，我是客服小猫，请问有什么可以帮您的？`.
- Every task ends with a `dev-notes/ch05.md` entry using the four required headings (user words, outputs, rejections, failures/rework).
- Use Context7 before changing a library-specific API if the documented signature differs from the spec; do not guess a version-specific API.

## Review Focus

These are the five failure modes most likely to bite a user; each is pinned to a task test.

1. **Reranker outage**: `rerank_failed=True` returns RRF scores, not rerank scores. It must refuse conservatively and must not write a low-confidence row. Owned by Task 6 + Task 9.
2. **Nested stream loss**: without `subgraphs=True`, Agent tokens/tool frames silently disappear. The `/ch05/chat` stream must show tool frames and answer tokens from the subgraph. Owned by Task 8 + Task 10.
3. **Broken multi-turn memory**: fixed-answer paths that only set `answer` leave consecutive `HumanMessage`s. Every terminal path must append an `AIMessage`. Owned by Task 8 + Task 9.
4. **Ticket/transfer coupling**: clicking either button must not trigger the other, and `create_ticket` must not flip conversation status. Owned by Task 3 + Task 11 + Task 12.
5. **Intent boundary errors**: “投诉” vs “退款退货” vs “售后”, and malformed classifier JSON. The classifier must fall back to `售后` and never auto-route to complaint/chitchat. Owned by Task 4 + Task 5.

---

### Task 1: Branch, dependency, config, and checkpoint path

**Files:**
- Modify: `requirements.txt`
- Modify: `app/config.py`
- Modify: `.env.example`
- Modify: `.gitignore`
- Create: `tests/test_ch05_config.py`
- Modify: `dev-notes/ch05.md`

**Interfaces:**
- Produces `Settings` fields: `agent_max_steps: int = 6`, `agent_token_budget: int = 12000`, `agent_tool_result_max_chars: int = 2000`, `confidence_min_score: float = 0.3`, `langgraph_checkpoint_db: str = ".data/ch05-checkpoints.sqlite"`.
- Produces installed import: `from langgraph.checkpoint.sqlite import SqliteSaver`.

- [ ] **Step 1: Confirm the pre-created worktree/branch and commit the approved design**

The controller already created `D:/AItest/.worktrees/ch05-agent-workflow` on branch `feature/ch05-agent-workflow` and copied the design, plan, and `dev-notes/ch05.md` into it. Do NOT run `git checkout -b`. Run:
```bash
git branch --show-current   # must print feature/ch05-agent-workflow
git add docs/superpowers/specs/2026-11-17-ecommerce-cs-ch05-agent-workflow-design.md docs/superpowers/plans/2026-11-17-ecommerce-cs-ch05-agent-workflow.md dev-notes/ch05.md
git commit -m "docs: add ch05 agent workflow design, plan, and notes"
```

- [ ] **Step 2: Write the failing config test**

```python
# tests/test_ch05_config.py
from app.config import Settings


def test_ch05_settings_defaults():
    s = Settings(_env_file=None, openai_api_key="x")
    assert s.agent_max_steps == 6
    assert s.agent_token_budget == 12000
    assert s.agent_tool_result_max_chars == 2000
    assert s.confidence_min_score == 0.3
    assert s.langgraph_checkpoint_db == ".data/ch05-checkpoints.sqlite"


def test_ch05_settings_env_override(monkeypatch):
    monkeypatch.setenv("AGENT_MAX_STEPS", "3")
    monkeypatch.setenv("CONFIDENCE_MIN_SCORE", "0.55")
    s = Settings(_env_file=None, openai_api_key="x")
    assert s.agent_max_steps == 3
    assert s.confidence_min_score == 0.55
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ch05_config.py -v`
Expected: FAIL with `AttributeError: 'Settings' object has no attribute 'agent_max_steps'` (or Pydantic validation error).

- [ ] **Step 4: Add the settings fields**

Add to `Settings` in `app/config.py`:
```python
agent_max_steps: int = 6
agent_token_budget: int = 12000
agent_tool_result_max_chars: int = 2000
confidence_min_score: float = 0.3
langgraph_checkpoint_db: str = ".data/ch05-checkpoints.sqlite"
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ch05_config.py -v`
Expected: 2 passed.

- [ ] **Step 6: Add the dependency and ignore the checkpoint directory**

Run:
```bash
.venv/Scripts/pip.exe install "langgraph==1.2.20" "langgraph-checkpoint-sqlite==3.1.3"
```
Add to `requirements.txt`:
```text
langgraph==1.2.20
langgraph-checkpoint-sqlite==3.1.3
```
Add to `.gitignore`:
```text
.data/
.worktrees/
```
Add to `.env.example`:
```dotenv
AGENT_MAX_STEPS=6
AGENT_TOKEN_BUDGET=12000
AGENT_TOOL_RESULT_MAX_CHARS=2000
CONFIDENCE_MIN_SCORE=0.3
LANGGRAPH_CHECKPOINT_DB=.data/ch05-checkpoints.sqlite
```

- [ ] **Step 7: Verify the import and commit**

Run:
```bash
.venv/Scripts/python.exe -c "from langgraph.checkpoint.sqlite import SqliteSaver; print('ok')"
git add requirements.txt app/config.py .env.example .gitignore tests/test_ch05_config.py
git commit -m "chore: add ch05 config and sqlite checkpointer dependency"
```

- [ ] **Step 8: Append the dev-notes entry**

Append `## 阶段：Task 1 完成（执行日期）` to `dev-notes/ch05.md` with the four required bullets. Record the dependency version, config defaults, and any install issue.

---

### Task 2: Naked ReAct loop and demo

**Files:**
- Create: `app/agent/__init__.py` (empty)
- Create: `app/agent/raw_loop.py`
- Create: `app/agent/raw_demo.py`
- Create: `tests/test_agent_raw_loop.py`
- Modify: `dev-notes/ch05.md`

**Interfaces:**
- Produces:
```python
@dataclass(frozen=True)
class AgentLoopResult:
    answer: str
    messages: list[AnyMessage]
    steps: int
    token_estimate: int
    exhausted: bool

def run_react_loop(
    llm,
    tools: Sequence[BaseTool],
    messages: Sequence[BaseMessage],
    *,
    max_steps: int = 6,
    token_budget: int = 12000,
    emit: Callable[[dict], None] | None = None,
) -> AgentLoopResult
```
- `emit` receives `{"type":"token","text":...}` and `{"type":"tool","tool":...,"label":...}`.
- Uses `app.services.context.estimate_tokens` (one character = one estimated token).
- Produces CLI: `python -m app.agent.raw_demo "<question>"`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_agent_raw_loop.py
from langchain_core.messages import AIMessageChunk, HumanMessage

from app.agent.raw_loop import AgentLoopResult, run_react_loop
from app.tools.registry import ALL_TOOLS


class ScriptedLLM:
    def __init__(self, chunks):
        self.chunks = list(chunks)

    def bind_tools(self, tools, **kwargs):
        return self

    def stream(self, messages):
        yield self.chunks.pop(0)


def text_chunk(text):
    return AIMessageChunk(content=text)


def tool_chunk(name, args, call_id="c1"):
    return AIMessageChunk(
        content="",
        tool_call_chunks=[{"name": name, "args": args, "id": call_id, "index": 0}],
    )


def test_raw_loop_converges_without_tool():
    events = []
    result = run_react_loop(
        ScriptedLLM([text_chunk("你好")]),
        ALL_TOOLS,
        [HumanMessage(content="在吗")],
        emit=events.append,
    )
    assert result.answer == "你好"
    assert result.steps == 1
    assert result.exhausted is False
    assert events == [{"type": "token", "text": "你好"}]


def test_raw_loop_executes_one_tool_then_converges(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "app.agent.raw_loop.execute_tool",
        lambda tool, args, timeout_s=5, retries=1: calls.append((tool.name, args))
        or '{"order_id":"1001","status":"已发货"}',
    )
    result = run_react_loop(
        ScriptedLLM([
            tool_chunk("query_order", '{"order_id":"1001"}'),
            text_chunk("订单已发货"),
        ]),
        ALL_TOOLS,
        [HumanMessage(content="订单 1001 状态")],
    )
    assert calls == [("query_order", {"order_id": "1001"})]
    assert result.answer == "订单已发货"
    assert result.steps == 2


def test_raw_loop_supports_two_tool_steps(monkeypatch):
    monkeypatch.setattr(
        "app.agent.raw_loop.execute_tool",
        lambda tool, args, timeout_s=5, retries=1: f"{tool.name}-ok",
    )
    result = run_react_loop(
        ScriptedLLM([
            tool_chunk("query_order", '{"order_id":"1001"}', "c1"),
            tool_chunk("query_logistics", '{"order_id":"1001"}', "c2"),
            text_chunk("已查到订单和物流"),
        ]),
        ALL_TOOLS,
        [HumanMessage(content="先查订单再查物流")],
    )
    assert result.steps == 3
    assert result.answer == "已查到订单和物流"


def test_raw_loop_stops_at_max_steps(monkeypatch):
    monkeypatch.setattr(
        "app.agent.raw_loop.execute_tool",
        lambda tool, args, timeout_s=5, retries=1: "ok",
    )
    result = run_react_loop(
        ScriptedLLM([tool_chunk("query_order", '{"order_id":"1001"}')] * 3),
        ALL_TOOLS,
        [HumanMessage(content="循环")],
        max_steps=2,
    )
    assert result.steps == 2
    assert result.exhausted is True
    assert result.answer == ""


def test_raw_loop_stops_at_token_budget():
    result = run_react_loop(
        ScriptedLLM([text_chunk("很长" * 100)]),
        ALL_TOOLS,
        [HumanMessage(content="预算")],
        token_budget=1,
    )
    assert result.exhausted is True
    assert result.answer == ""
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_raw_loop.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.agent'`.

- [ ] **Step 3: Implement the loop**

`run_react_loop` behavior:
1. `working = list(messages)`, `steps = 0`, `tokens = sum(estimate_tokens(getattr(m, "content", None)) for m in working)`.
2. Before each LLM call, if `steps >= max_steps` or `tokens >= token_budget`: return `AgentLoopResult("", working, steps, tokens, True)`.
3. `bound = llm.bind_tools(tools)`; `gathered = None`; iterate `bound.stream(working)`:
   - emit content chunks;
   - `gathered = chunk if gathered is None else gathered + chunk`.
4. `steps += 1`; normalize to `AIMessage(content=gathered.content, tool_calls=gathered.tool_calls)`; append it.
5. If no `tool_calls`: return `AgentLoopResult(gathered.content or "", working, steps, tokens, False)`.
6. For each tool call: emit `{"type":"tool","tool":name,"label":TOOL_LABELS.get(name,name)}`; look up `TOOLS_BY_NAME`; execute via `execute_tool(tool, args)` (query_faq timeout 25/retries 0 is not relevant here; use default or the existing `tool_timeout`); append `ToolMessage(content=result, tool_call_id=id)`.
7. Update `tokens` with output + tool-result lengths; loop.

- [ ] **Step 4: Run the loop tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_raw_loop.py -v`
Expected: 5 passed.

- [ ] **Step 5: Add the CLI demo**

`app/agent/raw_demo.py`:
- `main()` reads `sys.argv[1]`; if missing prints usage and exits 2.
- Builds `Settings`, `build_llm(settings)`, `render_system_prompt()`, and calls `run_react_loop`.
- Prints each emitted event as a line (`[tool] 查物流`, `[token] ...` buffered to stdout), then prints `steps=`, `token_estimate=`, `exhausted=`, `answer=`.
- Run the real demo:
```bash
.venv/Scripts/python.exe -m app.agent.raw_demo "订单 1001 的物流到哪了"
```
Expected: at least one `[tool] 查物流` event, then a final answer. Record the actual output in dev-notes. This is the “祛魅热身” validation; no unit test for the real model call.

- [ ] **Step 6: Run the full suite and commit**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: all existing + 5 new tests pass.
```bash
git add app/agent tests/test_agent_raw_loop.py
git commit -m "feat: add naked react loop and demo"
```

- [ ] **Step 7: Append the dev-notes entry**

Append `## 阶段：Task 2 完成（执行日期）` with the raw demo command/output, tests, and any model behavior surprises.

---

### Task 3: Make `create_ticket` ticket-only

**Files:**
- Modify: `app/tools/ticket.py`
- Modify: `tests/test_tools_ticket.py`
- Modify: `dev-notes/ch05.md`

**Interfaces:**
- Produces: `create_ticket` still returns the ticket number string or a `创建工单失败...` string; it no longer updates `Conversation.status`.
- Produces public constant `TICKET_NO_PATTERN = re.compile(r"^T\d{11}$")` in `app/tools/ticket.py` for the API.

- [ ] **Step 1: Update the failing test**

In `tests/test_tools_ticket.py`, change the assertion:
```python
def test_create_ticket_writes_only_ticket(session):
    cid = _make_conversation(session)
    current_conversation_id.set(cid)
    out = create_ticket.invoke({"description": "屏幕碎了", "ticket_type": "售后"})
    session.expire_all()
    assert TICKET_NO_PATTERN.match(out)
    t = session.query(Ticket).one()
    assert t.conversation_id == cid and t.status == "待处理"
    assert session.query(Conversation).get(cid).status == "进行中"
```
Import `TICKET_NO_PATTERN` from `app.tools.ticket`.

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tools_ticket.py -v`
Expected: FAIL because status is `已转人工` and/or `TICKET_NO_PATTERN` is missing.

- [ ] **Step 3: Remove the status side effect**

In `app/tools/ticket.py`:
- Remove the `Conversation` lookup and `conversation.status = "已转人工"` block.
- Remove the now-unused `Conversation` import.
- Add `TICKET_NO_PATTERN = re.compile(r"^T\d{11}$")`.
- Update the module and tool docstrings to say “只写 tickets 表，不改会话状态；转人工由前端独立模拟”.

- [ ] **Step 4: Run the ticket tests and full suite**

Run:
```bash
.venv/Scripts/python.exe -m pytest tests/test_tools_ticket.py -v
.venv/Scripts/python.exe -m pytest -q
```
Expected: all pass.

- [ ] **Step 5: Commit and record dev-notes**

```bash
git add app/tools/ticket.py tests/test_tools_ticket.py
git commit -m "fix: make ticket creation independent from human transfer"
```
Append `## 阶段：Task 3 完成（执行日期）`; record the semantic change and the test update.

---

### Task 4: State, intent classifier, and hardcoded routing

**Files:**
- Create: `app/agent/state.py`
- Create: `app/agent/routing.py`
- Create: `app/agent/intent.py`
- Create: `tests/test_agent_routing.py`
- Create: `tests/test_agent_intent.py`
- Modify: `dev-notes/ch05.md`

**Interfaces:**
- Produces:
```python
Intent = Literal["物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "闲聊"]
Route = Literal["knowledge", "business", "complaint", "chitchat"]

INTENT_ROUTES: dict[Intent, Route]
def route_for_intent(intent: str) -> Route

class IntentResult(BaseModel):
    intent: Intent
    reason: str = ""

INTENT_SYSTEM_PROMPT: str
def classify_intent(llm, message: str) -> IntentResult
```
- Produces `CustomerServiceState` and `AgentState` TypedDicts exactly as in spec §6.3/§6.5.
- `route_for_intent` returns `"business"` for unknown values and logs a warning.
- `classify_intent` catches any exception and returns `IntentResult(intent="售后", reason=f"fallback: {exc}")`.

- [ ] **Step 1: Write the failing routing test**

```python
# tests/test_agent_routing.py
from app.agent.routing import INTENT_ROUTES, route_for_intent


def test_every_intent_maps_to_one_of_four_exits():
    assert INTENT_ROUTES == {
        "商品咨询": "knowledge",
        "退款退货": "knowledge",
        "物流": "business",
        "订单": "business",
        "售后": "business",
        "投诉": "complaint",
        "闲聊": "chitchat",
    }


def test_unknown_intent_falls_back_to_business():
    assert route_for_intent("未知") == "business"
```

- [ ] **Step 2: Write the failing classifier tests**

```python
# tests/test_agent_intent.py
from app.agent.intent import IntentResult, classify_intent


class StructuredLLM:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error

    def with_structured_output(self, schema, method=None):
        return self

    def invoke(self, messages):
        if self.error:
            raise self.error
        return self.result


def test_classify_valid_json():
    llm = StructuredLLM(result=IntentResult(intent="物流", reason="问快递"))
    assert classify_intent(llm, "我的快递到哪了").intent == "物流"


def test_classify_malformed_json_falls_back_to_after_sale():
    llm = StructuredLLM(error=ValueError("bad json"))
    out = classify_intent(llm, "我要投诉")
    assert out.intent == "售后"
    assert "fallback" in out.reason


def test_intent_prompt_lists_all_seven_labels():
    from app.agent.intent import INTENT_SYSTEM_PROMPT
    for label in ["物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "闲聊"]:
        assert label in INTENT_SYSTEM_PROMPT
    assert "只输出" in INTENT_SYSTEM_PROMPT and "JSON" in INTENT_SYSTEM_PROMPT
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_routing.py tests/test_agent_intent.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 4: Implement state, routing, and intent**

- `app/agent/state.py`: define `Intent`, `Route`, `CustomerServiceState`, `AgentState` exactly as spec §6.3 and §6.5. Use `Annotated[list[AnyMessage], add_messages]` for messages.
- `app/agent/routing.py`: define the table and function. Use `logger.warning` for unknown.
- `app/agent/intent.py`: define the Pydantic model, a Chinese prompt with the seven definitions, a “不确定归售后，不要猜投诉/闲聊” rule, and “只输出 JSON”. Use `llm.with_structured_output(IntentResult, method="json_mode").invoke([SystemMessage(...), HumanMessage(...)])`. Catch all exceptions.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_routing.py tests/test_agent_intent.py -v`
Expected: all pass.

- [ ] **Step 6: Commit and record dev-notes**

```bash
git add app/agent/state.py app/agent/routing.py app/agent/intent.py tests/test_agent_routing.py tests/test_agent_intent.py
git commit -m "feat: add ch05 intent classifier and hardcoded routing"
```
Append `## 阶段：Task 4 完成（执行日期）`; record the route table and fallback decision.

---

### Task 5: Intent labeled sample set and live prompt validation

**Files:**
- Create: `tests/eval/ch05_intent_cases.json`
- Create: `tests/test_intent_cases.py`
- Create: `app/jobs/eval_intent.py`
- Modify: `app/agent/intent.py` (only if prompt tuning is needed)
- Create: `docs/superpowers/reports/2026-11-17-ch05-intent-eval.md`
- Modify: `dev-notes/ch05.md`

**Interfaces:**
- Dataset schema: list of `{"id": str, "text": str, "intent": Intent}`.
- Produces:
```python
def compute_metrics(expected: list[str], predicted: list[str]) -> dict
```
returns `{"total": int, "correct": int, "accuracy": float, "per_class": dict[str, dict[str, int]], "confusion": dict[str, dict[str, int]]}`.
- CLI: `python -m app.jobs.eval_intent` uses real `build_llm(settings, temperature=0.0)`, writes the markdown report.

- [ ] **Step 1: Write the failing dataset-schema test**

```python
# tests/test_intent_cases.py
import json
from pathlib import Path

from app.agent.routing import INTENT_ROUTES

CASES = Path(__file__).parent / "eval" / "ch05_intent_cases.json"


def test_intent_cases_cover_all_classes():
    rows = json.loads(CASES.read_text(encoding="utf-8"))
    assert len(rows) == 42
    ids = [row["id"] for row in rows]
    assert len(ids) == len(set(ids))
    by_intent = {}
    for row in rows:
        assert row["text"].strip()
        assert row["intent"] in INTENT_ROUTES
        by_intent[row["intent"]] = by_intent.get(row["intent"], 0) + 1
    assert by_intent == {intent: 6 for intent in INTENT_ROUTES}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_intent_cases.py -v`
Expected: FAIL with `FileNotFoundError`.

- [ ] **Step 3: Create the 42 labeled cases and the metrics function**

Create six cases per class. Include these boundaries:
- 商品咨询 vs 售后: “这个型号的功率是多少” vs “买回来坏了怎么修”.
- 退款退货 vs 售后: “签收后还能退吗” vs “补发一个配件”.
- 投诉 vs 退款退货: “太差了，我要投诉你们” vs “我要申请退款”.
- 闲聊 vs 订单: “你好，在吗” vs “我的订单在哪”.

Implement `compute_metrics` in `app/jobs/eval_intent.py`; write the report with overall accuracy, per-class accuracy, and a confusion matrix. The report must list every wrong case with expected vs predicted.

- [ ] **Step 4: Add the pure metrics test**

```python
def test_compute_metrics_counts_confusion():
    from app.jobs.eval_intent import compute_metrics
    out = compute_metrics(["物流", "物流", "投诉"], ["物流", "订单", "投诉"])
    assert out["total"] == 3
    assert out["correct"] == 2
    assert out["per_class"]["物流"] == {"total": 2, "correct": 1}
    assert out["confusion"]["物流"]["订单"] == 1
```

- [ ] **Step 5: Run the dataset + metrics tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_intent_cases.py -v`
Expected: all pass.

- [ ] **Step 6: Run the live validation and tune if needed**

Run:
```bash
.venv/Scripts/python.exe -m app.jobs.eval_intent
```
Pass criterion: overall accuracy ≥ 90% and each class ≥ 80%. If below, edit the prompt/few-shot examples in `app/agent/intent.py`, rerun, and record the before/after. This step is the TDD substitute for prompt work; no live model call in pytest.

- [ ] **Step 7: Commit and record dev-notes**

```bash
git add tests/eval/ch05_intent_cases.json tests/test_intent_cases.py app/jobs/eval_intent.py app/agent/intent.py docs/superpowers/reports/2026-11-17-ch05-intent-eval.md
git commit -m "test: add ch05 intent sample validation"
```
Append `## 阶段：Task 5 完成（执行日期）`; record accuracy, per-class weak spots, and any prompt changes.

---

### Task 6: Confidence gate

**Files:**
- Create: `app/agent/confidence.py`
- Create: `tests/test_agent_confidence.py`
- Modify: `dev-notes/ch05.md`

**Interfaces:**
- Produces:
```python
@dataclass(frozen=True)
class ConfidenceVerdict:
    passed: bool
    top_score: float | None
    reason: str  # "ok" | "empty" | "below_threshold" | "rerank_failed"

def assess_confidence(
    chunks: Sequence[RetrievedChunk],
    min_score: float,
    *,
    rerank_failed: bool = False,
) -> ConfidenceVerdict
```
- If `rerank_failed`: `passed=False`, `reason="rerank_failed"`, `top_score` may still be reported for logging.
- If empty: `passed=False`, `reason="empty"`, `top_score=None`.
- Else top = max chunk score; pass iff `top >= min_score`, reason `"ok"` or `"below_threshold"`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_agent_confidence.py
from app.agent.confidence import assess_confidence
from app.rag.retrieval import RetrievedChunk


def chunk(score):
    return RetrievedChunk(
        chunk_id=1, score=score, questions="q", answer="a",
        section_path="s", category="商品", content_type="faq",
    )


def test_empty_is_a_knowledge_gap():
    out = assess_confidence([], 0.3)
    assert (out.passed, out.reason, out.top_score) == (False, "empty", None)


def test_below_threshold_fails():
    out = assess_confidence([chunk(0.29)], 0.3)
    assert (out.passed, out.reason) == (False, "below_threshold")
    assert out.top_score == 0.29


def test_equal_threshold_passes():
    out = assess_confidence([chunk(0.3)], 0.3)
    assert (out.passed, out.reason) == (True, "ok")


def test_rerank_failure_is_not_a_knowledge_gap():
    out = assess_confidence([chunk(0.02)], 0.3, rerank_failed=True)
    assert (out.passed, out.reason) == (False, "rerank_failed")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_confidence.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the verdict function**

Implement `assess_confidence` with the exact reasons above. Do not call the reranker; this is a pure judgment over already-retrieved chunks.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_confidence.py -v`
Expected: 4 passed.

- [ ] **Step 5: Calibrate the threshold with real retrieval**

Run a small live script over 3 in-domain and 3 out-of-domain queries using `retrieve_hybrid`, print the Top-1 reranker score for each. If the positives clearly separate from the negatives, set `confidence_min_score` so positives pass and negatives fail; update `app/config.py`, `.env.example`, and `tests/test_ch05_config.py` if the default changes. If the environment cannot reach Milvus/embedding/reranker, record that and keep `0.3` as the provisional default; the gate logic is still covered by unit tests.

- [ ] **Step 6: Commit and record dev-notes**

```bash
git add app/agent/confidence.py tests/test_agent_confidence.py app/config.py .env.example tests/test_ch05_config.py
git commit -m "feat: add ch05 confidence gate"
```
Append `## 阶段：Task 6 完成（执行日期）`; record calibration scores or the reason calibration was skipped.

---

### Task 7: Agent prompt and `suggest_handoff` control tool

**Files:**
- Modify: `app/prompts/customer_service.py`
- Create: `app/agent/tools.py`
- Create: `tests/test_agent_tools.py`
- Modify: `dev-notes/ch05.md`

**Interfaces:**
- Produces:
```python
def render_agent_system_prompt(
    business_name: str = "本店",
    evidence: Sequence[dict] | None = None,
) -> str

@tool
def suggest_handoff(
    options: list[Literal["transfer", "ticket"]],
    ticket_type: Literal["售后", "投诉", "咨询"] = "咨询",
    reason: str = "",
) -> str

AGENT_TOOLS: list[BaseTool]
AGENT_TOOLS_BY_NAME: dict[str, BaseTool]
BUSINESS_TOOL_NAMES: frozenset[str] = frozenset({"query_order", "query_product", "query_logistics"})

def suggested_actions_from_handoff(
    options: Sequence[str],
    ticket_type: str = "咨询",
) -> list[dict]
```
- `suggested_actions_from_handoff` returns `[{"action":"transfer","label":"转人工"}]` and/or `[{"action":"ticket","label":"建工单","ticket_type": ticket_type}]`, deduped and order-preserving; invalid actions are dropped.
- `AGENT_TOOLS` order is exactly `["query_order", "query_product", "query_logistics", "suggest_handoff"]`; no `query_faq`, no `create_ticket`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_agent_tools.py
from app.agent.tools import (
    AGENT_TOOLS,
    suggested_actions_from_handoff,
)


def test_agent_tools_exclude_faq_and_create_ticket():
    assert [t.name for t in AGENT_TOOLS] == [
        "query_order", "query_product", "query_logistics", "suggest_handoff"
    ]


def test_suggested_actions_are_independent_and_deduped():
    out = suggested_actions_from_handoff(["transfer", "ticket", "transfer"], "投诉")
    assert out == [
        {"action": "transfer", "label": "转人工"},
        {"action": "ticket", "label": "建工单", "ticket_type": "投诉"},
    ]


def test_invalid_handoff_option_is_dropped():
    assert suggested_actions_from_handoff(["bogus", "ticket"], "咨询") == [
        {"action": "ticket", "label": "建工单", "ticket_type": "咨询"}
    ]


def test_agent_prompt_has_handoff_and_no_ticket_rules():
    from app.prompts.customer_service import render_agent_system_prompt
    prompt = render_agent_system_prompt(
        evidence=[{"n": 1, "section_path": "退货政策", "question": "能退吗", "answer": "7 天", "score": 0.9}]
    )
    assert "suggest_handoff" in prompt
    assert "不得调用 create_ticket" in prompt
    assert "工具返回" in prompt and "[n]" in prompt
    assert "串行" in prompt
    assert "[1] 章节:退货政策" in prompt
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_tools.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the prompt and control tool**

- Add `render_agent_system_prompt` to `app/prompts/customer_service.py`; reuse `_NEGATIVE_RULES` and the citation concepts, but add the split rule: knowledge evidence facts need `[n]`; tool-returned order/logistics/product facts must not use `[n]`.
- Include the exact “不得调用 create_ticket” rule, the `suggest_handoff` usage rule, and the “后一个工具依赖前一个结果时必须串行” rule.
- Add `app/agent/tools.py` with `suggest_handoff`, `AGENT_TOOLS`, `AGENT_TOOLS_BY_NAME`, `BUSINESS_TOOL_NAMES`, and `suggested_actions_from_handoff`. The tool function itself is a no-op acknowledgment; all state mutation happens in the subgraph `tools` node.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_tools.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit and record dev-notes**

```bash
git add app/prompts/customer_service.py app/agent/tools.py tests/test_agent_tools.py
git commit -m "feat: add ch05 agent prompt and handoff control tool"
```
Append `## 阶段：Task 7 完成（执行日期）`.

---

### Task 8: Message assembly and Agent ReAct subgraph

**Files:**
- Modify: `app/services/context.py`
- Create: `app/agent/subgraph.py`
- Create: `tests/agent_fakes.py`
- Create: `tests/test_agent_subgraph.py`
- Modify: `tests/test_context.py`
- Modify: `dev-notes/ch05.md`

**Interfaces:**
- Produces:
```python
def trim_messages(
    system: SystemMessage,
    history: Sequence[BaseMessage],
    max_tokens: int,
) -> list[BaseMessage]

def build_agent_messages(
    system: SystemMessage,
    messages: Sequence[BaseMessage],
    max_tokens: int,
) -> list[BaseMessage]

def build_agent_subgraph(
    llm,
    *,
    max_steps: int = 6,
    token_budget: int = 12000,
    tool_result_max_chars: int = 2000,
    max_context_tokens: int = 4096,
    tools: Sequence[BaseTool] | None = None,
):
    """Return a compiled Agent subgraph (no checkpointer; parent propagates it)."""
```
- `trim_messages` preserves the existing atomic tool-round behavior and always keeps `system` first.
- `build_messages` in `app/services/context.py` must keep its existing signature/behavior; it may delegate to `trim_messages`.
- Subgraph nodes: `agent`, `tools`, `finalize`; edges `START → agent`, `agent → tools | finalize | END`, `tools → agent`, `finalize → END`.
- Subgraph custom frames: `{"type":"token","text":...}`, `{"type":"tool","tool":...,"label":...}`.
- `suggest_handoff` is intercepted in `tools`; it must not call `execute_tool`.
- `create_ticket` is absent from the bound tool list.
- `finalize` emits a fallback token, appends `AIMessage`, and adds a transfer-only option if none exists.

- [ ] **Step 1: Add the failing context test for `trim_messages`**

```python
def test_trim_messages_keeps_system_and_atomic_tool_round():
    from app.services.context import trim_messages
    system = SystemMessage(content="S")
    history = [
        HumanMessage(content="old"),
        AIMessage(content="", tool_calls=[{"name": "query_order", "args": {}, "id": "c1"}]),
        ToolMessage(content="X" * 20, tool_call_id="c1"),
        AIMessage(content="answer"),
    ]
    msgs = trim_messages(system, history, max_tokens=30)
    assert [type(m).__name__ for m in msgs] == [
        "SystemMessage", "AIMessage", "ToolMessage", "AIMessage"
    ]
```

- [ ] **Step 2: Write the subgraph tests and fake helper**

`tests/agent_fakes.py`:
```python
from langchain_core.messages import AIMessageChunk


class ScriptedLLM:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.bound_tool_names = []

    def bind_tools(self, tools, **kwargs):
        self.bound_tool_names = [t.name for t in tools]
        return self

    def stream(self, messages):
        yield self.chunks.pop(0)


def text_chunk(text):
    return AIMessageChunk(content=text)


def tool_chunk(name, args, call_id="c1"):
    return AIMessageChunk(
        content="",
        tool_call_chunks=[{"name": name, "args": args, "id": call_id, "index": 0}],
    )


def make_chunk(score=0.9):
    from app.rag.retrieval import RetrievedChunk
    return RetrievedChunk(
        chunk_id=1, score=score, questions="q", answer="a",
        section_path="s", category="商品", content_type="faq",
    )


def make_retrieval_result(chunks=None, rerank_failed=False):
    from app.rag.query_understanding import QueryAnalysis
    from app.rag.retrieval import RetrievalResult, RetrievalStrategy
    return RetrievalResult(
        chunks=list(chunks or []),
        strategy=RetrievalStrategy.HYBRID_RERANK,
        analysis=QueryAnalysis(
            original="q", rewritten="q", keywords=[], rewritten_by="fallback"
        ),
        rerank_failed=rerank_failed,
    )
```

Tests in `tests/test_agent_subgraph.py`:
- `test_subgraph_converges_without_tool`: one text chunk; final state `answer == "你好"`, `step_count == 1`, last message is `AIMessage`.
- `test_subgraph_can_ask_follow_up`: one text chunk `"请提供订单号"`; final state `answer == "请提供订单号"` and no tool call; this pins the “缺信息就追问” behavior.
- `test_subgraph_executes_tool_then_converges`: tool chunk + text chunk; monkeypatch `app.agent.subgraph.execute_tool`; final state contains a `ToolMessage`, answer; `step_count == 2`.
- `test_subgraph_does_not_bind_create_ticket_or_query_faq`: fake LLM records `bound_tool_names`; assert exact AGENT_TOOLS names.
- `test_suggest_handoff_records_options_without_side_effect`: tool chunk for `suggest_handoff` with `{"options":["transfer","ticket"],"ticket_type":"投诉"}` + final text; monkeypatch `execute_tool` to raise if called; assert options in final state.
- `test_max_steps_runs_finalize`: repeated tool chunk with `max_steps=2`; assert `exhausted` true, fallback answer non-empty, `suggested_actions` includes transfer.
- `test_token_budget_runs_finalize`: `token_budget=1`; assert exhausted and fallback.
- `test_custom_stream_emits_tool_then_tokens`: use `build_agent_subgraph(...)` (no checkpointer needed) and `graph.stream(input, stream_mode="custom")`; assert a tool frame appears before the final token frame. State assertions in the other tests use `graph.invoke`.

- [ ] **Step 3: Run the tests to verify they fail**

Run:
```bash
.venv/Scripts/python.exe -m pytest tests/test_context.py tests/test_agent_subgraph.py -v
```
Expected: FAIL with missing `trim_messages`/`build_agent_subgraph`.

- [ ] **Step 4: Implement message trimming and the subgraph**

- Refactor `app/services/context.py` so `trim_messages` contains the existing block-trimming logic; `build_messages` delegates and keeps its current tests green.
- `app/agent/subgraph.py`:
  - `agent_node`: increments `step_count`; checks step/token budget; builds `SystemMessage(render_agent_system_prompt(evidence=state.get("evidence")))`; calls `build_agent_messages`; streams `llm.bind_tools(tools).stream(...)`; emits content via `get_stream_writer`; accumulates `AIMessageChunk`; appends normalized `AIMessage`; sets `answer` if no tool calls.
  - `tools_node`: emits tool frames; special-cases `suggest_handoff` using `suggested_actions_from_handoff` and appends a ToolMessage; executes business tools via `execute_tool` after `current_conversation_id.set(state["conversation_id"])`; truncates results to `tool_result_max_chars`; appends ToolMessages.
  - `finalize_node`: emits fallback via writer; appends `AIMessage`; adds transfer option if no options.
  - `should_continue`: if last message has tool calls and not exhausted → `tools`; if exhausted or no content → `finalize`; else `END`.
  - Compile without a checkpointer.

- [ ] **Step 5: Run the tests to verify they pass**

Run:
```bash
.venv/Scripts/python.exe -m pytest tests/test_context.py tests/test_agent_subgraph.py -v
```
Expected: all pass.

- [ ] **Step 6: Commit and record dev-notes**

```bash
git add app/services/context.py app/agent/subgraph.py tests/agent_fakes.py tests/test_agent_subgraph.py tests/test_context.py
git commit -m "feat: add ch05 react agent subgraph"
```
Append `## 阶段：Task 8 完成（执行日期）`; record the step limit and stream-frame test result.

---

### Task 9: Parent graph, confidence-gated routing, and checkpointer factory

**Files:**
- Create: `app/agent/graph.py`
- Create: `tests/test_agent_graph.py`
- Modify: `dev-notes/ch05.md`

**Interfaces:**
- Produces:
```python
@dataclass
class CheckpointerHandle:
    saver: SqliteSaver
    close: Callable[[], None]

def open_sqlite_checkpointer(path: str) -> CheckpointerHandle:
    """Open SQLite, configure WAL/busy_timeout, run setup(), return saver + closer."""

@contextmanager
def sqlite_checkpointer(path: str):
    """Yield a configured SqliteSaver; close the connection on exit."""

def build_graph(
    llm,
    intent_llm,
    *,
    checkpointer=None,
    settings: Settings | None = None,
):
    """Build and compile the parent workflow graph."""
```
- `sqlite_checkpointer` is a thin wrapper over `open_sqlite_checkpointer` for tests/one-shot use; the FastAPI dependency keeps the handle open and stores it on `app.state`.
- Graph nodes: `resolve_reference`, `classify_intent`, `route_intent`, `retrieve_knowledge`, `confidence_gate`, `complaint_reply`, `chitchat_reply`, `agent`, `record_log`.
- `route_intent` is a normal node that writes `route` into State; a conditional edge then reads `state["route"]` and maps to `retrieve_knowledge` / `agent` / `complaint_reply` / `chitchat_reply`.
- `confidence_gate` conditional: `gate_passed=True` → `agent`; else → `record_log`.
- All terminal paths append an `AIMessage` and set `answer`.
- `thread_id` is not set here; service sets `user:{session_id}`.

- [ ] **Step 1: Write the failing graph tests**

Use `tests/agent_fakes.py` plus:
```python
class FakeIntentLLM:
    def __init__(self, intent):
        self.intent = intent

    def with_structured_output(self, schema, method=None):
        return self

    def invoke(self, messages):
        from app.agent.intent import IntentResult
        return IntentResult(intent=self.intent, reason="test")
```
Tests:
- `test_resolve_reference_passthrough`: final state `resolved_message == user_message` and the log contains `ch05 resolve_reference passthrough`.
- `test_knowledge_route_calls_retrieval_and_gate`: monkeypatch `app.agent.graph.retrieve_hybrid` to return `make_retrieval_result([make_chunk(0.9)])`; `build_graph(ScriptedLLM([text_chunk("政策回答")]), FakeIntentLLM("退款退货"), checkpointer=InMemorySaver())`; invoke with `{"messages":[HumanMessage("退货政策")], "user_message":"退货政策", "resolved_message":"退货政策", "conversation_id":1, "step_count":0, "token_estimate":0, "evidence":[], "sources":[], "suggested_actions":[], "gate_passed":False, "exhausted":False}`; assert final `route=="knowledge"`, `gate_passed is True`, answer, and retrieval was called.
- `test_gate_failure_skips_agent`: retrieval returns `make_retrieval_result([])`; fake Agent LLM has no chunks; assert `gate_passed is False`, answer is the fallback, and `step_count` remains 0.
- `test_rerank_failure_skips_agent_and_uses_infra_text`: retrieval returns `make_retrieval_result([make_chunk(0.02)], rerank_failed=True)`; assert `gate_reason=="rerank_failed"`, `gate_passed is False`, answer contains `可靠性校验`, and the Agent LLM was not called.
- `test_infra_error_skips_agent_and_does_not_become_empty_gap`: monkeypatch retrieval to raise; assert `gate_reason=="infra_error"`, answer contains `知识库暂时不可用`, and no agent call.
- `test_business_route_skips_retrieval`: intent `"物流"`; monkeypatch retrieval to raise if called; fake Agent LLM returns a final text; assert retrieval not called.
- `test_complaint_and_chitchat_do_not_call_llm`: intents `"投诉"` / `"闲聊"`; pass an Agent LLM that raises if `.stream` is called; assert fixed answers and options for complaint.
- `test_fixed_paths_append_ai_message`: after complaint/chitchat/gate-fail, `final_state["messages"][-1]` is an `AIMessage`.
- `test_checkpointer_keeps_history_across_turns`: invoke twice with different messages; assert the second turn's state messages contain the first assistant `AIMessage` before the second `HumanMessage`.
- `test_parent_stream_receives_subgraph_custom_frames`: business intent `"物流"`, Agent LLM scripted with a tool chunk then a text chunk; `graph = build_graph(..., checkpointer=InMemorySaver())`; collect `list(graph.stream(inputs, config, stream_mode="custom", subgraphs=True, version="v2"))`; assert the child `tool` frame and final `token` frame are both present. This pins the propagation that Task 10 depends on.
- `test_sqlite_checkpointer_creates_file`: use `tmp_path` and `sqlite_checkpointer`; assert file exists and `setup()` did not raise.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_graph.py -v`
Expected: FAIL with `ModuleNotFoundError: app.agent.graph`.

- [ ] **Step 3: Implement the parent graph**

- `open_sqlite_checkpointer`: create parent dir, `sqlite3.connect(path, check_same_thread=False)`, `PRAGMA journal_mode=WAL`, `PRAGMA busy_timeout=5000`, construct `SqliteSaver(conn)`, call `setup()`, return `CheckpointerHandle(saver=saver, close=conn.close)`. `sqlite_checkpointer` wraps it in a context manager for tests.
- `build_graph`: build the Agent subgraph via `build_agent_subgraph(llm, max_steps=..., token_budget=..., tool_result_max_chars=..., max_context_tokens=...)`; add nodes/edges exactly as spec §6.1; compile with the supplied checkpointer.
- `resolve_reference`: return `{"resolved_message": state["user_message"]}` and log `ch05 resolve_reference passthrough`.
- `classify_intent`: call `classify_intent(intent_llm, state["resolved_message"])`; return `{"intent": result.intent}`.
- `route_intent`: ordinary node returning `{"route": route_for_intent(state["intent"])}`; the conditional edge maps `knowledge`→`retrieve_knowledge`, `business`→`agent`, `complaint`→`complaint_reply`, `chitchat`→`chitchat_reply`.
- `retrieve_knowledge`: call `retrieve_hybrid(state["resolved_message"])`; on exception set `gate_reason="infra_error"`, `answer=INFRA_TEXT`, `sources=[]`; on `rerank_failed` set `rerank_failed=True`, `gate_reason="rerank_failed"`, `answer=RERANK_FAILED_TEXT`; otherwise build ordered evidence + sources, call `assess_confidence(result.chunks, settings.confidence_min_score, rerank_failed=result.rerank_failed)`, and set `confidence`, `gate_passed`, `gate_reason`.
- `confidence_gate`: if `gate_reason` is `infra_error` or `rerank_failed`, emit the corresponding fallback token, append `AIMessage`, and do not call `assess_confidence` (so infra failures never become `empty`). Otherwise, if `gate_passed` is true, emit the sources frame via `get_stream_writer`; if false (`empty` / `below_threshold`), emit the knowledge-gap fallback and append `AIMessage`.
- `complaint_reply`: fixed text + both options; append `AIMessage`.
- `chitchat_reply`: fixed text; append `AIMessage`.
- `record_log`: log the structured JSON fields from spec §6.4; return `{}`.

- [ ] **Step 4: Run the graph tests and full suite**

Run:
```bash
.venv/Scripts/python.exe -m pytest tests/test_agent_graph.py -v
.venv/Scripts/python.exe -m pytest -q
```
Expected: all pass.

- [ ] **Step 5: Commit and record dev-notes**

```bash
git add app/agent/graph.py tests/test_agent_graph.py
git commit -m "feat: add ch05 parent workflow graph"
```
Append `## 阶段：Task 9 完成（执行日期）`; record graph routes, gate behavior, and checkpoint test.

---

### Task 10: `/ch05/chat` service, SSE mapping, DB audit, low-confidence pool

**Files:**
- Create: `app/agent/service.py`
- Create: `tests/test_agent_service.py`
- Modify: `tests/agent_fakes.py`
- Modify: `dev-notes/ch05.md`

**Interfaces:**
- Produces:
```python
def stream_chat_agent(
    graph,
    session,
    user_id: str,
    message: str,
    *,
    settings: Settings,
) -> Iterator[str]
```
- Yields raw SSE payload strings exactly like `app.services.chat.stream_chat`: plain tokens for text, JSON strings for `tool`/`sources`/`options`/`error`.
- Inputs reset every transient state field explicitly: `user_message`, `resolved_message`, `intent=""`, `route=""`, `evidence=[]`, `sources=[]`, `confidence=None`, `gate_passed=False`, `gate_reason=""`, `rerank_failed=False`, `answer=""`, `suggested_actions=[]`, `step_count=0`, `token_estimate=0`, `exhausted=False`, `conversation_id=<cid>`, `token_budget`/`max_steps` from settings. `messages` gets one new `HumanMessage`. Reset `rerank_failed` is required so a later turn cannot inherit a stale `True`.
- Consumes `graph.stream(inputs, config, stream_mode="custom", subgraphs=True, version="v2")`; on langgraph 1.2.20 each item is a dict `{"type":"custom","ns":...,"data":<node frame>}`, so read `item["data"]`. Filters the node frames to `token`/`tool`/`sources`; ignores unknown frames with a log.
- After the stream: `graph.get_state(config).values`; emit `options` if `suggested_actions`; persist assistant `answer`; write low-confidence row only for `gate_reason in {"empty","below_threshold"}`.
- On graph exception: log, yield an `{"error": ...}` frame; do not re-raise.

- [ ] **Step 1: Write the failing service tests**

Add `FakeGraph` to `tests/agent_fakes.py` (Task 11 reuses it):
```python
class FakeGraph:
    def __init__(self, frames, final):
        self.frames = frames
        self.final = final
        self.inputs = None
        self.config = None

    def stream(self, inputs, config, stream_mode, subgraphs, version):
        self.inputs = inputs
        self.config = config
        assert stream_mode == "custom" and subgraphs is True
        for frame in self.frames:
            # Real langgraph 1.2.20 shape for custom + subgraphs=True.
            yield {"type": "custom", "ns": (), "data": frame}

    def get_state(self, config):
        from types import SimpleNamespace
        return SimpleNamespace(values=self.final)
```
Tests:
- `test_service_forwards_tokens_tools_sources_and_options`: frames `[{"type":"tool",...},{"type":"sources",...},{"type":"token","text":"答"}]`, final `suggested_actions=[{"action":"transfer","label":"转人工"}]`; assert events include the JSON frames in order and an options frame last.
- `test_service_resets_transient_state`: assert `graph.inputs` has `step_count=0`, `evidence=[]`, `gate_passed=False`, `gate_reason=""`, `rerank_failed=False`, `conversation_id` set, and `messages` contains one `HumanMessage`.
- `test_service_persists_assistant_answer`: after run, `Message` rows are `user`, `assistant` with the answer.
- `test_service_records_low_confidence_for_gate_gap`: final `route="knowledge"`, `gate_passed=False`, `gate_reason="below_threshold"`; assert a `LowConfidenceQuestion` row with `source="confidence_gate"`.
- `test_service_does_not_pool_infra_or_rerank_failure`: parameterize `gate_reason` `infra_error`/`rerank_failed`; assert no row.
- `test_service_graph_error_yields_error_frame`: `FakeGraph.stream` raises; assert one error JSON frame.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_service.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the service**

Use `get_or_create_conversation`, `append_message`, and `record_low_confidence` from the existing services. Persist the user message before starting the graph and commit; persist the final assistant answer and commit after. The `thread_id` is `f"user:{user_id}"`. Convert custom frames to SSE payloads; JSON-dump with `ensure_ascii=False, separators=(",", ":")`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_service.py -v`
Expected: all pass.

- [ ] **Step 5: Commit and record dev-notes**

```bash
git add app/agent/service.py tests/test_agent_service.py
git commit -m "feat: add ch05 chat service and sse mapping"
```
Append `## 阶段：Task 10 完成（执行日期）`.

---

### Task 11: FastAPI routes, lazy graph wiring, and ticket endpoint

**Files:**
- Create: `app/api/ch05.py`
- Modify: `app/main.py`
- Modify: `app/schemas.py`
- Create: `sql/ch05_low_confidence_source.sql`
- Create: `tests/test_ch05_api.py`
- Modify: `tests/agent_fakes.py`
- Modify: `dev-notes/ch05.md`

**Interfaces:**
- Produces:
```python
# app/schemas.py
class TicketRequest(BaseModel):
    session_id: str
    description: str
    ticket_type: Literal["售后", "投诉", "咨询"] = "咨询"

# app/api/ch05.py
api_router = APIRouter(prefix="/ch05")
def get_ch05_graph_dep(request: Request): ...
```
- Task 11 implements only the API routes. Task 12 adds `page_router` + `GET /ch05` when the HTML file exists.
- `POST /ch05/chat` uses `ChatRequest`, `EventSourceResponse`, `stream_chat_agent`, and appends `data: [DONE]`.
- `POST /ch05/tickets` calls `execute_tool(create_ticket, args, timeout_s=5, retries=0)`, validates `TICKET_NO_PATTERN`, returns `{"ticket_no": ..., "message": "工单已创建"}` or 502.
- `app/main.py` uses a lifespan that only closes an existing lazy graph on exit; it does not build the graph at startup. `get_ch05_graph_dep` lazily builds the graph on the first `/ch05` request with `open_sqlite_checkpointer(settings.langgraph_checkpoint_db)` + `build_graph(...)`, caches it on `app.state.ch05_graph` under a lock, and returns it. This keeps existing tests that use `with TestClient(app)` from constructing real LLMs or touching `.data/`.
- `get_ch05_graph_dep` returns the cached graph; if construction raises, propagate a 503.

- [ ] **Step 1: Write the failing API tests**

Define a local `client` fixture in `tests/test_ch05_api.py` (the fixture in `tests/test_api.py` is module-local) and use `FakeGraph` from `tests/agent_fakes.py`:
```python
@pytest.fixture
def client(session):
    app.dependency_overrides[get_settings] = lambda: Settings(openai_api_key="test")
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_ch05_chat_streams_frames_and_done(client):
    app.dependency_overrides[get_ch05_graph_dep] = lambda: FakeGraph(...)
    with client.stream("POST", "/ch05/chat", json={"session_id": "s1", "message": "你好"}) as r:
        body = "".join(r.iter_text())
    assert "data: [DONE]" in body
    assert "options" in body or "token" in body


def test_ch05_ticket_writes_ticket_without_transfer(session):
    # create a conversation; post /ch05/tickets; assert Ticket exists and
    # Conversation.status == "进行中"
    ...
```
Also add tests:
- Blank message 422.
- Graph error yields `error` frame + `[DONE]`.
- Ticket invalid type 422.
- Ticket failure 502 (monkeypatch `execute_tool` to return `工具执行失败`).
- Lazy graph build failure → 503 (override `get_ch05_graph_dep` with a function that raises `HTTPException(status_code=503)`).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ch05_api.py -v`
Expected: FAIL with 404/ImportError.

- [ ] **Step 3: Implement schemas, routes, and lifespan**

- Add `TicketRequest` with a strip validator like `ChatRequest`.
- `app/api/ch05.py`: API router only (no page route yet), SSE route with the same error-frame pattern as `/chat`, and ticket route. For the ticket route, resolve conversation with `get_or_create_conversation(session, req.session_id)`, set `current_conversation_id`, call `execute_tool`, validate `TICKET_NO_PATTERN`, return 502 on failure.
- `app/main.py`: add a lifespan that only closes `app.state.ch05_checkpoint_handle` if it exists; keep existing routers; include the ch05 API router. `get_ch05_graph_dep(request, settings=Depends(get_settings))` lazily calls `open_sqlite_checkpointer`, builds under `app.state.ch05_graph_lock`, stores `app.state.ch05_graph` + `app.state.ch05_checkpoint_handle`, and returns the graph. Add a `close_ch05_graph(app)` helper that calls `handle.close()`.
- Ensure `app/main.py` import does not require an API key at import time; graph construction happens on the first `/ch05` request.
- Create `sql/ch05_low_confidence_source.sql` for existing MySQL databases:
  ```sql
  SET NAMES utf8mb4;
  ALTER TABLE low_confidence_questions
    MODIFY source ENUM('retrieval_low_conf','self_check','user_feedback','confidence_gate')
    NOT NULL COMMENT '入池入口:检索证据低 / 生成自评不足 / 用户反馈未解决 / ch05 置信度闸知识缺口';
  ```
  Verify it contains `confidence_gate` and preserves the column COMMENT. New databases already get the value from the updated ch04 DDL; this file is for existing ones.

- [ ] **Step 4: Run the tests and full suite**

Run:
```bash
.venv/Scripts/python.exe -m pytest tests/test_ch05_api.py -v
.venv/Scripts/python.exe -m pytest -q
```
Expected: all pass.

- [ ] **Step 5: Commit and record dev-notes**

```bash
git add app/api/ch05.py app/main.py app/schemas.py sql/ch05_low_confidence_source.sql tests/test_ch05_api.py
git commit -m "feat: add ch05 chat and ticket api"
```
Append `## 阶段：Task 11 完成（执行日期）`.

---

### Task 12: Native `/ch05` chat page with independent buttons

**Files:**
- Create: `app/static/ch05.html`
- Create: `tests/test_ch05_page.py`
- Modify: `app/api/ch05.py`
- Modify: `app/main.py`
- Modify: `dev-notes/ch05.md`

**Interfaces:**
- Page uses `POST /ch05/chat` and `POST /ch05/tickets`.
- `sessionStorage` key: `lamp-cs-ch05-session-id`.
- Buttons: exact labels `转人工` and `建工单`.
- Transfer click: appends `已转接人工客服`, then `您好，我是客服小猫，请问有什么可以帮您的？`; no fetch.
- Ticket click: `confirm("确认创建工单吗？")`; POST `/ch05/tickets` with the last user message and the option's `ticket_type`; on success shows the ticket number; on failure shows an error.
- Buttons are independent; clicking one does not disable or trigger the other; ignoring both leaves the composer usable.

- [ ] **Step 1: Write the failing page test**

```python
# tests/test_ch05_page.py
from pathlib import Path


def test_ch05_page_has_required_contract_strings():
    html = (Path(__file__).parents[1] / "app" / "static" / "ch05.html").read_text(
        encoding="utf-8"
    )
    for text in [
        "转人工", "建工单", "已转接人工客服", "客服小猫",
        "/ch05/chat", "/ch05/tickets", "lamp-cs-ch05-session-id",
    ]:
        assert text in html
    assert "confirm(" in html
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ch05_page.py -v`
Expected: FAIL with `FileNotFoundError`.

- [ ] **Step 3: Implement the page**

- Self-contained HTML with inline CSS/JS, matching the repo's warm dark style.
- Add `page_router = APIRouter()` + `GET /ch05` in `app/api/ch05.py` returning `FileResponse(app/static/ch05.html, media_type="text/html")`; register `page_router` in `app/main.py`.
- Render user/assistant/system bubbles; parse SSE manually with `fetch` + `TextDecoder`.
- Handle `tool` frames as a muted status line; ignore `sources` or show a compact “已检索知识库” note (optional).
- Handle `options` frames by appending buttons under the current assistant bubble.
- Transfer handler: append two local system/bot messages; no backend call.
- Ticket handler: confirm, POST JSON, append success/error; only disable the clicked button.
- Composer remains enabled throughout; a new send does not auto-click buttons.

- [ ] **Step 4: Run the page test and full suite**

Run:
```bash
.venv/Scripts/python.exe -m pytest tests/test_ch05_page.py -v
.venv/Scripts/python.exe -m pytest -q
```
Expected: all pass.

- [ ] **Step 5: Manual browser smoke**

Start the app:
```bash
.venv/Scripts/python.exe run.py
```
Open `http://localhost:8000/ch05`, send “你好”, and verify a fixed reply. Then send “我要投诉” and verify the two buttons render independently. Do not click anything yet; this step only checks rendering.

- [ ] **Step 6: Commit and record dev-notes**

```bash
git add app/static/ch05.html tests/test_ch05_page.py app/api/ch05.py app/main.py
git commit -m "feat: add native ch05 chat page"
```
Append `## 阶段：Task 12 完成（执行日期）`; record the rendering smoke result.

---

### Task 13: Live acceptance, full verification, and handoff

**Files:**
- Create: `docs/superpowers/reports/2026-11-17-ch05-acceptance.md`
- Modify: `dev-notes/ch05.md`

**Interfaces:**
- No new production interfaces. This task validates the five acceptance criteria and produces the demo evidence.

- [ ] **Step 1: Start the app and confirm the old `/chat` still works**

Run:
```bash
.venv/Scripts/python.exe run.py
```
In another shell:
```bash
curl -sS -N -X POST http://127.0.0.1:8000/chat -H 'Content-Type: application/json' --data-binary @- <<'JSON'
{"session_id":"old-chat-check","message":"你好"}
JSON
```
Expected: SSE with `[DONE]`; no regression in the old endpoint.

- [ ] **Step 2: Acceptance 1 — policy question forces retrieval**

Use `/ch05` with “你们的退货政策是什么？”. Verify:
- server log contains `ch05 retrieval_forced` and `ch05 confidence_gate`;
- an SSE `sources` frame is present;
- the answer cites `[n]` and matches evidence.
Record the log lines and response in the acceptance report.

- [ ] **Step 3: Acceptance 2 — order logistics tool call**

Send “订单 1001 的物流到哪了”. Verify a `query_logistics` tool frame and an answer based on the tool result. Record it.

- [ ] **Step 4: Acceptance 3 — complaint buttons and separation**

Send “我要投诉”. Verify both buttons render. Then:
- click “转人工”: frontend shows `已转接人工客服` and the 客服小猫 greeting; confirm no `tickets` row was created and no backend request was made;
- click “建工单” on a fresh complaint: confirm dialog, then verify a `tickets` row exists and `conversations.status` is still `进行中`;
- send another normal message without clicking either button: verify the conversation continues and no action occurs.
Record DB query results and screenshots.

- [ ] **Step 5: Acceptance 4 — chitchat fixed reply**

Send “你好”. Verify the fixed chitchat text and that the server did not call the Agent/retrieval. Record the log.

- [ ] **Step 6: Acceptance 5 — multi-step ReAct**

Send “先帮我确认订单 1001 的状态，如果已经发货，再查它的物流到哪了”. Verify logs/tool frames show at least two Agent steps (`query_order` then `query_logistics`). If the model calls both in parallel, tighten the serial-tool prompt and retest; record the final behavior. If it repeatedly fails, report the exact observed behavior rather than claiming pass.

- [ ] **Step 7: Full test suite and report**

Run:
```bash
.venv/Scripts/python.exe -m pytest -q
```
Expected: all tests pass. Write `docs/superpowers/reports/2026-11-17-ch05-acceptance.md` with the commands, actual outputs, acceptance pass/fail, screenshots, and any residual limitations.

- [ ] **Step 8: Commit and record dev-notes**

```bash
git add docs/superpowers/reports/2026-11-17-ch05-acceptance.md dev-notes/ch05.md
git commit -m "docs: record ch05 acceptance results"
```
Append `## 阶段：验收完成（执行日期）` with the five criteria and actual evidence.

- [ ] **Step 9: Code review and finish**

Invoke `superpowers:requesting-code-review` for the branch; fix findings and re-review. Then invoke `superpowers:finishing-a-development-branch`. Record the review conclusion and finish decision in `dev-notes/ch05.md` before merging/PR.
