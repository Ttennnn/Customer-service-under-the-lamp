# Ch07 会话上下文管理 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 ch06 的全量历史直塞模型升级为当前会话内的三层上下文：L1 原文、L2 规则截短、L3 异步分段摘要，预算从模型窗口倒推，并配多会话侧栏。

**Architecture:** runtime 在每轮入口从 checkpoint State 按 `db-{msg_id}` 用户锚点切出 L2/L1，产出 `prompt_history`；Agent 按固定顺序拼装 system → L2 → L1 → 当前用户 → 摘要+证据材料。L1 超预算只挪 `layer1_from_msg_id`；L2 超预算由线程池后台追加 `conversation_summaries` 一段并挪 `summary_upto_msg_id`。State 的 `messages` 由 `add_messages` 全量保留，checkpoint 与精简版互不影响。

**Tech Stack:** FastAPI, SQLAlchemy, MySQL/SQLite, LangGraph 1.2.20 (`add_messages` + `SqliteSaver`), LangChain `trim_messages`, pytest, 原生 HTML/CSS/JS。

**Spec:** `docs/superpowers/specs/2026-11-18-ecommerce-cs-ch07-context-management-design.md`

## Global Constraints

- 演示预算必须算出 `history=5650, layer1=3955, layer2=1695`；L1+L2=5650（用户写的 3954 是 70% 取整差 1，不改公式）。
- 只改 `MODEL_CONTEXT_WINDOW=18000`、其余默认时 `history=0`，启动日志必须出现 `上下文预算不足`。
- 默认 `MODEL_CONTEXT_WINDOW=65536` 下 20 轮稳态历史（20×300=6000）不触发降级/摘要；规划目标 30×300=9000，L1=6300 覆盖这 6000，L2=2700 作缓冲。
- 预算公式：`history = max(0, min(history_target_turns*history_tokens_per_turn, window - fixed - transient))`；`fixed = 1200 + rerank_top_k*250 + max(500, summary_tokens) + max_output + 10%*window`；`transient = max_user_input + max_agent_steps*tool_result_max`。
- 中文 token 口径固定 1 字符=1 token；所有预算常数按此校准，改估算必须同时重标常数。
- 摘要只提炼事实与诉求；不得编造；寒暄不留；prompt 目标 50–200 字，runtime 接受 20–400 字。
- 摘要一段一行只追加；旧段永不 UPDATE；`conversations.summary` 只做 `旧投影 + 新段` 追加。
- system 每个 route 恒定；摘要/证据/订单/意图一律不进 system；订单选择器说明不是 SystemMessage。
- `[订单选择]` 审计行不进前端历史；摘要任务按「用户选择了订单 N」系统事实行处理。
- `MAX_CONTEXT_TOKENS` / `AGENT_TOKEN_BUDGET` 不再 gate ch07 Agent；步数由 `MAX_AGENT_STEPS` 控。
- 工具结果不落 `messages` 表；摘要触发看 L2 渲染后的 token 用量，不看条数。
- 后台摘要用线程池 + 每会话在飞标记；提交即返回，不能用 `asyncio.create_task`。
- 每完成一个任务，先追加 `dev-notes/ch07.md` 的该任务段，再 commit。
- 本计划评审通过后，先在 `dev-notes/ch07.md` 追加“计划评审通过”段（用户原话、计划路径、评审结论），再开始 Task 1。

## Review Focus

- **Tool-heavy turn:** 大块工具结果在 L2 必须变成一行标识，L1 只在降级后进入 L2；触发用量要把工具结果算进去，摘要正文不能把工具 JSON 当用户发言。测试放在 Task 3/4/5。
- **Interrupt/resume:** resume 必须落在 `conv:{cid}` 同一 thread；订单选择器不是 system；不重复写用户消息。测试放在 Task 8。
- **旧会话/重启:** State 只覆盖新回合时，MySQL 历史必须每轮补到窗口前面，第二轮不能丢早期订单。测试放在 Task 4。
- **前缀缓存:** system 必须逐轮完全相同；摘要/证据/订单/意图/订单选择器都不能以 SystemMessage 出现在历史里。测试放在 Task 6。
- **多会话副作用:** 退款/工单必须归当前会话，消息接口不能越权读别人会话，切会话时不能串流。测试放在 Task 8/9。

---

### Task 1: 表结构、ORM 与会话/摘要 repository

**Files:**
- Create: `sql/ch07-ddl.sql`
- Create: `sql/ch07-layers.sql`
- Create: `app/models/conversation_summary.py`
- Modify: `app/models/conversation.py`
- Modify: `app/models/__init__.py`
- Modify: `app/services/history.py`
- Test: `tests/test_ch07_schema.py`
- Test: `tests/test_history.py`

**Interfaces:**
- Consumes: `app.db.base.Base`, existing `Conversation`, `Message`, `session` fixture.
- Produces:
  - `Conversation.summary: str | None`, `Conversation.summary_upto_msg_id: int | None`, `Conversation.layer1_from_msg_id: int | None`
  - `ConversationSummary(id, conversation_id, seq, from_msg_id, upto_msg_id, content, created_at)`
  - `ConversationNotFound(Exception)`
  - `create_conversation(session, user_id: str) -> Conversation`
  - `get_conversation(session, conversation_id: int, user_id: str | None = None) -> Conversation | None`
  - `resolve_conversation(session, user_id: str, conversation_id: int | None, new_conversation: bool) -> Conversation`
  - `list_conversations(session, user_id: str) -> list[dict]` (keys `id, preview, has_summary, created_at, updated_at`)
  - `list_dialog_messages(session, conversation_id: int) -> list[Message]` (user/assistant, id asc)
  - `previous_message_id(session, conversation_id: int, before_id: int) -> int | None`
  - `update_layer1_from(session, conversation_id: int, boundary: int | None) -> None`
  - `append_summary_segment(session, conversation_id: int, from_msg_id: int, upto_msg_id: int, content: str) -> ConversationSummary`
  - `list_summary_segments(session, conversation_id: int) -> list[ConversationSummary]`
  - `update_conversation_summary(session, conversation_id: int, summary: str, upto_msg_id: int) -> None`

- [ ] **Step 1: Write the failing schema/model tests**

`tests/test_ch07_schema.py`:

```python
from app.models import Conversation, ConversationSummary


def test_conversation_context_columns_roundtrip(session):
    conv = Conversation(user_id="u1", summary="第1段：订单 1001 退款", summary_upto_msg_id=42, layer1_from_msg_id=80)
    session.add(conv); session.commit(); session.refresh(conv)
    assert conv.summary == "第1段：订单 1001 退款"
    assert conv.summary_upto_msg_id == 42
    assert conv.layer1_from_msg_id == 80


def test_conversation_summary_unique_seq(session):
    from sqlalchemy.exc import IntegrityError
    import pytest
    session.add(ConversationSummary(conversation_id=1, seq=1, from_msg_id=1, upto_msg_id=2, content="a"))
    session.commit()
    session.add(ConversationSummary(conversation_id=1, seq=1, from_msg_id=3, upto_msg_id=4, content="b"))
    with pytest.raises(IntegrityError):
        session.commit()
```

- [ ] **Step 2: Run the schema tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ch07_schema.py -v`
Expected: FAIL with `ImportError: cannot import name 'ConversationSummary'`.

- [ ] **Step 3: Add the two DDL files and ORM models**

`sql/ch07-ddl.sql` and `sql/ch07-layers.sql` must be the user's two DDL blocks verbatim (including `SET NAMES utf8mb4` and Chinese comments). Add `summary`, `summary_upto_msg_id`, `layer1_from_msg_id` to `Conversation` using `Mapped[Optional[str]]` / `Mapped[Optional[int]]` with `Text` / `BigInteger`. Add `ConversationSummary` with `BigInteger().with_variant(Integer, "sqlite")` for `id`, `UniqueConstraint("conversation_id", "seq", name="uk_conv_seq")`, and `Index("idx_conv_upto", "conversation_id", "upto_msg_id")`. Export it in `app/models/__init__.py`.

- [ ] **Step 4: Run the schema tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ch07_schema.py -v`
Expected: PASS.

- [ ] **Step 5: Write the failing repository tests**

Append to `tests/test_history.py`:

```python
from app.models import Conversation, ConversationSummary
from app.services.history import (
    ConversationNotFound, append_message, append_summary_segment, create_conversation,
    get_conversation, list_conversations, list_dialog_messages, previous_message_id,
    resolve_conversation, update_conversation_summary, update_layer1_from,
)


def test_resolve_conversation_new_existing_and_foreign(session):
    c1 = resolve_conversation(session, "u1", None, True)
    c2 = resolve_conversation(session, "u1", c1.id, False)
    assert c2.id == c1.id
    assert get_conversation(session, c1.id, "u2") is None
    with pytest.raises(ConversationNotFound):
        resolve_conversation(session, "u2", c1.id, False)


def test_list_conversations_newest_first_preview_and_summary(session):
    c1 = create_conversation(session, "u1")
    append_message(session, c1.id, "user", "订单 1001 想退款")
    append_message(session, c1.id, "assistant", "好的")
    update_conversation_summary(session, c1.id, "第1段：订单 1001 退款", 2)
    c2 = create_conversation(session, "u1")
    append_message(session, c2.id, "user", "物流到哪")
    rows = list_conversations(session, "u1")
    assert [row["id"] for row in rows] == [c2.id, c1.id]
    assert rows[1]["preview"].startswith("订单 1001")
    assert rows[1]["has_summary"] is True
    assert rows[0]["has_summary"] is False


def test_list_dialog_messages_and_previous_id(session):
    cid = create_conversation(session, "u1").id
    u1 = append_message(session, cid, "user", "问1")
    a1 = append_message(session, cid, "assistant", "答1")
    u2 = append_message(session, cid, "user", "问2")
    session.commit()
    rows = list_dialog_messages(session, cid)
    assert [row.content for row in rows] == ["问1", "答1", "问2"]
    assert previous_message_id(session, cid, u2.id) == a1.id


def test_append_summary_segment_appends_projection_and_advances_upto(session):
    cid = create_conversation(session, "u1").id
    seg1 = append_summary_segment(session, cid, 1, 4, "第1段：订单 1001")
    seg2 = append_summary_segment(session, cid, 5, 8, "第2段：已退款")
    assert (seg1.seq, seg2.seq) == (1, 2)
    conv = get_conversation(session, cid)
    assert conv.summary_upto_msg_id == 8
    assert "第1段：订单 1001" in conv.summary and "第2段：已退款" in conv.summary
    assert [s.content for s in session.query(ConversationSummary).order_by(ConversationSummary.seq)] == [
        "第1段：订单 1001", "第2段：已退款",
    ]


def test_update_layer1_from_only_moves_anchor(session):
    cid = create_conversation(session, "u1").id
    update_layer1_from(session, cid, 42)
    assert get_conversation(session, cid).layer1_from_msg_id == 42
```

- [ ] **Step 6: Run the repository tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_history.py -v`
Expected: FAIL with `ImportError` / `AttributeError` for the new functions.

- [ ] **Step 7: Implement the repository functions in `app/services/history.py`**

Use the existing `get_or_create_conversation`, `append_message`, `load_history` patterns. Implement:
- `create_conversation`: add `Conversation(user_id=user_id, status="进行中")`, commit, refresh.
- `get_conversation`: filter by id and optional user_id.
- `resolve_conversation`: new → create; id → get or raise `ConversationNotFound`; neither → `get_or_create_conversation`.
- `list_conversations`: query the user's conversations `id desc`; for each, first user message content (correlated subquery or one query per conversation), `preview = content[:30]`, `has_summary = bool(summary)`.
- `list_dialog_messages`: `role.in_(("user", "assistant"))`, `id asc`.
- `previous_message_id`: `id < before_id`, `id desc`, first.
- `update_layer1_from`: set + commit.
- `append_summary_segment`: compute `seq = max(seq)+1`, insert row, set `conversation.summary = (old + "\n\n" + f"第{seq}段:{content}")` or content, `summary_upto_msg_id = upto_msg_id`, commit, refresh.
- `list_summary_segments`: `id asc`.
- `update_conversation_summary`: set summary + upto, commit.

- [ ] **Step 8: Run the repository and schema tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_history.py tests/test_ch07_schema.py -v`
Expected: PASS.

- [ ] **Step 9: Append the Task 1 dev-note and commit**

Append to `dev-notes/ch07.md` under a new `## 阶段：任务 1 完成（表结构与 repository）` section: what was built, test command/result, any deviation.

```bash
git add docs/superpowers/specs/2026-11-18-ecommerce-cs-ch07-context-management-design.md docs/superpowers/plans/2026-11-18-ecommerce-cs-ch07-context-management.md sql/ch07-ddl.sql sql/ch07-layers.sql app/models/conversation.py app/models/conversation_summary.py app/models/__init__.py app/services/history.py tests/test_ch07_schema.py tests/test_history.py dev-notes/ch07.md
git commit -m "feat: add ch07 context columns, summary table and repository"
```

---

### Task 2: 窗口预算计算与启动自检

**Files:**
- Modify: `app/config.py`
- Modify: `.env.example`
- Create: `app/services/context_budget.py`
- Test: `tests/test_ch07_config.py`

**Interfaces:**
- Consumes: `Settings`, `estimate_tokens` (Task 3 not yet needed; summary_tokens passed in).
- Produces:
  - `Settings` fields: `model_context_window=65536`, `max_output_tokens=2000`, `max_user_input_tokens=2000`, `max_agent_steps: int | None = None`, `tool_result_max_tokens: int | None = None`, `rerank_top_k: int | None = None`, `safety_margin_ratio=0.10`, `system_prompt_reserve_tokens=1200`, `evidence_tokens_per_chunk=250`, `summary_reserve_tokens=500`, `history_target_turns=30`, `history_tokens_per_turn=300`, `layer1_ratio=0.7`, `layer1_degrade_low_water_ratio=0.8`, `assistant_l2_chars=40`, `tool_result_marker_chars=200`, `summary_model=""`, `summary_max_chars=200`, `ch07_checkpoint_db=".data/ch07-checkpoints.sqlite"`, `context_log_preview_chars=40`, `log_to_file=True`.
  - Properties: `effective_max_agent_steps`, `effective_tool_result_max_tokens`, `effective_rerank_top_k`.
  - `build_llm(..., max_tokens: int | None = None)` passes `max_tokens or settings.max_output_tokens` to `ChatOpenAI`.
  - `ContextBudget` dataclass with `window, fixed, transient, window_surplus, target, history, layer1, layer2, safety, summary_tokens`.
  - `compute_budget(settings: Settings, summary_tokens: int = 0) -> ContextBudget`
  - `validate_budget(settings: Settings) -> ContextBudget`

- [ ] **Step 1: Write the failing config/budget tests**

`tests/test_ch07_config.py`:

```python
import pytest
from app.config import Settings
from app.services.context_budget import compute_budget, validate_budget


def _base(**kw):
    return Settings(_env_file=None, openai_api_key="x", **kw)


def test_effective_aliases():
    s = _base(max_agent_steps=3, tool_result_max_tokens=1200, rerank_top_k=5)
    assert s.effective_max_agent_steps == 3
    assert s.effective_tool_result_max_tokens == 1200
    assert s.effective_rerank_top_k == 5
    s2 = _base()
    assert s2.effective_max_agent_steps == s2.agent_max_steps
    assert s2.effective_tool_result_max_tokens == s2.agent_tool_result_max_chars
    assert s2.effective_rerank_top_k == s2.rerank_top_n


def test_demo_budget_exact_numbers():
    s = _base(model_context_window=18000, max_output_tokens=2000, max_user_input_tokens=2000,
              max_agent_steps=3, tool_result_max_tokens=1200, rerank_top_k=5)
    b = compute_budget(s)
    assert (b.history, b.layer1, b.layer2) == (5650, 3955, 1695)
    assert b.layer1 + b.layer2 == b.history


def test_default_budget_keeps_twenty_turns_raw():
    b = compute_budget(_base(model_context_window=65536))
    assert b.history == 9000
    assert (b.layer1, b.layer2) == (6300, 2700)
    assert b.layer1 >= 20 * 300  # 20 轮稳态原文装得下


def test_window_only_goes_to_zero_and_warns(caplog):
    s = _base(model_context_window=18000)
    b = validate_budget(s)
    assert b.history == 0
    assert any("上下文预算不足" in rec.message for rec in caplog.records)
```

- [ ] **Step 2: Run the budget tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ch07_config.py -v`
Expected: FAIL with `ImportError: app.services.context_budget`.

- [ ] **Step 3: Implement the settings and budget module**

Add the fields/properties to `Settings` and document the new env vars in `.env.example`. Implement `compute_budget` exactly per the Global Constraints:
```python
import math
safety = math.ceil(settings.model_context_window * settings.safety_margin_ratio)
fixed = (
    settings.system_prompt_reserve_tokens
    + settings.effective_rerank_top_k * settings.evidence_tokens_per_chunk
    + max(settings.summary_reserve_tokens, summary_tokens)
    + settings.max_output_tokens
    + safety
)
transient = (
    settings.max_user_input_tokens
    + settings.effective_max_agent_steps * settings.effective_tool_result_max_tokens
)
surplus = settings.model_context_window - fixed - transient
target = settings.history_target_turns * settings.history_tokens_per_turn
history = max(0, min(target, surplus))
layer1 = history * 7 // 10
layer2 = history - layer1
```
`validate_budget` logs `context budget window=... fixed=... transient=... history=... layer1=... layer2=...` at INFO and `logger.warning("上下文预算不足 ...")` when `history < settings.history_tokens_per_turn or layer1 <= 0`.

- [ ] **Step 4: Run the budget tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ch07_config.py -v`
Expected: PASS.

- [ ] **Step 5: Append dev-note and commit**

```bash
git add app/config.py .env.example app/services/context_budget.py tests/test_ch07_config.py dev-notes/ch07.md
git commit -m "feat: derive ch07 context budget from the model window"
```

---

### Task 3: 分层切窗、L2 渲染与 prompt 拼装 helper

**Files:**
- Modify: `app/services/context.py`
- Create: `tests/test_ch07_layers.py`

**Interfaces:**
- Consumes: `estimate_tokens`; `Settings` values; LangChain `trim_messages`.
- Produces:
  - `message_tokens(messages: Sequence[BaseMessage]) -> int` (content tokens + 4/message)
  - `split_layers(messages, summary_upto_msg_id: int | None, layer1_from_msg_id: int | None) -> tuple[list[BaseMessage], list[BaseMessage]]` → `(l2, l1)`
  - `render_l2(messages: Sequence[BaseMessage], settings: Settings) -> list[BaseMessage]`
  - `choose_degradation(messages, layer1_from_msg_id: int | None, l1_budget: int, low_water_ratio: float) -> int | None` → target user `db-` id or None
  - `lc_trim_l1(messages: Sequence[BaseMessage], max_tokens: int) -> list[BaseMessage]`
  - `assemble_agent_prompt(system: SystemMessage, history: Sequence[BaseMessage], current: Sequence[BaseMessage], material: str | None, max_prompt_tokens: int) -> list[BaseMessage]`
  - `log_context(logger, tag: str, summary: str, messages: Sequence[BaseMessage], window_count: int, preview_chars: int = 40) -> None`

- [ ] **Step 1: Write the failing layer tests**

`tests/test_ch07_layers.py`:

```python
import logging
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from app.config import Settings
from app.services.context import (
    assemble_agent_prompt, choose_degradation, log_context, message_tokens,
    render_l2, split_layers,
)


def _settings():
    return Settings(_env_file=None, openai_api_key="x", assistant_l2_chars=40)


def _hist():
    return [
        HumanMessage(content="问1", id="db-10"),
        AIMessage(content="答1" * 40),
        HumanMessage(content="问2", id="db-20"),
        AIMessage(content="", tool_calls=[{"name": "query_order", "args": {"order_id": "1001"}, "id": "c1"}]),
        ToolMessage(content="X" * 500, tool_call_id="c1"),
        AIMessage(content="答2"),
        HumanMessage(content="问3", id="db-30"),
        AIMessage(content="答3"),
    ]


def test_split_layers_by_user_anchors():
    # layer1_from=10（问1 的 db id）→ L1 从下一条用户消息 问2 开始
    l2, l1 = split_layers(_hist(), summary_upto_msg_id=None, layer1_from_msg_id=10)
    assert [m.content for m in l2] == ["问1", "答1" * 40]
    assert l1[0].content == "问2" and l1[-1].content == "答3"


def test_split_layers_null_anchors():
    l2, l1 = split_layers(_hist(), None, None)
    assert l2 == [] and len(l1) == 8


def test_render_l2_keeps_user_full_prefixes_assistant_and_marks_tool():
    # summary_upto=10 → L2 从 问2 起；layer1_from=20 → L1 从 问3 起
    l2, _ = split_layers(_hist(), 10, 20)
    rendered = render_l2(l2, _settings())
    assert rendered[0].content == "问2"
    assert rendered[1].content.startswith("[调用工具: query_order]")
    assert rendered[2].content == "[工具结果: query_order 已省略 500 字]"
    assert len(rendered[3].content) <= 41  # 40 + ellipsis


def test_message_tokens_includes_role_overhead():
    assert message_tokens([HumanMessage(content="abcd")]) == 8


def test_choose_degradation_keeps_one_turn_and_drops_oldest_first():
    one_turn = [HumanMessage(content="问3", id="db-30"), AIMessage(content="答3")]
    assert choose_degradation(one_turn, 20, l1_budget=1, low_water_ratio=0.5) is None
    two_turns = [
        HumanMessage(content="问2", id="db-20"), AIMessage(content="答2"),
        HumanMessage(content="问3", id="db-30"), AIMessage(content="答3"),
    ]
    assert choose_degradation(two_turns, 20, l1_budget=1, low_water_ratio=0.5) == 30


def test_assemble_agent_prompt_inserts_material_after_user_before_tools():
    system = SystemMessage(content="SYS")
    current = [
        HumanMessage(content="当前问题"),
        AIMessage(content="", tool_calls=[{"name": "query_order", "args": {}, "id": "c1"}]),
        ToolMessage(content="工具", tool_call_id="c1"),
    ]
    prompt = assemble_agent_prompt(system, [], current, "【摘要】订单 1001", 10000)
    assert prompt[0] is system
    assert prompt[1].content == "当前问题"
    assert prompt[2].content.startswith("【摘要】")
    assert prompt[3].type == "ai"


def test_assemble_agent_prompt_trims_history_but_keeps_current_and_material():
    system = SystemMessage(content="SYS")
    history = [HumanMessage(content="旧" * 100, id=f"db-{i}") for i in range(5)]
    current = [HumanMessage(content="当前问题")]
    prompt = assemble_agent_prompt(system, history, current, "材料", 40)
    assert any(m.content == "当前问题" for m in prompt)
    assert any(m.content == "材料" for m in prompt)
    assert len(prompt) < 7


def test_log_context_contains_summary_preview_and_count(caplog):
    caplog.set_level(logging.INFO)
    logger = logging.getLogger("test.ch07")
    log_context(logger, "history_ctx", "第1段：订单 1001", [HumanMessage(content="问1")], 1)
    text = caplog.text
    assert "history_ctx" in text and "第1段：订单 1001" in text and "问1" in text and '"window_count": 1' in text
```

- [ ] **Step 2: Run the layer tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ch07_layers.py -v`
Expected: FAIL with `ImportError` for the new helpers.

- [ ] **Step 3: Implement the helpers in `app/services/context.py`**

Implementation rules:
- `_db_id(message)` parses `message.id` when it starts with `db-`.
- `split_layers`: find the first user index with db id > `summary_upto` (or 0), then the first user index with db id > `layer1_from` (or len). L2 = `[start:layer1_start]`, L1 = `[layer1_start:]`. If no user anchor is found, return `([], list(messages))`.
- `render_l2`: map `tool_call_id → tool name` from AIMessage tool_calls; Human full; AI `content[:assistant_l2_chars] + "…"`, with `[调用工具: names]` prefix when tool_calls; Tool > `tool_result_marker_chars` → `[工具结果: {name} 已省略 {len} 字]`, else content[:marker]; System → first 40 chars.
- `choose_degradation`: walk L1 user turns from oldest; compute tokens until `<= l1_budget * low_water_ratio`; always leave at least one turn; return the chosen user's db id, or None.
- `lc_trim_l1`: import `from langchain_core.messages import trim_messages as lc_trim_messages`; `token_counter=message_tokens`, `strategy="last"`, `start_on="human"`, `include_system=False`, `max_tokens=max_tokens`; if empty, return the last HumanMessage onward.
- `assemble_agent_prompt`: find the first HumanMessage in `current`; insert `HumanMessage(content=material)` after it (or at end if none); if `message_tokens(prompt) > max_prompt_tokens`, trim `history` with `lc_trim_messages` to `max_prompt_tokens - message_tokens(system/current/material)` and rebuild; log warning if still over.
- `log_context`: `logger.info("%s %s", tag, json.dumps({"summary": summary, "window_count": window_count, "tokens": message_tokens(messages), "messages": [{"role": type(m).__name__, "preview": str(m.content)[:preview_chars]} for m in messages]}, ensure_ascii=False))`.

- [ ] **Step 4: Run the layer tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ch07_layers.py -v`
Expected: PASS.

- [ ] **Step 5: Append dev-note and commit**

```bash
git add app/services/context.py tests/test_ch07_layers.py dev-notes/ch07.md
git commit -m "feat: add ch07 layer splitting, L2 rendering and prompt assembly"
```

---

### Task 4: Runtime 预处理、L1 降级与摘要调度

**Files:**
- Create: `app/services/session_context.py`
- Test: `tests/test_ch07_session_context.py`

**Interfaces:**
- Consumes: Task 1 repository, Task 2 `compute_budget`, Task 3 helpers, Task 5 `schedule_summary` (import lazily; the test monkeypatches it).
- Produces:
  - `ContextPlan` dataclass: `summary, summary_upto_msg_id, layer1_from_msg_id, history_l2, history_l1, prompt_history, turn_start_index, l2_tokens, l1_tokens, budget, degraded`.
  - `prepare_context(session, conversation: Conversation, prior_messages: Sequence[BaseMessage], turn_start_index: int, settings: Settings, *, current_msg_id: int | None = None) -> ContextPlan`

- [ ] **Step 1: Write the failing session-context tests**

`tests/test_ch07_session_context.py`:

```python
from langchain_core.messages import AIMessage, HumanMessage
from app.config import Settings
from app.models import Conversation
from app.services.history import append_message, create_conversation, get_conversation
from app.services.session_context import prepare_context


def _settings(**kw):
    base = dict(_env_file=None, openai_api_key="x", model_context_window=18000,
                max_output_tokens=2000, max_user_input_tokens=2000,
                max_agent_steps=3, tool_result_max_tokens=1200, rerank_top_k=5)
    base.update(kw)
    return Settings(**base)


def _seed(session, turns=4, answer="答" * 100, question="问"):
    cid = create_conversation(session, "u1").id
    msgs, ids = [], []
    for i in range(turns):
        u = append_message(session, cid, "user", f"{question}{i}")
        a = append_message(session, cid, "assistant", answer)
        msgs += [HumanMessage(content=u.content, id=f"db-{u.id}"), AIMessage(content=a.content)]
        ids.append((u.id, a.id))
    session.commit()
    return get_conversation(session, cid), msgs, ids


def test_prepare_context_splits_and_logs_history_ctx(session, caplog):
    caplog.set_level(logging.INFO)
    conv, msgs, ids = _seed(session, turns=2)
    conv.layer1_from_msg_id = ids[0][1]  # 第一轮 assistant 之后是 L1 起点
    session.commit()
    plan = prepare_context(session, conv, msgs, len(msgs), _settings())
    assert plan.history_l2 and plan.history_l1
    assert "history_ctx" in caplog.text
    assert plan.prompt_history == plan.history_l2 + plan.history_l1


def test_prepare_context_degrades_l1_and_persists_anchor(session, caplog, monkeypatch):
    caplog.set_level(logging.INFO)
    monkeypatch.setattr("app.services.session_context.schedule_summary", lambda cid, settings: True)
    conv, msgs, ids = _seed(session, turns=8, answer="答" * 200)
    plan = prepare_context(session, conv, msgs, len(msgs), _settings(model_context_window=4000, max_output_tokens=200))
    assert plan.degraded is True
    updated = get_conversation(session, conv.id)
    assert updated.layer1_from_msg_id is not None
    assert "层1 降级" in caplog.text


def test_prepare_context_merges_db_history_before_oldest_state_id(session):
    conv, msgs, ids = _seed(session, turns=3)
    # State 只剩最后一轮，前面两轮要从 MySQL 补回来
    plan = prepare_context(session, conv, msgs[4:], 0, _settings())
    assert any(m.content == "问0" for m in plan.prompt_history)
    assert any(m.content == "问1" for m in plan.prompt_history)


def test_prepare_context_triggers_summary_when_l2_over_budget(session, monkeypatch):
    conv, msgs, ids = _seed(session, turns=6, answer="答" * 400, question="问" + "细" * 500)
    conv.layer1_from_msg_id = ids[-2][1]  # 倒数第二轮 assistant 之后是最后一轮 L1
    session.commit()
    calls = []
    monkeypatch.setattr("app.services.session_context.schedule_summary", lambda cid, settings: calls.append(cid) or True)
    prepare_context(session, conv, msgs, len(msgs), _settings())
    assert calls == [conv.id]


def test_prepare_context_does_not_trigger_within_budget(session, monkeypatch):
    conv, msgs, _ = _seed(session, turns=2)
    calls = []
    monkeypatch.setattr("app.services.session_context.schedule_summary", lambda cid, settings: calls.append(cid) or True)
    prepare_context(session, conv, msgs, len(msgs), _settings())
    assert calls == []
```

- [ ] **Step 2: Run the session-context tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ch07_session_context.py -v`
Expected: FAIL with `ImportError: app.services.session_context`.

- [ ] **Step 3: Implement `prepare_context`**

Algorithm:
1. `summary = conversation.summary or ""`, `summary_upto = conversation.summary_upto_msg_id`, `layer1_from = conversation.layer1_from_msg_id`; `history = list(prior_messages)[:turn_start_index]` (normal 轮已是当前轮之前；resume 用最后一条 user 的索引切掉当前轮).
2. `_merge_db_history(session, conversation, history, summary_upto, current_msg_id)`:
   - If `history` empty: load `list_dialog_messages`, filter `id > summary_upto` and (`current_msg_id is None` or `id < current_msg_id`), convert user → `HumanMessage(id=f"db-{id}")`, assistant → `AIMessage`.
   - Else find the oldest `db-` user id in history; if it is greater than the conversation's first message id, load rows with `id < oldest_id`, `id > summary_upto`, and (`current_msg_id is None` or `id < current_msg_id`), convert, prepend. If no `db-` ids, leave history as-is (fallback).
3. `l2, l1 = split_layers(merged, summary_upto, layer1_from)`.
4. `l1_tokens = message_tokens(l1)`; `budget = compute_budget(settings, estimate_tokens(summary))`.
5. If `l1_tokens > budget.layer1` and len(l1) > 0:
   - `target_user = choose_degradation(l1, layer1_from, budget.layer1, settings.layer1_degrade_low_water_ratio)`.
   - If target_user is not None: `boundary = previous_message_id(session, conversation.id, target_user)`; `update_layer1_from(...)`; set `degraded=True`; log `层1 降级 {old}→{boundary} l1_tokens={before}→{after}`; re-split.
   - If only one turn remains, keep it and log warning.
6. `l2_rendered = render_l2(l2, settings)`; `l2_tokens = message_tokens(l2_rendered)`.
7. If `l2_tokens > budget.layer2` and l2: `from_id = _first_db_id(l2)`; `upto_id = max db id <= layer1_from` (use `layer1_from` if it is the last L2 id; otherwise `previous_message_id` of the first L1 user id); log `summary trigger ...`; call `schedule_summary(conversation.id, settings)`.
8. `prompt_history = l2_rendered + l1`.
9. `log_context(logger, "history_ctx", summary, prompt_history, len(prompt_history), settings.context_log_preview_chars)`.
10. Return `ContextPlan`.

- [ ] **Step 4: Run the session-context tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ch07_session_context.py -v`
Expected: PASS.

- [ ] **Step 5: Append dev-note and commit**

```bash
git add app/services/session_context.py tests/test_ch07_session_context.py dev-notes/ch07.md
git commit -m "feat: prepare layered session context and degrade L1"
```

---

### Task 5: 后台分段摘要任务

**Files:**
- Create: `app/services/summary_job.py`
- Test: `tests/test_ch07_summary_job.py`

**Interfaces:**
- Consumes: Task 1 repository; `build_llm`; `Settings.summary_model`, `summary_max_chars`.
- Produces:
  - `class SegmentSummary(BaseModel): summary: str`
  - `@dataclass SummaryResult: status: str, seq: int | None, from_msg_id: int | None, upto_msg_id: int | None, chars: int, elapsed_ms: int, reason: str`
  - `run_summary(conversation_id: int, settings: Settings, *, session=None, llm=None) -> SummaryResult`
  - `schedule_summary(conversation_id: int, settings: Settings) -> bool`
  - `shutdown_summary_executor() -> None`
  - `build_summary_prompt(prior_summary: str, dialogue: str) -> list[BaseMessage]`
  - `SUMMARY_SYSTEM_PROMPT`, `SUMMARY_MIN_CHARS=20`, `SUMMARY_MAX_CHARS=400`

- [ ] **Step 1: Write the failing summary-job tests**

`tests/test_ch07_summary_job.py`:

```python
import threading
import time
from langchain_core.messages import AIMessage
from app.config import Settings
from app.models import ConversationSummary
from app.services.history import append_message, create_conversation, get_conversation
from app.services.summary_job import run_summary, schedule_summary, shutdown_summary_executor


class FakeLLM:
    def __init__(self, summary="订单 1001，手机 13800000000，要求退款"):
        self.summary = summary
    def with_structured_output(self, schema, method=None):
        return self
    def invoke(self, messages):
        return type("R", (), {"summary": self.summary})()


def _settings():
    return Settings(_env_file=None, openai_api_key="x")


def _seed(session, summary_upto=None, layer1_from=None):
    cid = create_conversation(session, "u1").id
    for text in ["订单 1001 想退款", "好的，请提供手机号", "13800000000", "已登记"]:
        append_message(session, cid, "user" if text in ("订单 1001 想退款", "13800000000") else "assistant", text)
    session.commit()
    conv = get_conversation(session, cid)
    conv.summary_upto_msg_id = summary_upto
    conv.layer1_from_msg_id = layer1_from
    session.commit()
    return conv


def test_run_summary_appends_segment_and_advances_upto(session):
    conv = _seed(session, layer1_from=4)
    result = run_summary(conv.id, _settings(), session=session, llm=FakeLLM())
    assert result.status == "done" and result.seq == 1
    updated = get_conversation(session, conv.id)
    assert updated.summary_upto_msg_id == 4
    assert "订单 1001" in updated.summary


def test_run_summary_does_not_rewrite_old_segments(session):
    conv = _seed(session, layer1_from=4)
    old = "旧段" + "旧" * 25  # >= 20 chars, passes summary validation
    new = "新段" + "新" * 25
    run_summary(conv.id, _settings(), session=session, llm=FakeLLM(old))
    append_message(session, conv.id, "user", "后来问物流")
    append_message(session, conv.id, "assistant", "在路上")
    session.commit()
    conv = get_conversation(session, conv.id)
    conv.layer1_from_msg_id = 6
    session.commit()
    run_summary(conv.id, _settings(), session=session, llm=FakeLLM(new))
    rows = session.query(ConversationSummary).order_by(ConversationSummary.seq).all()
    assert [row.content for row in rows] == [old, new]


def test_run_summary_skips_empty_range(session):
    conv = _seed(session)
    result = run_summary(conv.id, _settings(), session=session, llm=FakeLLM())
    assert result.status == "skip" and result.reason == "no_range"


def test_run_summary_skips_invalid_output(session):
    conv = _seed(session, layer1_from=4)
    result = run_summary(conv.id, _settings(), session=session, llm=FakeLLM(""))
    assert result.status == "skip" and result.reason == "too_short"


def test_schedule_summary_is_non_blocking_and_deduplicated(monkeypatch):
    started, release = threading.Event(), threading.Event()
    def slow(cid, settings):
        started.set(); release.wait(2)
    monkeypatch.setattr("app.services.summary_job.run_summary", slow)
    assert schedule_summary(1, _settings()) is True
    assert started.wait(1)
    assert schedule_summary(1, _settings()) is False
    release.set(); shutdown_summary_executor()


def test_shutdown_is_idempotent():
    shutdown_summary_executor()
    shutdown_summary_executor()
```

- [ ] **Step 2: Run the summary tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ch07_summary_job.py -v`
Expected: FAIL with `ImportError: app.services.summary_job`.

- [ ] **Step 3: Implement the summary job**

- `_get_executor()` lazily creates/recreates a `ThreadPoolExecutor(max_workers=2, thread_name_prefix="ch07-summary")` if it is `None` or was shut down; module-level `_inflight: set[int]`, `_lock: threading.Lock`.
- `schedule_summary`: under lock, if cid in `_inflight` return False; add; `_get_executor().submit(_run_and_clear, cid, settings)`; return True. `_run_and_clear` calls `run_summary(cid, settings)` without a `session` kwarg (so tests can monkeypatch it), catches/logs all exceptions, and removes cid in `finally`; `run_summary` opens its own session when `session=None`.
- `run_summary`: open/close a session if none; read conversation; compute range `(summary_upto, layer1_from]`; if empty or `upto <= old upto`, skip; load `list_dialog_messages` filtered to the range, converting `[订单选择]` audit rows to `"用户选择了订单 N"`; `prompt = build_summary_prompt(prior_summary, dialogue)`; call `llm.with_structured_output(SegmentSummary, method="json_mode").invoke(prompt)`; if `llm` is None build with `build_llm(settings, temperature=0.0, model=settings.summary_model or None)`.
- `build_summary_prompt`: `[SystemMessage(SUMMARY_SYSTEM_PROMPT), HumanMessage(f"【旧摘要，仅作背景，不要合并】\n{prior}\n\n【本段新对话】\n{dialogue}")]`.
- Validate `20 <= len(summary) <= 400`; else skip with reason.
- `append_summary_segment(...)`; log start/done with elapsed ms; return `SummaryResult("done", ...)`.
- `shutdown_summary_executor`: `executor.shutdown(wait=False, cancel_futures=True)`; set `_executor = None` so a later `schedule_summary` can recreate it; make it idempotent.
- `SUMMARY_SYSTEM_PROMPT` must include the red lines from the spec.

- [ ] **Step 4: Run the summary tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ch07_summary_job.py -v`
Expected: PASS.

- [ ] **Step 5: Append dev-note and commit**

```bash
git add app/services/summary_job.py tests/test_ch07_summary_job.py dev-notes/ch07.md
git commit -m "feat: add asynchronous append-only conversation summaries"
```

---

### Task 6: 恒定 system、本轮材料与 Agent 拼装

**Files:**
- Modify: `app/prompts/customer_service.py`
- Modify: `app/agent/subgraph.py`
- Modify: `app/agent/state.py`
- Modify: `app/agent/raw_demo.py`
- Test: `tests/test_ch07_prompt.py`
- Test: `tests/test_agent_context.py`
- Update: `tests/test_prompt.py`, `tests/test_agent_subgraph.py`, `tests/test_agent_refund_flow.py`

**Interfaces:**
- Consumes: Task 3 `assemble_agent_prompt`, `log_context`, `message_tokens`; Task 2 `Settings`.
- Produces:
  - `render_agent_system_prompt(business_name: str = "本店") -> str` (constant, no evidence)
  - `render_after_sales_agent_system_prompt(business_name: str = "本店") -> str` (constant, no order/intent/evidence)
  - `render_turn_material(summary: str, evidence: Sequence[dict] | None, order_data: dict | None, intent: str) -> str` (empty string when all empty)
  - `build_agent_subgraph(llm, *, max_steps: int = 6, tool_result_max_tokens: int = 2000, settings: Settings | None = None, tools: Sequence[BaseTool] | None = None)`; `settings = settings or get_settings()` inside (so existing no-arg tests keep working).
  - `CustomerServiceState` / `AgentState` new plain fields: `summary`, `summary_upto_msg_id`, `layer1_from_msg_id`, `prompt_history`, `history_l2`, `history_l1`, `turn_start_index`, `l2_tokens`, `l1_tokens`, `context_budget`.
  - Agent node logs `model_ctx` with summary + prompt previews + `window_count=len(history)` + tokens.

- [ ] **Step 1: Write the failing prompt tests**

`tests/test_ch07_prompt.py`:

```python
from app.prompts.customer_service import (
    render_after_sales_agent_system_prompt, render_agent_system_prompt, render_turn_material,
)


def test_agent_system_prompt_is_constant():
    a = render_agent_system_prompt()
    b = render_agent_system_prompt()
    assert a == b
    assert "本轮知识库证据" not in a
    assert "订单号:" not in a


def test_after_sales_system_prompt_is_constant():
    a = render_after_sales_agent_system_prompt()
    assert "订单号:" not in a and "识别意图" not in a


def test_turn_material_only_includes_nonempty_sections():
    assert render_turn_material("", None, None, "") == ""
    text = render_turn_material("订单 1001 退款", [{"n": 1, "section_path": "售后", "question": "能退吗", "answer": "可以"}], {"order_id": "1001"}, "退款退货")
    assert "早前对话摘要" in text and "订单 1001 退款" in text
    assert "本轮知识库证据" in text and "[1]" in text
    assert "当前订单" in text and "订单号:1001" in text
    assert "识别意图" in text and "退款退货" in text
```

- [ ] **Step 2: Write the failing agent-context tests**

`tests/test_agent_context.py`:

```python
import logging
from types import SimpleNamespace
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from app.agent.subgraph import build_agent_subgraph
from app.config import Settings
from tests.agent_fakes import ScriptedLLM, text_chunk


def _settings():
    return Settings(_env_file=None, openai_api_key="x")


def test_agent_prompt_order_and_model_ctx(caplog):
    caplog.set_level(logging.INFO)
    llm = ScriptedLLM([text_chunk("答")])
    graph = build_agent_subgraph(llm, max_steps=3, tool_result_max_tokens=1200, settings=_settings(), tools=[])
    state = {
        "messages": [HumanMessage(content="当前问题", id="db-10")],
        "turn_start_index": 0,
        "prompt_history": [HumanMessage(content="旧问", id="db-1"), AIMessage(content="旧答")],
        "summary": "订单 1001 退款",
        "evidence": [{"n": 1, "section_path": "售后", "question": "能退吗", "answer": "可以"}],
        "route": "knowledge",
        "conversation_id": 1,
    }
    result = graph.invoke(state, {"configurable": {"thread_id": "t"}})
    assert result["answer"] == "答"
    text = caplog.text
    assert "model_ctx" in text and "订单 1001 退款" in text and "当前问题" in text


def test_agent_ignores_token_budget_gate():
    llm = ScriptedLLM([text_chunk("答")])
    graph = build_agent_subgraph(llm, max_steps=3, tool_result_max_tokens=1200, settings=_settings(), tools=[])
    state = {
        "messages": [HumanMessage(content="问", id="db-1")],
        "turn_start_index": 0,
        "prompt_history": [],
        "token_budget": 1,
        "token_estimate": 999999,
        "step_count": 0,
    }
    result = graph.invoke(state, {"configurable": {"thread_id": "t"}})
    assert result["answer"] == "答"


def test_no_variable_system_messages_in_prompt():
    # assemble_agent_prompt 只接受 system + history + current；材料是 HumanMessage
    from app.services.context import assemble_agent_prompt
    prompt = assemble_agent_prompt(
        SystemMessage(content="SYS"), [], [HumanMessage(content="问")], "材料", 10000
    )
    assert [m.type for m in prompt] == ["system", "human", "human"]
```

- [ ] **Step 3: Run the prompt/agent tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ch07_prompt.py tests/test_agent_context.py -v`
Expected: FAIL (new signatures / material missing).

- [ ] **Step 4: Implement the prompt split and agent bundle**

- Remove `evidence_block`, `order_block`, `intent_note` from both system templates; make the render functions take only `business_name`.
- Update citation wording from “下方「本轮知识库证据」” to “本轮材料中的「知识库证据」”.
- Add `render_turn_material` using the existing `_format_agent_evidence` / `_format_after_sales_order`; start with `【系统提供的本轮材料，不是用户发言】`; append sections only when non-empty.
- Add the new plain fields to both `CustomerServiceState` and `AgentState` (`messages` keeps `add_messages`).
- Update `app/agent/raw_demo.py` to the new `build_agent_subgraph` signature (`settings=settings`, `max_steps=settings.effective_max_agent_steps`, `tool_result_max_tokens=settings.effective_tool_result_max_tokens`; no `token_budget`).
- In `build_agent_subgraph`, remove `token_budget` / `max_context_tokens`; build the system via the constant renderers; history from `state["prompt_history"]`; current from `state["messages"][state.get("turn_start_index", 0):]`; material from `render_turn_material(...)`; `prompt = assemble_agent_prompt(system, history, current, material, settings.model_context_window - ceil(window*safety))`; log `model_ctx` before `llm.bind_tools(...).stream(prompt)`.
- Keep `token_estimate` accumulation for logs; remove the `>= token_budget` exhaustion check.
- Truncate tool results with `text[:tool_result_max_tokens]` (1 char = 1 token).
- Update existing prompt/subgraph/refund tests to the new constant-system assertions; run the full set.

- [ ] **Step 5: Run the prompt/agent tests and the updated existing tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ch07_prompt.py tests/test_agent_context.py tests/test_prompt.py tests/test_agent_subgraph.py tests/test_agent_refund_flow.py -v`
Expected: PASS.

- [ ] **Step 6: Append dev-note and commit**

```bash
git add app/prompts/customer_service.py app/agent/subgraph.py app/agent/state.py app/agent/raw_demo.py tests/test_ch07_prompt.py tests/test_agent_context.py tests/test_prompt.py tests/test_agent_subgraph.py tests/test_agent_refund_flow.py dev-notes/ch07.md
git commit -m "feat: keep the agent system prompt constant and move turn material after the user"
```

---

### Task 7: 父图使用 prompt_history、订单选择器与 effective top-k

**Files:**
- Modify: `app/agent/graph.py`
- Modify: `app/agent/coref.py`
- Modify: `app/agent/intent.py`
- Modify: `app/rag/retrieval.py`
- Test: `tests/test_ch07_graph.py`
- Update: `tests/test_agent_graph.py`, `tests/test_agent_coref.py`, `tests/test_agent_intent.py`, `tests/test_agent_refund_flow.py`, tests that assert SystemMessage picker

**Interfaces:**
- Consumes: Task 2 effective settings; Task 6 subgraph signature.
- Produces:
  - `resolve_query(llm, message, history=(), summary: str = "") -> str` (summary line prepended to the human message)
  - `classify_intent(llm, message, history=(), *, raise_on_error=False, summary: str = "") -> IntentResult`
  - `resolve_reference` / `classify_intent_node` use `state.get("prompt_history") or []` and pass `summary=state.get("summary") or ""`.
  - `fetch_order_node` appends `HumanMessage(content="【订单选择器】用户选择了订单 {id}")`.
  - `build_graph` calls `build_agent_subgraph(..., max_steps=settings.effective_max_agent_steps, tool_result_max_tokens=settings.effective_tool_result_max_tokens, settings=settings)`.
  - `retrieve_policy` uses `settings.effective_rerank_top_k`; `retrieve_hybrid` uses `settings.effective_rerank_top_k`.

- [ ] **Step 1: Write the failing graph tests**

`tests/test_ch07_graph.py`:

```python
from langchain_core.messages import HumanMessage
from tests.agent_fakes import ScriptedStructuredLLM
from app.agent.graph import build_graph
from app.config import Settings


def _settings(**kw):
    return Settings(_env_file=None, openai_api_key="x", **kw)


def test_coref_uses_prompt_history_not_full_state(monkeypatch):
    seen = {}
    def fake_resolve(llm, message, history):
        seen["history"] = list(history)
        return message
    monkeypatch.setattr("app.agent.graph.resolve_query", fake_resolve)
    monkeypatch.setattr("app.agent.graph.build_agent_subgraph", lambda *a, **k: lambda state: state)
    graph = build_graph(object(), ScriptedStructuredLLM("", None), settings=_settings())
    state = {
        "user_message": "它呢",
        "messages": [HumanMessage(content="旧问", id="db-1"), HumanMessage(content="新问", id="db-2")],
        "prompt_history": [HumanMessage(content="旧问", id="db-1")],
    }
    graph.invoke(state, {"configurable": {"thread_id": "t"}})
    assert [m.content for m in seen["history"]] == ["旧问"]


def test_coref_and_intent_receive_summary_line():
    from app.agent.coref import resolve_query
    from app.agent.intent import IntentResult, classify_intent
    from tests.agent_fakes import ScriptedStructuredLLM

    coref_llm = ScriptedStructuredLLM("完整问题", None)
    resolve_query(coref_llm, "它呢", [HumanMessage(content="旧问")], summary="订单 1001 退款")
    assert "早前对话摘要" in coref_llm.messages[-1].content
    assert "订单 1001 退款" in coref_llm.messages[-1].content

    intent_llm = ScriptedStructuredLLM("", IntentResult(intent="退款退货", confidence=0.9))
    classify_intent(intent_llm, "它呢", [HumanMessage(content="旧问")], summary="订单 1001 退款")
    assert "早前对话摘要" in intent_llm.messages[-1].content


def test_fetch_order_picker_note_is_human_message(monkeypatch):
    import app.agent.graph as graph_mod
    from langchain_core.messages import HumanMessage

    monkeypatch.setattr(graph_mod, "list_user_orders", lambda sid: [{"order_id": "1001"}])
    monkeypatch.setattr(graph_mod, "interrupt", lambda payload: "1001")
    monkeypatch.setattr(graph_mod, "execute_tool", lambda *a, **k: '{"order_id": "1001"}')
    out = graph_mod.fetch_order_node({"session_id": "s", "resolved_query": "退款"})
    assert isinstance(out["messages"][0], HumanMessage)
    assert "订单 1001" in out["messages"][0].content
```

- [ ] **Step 2: Run the graph tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ch07_graph.py -v`
Expected: FAIL (coref still slices `messages[:-1]`).

- [ ] **Step 3: Implement the graph changes**

- Replace both `history = list(state.get("messages") or [])[:-1]` with `history = list(state.get("prompt_history") or [])`; pass `summary=state.get("summary") or ""` to both helpers.
- Add optional `summary` to `resolve_query` / `classify_intent`; prepend `(早前对话摘要: {summary})\n` before `最近对话：` in `_build_human_message`. Keep the parameter optional so eval scripts and existing callers stay compatible.
- Import `HumanMessage`; change the picker note.
- Pass the new subgraph kwargs; remove `token_budget` / `max_context_tokens` from the build call.
- Change `[: settings.rerank_top_n]` to `[: settings.effective_rerank_top_k]`.
- In `app/rag/retrieval.py`, replace `settings.rerank_top_n` with `settings.effective_rerank_top_k` in the retrieval top-k selection.
- Update existing tests that assert `SystemMessage` for the picker or the old subgraph kwargs.

- [ ] **Step 4: Run the graph and existing agent tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ch07_graph.py tests/test_agent_graph.py tests/test_agent_coref.py tests/test_agent_intent.py tests/test_agent_refund_flow.py tests/test_retrieval.py -v`
Expected: PASS.

- [ ] **Step 5: Append dev-note and commit**

```bash
git add app/agent/graph.py app/agent/coref.py app/agent/intent.py app/rag/retrieval.py tests/test_ch07_graph.py tests/test_agent_graph.py tests/test_agent_coref.py tests/test_agent_intent.py tests/test_agent_refund_flow.py dev-notes/ch07.md
git commit -m "feat: drive ch07 graph from prompt_history and effective retrieval limits"
```

---

### Task 8: 会话感知的 runtime、SSE frame、退款/工单归属

**Files:**
- Modify: `app/agent/service.py`
- Modify: `app/api/ch06.py`
- Create: `app/api/ch07.py`
- Modify: `app/api/ch05.py`
- Modify: `app/schemas.py`
- Modify: `app/main.py`
- Test: `tests/test_ch07_service.py`
- Test: `tests/test_ch07_api.py`
- Update: `tests/test_agent_service.py`, `tests/test_ch06_api.py`, `tests/test_ch05_api.py`, `tests/test_schemas.py`

**Interfaces:**
- Consumes: Task 1 `resolve_conversation`, Task 4 `prepare_context`, Task 5 `schedule_summary` (already inside prepare), Task 7 graph.
- Produces:
  - `_thread_config(conversation_id: int) -> dict` → `{"configurable": {"thread_id": f"conv:{conversation_id}"}}`
  - `has_pending_interrupt(graph, conversation_id: int) -> bool`
  - `stream_chat_agent(..., conversation_id: int | None = None, new_conversation: bool = False)`
  - `resume_chat_agent(..., conversation_id: int | None = None)`
  - `ChatRequest.conversation_id: int | None`, `ChatRequest.new_conversation: bool = False`
  - `ResumeRequest.conversation_id: int | None`
  - `RefundRequest.conversation_id: int | None`, `TicketRequest.conversation_id: int | None`
  - `/ch07/chat`, `/ch07/chat/resume`, `/ch07/refunds` aliases on the same graph.

- [ ] **Step 1: Write the failing service tests**

`tests/test_ch07_service.py`:

```python
import json
from types import SimpleNamespace
from langchain_core.messages import HumanMessage
from app.agent.service import stream_chat_agent
from app.config import Settings
from app.models import Message
from tests.agent_fakes import FakeGraph


def _settings():
    return Settings(_env_file=None, openai_api_key="x")


def _run(session, frames=(), final=None, message="你好", user_id="u1", **kw):
    graph = FakeGraph(list(frames), final or {"answer": "答"}, snapshots=[SimpleNamespace(values={}, interrupts=())])
    events = list(stream_chat_agent(graph, session, user_id, message, settings=_settings(), **kw))
    return graph, events


def test_stream_agent_thread_is_conversation_scoped(session):
    graph, events = _run(session, new_conversation=True)
    assert graph.config["configurable"]["thread_id"].startswith("conv:")
    assert any('"type":"conversation"' in e for e in events)


def test_stream_agent_uses_existing_conversation(session):
    from app.services.history import create_conversation
    cid = create_conversation(session, "u1").id
    graph, _ = _run(session, conversation_id=cid)
    assert graph.config["configurable"]["thread_id"] == f"conv:{cid}"
    rows = session.query(Message).filter_by(conversation_id=cid).all()
    assert any(row.content == "你好" for row in rows)


def test_stream_agent_rejects_foreign_conversation(session):
    from app.services.history import create_conversation
    cid = create_conversation(session, "u2").id
    graph, events = _run(session, conversation_id=cid)
    assert any('"error"' in e for e in events)
    assert graph.inputs == []


def test_resume_uses_conversation_thread_and_audit_row(session):
    from langgraph.types import Command
    from app.agent.service import resume_chat_agent
    from app.services.history import create_conversation

    cid = create_conversation(session, "u1").id
    graph = FakeGraph(
        [], {"answer": "答"},
        snapshots=[SimpleNamespace(values={}, interrupts=(SimpleNamespace(value={"kind": "select_order"}),))],
    )
    list(resume_chat_agent(graph, session, "u1", "1001", settings=_settings(), conversation_id=cid))
    assert graph.config["configurable"]["thread_id"] == f"conv:{cid}"
    assert any(isinstance(call, Command) for call in graph.inputs)
    rows = session.query(Message).filter_by(conversation_id=cid, role="user").all()
    assert rows[-1].content == "[订单选择] 订单 1001"
```

- [ ] **Step 2: Write the failing API tests**

`tests/test_ch07_api.py`:

```python
from types import SimpleNamespace
import pytest
from fastapi.testclient import TestClient
from app.api.ch06 import get_ch06_graph_dep
from app.config import Settings, get_settings
from app.main import app
from tests.agent_fakes import FakeGraph


@pytest.fixture
def client(session):
    app.dependency_overrides[get_settings] = lambda: Settings(openai_api_key="test")
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_ch07_chat_alias_streams_conversation_frame(client):
    app.dependency_overrides[get_ch06_graph_dep] = lambda: FakeGraph(
        [], {"answer": "答"}, snapshots=[SimpleNamespace(values={}, interrupts=())]
    )
    with client.stream("POST", "/ch07/chat", json={"session_id": "s1", "message": "你好"}) as r:
        body = "".join(r.iter_text())
    assert "conversation" in body and "data: [DONE]" in body


def test_new_conversation_creates_a_row(client, session):
    graph = FakeGraph([], {"answer": "答"}, snapshots=[SimpleNamespace(values={}, interrupts=())])
    app.dependency_overrides[get_ch06_graph_dep] = lambda: graph
    with client.stream("POST", "/ch07/chat", json={"session_id": "s1", "message": "一", "new_conversation": True}) as r:
        body = "".join(r.iter_text())
    assert '"conversation_id"' in body


def test_refund_and_ticket_use_request_conversation(client, session):
    from app.models import Ticket
    from app.services.history import create_conversation
    c1 = create_conversation(session, "s1")
    c2 = create_conversation(session, "s1")
    r = client.post("/ch06/refunds", json={
        "session_id": "s1", "order_id": "1001", "reason": "质量问题", "conversation_id": c1.id,
    })
    assert r.status_code == 200
    assert session.query(Ticket).order_by(Ticket.id.desc()).first().conversation_id == c1.id
    r2 = client.post("/ch05/tickets", json={
        "session_id": "s1", "description": "售后", "ticket_type": "售后", "conversation_id": c2.id,
    })
    assert r2.status_code == 200
    assert session.query(Ticket).order_by(Ticket.id.desc()).first().conversation_id == c2.id
```

The `/ch07` page assertion moves to Task 10; this task only wires the chat aliases.

- [ ] **Step 3: Run the new service/API tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ch07_service.py tests/test_ch07_api.py -v`
Expected: FAIL (thread id still `user:`, no conversation frame, no /ch07).

- [ ] **Step 4: Implement the runtime/API changes**

- `_thread_config(cid)`; `has_pending_interrupt(graph, cid)`.
- `stream_chat_agent`: resolve conversation first (new/id/latest); build config; check pending; if pending emit `conversation` + `interrupt`; append user with `HumanMessage(content=message, id=f"db-{msg_id}")`; get prior from the snapshot; `plan = prepare_context(session, conversation, prior, len(prior), settings, current_msg_id=msg_id)`; `_reset_inputs` includes plan fields (`summary`, anchors, `prompt_history`, `history_l2`, `history_l1`, `turn_start_index`, `l2_tokens`, `l1_tokens`, `context_budget=dataclasses.asdict(plan.budget)`) and the user message; emit `{"type":"conversation","conversation_id":cid}` before streaming; stream; `_finish_turn` unchanged.
- `resume_chat_agent`: resolve conversation; pending check; audit row; `turn_start_index = last HumanMessage index`; call `prepare_context` for logging/trigger; stream `Command(resume=...)`; emit conversation frame.
- `ConversationNotFound` → yield `{"error":"会话不存在"}` and return.
- `app/api/ch06.py`: pass `req.conversation_id` / `req.new_conversation`; `require_pending_interrupt` resolves the conversation (latest if None) and calls `has_pending_interrupt(graph, cid)`; refunds use the request's conversation id or fall back. Change the lazy graph builder to `settings.ch07_checkpoint_db`; keep the existing `app.state.ch06_*` attribute names for compatibility so `/ch06` and `/ch07` share one graph/handle.
- `app/api/ch05.py` tickets: same conversation resolution.
- `app/schemas.py`: add the fields; validate positive ids.
- `app/api/ch07.py`: `APIRouter(prefix="/ch07")`; add `/chat`, `/chat/resume`, `/refunds` using the ch06 functions (pass `response_class=EventSourceResponse` for the two SSE routes). Do not add the `/ch07` page route here; Task 10 owns the page.
- `app/main.py`: include the ch07 router; keep the ch06 page; root still ch06 until Task 10.
- Update old tests: thread id assertions `user:u1` → `conv:{cid}`; `ChatRequest` extra fields; refund/ticket payloads.

- [ ] **Step 5: Run the new and updated service/API tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ch07_service.py tests/test_ch07_api.py tests/test_agent_service.py tests/test_ch06_api.py tests/test_ch05_api.py tests/test_schemas.py -v`
Expected: PASS.

- [ ] **Step 6: Append dev-note and commit**

```bash
git add app/agent/service.py app/api/ch06.py app/api/ch07.py app/api/ch05.py app/schemas.py app/main.py tests/test_ch07_service.py tests/test_ch07_api.py tests/test_agent_service.py tests/test_ch06_api.py tests/test_ch05_api.py tests/test_schemas.py dev-notes/ch07.md
git commit -m "feat: make chat runtime conversation-aware and add ch07 aliases"
```

---

### Task 9: 会话列表与历史只读接口

**Files:**
- Create: `app/api/conversations.py`
- Modify: `app/main.py`
- Test: `tests/test_conversations_api.py`

**Interfaces:**
- Consumes: Task 1 `list_conversations`, `get_conversation`, `list_dialog_messages`.
- Produces:
  - `GET /api/conversations?user_id=` → `{"conversations": [{"id", "preview", "has_summary", "created_at", "updated_at"}]}`
  - `GET /api/conversations/{id}/messages?user_id=` → `{"conversation_id", "messages": [{"id", "role", "content", "created_at"}]}`; 404 missing or user mismatch; filters `[订单选择]`.

- [ ] **Step 1: Write the failing API tests**

`tests/test_conversations_api.py`:

```python
from app.models import Conversation
from app.services.history import append_message, create_conversation, update_conversation_summary


def test_list_conversations_newest_first(client, session):
    c1 = create_conversation(session, "u1")
    append_message(session, c1.id, "user", "订单 1001 退款")
    update_conversation_summary(session, c1.id, "第1段：订单 1001", 1)
    c2 = create_conversation(session, "u1")
    append_message(session, c2.id, "user", "物流到哪")
    r = client.get("/api/conversations", params={"user_id": "u1"})
    assert r.status_code == 200
    rows = r.json()["conversations"]
    assert [row["id"] for row in rows] == [c2.id, c1.id]
    assert rows[1]["preview"].startswith("订单 1001") and rows[1]["has_summary"] is True


def test_messages_404_and_user_scope(client, session):
    cid = create_conversation(session, "u1").id
    assert client.get(f"/api/conversations/{cid}/messages").status_code == 200
    assert client.get(f"/api/conversations/{cid}/messages", params={"user_id": "u2"}).status_code == 404
    assert client.get("/api/conversations/9999/messages").status_code == 404


def test_messages_filter_audit_rows(client, session):
    cid = create_conversation(session, "u1").id
    append_message(session, cid, "user", "问")
    append_message(session, cid, "user", "[订单选择] 订单 1001")
    append_message(session, cid, "assistant", "答")
    session.commit()
    rows = client.get(f"/api/conversations/{cid}/messages").json()["messages"]
    assert [row["content"] for row in rows] == ["问", "答"]
```

Use the `client` fixture from `tests/test_ch06_api.py` style (Settings override + TestClient).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_conversations_api.py -v`
Expected: FAIL with 404 for the new routes.

- [ ] **Step 3: Implement the endpoints**

`app/api/conversations.py`: `APIRouter(prefix="/api/conversations")`; list uses `list_conversations`; messages uses `get_conversation(session, id, user_id)` (404 if None), `list_dialog_messages`, filters `[订单选择]`, returns rows. Include the router in `app/main.py`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_conversations_api.py -v`
Expected: PASS.

- [ ] **Step 5: Append dev-note and commit**

```bash
git add app/api/conversations.py app/main.py tests/test_conversations_api.py dev-notes/ch07.md
git commit -m "feat: add conversation list and history APIs"
```

---

### Task 10: 多会话前端页面

**Files:**
- Create: `app/static/ch07.html`
- Modify: `app/api/ch07.py`
- Modify: `app/main.py`
- Modify: `app/static/ch06.html`
- Test: `tests/test_ch07_page.py`

**Interfaces:**
- Consumes: Task 8 `/ch07/chat`, `/ch07/chat/resume`, `/ch07/refunds`; Task 9 conversation APIs.
- Produces: `/ch07` page, root `/` serves ch07; ch06 page ignores `conversation` frames.

- [ ] **Step 1: Write the failing page tests**

`tests/test_ch07_page.py`:

```python
def test_ch07_page_has_sidebar_and_apis(client):
    r = client.get("/ch07")
    assert r.status_code == 200
    text = r.text
    assert "lamp-cs-ch07-user-id" in text
    assert "新对话" in text
    assert "/api/conversations" in text
    assert "conversation_id" in text
    assert "@media (max-width: 780px)" in text
    assert "streaming || pickerPending" in text


def test_root_serves_ch07_and_ch06_still_works(client):
    assert "lamp-cs-ch07-user-id" in client.get("/").text
    assert "lamp-cs-ch06-session-id" in client.get("/ch06").text


def test_ch06_page_ignores_conversation_frame():
    text = open("app/static/ch06.html", encoding="utf-8").read()
    assert 'frame.type === "conversation"' in text
```

Use the `client` fixture from `tests/test_ch06_api.py` style (Settings override + TestClient + `session` fixture).

- [ ] **Step 2: Run the page tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ch07_page.py -v`
Expected: FAIL (`/ch07` missing, root still ch06, ch06 frame branch missing).

- [ ] **Step 3: Implement the page**

Copy `app/static/ch06.html` to `app/static/ch07.html`; change title/tag/session key to `lamp-cs-ch07-user-id`; add:
- A `.layout` wrapper with `<aside class="sidebar">` + existing chat panel.
- Sidebar header with 「新对话」 button, `#conversation-list`.
- CSS: sidebar width 240px; `@media (max-width: 780px) { .sidebar { display: none; } }`.
- JS state `activeConversationId = null`; `loadConversations()` fetch `/api/conversations?user_id=`; `loadConversation(id)` fetch `/api/conversations/{id}/messages?user_id=` and render; `newConversation()` clears; `handlePayload` handles `frame.type === "conversation"` by setting `activeConversationId` and refreshing; send body includes `conversation_id` / `new_conversation`; refunds and tickets payloads include `conversation_id`; `.finally` refreshes the sidebar; sidebar clicks disabled while `streaming || pickerPending`.
- `app/api/ch07.py`: serve `CH07_PAGE` at `/ch07`.
- `app/main.py`: root serves `CH07_PAGE`; import it.
- `app/static/ch06.html`: in `handlePayload`, add `if (frame.type === "conversation") return;` before the unknown-frame fallback.
- Use `/ch07/chat` and `/ch07/chat/resume` in the page.

- [ ] **Step 4: Run the page tests and the full API smoke**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ch07_page.py tests/test_ch07_api.py tests/test_ch06_api.py -v`
Expected: PASS.

- [ ] **Step 5: Append dev-note and commit**

```bash
git add app/static/ch07.html app/api/ch07.py app/main.py app/static/ch06.html tests/test_ch07_page.py dev-notes/ch07.md
git commit -m "feat: add the ch07 multi-conversation chat page"
```

---

### Task 11: log/app.log 文件日志与启动自检接线

**Files:**
- Create: `app/logging_config.py`
- Modify: `run.py`
- Modify: `app/main.py`
- Modify: `.gitignore`
- Test: `tests/test_logging_config.py`

**Interfaces:**
- Produces: `configure_file_logging(path: str = "log/app.log", *, force: bool = False) -> None`; idempotent; `LOG_TO_FILE=0` disables; root INFO.
- `run.py` and `app.main` lifespan call it; lifespan calls `validate_budget(get_settings())`.

- [ ] **Step 1: Write the failing logging tests**

`tests/test_logging_config.py`:

```python
import logging
from pathlib import Path
from app.logging_config import configure_file_logging


def test_configure_creates_file_and_is_idempotent(tmp_path, monkeypatch):
    path = tmp_path / "app.log"
    monkeypatch.setenv("LOG_TO_FILE", "1")
    configure_file_logging(str(path), force=True)
    logging.getLogger("test.ch07").info("model_ctx hello")
    configure_file_logging(str(path), force=True)
    logging.getLogger("test.ch07").info("history_ctx world")
    text = path.read_text(encoding="utf-8")
    assert text.count("model_ctx hello") == 1
    assert "history_ctx world" in text


def test_log_to_file_zero_disables(tmp_path, monkeypatch):
    monkeypatch.setenv("LOG_TO_FILE", "0")
    path = tmp_path / "app.log"
    configure_file_logging(str(path), force=True)
    logging.getLogger("test.ch07").info("should not appear")
    assert not Path(path).exists()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_logging_config.py -v`
Expected: FAIL with `ImportError: app.logging_config`.

- [ ] **Step 3: Implement the logging config and wiring**

`configure_file_logging` creates the parent dir, adds a `FileHandler(path, encoding="utf-8")` with `%(asctime)s %(levelname)s %(name)s %(message)s`, adds a `StreamHandler` if the root has no console handler (so `basicConfig` doesn't silently skip it), sets root level INFO, and removes an existing handler for the same path when `force=True` (close it first to avoid Windows file locks; avoid duplicates). If `LOG_TO_FILE=0`, return without creating. If `pytest` is in `sys.modules` and `LOG_TO_FILE` is not explicitly `1`, return (tests don't create the repo log). In `run.py`, call it before `logging.basicConfig`/uvicorn. In `app.main` lifespan, call it once and `validate_budget(get_settings())`; on shutdown call `shutdown_summary_executor()`. Add `log/` to `.gitignore`.

- [ ] **Step 4: Run the logging tests and a startup smoke**

Run: `.venv/Scripts/python.exe -m pytest tests/test_logging_config.py -v`
Expected: PASS.

- [ ] **Step 5: Append dev-note and commit**

```bash
git add app/logging_config.py run.py app/main.py .gitignore tests/test_logging_config.py dev-notes/ch07.md
git commit -m "feat: log ch07 context traces to log/app.log and self-check the budget"
```

---

### Task 12: 摘要 prompt 标注样例评估

**Files:**
- Create: `tests/eval/ch07_summary_cases.json`
- Create: `app/jobs/eval_ch07_summary.py`
- Create: `tests/test_eval_ch07_summary.py`
- Create: `docs/superpowers/reports/2026-11-18-ch07-summary-eval.md`

**Interfaces:**
- Consumes: `SUMMARY_SYSTEM_PROMPT` and `build_llm`; the prompt-format function `build_summary_prompt(prior_summary, dialogue)` from Task 5.
- Produces: `evaluate_case(case, summary) -> dict`; `summarize(results) -> dict`; CLI `python -m app.jobs.eval_ch07_summary`.

- [ ] **Step 1: Write the annotated cases**

`tests/eval/ch07_summary_cases.json` with at least 5 cases:
1. `order-refund`: dialog has order `1001`, phone `13800000000`, "想退款", plus "你好/谢谢"; `contains: ["1001", "13800000000", "退款"]`, `not_contains: ["你好", "谢谢"]`.
2. `product-warranty`: model `LD-200`, "保修", unresolved "灯不亮"; `contains: ["LD-200", "保修", "灯不亮"]`, `not_contains: ["退款"]` (unless the case says refund).
3. `chitchat-only`: only greetings; `contains: []`, `not_contains: ["订单", "退款", "手机"]`, `max_chars: 60`.
4. `order-corrected`: user first says 1001 then corrects to 1002; `contains: ["1002"]`, `not_contains: ["1001"]`.
5. `no-fabrication`: dialog has no order number; `not_contains_regex: "\\d{4,}"`.
6. `refund-rejected`: "已拒绝退款", reason "超过七天"; `contains: ["拒绝", "超过七天"]`.

Each case: `id`, `history` (list of `{role, content}`), `prior_summary`, `contains`, `not_contains`, optional `max_chars`.

- [ ] **Step 2: Write the failing pure-function tests**

`tests/test_eval_ch07_summary.py`:

```python
from app.jobs.eval_ch07_summary import evaluate_case, summarize


def test_evaluate_case_contains_and_banned():
    case = {"id": "c1", "contains": ["1001", "退款"], "not_contains": ["你好"]}
    assert evaluate_case(case, "订单 1001 要求退款")["passed"] is True
    assert evaluate_case(case, "你好，订单 1001 要求退款")["passed"] is False


def test_summarize_counts_only_valid():
    results = [{"valid": True, "passed": True}, {"valid": True, "passed": False}, {"valid": False, "passed": False}]
    out = summarize(results)
    assert out["total"] == 2 and out["passed"] == 1 and out["accuracy"] == 0.5
```

- [ ] **Step 3: Run the pure tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_eval_ch07_summary.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 4: Implement the eval script**

`app/jobs/eval_ch07_summary.py`:
- `load_cases(path)`.
- `build_dialogue(history)` → lines.
- `evaluate_case(case, summary)` checks contains/not_contains/max_chars/regex.
- `summarize(results)` like `eval_coref`.
- `main()` builds `build_llm(get_settings(), temperature=0.0, model=get_settings().summary_model or None)`, calls the same structured-output path as `run_summary` (import `SegmentSummary` and `SUMMARY_SYSTEM_PROMPT`), writes a Markdown report to `docs/superpowers/reports/2026-11-18-ch07-summary-eval.md` with per-case pass/fail and prompt note.
- If no API key, print a skip message and exit 0.

- [ ] **Step 5: Run the pure tests and the real eval**

Run: `.venv/Scripts/python.exe -m pytest tests/test_eval_ch07_summary.py -v`
Expected: PASS.
Run: `.venv/Scripts/python.exe -m app.jobs.eval_ch07_summary`
Expected: report written; accuracy recorded. If the model is unavailable, record `skipped: no API key` in the report.

- [ ] **Step 6: Append dev-note and commit**

```bash
git add tests/eval/ch07_summary_cases.json app/jobs/eval_ch07_summary.py tests/test_eval_ch07_summary.py docs/superpowers/reports/2026-11-18-ch07-summary-eval.md dev-notes/ch07.md
git commit -m "test: evaluate the ch07 summary prompt against annotated cases"
```

---

### Task 13: 全量回归、手动验收与收尾

**Files:**
- Modify: `dev-notes/ch07.md`
- Possibly fix any test/code found by regression.

- [ ] **Step 1: Run the full test suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: all tests pass. Fix failures task-by-task; do not weaken assertions silently.

- [ ] **Step 2: Default-window 20-turn check**

Start with a fresh log and the default window:

```bash
rm -f log/app.log
python run.py
```

Chat 20+ short turns (the budget is calibrated at 300 tokens/turn; keep each turn to a sentence or two), then:

```bash
grep -c "层1 降级" log/app.log
grep -c "summary trigger" log/app.log
```

Expected: `0` and `0`. Also verify no `上下文预算不足`.

- [ ] **Step 3: Demo-config cascade check**

Set in `.env`:

```
MODEL_CONTEXT_WINDOW=18000
MAX_OUTPUT_TOKENS=2000
MAX_USER_INPUT_TOKENS=2000
MAX_AGENT_STEPS=3
TOOL_RESULT_MAX_TOKENS=1200
RERANK_TOP_K=5
```

Restart with a fresh log (`rm -f log/app.log`), chat, then:

```bash
grep "context budget" log/app.log
grep "层1 降级" log/app.log
grep "summary trigger" log/app.log
grep "summary done" log/app.log
grep "model_ctx" log/app.log | tail -3
grep "history_ctx" log/app.log | tail -3
```

Expected: budget line shows `history=5650 layer1=3955 layer2=1695`; `层1 降级 X→Y`; `summary trigger ... > 1695`; `summary done ... 第N段`; `model_ctx`/`history_ctx` show summary and window previews. Ask “最开始那个订单后来怎么说” and confirm the answer contains the early order id and request.

- [ ] **Step 4: Frontend multi-conversation check**

Open `/ch07`, create two conversations, switch back and forth, confirm history reloads, continue chatting, confirm the sidebar shows preview + 已摘要 after a summary, and the page still works at <780px with the sidebar hidden.

- [ ] **Step 5: Code review and dev-note finish**

Run the Superpowers code review skill on the branch; record its conclusion in `dev-notes/ch07.md` (what was found, what was fixed, what was accepted). Then append the finish section with the acceptance evidence and commands. Do not claim completion before the commands above have been run.

- [ ] **Step 6: Finish the branch**

Use `superpowers:finishing-a-development-branch` to decide integration (merge/PR/cleanup). Record the decision and any PR link in `dev-notes/ch07.md`.

- [ ] **Step 7: Final commit**

```bash
git add dev-notes/ch07.md
git commit -m "docs: record ch07 acceptance and review"
```

---

## Plan Self-Review

- **Spec coverage:** DDL/models/repo (Task 1), budget (Task 2), layers/L2 (Task 3), runtime/degradation/trigger (Task 4), summary job (Task 5), constant system/material/model_ctx (Task 6), graph/coref/picker/rerank (Task 7), conversation thread/API/refund/ticket (Task 8), conversation APIs (Task 9), frontend (Task 10), logging/self-check (Task 11), summary eval (Task 12), regression/manual acceptance (Task 13). All spec sections mapped.
- **Type consistency:** `ContextPlan` fields match the service inputs; `schedule_summary(cid, settings)` matches Task 4's monkeypatch and Task 5's signature; `assemble_agent_prompt` is defined in Task 3 and used in Task 6; `ConversationSummary` and repository names match Task 1 and Task 5.
- **Review Focus tests:** tool-heavy L2/trigger (Task 3/4/5), resume (Task 8), old-history merge (Task 4), constant system/no variable system (Task 6), multi-conversation side effects (Task 8/9).
- **Proportion:** Tasks specify signatures, test names/assertions, and only the algorithms that are not obvious (budget formula, layer split, degradation, summary append). No full-file transcripts.
