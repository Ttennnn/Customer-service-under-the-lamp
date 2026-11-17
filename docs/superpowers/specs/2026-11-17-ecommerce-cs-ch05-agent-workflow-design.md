# 第五章 · Agent 工作流 — 设计文档

- 日期：2026-11-17
- 章节：ch05
- 状态：待用户定稿
- 关联：ch02 工具调用、ch03 RAG、ch04 混合检索 + 重排 + 自评闸

## 1. 意图与为什么

把客服系统从「单轮工具调用」升级为「确定性工作流骨架 + 主力 ReAct Agent 核心节点」。

为什么不是「让 Agent 自己决定一切」：

- 政策类问题必须**强制先检索**，不能由模型自由决定是否查知识库；
- 检索结果必须先过**置信度闸**，否则流式答案已经吐给用户，再判就晚了；
- 投诉不能由模型自动转人工或自动建工单，必须把选择权交给用户；
- 闲聊不必花模型调用；
- 生产系统需要可预测的分流、可复查的日志、可延续的会话状态。

所以骨架用 LangGraph 的 StateGraph 把「指代消解 → 意图识别 → 分流 → 强制检索 → 置信度闸 → Agent → 日志」写死；Agent 只负责骨架放行后的工具推理与作答。

## 2. 目标与非目标

### 2.1 目标

1. 先提供一个不依赖 LangGraph 的裸 ReAct 循环 demo，能真实调 LLM + ch02 工具，看清「带工具的循环」本质。
2. 用 LangGraph 重构为父图 + Agent 子图，State 贯穿全流程，SQLite checkpointer 持久化。
3. 七类意图归入四个出口，分流规则硬编码在代码里：
   - 知识类（商品咨询、退款退货）→ 强制 RAG → 置信度闸 → Agent
   - 业务数据类（物流、订单、售后）→ 直接 Agent
   - 投诉 → 固定安抚话术 + 「转人工」「建工单」两个独立选项
   - 闲聊 → 固定话术，不调模型
4. Agent 是 `agent ↔ tools` 的 LangGraph 子图，支持多步工具调用、流式 token、追问、步数/token 上限。
5. 置信度闸只做最简版：Top-1 重排分 ≥ 可配置阈值才放行；弱证据直接兜底并记录低置信度问题。
6. 前端新增原生 HTML/CSS/JS 页面 `/ch05`，渲染两个独立按钮；「转人工」纯前端模拟，「建工单」才调 ch02 工单工具写 `tickets` 表。
7. 日志能证明强制检索节点、分流、闸门、Agent 步数都走到了。

### 2.2 非目标（本章不做）

- 指代消解正式版：本章原样透传。
- 意图识别正式版/微调模型：本章只用简单 Prompt + JSON。
- 上下文管理策略升级。
- MCP 接入。
- 飞轮入库/低置信度问题的后续挖掘。
- 正式可观测性/置信度检查：留给可观测章节。
- 真实人工客服系统对接。
- 新写业务工具：Agent 只复用 ch02 的 `query_order` / `query_product` / `query_logistics`。

### 2.3 反目标（即使跑通也算失败）

- Agent 自行调用 `create_ticket`，或后端自动转人工/自动建工单。
- 知识类问题绕过强制检索或绕过置信度闸直接作答。
- 「转人工」和「建工单」互相绑定：点一个导致另一个也执行。
- 用户两个按钮都不点、继续发消息时，系统擅自执行任何动作。
- ReAct 无步数上限或无 token 预算，可能死循环。
- 为流式输出而在闸门之前把答案发给用户。

## 3. 约束与已核实的技术事实

### 3.1 固定选型

- 编排：LangGraph 1.2.20（已安装）
- 状态持久化：`langgraph-checkpoint-sqlite` 3.1.3（需新增依赖），`SqliteSaver`
- 现有：FastAPI、LangChain 1.4.10 / langchain-core 1.7.7、SQLAlchemy、DeepSeek OpenAI 协议
- 检索：复用 ch04 `retrieve_hybrid`（默认 `hybrid_rerank`）
- 工具：复用 `app/tools/registry.py` 中的业务工具
- 前端：原生 HTML + CSS + JavaScript，自包含页面 `app/static/ch05.html`

### 3.2 Context7 / 本地已核实

- LangGraph `StateGraph` + `compile(checkpointer=...)`；调用时用 `config={"configurable":{"thread_id": ...}}` 区分会话。
- 子图作为父图节点时，只需在父图 compile 时传 checkpointer，LangGraph 会自动传播给子图。
- 节点内用 `from langgraph.config import get_stream_writer` 发自定义流。
- **子图自定义流必须显式 `subgraphs=True`**：在已安装的 langgraph 1.2.20 上，`graph.stream(..., stream_mode="custom")` 默认不会收到嵌套 Agent 子图节点发出的 token/tool 帧；只有 `subgraphs=True` 才会收到，且子图帧带非空 `ns`。本章只消费 `custom` 帧，忽略 `updates` 帧，最终状态用 `graph.get_state(config)` 读取，避免父子 scope 的 updates 重复。
- `stream_mode="messages"` 可逐 token 流 LLM 输出；本章 Agent 节点用 `get_stream_writer` 转发，避免依赖全局 messages 流。
- LangChain 流式工具调用：用 `gathered = gathered + chunk` 累加 `AIMessageChunk`，结束后 `gathered.tool_calls` 是完整工具调用。
- `SqliteSaver(sqlite3.connect(path, check_same_thread=False))`；首次使用调用 `setup()` 建表；适合本地/单机持久化，生产应换 Postgres。
- 现有 `execute_tool` 已提供参数校验、daemon 线程超时、contextvar 传播；Agent 工具执行复用它。
- 现有 `order_for_prompt` 负责「最强在首尾」排序；证据编号与 sources 帧跟随该顺序。

## 4. 现有系统现状

- `/chat`（ch02/ch04）：单轮工具调用；`query_faq` 可被强制；生成一次后流式输出；没有多步 ReAct。
- `/`：React/Vite 聊天页，构建产物由 FastAPI 托管；本次不改。
- 工具：`query_order`、`query_product`、`query_logistics`、`query_faq`、`create_ticket`。
- `create_ticket` 当前副作用：写入 `tickets` 的同时把 `conversations.status` 改为「已转人工」。本章必须去掉该副作用（见 §6.4）。
- 会话记忆：当前从 MySQL `messages` 重建，没有 checkpointer。

## 5. 方案对比与选择

### 5.1 方案 A：Agent 做成单个节点，节点内部跑手写 ReAct 循环

- 优点：和热身裸循环一一对应；State/流式/测试最简单。
- 缺点：循环内部不在图上，单步不可 checkpoint；「用 LangGraph 重构」只重构了外层骨架。

### 5.2 方案 B：Agent 做成嵌套 LangGraph ReAct 子图（已选）

- `agent`（调 LLM）↔ `tools`（执行工具）两个节点 + 条件边，作为父图的核心节点。
- 优点：每一步工具调用都进 State/checkpoint；步数/token 预算由条件边统一裁决；更贴近「把裸循环重构成图」。
- 缺点：子图与父图共享 `messages` channel 要小心；测试和流式转发多一层。
- 选择理由：本章重点是「Workflow 骨架 + 主力 Agent 节点」，B 让 ReAct 循环本身也成为图的一部分，而不是节点里的黑盒。

### 5.3 已确认的配套决策

| 决策 | 选择 | 理由 |
| --- | --- | --- |
| 前端落点 | 新增 `/ch05` 原生页 + `/ch05/chat`，保留现有 React `/chat` | 章节隔离，满足原生技术栈要求 |
| 根路径入口 | `GET /` 307 重定向到 `/ch05` | 用户在 8000 根路径直接测试投诉按钮；旧 React 聊天页让位，`POST /chat` 保留 |
| 工单语义 | 修改 `create_ticket`：只写 `tickets`，不再改会话状态 | 「转人工」与「建工单」互不绑定 |
| Checkpointer | 新增 `langgraph-checkpoint-sqlite`，SQLite 持久化 | 进程重启后仍能续聊，符合持久化要求 |
| 置信度闸 | Top-1 重排分 ≥ `CONFIDENCE_MIN_SCORE` | 本章最简版；正式版留给可观测章节 |
| 意图识别兜底 | 解析失败/异常 → 「售后」→ Agent | 不误判成闲聊或投诉，交给 Agent 继续处理 |
| Agent 工具范围 | 仅 `query_order` / `query_product` / `query_logistics` + 控制工具 `suggest_handoff` | 禁止自动建工单；知识类已预检索，禁止 Agent 再调 `query_faq` 绕过闸门 |
| 物流工具契约（D7） | `query_order` 返回 `tracking_no`；`query_logistics(tracking_no)` 只接受该单号 | 形成真实数据依赖：两个工具都吃 `order_id` 时模型可能并行一步调俩，改后必须先查订单拿到单号 |
| 会话记忆 | checkpointer 里的 State 为准；同时写 MySQL 做审计/工单关联 | 满足 State 贯穿，又不丢现有审计能力 |

## 6. 总体架构

### 6.1 父图

```text
START
  → resolve_reference        # 指代消解：本章原样透传 + 日志
  → classify_intent          # 简单 Prompt + JSON，七类
  → route_intent             # 硬编码七类 → 四出口
      ├─ knowledge  → retrieve_knowledge → confidence_gate
      │                                   ├─ pass → agent
      │                                   └─ fail → record_log
      ├─ business   → agent
      ├─ complaint  → complaint_reply → record_log
      └─ chitchat   → chitchat_reply  → record_log
  agent → record_log
  record_log → END
```

- `confidence_gate` 失败时直接把兜底话术写进 `answer`，不进 Agent。
- `complaint_reply` / `chitchat_reply` 不调用 LLM。
- `record_log` 统一输出结构化日志。

### 6.2 分流表（硬编码）

| 意图 | 出口 | 是否预检索 | 是否过闸 | 进 Agent | 工具 |
| --- | --- | --- | --- | --- | --- |
| 商品咨询 | knowledge | 是，强制 | 是 | 闸通过才进 | order/product/logistics |
| 退款退货 | knowledge | 是，强制 | 是 | 闸通过才进 | order/product/logistics |
| 物流 | business | 否 | 否 | 是 | order/product/logistics |
| 订单 | business | 否 | 否 | 是 | order/product/logistics |
| 售后 | business | 否 | 否 | 是 | order/product/logistics |
| 投诉 | complaint | 否 | 否 | 否 | 无 |
| 闲聊 | chitchat | 否 | 否 | 否 | 无 |

> 说明：知识类 Agent 不绑定 `query_faq`，避免它在闸门之后再次检索、绕过闸门；退款退货仍可调订单/物流工具核实订单。

### 6.3 Parent State

```python
class CustomerServiceState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]  # checkpointer 累积
    user_message: str
    resolved_message: str          # 指代消解结果；本章 = user_message
    intent: str                    # 七类之一
    route: str                     # knowledge | business | complaint | chitchat
    evidence: list[dict]           # [{"n","section_path","question","answer","score"}]
    sources: list[dict]            # 前端 sources 帧
    confidence: float | None       # Top-1 重排分
    gate_passed: bool
    gate_reason: str               # ok | empty | below_threshold | rerank_failed | infra_error
    rerank_failed: bool
    answer: str
    suggested_actions: list[dict]  # [{"action","label","ticket_type"?}]
    step_count: int
    token_estimate: int
    token_budget: int
    max_steps: int
    conversation_id: int
    exhausted: bool
```

每轮开始时，service 显式重置本轮瞬态字段（`evidence/sources/answer/suggested_actions/step_count/token_estimate/...`），只向 `messages` 追加新的 `HumanMessage`；checkpointer 保留历史消息。

### 6.4 节点行为

#### `resolve_reference`

- 本章：`resolved_message = user_message`，不调模型。
- 记 `logger.info("ch05 resolve_reference passthrough")`，为后续正式版留钩子。

#### `classify_intent`

- 独立 temperature=0 LLM，简单 Prompt，输出 JSON：`{"intent": "...", "reason": "..."}`。
- 七类定义与判定规则写入 Prompt，附带每类 1–2 个短样例；明确「不确定归售后，不要猜投诉/闲聊」。
- 使用 `with_structured_output(IntentResult, method="json_mode")`；解析失败/异常 → `intent="售后"`，记 warning。
- 指代消解目前是透传，所以分类输入就是 `resolved_message`。

#### `route_intent`

```python
INTENT_ROUTES = {
    "商品咨询": "knowledge",
    "退款退货": "knowledge",
    "物流": "business",
    "订单": "business",
    "售后": "business",
    "投诉": "complaint",
    "闲聊": "chitchat",
}
```

未知意图兜底 `business`（并记 warning）。

`route_intent` 是一个普通节点：把 `route_for_intent(intent)` 的结果写入 State 的 `route` 字段；随后由条件边读取 `state["route"]` 决定下一节点。这样日志、闸门和服务层都能看到同一个 `route`。

#### `retrieve_knowledge`

- 直接调用 ch04 `retrieve_hybrid(resolved_message)`，默认 `hybrid_rerank`；不经过 `query_faq` 工具。
- 用 `order_for_prompt` 排序，按顺序生成 `[n]` 证据块和 `sources`。
- 调用 `assess_confidence(result.chunks, settings.confidence_min_score, rerank_failed=result.rerank_failed)`，把 `confidence` / `gate_passed` / `gate_reason` 写入 State；这样判定使用原始 `RetrievedChunk`，不需要把 chunk 对象塞进可序列化 State。
- 记录 `logger.info("ch05 retrieval_forced ...")`，包含意图、命中数、Top-1 分数、`rerank_failed`。
- 检索异常：`gate_reason="infra_error"`、`answer=知识库暂时不可用...`；不写低置信度池（与 ch04 一致，基础设施故障不是知识缺口）。
- `result.rerank_failed=True`：此时 `score` 是 RRF 融合分而非重排分（通常 0.01–0.03），不能拿 `CONFIDENCE_MIN_SCORE` 判断，也不能当弱证据落池。标记 `gate_reason="rerank_failed"`，走保守兜底，不写低置信度池。

#### `confidence_gate`

- 输入：retrieve 阶段写好的 `evidence` / `confidence` / `gate_passed` / `gate_reason` / `rerank_failed`。
- 放行条件（已在 `assess_confidence` 判定）：`evidence` 非空、`rerank_failed=False`，且 Top-1 重排分 ≥ `CONFIDENCE_MIN_SCORE`（配置，默认 0.3，实施时用少量正/负样例校准后定值）。
- 放行：`gate_passed=True`，通过 `get_stream_writer` 先发 `sources` 帧，再进 Agent。
- 失败：`gate_passed=False`，`answer` 写兜底话术，`gate_reason` 为 `empty` / `below_threshold`；不进 Agent。
- `infra_error` / `rerank_failed` 由闸门短路处理：不调用 `assess_confidence`、不进 Agent、不写低置信度池。
- 服务层在整轮结束后，对 `route=="knowledge"` 且 `gate_reason in {"empty","below_threshold"}` 的记录调用 `record_low_confidence(..., source="confidence_gate")`；`infra_error` / `rerank_failed` 不落池。
- 兜底话术（知识缺口）：`抱歉,我暂时没有找到足够可靠的知识库证据来确认这个问题。建议您联系人工客服进一步处理。`
- 检索异常兜底话术：`知识库暂时不可用,请稍后重试或联系人工客服。`
- 重排失败兜底话术：`知识库暂时无法完成可靠性校验,请稍后重试或联系人工客服。`
- 不在闸门失败时给按钮（避免和投诉/Agent 建议的按钮语义混淆）。

#### 固定话术节点的通用约定

`confidence_gate` 失败、`complaint_reply`、`chitchat_reply`、`finalize` 都必须：

1. 通过 `get_stream_writer` 发一个 `{"type":"token","text": <完整话术>}` 帧（固定话术一次性发送，不再拆分）；
2. 返回 `{"answer": text, "messages": [AIMessage(content=text)]}`，让 assistant 回合进入 State/checkpointer；否则下一轮会出现连续 `HumanMessage`，checkpointer 作为会话记忆的目标不成立。
3. 服务层不再重复发送 `answer` 文本，只负责持久化 MySQL 和发送 `options` 帧。

> `finalize` 若在部分 token 已经流出后才触发，追加 fallback `AIMessage`，不撤回已发文本；前端看到「前置说明 + 兜底话术」是可接受的恢复路径。

#### `complaint_reply`

- 固定安抚话术（不调模型）：

  > 非常抱歉给您带来不好的体验，您的感受我们很重视。您可以选择转接人工客服进一步处理，或创建一个工单，我们会尽快跟进。

- 设置 `suggested_actions=[{"action":"transfer","label":"转人工"},{"action":"ticket","label":"建工单","ticket_type":"投诉"}]`。
- 后端不执行转人工、不建工单、不改会话状态。

#### `chitchat_reply`

- 固定话术，例如：

  > 您好，我是客服小猫，很高兴为您服务。请问有什么可以帮您的？

- 不调模型，不检索，不产生选项。

#### `agent`（子图）

详见 §7。

#### `record_log`

- 输出一条结构化 JSON 日志，字段：
  `thread_id/conversation_id, intent, route, gate_passed, gate_reason, confidence, retrieval_hits, rerank_failed, tool_steps, token_estimate, suggested_actions, answer_chars`。
- 不写数据库；数据库审计由 service 负责。

### 6.5 Agent 子图

```text
START → agent
agent ──有 tool_calls 且未超步数/预算──→ tools → agent
agent ──无 tool_calls──────────────────→ END
agent ──有 tool_calls 但超限────────────→ finalize → END
```

State（与父图共享键）：

```python
class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    evidence: list[dict]
    route: str
    step_count: int
    token_estimate: int
    token_budget: int
    max_steps: int
    answer: str
    suggested_actions: list[dict]
    exhausted: bool
    conversation_id: int
```

#### `agent` 节点

1. `step_count += 1`；若超过 `max_steps` 或 `token_estimate >= token_budget`，返回 `exhausted=True`，交给 `finalize`。
2. 用 `build_agent_messages(system_prompt, state["messages"], max_context_tokens)` 组装上下文；system prompt 包含：
   - 客服角色、礼貌简洁、不编造；
   - 本轮知识库证据（仅知识类）与引用规则；
   - 可用业务工具说明；
   - 「需要人工介入时调用 `suggest_handoff`，不得调用 `create_ticket`」；
   - 「后一个工具依赖前一个结果时必须串行；如先确认订单再查物流，不要并行」；
   - 「信息不足时向用户追问」。
   - **引用规则分层**：知识库证据里的事实必须带 `[n]`，编号只能来自本轮证据；业务工具返回的订单/物流/商品事实直接转述，**不需要也不能**给 `[n]`；不得给工具事实编造引用编号。这样复用 ch04 引用约束时，不会把允许的工具调用误判为违规。
3. `llm.bind_tools(tools_for_route(route)).stream(messages)` 逐 chunk 累加：
   - 有文本内容：立即通过 `get_stream_writer` 发 `{"type":"token","text": ...}`；
   - 用 `gathered = gathered + chunk` 累加完整 `AIMessageChunk`。
4. 结束后规范化为 `AIMessage(content=..., tool_calls=gathered.tool_calls)`，追加到 `messages`。
5. 若无 `tool_calls`：`answer = content`；若内容为空，置 `exhausted=True` 交给 `finalize`。
6. 估算 token：`token_estimate += estimate(prompt) + estimate(output)`。

> 说明：工具调用轮如果模型先输出了一段「我来查一下」文本，也会实时显示；这符合流式体验，不视为错误。DeepSeek 工具调用轮通常 content 为空。

#### `tools` 节点

- 遍历上一条 AI 消息的全部 `tool_calls`：
  - 先发 `{"type":"tool","tool":name,"label":中文}`；
  - `suggest_handoff`：**控制工具，不执行任何业务动作**。契约见 §6.5.1；`tools` 节点识别到该工具名后直接处理，不进入 `execute_tool`；
  - 业务工具：从 `AGENT_TOOLS_BY_NAME` 取工具，设置 `current_conversation_id`，调 `execute_tool`（复用超时/重试/线程），结果截断到 `AGENT_TOOL_RESULT_MAX_CHARS` 后作为 ToolMessage 回填；
  - 未知工具：回一条错误 ToolMessage，不中断循环。
- 更新 `token_estimate`。
- `create_ticket` **不在** `tools_for_route` 中，因此 Agent 永远无法自动建工单。

#### 6.5.0 物流工具链式契约（D7）

```python
@tool
def query_order(order_id: str) -> str:
    """...返回 {"order_id","items","amount","status","tracking_no"}"""

@tool
def query_logistics(tracking_no: str) -> str:
    """...返回 {"tracking_no","carrier","current_node","traces"}"""
```

- `query_order` 每次返回都带 `tracking_no`（`SF` + 10 位数字）。
- `query_logistics` 只接受 `tracking_no`，不再接受 `order_id`；参数 schema 必填，旧 `{"order_id": ...}` 调用会得到工具参数校验错误。
- 不做额外的单号格式闸：mock 没有可核验的签发表，schema + docstring 已足以强制「先查订单」；模型若猜测单号，属于 Prompt 服从性问题，由验收日志观察。
- 该变更修改的是 ch02 现有工具，不新增业务工具；旧 `/chat` 的模型也会被迫先调 `query_order`。

#### 6.5.1 `suggest_handoff` 契约

```python
@tool
def suggest_handoff(
    options: list[Literal["transfer", "ticket"]],
    ticket_type: Literal["售后", "投诉", "咨询"] = "咨询",
    reason: str = "",
) -> str:
    """建议前端展示「转人工」/「建工单」按钮；只记录建议，不执行任何动作。"""
```

- 只允许 `transfer` / `ticket`；`ticket_type` 只用于 ticket 选项，默认 `咨询`。
- 一次可只给一个或两个；`tools` 节点把合法选项写入 `suggested_actions`：
  - `transfer` → `{"action":"transfer","label":"转人工"}`
  - `ticket` → `{"action":"ticket","label":"建工单","ticket_type": ticket_type}`
- 重复调用按 `action` 去重合并（不覆盖已存在的 action；同一 action 的 `ticket_type` 以后一次为准）。
- 非法/空选项：不写入 State，返回 ToolMessage `参数无效: options 只能是 transfer/ticket。`，让模型继续作答；不抛异常。
- 无论是否调用该工具，后端都不会自动转人工或建工单；只有前端按钮会触发 `/ch05/tickets`。

#### `finalize`

- 若被步数/token 上限截断，或最终答案为空：
  - 通过 writer 发 fallback 文本，并追加 `AIMessage(fallback)`；
  - 固定兜底：「抱歉,这个问题需要更多信息才能继续处理。您可以补充订单号或具体诉求,也可以转人工客服进一步处理。」
  - `suggested_actions` 若为空，给 `[{"action":"transfer","label":"转人工"}]`（只给一个，符合「可只给一个」）。
- 不调用 LLM，不再消耗 token。

#### 停止条件

1. 模型不再发起工具调用（正常收敛）；
2. `step_count >= AGENT_MAX_STEPS`（默认 6，覆盖「查订单 → 查物流 → 作答」）；
3. `token_estimate >= AGENT_TOKEN_BUDGET`（默认 12000，字符估算）；
4. 最终答案为空 → `finalize`；
5. 工具异常不终止循环，交给下一步模型决定；未知工具同理。

#### 工具集

```python
BUSINESS_TOOLS = [query_order, query_product, query_logistics]
CONTROL_TOOLS = [suggest_handoff]
AGENT_TOOLS = BUSINESS_TOOLS + CONTROL_TOOLS
```

- 四个出口中只有 `knowledge` / `business` 进 Agent。
- `query_faq` 不绑定给 Agent；知识类已由 `retrieve_knowledge` 强制检索。
- `create_ticket` 不绑定给 Agent；只能由前端按钮触发 `/ch05/tickets`。

### 6.6 Checkpointer 与会话持久化

- 新增依赖 `langgraph-checkpoint-sqlite==3.1.3`。
- 配置 `LANGGRAPH_CHECKPOINT_DB`，默认 `.data/ch05-checkpoints.sqlite`；首次 `/ch05` 请求时懒加载：`open_sqlite_checkpointer(path)` 返回 `CheckpointerHandle(saver, close)`，编译父图（checkpointer 自动传播给 Agent 子图），把 graph 和 handle 缓存在 `app.state`，FastAPI shutdown 时调用 handle.close()。
- 懒加载而不是 lifespan 启动时构建：现有测试用 `with TestClient(app)` 会触发 lifespan，若此时构建会读真实模型/写真实 SQLite；懒加载让未访问 `/ch05` 的测试完全不碰新图。并发首次请求用 `app.state.ch05_graph_lock` 保护。
- 连接使用 `sqlite3.connect(path, check_same_thread=False)`，启用 `PRAGMA journal_mode=WAL` 和 `busy_timeout`，缓解并发写。
- `thread_id = f"user:{session_id}"`（即请求里的 `user_id`），不依赖 MySQL 自增 `conversation.id`：MySQL 重建后即使 id 复用，也不会串到旧 checkpoint；同一前端会话跨请求/重启沿用同一 thread。
- 每轮输入重置瞬态字段，`messages` 只追加新的 `HumanMessage`；checkpointer 保留历史与 Agent 工具轨迹。
- MySQL 仍写 `conversations` / `messages` 做审计：
  - 用户消息在启动图前写入；
  - 助手最终答案在图结束后写入；
  - 工具中间消息不重复写 MySQL（checkpointer 已有；需要审计时从 State 读取）。
- graph 不在 import 时构建：`app/agent/graph.py` 暴露 `build_graph(llm, intent_llm, checkpointer, settings)`；`app/api/ch05.py` 的 `get_ch05_graph_dep` 首次请求时构建并缓存。
- 测试注入缝：`build_graph(..., checkpointer=InMemorySaver(), llm=FakeLLM, intent_llm=FakeIntentLLM)` 可离线构建；`/ch05` API 依赖通过 FastAPI `dependency_overrides` 替换 graph，绝不触碰 `.data/ch05-checkpoints.sqlite` 或真实模型。

### 6.7 SSE 契约

`POST /ch05/chat`，请求与 ch01 相同：`{"session_id": "...", "message": "..."}`。

帧类型：

| 帧 | 内容 |
| --- | --- |
| 文本 token | 裸文本，直接拼接 |
| tool | `{"type":"tool","tool":"query_logistics","label":"查物流"}` |
| sources | `{"type":"sources","sources":[{n,chunk_id,section_path,question,answer,category,content_type,score}]}` |
| options | `{"type":"options","options":[{"action":"transfer","label":"转人工"},{"action":"ticket","label":"建工单","ticket_type":"投诉"}]}` |
| error | `{"error":"客服服务暂时不可用,请稍后重试"}` |
| 结束 | `[DONE]` |

- `sources` 只在置信度闸通过后、进入 Agent 前发出。
- 固定话术节点自己发 token 帧；Agent 节点边生成边发 token 帧。
- `options` 在 graph 结束后由 service 读取最终 State 发出，保证在答案 token 之后；前端据此渲染独立按钮。
- 错误发生且流已开始时，不返回 500，发 error 帧后仍以 `[DONE]` 结束（沿用 ch01/ch02 约定）。

#### 流式消费方式

```python
for item in graph.stream(
    inputs, config, stream_mode="custom", subgraphs=True, version="v2"
):
    # langgraph 1.2.20：subgraphs=True 时 custom 帧是
    # {"type": "custom", "ns": ..., "data": <节点发出的 dict>}，不是 (ns, frame) 元组。
    if item.get("type") == "custom":
        yield item["data"]
final_state = graph.get_state(config).values
```

- 只消费 `custom`，不消费 `updates`，避免父子 scope 的重复更新。
- 转发前按业务帧类型白名单过滤：`token` / `tool` / `sources`；未知帧只记日志。
- 最终 `options` 从 `final_state["suggested_actions"]` 读取。

### 6.8 工单接口

`POST /ch05/tickets`

```json
{
  "session_id": "前端会话 ID",
  "description": "用户投诉内容或工单描述",
  "ticket_type": "投诉"
}
```

- 解析/创建当前会话，设置 `current_conversation_id`，调用 ch02 `create_ticket` 工具（经 `execute_tool`）写 `tickets` 表。
- `create_ticket` 修改为**只写 tickets**，不再修改 `conversations.status`；同步更新 ch02 测试。
- `execute_tool(..., retries=0)`：工单写入不是幂等操作，超时后自动重试可能重复建单；宁可返回 502 让用户手动重试。
- 成功判定不用模糊字符串匹配：`create_ticket` 成功返回工单号，用公开正则 `^T\d{11}$`（T + YYYYMMDD + 3 位序号）校验；失败/超时统一 502。
- 返回 `{"ticket_no": "T20261117001", "message": "工单已创建"}`；失败返回 502 + 错误信息。
- 没有「转人工」后端接口；转人工完全由前端模拟。

## 7. 前端 `/ch05`

### 7.1 页面

- `app/static/ch05.html`，自包含 CSS/JS（与 `/kb`、`/rag-eval` 一致）。
- 路由 `GET /ch05` 返回该文件。
- 会话 ID 存 `sessionStorage`（独立 key，如 `lamp-cs-ch05-session-id`），刷新页面保持同一会话。

### 7.2 交互

- 输入框 + 发送按钮；SSE 流式追加到当前助手气泡。
- `tool` 帧显示一行灰色状态（如「正在查物流…」），不阻塞后续 token。
- `options` 帧在助手气泡下方渲染按钮：
  - 「转人工」：**不请求后端**；立即显示系统消息「已转接人工客服」，随后显示「您好，我是客服小猫，请问有什么可以帮您的」。只禁用该按钮。
  - 「建工单」：先 `confirm("确认创建工单吗？")`；确认后 POST `/ch05/tickets`，`description` 用最近一条用户消息，`ticket_type` 用 options 里的值或 `咨询`。成功后显示「工单已创建：T...」并只禁用该按钮；失败显示错误。
  - 两个按钮互不绑定：点一个不影响另一个；都不点继续发消息，正常走下一轮，不产生任何隐藏动作。
- 无数据超时 90 秒（Agent 多步 + 工具可能较慢）；超时显示可重试错误。
- 错误帧显示为一条系统提示，不破坏已有消息。

### 7.3 与旧页面的关系

- `GET /` 返回 307 重定向到 `/ch05`（2026-11-17 用户要求）：根路径直接进入带投诉按钮的 ch05 页面，旧 React 页面不再作为聊天入口；若存在 `frontend/dist`，其静态资源仍由 StaticFiles 挂载。
- `POST /chat` 与旧 React 组件完全保留，供旧接口/测试兼容；`/` 不再返回 React 页面或「尚未构建」503。
- `create_ticket` 的副作用移除会让旧 `/chat` 在 Agent 调工单时不再改会话状态；这是本次明确的产品语义修正。

## 8. 文件变更清单

### 新增

- `app/agent/__init__.py`
- `app/agent/state.py` — Parent/Agent State、常量、类型
- `app/agent/intent.py` — 意图 Prompt、`classify_intent`、兜底
- `app/agent/routing.py` — `INTENT_ROUTES` / `tools_for_route`
- `app/agent/confidence.py` — 置信度判定
- `app/agent/tools.py` — `suggest_handoff` 控制工具、`AGENT_TOOLS`
- `app/agent/raw_loop.py` — 裸 ReAct 循环（热身）
- `app/agent/raw_demo.py` — CLI demo
- `app/agent/subgraph.py` — Agent 子图
- `app/agent/graph.py` — 父图、checkpointer 构建
- `app/agent/service.py` — `/ch05/chat` 的流式编排、DB 审计、低置信度落池
- `app/api/ch05.py` — `GET /ch05`、`POST /ch05/chat`、`POST /ch05/tickets`
- `app/static/ch05.html` — 原生聊天页
- `tests/eval/ch05_intent_cases.json` — 七类标注样例
- `app/jobs/eval_intent.py` — 意图样例评估脚本（真实 LLM）
- 测试文件：`tests/test_agent_raw_loop.py`、`test_agent_intent.py`、`test_agent_routing.py`、`test_agent_confidence.py`、`test_agent_subgraph.py`、`test_agent_graph.py`、`test_ch05_api.py`、`test_intent_cases.py`

### 修改

- `app/config.py` — 新增 `AGENT_MAX_STEPS`、`AGENT_TOKEN_BUDGET`、`AGENT_TOOL_RESULT_MAX_CHARS`、`CONFIDENCE_MIN_SCORE`、`LANGGRAPH_CHECKPOINT_DB`
- `app/prompts/customer_service.py` — 新增 Agent system prompt renderer（复用引用/负面约束，去掉 query_faq 路由，加入 suggest_handoff 规则）
- `app/services/context.py` — 提供可复用的历史裁剪函数（供 Agent 多步消息裁剪）
- `app/tools/ticket.py` — 去掉改 `Conversation.status` 的副作用
- `tests/test_tools_ticket.py` — 同步断言
- `app/main.py` — 注册 ch05 路由；构建 checkpointer/graph；意图 LLM 依赖
- `requirements.txt` — `langgraph-checkpoint-sqlite==3.1.3`
- `.gitignore` — `.data/` 或 checkpoint 文件
- `dev-notes/ch05.md` — 分阶段追记

## 9. 测试与验证

### 9.1 TDD 单元测试（无真实网络）

- `test_agent_raw_loop.py`：无工具直接收敛；一次工具调用后收敛；两次工具调用多步；超步数；超 token 预算；emit 事件顺序。
- `test_agent_intent.py`：Prompt 包含七类；合法 JSON 解析；非法 JSON/异常兜底「售后」。
- `test_agent_routing.py`：七类逐一映射到四出口；未知意图兜底。
- `test_agent_confidence.py`：空证据、低于阈值、等于阈值、高于阈值、infra_error 不落池的判定输入。
- `test_agent_subgraph.py`：Agent 子图一步收敛；两次工具多步；`suggest_handoff` 写入选项且不执行动作；`create_ticket` 不在绑定工具中；步数/token 上限进 finalize；流式 token/tool 帧。
- `test_agent_graph.py`：知识类走 retrieve → gate → agent；闸失败不进 agent；业务类跳过检索；投诉/闲聊不调 LLM；日志节点字段；低置信度落池（service 层）。
- `test_ch05_api.py`：`/ch05` 200；`/ch05/chat` SSE 帧顺序（sources/tool/token/options/DONE）；`/ch05/tickets` 写 tickets 且不改 status；转人工不调后端。
- `test_intent_cases.py`：数据集 schema 校验（七类齐全、数量下限、无重复）。

### 9.2 Prompt/数据类验证（替代 TDD 的部分）

- `tests/eval/ch05_intent_cases.json`：42 条（每类 6 条），含边界样例：
  - 商品咨询 vs 售后；
  - 退款退货 vs 售后；
  - 投诉 vs 退款退货（带情绪的退款诉求）；
  - 闲聊 vs 订单（「你好，在吗」）。
- 运行 `python -m app.jobs.eval_intent`（真实 DeepSeek），输出总体准确率、每类准确率、混淆矩阵、错例。
- 通过标准：总体 ≥ 90%，每类 ≥ 80%；低于则调整 Prompt/样例后重跑，把最终结果写入 `docs/superpowers/reports/` 和 dev-notes。

### 9.3 端到端验收（对应验收标准）

1. **政策类强制检索**：问「你们的退货政策是什么？」→ 终端日志出现 `ch05 retrieval_forced`、`ch05 confidence_gate`，sources 帧出现；回答带引用。
2. **物流问题 Agent 自调工具**：问「订单 1001 的物流到哪了」→ 日志/帧出现 `query_logistics`，回答按工具结果作答。
3. **投诉按钮**：说「我要投诉」→ 前端出现「转人工」「建工单」两个独立按钮；点转人工只出现「已转接人工客服」+ 客服小猫问候，不写库；点建工单才写 `tickets`，且 `conversations.status` 仍为「进行中」；都不点继续聊天，系统不做任何动作。
4. **闲聊固定话术**：说「你好」→ 固定话术，日志无 LLM/检索调用。
5. **复杂问题多步 ReAct**：问「先帮我确认订单 1001 的状态，如果已经发货，再查它的物流到哪了」→ 日志/工具帧显示至少两次 Agent 步进（`query_order` → `query_logistics`）；`query_logistics` 的 `tracking_no` 必须等于前一步 `query_order` 返回的单号，证明不是并行调用。若模型猜测单号，收紧 Prompt 后复测。

### 9.4 质量门

- 全量 pytest 通过。
- 代码走 `superpowers:requesting-code-review`；修复后 scoped re-review。
- 不破坏旧 `/chat` 现有测试；`create_ticket` 语义变更同步测试。
- 子图流式测试必须覆盖 `stream(..., stream_mode="custom", subgraphs=True)` 能收到 Agent 子图 token/tool 帧；若真实 LangGraph 行为与预审结论不符，先修流式适配再继续。

## 10. Builder 预审裁决

独立 builder 读了设计与现有代码，提出 8 个会改变计划的问题；均已在本文件裁决，无须打断用户的开放项。

| # | 问题 | 裁决 |
| --- | --- | --- |
| 1 | 子图 custom 帧收不到 | `subgraphs=True`；只消费 `custom`，最终状态用 `get_state`；白名单转发 token/tool/sources |
| 2 | rerank 失败时 RRF 分数不能过 0.3 闸 | 新增 `gate_reason="rerank_failed"`，保守兜底，不落低置信度池 |
| 3 | `suggest_handoff` 契约不清 | 固定 schema：`options: list[Literal["transfer","ticket"]]`、`ticket_type`、`reason`；union 去重；非法选项回错误 ToolMessage |
| 4 | 固定话术节点没写回 messages | 所有终态节点都发 token 帧并追加 `AIMessage`；service 不重复发文本 |
| 5 | graph/checkpointer 构建与测试注入 | lifespan 构建一次，`app.state.ch05_graph`；`build_graph(...)` 可注入 fake LLM + InMemorySaver；API 用 dependency_overrides |
| 6 | `/ch05/tickets` 成败判定与重复建单 | `retries=0`；成功用 `^T\d{11}$` 校验；失败 502 |
| 7 | 知识轮工具事实的引用规则 | 新增分层引用规则：知识事实必须 `[n]`，工具事实不得带 `[n]` |
| 8 | `thread_id` 在 MySQL 重建后复用 id | 改用 `user:{session_id}`，不依赖自增 conversation.id |

## 11. 风险与开放项

1. **置信度阈值未校准**：默认 0.3 只是起点；实施时用少量正/负样例实测重排分后定值，并把校准结果写进 dev-notes。若无法调用真实 reranker，保留配置项并在验收时用构造数据验证闸门逻辑。
2. **多步串行依赖模型的服从性**：D7 已把 `query_logistics` 入参改为 `tracking_no`，单号只能从 `query_order` 返回中获得，理论上无法并行；但模型仍可能猜测或伪造单号。Prompt 明确串行规则与「单号须取自 query_order 返回」，验收时核对两工具的帧内容与步数；若稳定性不足，再收紧工具 docstring，但不新增业务工具。旧 `/chat` 的历史消息中可能残留 `{"order_id": ...}` 的旧工具调用参数，属于只读历史回放，不回放执行。
3. **SQLite checkpointer 并发**：单文件单写者，适合本地/单机；WAL + busy_timeout 缓解。生产应换 Postgres，本章不做。
4. **流式 token 与工具轮**：工具调用轮若模型先输出文本，会实时显示；不尝试撤回。若验收要求「工具轮不显示前置文本」，可改为缓冲到该轮结束再决定是否展示，但会牺牲流式感。
5. **意图分类成本**：每轮一次额外 LLM 调用；可接受。闲聊本来不花模型调用，但意图分类本身会花一次——需求中的「闲聊不花模型调用」指闲聊回复不调模型；分类是必要的路由成本，设计文档明确这一点。
6. **旧 `/chat` 与新 `/ch05` 并存**：旧页面不展示新按钮；ch05 的验收与旧页面隔离，避免回归。
7. **工单超时但已提交**：`retries=0` 已消除自动重试导致的重复建单；若工具内部已提交但响应超时，前端会看到 502，用户手动重试仍可能重复。本章不引入幂等键，前端在请求期间禁用按钮降低概率。

## 12. 交付物

- 设计文档：本文件
- 实施计划：`docs/superpowers/plans/2026-11-17-ecommerce-cs-ch05-agent-workflow.md`
- 开发记录：`dev-notes/ch05.md`（分阶段追记，不一次性补）
- 意图评估报告：`docs/superpowers/reports/2026-11-17-ch05-intent-eval.md`
- 演示命令：
  - 启动：`.venv/Scripts/python.exe run.py`
  - 裸循环：`.venv/Scripts/python.exe -m app.agent.raw_demo "订单 1001 的物流到哪了"`
  - 页面：`http://localhost:8000/ch05`
  - 测试：`.venv/Scripts/python.exe -m pytest -q`
