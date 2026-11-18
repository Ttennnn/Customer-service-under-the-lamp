# 第六章 · 意图识别与对话管理（占位分流器 → 正式版）— 设计文档

- 日期：2026-11-18
- 分支：`feature/ch06-dispatcher`（基于 `feature/ch05-agent-workflow` / PR #3）
- 状态：待用户定稿
- 权威来源：用户提供的 ch06 参考 spec（基于 `mewhelp-course/ch06-intent-recognition/README.md`）+ 本仓 ch05 实现
- 关联：ch01 会话、ch02 工具、ch03/ch04 RAG、ch05 Workflow + ReAct 子图

## 1. 意图与为什么

ch05 已经把 Workflow 骨架搭好：指代消解、意图识别、按意图分流、知识检索、置信度闸、主力 Agent、日志。链路能跑，但**分流器前段仍是占位**：

- `resolve_reference` 原样透传，多轮里的「它能退吗」无法补全；
- `classify_intent` 是单标签七类，没有 confidence、没有「其他」，异常回退「售后」；
- 知识路只做单 query 检索，退款/售后没有确定性前置步骤；
- 退款/售后没有槽位处理，缺订单号时只能靠模型追问或猜。

本章把分流器做成正式版：**先理解、再判意图、再按五条确定性出口分流**。目标不是让模型更自由，而是让多轮漂移、口语指代、退款这类高风险场景**可预测、可评估、可复查**。

## 2. 目标与非目标

### 2.1 目标

1. **指代消解 + Query 改写合一**：一次 LLM 调用，结合最近几轮历史，把「它能退吗」补成脱离上下文也能看懂的 `resolved_query`；已完整、指代已明确的原样透传，不强行改写。
2. **意图识别四件套**：八类枚举成选择题（物流/订单/商品咨询/退款退货/售后/投诉/闲聊/**其他**）；强制 JSON `{intent, confidence}`；边界 few-shot；拿不准归「其他」。
3. **分流精化**：`route_by_intent` 返回五出口 `escalate / fallback_script / knowledge / refund_flow / business`；退款退货/售后走确定性子流程。
4. **Query 扩写**：退款/售后场景把问题泛化成**严格 3 条**检索查询，强制 JSON 单字段 `{queries}`；多 query 检索后去重合并；只在检索侧现查现用，入库侧不拆存多份。
5. **槽位处理**：缺订单号不让模型猜；执行阶段 `interrupt` 弹订单选择器，点选后 `Command(resume)` 回填并继续子流程；退款原因不追问，用户提交退款单时从固定类目下拉选。
6. **前端配套**：订单选择器做成聊天流里可点选的订单卡片；退款单是简单表单，原因固定下拉。
7. **模型配置**：意图识别默认走够格大模型求准；小模型降级路本章只落配置种子，不实现运行时。

### 2.2 非目标（本章不做）

- 微调小模型 / BERT 意图分类。
- 跨会话记忆。
- 意图降级路的运行时实现（`intent_mode="cost"` 的判定/升级逻辑）。
- 两级分类、检索式意图识别（参考 README 明标为面试延伸）。
- 新增检索组件、新增业务表、入库侧多份知识。
- ch06 图 21 SVG 调整。
- 新增非流式 `/api/agent` 端点和 `Makefile`；eval 直接调用图，不新增端点。

### 2.3 反目标（即使跑通也算失败）

- 模型猜订单号，或把上一次的订单号硬塞给新问题。
- 知识类/退款类绕过强制检索，或在置信度闸之前把答案流给用户。
- Agent 自动提交退款/自动建单；退款/建单必须由用户在前端确认。
- 订单选择器用「再发一条普通消息」重跑分流，导致重复分类、重复检索。
- 指代消解引入历史里没有的事实，或把已完整的问句改出原意。
- 把「其他」硬塞成某个业务意图。
- 扩写用于商品咨询/闲聊等简单场景。
- 入库侧按 query 拆存多份知识。

## 3. 约束与已核实事实

### 3.1 固定选型

- 编排：沿用 ch05 的 LangGraph **1.2.20**（已安装）。
- 持久化：沿用同步 `SqliteSaver` + `langgraph-checkpoint-sqlite` 3.1.3；不引入 `AsyncSqliteSaver`。
- 上游：沿用 DeepSeek OpenAI 协议（`ChatOpenAI`）；参考 spec 里的 glm 502 经验不适用于本仓，但扁平 schema 仍保留。
- 检索：沿用 ch04 `retrieve_hybrid`（默认 `hybrid_rerank`）。
- 业务工具：沿用 ch02 `query_order` / `query_product` / `query_logistics` / `create_ticket`；新增 mock `list_user_orders` 和 `submit_refund` 控制工具。
- 工单：复用 `tickets` 表，`ticket_type` 增加 `退款`。
- 前端：原生 HTML/CSS/JS，Vibe Coding，不套 brainstorm/TDD/code review。
- 无新增组件、无新增依赖、无新增业务表。

### 3.2 Context7 / 本地已核实

- LangGraph `interrupt()` / `Command(resume=...)` 是 1.2.20 自带能力；`StateSnapshot.interrupts: tuple[Interrupt, ...]`、`PregelTask.interrupts` 已在本机源码确认。
- 恢复时**被中断节点从头重跑**，`interrupt()` 的返回值是 resume 值。因此 `fetch_order` 中 `interrupt()` 之前的代码必须只读、不可有副作用。
- 现有流式消费是 `graph.stream(..., stream_mode="custom", subgraphs=True, version="v2")`，只收自定义帧；中断不作为 custom 帧转发，而是在流结束后用 `graph.get_state(config).interrupts` 探测，再由 service 发 `interrupt` 帧。
- `SqliteSaver` 为同步 saver，ch05 已验证；本章不换 async。
- DeepSeek 的 `with_structured_output(..., method="json_mode")` 可用；扩写使用扁平 `list[str]`，避免嵌套 schema 的不稳定。
- **R1 红线**：动手前先跑 `tests/test_interrupt_smoke.py`，用 `InMemorySaver` 钉死 1.2.20 的实测 surface（custom stream + `get_state().interrupts` + `Command(resume)` + 节点重跑次数）。若与预审不符，先停，按选型内解决或问用户。

### 3.3 基线映射（参考 spec → 本仓）

| 参考 spec | 本仓 | 处理 |
|---|---|---|
| `app/graph/` | `app/agent/` | 直接映射 |
| `app/core/intent.py` | `app/agent/intent.py` | 重写 |
| `app/core/query_understanding.py` | `app/rag/query_understanding.py` | 增加扩写 |
| `app/core/prompts.py` | `app/prompts/customer_service.py` + `app/agent/coref.py` | 拆分 |
| `/api/chat`、`/api/actions` | `/ch06/chat`、`/ch06/chat/resume`、`/ch06/refunds` | 新增 `/ch06/*` |
| `AsyncSqliteSaver` | 同步 `SqliteSaver` | 不换 |
| langgraph 1.2.9 | 1.2.20 | 以本机为准 |
| glm | DeepSeek | 以本机为准 |
| `main_agent` 节点 | 父图节点当前叫 `agent` | ch06 改名 `main_agent` |
| 路由 `escalate/fallback_script` | 当前 `complaint/chitchat` | ch06 改名 |
| `confidence_check` / `log` | 当前 `confidence_gate` / `record_log` | 保留原名，spec 标映射 |
| `app/static/index.html` | `app/static/ch06.html` | 新页面 |
| `scripts/eval_*.py` | `app/jobs/eval_*.py` | 沿用本仓约定 |
| `Makefile` | 无 | 不新增，文档写命令 |

## 4. 现有系统现状

- 父图（`app/agent/graph.py`）：`resolve_reference → classify_intent → route_intent → {knowledge/business/complaint/chitchat}`；知识类 `retrieve_knowledge → confidence_gate → agent`；业务类直接 `agent`；投诉/闲聊固定话术；`record_log` 统一日志。
- 意图：`app/agent/intent.py` 七类 + `reason`，异常兜底「售后」。
- 检索：`app/rag/retrieval.py` 的 `retrieve_hybrid` 支持 `analysis` 参数，可跳过内部 LLM 改写；`app/rag/query_understanding.py` 已有单 query 改写 + 同义词。
- 订单：`app/tools/business.py` 的 `query_order` 随机 mock，无订单表、无订单列表。
- 工单：`tickets.ticket_type` 枚举为「售后/投诉/咨询」；`create_ticket` 只写 tickets，不改会话状态。
- 页面/API：`GET /` 与 `GET /ch05` 服务 `app/static/ch05.html`；`POST /ch05/chat` 流式；`POST /ch05/tickets` 建单；checkpointer 懒加载。
- 测试：ch05 已有 469 passed 基线；本章会更新受影响的 ch05 测试。

## 5. 方案对比与选择

### 5.1 退款/售后子流程放哪

- **A（已选）**：放父图，做成普通节点 `fetch_order → retrieve_policy`，再接现有 `confidence_gate` / `main_agent`。确定性最强，每一步进 checkpoint，日志清楚。
- B：做成嵌套 `refund_flow` 子图。更模块化，但子图流式 + 中断 + resume 要多一层适配，ch05 已经踩过 `subgraphs=True` 的坑；收益不足以抵消复杂度。

### 5.2 缺订单号怎么暂停

- **A（已选）**：LangGraph `interrupt()` + `Command(resume=order_id)`。官方 HITL 语义，状态由 checkpointer 保存，恢复从暂停点继续。
- B：前端发普通消息带 order_id，图重新跑一遍分流。简单，但会把「点选」变成新问题，重复分类/检索，且无法保证「执行阶段」语义。

### 5.3 指代消解 + 改写的输出

- **A（已选，参考 D5）**：一次 LLM 调用，单字段 `{"resolved_query": "..."}`。指代消解与口语归一在同一个 prompt 里完成；日志同时打 original/resolved，验收能看到补全结果。
- B：返回 `{"standalone_query","normalized_query"}` 两个字段。验收更直观，但与参考 spec 的单字段约定不一致，prompt/评估也更复杂；本章不采用。

### 5.4 多 query 结果怎么合并

- **A（已选）**：每条 query 各检索一次；按 `chunk_id` 去重，保留最高 rerank 分，按分降序；再交给 `order_for_prompt` / `CitationCollector`。闸门继续用 Top-1 重排分，语义不变。
- B：跨 query 做 RRF 融合。排序更鲁棒，但会把「重排分」替换成「融合分」，与 `CONFIDENCE_MIN_SCORE` 的语义不一致；本章不采用。

### 5.5 「其他」怎么处理

- A：其他 → `business` → `main_agent`，由 Agent 追问澄清（符合最初需求「需求澄清留在主力 Agent」）。
- **B（用户已选）**：其他 → `fallback_script` → `script_reply` 固定文案「请具体说明您要咨询的订单、物流、商品或售后问题」。
- 后果：最初需求里「需求澄清留在主力 Agent」对「其他」这一类不再成立；业务/知识/退款流程内的缺信息追问仍由 `main_agent` 完成。dev-notes 记录此变更。

### 5.6 已确认决策

| # | 决策 | 选择 |
|---|---|---|
| D1 | 订单选择器 | `interrupt` / `Command(resume)`，不用两轮重入 |
| D2 | 退款单落库 | 复用 `tickets`，`ticket_type='退款'`，不新建 refund 表 |
| D3 | 意图降级路 | 只落配置种子，默认 `accuracy` 走大模型，不实现运行时升级 |
| D4 | 退款两出口 | `submit_refund` 控制工具被 `tools` 节点拦截成 `refund_form` 动作，不写库 |
| D5 | 指代消解 + 改写 | 合一节点 `resolve_reference`，单字段 `resolved_query`；意图只判不改写 |
| D6 | Query 扩写 | 严格 3 条，只在 `refund_flow` 检索侧；商品咨询走单 query + 同义词 |
| D7 | 五出口 | `escalate / fallback_script / knowledge / refund_flow / business`；其他归 `fallback_script` |
| D8 | 置信度闸 | 沿用 ch05，在 Agent 之前；知识类和退款类都过闸 |

## 6. 总体架构

```text
START
 → resolve_reference        # coref + 口语归一（吃最近几轮历史；已完整透传）
 → classify_intent          # 八类选择题 + {intent, confidence}（吃 resolved_query + 历史）
 → route_by_intent          # 写 route，五出口
      ├─ escalate           → complaint_reply ───────────────────────────→ record_log
      ├─ fallback_script    → script_reply（闲聊/其他固定文案）──────────→ record_log
      ├─ knowledge          → retrieve_knowledge → confidence_gate ──┬─ 强 → main_agent
      │                                                              └─ 弱/异常 → record_log
      ├─ refund_flow        → fetch_order → retrieve_policy
      │                       → confidence_gate ─────────────────────┬─ 强 → main_agent
      │                                                              └─ 弱/异常 → record_log
      └─ business           → main_agent
 main_agent → record_log
 record_log → END
```

- `main_agent` 是父图节点，内部是 ch05 的 `agent ↔ tools` ReAct 子图；全图共用一个。
- `confidence_gate` 沿用 ch05 节点名（参考 spec 叫 `confidence_check`），负责发 `sources`、写兜底、短路 infra/rerank。
- `record_log` 沿用 ch05 节点名（参考 spec 叫 `log`）。

### 6.1 八类意图 → 五出口

```python
def route_for_intent(intent: str) -> Route:
    if intent == "投诉":
        return "escalate"
    if intent in ("闲聊", "其他"):
        return "fallback_script"
    if intent == "商品咨询":
        return "knowledge"
    if intent in ("退款退货", "售后"):
        return "refund_flow"
    if intent in ("物流", "订单"):
        return "business"
    logger.warning("ch06 unknown intent %r, falling back to business", intent)
    return "business"
```

- 未知意图（编程错误/脏数据）仍兜底 `business` 并记 warning；`classify_intent` 的异常/越界会先归「其他」，所以正常链路不会走到这里。
- 条件边映射：`escalate→complaint_reply`、`fallback_script→script_reply`、`knowledge→retrieve_knowledge`、`refund_flow→fetch_order`、`business→main_agent`。

## 7. 节点行为

### 7.1 `resolve_reference`

- 输入：`user_message` + `state["messages"]` 中除当前消息外的最近历史。
- 历史渲染：取 user/assistant 的文本，以及 `[订单选择器]` 这类系统标记；跳过 tool 消息；从新到旧取，受字符预算限制（默认 1200 字符），再按时间正序拼成 `最近对话：...`。
- 一次 LLM 调用，结构化输出：

```python
class CorefRewrite(BaseModel):
    resolved_query: str
```

- Prompt 规则：
  - 把指代词（它/这个/那个/该订单/这单/刚才那单/上面的）补成历史里明确的对象；
  - 保留所有实体（订单号、商品名、型号、金额）原样，不引入历史没有的条件或事实；
  - 问题已完整、指代已明确时**原样返回**，不要为了“更标准”而改写；
  - 口语归一：邮费→运费、退钱→退款、灯不亮→故障/维修、换个新的→换货；
  - 只输出 JSON `{"resolved_query": "..."}`，不要解释或 Markdown。
- 失败/空输出：`resolved_query = user_message`，记 warning。
- 日志：`ch06 resolve_reference original=%r resolved=%r changed=%s`，供验收 3 看补全结果。
- 下游：`classify_intent` 与两条检索路都用 `resolved_query`。

### 7.2 `classify_intent`

- 输入：`resolved_query` + 最近几轮历史（用于多轮漂移：物流→退款→物流；包含 `[订单选择器]` 系统标记）。
- Prompt 四件套：
  1. **选择题**：八类逐条给出定义、该归/不该归边界：
     - 物流：包裹位置、配送进度、快递单号；不包括订单内容/退款政策。
     - 订单：订单状态、订单内容、取消订单；不包括物流轨迹、商品参数。
     - 商品咨询：商品参数、库存、价格、型号推荐；不包括已购买后的质量问题。
     - 退款退货：退款进度、退货流程、退款/退货政策、明确要求退钱/退货；不包括单纯质量维修。
     - 售后：商品质量、使用、维修、安装、补发等售后处理；不包括退款/退货流程本身。
     - 投诉：明确表达不满、要求追责、找主管、曝光；带情绪的退款诉求，如果核心是退款，仍归退款退货。
     - 闲聊：寒暄、无关闲聊、问机器人身份。
     - 其他：与购物/售后无关、语义不明、无法归入以上任何一类；拿不准就选它。
  2. **强制 JSON**：字段只有 `intent` 和 `confidence`（0–1），不要 `reason`、不要解释、不要 Markdown。
  3. **边界 few-shot**（每条都标明“为什么不是另一类”）：
     - 「这个还能退吗」→ 退款退货（不是商品咨询）。
     - 「刚收到就坏了，怎么保修」→ 售后（不是退款退货）。
     - 「我要投诉，顺便把钱退了」→ 投诉（核心是投诉；若只是「顺便退款」的抱怨但核心诉求是退款，则应归退款退货）。
     - 「你好，在吗」→ 闲聊（不是订单）。
     - 「帮我写一首诗」→ 其他（不是闲聊/商品咨询）。
  4. **其他兜底**：语义不明或无法归入业务类时选「其他」，不要硬塞。
- 结构化输出：

```python
class IntentResult(BaseModel):
    intent: Literal["物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "闲聊", "其他"]
    confidence: float = 0.0
```

- 解析失败/越界/异常/空输出 → `intent="其他"`, `confidence=0.0`，记 warning。**不再回退「售后」**。
- `confidence` 夹到 `[0,1]`；缺失/非法按 0.0。
- 模型：默认 `settings.intent_model or settings.model_name`，temperature=0.0。配置种子见 §9.3；本章不实现 cost 模式运行时。
- 日志：`ch06 classify_intent intent=%s confidence=%.2f`（不打印用户原文，避免日志污染）。

### 7.3 `route_by_intent`

- 普通节点：写入 `route`，再由条件边读取 `route`。这样日志、闸门、service 看到同一个值。
- 映射见 §6.1。

### 7.4 `script_reply`

- 替代 ch05 `chitchat_reply`；按 `intent` 选固定文案，不调模型、不检索、不给按钮：
  - 闲聊：`您好，我是客服小猫，主要帮您处理订单、物流和售后问题。请问有什么可以帮您的？`
  - 其他：`抱歉，我还没完全理解您的需求。请具体说明您要咨询的订单、物流、商品或售后问题，我再帮您处理。`
- 通过 `get_stream_writer` 发一个 `token` 帧；返回 `answer` + `AIMessage`，保证下一轮 history 不出现连续 `HumanMessage`。
- `suggested_actions=[]`。

### 7.5 `complaint_reply`

- 行为与 ch05 一致：固定安抚话术，`suggested_actions=[转人工, 建工单]`，后端不自动执行；路由键从 `complaint` 改为 `escalate`，节点名保留。

### 7.6 `retrieve_knowledge`（商品咨询）

- 只用于 `商品咨询`；`退款退货/售后` 走 `retrieve_policy`。
- 调用 `retrieve_hybrid(resolved_query, analysis=...)`：
  - `analysis = QueryAnalysis(original=user_message, rewritten=resolved_query, keywords=expand_synonyms(resolved_query), rewritten_by="llm")`。
  - 这样避免 ch04 `analyze` 再调一次 LLM 改写；同义词轻扩保留。
- 证据、sources、confidence、gate 判定与 ch05 相同：`order_for_prompt` + `CitationCollector` + `assess_confidence`。
- 异常 → `gate_reason="infra_error"`；`rerank_failed` → 保守兜底；不写低置信度池。
- 日志增加 `resolved_query`。

### 7.7 `fetch_order`（退款子流程第一站）

- 槽位提取（只看本轮 `resolved_query`，不扫描原始历史）：
  - 先看 `state["order_id"]`（resume 回填或同轮已写入）；
  - 否则从 `resolved_query` 用正则抽订单号：`订单(?:号)?[：: ]*([0-9]{4,})`、`([0-9]{4,})\s*号订单`，最后才退回 `\b([0-9]{4,})\b`；
  - 多个数字时优先“订单”附近的那个；不把金额/日期当订单号；
  - **不直接扫描 `messages` 历史**。历史里的订单号必须先由 `resolve_reference` 补进 `resolved_query`；否则宁可弹订单选择器，也不复用旧订单（反目标 #1）。
- 订单号校验：
  - `orders = list_user_orders(state["session_id"])`（只读 mock，返回 1001/1002/1003 三张卡）；
  - 抽到的 `order_id` 必须在 `orders` 的 order_id 集合里；不在（如 1004、年份数字）→ 视为“没有可用订单号”，走选择器，不调用 `query_order`；
  - 这样退款流程的 `order_data` 始终来自演示订单目录，含 `created_at` / `signed_at`，退款 prompt 可以依赖这些字段；`/ch06/refunds` 的校验也一致。
- 缺订单号/订单号无效：
  - `selected = interrupt({"kind": "select_order", "slot": "order_id", "orders": orders})`；
  - 校验 `selected` 在 `orders` 的 order_id 集合里；合法则写入 `order_id`，并追加一条 `SystemMessage(content="[订单选择器] 用户选择了订单 <id>")` 到 `messages`（见 §8.2）；非法则走兜底。
  - **`interrupt()` 之前只做只读操作**（正则、`list_user_orders`、日志），不查订单、不写消息；否则 resume 重跑会重复副作用。
- 取订单：`execute_tool(query_order, {"order_id": order_id}, timeout_s=..., retries=0)`，`json.loads` 结果写入 `order_data`。
- 失败：工具异常/JSON 解析失败 → `answer=ORDER_FAILED_TEXT`、`gate_reason="order_fetch_failed"`，发 token 帧并追加 `AIMessage`，条件边转 `record_log`，不进 `retrieve_policy`。
- 成功：写 `order_id`、`order_data`，条件边继续 `retrieve_policy`。
- 日志：`ch06 fetch_order order_id=... source=message|state|picker`（`state` 表示槽位已由本轮 state 提供且通过目录校验，`picker` 表示 resume 后的选择器回填）。

订单 mock（`app/tools/business.py`）：

```python
DEMO_ORDERS = {
    "1001": {"order_id": "1001", "items": [...], "amount": 299.0,
             "status": "已签收", "tracking_no": "SF...", "created_at": ..., "signed_at": ...},
    "1002": {... "status": "运输中", "signed_at": None ...},
    "1003": {... "status": "待发货", "signed_at": None ...},
}
```

- `query_order("1001"/"1002"/"1003")` 返回 `DEMO_ORDERS[id]` 的同一份数据；其他 ID 保持原有随机行为（兼容 ch02）。
- 退款流程的 `fetch_order` 只允许目录内 ID；目录外 ID 不会进入退款子流程，只可能出现在物流/订单业务路。
- `list_user_orders(session_id)` 返回三张卡（order_id、首个商品摘要、金额、状态、签收时间）；不落库。
- 日期按 `datetime.now()` 相对生成（如 1001 已签收 3 天），保证 7 天政策演示稳定。

### 7.8 `retrieve_policy`（退款子流程第二站）

- 输入：`resolved_query` + `order_data`。
- 扩写：`expand_queries(llm, resolved_query, order_data)`：
  - Prompt：电商退款/售后检索查询生成器；输出**严格 3 条**短查询，分别侧重量化条件/例外/流程；保留订单号、商品名、型号；不引入原问题没有的参数；只输出 JSON `{"queries": [...]}`。
  - 结构化输出：`class QueryExpansion(BaseModel): queries: list[str]`（扁平 list）。
  - 校验：去空、去重；数量不是 3 时记 warning，回退为 `[resolved_query]`（不硬凑重复 query）。
- 多 query 检索：
  - 每条 query 调 `retrieve_hybrid(query, analysis=QueryAnalysis(original=query, rewritten=query, keywords=expand_synonyms(query), rewritten_by="llm"))`，避免二次 LLM 改写。
  - 任一 query 抛异常 → `gate_reason="infra_error"`，整趟兜底，不进 Agent。
  - 任一 query `rerank_failed=True` → 整趟 `gate_reason="rerank_failed"`，保守兜底；不把 RRF 分和重排分混在一起。
  - 合并：按 `chunk_id` 去重，保留最高 `score`；按分降序取 `settings.rerank_top_n` 条。
- 证据：`order_for_prompt` + `CitationCollector`，写 `evidence/sources/confidence/gate_passed/gate_reason/expanded_queries`。
- 置信度：`assess_confidence(merged_chunks, settings.confidence_min_score, rerank_failed=False)`；Top-1 重排分低于阈值 → 闸失败，走知识缺口兜底，不进 Agent。
- 日志：`ch06 retrieve_policy order_id=%s queries=%s hits=%d top_score=%s`。

### 7.9 `confidence_gate`

- 沿用 ch05：`infra_error` / `rerank_failed` 短路；通过则发 `sources` 帧，进 `main_agent`；知识缺口发 `KNOWLEDGE_GAP_TEXT`，不进 Agent。
- 知识类和退款类共用；退款类失败时不给退款表单按钮。
- service 在 `gate_reason in {"empty","below_threshold"}` 时写低置信度池（知识缺口），`source="confidence_gate"`。

### 7.10 `main_agent`（父图节点，内部 ReAct 子图）

- 父图节点名从 `agent` 改为 `main_agent`；内部子图仍是 `agent ↔ tools → finalize`。
- 按 `route` 选 prompt 和工具：
  - `business` / `knowledge`：`render_agent_system_prompt(evidence)`；工具 `query_order/query_product/query_logistics/suggest_handoff`。
  - `refund_flow`：`render_after_sales_agent_system_prompt(order_data, evidence, intent)`；工具 `submit_refund/suggest_handoff`。
    - 一条通用售后判定 prompt，同时覆盖 `退款退货` 和 `售后`；Agent 按用户诉求作答，不强行把所有售后问题都判成退款。
    - 不绑定 `query_order`/`query_logistics`，避免 Agent 重新查订单拿到与 `order_data` 不一致的随机数据。
    - 不绑定 `query_faq`，政策证据由父图强制检索后注入。
- 售后 prompt 要点：
  - 角色：售后处理助手；先判断用户诉求是「退款/退货」还是「保修/维修/补发/使用问题」；
  - 订单事实（订单号、商品、金额、状态、签收时间）直接转述，不带 `[n]`；
  - 政策事实必须带 `[n]`，编号只能来自本轮证据；
  - 退款/退货诉求：判断这一单能不能退；能退 → 调 `submit_refund(order_id=...)`；不能退 → 说明原因，必要时 `suggest_handoff` 转人工；
  - 保修/维修/补发/使用问题：按政策证据说明流程和条件，必要时 `suggest_handoff`（ticket_type=售后），**不调用 `submit_refund`、不出退款表单**；
  - 不追问退款原因（表单负责）；不承诺一定退款；不编造政策。
- `submit_refund` 控制工具：
  ```python
  @tool
  def submit_refund(order_id: str) -> str:
      """准备退款申请表单；只记录意图，不提交、不写库。仅当用户诉求是退款/退货且政策允许时调用。"""
  ```
  - `tools` 节点识别到该工具：校验 `order_id == state["order_data"]["order_id"]`；合法则合并 `suggested_actions=[{"action":"refund_form","label":"提交退款工单","order_id":...}]`，回确认 `ToolMessage`；非法则回错误 `ToolMessage`，不写 State。
  - 不经过 `execute_tool`；与 `suggest_handoff` 同构。
- `suggest_handoff` 行为不变；`create_ticket` 仍不绑定给 Agent。
- 停止条件、步数/token 预算、`finalize` 沿用 ch05。

### 7.11 `record_log`

- 在 ch05 字段基础上增加：`resolved_query`、`intent_confidence`、`order_id`、`expanded_queries`、`refund_form`（是否建议了退款表单）。
- 中断路径不会经过 `record_log`；service 单独记 `ch06 interrupt` 日志。
- 不写数据库；数据库审计由 service 负责。

## 8. 退款子流程与 interrupt/resume 全链路

### 8.1 流程

```text
route=refund_flow
 → fetch_order
     order_id = state.order_id | 正则( resolved_query + history )
     缺 → interrupt({"kind":"select_order","slot":"order_id","orders":[...]})
        → 前端订单卡片 → POST /ch06/chat/resume {session_id, order_id}
        → Command(resume=order_id) → 同一节点重跑 → order_id 回填
     order_data = query_order(order_id)          # 只执行一次
 → retrieve_policy
     queries = expand_queries(resolved_query, order_data)   # 严格 3 条
     merged  = 多 query retrieve_hybrid → chunk_id 去重、max rerank 分
 → confidence_gate
     强 → main_agent（order_data + 政策证据；工具 submit_refund/suggest_handoff）
     弱/异常 → 兜底，不进 Agent
 → main_agent（通用售后判定）
     退款/退货：能退 → submit_refund → tools 节点拦截为 refund_form；不能退 → 说明原因，必要时 suggest_handoff
     保修/维修/补发：说明政策/流程，必要时 suggest_handoff（售后），不出退款表单
 → record_log
```

### 8.2 interrupt/resume 细节

- `interrupt()` 必须在 `fetch_order` 节点内；payload：

```json
{
  "kind": "select_order",
  "slot": "order_id",
  "orders": [
    {"order_id": "1001", "title": "无线蓝牙耳机 ×1", "amount": 299.0,
     "status": "已签收", "signed_at": "2026-11-15"}
  ]
}
```

- service 在 `graph.stream` 结束后读取：

```python
snapshot = graph.get_state(config)
if snapshot.interrupts:
    payload = snapshot.interrupts[0].value
    yield _dumps({"type": "interrupt", **payload})
    return   # 不写 assistant 答案、不发 options
```

- resume：

```python
from langgraph.types import Command
graph.stream(Command(resume=order_id), config,
             stream_mode="custom", subgraphs=True, version="v2")
```

- 节点重跑：`fetch_order` 在 `interrupt()` 前只做正则、`list_user_orders`、日志；`query_order` 在 `interrupt()` 之后，只跑一次。回归测试必须覆盖“缺单→interrupt→resume→query_order 只一次”。
- 非法 resume 值（不在订单列表）：`fetch_order` 返回 `answer=ORDER_INVALID_TEXT` + `AIMessage`，条件边转 `record_log`；前端显示错误并允许重试。
- 选择器回填的消息可见性：`fetch_order` 在 resume 分支（`interrupt()` 返回合法订单号）追加一条 `SystemMessage(content="[订单选择器] 用户选择了订单 <id>")` 到 `messages`。这条消息进入 checkpoint，后续 `resolve_reference` / `classify_intent` / `main_agent` 都能看到；MySQL 审计另写一条 user 消息 `[订单选择] 订单 <id>`。这样下一轮「这个能退吗」即使 assistant 答案没重复订单号，coref 也能从系统消息里补全。
- 并发/刷新：
  - `/ch06/chat` 在启动新回合前先 `graph.get_state(config)`；若线程有 pending interrupt，不写用户消息、不跑新回合，直接重新发 `interrupt` 帧并结束。这样刷新/新标签页/API 客户端发新消息时，用户会重新看到订单卡片，而不是把线程卡死。
  - `/ch06/chat/resume` 在流开始前检查 pending interrupt；没有则返回 409 `{"detail":"当前没有待选择的订单"}`，不调用 `Command(resume=...)`。
  - 前端在订单选择器打开期间禁用输入框；点选或失败后恢复。

### 8.3 退款表单与工单

- Agent 在退款/退货诉求下只建议 `refund_form` 动作；真正的退款单由用户在前端提交。保修/维修/补发诉求不出这个动作。
- 固定退款原因（前后端同一份）：

```python
REFUND_REASONS = ("七天无理由", "质量问题", "发错货", "不想要了", "其他")
```

- `POST /ch06/refunds` 请求：

```json
{"session_id": "...", "order_id": "1001", "reason": "质量问题"}
```

- 校验：`reason` 必须在 `REFUND_REASONS`；`order_id` 必须在 `list_user_orders(session_id)` 里。
- 落库：复用 `create_ticket`，`ticket_type="退款"`，`description=f"订单 {order_id} 退款申请，原因：{reason}"`，`timeout_s=5`、`retries=0`（非幂等，失败返回 502 让用户手动重试）。
- 成功返回 `{"ticket_no": "T...", "message": "退款单已创建"}`；前端显示退款单号。
- `tickets.ticket_type` 枚举增加 `退款`：
  - `app/models/ticket.py`：`Enum("售后","投诉","咨询","退款", name="ticket_type")`；
  - `app/tools/ticket.py`：`ticket_type: Literal["售后","投诉","咨询","退款"]`；
  - `app/schemas.py`：`TicketRequest.ticket_type` 同步加 `退款`；
  - `sql/ch06_ticket_type.sql`：MySQL `ALTER TABLE tickets MODIFY COLUMN ticket_type ENUM('售后','投诉','咨询','退款') NOT NULL;`
  - 旧数据不受影响；SQLite 测试从模型建表，不需迁移。

## 9. State 与配置

### 9.1 `CustomerServiceState`

```python
Intent = Literal["物流","订单","商品咨询","退款退货","售后","投诉","闲聊","其他"]
Route = Literal["escalate","fallback_script","knowledge","refund_flow","business"]

class CustomerServiceState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    user_message: str
    resolved_query: str          # 替换 ch05 resolved_message
    intent: str
    intent_confidence: float
    route: str
    session_id: str              # 订单 mock / 审计
    order_id: str
    order_data: dict
    expanded_queries: list[str]
    evidence: list[dict]
    sources: list[dict]
    confidence: float | None
    gate_passed: bool
    gate_reason: str
    rerank_failed: bool
    answer: str
    suggested_actions: list[dict]
    step_count: int
    token_estimate: int
    token_budget: int
    max_steps: int
    conversation_id: int
    exhausted: bool
```

- `AgentState` 增加 `order_data`（prompt 用）与既有 `route`；其余共享键不变。
- 每轮 `_reset_inputs` 显式重置：`resolved_query=""`、`intent_confidence=None`、`order_id=""`、`order_data=None`、`expanded_queries=[]`、`evidence/sources/answer/suggested_actions/...`。`session_id` 每轮写入。`order_edge` 以 `order_data is None` 判断取单失败，因此 `order_data` 必须重置为 `None` 而不是 `{}`（`intent_confidence` 同样使用 `None` 哨兵）。
- `messages` 只追加新的 `HumanMessage`；checkpointer 保留历史。

### 9.2 配置

```python
# app/config.py
intent_model: str = ""                 # 空 -> model_name
intent_small_model: str = ""           # 仅配置种子
intent_mode: Literal["accuracy","cost"] = "accuracy"
intent_conf_threshold: float = 0.7     # 仅配置种子
ch06_checkpoint_db: str = ".data/ch06-checkpoints.sqlite"
```

- `build_llm(settings, temperature=0.7, model: str | None = None)`：`model` 为空时用 `settings.model_name`。
- 聊天模型：`build_llm(settings)`；意图/核心指代/扩写模型：`build_llm(settings, temperature=0.0, model=settings.intent_model or None)`。
- **不实现** `intent_mode="cost"` 的小模型→大模型升级；只保留字段和 spec 说明。运行时即使配置为 cost 也按 accuracy 处理，并记 warning。

## 10. API 与 SSE 契约

### 10.1 端点

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 服务 `app/static/ch06.html`（新聊天页） |
| GET | `/ch06` | 同上 |
| GET | `/ch05` | 307 → `/ch06`（兼容旧链接） |
| POST | `/ch06/chat` | SSE；请求 `{session_id, message}` |
| POST | `/ch06/chat/resume` | SSE；请求 `{session_id, order_id}`，`Command(resume=...)` 续跑 |
| POST | `/ch06/refunds` | JSON；请求 `{session_id, order_id, reason}`，写 tickets |
| POST | `/ch05/chat` | 307 → `/ch06/chat`（兼容） |
| POST | `/ch05/tickets` | 保留旧建单端点；`ticket_type` 同步支持「退款」 |

- 不新增 `/api/agent` 非流式端点；eval 直接 `build_graph(...).invoke(...)`。
- 旧 `/ch05` 页面文件不再挂载；`app/static/ch05.html` 可删除或保留但不服务（推荐删除，避免误用）。
- `app/api/ch05.py` 缩减为兼容层：`/ch05` 和 `/ch05/chat` 重定向，`/ch05/tickets` 保留；ch05 的 graph 依赖/lifespan 关闭逻辑移除。
- `app/main.py` 根路由改为 `ch06.html`；注册 ch06 路由；lifespan 关闭 ch06 checkpointer。

### 10.2 SSE 帧

| 帧 | 内容 |
|---|---|
| token | 裸文本（直接拼接） |
| tool | `{"type":"tool","tool":"query_logistics","label":"查物流"}` |
| sources | `{"type":"sources","sources":[...]}` |
| options | `{"type":"options","options":[{"action":"transfer","label":"转人工"},{"action":"ticket","label":"建工单","ticket_type":"投诉"},{"action":"refund_form","label":"提交退款工单","order_id":"1001"}]}` |
| interrupt | `{"type":"interrupt","kind":"select_order","slot":"order_id","orders":[...]}` |
| error | `{"error":"..."}` |
| 结束 | `[DONE]` |

- `interrupt` 帧发出后，service 不再发 `options`、不写 assistant 答案；API 仍以 `[DONE]` 结束。
- `options` 在 graph 结束后从最终 State 的 `suggested_actions` 读取，保证在答案 token 之后。
- `sources` 只在闸通过后、进入 `main_agent` 前发出。

### 10.3 图构建与 checkpointer

- 沿用 ch05 懒加载模式：`app.state.ch06_graph`、`ch06_checkpoint_handle`、`ch06_graph_lock`；首次 `/ch06/*` 请求构建，FastAPI shutdown 关闭。
- `thread_id = f"user:{session_id}"`；使用独立的 `CH06_CHECKPOINT_DB`，避免与 ch05 旧图状态/结构冲突。
- `build_graph(llm, intent_llm, *, checkpointer=None, settings=None)` 不变；测试可注入 `InMemorySaver` + fake LLM。
- API 依赖 `get_ch06_graph_dep` 可被 `dependency_overrides` 替换；测试不碰真实 SQLite/模型。

## 11. 前端配套（Vibe Coding）

页面 `app/static/ch06.html`，基于 ch05 页面升级，自包含 CSS/JS。**前端不套 brainstorm/TDD/code review；用户描述效果，我直接改。**

- 会话 key：`lamp-cs-ch06-session-id`（与 ch05 隔离）。
- 订单选择器：
  - 收到 `interrupt`（`kind=select_order`）帧 → 在聊天流里渲染可点选订单卡片（订单号、商品摘要、金额、状态、签收时间）；
  - 点选 → 禁用卡片，追加系统消息「已选择订单 1001」→ `POST /ch06/chat/resume`，把续流渲染到新的助手气泡；
  - 等待选择期间禁用输入框；失败恢复卡片可重试。
- 退款表单：
  - 收到 `options` 里的 `refund_form` 动作 → 渲染「提交退款工单」按钮；
  - 点开表单：订单号只读 + 退款原因下拉（七天无理由/质量问题/发错货/不想要了/其他）+ 提交/取消；
  - 提交 → `POST /ch06/refunds` → 显示退款单号；成功禁用表单，失败可重试。
- 保留 ch05 的转人工/建工单两个独立按钮、90 秒无数据超时、错误帧处理。
- `tool` 帧显示状态行；`sources` 帧显示「已检索知识库」。

## 12. 文件变更清单

### 新增

- `app/agent/coref.py` — `CorefRewrite`、prompt、`resolve_query(llm, message, history)`
- `app/api/ch06.py` — `/ch06` 页面、`/ch06/chat`、`/ch06/chat/resume`、`/ch06/refunds`、懒加载 graph
- `app/static/ch06.html` — 新聊天页
- `sql/ch06_ticket_type.sql` — tickets 枚举加 `退款`
- `tests/test_interrupt_smoke.py` — interrupt/resume 红线冒烟
- `tests/test_agent_coref.py`
- `tests/test_agent_expand.py`
- `tests/test_agent_refund_flow.py`
- `tests/test_ch06_api.py`
- `tests/eval/ch06_coref_cases.json`
- `tests/eval/ch06_intent_cases.json`
- `tests/eval/ch06_expand_cases.json`
- `app/jobs/eval_coref.py`
- `app/jobs/eval_expand.py`
- `app/jobs/eval_ch06.py`

### 修改

- `app/agent/state.py` — Intent/Route 扩类、State 字段
- `app/agent/intent.py` — 四件套 prompt、`{intent,confidence}`、历史、其他兜底
- `app/agent/routing.py` — 八类→五出口、`tools_for_route`
- `app/agent/graph.py` — `resolve_reference` 做实、新节点、五出口、退款子流程、gate 路由 `main_agent`
- `app/agent/subgraph.py` — 按 route 选 prompt/工具、`submit_refund` 拦截、退款 prompt
- `app/agent/tools.py` — `submit_refund`、工具子集
- `app/agent/service.py` — 重置字段、interrupt 帧、`resume_chat_agent`、审计
- `app/rag/query_understanding.py` — `expand_queries` + prompt/schema
- `app/prompts/customer_service.py` — `render_after_sales_agent_system_prompt`
- `app/tools/business.py` — DEMO_ORDERS、`list_user_orders`、`query_order` 固定演示订单、日期字段
- `app/tools/ticket.py` — `ticket_type` 加 `退款`
- `app/models/ticket.py` — 枚举加 `退款`
- `app/schemas.py` — `ResumeRequest`、`RefundRequest`、`TicketRequest` 加 `退款`
- `app/config.py` — 意图模型/模式/阈值、ch06 checkpointer
- `app/main.py` — 根页面改 ch06、注册路由、lifespan
- `app/api/ch05.py` — 改为兼容层（`/ch05`、`/ch05/chat` 重定向；保留 `/ch05/tickets`）
- `tests/test_agent_intent.py`、`test_agent_routing.py`、`test_agent_graph.py`、`test_agent_subgraph.py`、`test_agent_service.py`、`test_agent_tools.py` — 同步
- `tests/test_intent_cases.py` — 八类、held-out、多轮 schema
- `tests/test_tools_business.py`、`test_tools_ticket.py`、`test_models.py`、`test_schemas.py` — 同步
- `tests/test_ch05_api.py`、`test_ch05_page.py` — 兼容层/页面同步
- `app/jobs/eval_intent.py` — 八类、confidence、history
- `dev-notes/ch06.md` — 分阶段追记

### 删除/替换

- `app/static/ch05.html` — 由 `ch06.html` 替换（文件可删；旧链接由 `/ch05` 重定向兼容）。

## 13. 测试与验证策略

### 13.1 红线冒烟（先于实现）

`tests/test_interrupt_smoke.py`：用 `InMemorySaver` + 最小图，验证：
1. `interrupt()` 后 `graph.stream(..., stream_mode="custom")` 正常结束；
2. `graph.get_state(config).interrupts[0].value` 是 payload；
3. `Command(resume=...)` 能续跑；被中断节点从头重跑，`interrupt()` 返回 resume 值；
4. `interrupt()` 前代码执行两次、之后代码执行一次（用计数器钉死）。
若与预审不符，先停，按选型内解决或问用户。

### 13.2 TDD 单元测试（无真实网络）

- `test_agent_coref.py`：prompt 规则、合法解析、异常/空输出回退原文、history 渲染、日志。
- `test_agent_intent.py`：八类齐全、四件套关键文案、`{intent,confidence}` 解析、confidence 夹取、异常/越界→其他、history 传入、prompt 不含 `reason`。
- `test_agent_routing.py`：八类逐一映射五出口；其他→`fallback_script`；未知→`business`。
- `test_agent_expand.py`：合法 3 条、数量不对回退单 query、异常回退、prompt 只要求 `queries`。
- `test_agent_refund_flow.py`：
  - 有订单号 → `query_order` 一次、order_data 写入；
  - 目录外订单号（1004/年份）→ 弹选择器，不调 `query_order`；
  - 缺订单号 → interrupt payload 含三张卡；resume 回填；`query_order` 只一次；interrupt 前代码只读；回填后 `messages` 含 `[订单选择器]` SystemMessage；
  - 非法 resume → 兜底；
  - `retrieve_policy` 调 `expand_queries` + 3 次 `retrieve_hybrid`（带 `analysis` 跳过二次改写）；去重/取最高分；闸门；
  - 退款/售后 prompt 含 order_data + 证据；工具只有 `submit_refund/suggest_handoff`；
  - 退款诉求可出 `refund_form`；保修/维修/补发诉求不调用 `submit_refund`、不出 `refund_form`；
  - `submit_refund` 拦截为 `refund_form`，非法 order_id 回错误 ToolMessage，不调 `execute_tool`。
- `test_agent_graph.py`：商品咨询/退款/售后/物流/订单/投诉/闲聊/其他八条路径；闸失败不进 Agent；固定话术不调 LLM；日志字段；跨轮状态重置。
- `test_agent_service.py`：interrupt 帧、不写 assistant 答案；resume 用 `Command`；options 位置；审计；低置信度池；pending interrupt 时 `/ch06/chat` 重发 interrupt；无 pending 时 `/ch06/chat/resume` 返回 409。
- `test_ch06_api.py`：页面 200；`/ch06/chat` SSE；`/ch06/chat/resume` 无 pending 返回 409；`/ch06/refunds` 写 tickets 且 `ticket_type='退款'`；非法 reason 422；失败 502 且 `retries=0`；懒加载 build/close；`/ch05` 重定向。
- `test_tools_business.py`：演示订单确定性、列表与 query_order 一致、未知 ID 仍随机、日期字段。
- `test_models.py` / `test_tools_ticket.py` / `test_schemas.py`：`退款` 枚举/校验。

### 13.3 Prompt/数据类验证（替代 TDD 的部分）

- `tests/eval/ch06_coref_cases.json`：多轮指代、订单/商品引用、已完整透传、无历史；每条带 `history`、`message`、`expect.contains` / `expect.not_contains` / `passthrough`。脚本 `app/jobs/eval_coref.py` 用真实 LLM 跑，输出通过率与错例。
- `tests/eval/ch06_intent_cases.json`：八类各 6 条 + 边界/其他/多轮漂移；held-out（不得复用 prompt few-shot 原文）；脚本 `app/jobs/eval_intent.py` 输出总体/每类准确率、混淆矩阵、有效输出率、fallback 列表。
- `tests/eval/ch06_expand_cases.json`：退款/售后核心场景；脚本 `app/jobs/eval_expand.py` 检查严格 3 条、JSON 可解析、关键实体保留。
- 通过标准：coref 关键实体/去指代 ≥90%，透传用例 100%（或至少 90%，以评估报告为准）；intent 总体 ≥90%、每类 ≥80%、JSON 解析率 100%（fallback 视为失败）；expand 3 条率 ≥90%、JSON 解析率 100%。
- 报告写入 `docs/superpowers/reports/2026-11-18-ch06-*.md`，并把结论写进 dev-notes。

### 13.4 端到端验收

1. **多轮漂移**：`app/jobs/eval_ch06.py` 跑「物流→退款→物流」三轮（真实 LLM + 真实检索/工具），每轮记录 `resolved_query`、`intent`、`route`、`order_id`、`expanded_queries`、`gate`、`answer`；要求意图全对、指代补全正确。
2. **意图 JSON/其他**：`eval_intent.py` 覆盖怪问题落「其他」、JSON 稳定可解析。
3. **「这个能退吗」**：先有订单 1001 的物流上下文，再问「这个能退吗」；日志出现 `resolved_query` 含 1001、`route=refund_flow`、`fetch_order order_id=1001`、`expanded_queries` 3 条、政策证据；dev-notes 贴实际日志。
4. **浏览器**：`http://localhost:8000/` 新页面；不带订单号问退款 → 订单卡片；点选 1001 → 续流 → 政策/判断 → 「提交退款工单」→ 选原因 → 提交 → 显示退款单号；用 Playwright（Edge headless）截图并写 dev-notes。
5. 全量 `pytest -q` 通过；旧 `/chat`、`/ch05/tickets` 不破坏。

### 13.5 质量门

- 全量 pytest 通过。
- 代码走 `superpowers:requesting-code-review`；修复后 scoped re-review。
- 前端 Vibe Coding 不套 review，但浏览器验收必须真跑。
- 评估报告与 dev-notes 分阶段追记，不一次性补。

## 14. 风险与开放项

1. **R1 interrupt surface**：已在 1.2.20 源码确认 API，但仍先用 smoke test 钉死流式/恢复形状；不符则停。
2. **R2 resume 重跑**：`fetch_order` interrupt 前只读；回归测试钉死 `query_order` 只一次。
3. **R3 LLM 输出稳定性**：coref/intent/expand 都依赖真实模型；用 held-out 评估集验证，解析失败走安全兜底；不把 fallback 算作正确。
4. **R4 tickets 枚举迁移**：MySQL 存量数据用 `MODIFY COLUMN` 加值；SQLite 测试从模型建表；旧行不变。
5. **R5 多 query 成本/延迟**：退款/售后每轮 3 次检索 + 1 次扩写；可接受；若延迟明显，后续再谈缓存/并行，不在本章换方案。
6. **R6 退款闸门**：退款类也过置信度闸；政策证据弱时不进 Agent、不出退款表单，走人工兜底。参考 spec 未显式画闸，本章按 ch05 D8 保留。
7. **R7 订单 mock 与真实订单**：1001/1002/1003 是演示数据；接真实订单接口不在本章范围。
8. **R8 页面刷新/并发**：订单选择器是 pending interrupt；刷新后前端不自动恢复聊天记录，但用户再发消息时 `/ch06/chat` 会检测 pending interrupt 并重发订单卡片，不会卡死线程；无 pending 的 `/ch06/chat/resume` 返回 409。前端在等待选择期间禁用输入框。
9. **「其他」口径变更**：用户选 B，其他走固定文案，不再由主力 Agent 澄清；与最初需求第 6 条不同，已记录并将在 dev-notes 说明。

## 15. 验收映射

| 验收 | 证据 |
|---|---|
| 1. 多轮物流→退款→物流，意图/指代全对 | `eval_ch06.py` 三轮 trace + `eval_coref.py` / `eval_intent.py` 报告 |
| 2. 意图 JSON 稳定、怪问题落其他 | `eval_intent.py` 解析率/混淆矩阵/其他类用例 |
| 3. 「这个能退吗」先补指代、再走退款子流程 | trace 日志：`resolve_reference` original/resolved、`route=refund_flow`、`fetch_order`、`expanded_queries`、政策证据 |
| 4. 浏览器订单选择器 → 点选 → 子流程走完 | Playwright 截图 + `/ch06/chat/resume` 日志 + 退款表单/退款单号 |

## 16. 交付物

- 本设计文档
- 实施计划：`docs/superpowers/plans/2026-11-18-ecommerce-cs-ch06-dispatcher.md`
- 开发记录：`dev-notes/ch06.md`（分阶段追记）
- 评估报告：`docs/superpowers/reports/2026-11-18-ch06-coref-eval.md`、`...-intent-eval.md`、`...-expand-eval.md`
- 演示命令：
  - 启动：`.venv/Scripts/python.exe run.py`
  - 页面：`http://localhost:8000/`
  - 测试：`.venv/Scripts/python.exe -m pytest -q`
  - 评估：`.venv/Scripts/python.exe -m app.jobs.eval_coref`、`... eval_intent`、`... eval_expand`、`... eval_ch06`
