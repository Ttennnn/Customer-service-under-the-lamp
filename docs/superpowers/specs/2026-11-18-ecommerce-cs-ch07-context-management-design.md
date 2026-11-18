# 第七章 · 会话上下文管理(三层 + 异步分段摘要)— 设计文档

- 日期：2026-11-18
- 章节：ch07
- 状态：已定稿（2026-11-18 用户确认）
- 关联：ch02 `conversations` / `messages`、ch05 ReAct Agent、ch06 父工作流与指代消解

## 1. 意图与为什么

ch05/ch06 的 `main_agent` 把 checkpoint 里的**全量历史**原样递给模型：system 拼接
本轮证据，`state["messages"]` 整条接在后面。轮数一多 token 必爆，且 system 每轮
随证据/订单变化，前缀缓存整段失效。ch02 遗留的 `trim_history` 只是按 token 从
尾部截断，既不保早期事实，也没有摘要。

本章把“简单裁剪”升级成 **当前会话内的三层上下文管理**：

- **层 1（L1）**：离当前最近的几轮，保留原文，一个字都不压；
- **层 2（L2）**：中间层，按规则截短——用户原话不动、客服答复只留开头几十字、
  大块工具结果换成一行标识；
- **层 3（L3）**：更早的轮次，后台异步压成一段段梗概，追加进分段摘要表。

两个数据库 id 锚点只移动边界，不搬消息数据。摘要按 **L2 的 token 用量**触发，
不按消息条数——工具结果压根不落 `messages` 表，数条数看不见它涨。摘要一段段
追加、压完不回头重写，旧梗概只作背景给模型看、不参与合并，避免同一个事实被
反复有损压缩。

**只管当前会话，不跨会话、不存用户画像。**

## 2. 目标、非目标与反目标

### 2.1 目标

1. 三层结构 + 两个锚点：`summary_upto_msg_id` 标出摘要覆盖到哪条，
   `layer1_from_msg_id` 标出 L1 边界；降级只挪 `layer1_from_msg_id`，摘要完成只挪
   `summary_upto_msg_id`，不搬、不改 `messages` 行。
2. 摘要后台异步：L2 超预算就在后台起一次摘要任务，把这一批压成**新的一段**追加
   进 `conversation_summaries`，边界追到 L1 起点；不阻塞用户当前这轮回复。
3. 调模型的上下文按固定顺序拼装：system 人设/红线/工具定义打头（每轮一样）→
   L2 → L1 原文 → 用户当前这句 → 早期梗概 + 检索证据合成一条挂在用户这句之后。
   梗概绝不单独占一条 system。
4. token 预算从模型窗口倒推，不写死成常量；L1 拿七成、L2 拿三成；启动时自检，
   连一轮都装不下就报警。
5. 对话历史落 `conversations` / `messages` 两表；LangGraph State 的 `messages`
   挂 `add_messages`，checkpoint 留全量历史；调模型前另整理一份裁过、带摘要的
   精简版递进去，双轨互不影响。
6. 上下文可观测：每轮调模型前打 `model_ctx`；指代消解/意图识别共用的历史打
   `history_ctx`（每轮必打）；后台摘要触发/开始/完成/跳过/失败全留痕。
7. 前端多会话页面：侧栏列出该用户全部会话（新在前、首问预览、已摘要标记），
   点击切换并回载历史原文继续聊；「新对话」只是开新会话；配两个只读接口。

### 2.2 非目标（本章不做）

- 语义检索捞历史。
- 主题重要度筛选、关键事实抽取。
- 跨会话长期记忆、用户画像。
- 摘要的滚动重写/多版本合并。
- 前端保留任何跨会话缓存或画像。

### 2.3 反目标（即使跑通也算失败）

- 缩写后的梗概单独占一条 system，导致上游模板把所有 system 上提、工具定义被挤到
  可变内容之后、前缀缓存整段作废。
- 后台摘要阻塞当轮回复，或摘要失败影响用户链路。
- 装得下仍然压缩；或按消息条数而不是 token 用量触发摘要。
- 摘要里出现对话中没有的订单号、手机号、商品、诉求（编造）。
- 旧摘要回炉重写，同一事实被反复有损压缩。
- 三层边界靠复制/删除消息实现，`messages` 表被改写。
- State 的完整历史被裁剪或删除。

## 3. 固定选型与已核实事实

### 3.1 固定选型

- LangGraph `StateGraph` + `add_messages` reducer + SQLite checkpointer。
- LangChain `trim_messages` 作为 L1 的 token 兜底。
- MySQL `conversations` / `messages`，新增 `conversation_summaries`。
- 现有 FastAPI、SQLAlchemy、DeepSeek OpenAI 协议、原生 HTML/CSS/JS 前端。

### 3.2 Context7 / 本地已核实

- `add_messages` 按消息 id 追踪：同 id 更新覆盖，新 id 追加。因此给用户消息打
  `id=f"db-{msg_id}"` 不会撞车（MySQL 自增主键唯一），也不会重复追加。
- `graph.get_state(config)` 必须给 `thread_id`，返回最新快照；新输入不会清空
  thread，要隔离会话必须换 `thread_id`。本章因此用 `conv:{conversation_id}`。
- `Command(resume=...)` 用同一个 `thread_id` 恢复中断；被中断节点会从头重跑。
- `trim_messages(strategy="last", token_counter=..., max_tokens=..., start_on="human")`
  在调模型前裁剪、不改 State；本章只把它当 L1 的最后一道 token 兜底。
- 现有 `app/services/context.py` 的 `estimate_tokens` 是 1 字符 = 1 token 的保守
  口径；本章所有预算常数都按这个口径校准，改口径必须同时重标常数。

### 3.3 现有系统现状（ch06 基线）

- `thread_id = user:{session_id}`；`get_or_create_conversation` 取该用户最近一条
  “进行中”会话。多会话侧栏必须改成按 conversation 建 thread。
- `main_agent` 的 system 里拼了 evidence、订单数据、意图，与“system 每轮一样”
  冲突；要搬到用户这句之后的本轮材料。
- `fetch_order` 在 resume 时追加一条 `[订单选择器]` `SystemMessage`；它也是可变
  system，会和摘要一样被上游模板上提，必须改成非 system 的本轮材料。
- 工具结果只存在 State / checkpoint，不落 `messages` 表；摘要任务读
  user/assistant 原文即可，触发用量要看 State 渲染结果。
- `MAX_CONTEXT_TOKENS`（默认 4096）和 `AGENT_TOKEN_BUDGET`（默认 12000）现在
  仍在 gate：subgraph 会把整条 prompt 再裁到 4096，`agent_node` 在累计
  prompt token ≥ 12000 时直接进兜底。两者都会让窗口倒推的预算失效，ch07 必须
  退役它们（见 §6.1.1）。
- SSE 端点是同步生成器，跑在线程池；没有事件循环，后台任务不能用
  `asyncio.create_task`。
- 前端当前是自包含的 `app/static/ch06.html`，`localStorage/sessionStorage` 只记
  一个 session id；“新对话”会覆盖旧会话。

### 3.4 表结构（用户已给定，原文落地）

`sql/ch07-ddl.sql`：

- `conversations.summary TEXT NULL`：最近几段梗概拼成的投影，拼装时跟证据一起
  挂在用户这句之后。
- `conversations.summary_upto_msg_id BIGINT UNSIGNED NULL`：摘要已覆盖到哪条。

`sql/ch07-layers.sql`：

- `conversations.layer1_from_msg_id BIGINT UNSIGNED NULL`：L1 边界。
- `conversation_summaries(id, conversation_id, seq, from_msg_id, upto_msg_id,
  content, created_at)`，`UNIQUE(conversation_id, seq)`，
  `KEY(conversation_id, upto_msg_id)`，只追加。

ORM 同步三列和 `ConversationSummary`；DDL 无外键，ORM 不加外键；测试库 SQLite
用 `BigInteger().with_variant(Integer, "sqlite")` 保证自增。

## 4. 方案对比与选择

### 4.1 窗口拼装位置

**方案 A（已选）runtime 预处理。** 每轮入口读一次 checkpoint State，用
`db-{msg_id}` 标在用户 `HumanMessage` 上，按两个锚点切出 L2/L1，把整理好的
`prompt_history` 放进 State；coref、意图识别、`main_agent` 都消费这一份。State 的
`messages` 仍由 `add_messages` 全量保留。边界只落在用户消息上，assistant/tool 消息
不需要 id。checkpoint 缺失时用 MySQL 行兜底重建窗口。

**方案 B MySQL 重建整个窗口。** id 最全，但 tool 结果不落库，L1 的原始 tool 结果
和 L2 的用量都看不见，违背“触发看用量”的理由。不选。

**方案 C 每个节点各自按锚点切 State。** 少一个 State 字段，但 coref/agent/日志三处
重复切窗，降级边界更新没有单一入口。不选。

方案 A 让“完整历史”和“精简版”各有唯一入口：完整历史是 checkpoint 的
`messages`，精简版是 runtime 产出的 `prompt_history`。

### 4.2 多会话与 thread

**按 conversation 建 thread（已选）：** `thread_id = conv:{cid}`。聊天请求加可选
`conversation_id` / `new_conversation`；后端先解析或创建会话，再检查该 thread 的
pending interrupt。SSE 开头先回一帧 `{"type":"conversation","conversation_id":cid}`，
前端据此选中新会话。旧客户端不带 `conversation_id` 时保持兼容：取该用户最近一条
“进行中”会话。

不选“继续用 user thread + 每轮从 DB 重置 State”：会把不同会话的 checkpoint 混在
一起，且要频繁 `RemoveMessage`，与“checkpoint 全量保留”冲突。

### 4.3 后台摘要执行

**线程池 + 每会话在飞标记（已选）：** 模块级 `ThreadPoolExecutor`，提交即返回；
任务自己开 DB session。同步 SSE 端点没有事件循环，不能用 `asyncio.create_task`。
每会话一个在飞标记，防重复触发；服务关闭时 `shutdown(wait=False)`。

## 5. 数据模型与边界语义

### 5.1 锚点定义（与 DDL 注释一致）

```
id ≤ summary_upto_msg_id                 已进 L3（摘要覆盖）
summary_upto_msg_id < id ≤ layer1_from   L2（半压形态）
id > layer1_from                         L1（原文）
```

- `layer1_from_msg_id` 是 **L2 的最后一条消息 id**（通常是某轮客服答复）；
  L1 从它的下一条开始。这样 DDL 的 `>` 条件成立。
- `summary_upto_msg_id` 是 **L3 的最后一条消息 id**；L2 从它的下一条开始。
- 两个锚点都只在**用户轮边界**之间移动：L2 起点和 L1 起点都是用户消息；锚点本身
  可以指向该用户消息前一条 assistant 消息。这样 State 切窗只需要给用户消息打
  `db-` id。
- 锚点为 NULL：`summary_upto=NULL` 表示还没有摘要；`layer1_from=NULL` 表示全部历史
  都是 L1。

### 5.2 分段摘要表

- 一段一行，只追加；`seq` 从 1 开始，`(conversation_id, seq)` 唯一。
- `from_msg_id` / `upto_msg_id` 是这段覆盖的闭区间。
- `content` 只包含这一段新消息提炼出的事实与诉求，不回写旧段。
- `conversations.summary` 是投影：把各段按 `seq` 顺序拼起来，供拼装时整体注入。
  新段追加后只做 `旧投影 + "\n\n" + 新段`，不重新摘要旧段。

### 5.3 不搬数据

- 降级只 `UPDATE conversations.layer1_from_msg_id`。
- 摘要完成只 `INSERT conversation_summaries` + `UPDATE conversations.summary,
  summary_upto_msg_id`。
- 不 UPDATE/DELETE `messages` 行；不改写旧摘要行。

## 6. Token 预算

### 6.1 公式

```python
safety_margin = ceil(window * safety_margin_ratio)          # 默认 10%
fixed = (
    system_prompt_reserve_tokens                          # 1200
    + effective_rerank_top_k * evidence_tokens_per_chunk  # 250/条
    + max(summary_reserve_tokens, actual_summary_tokens)  # 下限 500
    + max_output_tokens                                   # 2000
    + safety_margin
)
transient = (
    max_user_input_tokens                                 # 2000
    + effective_max_agent_steps * tool_result_max_tokens  # ReAct 瞬时峰值
)
window_surplus = window - fixed - transient
target = history_target_turns * history_tokens_per_turn    # 30 × 300
history_budget = max(0, min(target, window_surplus))
layer1_budget = history_budget * 7 // 10
layer2_budget = history_budget - layer1_budget
```

- `actual_summary_tokens` 是当前 `conversations.summary` 投影的估算值；没有摘要时
  取 0，但固定开销仍按下限 500 计，保证验收数字可复现。投影变大时历史预算相应
  变小；摘要比原始历史更值钱，不能为了保住历史把摘要丢掉。
- `effective_*` 是新配置对旧配置的兼容取值：`max_agent_steps` 未设置时回落
  `agent_max_steps`，`tool_result_max_tokens` 未设置时回落
  `agent_tool_result_max_chars`，`rerank_top_k` 未设置时回落 `rerank_top_n`。
- 预算只覆盖**历史**；当前轮的 user 输入和 ReAct 工具结果由 `transient` 预留，
  不算进 L1/L2。

### 6.1.1 旧预算开关退役

- `MAX_CONTEXT_TOKENS` 不再对 ch07 拼好的 prompt 做二次裁剪。最终 prompt 的上界
  由 §6.1 的窗口预算 + safety margin 保证；组装后只做“超出则 warning”的检查。
- `AGENT_TOKEN_BUDGET` 不再作为 ReAct 的耗尽条件。步数上限仍由
  `MAX_AGENT_STEPS` 控制；ReAct 瞬时峰值由 `transient` 预留。`token_estimate`
  仍累计并写日志，只作观测。
- 两个旧配置保留给 `/chat` 与 `raw_demo` 的旧链路，不传给 ch07 的
  `build_agent_subgraph`。
- 最终 prompt 若仍超过 `window - safety`：记 warning，只用 `trim_messages` 裁
  L1 尾部；不裁摘要、证据和当前用户句。正常路径不应触发。

### 6.2 验收数字

演示配置：

```
MODEL_CONTEXT_WINDOW=18000
MAX_OUTPUT_TOKENS=2000
MAX_USER_INPUT_TOKENS=2000
MAX_AGENT_STEPS=3
TOOL_RESULT_MAX_TOKENS=1200
RERANK_TOP_K=5
```

计算：

```
fixed     = 1200 + 5*250 + 500 + 2000 + 1800 = 6750
transient = 2000 + 3*1200 = 5600
surplus   = 18000 - 6750 - 5600 = 5650
target    = 30*300 = 9000
history   = min(9000, 5650) = 5650
layer1    = 5650*7//10 = 3955
layer2    = 1695
```

用户写的是 L1=3954；这是 70% 向下取整差 1 token，L1+L2 仍应等于 5650。实现按
3955/1695，日志同时打印 `history=5650 layer1=3955 layer2=1695`。

只改 `MODEL_CONTEXT_WINDOW=18000`、其余保持默认：

```
fixed     = 1200 + 10*250 + 500 + 2000 + 1800 = 8000
transient = 2000 + 6*2000 = 14000
surplus   = 18000 - 8000 - 14000 = -4000
history   = 0
```

启动自检必须报警 `上下文预算不足`，滑窗归零。

### 6.3 默认窗口与二十轮

`MODEL_CONTEXT_WINDOW` 默认 65536。默认参数下：

```
fixed     = 1200 + 10*250 + 500 + 2000 + 6554 = 12754
transient = 2000 + 6*2000 = 14000
surplus   = 65536 - 12754 - 14000 = 38782
target    = 30*300 = 9000
history   = 9000, layer1=6300, layer2=2700
```

规划目标取 30 轮，是因为 L1 只拿七成：`30*0.7=21` 轮，L1 的 6300 token
覆盖验收要求的 20 轮 × 300 = 6000 token 原文，L2 的 2700 留作降级缓冲。
因此默认窗口下 20 轮稳态短对话**不应触发任何降级和摘要**。压缩是成本不是
美德；装得下就不压。

### 6.4 中文口径

`estimate_tokens(text) = len(text)`（1 字符 = 1 token，保守）。预算常数
（每轮 300、system 1200、证据 250/条、摘要 500）全部按这个口径校准。若要把
估算改成更贴近真实 tokenizer 的比例，必须同时重标全部常数；只改估算会让
“预算够/不够”的结论整体反转。

## 7. 三层切分与降级

### 7.1 L2 渲染规则

L2 是**派生视图**，不落库；每轮从 State 消息按锚点范围渲染：

| 消息 | 渲染 |
|---|---|
| HumanMessage | 原文，一个字不压 |
| AIMessage 文本 | 前 `assistant_l2_chars`（默认 40）个字符 + `…` |
| AIMessage 带 tool_calls | `[调用工具: query_order]` + 文本前缀；无文本则只留工具标识 |
| ToolMessage（大块） | `[工具结果: query_order 已省略 N 字]` 一行；工具名从前一条 AIMessage 的 tool_call_id 映射 |
| ToolMessage（小块，≤ `tool_result_marker_chars`） | 原文，最多 200 字 |
| SystemMessage（历史遗留） | 前 40 字；新代码不再产生可变 system |

L2 的 token 数 = 每条渲染文本的 `estimate_tokens` + 固定消息开销（默认 4/条）。
触发摘要看这个数，不看消息条数。

### 7.2 L1 原文

L1 是锚点之后的历史消息，连同 tool 调用/结果原样保留。切窗：

1. 找 State 里第一条用户消息，其 `db-{id}` 数值 `> layer1_from_msg_id`；
2. 从它切到历史末尾（不含当前轮；当前轮由 runtime 单独追加）；
3. 用 LangChain `trim_messages(strategy="last", max_tokens=layer1_budget,
   start_on="human", token_counter=<按消息估算>)` 做最后兜底，防止单轮超大时
   溢出。正常路径不应触发兜底；触发时记 `L1 trim fallback` warning。

锚点缺失（旧会话没有 `db-` id、或找不到边界）时退化为纯 token 裁剪：从历史尾部
按 `trim_messages` 保留，不崩。

### 7.3 降级（L1 超预算）

每轮入口、拼 prompt 之前检查**历史** L1 的 token 数（不含当前轮）：

1. 若 `l1_tokens <= layer1_budget`，不降级。
2. 若超预算，从 L1 最老的轮次开始，把 `layer1_from_msg_id` 往后移到该轮用户消息的
   前一条 assistant 消息 id，直到 `l1_tokens <= layer1_budget * 
   layer1_degrade_low_water_ratio`（默认 0.8）或 L1 只剩最近一轮。
3. 至少保留最近一轮原文；单轮本身超预算时保留这一轮并记 warning。
4. 只 `UPDATE conversations.layer1_from_msg_id`，不搬消息；日志
   `层1 降级 {old}→{new} l1_tokens={before}→{after}`。旧值为 NULL 记 `none`。
5. 降级后重新渲染 L2；被降级进 L2 的消息立即按 7.1 截短。

### 7.4 L2 超预算

渲染后的 `l2_tokens > layer2_budget` 且 L2 至少有一整轮、且该会话没有在飞摘要任务 →
提交后台摘要（§9）。日志 `summary trigger l2_tokens=... > layer2_budget=...`。
摘要完成后 L2 归零，`summary_upto_msg_id` 追到 `layer1_from_msg_id`，L1 边界不动。

## 8. 拼装顺序与 Prompt

### 8.1 固定顺序

```
[SystemMessage] 人设 + 红线 + 工具调用规则（每个 route 恒定）
[L2 渲染消息...]
[L1 原文消息...]
[当前轮 UserMessage(原文)]
[本轮材料 HumanMessage: 摘要 + 证据 + 订单数据 + 意图]   ← 仅当有材料
[当前轮后续 AIMessage/ToolMessage...]                      ← ReAct 循环中
```

- system 只允许一条；摘要、证据、订单数据、意图**不进 system**。
- 本轮材料挂在**当前用户消息之后**；ReAct 后续轮次仍插在同一位置，不追加到
  tool 结果后面。
- 工具定义通过 `bind_tools` 传给模型，和 system 一起构成稳定前缀；同一 route
  内每轮一致。route 变化（业务→退款）时工具集变化，前缀本来就会失效，属预期。

### 8.2 本轮材料

新增 `render_turn_material(summary, evidence, order_data, intent)`，输出一段带标签的
`HumanMessage`：

```
【系统提供的本轮材料，不是用户发言】
【早前对话摘要】...
【本轮知识库证据】[1] 章节... 问... 答...
【当前订单】订单号... 状态...
【识别意图】...
```

只拼非空段；全空则不插入。证据编号仍按 `order_for_prompt` 的顺序，`sources` 帧
不变。引用规则里的“下方证据”改成“本轮材料中的证据”，避免指向 system 下面。

### 8.3 coref / 意图识别

`resolve_reference` 和 `classify_intent` 不再取 `state["messages"][:-1]`，改用
`state["prompt_history"]`（L2+L1）的最后几条；摘要行拼进同一个 HumanMessage：

```
(早前对话摘要: ...)
最近对话：
用户：...
客服：...
当前用户消息：...
```

摘要行只作背景，不单独占 system。

### 8.4 ReAct Agent

- 历史 = `state["prompt_history"]`（L2+L1）。
- 当前轮 = `state["messages"][state["turn_start_index"]:]`；普通轮从新用户消息起，
  resume 从最后一条用户消息起。
- 每步把本轮材料插在当前轮第一条 HumanMessage 之后，再调 `llm.bind_tools(...)`。
- `model_ctx` 在每次调模型前打印实际发出的摘要全文、逐条消息角色+前 40 字、窗口
  条数、token 估算。
- 工具结果仍按 `tool_result_max_tokens` 截断（1 字符=1 token 口径）；工具结果不落库。

### 8.5 禁止可变 system

- 摘要不建 `SystemMessage`。
- `fetch_order` 的 `[订单选择器]` 从 `SystemMessage` 改为 `HumanMessage`（或并入本轮
  材料），避免被上游模板上提。
- 任何历史 system 消息在拼装前都过滤/转成材料，最终 prompt 只有开头一条 system。

## 9. 后台摘要任务

### 9.1 触发

- 时机：每轮入口，降级检查之后、拼 prompt 之前。
- 条件：`l2_tokens > layer2_budget`；L2 至少有完整一轮；该会话无在飞任务。
- 触发后提交线程池并立即返回；当轮 prompt 仍用提交前的 L2，下一轮才看到新摘要。
- 触发日志：`summary trigger conv=... l2_tokens=... layer2_budget=... from=...
  upto=...`。

### 9.2 任务体

1. 自开 DB session，重新读 `conversations.summary / summary_upto_msg_id /
   layer1_from_msg_id`。
2. 计算范围 `(summary_upto, layer1_from]`；若为空、或 `upto <= 旧 summary_upto`，
   `summary skip reason=no_range|boundary_moved` 并返回。
3. 查该范围 `role in ('user','assistant')` 的消息，按 id 升序；`from_msg_id` 取
   第一条，`upto_msg_id` 取最后一条。
4. 旧 `conversations.summary` 只作背景传给摘要模型，**不参与合并**；提示词要求只
   输出这一段新消息的事实与诉求。
5. 严格 JSON `{"summary": "..."}`；解析失败/空/超长 → `summary skip
   reason=invalid_json|blank|too_long`，不写库。
6. 追加 `conversation_summaries`：`seq = max(seq)+1`；`content` 为新段。
7. 同一事务更新 `conversations.summary = 旧投影 + "\n\n" + 新段`，
   `summary_upto_msg_id = upto_msg_id`；L1 边界不动。
8. 日志：`summary start`、`summary done conv=... seq=... from=... upto=...
   chars=... elapsed_ms=...`；异常 `logger.exception("summary failed ...")`，
   rollback，不重试（下一轮条件仍满足会自然重触发）。

### 9.3 摘要提示词红线

- 只提炼**事实与诉求**：问过哪款商品/型号、报过的订单号/手机号、明确的退款/退货/
  售后诉求、尚未解决的问题、已确认的处理结论。
- 对话里没出现的内容一个字不许编；不确定就不写。
- 寒暄、闲聊、客套、重复确认不留。
- 长度控制在几十到一两百字（目标 50–200 个汉字）。
- 只输出 JSON：`{"summary": "<这一段的事实与诉求>"}`，不要解释、不要 Markdown。
- 旧摘要只作背景；不要把旧摘要的内容重新概括进新段。

### 9.4 投影增长

`conversations.summary` 按 `seq` 拼接**所有**段，保证最早的事实（如第一次报的订单号）
不会被“只留最近几段”丢掉。预算里用 `max(500, actual_summary_tokens)`：投影变大时
历史预算相应缩小；若摘要本身就把窗口吃穿，预算自检报警，但摘要不丢。

## 10. State 与 Runtime 双轨

### 10.1 State 新字段

`CustomerServiceState` / `AgentState` 增加：

```python
summary: str
summary_upto_msg_id: int | None
layer1_from_msg_id: int | None
prompt_history: list[AnyMessage]     # L2 渲染 + L1 原文
history_l2: list[AnyMessage]
history_l1: list[AnyMessage]
turn_start_index: int
l2_tokens: int
l1_tokens: int
context_budget: dict
```

除 `messages` 外都是普通覆盖字段，每轮入口重写。`messages` 仍：
`Annotated[list[AnyMessage], add_messages]`，checkpoint 全量保留。

### 10.2 Runtime 预处理

`app/services/session_context.py`（新模块）提供：

```python
@dataclass
class ContextPlan:
    summary: str
    summary_upto_msg_id: int | None
    layer1_from_msg_id: int | None
    history_l2: list[BaseMessage]
    history_l1: list[BaseMessage]
    prompt_history: list[BaseMessage]
    turn_start_index: int
    l2_tokens: int
    l1_tokens: int
    budget: ContextBudget
    degraded: bool

def prepare_context(session, conversation, prior_messages, turn_start_index,
                    settings) -> ContextPlan: ...
```

流程：

1. 取 prior State 消息（当前轮之前）。若 State 里最老的 `db-` 用户 id 大于会话
   第一条消息 id，说明有 MySQL 历史没进 checkpoint（旧会话/重启/迁移）：用
   `list_dialog_messages` 把 **`summary_upto` 之后、最老 State `db-` id 之前** 的
   user/assistant 行补到窗口前面，用户消息补 `db-{id}`。这个合并每轮都做，直到
   State 覆盖到会话开头，避免第二轮把旧历史丢掉。
2. 按锚点切 L2/L1；L2 渲染；L1 原文。
3. L1 超预算 → 降级并持久化锚点。
4. L2 超预算 → 提交后台摘要。
5. 打 `history_ctx`。
6. 返回 plan；runtime 把 plan 字段和 `HumanMessage(id=db-{msg_id})` 一起作为 graph
   inputs。

### 10.3 一张图、一个 checkpoint

ch07 不新建第二套图：`/ch07/chat` 与 `/ch06/chat` 挂同一份懒加载图，checkpoint
统一用 `CH07_CHECKPOINT_DB`；`CH06_CHECKPOINT_DB` 仅保留给旧测试/旧进程，不再
读写。ch06 页面继续在 `/ch06` 可用（不带 `conversation_id` 时取最近进行中会话），
ch07 页面在 `/ch07` 与 `/`。旧 ch06 的 user-thread checkpoint 不迁移；切到新
`conv:` thread 后第一次用 MySQL 行兜底重建，见 §10.2。

### 10.3.1 每轮请求流程

```
POST /ch06/chat
  resolve/create conversation (new_conversation / conversation_id / latest)
  config = {"configurable": {"thread_id": f"conv:{cid}"}}
  snapshot = graph.get_state(config)
  if snapshot.interrupts: emit conversation frame + interrupt; return
  append user message -> msg_id
  prior = snapshot.values["messages"]  # 当前轮之前
  plan = prepare_context(..., prior, len(prior), ...)
  inputs = _reset_inputs(...) + plan + HumanMessage(id=f"db-{msg_id}")
  emit {"type":"conversation","conversation_id":cid}
  stream graph ...
  persist assistant; trigger summary was already submitted at entry
```

Resume：

```
POST /ch06/chat/resume
  resolve conversation (conversation_id required for ch07; legacy latest for ch06)
  config = conv:{cid}
  pending check -> 409 if none
  append audit user row [订单选择]
  prior = snapshot.values["messages"]
  turn_start = index of last HumanMessage (pending turn)
  plan = prepare_context(...)   # 刷新摘要/锚点，但 Command 无法带 inputs
  stream graph Command(resume=...)
```

已知边界：resume 期间后台摘要若刚好完成，挂起轮仍用初始轮放进 checkpoint 的
`prompt_history`；下一轮才看到新摘要。接受这个一码宽的不一致，不做
`update_state` 热刷新，避免和 interrupt 互相干扰。

### 10.4 多会话 API

`ChatRequest` 增加：

```python
conversation_id: int | None = None
new_conversation: bool = False
```

- `new_conversation=True`：新建会话。
- `conversation_id`：用该会话；不存在或不属于该 user → SSE error frame。
- 都没有：兼容旧客户端，取该 user 最近一条“进行中”会话。
- resume 同样接受可选 `conversation_id`；ch07 前端必须带。

SSE 第一帧：`{"type":"conversation","conversation_id":123}`。ch06 旧页面要加一条
忽略该 frame 的分支，否则会把 JSON 当 token 显示。

### 10.4.1 退款/工单归属

多会话下 `POST /ch06/refunds` 与 `POST /ch05/tickets` 不能再按“最近进行中”
归会话。`RefundRequest` / `TicketRequest` 增加可选 `conversation_id`：

- 带值时校验属于该 user，用它设置 `current_conversation_id`；
- 不带时回落旧的“最近进行中”行为，兼容 ch06 页面；
- ch07 退款表单和建工单按钮都带当前 `conversation_id`；
- ticket 的 `conversation_id` 只用于归属，不改 `conversations.status`。

### 10.5 只读接口

`app/api/conversations.py`：

- `GET /api/conversations?user_id=`：该用户全部会话，`id desc`；每项
  `{id, preview, has_summary, created_at, updated_at}`。`preview` 取第一条 user
  消息前 30 字；`has_summary = summary IS NOT NULL`。
- `GET /api/conversations/{id}/messages`：会话不存在 404；可选 `user_id` 查询
  参数，若提供且不属于该会话则 404（ch07 页面总是带）。按 id 升序返回
  user/assistant 原文 `{id, role, content, created_at}`；`[订单选择]` 审计行
  不返回给前端（见 §10.6）。

### 10.6 审计行与订单选择器

- `resume_chat_agent` 写的 `[订单选择] 订单 N` 是审计行，不是用户原话；
  `GET .../messages` 过滤掉以 `[订单选择]` 开头的 user 行，前端不渲染假气泡。
- 摘要任务仍能读到这条行：把它规范成「用户选择了订单 N」的系统事实行再交给
  摘要模型，不当作 user 发言，避免摘要写成“用户说 [订单选择]…”。
- `fetch_order` 在 checkpoint 里追加的订单选择说明改为带标签的 `HumanMessage`
  （`【订单选择器】用户选择了订单 N`），不占 system；它参与当前轮，不进入历史
  摘要范围。

## 11. 可观测

| 日志 | 时机 | 内容 |
|---|---|---|
| `history_ctx` | 每轮入口，coref/意图之前 | `summary` 行（全文）、滑窗逐条角色+前 40 字、条数、token 估算 |
| `model_ctx` | main_agent 每次调模型前 | 摘要全文、实际发出的逐条消息角色+前 40 字、窗口条数、token 估算 |
| `层1 降级` | L1 超预算 | 旧边界→新边界、前后 token |
| `summary trigger` | L2 超预算 | 会话、L2 token、预算、范围 |
| `summary start/done/skip/fail` | 后台任务 | 会话、seq、范围、字符数、耗时 ms；失败带异常 |
| `context budget` | 启动/图构建 | window/fixed/transient/history/layer1/layer2；预算不足 warning |

`history_ctx` 每轮必打，闲聊兜底轮（不进 main_agent）也看得到。

`log/app.log` 由新模块 `app/logging_config.py` 的 `configure_file_logging()` 建：
UTF-8 `FileHandler` + console，INFO 级别，幂等（重复调用不叠加 handler）。
`run.py` 启动时调用；FastAPI lifespan 在未配置时也调用一次。测试进程默认不落
文件（`LOG_TO_FILE=0` 可显式关闭）。验收统一用 `python run.py` 启动后
`grep model_ctx log/app.log` / `grep history_ctx log/app.log`。

## 12. 前端多会话侧栏

新增 `app/static/ch07.html`（沿用 ch06 像素风），并在 `/ch07` 和 `/` 提供；ch06 页面
保留在 `/ch06`。

- 布局：左侧会话侧栏 + 右侧聊天面板。窄屏 `<780px` 隐藏侧栏，聊天不受影响。
- 侧栏：
  - 「新对话」按钮：清空当前视图，`activeConversationId=null`；下一条消息带
    `new_conversation=true`，后端建会话后回 `conversation` frame，前端选中它。
  - 会话列表：新在前、首问预览、已摘要徽标、active 高亮；点击切换并回载历史。
  - 每轮结束后刷新列表（预览/已摘要/顺序）。
- 历史回载：`GET /api/conversations/{id}/messages`，逐条 user/assistant 原文渲染；
  不依赖 localStorage 存消息。
- 聊天：沿用 ch06 的 SSE 解析、订单选择器、退款表单；请求带 `conversation_id` 或
  `new_conversation`。
- 失败降级：侧栏/历史加载失败只静默隐藏或保持旧列表，不影响聊天主链路。
- 切换保护：流式/订单选择中禁用侧栏点击，避免半截消息串会话。
- ch06 页面：只加一条忽略 `conversation` frame 的分支，其余不动。

## 13. 模块改动清单

| 文件 | 改动 |
|---|---|
| `sql/ch07-ddl.sql` | 用户给定第一段 DDL |
| `sql/ch07-layers.sql` | 用户给定第二段 DDL |
| `app/models/conversation.py` | 三列 |
| `app/models/conversation_summary.py` | 新模型 |
| `app/models/__init__.py` | 导出新模型 |
| `app/config.py` | 新配置、effective 属性、预算自检 |
| `app/services/context.py` | token 估算/块切分/L2 渲染/trim 兜底/材料拼装 |
| `app/services/session_context.py` | 新：预算、切层、降级、摘要触发、ContextPlan |
| `app/services/summary_job.py` | 新：线程池摘要任务、提示词、JSON 解析、追加段 |
| `app/services/history.py` | 会话解析/创建、锚点读写、分段摘要读写、对话读取 |
| `app/agent/state.py` | 新字段 |
| `app/agent/service.py` | 会话解析、conv thread、plan 注入、conversation frame、resume |
| `app/agent/graph.py` | coref/意图用 prompt_history；订单选择器改 HumanMessage；rerank 用 effective top_k |
| `app/agent/subgraph.py` | system 恒定；材料消息插入；model_ctx；工具结果 token 截断 |
| `app/prompts/customer_service.py` | system 去可变块；新增 render_turn_material |
| `app/api/ch06.py` | 可选 conversation_id/new_conversation、conv thread、conversation frame；refund 带 conversation_id |
| `app/api/ch07.py` | 新：/ch07 页面 + /ch07/chat 别名；与 /ch06 共用一张图 |
| `app/api/ch05.py` | tickets 带可选 conversation_id，按会话归属 |
| `app/api/conversations.py` | 新：两个只读接口（messages 支持 user_id 校验） |
| `app/schemas.py` | ChatRequest / RefundRequest / TicketRequest / ResumeRequest 新字段 |
| `app/logging_config.py` | 新：log/app.log FileHandler，幂等配置 |
| `run.py` | 启动时 configure_file_logging() |
| `app/main.py` | 挂 ch07/会话接口；启动预算自检与日志配置；关闭线程池 |
| `app/static/ch07.html` | 新多会话页面 |
| `app/static/ch06.html` | 忽略 conversation frame |
| `app/jobs/eval_ch07_summary.py` | 新：摘要 prompt 标注样例评估 |
| `tests/eval/ch07_summary_cases.json` | 新：标注样例 |
| `.env.example` | 新配置说明 |
| `dev-notes/ch07.md` | 过程留痕 |

## 14. 错误处理与边界

- 摘要失败/JSON 解析失败：log + 放弃本次，下轮重触发；summary 字段只在成功后更新。
- 降级持久化失败：log warning，继续用旧锚点；prompt 可能略超预算，不崩。
- checkpoint 缺失/旧消息无 `db-` 锚点：用 DB 行重建窗口；仍找不到边界则纯 token 裁剪。
- 旧会话（ch06 的 user thread）：切到按 conversation 的 thread 后 checkpoint 为空，
  用 `list_dialog_messages` 重建 prompt_history；全量 State 历史不保证补回，但
  继续聊可用。
- interrupt/resume：resume 不改写 State 的 `messages`；订单选择器 note 改
  HumanMessage；挂起期间摘要完成的一码宽延迟见 §10.3。
- 用户输入超 `max_user_input_tokens`：模型输入截断并 warning，DB 保留全文。
- 并发摘要：每会话在飞标记；不同会话可并行；服务关闭 `shutdown(wait=False)`。
- 服务重启：在飞任务丢失无妨；锚点已落库，下轮条件仍满足会重触发。
- 摘要与下一轮并发读：任务只写 conversations 两字段 + 新段；入口读到旧值最多晚
  一轮生效，无一致性风险。
- 侧栏/历史接口失败：前端静默降级；聊天主链路不受影响。

## 15. 测试与验证

### 15.1 代码 TDD（测试库 `mewhelp_ch02_test` / SQLite 惯例）

- `test_context_budget.py`：演示配置 5650/3955/1695；只改窗口归零 + warning；
  默认窗口 20 轮不压；70/30。
- `test_context_layers.py`：L2 渲染（用户原文、客服 40 字、tool 一行标识）；
  锚点命中/缺失退化；L1 超预算降级（低水位、至少留一轮、只挪 id）；L2 超预算。
- `test_context_summary.py`：范围计算、seq 追加、投影拼接、旧段不变、跳过/失败、
  防抖（并发只起一个）、触发异常不外抛。
- `test_history.py` 扩展：会话创建/读取/列表、锚点读写、分段摘要读写、对话读取。
- `test_agent_context.py`：拼装顺序（system → L2 → L1 → 当前 user → 材料）；
  材料在 user 之后、tool 之前；无摘要/证据不插；system 只有一条；caplog 验证
  `model_ctx` / `history_ctx` 内容。
- `test_conversations_api.py`：两个只读接口（列表顺序/预览/has_summary、404）。
- `test_ch07_api.py`：new_conversation / conversation_id / 兼容无 id；conv thread；
  conversation frame；resume 带 conversation_id。
- `test_ch07_page.py`：页面含侧栏、两个接口、移动端隐藏、conversation frame 处理。
- 更新 ch05/ch06 既有测试：thread id 断言、prompt 渲染、subgraph 参数、
  ChatRequest 新字段、订单选择器消息类型。

### 15.2 摘要 prompt 评估（替代不可单测的 TDD）

`app/jobs/eval_ch07_summary.py` + `tests/eval/ch07_summary_cases.json`：

- 样例含订单号、手机号、商品型号、明确诉求、未决问题、寒暄。
- 真实模型跑，断言：严格 JSON；订单号/手机号/诉求在；无编造实体；长度 20–200 字；
  寒暄不出现。
- 报告写 `docs/superpowers/reports/2026-11-18-ch07-summary-eval.md`，记录 prompt
  调整与通过率。

### 15.3 手动验收（用户全浏览器）

1. 默认窗口聊 20+ 轮：token 不爆、系统不崩；日志无 `层1 降级`/`summary trigger`。
2. 演示配置再聊：
   - 看到 `层1 降级 X→Y`；
   - 看到 `summary trigger 层2 约 N token > 预算 M`；
   - 看到 `summary done 第N段`；
   - 问「最开始那个订单后来怎么说」，答复含早期订单号与诉求。
3. `grep model_ctx log/app.log` / `grep history_ctx` 看到每轮摘要+滑窗；
   `grep summary` 看到触发/完成/耗时，且当轮回复未被阻塞。
4. 前端侧栏开多个会话对照、切回旧会话历史完整回载、继续聊。

## 16. 配置

| 配置 | 默认 | 含义 |
|---|---|---|
| `MODEL_CONTEXT_WINDOW` | 65536 | 模型窗口 |
| `MAX_OUTPUT_TOKENS` | 2000 | 输出预留；同时设给 LLM |
| `MAX_USER_INPUT_TOKENS` | 2000 | 当前用户输入预留 |
| `MAX_AGENT_STEPS` | 空 | ReAct 步数；空回落 `AGENT_MAX_STEPS` |
| `TOOL_RESULT_MAX_TOKENS` | 空 | 工具结果上限；空回落 `AGENT_TOOL_RESULT_MAX_CHARS` |
| `RERANK_TOP_K` | 空 | 证据条数；空回落 `RERANK_TOP_N` |
| `SAFETY_MARGIN_RATIO` | 0.10 | 安全余量 |
| `SYSTEM_PROMPT_RESERVE_TOKENS` | 1200 | system 固定预留 |
| `EVIDENCE_TOKENS_PER_CHUNK` | 250 | 每条证据预留 |
| `SUMMARY_RESERVE_TOKENS` | 500 | 注入梗概下限预留 |
| `HISTORY_TARGET_TURNS` | 30 | 规划轮数（L1 七成约 21 轮，覆盖 20 轮原文验收） |
| `HISTORY_TOKENS_PER_TURN` | 300 | 每轮稳态占用 |
| `LAYER1_RATIO` | 0.7 | L1 占比 |
| `LAYER1_DEGRADE_LOW_WATER_RATIO` | 0.8 | 降级低水位 |
| `ASSISTANT_L2_CHARS` | 40 | L2 客服答复保留字数 |
| `TOOL_RESULT_MARKER_CHARS` | 200 | 大块工具结果阈值 |
| `SUMMARY_MODEL` | 空 | 摘要模型；空回落 `MODEL_NAME` |
| `SUMMARY_MAX_CHARS` | 200 | 摘要长度上限 |
| `CH07_CHECKPOINT_DB` | `.data/ch07-checkpoints.sqlite` | 共用图的 checkpoint |
| `CH06_CHECKPOINT_DB` | 旧值 | ch07 起不再读写，仅兼容旧进程/测试 |
| `MAX_CONTEXT_TOKENS` / `AGENT_TOKEN_BUDGET` | 旧值 | 仅 `/chat`、`raw_demo` 旧链路使用，ch07 退役 |
| `LOG_TO_FILE` | 1 | 0 关闭 log/app.log FileHandler（测试用） |
| `CONTEXT_LOG_PREVIEW_CHARS` | 40 | 日志预览字数 |

## 17. 验收标准映射

| 验收 | 设计落点 |
|---|---|
| 20+ 轮不爆不崩 | §6 预算、§7 降级、§9 摘要 |
| 演示配置完整级联、早期订单答对 | §6.2、§7、§9、§15.3 |
| 默认窗口 20 轮不压 | §6.3、§15.1 |
| 摘要不阻塞 + 日志可 grep | §9.3 线程池、§11 日志 |
| 侧栏多会话、回载、续聊 | §10.4、§10.5、§12 |

## 18. 留给 builder

- 具体函数拆分、命名、测试用例粒度。
- ch07 页面沿用 ch06 的订单选择器/退款表单代码的组织方式。
- 线程池 worker 数量（建议 2）。
- 摘要评估样例的具体条目。
- 是否需要把 ch06 旧页面也升级成会话侧栏（本章只保证 ch07 页面）。

## 19. 核对后定下的实现取舍

独立 builder 通读后提了 7 个会影响计划的问题，已在正文落定，列在这里供你复核：

1. **旧预算开关退役**：`MAX_CONTEXT_TOKENS` 不再二次裁剪 ch07 prompt，
   `AGENT_TOKEN_BUDGET` 不再作为 ReAct 耗尽条件；步数由 `MAX_AGENT_STEPS` 控，
   瞬时峰值由 `transient` 预留。两者保留给旧 `/chat` / `raw_demo`。
2. **一张图、一个 checkpoint**：`/ch07/chat` 与 `/ch06/chat` 共用同一份懒加载图，
   统一用 `CH07_CHECKPOINT_DB`；旧 ch06 user-thread checkpoint 不迁移，靠 MySQL
   行兜底重建。ch06 页面保留可用。
3. **退款/工单归属**：`RefundRequest` / `TicketRequest` 增加可选
   `conversation_id`，带值时校验归属并用它设置 `current_conversation_id`；不带
   时回落旧行为。ch07 表单总是带。
4. **消息接口按用户校验**：`GET /api/conversations/{id}/messages` 接受可选
   `user_id`，提供且不匹配则 404；ch07 页面总是带。
5. **旧历史每轮补合并**：不只 checkpoint 为空时重建；只要 State 最老 `db-` id
   晚于会话第一条消息，就把 `summary_upto` 之后、State 之前的 MySQL 行补到窗口
   前面，避免第二轮丢旧历史。
6. **日志落文件**：新增 `app/logging_config.py`，`run.py` / lifespan 幂等配置
   `log/app.log`（UTF-8 FileHandler + console，INFO）；测试可 `LOG_TO_FILE=0`。
7. **审计行与订单选择器**：`[订单选择]` 审计行不进前端历史；摘要任务把它规范成
   「用户选择了订单 N」的系统事实行；`fetch_order` 的订单选择说明从 system 改为
   带标签的 HumanMessage。
