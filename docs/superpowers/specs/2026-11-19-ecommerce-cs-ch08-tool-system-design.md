# 第八章 · 工具系统 设计文档（对齐参考 spec 版）

- 日期：2026-11-19
- 分支基线：`feature/ch07-context-management`
- 目标分支：`feature/ch08-tool-system`
- 本版依据：用户提供的《Ch08 设计：即插即用工具系统》参考文档，以及本仓库 ch01–ch07 的实际代码结构。
- 范围：注册中心、统一执行引擎、MCP 接入、建工单确认流、审计。

## 0. 与参考 spec 的映射

参考 spec 的架构决策全部采纳；其中文件/接口名与本仓库不一致的，按下表落地：

| 参考 spec | 本仓库落地 |
|---|---|
| `app/tools/registry.py` 重造 | 同名重造 |
| `app/tools/builtin/` 包 | 新建同名包 |
| `app/tools/infra.py` | 本仓库是 `app/tools/executor.py`，重造为 `app/tools/engine.py`，`executor.py` 保留兼容薄壳或删除 |
| `app/tools/engine.py` | 新建 |
| `agent_tools` 节点 | `app/agent/subgraph.py` 的 `tools_node` |
| `runtime.py` 的 interrupt 帧/恢复 | `app/agent/service.py` + `app/api/ch06.py` 的 `/ch07/chat/resume` |
| `repository.insert_tool_audit` | 新建 `app/tools/audit.py`；本仓库没有 repository 层 |
| `models.py` 加 `ToolAuditLog` | `app/models/tool_audit.py` + `app/models/__init__.py` |
| `POST /api/actions/resume` | `POST /ch07/chat/resume`（`/ch06` 同） |
| `POST /api/actions/create-ticket` | `POST /ch05/tickets`（保持不动） |
| `tests/graph/test_ch06_nodes` | `tests/test_agent_subgraph.py` 等现有测试 |
| `tests/conftest.py` 的 `_DDL_FILES` | 本仓库 conftest 用 `Base.metadata.create_all`，新模型导入即可建表 |
| `uv add` | 本仓库用 `requirements.txt` + `.venv/Scripts/python.exe -m pip install` |

参考 spec 中声明的用户决策，本设计全部继承：

1. 架构方案 A：进程内动态注册表 + 四段式执行管道；否掉工具表落 MySQL、否掉 LangGraph `ToolNode` 黑盒定制。
2. 超时注入走环境变量：物流 MCP Server 读 `MOCK_DELAY_SECONDS`；主服务读 `DEMO_TICKET_DELAY_SECONDS` 让 `create_ticket` 变慢；不埋工具参数后门。
3. 意图新增第九类「人工」：明确要求建工单/转人工跟进 → `business`，让验收 4 真正走到主力 Agent。验收 2 的对话演示只做物流轨迹；在保/退货进度由注册中心和集成测试证明可用。

## 1. 背景与目标

ch02–ch07 的工具层是几个写死的内置工具：静态 `ALL_TOOLS`、`execute_tool` 只做超时和粗重试、没有 JSON Schema 校验、没有权限分层、没有审计，`create_ticket` 在 Agent 侧被直接排除。

本章升级为即插即用的工具系统：

1. **注册中心**：内置与 MCP 工具统一带齐名称、用途描述、JSON Schema；内置启动登记，MCP 每步现拉；新增内置工具只丢文件，新增 MCP 工具只重启对应 Server。
2. **参数校验**：JSON Schema 校验，错误包成工具结果回灌，不抛异常。
3. **权限控制**：只读/写两类；写操作只有 `create_ticket`；模型无法伪造确认；MCP 声明不可信，只认本侧规则。
4. **执行引擎**：所有调用同一管道；超时、重试、错误分诊、结果格式化；写操作 0 重试；业务空结果不重试。
5. **审计**：`tool_audit_logs` 每次最终调用落一条，含被拦调用；无外键；审计失败不反拦。
6. **MCP 接入**：物流、售后两台独立 Server，Streamable HTTP；内置 `query_logistics` 下线，MCP 接管，不留重名。
7. **建工单确认流**：明确要求 + 信息齐全才发起；interrupt 预览；确认落库回工单号；取消落「权限拒绝」审计；ch05 按钮建单路径不动。

非目标：

- Skill 机制。
- 仓储等更多外部系统。
- MCP Server 鉴权。
- 注册表落库/跨进程注册中心。
- 工具清单缓存与失效策略：现问现拿，不做缓存层。

## 2. 目标 / 非目标 / 反目标

### 目标

- 验收 1–6 全部可演示。
- 新增内置工具零核心代码改动；MCP 工具热增只重启 MCP Server。
- 校验、权限、重试、格式化、审计只有一处实现。
- 建工单确认流不重跑已执行的工具，模型无后门。

### 反目标

- MCP Server 通过声明伪装写操作，或遮蔽内置写工具。
- 模型在参数里伪造确认。
- 审计写失败拦掉工具执行。
- 写操作超时自动重试。
- resume 重跑已经执行过的读工具。

## 3. 架构与数据流

```
app/tools/builtin/            启动时 pkgutil 扫描；每模块 = @tool + registry.register(ToolSpec)
  orders.py / faq.py / tickets.py / refunds.py / promotions.py（验收 1）
        │
app/tools/registry.py         内置登记表
        │
app/tools/mcp_client.py       MultiServerMCPClient；每次 get_all_specs() 现查
        ├ logistics_server   :8101  （mcp_servers/，独立进程）
        └ aftersales_server  :8102  （mcp_servers/，独立进程）
        ↓
get_all_specs() = builtin + mcp；重名 builtin 优先，MCP 丢弃并 warning
        ↓
main_agent  bind_tools([spec.tool]) ── 模型发起 tool_calls
        ↓
subgraph.tools_node
  ├ 含 create_ticket → 纯校验 → interrupt(confirm_ticket 预览) ⇄ 前端
  │                       resume {"confirmed": bool}
  ├ 其余工具 / 确认后的 create_ticket → engine.execute_tool_call
        ↓
engine.py 统一管道：查工具 → jsonschema 校验 → 权限门 → 执行（超时/重试）
                     → 分诊 → 格式化 → 审计
        ↓
tool_audit_logs
```

关键点：

- `get_all_specs()` 在每次 `main_agent` 绑定工具前、每次 `tools_node` 执行前各拉一次。一步 ReAct 约 2 次本地 HTTP 清单拉取，毫秒级；没有缓存失效逻辑。
- 确认中断放在 `tools_node` **顶部、执行任何工具之前**。interrupt 之前只做纯计算；resume 重跑该节点时不会重复执行已跑过的工具。
- MCP 拉取失败：告警并只用内置工具，不拖垮对话。

## 4. 工具注册中心（`app/tools/registry.py` 重造）

```python
@dataclass
class ToolSpec:
    name: str                    # 工具名（唯一键）
    description: str             # 用途描述
    json_schema: dict            # 参数 JSON Schema
    tool: BaseTool               # 底层可调用对象
    permission: str              # "read" | "write"，只认我们侧规则
    source: str                  # "builtin" | "mcp"
    mcp_server: str | None       # 来源 Server 名，内置为 None
    timeout: float | None        # 超时覆盖；None 用默认（内置 5s / MCP 10s）
    inject_conversation: bool    # 是否注入 conversation_id
    format_result: Callable | None
    classify: Callable | None    # 把裸返回值判成 ok/empty/error
    exposed_to_agent: bool       # 是否绑定给 Agent
    routes: frozenset[str] | None  # 允许的 Agent 路由；None 表示全部
    label: str | None            # 流式 tool 帧展示名
```

### 内置登记

- `app/tools/builtin/` 包，每个模块 = `@tool` 实现 + `registry.register(ToolSpec(...))`。
- 服务启动（lifespan）时 `pkgutil.iter_modules` 扫描导入整包，完成登记。
- 模块划分：
  - `orders.py`：`query_order`、`query_product`
  - `faq.py`：`query_faq`（30s 超时，父图专用，`exposed_to_agent=False`）
  - `tickets.py`：`create_ticket`（`write`）
  - `refunds.py`：`submit_refund`（只读控制工具）
  - `promotions.py`：`query_promotions`，验收 1 的丢文件示例
- 现有 `app/tools/business.py` 瘦身成 mock 数据纯函数：`DEMO_ORDERS`、`list_user_orders`、`order_snapshot`；`graph.fetch_order` 继续用。
- `query_logistics` 不迁入 builtin，直接从清单下线。

### MCP 现问现拿

- `app/tools/mcp_client.py` 持有 `MultiServerMCPClient`，两台 Server 的 URL 来自 settings。
- `fetch_mcp_specs()` 逐 Server `get_tools(server_name=...)`，单台 try/except：失败告警 + 跳过。
- 工具三要素来自适配器：`tool.name`、`tool.description`、`tool.args` / `args_schema` 转 JSON Schema。
- `source="mcp"`、`mcp_server=Server 名`、`permission="read"`（本侧强制，不看 Server 声明）。
- 不做缓存：`get_all_specs()` 每次现拉，Server 侧加工具下一轮（或下一步）即可见。

### 合并与重名

- `get_all_specs()` = builtin + MCP。
- 重名后到者丢弃并 `logger.warning`；内置永远优先。
- 本章靠下线内置 `query_logistics` 避免撞名，MCP 物流工具沿用 `query_logistics` 名。

## 5. 统一执行引擎（`app/tools/engine.py`）

对外入口：

```python
def execute_tool_call(
    tool_call: dict,
    conversation_id: int | None,
    *,
    confirmed: bool = False,
    specs: dict[str, ToolSpec] | None = None,
) -> ToolRun: ...
```

`agent_tools` 是模型调用的唯一入口；ch05/ch06 使用同一引擎的 `execute(spec, args, ...)` 底层入口，并显式传 `confirmed=True`。

管道：

| 段 | 规则 | 审计 status |
|---|---|---|
| 查工具 | 名字查 spec；查不到 → 「未知工具」回灌 | 失败 |
| 校验 | `Draft202012Validator(spec.json_schema).validate(args)`；错误包成「参数校验未通过：<具体原因>，请修正参数或向用户追问」，不抛异常 | 校验拦下 |
| 权限门 | `permission=="write"` 且 `confirmed != True` → 拒绝；模型参数无法提供 confirmed | 权限拒绝 |
| 执行 | 超时：内置 5s、MCP 10s、`query_faq` 30s、spec.timeout 覆盖；重试只给 `TimeoutError` + 网络/传输类异常，指数退避 0.2s × attempt，默认最多 2 次；write 恒 0 次 | 成功/失败/超时 |
| 分诊 | 参数不合法（校验层拦）；查询落空（工具语义表达）；真故障（异常 → 「工具暂时不可用：<类名>」如实回灌） | — |
| 格式化 | 字段白名单、内部枚举码翻中文、`json.dumps(ensure_ascii=False)`、截断 | — |
| 审计 | 每次最终调用落一条；含校验拦下、权限拒绝；审计失败只 log | — |

### 执行线程与异步桥

- 本仓库的 LangGraph 节点是同步的，因此引擎对外保持同步：同步工具在 daemon 子线程执行，主线程带超时等待；开线程前 `contextvars.copy_context()`，传播 `current_conversation_id` 与 `current_citation_collector`。
- MCP 工具是异步工具：`app/tools/async_bridge.py` 提供常驻事件循环线程和 `run_async(coro, timeout)`，用 `asyncio.run_coroutine_threadsafe` + `asyncio.wait_for`。`get_all_specs()` 也走这个桥。
- 检测到当前线程已有运行中事件循环时，桥自动切到工作线程，绝不嵌套 `asyncio.run`。
- 写操作超时后 daemon 线程可能仍在跑，但引擎不重试、不重复调用；这是「超时未必没执行」的防线。
- 任何路径都不向调用方抛异常；最坏返回 `ToolRun(status="error")`。

### 分类（`classify` 契约）

- 默认：`None` 或空字符串 → `empty`；其他 → `ok`。
- `query_faq`：`MISS_TEXT` → `empty`；`ERROR_TEXT` → `error`；其余原文 → `ok`，内容不得 JSON 包装。
- `create_ticket`：匹配 `T\d{11}` → `ok`；「创建工单失败」开头或无会话上下文 → `error`。
- MCP：JSON 对象 `found: false` → `empty`；正常 JSON → `ok`；异常/`isError` → `error`。
- `empty` 审计记「成功」（执行本身成功），结果如实说未查到，不重试。
- `error` 审计记「失败」，不重试。
- `timeout` 审计记「超时」，只读工具重试耗尽后记一次。

### 结果格式化

按工具名维护字段白名单与枚举翻译：

- `query_order`：`order_id, items, amount, status, tracking_no, created_at, signed_at`
- `query_product`：`product_name, price, stock, specs`
- `query_logistics`：`tracking_no, carrier, status_code, traces`；`status_code` 翻译：`PICKED_UP → 已揽收`、`TRANSIT → 运输中`、`ARRIVED → 到达转运中心`、`DELIVERING → 派送中`、`DELIVERED → 已签收`
- `query_warranty`：`order_id, product_name, warranty_status, purchase_date, expire_date, coverage`；`IN_WARRANTY → 在保`、`OUT_OF_WARRANTY → 已过保`、`UNKNOWN → 无法确认`
- `query_return_status`：`order_id, return_no, status, progress, updated_at`；`APPLIED → 已申请`、`PICKING → 待取件`、`IN_TRANSIT → 退货运输中`、`RECEIVED → 已收到退货`、`REFUNDING → 退款中`、`DONE → 已完成`
- `create_ticket`：`ticket_no`；模型可见内容 `{"ticket_no":"T..."}`，原始工单号放 `ToolRun.data`，ch05/ch06 不再正则匹配文本
- `query_faq`：不裁剪、不包装，原样返回 `[n]` 证据
- 未知工具：不裁剪，只做通用序列化和截断

空结果、错误、超时都用中文句子。截断长度沿用 `agent_tool_result_max_chars`（默认 2000），截断后追加「…（结果已截断）」。

### 审计

- DDL 见附录 A，原样落 `sql/ch08-ddl.sql`。
- ORM：`app/models/tool_audit.py`，`Base.metadata` 导入即可在 SQLite 测试库建表。
- 字段填充：
  - `tool_source`、`mcp_server` 来自 ToolSpec。
  - `arguments` 用 `json.dumps(default=str, ensure_ascii=False)`。
  - `result_summary` 取模型可见结果，截断 500 字符。
  - `status`：`成功 / 失败 / 超时 / 校验拦下 / 权限拒绝`。
  - `retry_count` 实际重试次数；写操作恒 0。
  - `duration_ms` 总耗时。
- 审计用独立 session（`get_session_factory()()`），异常只 `logger.exception`，绝不向上抛。
- `confirmation_required` 阶段不落审计；确认/取消时落最终结局。

## 6. 权限控制

- 本侧唯一真相：`WRITE_TOOLS = {"create_ticket"}`。
- 内置工具按登记 `permission` 判定；MCP 工具一律 `read`，忽略 Server 的 `destructiveHint`、描述或自定义字段。
- 模型发起的写调用：
  1. `tools_node` 先做纯参数校验；校验失败 → 引擎校验拦截路径回灌，审计「校验拦下」，不 interrupt。
  2. 校验通过 → 执行任何工具之前 `interrupt({"kind": "confirm_ticket", "preview": {...}})`。
  3. resume `{"confirmed": true}` → 引擎 `confirmed=True` 真执行；`false` → 引擎权限门拒绝，审计「权限拒绝」，回灌「用户取消了本次建工单，请不要再发起」。
  4. 任意其他 resume 值按取消处理（fail closed）。
- 可信服务端路径：`/ch05/tickets` 与 `/ch06/refunds` 是用户点按钮/交表单触发的服务端动作，调用引擎时写死 `confirmed=True`，不经过 interrupt。
  - 审计照常：`tool_source=builtin`、`tool_call_id=NULL`、`conversation_id` 取目标会话；状态按真实结局。
  - `/ch06/refunds` 继续 `retries=0`，失败 502。
- 有 pending 工单确认时，新聊天消息只重发 interrupt 帧，不运行图、不写用户消息，和订单选择器一致。

## 7. 建工单确认流（graph 改造）

`app/agent/subgraph.py` 的 `tools_node` 重写：

1. 收集本轮 `tool_calls`。若含 `create_ticket`：
   - 先对第一个 `create_ticket` 做纯校验（不落审计、不执行）。
   - 校验失败：走引擎校验拦截路径，审计「校验拦下」，ToolMessage 回灌，模型向用户追问；不 interrupt。
   - 校验通过：在节点顶部、执行任何工具之前 `interrupt(...)`。
   - 一轮多个 `create_ticket`：只处理第一个；其余立即回 ToolMessage「一次只处理一个建工单请求」，并通过 `deny_validation` 落「校验拦下」审计。
2. resume：
   - `{"confirmed": true}` → 引擎带 `confirmed=True` 执行，落 tickets 表；ToolMessage 带工单号；其余工具照常执行。
   - `{"confirmed": false}` → 引擎按「权限拒绝」落审计，回灌取消说明；其余工具照常执行。
3. 其余只读/控制/MCP 工具：
   - 控制工具 `suggest_handoff` / `submit_refund` 保留现有业务守卫；守卫通过后走引擎，确保校验/审计/格式化统一，再合并 `suggested_actions`；守卫失败走 `deny_validation` 审计。
   - `query_faq` 仍由父图调用，不绑定给 Agent。
4. 工单号兜底：
   - 成功后把 `created_ticket_no` 写进 State；服务层 `_finish_turn` 在回答落库前检查最终 `answer` 是否包含该工单号，不包含就确定性追加「工单已创建，工单号：T...」。
5. State 新增 `created_ticket_no: str`（父图/子图同名共享）。不把工具对象或快照写进 State。

删除/保留：

- 删除任何 `create_ticket → suggested_actions` 的旧拦截（本仓库目前没有，确认不存在即可）。
- 保留 `submit_refund → refund_form` 拦截。
- 不动 ch05 投诉流程的 `/ch05/tickets` 按钮建单路径。

## 8. 意图与路由（第九类「人工」）

参考 spec 的实现期裁决，本仓库同样存在：

- 现状：`INTENT_ROUTES` 把「售后」送 `refund_flow`、「其他」送 `fallback_script`；「帮我建个工单」容易被判成「其他」，到不了主力 Agent。
- 改动：
  - `app/agent/state.py` 的 `Intent` 增加 `"人工"`。
  - `app/agent/intent.py` 的 `INTENT_LABELS`、prompt 类别定义、边界样例、输出格式说明同步加第九类。
  - `INTENT_ROUTES["人工"] = "business"`。
  - 意图评估集增加「人工」样例；`eval_intent` 从 16 例扩到 18 例，原 16 例零回归。
- 验收 2 口径：对话里只演示物流轨迹；`query_warranty` / `query_return_status` 由注册中心测试和 MCP 集成测试证明可用。

## 9. 两台业务 MCP Server

顶层新目录 `mcp_servers/`，官方 SDK，Streamable HTTP，独立进程，不接 DB 不建表。

| Server | 端口 | 工具 | 说明 |
|---|---|---|---|
| `logistics_server.py` | 8101 | `query_logistics(tracking_no)` | mock 轨迹；返回内部枚举 `status_code`；读 `MOCK_DELAY_SECONDS` 注入延迟（验收 6） |
| `aftersales_server.py` | 8102 | `query_warranty(order_id)`、`query_return_status(order_id)` | 查在保/查退货进度；mock 稳定种子 |

- mock 数据按第 2 章做法：随机字段 + 按单号稳定种子；JSON `ensure_ascii=False`。
- SDK 新老 API（`FastMCP` → `MCPServer`）以实装版本为准，动手前用 Context7 按版本核对导入路径。
- 启动命令：
  - `python -m mcp_servers.logistics_server`
  - `python -m mcp_servers.aftersales_server`
- 验收 3 演示：现场往 `logistics_server.py` 加 `query_delivery_eta`，只重启 8101；客服系统不动，下一轮对话即可用。

## 10. MCP Client 接入

- settings：
  - `mcp_enabled: bool = False`（代码默认；演示 `.env` 设 true）
  - `mcp_logistics_url = http://127.0.0.1:8101/mcp`
  - `mcp_aftersales_url = http://127.0.0.1:8102/mcp`
  - `mcp_discovery_timeout_s = 2`
  - `mcp_tool_timeout_s = 10`
- `MultiServerMCPClient({"logistics": {"transport": "http", "url": ...}, "aftersales": {...}})`。
- 工具调用走 `tool.ainvoke(args)`；适配器每次新建会话，无需长连接管理。
- `mcp_enabled=False` 时 `get_all_specs()` 完全不发起网络请求，保证 663 个现有测试零网络、零 2 秒超时。
- 测试验证发现逻辑时显式打开并注入假 MCP 客户端；真实联调用 `.env` 打开。

## 11. API 与前端

- `ResumeRequest` 扩展：
  - `order_id: str | None`（订单选择器）
  - `confirmed: bool | None`（工单确认）
  - 二者恰好一个。
- `resume_chat_agent` 接受字符串或 `{"confirmed": bool}`；审计文本「[订单选择] 订单 X」「[工单确认] 确认提交/取消」。
- `/ch07/chat/resume` 复用；`require_pending_interrupt` 先检查 pending。
- interrupt 帧泛化透传 payload，前端按 `kind` 分发。
- 前端只改 `app/static/ch07.html`：
  - `select_order` → 现有订单卡不动。
  - `confirm_ticket` → 新预览卡（工单类型 + 问题描述 + 「确认提交」「取消」），样式对齐订单卡。
  - 确认 → `POST /ch07/chat/resume {conversation_id, confirmed: true}`；取消 → `confirmed: false`；都复用现有 SSE 读取/看门狗/错误处理。
  - 刷新丢预览卡，和订单选择器同待遇；不做历史回载。
- 前端按用户要求 Vibe Coding，不套 brainstorm/TDD/code review。

## 12. legacy `/chat`

- 只绑定 `query_order`、`query_product`、`query_faq`。
- 不绑定 `create_ticket`（无确认 UI）、不绑定控制工具、不绑定 MCP 工具。
- 工具执行走新引擎；`query_logistics` 下线后 legacy `/chat` 不再提供物流。这是有意的回归，记入 dev-notes。
- 旧测试中「legacy 聊天可调 create_ticket」的断言改为「不绑定写工具」。

## 13. 配置 / 依赖 / DB

```text
MCP_ENABLED=false
MCP_LOGISTICS_URL=http://127.0.0.1:8101/mcp
MCP_AFTERSALES_URL=http://127.0.0.1:8102/mcp
MCP_DISCOVERY_TIMEOUT_S=2
MCP_TOOL_TIMEOUT_S=10
TOOL_DEFAULT_TIMEOUT_S=5
TOOL_MAX_RETRIES=2
TOOL_AUDIT_RESULT_MAX_CHARS=500
MOCK_DELAY_SECONDS=0              # 物流 MCP Server 读
DEMO_TICKET_DELAY_SECONDS=0       # 主服务读；create_ticket 在 commit 之后 sleep
```

- 依赖：`mcp`、`langchain-mcp-adapters`、`jsonschema`，安装后锁版本写进 `requirements.txt`。
- DB：`sql/ch08-ddl.sql` 原样落附录 A；本地 MySQL 手动 apply；SQLite 测试由 `create_all` 建表。
- `DEMO_TICKET_DELAY_SECONDS` 的 sleep 放在 DB commit 之后：超时上报但工单只落一次，正好演示「超时未必没执行」且不重复执行。

## 14. 验收路径

1. **新工具只注册即可用**：`app/tools/builtin/promotions.py` 丢文件；启动后问优惠活动，Agent 能绑定并调用。
2. **MCP 物流**：启动两台 Server，问「订单 1001 的物流到哪了」，Agent 先 `query_order` 拿 `tracking_no`，再经 MCP `query_logistics` 返回轨迹并作答。售后工具由注册中心/集成测试证明。
3. **MCP 热加工具**：往物流 Server 加 `query_delivery_eta`，只重启 8101；不重启客服，下一轮问预计送达，Agent 能用新工具。
4. **建工单确认**：说「帮我建个工单」但没说问题 → 意图「人工」→ Agent → 缺描述先追问；补齐 → 预览卡 → 确认 → tickets 落一条 → 回复带工单号。
5. **取消**：预览卡点取消 → tickets 不新增 → 该 `create_ticket` 审计「权限拒绝」。
6. **超时与重试**：
   - 读：物流 Server 设 `MOCK_DELAY_SECONDS` 大于超时 → 重试、审计 `超时`、`retry_count` 有值、`duration_ms` 有值。
   - 写：主服务设 `DEMO_TICKET_DELAY_SECONDS` 大于超时 → 确认后超时，`retry_count=0`、审计「超时」，不产生第二张工单。

## 15. 测试策略

- 代码任务按 TDD。
- 单元：
  - registry：pkgutil 扫描、注册、重名丢弃、MCP 失败降级、每步现拉。
  - engine：校验拦下、权限拒绝、超时重试、写不重试、分诊、审计字段、审计失败不抛。
  - 审计：真测试库插入与回查。
  - 格式化：字段白名单、枚举翻译、`ensure_ascii=False`、`query_faq` 原文。
  - 意图：新增「人工」样例，`eval_intent` 18/18。
- 集成：
  - 图级 `tools_node` interrupt→resume 两分支（确认/取消），tickets 与审计断言。
  - 同批两个 `create_ticket` 的「校验拦下」。
  - 回复漏工单号的服务层兜底。
  - ch05/ch06 写路径 `confirmed=True` 后仍建单、失败仍 502。
  - 假 MCP 客户端两轮不同清单，验证现拉热替换。
- MCP：pytest fixture 子进程拉起两台 Server，客户端拉清单断言三要素并真调一次；若环境不允许起进程，保留函数级测试并手工演示。
- 全量回归：现有 663 个测试继续通过；预期更新工具清单、Agent 工具断言、prompt 断言、legacy 断言。
- 前端 Vibe Coding + 浏览器手工验收。

## 16. 清理与迁移

- 删除 `app/tools/business.py` 里的 `query_logistics` 实现，保留 mock 数据纯函数。
- 旧 `ALL_TOOLS` / `TOOLS_BY_NAME` / `TOOL_TIMEOUTS` 不再作为主入口；如保留，只做兼容薄壳。
- `app/tools/executor.py` 的旧签名退场，ch05/ch06 改走 engine。
- `app/agent/tools.py` 的控制工具迁移到 `app/tools/builtin/refunds.py` 或保留在 `app/tools/control.py`，以注册中心为准。
- 新审计表需 apply DDL；不做自动迁移。
- 第九类意图同步更新 `app/agent/state.py`、`app/agent/intent.py`、`app/agent/routing.py`、评估集、相关测试。

## 17. 交付物

- 本设计文档。
- 实施计划：`docs/superpowers/plans/2026-11-19-ecommerce-cs-ch08-tool-system.md`。
- 代码、`sql/ch08-ddl.sql`、`dev-notes/ch08.md`。
- 演示命令：两台 MCP Server、客服服务、验收 1–6。
- 测试结果、已知回归（legacy `/chat` 无物流、无建单；验收 2 对话只演示物流）。

## 18. 留给构建者的决定

- 文件拆分与模块命名在职责内可调整。
- 测试用假 MCP 客户端形态。
- 预览卡 CSS 与文案。
- 中文错误说明具体措辞（包含参数名和原因即可）。
- 进程级 MCP 测试是否起真实 Server，或降级为函数级 + 手工演示。

## 附录 A：`tool_audit_logs` DDL（用户给定，原样落 `sql/ch08-ddl.sql`）

```sql
-- =============================================================
-- ch08 · 工具系统 · 建表 DDL
-- 本章新建:tool_audit_logs(工具调用审计留痕,统一执行引擎每次调用落一条)
-- 不挂外键:审计写入不能被引用约束拦住,conversation_id 只建普通索引
-- 全库统一 ENGINE=InnoDB、CHARSET=utf8mb4
-- =============================================================

-- 确保中文 ENUM 定义值/DEFAULT/COMMENT 按 utf8mb4 解析
-- (否则 latin1 默认的 mysql client 会把中文 double-encode,ENUM 值存成乱码)
SET NAMES utf8mb4;

-- 工具调用审计:内置和 MCP 工具都记,被权限拒、被校验拦的调用同样落一条
CREATE TABLE tool_audit_logs (
  id              BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '审计主键',
  conversation_id BIGINT UNSIGNED NULL                    COMMENT '所属会话,无会话上下文的调用为 NULL',
  tool_call_id    VARCHAR(64)     NULL                    COMMENT '模型申请单 id,可对回 messages 流水',
  tool_name       VARCHAR(128)    NOT NULL                COMMENT '工具名',
  tool_source     ENUM('builtin','mcp') NOT NULL          COMMENT '工具来源:内置 / MCP 接入',
  mcp_server      VARCHAR(64)     NULL                    COMMENT '来源 MCP Server 名,内置工具为 NULL',
  arguments       JSON            NULL                    COMMENT '调用参数',
  result_summary  TEXT            NULL                    COMMENT '返回结果,过长截断存摘要',
  status          ENUM('成功','失败','超时','校验拦下','权限拒绝') NOT NULL COMMENT '本次调用结局',
  error_message   VARCHAR(512)    NULL                    COMMENT '失败 / 拦下时的原因说明',
  retry_count     TINYINT UNSIGNED NOT NULL DEFAULT 0     COMMENT '实际重试次数,写操作默认不重试恒为 0',
  duration_ms     INT UNSIGNED    NULL                    COMMENT '耗时毫秒',
  created_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '调用时间',
  PRIMARY KEY (id),
  KEY idx_conversation_id (conversation_id),
  KEY idx_tool_name (tool_name),
  KEY idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='工具调用审计留痕';
```

ORM 映射：

- `id` 用 `BigInteger` + `autoincrement=True`（MySQL 端实际是 `BIGINT UNSIGNED`）。
- `arguments` 用 SQLAlchemy `JSON`；MySQL 原生 JSON，SQLite 存 TEXT。
- `status` 用 `Enum("成功","失败","超时","校验拦下","权限拒绝", name="tool_audit_status")`。
- `tool_source` 用 `Enum("builtin","mcp", name="tool_source")`。
- `created_at` 用 `server_default=func.now()`。
- 不声明 `ForeignKey`。
