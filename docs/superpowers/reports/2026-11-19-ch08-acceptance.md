# Ch08 工具系统 — 验收记录（2026-11-19）

- 分支：`feature/ch08-tool-system`
- 设计：`docs/superpowers/specs/2026-11-19-ecommerce-cs-ch08-tool-system-design.md`
- 计划：`docs/superpowers/plans/2026-11-19-ecommerce-cs-ch08-tool-system.md`
- DDL：`sql/ch08-ddl.sql`

## 自动化测试

```bash
MAX_AGENT_STEPS=6 TOOL_RESULT_MAX_TOKENS=2000 RERANK_TOP_K= \
MODEL_CONTEXT_WINDOW=65536 MAX_OUTPUT_TOKENS=2000 MAX_USER_INPUT_TOKENS=2000 \
.venv/Scripts/python.exe -m pytest -q
```

结果：**772 passed, 1 warning**（既有 SQLAlchemy LegacyAPIWarning）。

意图评估：

```bash
.venv/Scripts/python.exe -m app.jobs.eval_intent
```

结果：64 例，valid 64/64，correct 63，accuracy 0.984，errors 0，人工 2/2 = 1.000。
报告：`docs/superpowers/reports/2026-11-19-ch08-intent-eval.md`。

## 验收 1：新工具只注册即可用

- 新工具：`app/tools/builtin/promotions.py` 的 `query_promotions`，仅注册，不改核心注册表。
- 实测对话：`订单 1001 能参加什么优惠活动`（business 路由，避免知识闸拦截）。
- SSE 工具帧：`query_order`、`query_promotions`（label `查优惠`）。
- 回答使用了活动名称、门槛、有效期，并说明订单 299 元未达满 300 减 50。

## 验收 2：MCP 物流经 Agent 查询

- 两台 Server 独立进程：8101 logistics、8102 aftersales。
- `fetch_mcp_specs` 返回：`query_logistics`、`query_warranty`、`query_return_status`。
- 实测对话（MCP_ENABLED=true 的 8011 实例）：`订单 1001 的物流到哪了`。
- SSE 工具帧：`query_order` → `query_logistics`。
- 回答包含顺丰速运、运单号、已揽收/运输中轨迹。
- 直接调用三个 MCP 工具的格式化结果均把内部枚举码翻成中文（含 `traces[].status_code`）。
- SQLite 审计行：三条均 `tool_source=mcp`、`status=成功`。

## 验收 3：MCP Server 热加工具

- 临时在 `logistics_server.py` 增加 `query_delivery_eta`，只重启 8101；客服服务（8011）不重启。
- `fetch_mcp_specs` 下一轮返回：`query_delivery_eta`、`query_logistics`、`query_return_status`、`query_warranty`。
- 实测对话：`订单 1001 预计什么时候送达`；SSE 工具帧 `query_order` → `query_delivery_eta`，回答使用预计送达日期/时段。
- 验证后已还原 `logistics_server.py`，未提交演示工具。

## 验收 4：建工单确认流

真实 LLM + SQLite 应用实例（8010，MCP_ENABLED=false）：

1. 用户：`帮我建个工单`
   - 回复追问工单类型与问题描述，未建单。
2. 用户：`猫砂盆漏电了,帮我建个工单跟进`
   - SSE `interrupt`：`kind=confirm_ticket`，preview `ticket_type=售后`，description 含漏电问题。
3. 前端 resume：`confirmed=true`
   - SSE 工具帧 `create_ticket`。
   - 最终回答包含 `工单号：T20261119001`。
   - SQLite `tickets`：一条 `T20261119001`。

## 验收 5：取消

同一流程，resume `confirmed=false`：

- 回答：本次建单已取消，不再重复发起。
- `tickets` 未新增（全库仍只有 T20261119001）。
- `tool_audit_logs`：`create_ticket` / `权限拒绝` / error_message `用户取消了本次建工单,请不要再发起。` / retry_count 0。

## 验收 6：超时与重试

脚本：`.superpowers/sdd/.../accept_timeouts.py`（SQLite 审计）。

- 只读慢工具：`status=timeout`，`retry_count=1`；审计 `超时`，`duration_ms=222`。
- 写 `create_ticket`（`DEMO_TICKET_DELAY_SECONDS=6`，timeout=1s）：`status=timeout`，`retry_count=0`；审计 `超时`，`duration_ms=1013`。
- 写超时后 `tickets` 只有一条，未重复执行。
- 另：MySQL 未启动时审计写入失败只记日志，工具结果仍返回成功，验证「审计失败不反拦执行」。

## 环境限制与待用户确认

- 本机 MySQL 服务未运行，且 `net start MySQL` 返回「拒绝访问」（系统错误 5），因此：
  - `sql/ch08-ddl.sql` 未在本机 MySQL 实际 apply。
  - 验收 4/5 的实时 HTTP 流程使用 SQLite 应用实例验证；生产 MySQL 需用户启动服务后执行 DDL。
- 浏览器工单预览卡的视觉效果未做自动化截图验收；后端 SSE 契约与前端代码已就绪，需用户在 `http://localhost:8000/` 手工点一遍。
- 已知有意回归：legacy `/chat` 不再提供物流与建单，只保留 `query_order`/`query_product`/`query_faq`。

## 演示命令

```bash
# 1) 依赖与 DDL（MySQL 可用时）
mysql -uroot -proot cs_customer < sql/ch08-ddl.sql

# 2) 两台 MCP Server（独立进程）
python -m mcp_servers.logistics_server     # 8101
python -m mcp_servers.aftersales_server    # 8102
# 演示 6 的读超时：MOCK_DELAY_SECONDS=6 python -m mcp_servers.logistics_server

# 3) 客服服务（.env 设 MCP_ENABLED=true）
python run.py
# 浏览器打开 http://localhost:8000/
#   验收 1：订单 1001 能参加什么优惠活动
#   验收 2：订单 1001 的物流到哪了
#   验收 3：往 logistics_server.py 加 query_delivery_eta，仅重启 8101，再问预计送达
#   验收 4：帮我建个工单 → 补充描述 → 预览卡确认提交
#   验收 5：重复到预览卡 → 取消 → 查 tool_audit_logs 权限拒绝
```
