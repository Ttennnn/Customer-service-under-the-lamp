# ch05 Agent 工作流验收报告

- 执行日期：2026-11-17
- 分支：`feature/ch05-agent-workflow`
- 起点提交：`c47c79a`（docs: record ch05 task 12 notes）
- 运行环境：Windows 11 + Git Bash；`.venv`（Python 3.13.13）；MySQL `cs_customer`、Milvus `knowledge_hybrid`（136 行）均在线
- 服务端口：`8000`（启动前确认空闲）
- 验收会话：全部使用 fresh session id（`acc-ch05-20261117-2224-*` 或 Playwright 新 context 生成的 UUID），避免与历史 SQLite checkpoint / MySQL 会话串线

## 启动方式

```bash
PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 .venv/Scripts/python.exe -c "import logging; logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s'); import uvicorn; uvicorn.run('app.main:app', host='127.0.0.1', port=8000, reload=False)" > .superpowers/sdd/2026-11-17-ecommerce-cs-ch05-agent-workflow/ch05-acceptance-server.log 2>&1
```

说明：仓库没有配置应用 logger 的 handler，直接 `run.py` 时 `app.*` 的 INFO 日志不会落盘。本次用一次性 `logging.basicConfig(level=INFO)` 启动（不改仓库代码），日志写入 `.superpowers/.../ch05-acceptance-server.log`。`GET /ch05` 返回 200 后开始验收，结束时已终止服务进程。

## 旧 `/chat` 回归

请求（Windows shell 下为避免中文内联编码问题，body 写入 UTF-8 文件后 `--data-binary @file`）：

```bash
curl -sS -N --max-time 180 -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json; charset=utf-8' \
  --data-binary @.superpowers/sdd/2026-11-17-ecommerce-cs-ch05-agent-workflow/old-chat-request.json
# body: {"session_id":"old-chat-check-acc13","message":"你好"}
```

实际结果：HTTP 200；共 40 个 SSE data 帧，最后帧为 `data: [DONE]`。拼接后的回答：

> 您好，很高兴为您服务。请问有什么售后问题需要帮您处理？比如订单查询、退换货、退款、物流或维修等，您说下具体情况，我来协助您。

日志含 `POST /chat HTTP/1.1" 200 OK`，无异常栈。**旧 `/chat` 无回归。**

## 验收 1：知识类问题强制检索

请求：

```bash
curl -sS -N --max-time 180 -X POST http://127.0.0.1:8000/ch05/chat \
  -H 'Content-Type: application/json; charset=utf-8' \
  --data-binary @.superpowers/sdd/2026-11-17-ecommerce-cs-ch05-agent-workflow/acc1-request.json
# {"session_id":"acc-ch05-20261117-2224-pol","message":"你们的退货政策是什么？"}
```

SSE 实际证据：

- 第一帧为 `sources`，共 10 条；首条 `section_path=灯下商品与购物 FAQ > 退货政策是什么`，`score=0.9883095026016235`；
- 回答 `answer_chars=583`，引用编号覆盖 `[1]`–`[9]`，例如：
  > 您好，本店退货政策如下：**7天无理由退货** -自签收之日起7天内，商品完好、不影响二次销售的，支持无理由退货；定制类、生鲜类商品除外[1][3]。……电子类商品自签收起保修12个月，保修期内非人为损坏可申请免费维修或换货[2]。……
- 回答内容与 sources 中的证据一致。

服务端日志：

```text
2026-11-17 22:23:57,302 INFO app.agent.graph ch05 retrieval_forced intent=退款退货 hits=10 top_score=0.9883095026016235 rerank_failed=False
2026-11-17 22:23:59,294 INFO app.agent.graph ch05 turn_log {"thread_id": "user:acc-ch05-20261117-2224-pol", "conversation_id": 28, "intent": "退款退货", "route": "knowledge", "gate_passed": true, "gate_reason": "ok", "confidence": 0.9883095026016235, "retrieval_hits": 10, "rerank_failed": false, "tool_steps": 1, "token_estimate": 2402, "suggested_actions": [], "answer_chars": 583}
```

结论：**PASS**。强制检索日志存在；置信度闸通过由 `turn_log` 的 `gate_passed=true / gate_reason=ok` 以及只有 `confidence_gate` 节点才会发出的 `sources` 帧共同证明。

口径说明：代码中**不存在**字面量 `ch05 confidence_gate` 日志（`confidence_gate_node` 没有 logger 调用）。brief 期望的该字符串无法命中，本报告以等价的闸门状态字段 + `sources` 帧作为证据，未修改生产日志代码。

## 验收 2：订单物流工具调用

请求：

```bash
curl -sS -N --max-time 180 -X POST http://127.0.0.1:8000/ch05/chat \
  -H 'Content-Type: application/json; charset=utf-8' \
  --data-binary @.superpowers/sdd/2026-11-17-ecommerce-cs-ch05-agent-workflow/acc2-request.json
# {"session_id":"acc-ch05-20261117-2224-logi","message":"订单 1001 的物流到哪了"}
```

SSE 首帧：

```json
{"type":"tool","tool":"query_logistics","label":"查物流"}
```

回答（基于工具返回的物流单号/承运商/当前节点/轨迹，无 `[n]`，符合“工具事实不带引用”的规则）：

> 您好，订单1001的物流信息如下：-物流单号：SF1912454366 -承运商：韵达快递 -当前节点：派送中……物流轨迹：1.已揽收 2.到达转运中心 3.派送中……具体送达时间我这边无法确认，也不做时效承诺，请以承运商实际派送为准。

服务端日志：

```text
2026-11-17 22:24:09,218 INFO app.agent.graph ch05 turn_log {"thread_id": "user:acc-ch05-20261117-2224-logi", "conversation_id": 29, "intent": "物流", "route": "business", "gate_passed": false, "gate_reason": "", "confidence": null, "retrieval_hits": 0, "rerank_failed": false, "tool_steps": 2, "token_estimate": 2581, "suggested_actions": [], "answer_chars": 231}
```

结论：**PASS**。出现 `query_logistics` 工具帧，回答基于工具结果，`tool_steps=2`（工具轮 + 最终回答轮）。

## 验收 3：投诉按钮、转人工与建工单

工具：Playwright for Python（仅安装到 `.venv`，未写入 `requirements.txt`），`channel="msedge"` headless，真实点击页面按钮并用 `page.on("request")` 记录网络请求。脚本：`.superpowers/sdd/2026-11-17-ecommerce-cs-ch05-agent-workflow/acc3_playwright.py`。

每次场景使用独立 browser context（独立 `sessionStorage`）。网页发送“我要投诉”后均渲染出两个按钮：`["转人工", "建工单"]`，投诉固定话术正常。

### 3a 转人工

- 点击「转人工」后页面依次出现：`已转接人工客服`、`您好，我是客服小猫，请问有什么可以帮您的？`；
- 点击前后请求数 `2 -> 2`，新增请求 `[]`（没有任何 `/ch05/tickets` 或其它后端请求）；
- 该会话 `f249e3fc-1a99-4525-8baa-fc0b9bcfe824`（conversation 34）tickets `0 -> 0`，`conversations.status='进行中'`；
- 服务端日志中该 session 只有一条投诉 turn_log，无工单接口访问。

### 3b 建工单

- 点击「建工单」弹出 `confirm`：`确认创建工单吗？`，脚本接受；
- 页面显示 `工单已创建，工单号：T20261117001`，按钮变为 `已建工单`；
- 该会话 `ea084dbc-c244-4d93-9d54-cc0982899ecb`（conversation 35）tickets `0 -> 1`；
- DB 行：`T20261117001 | 投诉 | 待处理 | 我要投诉 | 进行中`；
- 会话状态点击前后均为 `35  进行中`；
- 服务端日志只有一条 `POST /ch05/tickets HTTP/1.1" 200 OK`，来自本场景。

### 3c 不点按钮继续聊天

- 会话 `ac3c35f7-87f9-43db-bc8f-6b1e80d647aa`（conversation 36）在投诉后不点任何按钮，继续发送“好的，那我先看看，谢谢”；
- 页面消息：user 2 条、assistant 2 条；无 `已转接人工客服`、无 `工单已创建`；
- 无 `/ch05/tickets` 请求；tickets `0`；`conversations.status='进行中'`；
- 服务端日志该 session 有两条 turn_log：第一条 `intent=投诉 route=complaint`，第二条 `intent=闲聊 route=chitchat`，说明会话正常继续且没有触发任何动作。

### DB 汇总

```text
id  user_id                                     status  tickets messages
34  f249e3fc-1a99-4525-8baa-fc0b9bcfe824        进行中       0        2
35  ea084dbc-c244-4d93-9d54-cc0982899ecb        进行中       1        2
36  ac3c35f7-87f9-43db-bc8f-6b1e80d647aa        进行中       0        4
TICKETS  T20261117001  投诉  待处理  我要投诉
```

结论：**PASS**（三个子场景全部满足）。

截图（本地证据，位于 `.superpowers/sdd/2026-11-17-ecommerce-cs-ch05-agent-workflow/`；按提交约束未提交）：

- `acc3-1-buttons.png`：两个按钮渲染
- `acc3-2-transferred.png`：已转接人工客服 + 客服小猫问候
- `acc3-3-ticket-before.png`：确认前
- `acc3-4-ticket-created.png`：工单号 + 已建工单
- `acc3-5-ignored-continue.png`：忽略按钮后继续对话

## 验收 4：闲聊固定话术

请求：

```bash
curl -sS -N --max-time 120 -X POST http://127.0.0.1:8000/ch05/chat \
  -H 'Content-Type: application/json; charset=utf-8' \
  --data-binary @.superpowers/sdd/2026-11-17-ecommerce-cs-ch05-agent-workflow/acc4-request.json
# {"session_id":"acc-ch05-20261117-2224-chit","message":"你好"}
```

SSE 只有 2 帧：

```text
data: 您好，我是客服小猫，很高兴为您服务。请问有什么可以帮您的？
data: [DONE]
```

服务端日志：

```text
2026-11-17 22:24:15,092 INFO app.agent.graph ch05 turn_log {"thread_id": "user:acc-ch05-20261117-2224-chit", "conversation_id": 30, "intent": "闲聊", "route": "chitchat", "gate_passed": false, "gate_reason": "", "confidence": null, "retrieval_hits": 0, "rerank_failed": false, "tool_steps": 0, "token_estimate": 0, "suggested_actions": [], "answer_chars": 29}
```

该 session 无 `retrieval_forced` 日志；`route=chitchat`、`retrieval_hits=0`、`tool_steps=0`、`token_estimate=0`，即未进入检索/Agent。结论：**PASS**。

## 验收 5：多步 ReAct

请求：

```bash
# {"session_id":"acc-ch05-20261117-2224-multi","message":"先帮我确认订单 1001 的状态，如果已经发货，再查它的物流到哪了"}
curl -sS -N --max-time 240 -X POST http://127.0.0.1:8000/ch05/chat \
  -H 'Content-Type: application/json; charset=utf-8' \
  --data-binary @.superpowers/sdd/2026-11-17-ecommerce-cs-ch05-agent-workflow/acc5-request.json
```

第一次运行（conversation 31）：`query_order` 返回随机状态「待付款」，模型正确地只查订单并说明“尚未付款、也未发货，暂时无法查询物流”。工具帧只有 `query_order`，`tool_steps=2`。这是符合业务条件的正确分支，但没有覆盖“两个工具串行”的验收目标。

重试一次（fresh session `acc-ch05-20261117-2224-multi-r2`，conversation 32）：`query_order` 返回「运输中」，模型随后调用 `query_logistics`。SSE 工具帧按顺序为：

```json
{"type":"tool","tool":"query_order","label":"查订单"}
{"type":"tool","tool":"query_logistics","label":"查物流"}
```

回答同时包含订单状态与物流单号/承运商/轨迹，`tool_steps=3`：

```text
2026-11-17 22:24:30,334 INFO app.agent.graph ch05 turn_log {"thread_id": "user:acc-ch05-20261117-2224-multi-r2", "conversation_id": 32, "intent": "订单", "route": "business", ..., "tool_steps": 3, "token_estimate": 4192, ..., "answer_chars": 310}
```

结论：**PASS**（重试 1 次后满足“query_order → query_logistics”至少两步）。未出现并行调用，因此**没有修改 Agent prompt**。

## 全量测试

```bash
.venv/Scripts/python.exe -m pytest -q
```

实际结果：

```text
460 passed, 1 warning in 16.98s
```

唯一 warning 是既有 `tests/test_tools_ticket.py::test_create_ticket_writes_only_ticket` 的 SQLAlchemy `Query.get()` LegacyAPIWarning，与本任务无关。

## 数据库最终状态

```text
LOWCONF 0   -- 所有验收会话均未落低置信度池（验收 1 闸门通过，符合预期）
```

各验收会话 `conversations.status` 均为 `进行中`；仅建工单场景产生 1 条 ticket。`low_confidence_questions.source` ENUM 已确认包含 `confidence_gate`（本次未触发写入）。

## 偏差与残留限制

1. **日志字面量差异**：brief 要求日志包含 `ch05 confidence_gate`，但代码从未输出该字符串。闸门执行改用 `ch05 turn_log` 的 `gate_passed/gate_reason` 与 `sources` 帧证明；未为验收去改生产日志。
2. **Mock 业务数据随机**：`query_order` 的状态/金额/商品为随机值，验收 5 第一次运行命中「待付款」，因此重试一次才得到“已发货”分支。两次行为都正确；报告保留两次结果，不把第一次算作失败。
3. **浏览器范围**：Playwright 使用系统 Edge headless、1000×900 桌面视口；未重新覆盖移动端布局和真实人工客服系统（转人工本来就是前端模拟）。
4. **截图与日志未提交**：按提交约束，截图保存在 `.superpowers/...`、服务日志为临时证据；仓库提交只包含本报告与 `dev-notes/ch05.md`。
5. **Windows 编码**：中文请求体统一写入 UTF-8 文件后用 `--data-binary @file`。Task 12 曾记录内联中文在 Windows shell 下触发 400，本次直接规避；属命令行环境问题，不是应用缺陷。
6. Playwright 仅安装到本地 `.venv`，未修改 `requirements.txt`；验收结束后服务进程已终止。

## 结论

| 项目 | 结果 |
| --- | --- |
| 旧 `/chat` 回归 | PASS |
| 验收 1 知识类强制检索 + 引用 | PASS（`ch05 confidence_gate` 字面日志缺失，已用等价闸门证据说明） |
| 验收 2 物流工具调用 | PASS |
| 验收 3 按钮/转人工/建工单/忽略按钮 | PASS |
| 验收 4 闲聊固定话术 | PASS |
| 验收 5 多步 ReAct | PASS（重试 1 次，串行两工具） |
| 全量 pytest | 460 passed, 1 warning |
