# 电商智能客服系统 · 第二章:工具调用(查数据 + 建工单)— 设计文档

日期:2026-11-12
状态:待批准(已过 builder 预审 / 待用户批准)

## Intent & Why

第一章已跑通纯对话闭环:SSE 流式对话 + 结构化提取 + 内存多轮上下文,但客服只能"说",不能"查"。真实客服的价值在于能查订单、查物流、查 FAQ、把解决不了的单子转人工。

第二章给客服装上"手":用 Function Calling 让模型自己决定调哪个业务工具,工具链长在客服聊天本身——用户在聊天页问一句"订单 1001 的物流到哪了",模型选中 `query_logistics` 工具、执行、把结果喂回模型、最终答案仍逐 token 流式吐给用户。同时把会话与消息从内存搬到 MySQL,打通持久化这条线。

为什么先做单轮工具调用:它是 Agent 的地基。单轮(模型调一次工具就收敛)把"模型定工具 → 执行 → 回灌 → 组织回答"这条主链路跑通并验收后,多轮 Agent Loop 只是在其上再加一层循环控制。

## Goals

1. **持久化基座**:FastAPI + SQLAlchemy 分层骨架;Docker 起 MySQL;建 `faq` / `conversations` / `messages` / `tickets` 四张表并灌测试数据。
2. **五个业务工具**(LangChain `@tool` 装饰器):
   - `query_order` 查订单、`query_product` 查商品、`query_logistics` 查物流 —— 真实场景应接电商/物流系统 API,演示起见内部随机生成数据返回,不接真实接口、不建表。
   - `query_faq` 查常见问题 —— SQL `LIKE` 关键词查 `faq` 表。
   - `create_ticket` 创建人工工单 —— 写 `tickets` 表。
3. **工具基础设施**:注册管理、参数 Schema 校验、执行错误处理、超时重试;工具结果回灌给模型组织回答。
4. **接入现有 `/chat` SSE 入口**:用户问一句 → 后端走「模型定工具 → 执行 → 回灌收敛」;最终回答仍逐 token 流式吐出,工具执行那一段先推一个状态帧;聊天记录(含工具调用与结果)落 `conversations`/`messages` 表;聊天页气泡显示这轮调了哪个工具(工具轨迹小徽章)。
5. **只做单轮调用**:模型调一次工具就收敛。

## Non-Goals(本章不做)

- 多轮自动循环的 Agent Loop。
- 向量检索、RAG。
- 认证、鉴权、限流、计费(同第一章)。
- 真实的订单/商品/物流 API 对接(演示随机生成)。
- **历史回放**:前端刷新页面后不回放 DB 里的历史消息(用户已拍板 scope A);DB 只保证服务端上下文连续性,前端仍像 ch01 在本地 React state 保留当前页对话。

## Anti-Goals(技术上能跑通也算失败)

- 工具定义散落、各自为政 —— 必须集中注册、统一执行入口。
- 工具执行无错误兜底 —— 工具抛异常/超时不能把整个流式对话搞挂,必须转成可回灌的错误结果。
- 模型地址/模型名/密钥、数据库连接串硬编码进代码 —— 全走 `.env`。
- 会话/消息继续依赖内存 —— 本章起 MySQL 是唯一真相源,服务重启上下文不丢。
- 工具结果不落库 —— 工具调用申请(`tool_calls`)与结果(`tool` 消息)必须写进 `messages` 表。

## Constraints(定死,不可自行更改)

- 技术栈:**Python + FastAPI + SQLAlchemy + MySQL(Docker 起)+ LangChain(`@tool`)**。
- 模型接入:沿用第一章的 OpenAI 协议直连 DeepSeek,地址/模型名/密钥全在 `.env`。
- 工具定义用 LangChain `@tool` 装饰器,五个工具的名称与用途如上,不可增删改用途。
- 单轮调用:模型调一次工具就收敛,不引入 Agent Loop。
- 工具执行那一段先推状态帧,最终回答仍 SSE 逐 token 流式。
- **四张表结构以用户提供的 DDL 为准(见下),不自行增删列。**

## 已确认的决策(来自与用户的对话)

- **MySQL 取代内存 SessionStore,成为会话与消息的唯一来源**:每轮从 `conversations`/`messages` 读历史组上下文;`SessionStore` 移除。用户拍板选 A。
- **历史回放 scope = A**:只做服务端上下文连续性,本章不加 GET 历史接口、前端不做刷新回放。
- 技术选型定死,若 Function Calling 走不通停下再问用户(经 Context7 预验证可行,见下)。
- 聊天页改造走 Vibe Coding,不套 brainstorm / TDD / code review 流程。
- **用户提供了权威建表 DDL**(faq / conversations / messages / tickets),覆盖我此前 draft 的 schema my-call。

## 关键可行性预验证(本章设计定稿前已做)

- **DeepSeek Function Calling 可用**:官方文档(api-docs.deepseek.com/guides/tool_calls)给出 OpenAI 协议 `tools` 参数 + `tool` 角色消息(`tool_call_id`)回环的完整示例,`deepseek-chat`/`deepseek-flash` 均支持;普通(非 strict)function calling 即可用,不要求 strict/beta 端点。
- **LangChain 工具调用链路**:`@tool` 装饰器 + `model.bind_tools(tools)` → `AIMessage.tool_calls`(含 `name`/`args`/`id`)→ 执行 → `ToolMessage(content, tool_call_id, name)` 回灌 → 再 `model.invoke/stream` 出最终答案。与设计一致。
- 实现阶段会再按 Context7 复核 `ChatOpenAI.bind_tools` 对 DeepSeek 的具体行为(如是否需 `extra_body={"thinking": {"type": "disabled"}}`),以实测跑通为准。

## 表结构(用户提供,权威)

```sql
-- 会话壳
CREATE TABLE conversations (
  id          BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '会话主键',
  user_id     VARCHAR(64)     NOT NULL                COMMENT '用户标识',
  status      ENUM('进行中','已转人工','已结束') NOT NULL DEFAULT '进行中' COMMENT '处理状态',
  created_at  DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '开启时间',
  updated_at  DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (id),
  KEY idx_user_id (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='客服会话';

-- 消息流水
CREATE TABLE messages (
  id              BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '消息主键',
  conversation_id BIGINT UNSIGNED NOT NULL                COMMENT '所属会话',
  role            ENUM('user','assistant','tool') NOT NULL COMMENT '角色:用户/助手/工具结果',
  content         TEXT            NULL                     COMMENT '消息正文,assistant 纯工具调用时可为空',
  tool_calls      JSON            NULL                     COMMENT 'assistant 消息带的工具调用申请单',
  tool_call_id    VARCHAR(64)     NULL                     COMMENT 'tool 消息对应的申请单 id,回灌时对号入座',
  created_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '产生时间',
  PRIMARY KEY (id),
  KEY idx_conversation_id (conversation_id),
  CONSTRAINT fk_messages_conversation FOREIGN KEY (conversation_id) REFERENCES conversations (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='会话消息流水';

-- FAQ
CREATE TABLE faq (
  id          BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT 'FAQ 主键',
  question    VARCHAR(512)    NOT NULL                COMMENT '问题',
  answer      TEXT            NOT NULL                COMMENT '答案',
  category    VARCHAR(64)     NOT NULL                COMMENT '分类',
  created_at  DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  updated_at  DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (id),
  KEY idx_category (category)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='常见问答';

-- 人工工单
CREATE TABLE tickets (
  ticket_no       VARCHAR(32)     NOT NULL                COMMENT '工单号,如 T20260701008',
  conversation_id BIGINT UNSIGNED NOT NULL                COMMENT '关联会话,可倒查当时聊了什么',
  description     TEXT            NOT NULL                COMMENT '问题描述',
  ticket_type     ENUM('售后','投诉','咨询') NOT NULL     COMMENT '工单类型',
  status          ENUM('待处理','已处理') NOT NULL DEFAULT '待处理' COMMENT '处理状态',
  created_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  PRIMARY KEY (ticket_no),
  KEY idx_conversation_id (conversation_id),
  CONSTRAINT fk_tickets_conversation FOREIGN KEY (conversation_id) REFERENCES conversations (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='人工工单';
```

## 我替用户做的决定(标记为 my call,用户可在审阅时推翻)

1. **MySQL 实例**:Docker Compose 起 `mysql:8.0`,库名 `cs_customer`,字符集 `utf8mb4`,端口 `3306`,root 口令与库名全走 `.env`(附 `.env.example`),数据用 named volume 持久化;同步驱动 `pymysql`(新增依赖)。建表用 `metadata.create_all` + seed 脚本,不引 Alembic。
2. **session_id → user_id 映射**:客户端继续发 `session_id`(ch01 契约不变);后端把它当作 `user_id` 存进 `conversations.user_id`(无鉴权 demo 里,per-tab UUID 即匿名用户标识)。`/chat` 每轮按 `user_id` 找「最近一条 `status='进行中'` 的会话」,找不到则新建;找到的会话 `id`(bigint)作为本轮上下文与落库的 `conversation_id`。
3. **会话生命周期**:`create_ticket` 把当前 `conversation.status` 置为 `已转人工`;之后同一 `user_id` 再发消息,因无 `进行中` 会话而自动开新会话(不翻回旧会话、不拒绝)。`已结束` 状态本章 bot 流程不主动置,仅保留给后续/人工侧。
4. **工单号生成**:`ticket_no` = `T` + `YYYYMMDD` + 当日三位递增序号(如 `T20260701008`),实现为「当日已有工单数 + 1」并加锁/重试防撞;格式对齐用户 DDL 示例。
5. **五个工具签名与 Schema**(LangChain `@tool` 由类型注解 + docstring 生成参数 Schema):
   - `query_order(order_id: str) -> str`:按订单号查订单,返回 JSON(订单号/商品/金额/状态等随机生成)。
   - `query_product(product_name: str) -> str`:按商品名查商品,返回 JSON(名称/价格/库存/规格等随机生成)。
   - `query_logistics(order_id: str) -> str`:按订单号查物流,返回 JSON(物流单号/承运商/当前节点/轨迹等随机生成)。
   - `query_faq(keyword: str) -> str`:用 `LIKE %keyword%` 查 `faq.question`(或 `answer`),命中返回问答,未命中返回"未找到"。
   - `create_ticket(description: str, ticket_type: Literal["售后","投诉","咨询"]) -> str`:写 `tickets` 表(`ticket_no` 自动生成、`status` 默认 `待处理`、`conversation_id` 经 contextvar 注入),返回工单号;`ticket_type` 用 `Literal` 约束为三值枚举。
   - 五个工具返回类型统一为 `str`(JSON 或人类可读文本),便于 `ToolMessage` 回灌。
6. **工具执行基础设施**:`tools/executor.py` 提供统一 `execute_tool(name, args, timeout_s=5, retries=1) -> str`——先用 `@tool` 生成的 Pydantic Schema 校验参数(非法参数直接返回错误串给模型),执行包 `try/except` + `threading` 超时 + 失败重试 1 次,任何失败都转成"工具执行失败"字符串回灌,不让流挂掉。
7. **`/chat` 单轮工具编排流程**:
   1. 落用户消息(`messages.role=user`),按 user_id 找/建 `conversations`,把 `conversation.id` 写入 contextvar。
   2. 从 DB 读该会话历史,组上下文(沿用 `build_messages` 的裁剪,历史来源改为 DB)。
   3. `model.bind_tools(ALL_TOOLS).invoke(messages)` 判断:无 `tool_calls` → 直接走 ch01 的流式回答并落库;有 `tool_calls` → 落 assistant 的 `tool_calls` 消息 → 推状态帧 → 逐个执行工具(单轮,不回环)→ 落 `tool` 结果消息 → `ToolMessage` 回灌 → 再 `model.stream(full_messages)` 逐 token 流式出最终答案并落库。
   4. **单轮硬约束**:最终回答那次调用若又吐 `tool_calls`,只取文本 `content` 忽略新工具调用;若 `content` 为空只剩工具调用,推兜底文案("抱歉,暂时无法完成查询,已为您转人工")。
8. **上下文重建与裁剪的工具轮原子性**:从 `messages` 重建 LangChain 消息时,把「assistant 的 `tool_calls` + 对应的 `tool` 结果」当作原子块;`build_messages` 裁剪时整块保留/整块丢弃,`content` 为 NULL 按 0 计 token。这样不拆散 tool 对(避免 OpenAI 协议 400),下一轮模型也能引用"刚才查到的物流"。
9. **SSE 状态帧契约**:工具执行前推一个 JSON 帧 `{"type":"tool","tool":"<tool_name>","label":"<中文名>"}`(label 映射:`query_order`查订单、`query_product`查商品、`query_logistics`查物流、`query_faq`查常见问题、`create_ticket`创建工单);前端据此在气泡渲染工具徽章。最终答案仍按 ch01 的裸 token 帧逐字推送。
10. **System Prompt 增量**:在 ch01 客服角色/约束基础上,追加"可用的查询能力与建工单能力、何时该用、查到结果如何礼貌转述、查不到时如实告知并引导转人工"的行为约束;措辞由实现者起草。
11. **测试数据(faq 种子)**:灌入含"退货政策/换货/退款/物流时效"等条目;刻意**不含"邮费"关键词**,以支撑验收第 3 条"邮费是多少"的预期漏召回(留作下一步升级)。

## 验收标准(来自用户,不可改动)

1. 浏览器打开聊天页,问「订单 1001 的物流到哪了」,能看到模型选中工具(气泡带工具徽章)并按返回结果作答。
2. 问「退货政策是什么」,`query_faq` 查得到并作答。
3. 换个说法问「邮费是多少」,确认关键词查表查不出来——这个漏召回是预期结果,记下来留给下一步升级。

## 留给 builder 的

- 具体类/函数签名、依赖版本、SQLAlchemy 2.0 声明式模型映射(含 ENUM/JSON/BigInteger 的映射方式)、seed 脚本与 Docker Compose 细节、测试组织、异常处理细节、实现顺序。
- 上述"my call"标记项的具体落地,除非用户推翻。
