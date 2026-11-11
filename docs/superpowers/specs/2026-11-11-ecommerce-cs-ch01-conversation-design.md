# 电商智能客服系统 · 第一章:纯对话闭环 — 设计文档

日期:2026-11-11
状态:待批准(已通过 builder 预审 / 待用户批准)

## Intent & Why

目标是做一个电商智能客服系统。整个系统最终会走向工具调用、Agent 循环、RAG 等能力,但**第一章只做纯对话闭环**,先把架构骨架和"应用侧统一 OpenAI 协议、可换模型"这条主线跑通并验收,后续章节在此之上叠加。

为什么先做纯对话:对话是所有能力的地基。逐 token 流式、多轮上下文、结构化输出这三块一旦跑通并验证,工具调用和 Agent 只是在其上增加一层决策与动作。

## Goals

1. **对话接口**:`POST /chat`,支持多轮对话,SSE 流式逐 token 推送回复。
2. **Prompt 管理**:客服角色设定 + 行为约束放在 `PromptTemplate` 模板里,与业务代码分离。
3. **结构化输出**:独立接口 `POST /extract`,把一段售后描述提取为 `{订单号, 诉求类型, 期望方案}` 的 JSON,用 `with_structured_output` 实现。
4. **多轮上下文(最简版)**:带进模型前做历史消息裁剪 + token 预算控制,不做向量库/记忆库。

## Non-Goals(本章不做)

- 工具调用、Agent 循环。
- 聊天页面(后续章节,按 Vibe Coding 单独做)。
- 向量库 / 长期记忆 / RAG。
- 多用户会话持久化(会话状态仅内存保存,或最简会话对象;本章验收不要求重启存活)。
- 认证、鉴权、限流、计费。

## Anti-Goals(技术上能跑通也算失败)

- 把模型地址 / 模型名 / API Key 硬编码进代码 —— 必须全部走 `.env`。
- 把 System Prompt 写死在路由函数里 —— 必须 `PromptTemplate` 分离。
- 依赖某个具体模型的私有行为,导致换模型(GPT/Claude/Ollama)要改应用代码。
- 结构化输出的字段 / 枚举散落成字符串魔法值。

## Constraints(定死,不可自行更改)

- 技术栈:**Python + FastAPI + LangChain**。
- 模型接入:**应用侧统一说 OpenAI 协议,直连上游**;地址、模型名、密钥全在 `.env`。
- 本次验收上游:**DeepSeek**(用户已提供 API Key)。
- 诉求类型:**固定枚举**。

## 已确认的决策(来自与用户的对话)

- 上游模型:DeepSeek,OpenAI 协议兼容,base_url `https://api.deepseek.com`。
- 诉求类型为固定枚举,取值:退款、退货退款、换货、仅退款、补发、维修、物流催单、投诉、其他(共 9 类,"其他"兜底)。
- 结构化输出是**独立接口**,与对话接口分开调用,不嵌在对话流里自动触发。
- 模型名等具体取值放 `.env`,应用代码不写死。
- **会话标识**:客户端传 `session_id`,服务端按 session 在内存保存历史,超出 token 预算时裁剪。验收第 2 条(两轮上下文)用同一个 `session_id` 连发两轮来演示。
- **可换模型**:本章只做配置化(地址/模型名/密钥全在 `.env`,无模型专用代码),只用 DeepSeek 验收,不实际切换第二个 provider。

## 我替用户做的决定(标记为我的 call,用户可在审阅时推翻)

1. **路由与端口**:`POST /chat`、`POST /extract`,服务跑在 `8000` 端口(可用 `.env` 的 `HOST`/`PORT` 覆盖)。
2. **目录结构**:
   ```
   app/
     main.py            # FastAPI 应用装配、路由挂载
     config.py          # 读 .env,构建统一 OpenAI 协议的模型配置
     schemas.py         # 结构化输出的 Pydantic 模型(诉求类型枚举)
     prompts/
       customer_service.py  # 客服 System Prompt 模板 + PromptTemplate 实例
     services/
       chat.py          # 对话逻辑(流式、多轮上下文裁剪)
       extract.py       # 结构化提取逻辑
   ```
3. **System Prompt 内容**:客服角色设定 + 行为约束(礼貌、只答售后相关、不编造、涉及真实订单信息需引导提供、超范围拒答)。具体措辞由我起草,实现时给出。
4. **多轮上下文裁剪策略**:最简版。用消息列表 + 粗略 token 估算(按字符数折算),设定历史上下文预算上限,超出时从最早的消息开始裁剪,始终保留 System Prompt 和最近一轮用户消息。预算上限放 `.env`(如 `MAX_CONTEXT_TOKENS`)。
5. **流式格式**:SSE,`data:` 帧逐 token 推送,末尾发 `data: [DONE]` 结束(应用侧语义,非上游原样转发)。
6. **`.env` 字段**:`OPENAI_BASE_URL`、`OPENAI_API_KEY`、`MODEL_NAME`(默认 `deepseek-chat`)、`HOST`、`PORT`、`MAX_CONTEXT_TOKENS`。附带 `.env.example`(不含真实密钥)。
7. **结构化输出实现方式**:LangChain `ChatOpenAI.with_structured_output(...)` 的 Pydantic 模式;DeepSeek 同时支持 JSON mode 与 function calling,实现时按 LangChain 最新文档选能稳定跑通的那一种并验证。
8. **结构化输出字段可空性**:`诉求类型` 为必填枚举(带"其他"兜底);`订单号`、`期望方案` 为可空字段(`Optional[str]`),描述中缺失时返回 `null`,不让模型编造。接口固定返回三个字段。
9. **会话存储**:`session_id` → 历史消息列表的内存字典,带简单线程安全与容量上限(超出上限清理最旧会话),本章不要求重启存活。

## 验收标准(来自用户,不可改动)

1. `curl` 调对话接口能看到流式回复。
2. 连续问两轮,第二轮能接住第一轮的上下文。
3. 发一段售后描述,能拿到结构化 JSON。

## 留给 builder 的

- 具体类/函数签名、依赖版本、测试组织、异常处理细节、实现顺序。
- 上述"我的 call"标记项的具体落地,除非用户推翻。
