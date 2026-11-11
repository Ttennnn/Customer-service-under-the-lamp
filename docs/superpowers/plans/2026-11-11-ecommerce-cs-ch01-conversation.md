# 电商智能客服 · 第一章(纯对话闭环)实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 交付一个纯后端最小闭环:SSE 流式对话接口 + PromptTemplate 客服角色管理 + `with_structured_output` 结构化提取 + 最简多轮上下文(内存会话 + token 预算裁剪),用 DeepSeek 通过 OpenAI 协议直连验收。

**Architecture:** FastAPI 暴露 `POST /chat`(SSE 逐 token)与 `POST /extract`(结构化 JSON)。模型统一走 LangChain `ChatOpenAI`(OpenAI 协议,base_url/model/key 全在 `.env`)。会话历史存内存 `SessionStore`,带进模型前做 token 预算裁剪。System Prompt 用 `PromptTemplate` 与业务代码分离。

**Tech Stack:** Python 3.11+ · FastAPI · LangChain(`langchain` + `langchain-openai`)· pydantic-settings · DeepSeek(OpenAI 协议)。

**Spec:** `docs/superpowers/specs/2026-11-11-ecommerce-cs-ch01-conversation-design.md`

## Global Constraints

- 技术栈定死:Python + FastAPI + LangChain;不引入 LangGraph / Milvus / Langfuse。
- 模型接入:应用侧统一 OpenAI 协议,直连上游;`OPENAI_BASE_URL` / `OPENAI_API_KEY` / `MODEL_NAME` 全在 `.env`,应用代码不写死、无模型专用分支。
- 本章不实现:工具调用、Agent 循环、聊天页面、会话持久化、认证/限流。
- 诉求类型固定枚举:退款 / 退货退款 / 换货 / 仅退款 / 补发 / 维修 / 物流催单 / 投诉 / 其他(共 9 类)。
- 每个涉及库 API 的任务,实现前先用 Context7 查最新文档(`langchain-openai` 的 `ChatOpenAI` 构造与 `stream`、`with_structured_output` 的方法选择;`fastapi.sse` 的 `EventSourceResponse` / `ServerSentEvent`)。

## Review Focus

以下是 spec 未明说、但真实用户会撞上、且无任务自带测试最容易翻车的输入/失败模式,按最可能踩中排序:

1. **不存在的 `session_id`** → 应新建空会话、正常回复,而非 500。→ Task 4 测试:缺失 key 的 `get` 返回空列表。
2. **单条消息超过 token 预算**(超长输入)→ 不崩溃,仍保留 System Prompt + 当前用户消息。→ Task 4 测试:`build_messages` 在 current 单条即超预算时仍返回 `[system, current]`。
3. **空 / 空白 `message`** → 返回 4xx 校验错误,不把空串发给模型。→ Task 7 测试:空 message 得 422。
4. **结构化描述缺 `order_id` / `expected_solution`** → 字段为 `null` 不编造;诉求类型缺失/非法 → 模型输出不合枚举时给出明确错误,不裸崩。→ Task 2 测试(可空字段)、Task 6 测试(缺字段返回 null)、Task 7 测试(返回 JSON 结构含三字段)。
5. **上游 LLM 报错**(密钥错 / 网络超时)→ 返回明确错误信息,不挂起 SSE 连接。→ Task 7 测试:mock llm 抛异常时端点返回错误而非空挂。

---

## File Structure

```
app/
  __init__.py
  main.py                 # FastAPI 装配、路由、依赖注入(llm/store 经 DI 便于测试 override)
  config.py               # Settings(pydantic-settings 读 .env)+ build_llm()
  schemas.py              # AppealType 枚举、AfterSaleExtraction、ChatRequest、ExtractRequest
  prompts/
    __init__.py
    customer_service.py   # 客服 System Prompt 模板 + render_system_prompt()
  services/
    __init__.py
    session.py            # SessionStore:session_id -> 历史消息列表(内存 + 锁)
    context.py            # estimate_tokens()、build_messages()(裁剪)
    chat.py               # stream_chat():流式生成器
    extract.py            # extract_after_sale():with_structured_output
tests/
  test_config.py
  test_schemas.py
  test_session.py
  test_context.py
  test_chat.py
  test_extract.py
  test_api.py
requirements.txt
.env.example
.gitignore
```

---

## Task 1: 项目脚手架 + 配置

**Files:**
- Create: `requirements.txt`, `.env.example`, `.gitignore`, `app/__init__.py`, `app/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces:
  - `app.config.Settings`(pydantic-settings `BaseSettings`,读 `.env`)
  - `app.config.build_llm(settings: Settings) -> ChatOpenAI`
  - `app.config.get_settings() -> Settings`

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_config.py
def test_settings_defaults(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    s = Settings()
    assert s.openai_base_url == "https://api.deepseek.com"
    assert s.model_name == "deepseek-chat"
    assert s.max_context_tokens == 4096

def test_build_llm_uses_settings():
    s = Settings(openai_api_key="sk-test", openai_base_url="http://localhost:11434/v1", model_name="qwen")
    llm = build_llm(s)
    assert llm.model_name == "qwen"
    assert llm.openai_api_base == "http://localhost:11434/v1"
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_config.py -v`
Expected: FAIL(`Settings` / `build_llm` 未定义)。

- [ ] **Step 3: 实现 `app/config.py`**

`Settings` 字段(值即 spec 里的默认):`openai_base_url="https://api.deepseek.com"`、`openai_api_key`(必填)、`model_name="deepseek-chat"`、`host="0.0.0.0"`、`port=8000`、`max_context_tokens=4096`。`model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")`。
`build_llm` 返回 `ChatOpenAI(model=s.model_name, api_key=s.openai_api_key, base_url=s.openai_base_url, streaming=True, temperature=0.7)`。实现前先 Context7 查 `langchain-openai` 的 `ChatOpenAI` 构造参数名(`base_url`/`openai_api_base` 以查到的为准,测试断言跟着改)。

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/test_config.py -v`
Expected: PASS。

- [ ] **Step 5: 写 `.env.example`(不含真实密钥)、`requirements.txt`(fastapi、uvicorn[standard]、langchain、langchain-openai、pydantic-settings、pytest、httpx)、`.gitignore`(忽略 `.env`、`__pycache__`、`.venv`)**

- [ ] **Step 6: Commit**

```bash
git add requirements.txt .env.example .gitignore app/__init__.py app/config.py tests/test_config.py
git commit -m "chore: scaffold project with OpenAI-protocol config"
```

---

## Task 2: 数据模型与诉求类型枚举

**Files:**
- Create: `app/schemas.py`
- Test: `tests/test_schemas.py`

**Interfaces:**
- Produces:
  - `app.schemas.AppealType(str, Enum)`:成员 `REFUND="退款"`, `RETURN_REFUND="退货退款"`, `EXCHANGE="换货"`, `REFUND_ONLY="仅退款"`, `RESHIPMENT="补发"`, `REPAIR="维修"`, `LOGISTICS="物流催单"`, `COMPLAINT="投诉"`, `OTHER="其他"`。
  - `app.schemas.AfterSaleExtraction`:字段 `order_id: str | None`、`appeal_type: AppealType`、`expected_solution: str | None`。
  - `app.schemas.ChatRequest`:字段 `session_id: str`、`message: str`(两者 `min_length=1`)。
  - `app.schemas.ExtractRequest`:字段 `text: str`(`min_length=1`)。

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_schemas.py
def test_appeal_type_values():
    assert [m.value for m in AppealType] == ["退款", "退货退款", "换货", "仅退款", "补发", "维修", "物流催单", "投诉", "其他"]

def test_extraction_requires_appeal_type_but_nulls_others():
    m = AfterSaleExtraction.model_validate({"appeal_type": "退款"})
    assert m.order_id is None and m.expected_solution is None
    assert m.appeal_type == AppealType.REFUND

def test_chat_request_rejects_blank_message():
    with pytest.raises(ValidationError):
        ChatRequest(session_id="s1", message="   ")
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_schemas.py -v`
Expected: FAIL(导入错误)。

- [ ] **Step 3: 实现 `app/schemas.py`**

按 Interfaces 定义枚举与 4 个 Pydantic 模型;可空字段用 `Field(default=None, description=...)`,`appeal_type` 用 `Field(description="诉求类型")`。

- [ ] **Step 4: 运行确认通过** → `pytest tests/test_schemas.py -v` → PASS

- [ ] **Step 5: Commit**

```bash
git add app/schemas.py tests/test_schemas.py
git commit -m "feat: add appeal-type enum and request/extraction schemas"
```

---

## Task 3: 客服 System Prompt 模板

**Files:**
- Create: `app/prompts/__init__.py`, `app/prompts/customer_service.py`
- Test: `tests/test_prompt.py`

**Interfaces:**
- Produces:
  - `app.prompts.customer_service.CUSTOMER_SERVICE_SYSTEM_PROMPT: PromptTemplate`(变量 `business_name`)
  - `app.prompts.customer_service.render_system_prompt(business_name: str = "本店") -> str`

**说明:** 这是纯 Prompt 产出。按你的要求,可单测的只有「模板能渲染、且渲染结果包含角色设定与行为约束」的冒烟测试;「客服行为是否正确」的验证放到 Task 8 端到端验收,用标注样例(售后咨询应礼貌回复、问及无关话题应婉拒)对着真实模型跑。

- [ ] **Step 1: 写失败的冒烟测试**

```python
# tests/test_prompt.py
def test_render_contains_role_and_constraints():
    p = render_system_prompt(business_name="XX商城")
    assert "XX商城" in p
    assert "客服" in p
    assert "不" in p  # 含约束/拒答措辞
```

- [ ] **Step 2: 运行确认失败** → `pytest tests/test_prompt.py -v` → FAIL

- [ ] **Step 3: 实现 `app/prompts/customer_service.py`**

用 `PromptTemplate(input_variables=["business_name"], template=...)` 写清:客服角色(电商售后)、语气(礼貌、简洁)、行为约束(只答售后相关、不编造订单/物流信息、需要订单号等实据时引导用户提供、超出售后范围婉拒)、不承诺无法兑现的方案。模板文本由实现者起草,`render_system_prompt` 调 `.format(...)`。

- [ ] **Step 4: 运行确认通过** → `pytest tests/test_prompt.py -v` → PASS

- [ ] **Step 5: Commit** → `git add app/prompts/ tests/test_prompt.py && git commit -m "feat: add customer-service system prompt template"`

---

## Task 4: 会话存储 + 上下文裁剪(纯逻辑)

**Files:**
- Create: `app/services/__init__.py`, `app/services/session.py`, `app/services/context.py`
- Test: `tests/test_session.py`, `tests/test_context.py`

**Interfaces:**
- Produces:
  - `app.services.session.SessionStore`:构造 `SessionStore(max_sessions: int = 1000)`;方法 `get(session_id: str) -> list[BaseMessage]`(缺失返回新空列表)、`append(session_id: str, message: BaseMessage) -> None`(超 `max_sessions` 清最旧会话)。写操作用 `threading.Lock` 保护,v1 单用户够用。
  - `app.services.context.estimate_tokens(text: str) -> int`(最简:`len(text)`,1 字符 ≈ 1 token 的保守上界)。
  - `app.services.context.build_messages(system: SystemMessage, history: list[BaseMessage], current: HumanMessage, max_tokens: int) -> list[BaseMessage]`

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_session.py
def test_get_missing_returns_empty():
    store = SessionStore()
    assert store.get("nope") == []

def test_append_then_get_roundtrip():
    store = SessionStore()
    store.append("s1", HumanMessage(content="hi"))
    assert len(store.get("s1")) == 1

# tests/test_context.py
def test_build_messages_keeps_system_and_current_when_over_budget():
    system = SystemMessage(content="SYS" * 500)          # 大 system
    current = HumanMessage(content="CUR" * 500)          # 大 current,单条即超预算
    msgs = build_messages(system, [HumanMessage(content="old")], current, max_tokens=10)
    assert msgs[0] == system and msgs[-1] == current

def test_build_messages_trims_oldest_first():
    system = SystemMessage(content="S")
    history = [HumanMessage(content="a"), AIMessage(content="b" * 100)]
    current = HumanMessage(content="c")
    msgs = build_messages(system, history, current, max_tokens=20)
    # 最旧的 a 被裁掉,最近的 b 保留
    assert msgs[1].content == "b" * 100

def test_estimate_tokens_is_len():
    assert estimate_tokens("你好") == 2
```

- [ ] **Step 2: 运行确认失败** → `pytest tests/test_session.py tests/test_context.py -v` → FAIL

- [ ] **Step 3: 实现 `session.py` 与 `context.py`**

`build_messages` 算法:结果 = `[system] + (裁剪后的 history) + [current]`;预算约束为 `estimate_tokens(system)+estimate_tokens(current)+Σhistory ≤ max_tokens`,超出时从 history 最左(最旧)开始丢弃,直到满足;system 与 current 永不丢弃(即便单条超预算,如测试 1)。

- [ ] **Step 4: 运行确认通过** → `pytest tests/test_session.py tests/test_context.py -v` → PASS

- [ ] **Step 5: Commit** → `git add app/services/ tests/test_session.py tests/test_context.py && git commit -m "feat: add in-memory session store and token-budget context trimming"`

---

## Task 5: 对话流式服务

**Files:**
- Create: `app/services/chat.py`
- Test: `tests/test_chat.py`

**Interfaces:**
- Consumes: `SessionStore`(Task 4)、`build_messages`(Task 4)
- Produces:
  - `app.services.chat.stream_chat(llm, store: SessionStore, system_prompt: str, session_id: str, message: str, max_tokens: int) -> Iterator[str]`(同步生成器,逐 token yield;结束后把当前 `HumanMessage` 与累积的 `AIMessage` 追加进 store)

- [ ] **Step 1: 写失败的测试(用假 LLM)**

```python
# tests/test_chat.py
class FakeChunk:
    def __init__(self, content): self.content = content
class FakeLLM:
    def stream(self, messages):
        yield FakeChunk("你"); yield FakeChunk("好")

def test_stream_chat_yields_tokens_and_persists_history():
    store = SessionStore()
    tokens = list(stream_chat(FakeLLM(), store, "SYS", "s1", "在吗", max_tokens=100))
    assert tokens == ["你", "好"]
    history = store.get("s1")
    assert history[0].content == "在吗" and history[1].content == "你好"
```

- [ ] **Step 2: 运行确认失败** → `pytest tests/test_chat.py -v` → FAIL

- [ ] **Step 3: 实现 `stream_chat`**

实现前 Context7 查 `ChatOpenAI.stream()` 的返回:逐块为 `AIMessageChunk`,取 `.content` 得文本。流程:`history = store.get(session_id)` → `messages = build_messages(SystemMessage(system_prompt), history, HumanMessage(message), max_tokens)` → 遍历 `llm.stream(messages)` 累积并 yield → 结束后 `store.append(session_id, HumanMessage(message))`、`store.append(session_id, AIMessage(full))`。

- [ ] **Step 4: 运行确认通过** → `pytest tests/test_chat.py -v` → PASS

- [ ] **Step 5: Commit** → `git add app/services/chat.py tests/test_chat.py && git commit -m "feat: add streaming chat service with session persistence"`

---

## Task 6: 结构化提取服务

**Files:**
- Create: `app/services/extract.py`
- Test: `tests/test_extract.py`

**Interfaces:**
- Consumes: `AfterSaleExtraction`(Task 2)
- Produces:
  - `app.services.extract.EXTRACT_SYSTEM_PROMPT: str`
  - `app.services.extract.extract_after_sale(llm, text: str) -> AfterSaleExtraction`

- [ ] **Step 1: 写失败的测试(用假结构化 LLM)**

```python
# tests/test_extract.py
class FakeStructuredLLM:
    def __init__(self): self.captured = None
    def invoke(self, messages):
        self.captured = messages
        return AfterSaleExtraction(order_id=None, appeal_type="退款", expected_solution="退货退款")

class FakeLLM:
    def with_structured_output(self, schema):
        self.fake = FakeStructuredLLM()
        return self.fake

def test_extract_returns_model_and_handles_missing_order_id():
    llm = FakeLLM()
    result = extract_after_sale(llm, "东西坏了要退款")
    assert result.appeal_type == AppealType.REFUND
    assert result.order_id is None
    assert llm.fake.captured[0].content  # system 提示已传入
```

- [ ] **Step 2: 运行确认失败** → `pytest tests/test_extract.py -v` → FAIL

- [ ] **Step 3: 实现 `extract_after_sale`**

实现前 Context7 查 `ChatOpenAI.with_structured_output` 对 DeepSeek 的可用方法:优先 `method="function_calling"`(默认),若 DeepSeek 不兼容则试 `method="json_schema"` 或 `method="json_mode"`(`response_format: json_object`),以实际跑通为准,并把结论写进 dev-notes。逻辑:`structured = llm.with_structured_output(AfterSaleExtraction)`;`structured.invoke([SystemMessage(EXTRACT_SYSTEM_PROMPT), HumanMessage(text)])` 返回 `AfterSaleExtraction`。

- [ ] **Step 4: 运行确认通过** → `pytest tests/test_extract.py -v` → PASS

- [ ] **Step 5: Commit** → `git add app/services/extract.py tests/test_extract.py && git commit -m "feat: add structured after-sale extraction service"`

---

## Task 7: FastAPI 应用与路由(SSE + 结构化)

**Files:**
- Create: `app/main.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `Settings`/`build_llm`(Task 1)、`ChatRequest`/`ExtractRequest`(Task 2)、`render_system_prompt`(Task 3)、`SessionStore`(Task 4)、`stream_chat`(Task 5)、`extract_after_sale`(Task 6)
- Produces:
  - `app.main.app: FastAPI`
  - `POST /chat`(SSE)、`POST /extract`(JSON)
  - `app.main.get_llm_dep()`、`app.main.get_store_dep()` 作为依赖,便于测试用 `app.dependency_overrides` 注入假实现。

- [ ] **Step 1: 写失败的测试(TestClient + dependency_overrides 注入假 LLM/store)**

```python
# tests/test_api.py
def test_chat_streams_sse_and_done(client):
    with client.stream("POST", "/chat", json={"session_id": "s1", "message": "在吗"}) as r:
        body = "".join(r.iter_text())
    assert "data: 你" in body and "data: [DONE]" in body

def test_chat_blank_message_422(client):
    assert client.post("/chat", json={"session_id": "s1", "message": "  "}).status_code == 422

def test_extract_returns_three_fields(client):
    r = client.post("/extract", json={"text": "订单123 退款"}).json()
    assert set(r.keys()) == {"order_id", "appeal_type", "expected_solution"}

def test_chat_upstream_error_not_hang(client):
    r = client.post("/chat", json={"session_id": "s1", "message": "hi"})
    assert r.status_code == 500 and r.json()["detail"] != ""
```

- [ ] **Step 2: 运行确认失败** → `pytest tests/test_api.py -v` → FAIL

- [ ] **Step 3: 实现 `app/main.py`**

`POST /chat`:参数 `ChatRequest`;用 `response_class=EventSourceResponse`,同步生成器对 `stream_chat` 每个 token `yield ServerSentEvent(raw_data=token)`,末尾 `yield ServerSentEvent(raw_data="[DONE]")`;捕获上游异常并产出明确错误帧(或非流式 500)。实现前 Context7 查 `fastapi.sse` 的 `EventSourceResponse` / `ServerSentEvent` 用法(确认 `raw_data` 原样发 `data:`、支持 POST)。`POST /extract`:调 `extract_after_sale`,返回 `result.model_dump()`。llm 与 store 通过 `Depends(get_llm_dep)` / `Depends(get_store_dep)` 注入;`get_store_dep` 返回模块级单例 `SessionStore`。

- [ ] **Step 4: 运行确认通过** → `pytest tests/test_api.py -v` → PASS

- [ ] **Step 5: Commit** → `git add app/main.py tests/test_api.py && git commit -m "feat: wire FastAPI chat(SSE) and extract endpoints"`

---

## Task 8: 端到端验收(真实 DeepSeek)+ dev-notes

**Files:**
- Modify: `dev-notes/ch01.md`(补记验收结果)

**说明:** 此任务不走 TDD,是对着真实模型跑三条验收标准,并留痕。

- [ ] **Step 1: 建 `.env`(填真实 `OPENAI_API_KEY`,base_url 用 DeepSeek 默认),`pip install -r requirements.txt`,启动 `uvicorn app.main:app --reload`**

- [ ] **Step 2: 验收 1 —— 流式对话**

```bash
curl -N -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"session_id":"s1","message":"你好,我想咨询售后"}'
```
Expected: 逐 token 滚动输出,末尾 `data: [DONE]`。

- [ ] **Step 3: 验收 2 —— 两轮上下文**

```bash
curl -N -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"session_id":"s1","message":"我买了个手机壳"}'
curl -N -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"session_id":"s1","message":"我刚才说的那个,想退货"}'
```
Expected: 第二轮回复能指代"手机壳",说明接住了第一轮上下文。

- [ ] **Step 4: 验收 3 —— 结构化 JSON**

```bash
curl -X POST http://localhost:8000/extract -H "Content-Type: application/json" -d '{"text":"订单号 20231111001,收到货屏幕碎了,要求退货退款"}'
```
Expected: 返回 `{"order_id":"20231111001","appeal_type":"退货退款","expected_solution":...}` 形状的 JSON。

- [ ] **Step 5: 全量单测复跑 + dev-notes 补记验收结果(三条各 pass/fail、关键输出片段),并 commit 若验收过程中有代码修正则一并提交**

```bash
pytest -q
git add dev-notes/ch01.md
git commit -m "docs: record ch01 end-to-end acceptance"
```

---

## Self-Review 结论

- **Spec 覆盖**:对话接口(Task 5/7)、Prompt 管理(Task 3)、结构化输出(Task 6/7)、多轮上下文最简版(Task 4/5)、`.env` 可换模型(Task 1)、三条验收(Task 8)—— 全部有对应任务。
- **Step 可执行性**:每个测试步给了断言与命令,每个实现步给了签名与算法边界。
- **类型一致**:`SessionStore`/`build_messages`/`stream_chat`/`extract_after_sale`/`AfterSaleExtraction`/`AppealType` 的签名在各自任务的 Interfaces 与测试中一致。
- **Review Focus**:5 条均有归属任务与测试(Task 4 / 4 / 7 / 2·6·7 / 7)。
- **比例**:计划长度与 spec 相当,未写成代码转录。
