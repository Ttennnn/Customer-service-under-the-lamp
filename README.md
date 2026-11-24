# 灯下 · 电商售后智能客服 Agent

一个按章节逐步构建的电商售后客服 Agent：从纯对话闭环一路做到 RAG 知识库、
LangGraph 工作流、意图识别、上下文管理、MCP 工具系统、可观测性与数据飞轮。
默认以「灯下」店铺的售后知识（商品、退货退款、售后手册、FAQ）为知识库。

- 对话接口：`POST /chat`，SSE 流式输出
- 模型：默认 DeepSeek（OpenAI 兼容协议，可换任意兼容服务）
- 检索：Milvus 混合检索 + BGE 重排 + 证据置信度门控
- 数据：MySQL 8 存会话/工单/审计，SQLite 存 LangGraph checkpoint

## 功能概览

| 章节 | 能力 |
|---|---|
| ch01 | 纯对话闭环，多轮消息落库 |
| ch02 | 工具调用（订单、FAQ、工单） |
| ch03 | RAG 知识库：Markdown 切分、向量化、FAQ 召回 |
| ch04 | RAG 进阶：混合检索、RRF、重排、低置信度问题池 |
| ch05 | LangGraph Agent 工作流与置信度闸门 |
| ch06 | 意图识别、指代消解、订单选择中断 |
| ch07 | 会话上下文管理：摘要与分层压缩、token 预算 |
| ch08 | 工具系统：注册、校验、超时重试、审计、MCP 接入 |
| ch09 | 可观测性与数据飞轮：Langfuse、成本报表、待审队列 |
| ch10 | 多标签主题分类器（独立 ML 环境，可选 ONNX 服务） |

## 技术栈

- **Web**：FastAPI、Uvicorn、Server-Sent Events
- **编排**：LangChain、LangGraph（SQLite checkpoint）
- **模型**：DeepSeek（`OPENAI_BASE_URL` 可切换 OpenAI 兼容服务）
- **存储**：MySQL 8 + SQLAlchemy + PyMySQL
- **检索**：Milvus 3.0、`BAAI/bge-m3` 向量、`BAAI/bge-reranker-v2-m3` 重排
  （SiliconFlow API）
- **观测**：Langfuse v3（可选）
- **前端**：管理页为原生 HTML/JS/CSS；`frontend/` 是旧的 React + Vite 聊天组件
- **分类器（ch10）**：PyTorch、Transformers、ONNX（独立 `.venv-ml`，Python 3.12）

## 目录结构

```
app/
  api/         页面与 API 路由（chat、kb、review、topics、admin…）
  agent/       LangGraph 工作流、意图识别、指代消解、置信度
  rag/         文档切分、embedding、Milvus、重排、证据自检
  services/    聊天编排、历史/上下文、摘要任务
  tools/       工具注册与执行、内置 FAQ/订单/物流/工单、审计
  db/          模型、按章节的 DDL 初始化、seed
  jobs/        知识库构建、评估、成本报表、数据飞轮
  static/      管理页面 HTML/JS/CSS
knowledge/     知识库 Markdown（商品、售后、退货退款、FAQ）
sql/           各章节 DDL
mcp_servers/   物流（8101）与售后（8102）MCP Server
scripts/ch10/  分类器语料、训练、导出、评估脚本
tests/         pytest 测试 + quickstart shell 测试
dev-notes/     每章开发记录（设计取舍、验收证据）
docs/          运维手册与设计文档
```

## 快速开始

### 前置条件

- Windows 10/11 + Git Bash（`start.bat` 依赖 Git Bash 和 PowerShell）
- Python 3.13（仓库的 `.venv` 已验证 3.13.13；ch10 分类器另需 Python 3.12）
- Docker Desktop（提供 MySQL 8 与 Milvus）
- 一个 OpenAI 兼容的 LLM API Key，以及 SiliconFlow embedding Key

### 1. 配置环境变量

```bash
cp .env.example .env
```

至少填写：

```dotenv
OPENAI_API_KEY=sk-...
EMBEDDING_API_KEY=sk-...
```

其余字段按需调整（数据库、Milvus、检索参数、ch07 预算、ch09 Langfuse 等）。
`.env` 已在 `.gitignore` 中，不要提交。

### 2. 安装依赖

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

Linux/macOS 将路径换成 `.venv/bin/python`。

> `start.sh` 优先使用最新章节 worktree 自己的 `.venv`，没有才回退仓库根的
> `.venv`。依赖请装在脚本实际使用的那个环境里（用 `./start.sh --print-target`
> 可以看到 `python=` 指向哪个解释器）。

### 3. 启动依赖（MySQL + Milvus）

```bash
docker compose up -d
```

也可以跳过这一步：`start.sh` 会在启动前检查 `3306` / `19530`，未就绪时自动
启动 Docker Desktop 并执行 `docker compose up -d`，最多等待 180 秒。

> 四个服务已配置 `restart: unless-stopped`，Docker Desktop 也已开启登录自启，
> 正常情况下开机后会自动恢复。

### 4. 初始化数据库（仅首次）

```bash
# ch03 知识库表（幂等）
.venv/Scripts/python.exe -m app.db.init_ch03

# 基础表：conversations / messages / faq / tickets + FAQ 种子数据
.venv/Scripts/python.exe -m app.db.seed

# ch04 问题池 / 忠实性用例表（幂等）
.venv/Scripts/python.exe -m app.db.init_ch04

# ch05–ch08 是一次性增量 DDL（ALTER / CREATE，不要重复执行）
docker exec -i cs_customer_mysql mysql -uroot -proot cs_customer < sql/ch05_low_confidence_source.sql
docker exec -i cs_customer_mysql mysql -uroot -proot cs_customer < sql/ch06_ticket_type.sql
docker exec -i cs_customer_mysql mysql -uroot -proot cs_customer < sql/ch07-ddl.sql
docker exec -i cs_customer_mysql mysql -uroot -proot cs_customer < sql/ch07-layers.sql
docker exec -i cs_customer_mysql mysql -uroot -proot cs_customer < sql/ch08-ddl.sql

# 幂等初始化脚本
.venv/Scripts/python.exe -m app.db.init_ch09
.venv/Scripts/python.exe -m app.db.init_ch10
```

已有数据卷的环境不需要重复执行。

### 5. 构建知识库（仅首次）

```bash
.venv/Scripts/python.exe -m app.jobs.build_knowledge
```

从 `knowledge/` 读取 Markdown，切分后写入 MySQL 并向量化到 Milvus。任务可续跑、
按指纹复用已存在的块；常用参数：

```bash
# 只重新摄入文档块，不向量化（可稍后补跑）
.venv/Scripts/python.exe -m app.jobs.build_knowledge --reset --no-vectorize

# 清空全部（含挖掘出的 QA）后重建，必须显式确认
.venv/Scripts/python.exe -m app.jobs.build_knowledge --reset-all --yes
```

### 6. 启动服务

推荐直接双击 `start.bat`，或在 Git Bash 中：

```bash
./start.sh
```

脚本会：

1. 选择最新的 `feature/chNN-*` 分支（本地优先），必要时创建 worktree 并复制 `.env`；
2. 选择 worktree 的 `.venv`，没有则回退仓库根 `.venv`；
3. 检查并在需要时启动 MySQL / Milvus，等待端口就绪；
4. 结束占用 8000 端口的旧进程（含 uvicorn reload 父进程）；
5. 启动 `python run.py`，轮询 `http://127.0.0.1:8000/` 后打开浏览器。

可选环境变量：

| 变量 | 默认 | 作用 |
|---|---|---|
| `NO_BROWSER` | `0` | 设为 `1` 不自动打开浏览器 |
| `START_TIMEOUT` | `60` | 等待应用 HTTP 就绪的秒数 |
| `INFRA_TIMEOUT` | `180` | 等待 Docker / MySQL / Milvus 的秒数 |
| `INFRA_PORTS` | `3306 19530` | 启动前必须监听的端口 |
| `SKIP_INFRA_CHECK` | `0` | 设为 `1` 跳过依赖预检 |

也可以手动启动（需自行保证依赖已就绪）：

```bash
python run.py
```

### 访问页面

| 路径 | 用途 |
|---|---|
| `/` | 客服聊天（ch07 会话上下文 + 订单选择 / 退款表单） |
| `/ch05` `/ch06` `/ch07` | 各阶段的聊天页面 |
| `/kb` | 知识库录入与重建 |
| `/rag-eval` | RAG 四策略对照评估 |
| `/review` | 低置信度问题待审队列 |
| `/topics` `/topics/questions` | 主题分布与类目问题 |
| `/acceptance` | 验收总览（数据 / 评测 / 错例） |
| `/admin` | 后台首页 |
| `/observability` | 观测总览 |

## 测试

```bash
make test
# 等价于
.venv/Scripts/python.exe -m pytest -q

# 若依赖装在 worktree 的 venv 里，显式指定解释器：
make PYTHON=.worktrees/<slug>/.venv/Scripts/python.exe test
```

quickstart 启动脚本的 shell 测试：

```bash
bash tests/quickstart/test_start_sh.sh
bash tests/quickstart/test_kill_tree.sh
```

## 可选组件

### MCP 物流 / 售后工具（ch08）

默认关闭（`MCP_ENABLED=false`）。需要时先启动两个 Server：

```bash
.venv/Scripts/python.exe -m mcp_servers.logistics_server    # 8101
.venv/Scripts/python.exe -m mcp_servers.aftersales_server   # 8102
```

然后在 `.env` 中设置 `MCP_ENABLED=true` 并重启主服务。注意 `mcp` 包必须装在
运行这两个 Server 的解释器里（与主服务使用的 venv 保持一致）。

### Langfuse 可观测性（ch09）

```bash
make langfuse-up    # Web UI: http://localhost:3000
make langfuse-down
```

未启动 Langfuse 时，日志里会出现
`opentelemetry ... Failed to export spans`，属于噪音，不影响聊天。
详见 `docs/ch09-operations.md`。

### 主题分类器（ch10）

分类器使用独立的 `.venv-ml`（Python 3.12），不影响主服务：

```bash
make ch10-env        # 创建 .venv-ml 并安装 requirements-ch10.txt
make classifier-up   # 启动 HTTP 分类服务，端口 8110
make classifier-down # 停止
make ch10-eval       # 启动分类服务并跑评测
```

### 前端聊天组件（旧版）

主页面由后端直接提供。`frontend/` 是早期的 React + Vite 组件，仍可单独开发：

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173，后端已允许该来源的 CORS
```

## 常见问题

**聊天回复「客服服务暂时不可用,请稍后重试」**

99% 是后端依赖未启动：`/chat` 读取会话历史时连不上 MySQL。排查顺序：

```bash
docker compose ps -a                 # 容器是否 Exited
netstat -ano | grep ":3306 "         # 3306 是否监听
tail -n 50 <worktree>/log/app.log    # 找 SQLAlchemy / pymysql 堆栈
```

恢复：启动 Docker Desktop → `docker compose up -d`，或直接重新运行 `start.sh`
（会自动完成这两步）。

**本机装了原生 MySQL，为什么不用它？**

本项目的数据在 compose 的 `cs_customer_mysql` 容器（数据卷 `aitest_mysql_data`）
里。仓库根的 `.env` 指向 `localhost:3306`，原生 MySQL 如果占用该端口会冲突，
请保持原生 MySQL 服务停止。

**冷启动要多久？**

Docker Desktop + MySQL + Milvus 冷启动一般 30–60 秒。`start.sh` 会等到端口就绪
才启动应用；Milvus 在端口打开后可能还需数秒才完全健康，极短时间内问知识库问题
可能得到保守拒答，稍等即可。

**`docker compose down` 会丢数据吗？**

不会。`down` 只删除容器，保留数据卷；`down -v` 才会删除数据卷（`aitest_*`）。
日常停止用 `docker compose stop` 或直接关 Docker Desktop 即可。

**8000 端口被占用**

`start.sh` 会自动结束监听 8000 的进程及其 python 父进程；手动启动 `run.py` 时
需要自己处理。

## 开发记录与文档

- `dev-notes/ch01.md` … `dev-notes/ch10.md`：每章的设计取舍、实现与验收证据
- `docs/ch09-operations.md`：Langfuse、批处理任务与运维操作
- `docs/superpowers/`：快捷启动脚本的设计与实施计划
- `sql/`：各章节权威 DDL；`tests/`：pytest 与 shell 测试
