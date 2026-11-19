# 第九章 · 可观测性与数据飞轮 · 本地操作手册

本章依赖两类外部设施：Langfuse v3（trace / 成本报表的数据源）和 MySQL
（待审队列、评估趋势、低置信度问题池）。批处理脚本按**单进程**设计，
调度交给 Makefile + cron，不在应用进程内起定时器。

默认解释器：`PYTHON ?= .venv/Scripts/python.exe`（Windows）。Linux/macOS
可在命令行覆盖，例如 `make PYTHON=.venv/bin/python test`。

---

## 1. 启动 Langfuse v3

独立 compose 栈在 `docker-compose.langfuse.yml`，只把 Web UI 的 `3000`
发布到宿主机；Postgres / ClickHouse / Redis / MinIO 留在 compose 网络内，
不会和 Milvus 的 `9000/9001` 冲突。

```bash
make langfuse-up      # docker compose -f docker-compose.langfuse.yml up -d
# 首次启动会拉取镜像并跑迁移，等 Web 健康检查通过
# 打开 http://localhost:3000

make langfuse-down    # 停止容器（保留数据卷）
```

### 首次启动时自动建好的开发凭据

compose 里的 `LANGFUSE_INIT_*` 只在**首次**启动、数据卷为空时生效：

| 项 | 值 |
|---|---|
| 组织 / 项目 | `cs` / `cs-chat` |
| Public key | `pk-lf-ch09` |
| Secret key | `sk-lf-ch09` |
| 登录账号 | `dev@example.com` / `langfuse-dev` |

把 key 写进 `.env`（`app/config.py` 也读取同名变量）：

```dotenv
LANGFUSE_PUBLIC_KEY=pk-lf-ch09
LANGFUSE_SECRET_KEY=sk-lf-ch09
LANGFUSE_BASE_URL=http://localhost:3000
LANGFUSE_ENVIRONMENT=development
LANGFUSE_TRACE_NAME=cs-chat
```

三个字段（public / secret / base URL）齐全时才会在编译期挂上 Langfuse
回调；缺任意一个就完全不追踪，聊天照常工作。

### 想自己建 key 或改了 INIT 配置

- 数据卷已存在时，`LANGFUSE_INIT_*` 不会重新生效。到 UI 的
  Settings → API Keys 新建 key，复制到 `.env` 即可。
- 如果需要重新初始化开发环境，可 `docker compose -f
  docker-compose.langfuse.yml down -v` 清空全部 Langfuse 数据卷再
  `make langfuse-up`（**会删除已有 trace，慎用**）。

---

## 2. 初始化 MySQL 表结构

DDL 原文在 `sql/ch09-ddl.sql`（`review_queue`、`eval_runs`，以及
`low_confidence_questions` 的两个新列和外键）。幂等初始化脚本：

```bash
.venv/Scripts/python.exe -m app.db.init_ch09
```

行为：

- `review_queue`、`eval_runs` 和两个新列都已存在 → 打印 no-op 后退出。
- 全部缺失 → 按顺序执行 `sql/ch09-ddl.sql` 的所有语句。
- 只存在一部分 → 报错退出，不猜测、不半途补表；需要人工检查后补完或
  清理残留再重跑。

其他章节的表仍按各自 `app/db/init_ch0*.py` 初始化；本章不重复建旧表。

---

## 3. Makefile 目标

| 目标 | 等价命令 | 用途 |
|---|---|---|
| `make langfuse-up` | `docker compose -f docker-compose.langfuse.yml up -d` | 启动 Langfuse v3 栈 |
| `make langfuse-down` | `docker compose -f docker-compose.langfuse.yml down` | 停止栈（保留卷） |
| `make flywheel` | `python -m app.jobs.flywheel_pipeline` | 标准化 + 查重低置信问题，写入待审队列 |
| `make cost-report` | `python -m app.jobs.cost_by_intent --days 7` | 按 intent 聚合 Langfuse token 成本 |
| `make eval-flywheel` | `python -m app.jobs.eval_flywheel --triggered-by 手动` | 跑 `hybrid_rerank` 评估并落 `eval_runs` |
| `make calibrate-confidence` | `python -m app.jobs.calibrate_confidence` | 用评估集校准证据置信度阈值 |
| `make test` | `python -m pytest -q` | 全量测试 |

`scripts/` 下的同名脚本只是薄包装，逻辑都在 `app/jobs/`：

```bash
.venv/Scripts/python.exe scripts/flywheel_pipeline.py --limit 50
.venv/Scripts/python.exe scripts/cost_by_intent.py --days 30 --json
.venv/Scripts/python.exe scripts/eval_flywheel.py --triggered-by 定时 --limit 50
.venv/Scripts/python.exe scripts/calibrate_confidence.py --json
```

直接以模块方式运行也可以；两者参数一致。

---

## 4. 校准置信度并把推荐阈值回填

```bash
make calibrate-confidence
```

- 输入：默认 `tests/eval/rag_cases.json`（可用 `--cases` 指定）。
- 输出：`data/ch09/reports/confidence_calibration.txt` 与
  `confidence_calibration.json`。
- 终端会打印推荐阈值、Youden J、TPR/FPR/precision。

推荐阈值需要人工回填后再重启应用：

1. 打开 `data/ch09/reports/confidence_calibration.json`，取
   `recommended.threshold`（两位小数）。
2. 优先写到 `.env`（不改代码）：
   ```dotenv
   EVIDENCE_CONFIDENCE_THRESHOLD=0.42
   ```
3. 若希望默认值随仓库提交，也同步修改 `app/config.py` 的
   `evidence_confidence_threshold` 默认值。
4. 重启 `python run.py` 生效。

`/observability` 页面会对比“推荐阈值”和“在用阈值”，不一致时提示回填。
`--top-k`、`--out-dir`、`--json` 可用于试算或改输出目录。

---

## 5. 飞轮批处理与 cron

`make flywheel` 扫描 `low_confidence_questions.matched_review_id IS NULL`
的行（每轮最多 `FLYWHEEL_CANDIDATE_LIMIT` 或 `--limit` 条），一次 LLM
调用完成标准化、与 `review_queue` **全部行**查重、生成示例答案：

- 命中：`occurrence_count` 累加，原话回写 `matched_review_id`；已驳回 /
  已通过的行只累加次数、状态不变。
- 未命中：新建 `review_queue`（状态 `待审`）。
- 坏 JSON、幻觉 id、模型异常：该条跳过，游标留在原地，下一轮重试；
  低置信度原话不丢。
- 失败条数非零时进程退出码为 1，便于 cron 告警。

cron 示例（每 30 分钟，按实际仓库路径和解释器调整）：

```cron
*/30 * * * * cd /path/to/ecommerce-cs && .venv/bin/python -m app.jobs.flywheel_pipeline >> log/flywheel.log 2>&1
```

也可直接 `cd /path/to/repo && make PYTHON=.venv/bin/python flywheel`。
Windows 计划任务用 `D:\...\.venv\Scripts\python.exe -m
app.jobs.flywheel_pipeline`，工作目录设为仓库根。

同一作业不要并发运行；见第 8 节“单进程限制”。

---

## 6. 自动化评估与趋势

```bash
make eval-flywheel                 # 手动触发
```

- 调用 `app.jobs.eval_rag` 只跑 `hybrid_rerank` 策略，用例默认
  `tests/eval/rag_cases.json`。
- 指标写入 `eval_runs`（`triggered_by`、`dataset_size`、`metrics` JSON、
  `created_at`）；同时生成
  `data/ch09/reports/eval_flywheel.md`、`.json` 和趋势副本
  `eval_trend.txt`。
- 评估非零退出或报告缺失/不可读时不插入 `eval_runs` 行，失败不会伪装成
  趋势点。

cron 示例（每天 03:00，标记为“定时”）：

```cron
0 3 * * * cd /path/to/ecommerce-cs && .venv/bin/python -m app.jobs.eval_flywheel --triggered-by 定时 >> log/eval_flywheel.log 2>&1
```

趋势以 `eval_runs` 为唯一权威来源，`/observability` 读取最近 10 轮。

---

## 7. 报表路径与后台页面

默认 `CH09_REPORT_DIR=data/ch09/reports`（`.env` 可改）：

| 报表 | 文件 | 生产者 |
|---|---|---|
| 意图成本账 | `cost_by_intent.txt` / `.json` | `app.jobs.cost_by_intent` |
| 置信度校准 | `confidence_calibration.txt` / `.json` | `app.jobs.calibrate_confidence` |
| 评估报告 | `eval_flywheel.md` / `.json` | `app.jobs.eval_flywheel` → `eval_rag` |
| 评估趋势文本 | `eval_trend.txt` | `app.jobs.eval_flywheel` |
| 评估趋势数据 | MySQL `eval_runs` 表 | `app.jobs.eval_flywheel` |

成本报表需要 Langfuse 里已有 trace：先启动 Langfuse、配好 key、聊几轮，
再跑 `make cost-report`（`--days` 可改窗口，`--json` 打印 JSON）。缺
key 时脚本提示先 `make langfuse-up` 并以退出码 1 结束。

页面（`python run.py` 后访问）：

| 路径 | 内容 |
|---|---|
| `/observability` | 意图成本账、评估趋势、置信度校准三块；可触发白名单作业 |
| `/review` | 待审队列列表、详情（含召回快照）、通过 / 驳回 |
| `/admin` | 待审 / 通过 / 驳回计数与最高频缺口摘要 |

后台作业 API 只接受固定作业名：`cost-report`、`eval-flywheel`、
`calibrate-confidence`；`POST /api/admin/jobs/{name}/run` 启动，
`GET /api/admin/jobs/{name}/status` 轮询。运行器不拼 shell，参数之外的
命令片段进不来；同名作业在进程内互斥（409），不同作业可并行。

---

## 8. 单进程限制

本章的批处理、后台作业运行器和审核写回都按**单进程**实现：

- 作业运行器用进程内 `threading.Lock` 保证同名作业互斥；审核写回也持有
  进程内模块锁，并在知识库构建进行时拒绝写回（409）。
- 这些锁**不跨进程**。用 `uvicorn --workers N`、多台机器或多个 cron
  同时跑同一作业时，没有全局互斥，可能出现重复处理或写回竞争。
- 因此：`python run.py` 按单 worker 运行；cron 与后台页面不要同时触发
  同一个作业；飞轮 / 评估 / 审核写回不要并行跑。
- 要多 worker / 跨进程安全，需要外部队列或分布式锁，本章不实现。

---

## 9. 反馈与审核闭环

1. 聊天页对一轮回答点 👎，前端 `POST /api/feedback`：
   `{session_id, conversation_id, rating: "down", question}`。
   后端校验会话归属，从 checkpointer 最近一次 state 回捞该问题的 Top-3
   召回快照；图不可用时降级为无快照落池，不会返回 500。
   `up` 只记日志，不落库。
2. 低置信度问题池的 `source` 有三种：`retrieval_low_conf`（检索证据
   不足）、`self_check`（selfcheck 判定不够答）、`user_feedback`（👎）。
   基础设施 / rerank / 订单查询失败不落池。
3. 跑 `make flywheel`（或等 cron）把池里的原话标准化、查重后写入
   `review_queue`。
4. 打开 `/review` 审核：
   - **通过**：弹框填核准答案（可预填 `ai_suggested_answer`），提交
     `POST /api/review/{id}/approve`。后端用 ch03 的 `insert_mined_qa`
     写入知识库为 `mined_qa` 分类，并同步 `vectorize_pending`；只有向量化
     成功才置为「通过」并保存 `approved_answer`。向量化失败时状态保持
     「待审」，已落库 chunk 保持 pending，重试会复用同一行。
   - **驳回**：`POST /api/review/{id}/reject`，置「驳回」。驳回是终审，
     飞轮再次命中只累加次数，不复活、不新建。
   - 仅「待审」可操作，否则 409；行不存在 404；核准答案为空 422；
     写回或知识库构建进行中返回 409。
5. 通过后重新提问同一问题，应能命中新写入的 `mined_qa` 知识。

开发环境的 `/review`、`/admin`、`/observability` 和作业 API 没有登录
鉴权（与既有后台页一致）；生产部署需要自行补鉴权 / 网络隔离。
