# 电商智能客服系统 · 第三章:RAG 知识库 — 设计文档

日期:2026-11-12
状态:待批准(已过 builder 预审 / 待用户批准)

## Intent & Why

第二章的 `query_faq` 是 SQL `LIKE` 关键词查表,验收第 3 条已经记录了一个预期漏召回:用户问「邮费是多少」,`faq` 表里没有「邮费」二字,工具只能回「未找到」。真实客服场景里,同一件事会被问成「邮费」「运费」「快递费」「包不包邮」;关键词查表天然接不住换说法。

第三章把 `query_faq` 的内部实现换成向量语义检索,并把知识来源从一张手写 FAQ 表升级成一条可持续的知识库流水线:

- 离线把 Markdown 知识文档按结构切成 chunk,双写 MySQL(原文权威源)+ Milvus(向量索引);
- 离线从历史客服对话里挖可复用的问答对,分批 LLM 抽取、整体去重后入库;
- 在线把用户问题向量化,Milvus 取 Top-K,回 MySQL 拿原文拼答案。

**工具契约不变**:`query_faq(keyword: str) -> str` 的名称、入参、返回文本形态、未命中话术都保持第二章的样子,上层 Agent/`/chat` 不需要改。

## Goals

1. **样本文档**:在 `knowledge/` 造一套客服 Markdown 知识文档(退货政策、商品 FAQ、售后手册、运费与配送说明),含可直接回答「邮费是多少」的运费条款。
2. **结构感知切分**:按标题层级切 leaf section;超长内容递归切;块间重叠并裁到最近句号;大表格按行切且每块复制表头。
3. **离线建库命令**:文档加载 → 切分 → MySQL 记 `pending` → 云端 BGE-M3 向量化 → Milvus 写向量 → 回填 `vector_id` / 转 `done`;按主键幂等,中断可重跑补齐。
4. **对话挖知识命令**:从 MySQL 历史会话分批喂 LLM 抽问答对 → `qa_extraction_staging` → 整体精确 + 向量近似去重 → 保留项进 `knowledge_chunks` → 复用同一条向量化流水线。
5. **落库结构**:`knowledge_chunks` 与 `qa_extraction_staging` 严格按用户 DDL 建表;`category + questions + answer` 拼成向量化文本,其余元数据只存不进向量。
6. **在线检索**:`query_faq(keyword)` 问题向量化 → Milvus COSINE Top-K → MySQL 原文 → 返回问答文本。
7. **验证**:代码逻辑走单测;Prompt 抽取与检索效果走标注样例/评估集;验收覆盖「邮费是多少」语义召回和「中断后重跑补齐」。

## Non-Goals(本章不做)

- 关键词召回、混合检索、重排(rerank);本章只跑 dense 向量单路。
- 文档上传 UI、知识管理后台、增量文档版本管理(改文档后默认需 `--reset` 重建)。
- 在线聊天过程中实时写知识库;建库/挖矿都是离线命令。
- 把 `faq` 表数据迁移进知识库或删除 `faq` 表;`query_faq` 不再读它,表保留。
- 前端改造;`/chat`、SSE、工具徽章行为不变。
- 认证、鉴权、限流、计费(同前两章)。
- 定时调度器;只交付可手动执行的 job 命令,部署方式由使用方决定。
- 向量库分片/分区/多租户/多集合管理。

## Anti-Goals(技术上能跑通也算失败)

- `query_faq` 仍走 `LIKE` 或保留关键词兜底 —— 本章只允许 dense 单路。
- 切块出现半截话:块首/块尾停在句子中间。
- 大表格切块丢失表头,后续块无法独立读懂。
- 双写不是幂等:中断后重跑产生重复 chunk 或重复向量;或 Milvus 写失败却把状态标成 `done`。
- 把 `section_path` / `content_type` / `is_key_clause` / `prev/next` 拼进向量化文本。
- 密钥、base_url、连接串硬编码;`.env` 进版本库。
- 建库命令要求「必须一次跑完」;任何一个 mid-run 失败都要能靠重跑收敛。

## Constraints(定死,不可自行更改)

- 嵌入模型:**BGE-M3**,走**硅基流动云端 API**,不在本机下载/运行模型。
- 向量库:**Milvus**(Docker standalone);原文权威源:**MySQL**。
- `query_faq` 工具契约不变:函数名、`keyword: str` 入参、`-> str` 返回、未命中话术保持第二章。
- `knowledge_chunks` / `qa_extraction_staging` 表结构以用户 DDL 为准,**不自行增删列**。
- 本章只做 dense 向量检索,不引入关键词召回/混合/重排。
- 任务必须是可手动启动的命令,不挂 APScheduler。
- 涉及库/框架/API 的用法先经 Context7 查证再实现;技术选型矛盾时停下问用户。

## Context7 已核实的接口事实

- **硅基流动 Embeddings**:`POST https://api.siliconflow.cn/v1/embeddings`,`Authorization: Bearer <key>`,`model` 可为 `BAAI/bge-m3`,`input` 支持单字符串或字符串数组,`encoding_format: "float"`;响应 `data[].embedding` + `data[].index`,`usage` 含 token 统计。`BAAI/bge-m3` 单条输入上限 8192 token。BGE-M3 dense 向量维度 1024,运行时校验返回长度。
- **pymilvus**:当前 PyPI 最新 `3.0.3`,与 Milvus `3.0.x` 配套。`MilvusClient(uri=...)`;`create_collection(collection_name, schema=CollectionSchema(...), index_params=...)`;`upsert(collection_name, data: list[dict])` 按主键插入/更新;`search(collection_name, data=[[...]], anns_field="vector", search_params={"metric_type": "COSINE"}, limit=K, output_fields=[...])` 返回 `list[list[dict]]`,hit 含 `id`、`distance` 与输出字段。
- **Milvus standalone**:官方 Docker Compose 为 `etcd` + `minio` + `standalone` 三容器,standalone 暴露 `19530`(gRPC)与 `9091`(metrics/health),依赖 `ETCD_ENDPOINTS=etcd:2379`、`MINIO_ADDRESS=minio:9000`。

## 云端连通性预验证(2026-11-12,已实测)

- 用真实 key 调 `POST /v1/embeddings`,`model=BAAI/bge-m3`,一次 3 条数组输入,HTTP 200。
- 返回 3 条向量,维度均为 **1024**;`usage.prompt_tokens=59`。
- 中文相似度量级:「邮费是多少」vs 运费说明 **0.7037**,vs 退货政策 **0.4665**。早期以 0.35 作为保守起点,Task 15 用评估集校准后正式默认值定为 **0.5**。
- 结论:eval/验收可以用真实云端嵌入,不需要 fake 兜底;`EMBEDDING_API_KEY` 只写本地 `.env`,不进仓库。

## 表结构(用户提供,权威)

```sql
-- =============================================================
-- ch03 · RAG 基础 · 建表 DDL
-- 本章新建:knowledge_chunks(知识库原文权威源)
-- 向量落 Milvus 集合 knowledge(非 MySQL,DDL 不含);MySQL 存原文 + 双写状态
-- category + questions + answer 三格拼成向量化文本;其余字段是元数据,只存不进向量
-- =============================================================

-- 确保中文 COMMENT 按 utf8mb4 解析(latin1 默认的 mysql client 会把中文 double-encode)
SET NAMES utf8mb4;

CREATE TABLE knowledge_chunks (
  id               BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT 'chunk 主键,与 Milvus 集合主键对齐',
  category         VARCHAR(255)    NOT NULL                COMMENT '分类 / 上级标题路径,进向量化文本',
  questions        TEXT            NOT NULL                COMMENT '问法或本节标题,多个问法换行分隔,进向量化文本',
  answer           TEXT            NOT NULL                COMMENT '正文答案,进向量化文本',
  section_path     VARCHAR(512)    NULL                    COMMENT '章节路径,元数据,溯源用,不进向量',
  content_type     VARCHAR(32)     NULL                    COMMENT '内容类型:faq / policy / manual 等,元数据',
  is_key_clause    TINYINT(1)      NOT NULL DEFAULT 0      COMMENT '是否关键条款,0 否 1 是,元数据',
  prev_chunk_id    BIGINT UNSIGNED NULL                    COMMENT '前一块指针,元数据',
  next_chunk_id    BIGINT UNSIGNED NULL                    COMMENT '后一块指针,元数据',
  vector_id        VARCHAR(64)     NULL                    COMMENT 'Milvus 集合 knowledge 里的主键,写入后回填',
  vectorize_status ENUM('pending','done') NOT NULL DEFAULT 'pending' COMMENT '待向量化 / 已向量化,双写幂等靠它',
  created_at       DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  updated_at       DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (id),
  KEY idx_category (category),
  KEY idx_vectorize_status (vectorize_status),
  CONSTRAINT fk_chunks_prev FOREIGN KEY (prev_chunk_id) REFERENCES knowledge_chunks (id) ON DELETE SET NULL,
  CONSTRAINT fk_chunks_next FOREIGN KEY (next_chunk_id) REFERENCES knowledge_chunks (id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='知识库 chunk 原文权威源';

CREATE TABLE qa_extraction_staging (
  id               BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '暂存行主键',
  batch_no         VARCHAR(64)     NOT NULL                COMMENT '抽取批次号,一批几十个会话跑一次,分批防串味、按批追溯',
  source_ref       VARCHAR(255)    NULL                    COMMENT '来源会话 / 导出文件标识,溯源用,不入最终知识库',
  question         TEXT            NOT NULL                COMMENT 'LLM 从会话抽出的用户问法',
  answer           TEXT            NOT NULL                COMMENT 'LLM 从会话抽出的客服答案',
  status           ENUM('extracted','kept','discarded') NOT NULL DEFAULT 'extracted' COMMENT '已抽出待去重 / 去重保留 / 去重丢弃',
  created_at       DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '抽取写入时间',
  PRIMARY KEY (id),
  KEY idx_batch_no (batch_no),
  KEY idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='历史对话抽 QA 的离线中转暂存表:分批抽取、整体去重,保留项入 knowledge_chunks,建库完成可清空';
```

**长度边界**:`category VARCHAR(255)`、`section_path VARCHAR(512)`、`vector_id VARCHAR(64)`。切分/写入前做长度保护:超长 `section_path` 截断并保留末级标题;`vector_id` 固定为 `chunk_{id}`,最长 `chunk_` + 20 位数字,远小于 64。

## 架构与数据流

```
离线·文档:
knowledge/*.md
  → 文档加载(frontmatter + 标题树)
  → 结构感知切分(section / 递归 / 重叠 / 表格)
  → MySQL knowledge_chunks(pending)
  → SiliconFlow BGE-M3 向量化
  → Milvus knowledge upsert(vector_id = chunk_{id})
  → MySQL 回填 vector_id + status=done

离线·对话:
MySQL conversations/messages
  → 按批组会话 → LLM 抽 QA(JSON)
  → MySQL qa_extraction_staging(extracted)
  → 整体去重(精确归一化 + 向量近似)
  → kept 行入 knowledge_chunks(pending)
  → 复用上面的向量化流水线

在线:
query_faq(keyword)
  → BGE-M3 向量化
  → Milvus COSINE Top-K
  → MySQL 按 chunk_id 取原文
  → "问:...\n答:..."
```

## 样本文档与切分约定

### `knowledge/` 文件

| 文件 | content_type | 说明 |
|---|---|---|
| `退货政策.md` | `policy` | 退货条件、时限、退款、运费承担、关键条款 |
| `商品FAQ.md` | `faq` | 真实问法标题,如「尺码怎么选?」「发票怎么开?」 |
| `售后手册.md` | `manual` | 换货、维修、投诉、上门取件等流程 |
| `运费与配送说明.md` | `policy` | 包邮门槛、未达标运费、偏远地区、退换货运费承担 |

frontmatter 约定:

```markdown
---
content_type: policy        # policy / manual / faq
category: 售后政策           # 仅 faq 文档必填;policy/manual 由标题路径推导
---
```

### 字段推导

- **policy / manual**:每个 leaf section 生成 chunk;`questions` = leaf 标题;`category` = leaf 的上级标题路径(如 `退货政策 > 退货时限`);`section_path` = 含 leaf 的完整路径(如 `退货政策 > 退货时限 > 七天无理由`)。
- **faq**:每个 H2 是一个真实问法;`questions` = H2 标题;`category` 取 frontmatter;`section_path` = 文档标题 > H2。
- **mined_qa**:`questions` = LLM 抽出的用户真实问法;`category` = LLM 从固定枚举 `售后/物流/支付/商品/其他` 中选;`section_path` 为空。
- **is_key_clause**:正文含显式标记 `【关键条款】` 记 1;否则 policy/manual 正文命中 `不予|不得|必须|免责|期限内|时效` 记 1;faq/mined_qa 默认 0。
- **prev/next**:同一文档内按切分顺序链接;跨文档不链。mined_qa 不链。
- **向量化文本**:`f"{category}\n{questions}\n{answer}"`,不做任何前缀/指令模板。

### 切分算法

1. 解析 YAML frontmatter;代码围栏内的 `#` 不当标题。
2. 维护标题栈(`#`~`######`),每个标题开始一个 section;记录完整 heading path 与正文行。
3. 叶子 section 正文 ≤ `CHUNK_MAX_CHARS`(默认 500 字)时整段成块。
4. 超长正文按优先级递归切:空行段落 → 列表项/表格 → 句末标点(`。!?;!?`)→ 硬切。硬切是最后手段,并尽量回退到最近的标点。
5. 块间重叠 `CHUNK_OVERLAP_CHARS`(默认 80 字):从上一块尾部取最多 80 字,向前/向后找最近的句子边界,保证新块不从半句话开始、上一块也不在半句话结束。
6. 表格:识别连续 Markdown 表格(表头行 + `|---|` 分隔行 + 数据行)。整表超长时按数据行分组,每组块都复制表头 + 分隔行;表格与普通正文混排时分别切块。
7. 空 section、纯标题、只有分隔线的 section 不产 chunk。
8. 同一文档内完全相同的 `(category, questions, answer)` 只保留首次出现。

### 幂等入库键

DDL 没有来源文件列,因此文档入库用**内容指纹**判重:`(content_type, section_path, category, questions, answer)` 完全相等视为同一 chunk,已存在则复用其 `id`,不重复插入。这样同一份文档重跑不会膨胀;改内容后旧 chunk 会保留,需 `build_knowledge --reset` 全量重建(本章明确的 Non-Goal)。

## 双写与向量化

### 写入顺序

1. MySQL 插入/复用 chunk,`vectorize_status='pending'`,`vector_id=NULL`。
2. 同一文档内回填 `prev_chunk_id` / `next_chunk_id`。
3. 取所有 `pending` 行(不只本次插入,以恢复上次中断)。
4. 拼向量化文本 → 分批调硅基流动 Embeddings。
5. Milvus `upsert`:主键 `vector_id = f"chunk_{id}"`,字段 `id`、`chunk_id`、`vector`。
6. MySQL 更新 `vector_id`、`vectorize_status='done'`,按批提交。

### 幂等与失败语义

- Milvus 主键由 MySQL 主键确定,`upsert` 天然幂等:即使「Milvus 已写、MySQL 未回填」时崩溃,重跑也是同一主键覆盖。
- 只有 Milvus `upsert` 成功且 MySQL 更新成功才算 `done`;任何一步失败保持 `pending`,命令以非零退出并打印失败 chunk id。
- Embeddings 429/503/504 与网络错误做有限指数退避重试;重试耗尽仍失败则中止该批,不静默吞掉。
- 每次向量化后校验 `len(vector) == EMBEDDING_DIM`(默认 1024),不一致立即报错。
- 重跑不产生重复 MySQL 行(内容指纹)也不产生重复 Milvus 实体(主键 upsert)。

### 文档重建(`--reset`)语义

`--reset` 是**只清文档域、保留挖矿知识**的安全重建,不是整表清空:

1. 选出 `content_type IN ('policy','manual','faq')`(含 `content_type IS NULL` 的历史行)的文档域 chunk,收集其非空 `vector_id`。
2. 先删 Milvus 中这些 `vector_id` 对应的实体(按 id 批量删;无 `vector_id` 的 pending 行无需删)。
3. 再删 MySQL 文档域行(自引用 FK 由 `ON DELETE SET NULL` 处理)。
4. 保留 `content_type='mined_qa'` 的行与其 Milvus 向量;挖矿知识不受文档重建影响。
5. 之后按 `knowledge/` 当前文件重新切分、入库、向量化。文件被删掉后其旧 chunk 也就不会复活。
6. 若确实要连挖矿知识一起清空,使用显式的 `--reset-all`;执行前打印将删除的行数/实体数并等待确认(非交互环境需 `--yes`)。默认不带任何 reset 标志时只做增量幂等入库。

### Milvus 集合

- 集合名 `knowledge`(可配)。
- 字段:`id VARCHAR(64)` 主键;`chunk_id INT64`;`vector FLOAT_VECTOR(1024)`。
- 索引:`FLAT` + `COSINE`(小规模知识库用精确检索,后续可换 HNSW/IVF_FLAT)。
- 创建集合时使用强一致(或写入后 flush),保证建库命令结束后立即可检索;检索结果用 `chunk_id` 回 MySQL 取权威原文。
- 章节路径、内容类型、关键条款、前后指针**只存 MySQL**,不进 Milvus。

## 在线检索

`app/tools/faq.py` 重写,但保持:

```python
@tool
def query_faq(keyword: str) -> str:
    """查询常见问题(FAQ)。..."""
```

流程:

1. `keyword` 经同一个 EmbeddingClient 向量化。
2. Milvus `search` top_k(默认 3),`metric_type="COSINE"`,`output_fields=["chunk_id"]`。
3. 过滤 `distance < FAQ_SCORE_THRESHOLD`(默认 0.5,Task 15 用评估集校准后的正式值;早期默认 0.35)。
4. 按 `chunk_id` 到 MySQL 取 `questions`、`answer`,保持 Milvus 相似度顺序。
5. 返回 `"\n\n".join(f"问:{row.questions}\n答:{row.answer}")`;多条问题按原样换行。
6. 无命中返回原文案:`"未找到相关问题,请换个关键词再试试。"`
7. 向量服务/Milvus/DB 异常:记录日志,返回 `"知识库暂时不可用,请稍后重试或联系人工客服。"`,不向 `/chat` 抛异常。

工具内部自开 MySQL 会话(沿用第二章做法);Milvus 客户端与 Embedding HTTP 客户端用惰性单例 + 锁,避免每次调用重建连接,但不在 import 时联网。

**超时预算**:第二章 `execute_tool` 默认 `timeout_s=5, retries=1`。新链路多一次云端嵌入 + Milvus 检索,给 `query_faq` 单独放宽到 **8 秒**(registry 增加 `TOOL_TIMEOUTS = {"query_faq": 8.0}`,`stream_chat` 按工具名取超时,其余工具仍 5 秒);客户端惰性单例复用连接以压低首次延迟。工具契约(`keyword` 入参、`str` 返回)不变,只动执行器的内部超时参数。

## 对话挖知识

命令:`python -m app.jobs.mine_qa [--limit N] [--batch-size N] [--since ISO] [--dry-run] [--no-vectorize] [--force]`

1. **选会话**:从 `conversations` + `messages` 取有至少一条 user 和一条 assistant 消息的会话;默认跳过 `qa_extraction_staging.source_ref` 已出现过的会话(`--force` 可重跑);`--since` 按 `conversations.updated_at` 过滤。
2. **分批**:每 `--batch-size`(默认 10)个会话一次 LLM 调用;会话序列化为 `[{role, content}]`,只取 user/assistant,丢弃 tool 消息;超长会话按消息条数截断并保留首尾。
3. **抽取 Prompt**:要求只抽「可复用、与具体订单号/个人信息无关」的通用问答;`question` 保留用户真实问法;`answer` 用客服原话压缩;`category` 限定 `售后/物流/支付/商品/其他`;无可抽取内容返回空数组。用 `with_structured_output` 约束为 Pydantic 列表。
4. **落暂存**:每批生成 `batch_no`(如 `qa-20261112153000-ab12`),逐条写入 `qa_extraction_staging(status='extracted', source_ref=会话id)`,按批提交。
5. **整体去重**(所有批次抽完后统一做):
   - 问题归一化:去首尾空白、去空白与常见中英文标点、全角转半角、英文小写。
   - 精确去重:归一化问题相同,保留 staging `id` 最小的一条。
   - 近似去重:把 staging 问题与已入库 `knowledge_chunks.questions`(按换行拆开)一起向量化,余弦 ≥ `QA_DEDUP_THRESHOLD`(默认 0.92)视为重复,保留先出现的一条。
   - 被丢弃行置 `discarded`,保留行置 `kept`。
6. **入库**:`kept` 行按内容指纹插入 `knowledge_chunks`:`content_type='mined_qa'`、`questions=question`、`answer=answer`、`category=LLM category`、`section_path=NULL`、`is_key_clause=0`、`pending`。
7. **向量化**:默认复用 `vectorize_pending`;`--no-vectorize` 只落 pending,留给 `python -m app.jobs.vectorize`。
8. **可观测**:打印每批会话数、抽出条数、精确/近似丢弃数、入库条数;`--dry-run` 只打印抽取与去重结果,不写库。

### 中断恢复与 category 的持久化边界

`qa_extraction_staging` 按用户 DDL 没有 `category` 列,因此 **LLM 给的 category 只在单次 `mine_qa` 进程内存中存活**,不做跨进程持久化;靠下面的重跑语义保证不丢知识:

1. 每次 `mine_qa` 启动时,对本次选中的 `source_ref` 先删除状态为 `extracted` 的旧行(上次中断留下的半成品),再重新抽取。避免「有 staging 行、却没有 category」的僵尸数据。
2. `kept` 行一旦已写入 `knowledge_chunks`,重跑时靠内容指纹 `(content_type, section_path, category, questions, answer)` 判重,不会重复入库;`discarded` 行保留作追溯。
3. `--force` 额外删除选中 `source_ref` 的全部旧 staging 行(含 `kept`/`discarded`)后重新抽取;已入库的 `knowledge_chunks` 仍靠指纹幂等,不重复。
4. 因此挖矿是「可重跑收敛」而非「断点续跑」:中断后重跑会重做抽取,但不会产生重复知识;这与文档建库的 pending 断点续跑是两种恢复语义,均在验收里说明。

暂存表不自动清空,保留按批追溯;需要时另行清理(DDL 注释允许建库完成后清空)。

## 命令与配置

### 命令

```bash
# 1) 起基础设施(MySQL + Milvus standalone)
docker compose up -d

# 2) 建表(执行用户 DDL,幂等检查)
python -m app.db.init_ch03

# 3) 文档建库 + 向量化
python -m app.jobs.build_knowledge            # --reset 只重建文档域;--reset-all 连挖矿知识一起清;--no-vectorize 只落 pending
python -m app.jobs.vectorize                  # 单独补齐 pending(中断后重跑)

# 4) 对话挖知识
python -m app.jobs.mine_qa --limit 10         # --dry-run / --no-vectorize / --force

# 5) 检索评估
python -m app.jobs.eval_retrieval             # 跑标注评估集,打印 Top-K 命中与分数

# 6) 服务
uvicorn app.main:app --reload
```

### 新增配置(全部进 `.env` / `.env.example`,密钥只进本地 `.env`)

```dotenv
EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1
EMBEDDING_API_KEY=
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DIM=1024
EMBEDDING_BATCH_SIZE=16

MILVUS_URI=http://localhost:19530
MILVUS_COLLECTION=knowledge

FAQ_TOP_K=3
FAQ_SCORE_THRESHOLD=0.5

KNOWLEDGE_DOCS_DIR=knowledge
CHUNK_MAX_CHARS=500
CHUNK_OVERLAP_CHARS=80

QA_MINING_BATCH_SIZE=10
QA_DEDUP_THRESHOLD=0.92
```

## 建议文件布局

```
knowledge/                          # 样本文档(4 份)
sql/ch03_knowledge.sql              # 用户 DDL 原样
app/models/knowledge.py             # KnowledgeChunk / QaExtractionStaging ORM
app/rag/docs.py                     # frontmatter + 文档加载
app/rag/splitter.py                 # 结构感知切分
app/rag/embedding.py                # SiliconFlow EmbeddingClient
app/rag/milvus_store.py             # Milvus 集合初始化 / upsert / search
app/rag/vectorize.py                # pending → 向量 → Milvus → done
app/services/qa_mining.py           # 抽取 Prompt / Pydantic Schema / 解析
app/jobs/build_knowledge.py
app/jobs/mine_qa.py
app/jobs/vectorize.py
app/jobs/eval_retrieval.py
app/tools/faq.py                    # 重写为向量检索,契约不变
tests/test_knowledge_splitter.py
tests/test_knowledge_models.py
tests/test_vectorize.py
tests/test_query_faq_vector.py
tests/test_qa_mining.py
tests/eval/faq_cases.json           # 检索评估集
tests/eval/qa_mining_samples.json   # 挖掘标注样例
```

## 验证策略

### 可单测代码(TDD)

- 切分:标题层级、叶子识别、超长递归、重叠句界、表格表头复制、代码围栏、空 section。
- 字段推导:`category` / `questions` / `section_path` / `content_type` / `is_key_clause` / prev-next。
- 双写:fake embedder + fake Milvus,验证 pending→done、vector_id 回填、向量维度校验、失败保持 pending、重跑幂等。
- 检索工具:fake embedder + fake store + SQLite,验证 Top-K 顺序、阈值过滤、未命中话术、异常兜底、契约不变。
- 去重:归一化、精确重复、近似重复、与既有知识去重、保留最早一条。
- 挖矿:Prompt 输出解析、非法 category 归一化、空结果、分批入库。

### Prompt / 数据类(用标注样例或评估集)

- **检索评估集** `tests/eval/faq_cases.json`:至少覆盖
  - 「邮费是多少」→ 命中运费条款(包含 `包邮` / `运费` / `元`);
  - 「退货运费谁承担」「几天能退」「怎么开发票」「尺码怎么选」等换说法;
  - 1~2 条知识库外问题,期望低分/未命中。
  `eval_retrieval` 打印每条 query 的 Top-K、分数、是否命中,并汇总 top-1/top-K 命中率;用它校准 `FAQ_SCORE_THRESHOLD`。
- **挖掘标注样例** `tests/eval/qa_mining_samples.json`:2~3 组历史会话 + 期望抽取的问答对/期望去重结果;用 `mine_qa --dry-run` 跑,检查关键问法/答案是否被抽出、重复问法是否只保留一条。LLM 输出非确定,验收以「关键条目命中」为准,不逐字比对。

### 验收(来自用户,不可改动)

1. 「邮费是多少」这类换说法的问题,能召回运费说明并答对。
2. 故意中断建库任务再重跑,漏向量化的块能被捡起补齐。
   - 操作:向量化阶段 Ctrl-C / 或限制批次数后中断;确认 `pending > 0`;重跑 `build_knowledge`(或 `vectorize`);确认 `pending = 0`、Milvus 实体数 = `done` 行数、重复重跑不增实体。

## 我替用户做的决定(标记为 my call,用户可在审阅时推翻)

1. **样本文档 4 份**:退货政策、商品 FAQ、售后手册、运费与配送说明;frontmatter 标 `content_type`,faq 文档标 `category`。
2. **切分参数**:默认 `CHUNK_MAX_CHARS=500`、`CHUNK_OVERLAP_CHARS=80`,字符口径(中文按字),均可配。
3. **向量化文本**:`category\nquestions\nanswer`,无指令前缀;`questions` 多问法用换行分隔。
4. **主键映射**:Milvus 主键 `chunk_{MySQL id}`,`upsert` 幂等;Milvus 只存 `id/chunk_id/vector`,元数据全在 MySQL。
5. **文档入库幂等**:内容指纹 `(content_type, section_path, category, questions, answer)`;改文档需 `--reset`。
6. **Milvus 索引**:`FLAT` + `COSINE`;小库精确检索,规模上来再换 HNSW/IVF_FLAT。
7. **检索参数**:Top-3、阈值 0.5(Task 15 以评估集校准,早期 0.35 起步);未命中话术沿用第二章。
8. **挖矿去重**:归一化精确去重 + 余弦 0.92 近似去重,保留最早一条;与既有 `knowledge_chunks` 一并去重。
9. **挖矿 category**:固定枚举 `售后/物流/支付/商品/其他`,非法值归 `其他`。
10. **任务形态**:三个独立命令(`build_knowledge` / `mine_qa` / `vectorize`),不挂调度器;部署与触发由使用方决定。
11. **`faq` 表**:保留但不再被 `query_faq` 读取;不做数据迁移。
12. **异常兜底**:`query_faq` 内部捕获基础设施异常并返回安全话术,不把异常抛给 `/chat`。
13. **建表**:提交用户 DDL 原文到 `sql/ch03_knowledge.sql`,另写 `init_ch03` 执行器;ORM 仅用于测试与类型化读写,生产 schema 以 DDL 为准。
14. **重建语义**:`--reset` 只删文档域(`policy/manual/faq`)的 MySQL 行和 Milvus 实体,保留 `mined_qa`;`--reset-all` 才清空全部,且需打印影响面并二次确认(非交互 `--yes`)。理由:文档重建不应静默毁掉挖矿知识。
15. **在线超时**:`query_faq` 单独放宽到 8 秒(registry 的 `TOOL_TIMEOUTS`),其余工具保持 5 秒;不改变工具入参/出参契约。
16. **挖矿恢复语义**:`category` 只在内存,不做 staging 持久化;每次运行先清掉本次 `source_ref` 的 `extracted` 行后重抽;`--force` 清掉该 `source_ref` 的全部旧 staging 行;最终入库靠内容指纹幂等。挖矿是「重跑收敛」,不是断点续跑。
17. **真实嵌入可用**:已实测云端 key 可用、维度 1024;评估集与验收走真实 BGE-M3,不用 fake 冒充线上效果;`FAQ_SCORE_THRESHOLD` 以评估集结果校准后再定稿。

## 留给 builder 的

- 具体类/函数签名、pymilvus 3.0.3 的落地细节(集合 schema/index/强一致参数、hit 结构解析)、SiliconFlow 客户端重试细节、Pydantic 抽取 Schema、Docker Compose 的 Milvus 三件套版本与 healthcheck、SQLite 测试替身组织、命令参数解析、日志格式。
- 上述 my call 的具体落地,除非用户推翻。
