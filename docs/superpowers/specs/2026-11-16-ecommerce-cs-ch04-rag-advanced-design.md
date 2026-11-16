# 电商智能客服系统 · 第四章:RAG 进阶 — 设计文档

日期:2026-11-16
状态:已批准并实现(2026-11-16)
章节说明:原「第四章 · 知识录入页(`/kb`)」并入第三章,本章占用 ch04。

> 2026-11-17 语料更新:知识库已整体替换为 4 份「灯下」开头的喵喵优选文档——`灯下售后手册.md`、`灯下商品与购物FAQ.md`、`灯下商品规格手册.md`、`灯下退货退款政策.md`;旧 5 份灯下示例文档已删除。评估集、prompt 样例与四策略报告已同步。下文涉及旧示例型号/章节路径处,按历史设计说明理解。

## Intent & Why

第三章把 `query_faq` 从 SQL `LIKE` 升级成了 BGE-M3 + Milvus 的 dense 单路语义检索,解决了「邮费 / 运费」这类换说法召回。但它仍然有三个明显短板:

1. **具体型号、专有名词容易被 dense 检索稀释**:向量相似度对 `LX-A100` / `LX-B200` 这种只差一两个字符的型号区分度不够,关键词召回能补上。
2. **召回只取 Top-3、没有重排**:候选少、排序粗,容易把真正相关的条款挤掉。
3. **答案没有可追溯的引用,也没有「答不了就拒答」的硬闸门**:模型可能拿相邻证据硬编,或者承诺知识库里没有的到账时间。

本章把检索质量做成一条可评估的链路:查询理解 → dense + BM25 双路召回 → RRF 融合 → bge-reranker-v2-m3 精排 → 引用化组装 → 生成前自评 → 不足则拒答并落低置信度池。再用一套带 ground-truth 的评估集,把纯向量、纯 BM25、混合、混合+Rerank 四种策略放在同一张报告里对比。

## Goals

1. **Milvus 原生 BM25**:`knowledge_hybrid` 集合的 `text` 字段开 `enable_analyzer=True`、`analyzer_params={"type": "chinese"}`,用 `FunctionType.BM25` 生成 `sparse` 字段;`sparse` 建 `SPARSE_INVERTED_INDEX` + `metric_type="BM25"`(k1=1.2,b=0.75)。
2. **混合检索**:dense 与 BM25 各召回 Top-50,`hybrid_search` + `RRFRanker(k=60)` 融合;最终候选交给 bge-reranker-v2-m3 精排 Top-10。
3. **元数据前置过滤**:`category` / `content_type` 作为 Milvus 标量字段,通过 `AnnSearchRequest.expr` 在召回前过滤,两路都生效。
4. **Query 理解**:一次 LLM 调用把口语问法改写成标准问法,并产出同义检索词;`category` / `content_type` 过滤只来自显式参数或评估集,不由 LLM 猜;静态同义词表只拼进检索 query,不改变入库文本、不在库里拆存多份。
5. **引用与可追溯**:工具结果带 `[n]` 编号,编号映射到 chunk 的 `section_path`、原文答案;SSE 额外下发 `sources` 帧,聊天页点编号可展开来源。
6. **生成前自评与拒答**:工具执行后、最终答案生成前,用一次独立 LLM 调用判断「召回证据够不够回答」;`useful=false` 时显式拒答,原话、会话、入口、原因落 `low_confidence_questions`;自评/检索基础设施故障时也拒答,但按 infra 失败处理,不污染低置信度池。
7. **负面知识约束**:System Prompt 明确禁止承诺到账时间、一定退款/赔付、当天到货等。
8. **评估体系**:新增带难度梯度、按桶标注 ground-truth 的评估集;检索段评 Recall@5 / Recall@10 / MRR@10,生成段评 Faithfulness;四策略 × 分桶出报告;判出的编造落 `faith_cases`。
9. **前端配套**:引用编号可点开来源;每条 assistant 回答左下角 👍/👎,点后点亮、显示「已反馈」并一次性锁定;本章纯前端采集,不写后端。
10. **文档归位**:知识录入页的 spec / plan / dev-notes 并入第三章,`dev-notes/ch04.md` 让位给本章。

## Non-Goals(本章不做)

- 指代消解、多轮 query 改写:本章只处理单轮问法的归一与同义词扩展。
- 把同义词扩展结果写回 MySQL / Milvus,或在入库时拆存多份问法。
- 用户反馈写后端或建反馈表:👍/👎 只做纯前端采集,为后续数据飞轮留入口。
- 低置信度池的管理后台 / 查询页。
- 修改 `query_order` / `query_product` / `query_logistics` / `create_ticket` 的业务行为。
- 删除 `faq` 表、删除旧 `knowledge` 集合、改 ch01–ch03 的表结构。
- 多集合 / 多租户 / 分区 / 在线增量索引调度。
- Langfuse 等可观测平台的接入(本章不涉及;如后续需要另起)。

## Anti-Goals(技术上能跑通也算失败)

- 用应用层手写 BM25 或 LIKE 兜底代替 Milvus 原生 BM25。
- 混合检索只做一路召回、另一路仅作展示;RRF 融合必须是两路真实候选进入 `hybrid_search`。
- 同义词扩展进入入库文本,导致同一知识在库里有多份 chunk。
- 引用编号由模型自由编造,或编号无法映射回本轮工具返回的真实 chunk。
- 自评 `useful=false` 后仍然生成答案;或因知识不足拒答了却没有落 `low_confidence_questions`(基础设施故障导致的保守拒答除外)。
- 评估报告只给总体数字、不按桶拆分;或 ground-truth 用「看起来相关」代替可定位的 `section_path`。
- 为了跑分调低拒答闸门,把「知识库没有」的题答成有。
- 对旧 `knowledge` 集合做破坏性 schema 变更或删除。

## Constraints(定死)

- 检索:Milvus 原生 BM25 + `hybrid_search` + RRF;稠密向量仍用 BGE-M3(硅基流动云端,1024 维)。
- 重排模型:`BAAI/bge-reranker-v2-m3`,走硅基流动 `POST /v1/rerank`,复用现有 `EMBEDDING_API_KEY`(可用 `RERANK_API_KEY` 覆盖)。
- 原文权威源仍是 MySQL `knowledge_chunks`;Milvus 只存向量、BM25 文本与检索所需元数据。
- 新表结构以用户 DDL 为准,不自行增删列;`low_confidence_questions` 本章只写 `source='self_check'`。
- 技术选型矛盾时停下问用户,不自行换方案。
- 涉及库 / 框架 / API 的用法先过 Context7;已核实的接口事实见下节。

## Context7 已核实的接口事实(2026-11-16)

### SiliconFlow Rerank

- `POST https://api.siliconflow.cn/v1/rerank`,`Authorization: Bearer <key>`。
- 请求体:`model`(如 `BAAI/bge-reranker-v2-m3`)、`query`(string)、`documents`(string[])、可选 `top_n`、`return_documents`、`max_chunks_per_doc`、`instruction`。
- 响应:`results: [{index, document?, relevance_score}]`,`index` 是原 documents 下标,`relevance_score` 为相关性分数;`meta` 含 token 统计。
- 错误:400/401/403/404/429/503/504;429 与 5xx 可重试。

### Milvus 中文 analyzer / BM25

- 字段级配置:`schema.add_field("text", DataType.VARCHAR, max_length=..., enable_analyzer=True, analyzer_params={"type": "chinese"})`。
- 内置中文 analyzer 等价于 `{"tokenizer": "jieba", "filter": ["cnalphanumonly"]}`,不接受额外参数。
- BM25 Function:`Function(name="bm25_fn", function_type=FunctionType.BM25, input_field_names=["text"], output_field_names=["sparse"])`,输出字段必须是 `SPARSE_FLOAT_VECTOR`,输入必须是 VARCHAR/TEXT。
- 稀疏索引:`index_params.add_index(field_name="sparse", index_name="sparse_inverted", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25", params={"bm25_k1": 1.2, "bm25_b": 0.75})`。
- BM25 全文检索:`client.search(collection_name=..., data=["原始 query 文本"], anns_field="sparse", search_params={"metric_type": "BM25"}, limit=K, output_fields=[...])`;sparse 字段本身不能作为 output_fields。
- 混合检索:`MilvusClient.hybrid_search(collection_name, reqs: list[AnnSearchRequest], ranker=RRFRanker(k), limit=..., output_fields=[...])`;`AnnSearchRequest(data, anns_field, param, limit, expr=None)`。
- 已安装 `pymilvus==3.0.3`,本机签名与上一致;`ranker` 是客户端参数名(不是 `rerank`)。

## 现状与改动面

| 位置 | 现状 | 本章改动 |
|---|---|---|
| `app/rag/milvus_store.py` | `knowledge` 集合只有 `id/chunk_id/vector` | 新增 `knowledge_hybrid` 集合:文本、sparse、BM25 Function、标量字段;保留旧集合不动 |
| `app/rag/retrieval.py` | dense 单路 Top-3 + 阈值 | 改为四策略统一入口;默认 `hybrid_rerank` |
| `app/rag/vectorize.py` | 只写向量 | 写入 `text/category/content_type/section_path`;新增全量重建入口 |
| `app/rag/splitter.py` | manual/policy 的 category 只从标题路径推导 | frontmatter 显式 category 对所有 content_type 优先,无则维持原推导 |
| `app/tools/faq.py` | 返回「问/答」纯文本 | 返回带 `[n]` 的证据;可选 `category/content_type`;写入引用收集器 |
| `app/tools/context.py` | 只有 `current_conversation_id` | 新增可变 `CitationCollector` contextvar |
| `app/services/chat.py` | 单轮工具调用后直接生成 | 工具执行后下发 sources、跑自评、必要时拒答落池;接受 `self_check_llm` |
| `app/main.py` | 只建主 LLM | 新增 `get_self_check_llm_dep`(temperature=0)并传入 `stream_chat` |
| `app/prompts/customer_service.py` | 既有行为约束 | 增加引用、拒答、负面知识、型号问题路由 |
| `app/config.py` | 无混合/重排配置 | 增加召回深度、RRF k、重排模型与 key 回退 |
| `app/models/` + `sql/` | 无本章两表 | 新增 ORM、DDL、init 脚本 |
| `frontend/src` | 纯文本答案 | 引用 chip + 来源展开 + 点赞/点踩 |
| `tests/eval/` + `docs/` | 只有 FAQ 检索样例 | 新增 ch04 评估集、四策略报告 |
| `dev-notes/` + `docs/superpowers/` | ch04 是知识录入页 | 并入 ch03;ch04 让位本章 |

## 数据模型

### MySQL(用户 DDL,权威)

新增 `sql/ch04_rag_advanced.sql` 与 `app/db/init_ch04.py`(幂等 apply;两表全缺则建,只缺一张则报错;沿用 ch03 init 的写法)。DDL 原文:

```sql
-- =============================================================
-- ch04 · RAG 进阶 · 建表 DDL
-- 本章新建:low_confidence_questions(低置信度问题池)、faith_cases(编造个案台账)
-- 生成阶段模型自评知识不够答就拒答,把原话落进这张池子,是 ch09 数据飞轮的入口
-- 本章只靠 useful 自评判入池(useful=false 才入,不单独存该字段);ch09 会给这张表 ALTER 加归并字段
-- =============================================================

-- 确保中文 ENUM 定义值/DEFAULT/COMMENT 按 utf8mb4 解析
-- (否则 latin1 默认的 mysql client 会把中文 double-encode,ENUM 值存成乱码)
SET NAMES utf8mb4;

-- 2026-11-16 用户拍板:现有 conversations.id 是 signed BIGINT(不是 UNSIGNED),
-- 新表 conversation_id 改为 signed BIGINT 以匹配 FK;其余 DDL 保持用户原文。
CREATE TABLE low_confidence_questions (
  id              BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '主键',
  conversation_id BIGINT          NULL                    COMMENT '来源会话',
  raw_question    TEXT            NOT NULL                COMMENT '用户原话,带情绪口语',
  source          ENUM('retrieval_low_conf','self_check','user_feedback') NOT NULL COMMENT '入池入口:检索证据低 / 生成自评不足 / 用户反馈未解决',
  reason          TEXT            NULL                    COMMENT '判不能的原因,留作复盘',
  created_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '入池时间',
  PRIMARY KEY (id),
  KEY idx_source (source),
  KEY idx_created_at (created_at),
  CONSTRAINT fk_lcq_conversation FOREIGN KEY (conversation_id) REFERENCES conversations (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='低置信度问题池';

-- ch04 编造个案台账:忠实度裁判判出的每条编造,不只留在这一轮的报告里,进表长期管理。
--
-- 为什么要一张表:报告是产物,重跑一次就被覆盖,上一轮判出的个案连带它的处置状态一起没了。
-- 而这些个案的价值恰恰在跨轮追溯——同一道题反复被判编造,说明库里那一格一直没补对。
--
-- 一题一行(uk_eval_id):同一道题再次被判编造不新增行,只把答案/理由/角标快照更新成最近一次、
-- seen_count 加一。已经标「已解决」的题又被判出来,状态自动退回「未解决」——那是复发,
-- 不是新问题,得让它重新出现在待处理列表里。
--
-- citations 存的是这一轮喂给模型的 Top-K 证据**全集**(答案里的角标 [n] 就是这份列表的序号)。
-- 裁判说「证据里没有」,追溯时必须能当场看到当时喂进去的到底是什么,不能只留一句结论;
-- 而且要看得出「手里有哪几条、实际只引了哪几条」——没被引用的那些同样是判断依据。
CREATE TABLE faith_cases (
  id            BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '主键',
  eval_id       VARCHAR(16)     NOT NULL                COMMENT '评估集题号,如 A43;一题一行',
  bucket        VARCHAR(24)     NOT NULL                COMMENT '题目所属桶:A_policy / B_model / C_colloquial / E_multi',
  query         VARCHAR(512)    NOT NULL                COMMENT '用户问题原文',
  strategy      VARCHAR(24)     NOT NULL DEFAULT 'hybrid_rerank' COMMENT '产出这条答案的检索策略',
  answer        TEXT            NOT NULL                COMMENT '被判编造的那版生成答案原文',
  reason        TEXT            NOT NULL                COMMENT '裁判给的理由:编在哪一句',
  citations     JSON            NULL                    COMMENT '这一轮喂给模型的 Top-K 证据全集快照:[{n,chunk_id,section_path,question,answer}];答案里的角标 [n] 就是这份列表的序号,答案通常只引用其中两三条;老数据没记为 NULL',
  judge_model   VARCHAR(64)     NULL                    COMMENT '判这条的裁判模型',
  status        ENUM('未解决','已解决','无需解决') NOT NULL DEFAULT '未解决' COMMENT '处置状态,人工点按钮改',
  seen_count    INT UNSIGNED    NOT NULL DEFAULT 1      COMMENT '被判编造的累计次数(跨轮)',
  first_seen_at DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '第一次被判编造的时间',
  last_seen_at  DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '最近一次被判编造的时间',
  resolution    VARCHAR(300)    NULL                    COMMENT '处置说明:标已解决要写清怎么解决的,标无需解决要写清为什么不用改;退回未解决时清空。空着的处置在台账上等于没有交代',
  resolved_at   DATETIME        NULL                    COMMENT '最近一次被标为已解决/无需解决的时间;复发后仍保留,用来标「复发」',
  PRIMARY KEY (id),
  UNIQUE KEY uk_eval_id (eval_id),
  KEY idx_status (status),
  KEY idx_last_seen_at (last_seen_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='ch04 忠实度编造个案台账';
```

ORM 映射:

- `LowConfidenceQuestion`:`conversation_id` 可空、FK `conversations.id`;`source` 用 `Enum(..., name="low_confidence_source")`;`raw_question` / `reason` 为 `Text`;`created_at` 用 `func.now()`。
- `FaithCase`:`citations` 用 SQLAlchemy `JSON`;`status` 用中文枚举 `Enum("未解决","已解决","无需解决", name="faith_case_status")`;其余按 DDL。
- `eval_id` 唯一;upsert 逻辑见「评估体系」。

### Milvus `knowledge_hybrid`

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | VARCHAR(64), PK | `chunk_{knowledge_chunks.id}`,与 MySQL 对齐 |
| `chunk_id` | INT64 | MySQL 主键 |
| `vector` | FLOAT_VECTOR(1024) | BGE-M3 dense 向量,COSINE |
| `text` | VARCHAR(8192), `enable_analyzer=True`, `analyzer_params={"type":"chinese"}` | 与向量化文本同一份:`category\nquestions\nanswer` |
| `sparse` | SPARSE_FLOAT_VECTOR | BM25 Function 输出 |
| `category` | VARCHAR(255) | 前置过滤 + 展示 |
| `content_type` | VARCHAR(32) | 前置过滤 + 展示 |
| `section_path` | VARCHAR(512) | 引用展示 |

- `Function(name="bm25_fn", function_type=FunctionType.BM25, input_field_names=["text"], output_field_names=["sparse"])`。
- 索引:`vector` FLAT/COSINE;`sparse` `SPARSE_INVERTED_INDEX`/BM25(k1=1.2,b=0.75);`category`、`content_type` 建 INVERTED 标量索引(便于过滤)。
- `consistency_level="Strong"`,与现有集合一致。
- 旧 `knowledge` 集合保留,不删、不改;配置新增 `MILVUS_HYBRID_COLLECTION=knowledge_hybrid`。

## 组件设计

### 1. 配置(`app/config.py`)

新增(默认值即本章契约):

```python
milvus_hybrid_collection: str = "knowledge_hybrid"
hybrid_recall_top_k: int = 50
hybrid_rrf_k: int = 60
rerank_top_n: int = 10
rerank_model: str = "BAAI/bge-reranker-v2-m3"
rerank_base_url: str = ""        # 空则回退 embedding_base_url
rerank_api_key: str = ""         # 空则回退 embedding_api_key
rerank_timeout_s: float = 10.0
query_rewrite_enabled: bool = True
```

- `get_vector_store()` 改用 `milvus_hybrid_collection`;旧的 `milvus_collection` 字段保留给历史代码/测试引用,不参与本章检索。
- `build_llm(settings, temperature=0.7)` 增加 `temperature` 参数;自评与 Faithfulness 裁判使用 `temperature=0.0` 的独立实例(同 provider/model),避免裁判随机性。
- `.env.example` 同步新增;`.env` 不提交。
- `TOOL_TIMEOUTS["query_faq"]` 从 8.0 提到 **25.0** 秒,并对 `query_faq` 关闭 `execute_tool` 的整工具重试(`retries=0`):query 改写 + 双路召回 + 重排 + 回表比 dense 单路慢;25 秒覆盖尾延迟,且只发生一次,避免重试把整条链路再跑一遍。25 秒低于前端 30 秒无数据超时。

### 2. Milvus 封装(`app/rag/milvus_store.py`)

`VectorItem` 扩为:

```python
@dataclass(frozen=True)
class VectorItem:
    vector_id: str
    chunk_id: int
    vector: list[float]
    text: str
    category: str
    content_type: str | None
    section_path: str | None
```

新增方法(保持现有 `upsert/count/delete_ids` 兼容):

```python
def dense_search(self, vector, top_k, expr=None) -> list[SearchHit]
def bm25_search(self, query_text, top_k, expr=None) -> list[SearchHit]
def hybrid_search(self, vector, query_text, top_k, expr=None, rrf_k=60) -> list[SearchHit]
```

实现要点:

- `dense_search`:`client.search(..., data=[vector], anns_field="vector", search_params={"metric_type":"COSINE"}, limit=top_k, filter=expr or "", output_fields=["chunk_id"])`。
- `bm25_search`:`client.search(..., data=[query_text], anns_field="sparse", search_params={"metric_type":"BM25"}, limit=top_k, filter=expr or "", output_fields=["chunk_id"])`。
- `hybrid_search`:
  ```python
  reqs = [
      AnnSearchRequest([vector], "vector", {"metric_type": "COSINE"}, limit=top_k, expr=expr),
      AnnSearchRequest([query_text], "sparse", {"metric_type": "BM25"}, limit=top_k, expr=expr),
  ]
  client.hybrid_search(collection, reqs, ranker=RRFRanker(k=rrf_k), limit=top_k, output_fields=["chunk_id"])
  ```
- 结果解析沿用现有 `hit["distance"]` + `entity["chunk_id"]` 兼容写法;hybrid 的 `distance` 是 RRF 分数。
- `ensure_collection` 按上表建 schema/function/index;已有集合只 load,不重建。

### 3. 全量重建(`app/rag/vectorize.py` + `app/jobs/reindex_hybrid.py`)

- `vectorize_pending` 写入时带上 `text/category/content_type/section_path`;行为仍只处理 `pending`,`vector_id` 回填、`done` 标记不变。
- 新增 `vectorize_all(session, embedder, store, batch_size=16)`:按主键分批读取**全部** chunk,重新 embed + upsert,用于把 ch03 已有的 20 条(含 mined)迁到 `knowledge_hybrid`;不依赖 `vectorize_status`。
- 迁移选择**重新 embed**,不从旧集合复制向量:当前语料只有 20 条,成本可忽略;重新 embed 能保证向量、BM25 文本、元数据来自同一份 MySQL 行,避免复制路径掩盖数据不一致。真实大数据量可后续加 `--reuse-vectors`。
- 新 CLI:`python -m app.jobs.reindex_hybrid [--recreate]`。
  - 默认 upsert 覆盖,幂等;`--recreate` 只 drop/recreate `knowledge_hybrid`(不碰旧 `knowledge`),执行前打印明确的 destructive 提示并要求 `--yes`(无 `--yes` 则退出)。
  - 输出 `{chunks, embedded, upserted, elapsed_s}`。
- 建库命令 `build_knowledge` 后续也写 `knowledge_hybrid`(因为 `get_vector_store()` 已切换),新文档/上传重建自动带 BM25 字段。

### 4. Query 理解(`app/rag/query_understanding.py`)

```python
@dataclass(frozen=True)
class QueryAnalysis:
    original: str
    rewritten: str
    keywords: list[str]
    rewritten_by: Literal["llm", "fallback"]
```

- 一次 LLM 调用,`with_structured_output(..., method="json_mode")`(DeepSeek 仅支持 json_mode,沿用 `extract` 的写法)。
- Prompt 要求:
  - 把口语/模糊问法归一成一句标准客服问法,**不得新增事实**;
  - 输出 `keywords`:同义/近义检索词(如 邮费→运费、配送费;到账→退款、原路退回);
  - 只输出 JSON。
- **Query 理解不产出 `category` / `content_type` 过滤条件**。LLM 猜出来的品类名与库里的真实 `category` 值很容易对不上(例如把 `退款` 猜成品类,而库里是 `退货政策`),精确过滤会直接清空候选。过滤只来自两个明确入口:
  1. `query_faq(keyword, category=..., content_type=...)` 的显式参数(由聊天模型在用户明确指定品类时传入);
  2. 评估集的 `filters` 字段(直接调用 `retrieve_hybrid`)。
- 过滤语义:精确匹配;传了不存在的值就是 0 条候选,由自评/拒答路径处理,不静默去掉过滤。
- 静态同义词表 `SYNONYMS` 作为确定性兜底:即使 LLM 漏掉,也对原始 query 做子串扩展。LLM 失败时 `rewritten=original`,`rewritten_by="fallback"`,仍可检索。
- 同义词只拼到 BM25 查询串:`bm25_query = " ".join([rewritten, *keywords, *static_expansions])`;dense 只用 `rewritten`。入库侧完全不变。
- 测试用 fake LLM,覆盖:正常 JSON、非法 JSON 回退、同义词去重、`rewritten_by` 标记;不测 category 透传(该字段已移除)。

#### 新样本文档的 category 归属

- 现有 `splitter._build_chunk` 只对 `content_type == "faq"` 使用 frontmatter `category`;manual/policy 一律用标题路径。这样 `knowledge/商品型号与参数.md` 即使写了 `category: 商品`,`category` 也会变成 `商品型号与参数`,按 `商品` 过滤会落空。
- 本章做一个小而兼容的调整:**任何 content_type 下,frontmatter 显式 `category` 优先;没有时才按原规则从标题路径推导**。
  - 现有 `商品FAQ.md` 本来就有 `category: 商品`,行为不变;
  - 现有 policy/manual 文档没有 frontmatter category,行为不变;
  - 新文档因此得到 `category=商品`,与 B 桶过滤用例一致。
- 该改动配单测:显式 category 优先、无 category 时 manual/policy 仍按标题路径、FAQ 行为不变。

### 5. 重排(`app/rag/rerank.py`)

```python
class RerankError(RuntimeError): ...

@dataclass(frozen=True)
class RerankHit:
    index: int
    score: float

class SiliconFlowReranker:
    def rerank(self, query: str, documents: list[str], top_n: int | None = None) -> list[RerankHit]
```

- 请求体:`{"model": settings.rerank_model, "query": query, "documents": documents, "top_n": top_n, "return_documents": False}`。
- 解析 `results`,按 `relevance_score` 降序返回;校验 `index` 范围。
- 429/5xx/网络错误指数退避重试(最多 2 次,单次超时 `rerank_timeout_s=10.0`),其他错误直接 `RerankError`。
- 失败降级:检索层捕获 `RerankError`,记录日志,退回 hybrid RRF 顺序,并在 `RetrievalResult.rerank_failed=True` 上透出;不打断对话。
- 评估时 `rerank_failed=True` 的 (case, strategy) 不进 `hybrid_rerank` 的指标聚合,报告单列 fallback 数量与 case id;不能把 RRF 顺序冒充成重排结果参与对比。
- 文档文本:`f"{section_path or ''}\n{questions}\n{answer}"`,与入库文本同源但加章节路径。

### 6. 四策略检索(`app/rag/retrieval.py`)

```python
class RetrievalStrategy(str, Enum):
    DENSE = "dense"
    BM25 = "bm25"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid_rerank"

@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: int
    score: float
    questions: str
    answer: str
    section_path: str | None
    category: str
    content_type: str | None

@dataclass(frozen=True)
class RetrievalResult:
    chunks: list[RetrievedChunk]
    strategy: RetrievalStrategy
    analysis: QueryAnalysis
    rerank_failed: bool = False

def retrieve_hybrid(
    query: str,
    *,
    strategy: RetrievalStrategy = RetrievalStrategy.HYBRID_RERANK,
    session=None,
    embedder=None,
    store=None,
    reranker=None,
    analyzer=None,
    analysis: QueryAnalysis | None = None,
    category: str | None = None,
    content_type: str | None = None,
    recall_top_k: int | None = None,
    top_k: int | None = None,
) -> RetrievalResult:
```

执行顺序:

1. `analysis = analysis or analyze(query, ...)`:传了就用注入结果,没传才调 LLM。评估集每个 case **只调一次** `analyze()`,再把同一个 `QueryAnalysis` 传给四种策略,保证改写完全一致;线上 `query_faq` 不预计算,由 `retrieve_hybrid` 自己分析。
2. 组 `expr`:category/content_type 精确匹配;字符串转义用参数化 `expr_params` 或安全引号转义。过滤只来自显式参数,不读 `analysis`。
3. 按策略:
   - `dense`:dense Top-50,按 cosine 取 Top-10。
   - `bm25`:BM25 Top-50,按 BM25 取 Top-10。
   - `hybrid`:`hybrid_search` 两路 Top-50,RRF 取 Top-10。
   - `hybrid_rerank`:`hybrid_search` 两路 Top-50 → 用 MySQL chunk 文本重排 Top-10。
4. 用 `chunk_id` 回 MySQL 取 `questions/answer/section_path/category/content_type`;Milvus 有、MySQL 无的 id 跳过(沿用现有竞态处理)。
5. 返回 `RetrievalResult`。`chunks` 按策略分数降序;`score` 语义随策略不同(dense cosine / BM25 / RRF / rerank),跨策略不可直接比较。
6. 重排失败时 `rerank_failed=True`,`chunks` 退回 hybrid RRF 顺序;调用方必须把这个标记透出到日志/评估报告,不能假装 hybrid_rerank 正常跑完。

- 现有 `retrieve()` 保留为 `strategy=HYBRID_RERANK` 的薄封装,返回 `.chunks`,保持 `query_faq` 之外的调用方兼容;旧 `faq_top_k`/`faq_score_threshold` 只用于 dense-only 兼容路径,**不用于新默认路径**(混合/重排不再用 0.5 阈值砍候选;是否答由自评判定)。

### 7. 引用收集与工具改造

#### CitationCollector(`app/tools/context.py`)

```python
@dataclass
class CitationCollector:
    sources: list[dict] = field(default_factory=list)
    _by_chunk: dict[int, int] = field(default_factory=dict)
    faq_called: bool = False

    def add(self, chunks: list[RetrievedChunk]) -> list[int]: ...
    def payload(self) -> list[dict]: ...

current_citation_collector: ContextVar[CitationCollector | None] = ContextVar(..., default=None)
```

- 关键点:`execute_tool` 在子线程里执行工具,但会 `copy_context()`;context 里放的是**可变对象引用**,所以工具在子线程里 `collector.add(...)` 的修改对主线程可见。不要放不可变值或重新 `set()` 一个新对象。
- `add` 按 `chunk_id` 去重并复用编号;`faq_called=True` 即使没有命中也要置位,供自评判断。

#### `query_faq`(`app/tools/faq.py`)

```python
@tool
def query_faq(
    keyword: str,
    category: str | None = None,
    content_type: str | None = None,
) -> str:
```

- 调 `retrieve_hybrid(keyword, category=category, content_type=content_type)`,取 `.chunks`。
- 对返回的 Top-10 先做 `order_for_prompt()`,再 `collector.add(ordered)` 拿 `[n]`,格式:
  ```
  [1] 章节:退货政策 > 退款说明
  问:退款说明
  答:退货验收通过后,退款将在 1-3 个工作日内按原支付路径原路退回……
  ```
- 无命中:`collector.faq_called=True`,返回 `_MISS_TEXT`,不加证据。
- 基础设施异常:返回 `_ERROR_TEXT`,同样置位;该路径**不写低置信度池**(见「自评与拒答」对 infra 失败的区分)。
- `_MISS_TEXT` / `_ERROR_TEXT` 提升为模块级公开常量(如 `MISS_TEXT` / `ERROR_TEXT`),`stream_chat` 用它区分「真实无命中」与「知识库不可用」,不再靠字符串匹配私有变量。
- 工具描述更新:告知模型 `[n]` 是引用编号、回答必须引用、`category/content_type` 是精确匹配、仅在用户明确指定且确信值正确时才传,不确定就省略。
- 直接调用(测试/脚本)没有 collector 时,内部临时建一个,编号从 1 开始。
- 超时:注册表把 `query_faq` 的 `TOOL_TIMEOUTS` 设为 **25.0 秒**,并由 `stream_chat` 对 `query_faq` 显式传 `retries=0`。理由:整条链路内部已有 embedding/rerank 重试,整工具重试会把改写 + 召回 + 重排再跑一遍,还可能让超时线程继续往 collector 写;`retries=0` 让超时只发生一次。25 秒仍低于前端 30 秒无数据超时,且工具帧在调用前已下发,前端计时会从工具帧重新起算。

#### Prompt 组装顺序(`order_for_prompt`)

- 输入按分数降序的列表,输出「最强在首尾」的顺序:
  - 第 1 名放第 1 位,第 2 名放最后,第 3 名放第 2 位,第 4 名放倒数第 2…… 
  - 即 `[1,3,5,...,6,4,2]`。
- 引用编号、tool 结果、`sources` 帧、前端来源列表全部使用这个顺序;评估的 Recall/MRR 仍用分数顺序。
- 单测覆盖 n=0/1/2/10 的精确顺序。

### 8. 生成前自评与拒答(`app/rag/self_check.py` + `app/services/chat.py`)

```python
class EvidenceAssessment(BaseModel):
    useful: bool
    reason: str

def assess_evidence(llm, question: str, sources: list[dict]) -> EvidenceAssessment
```

- 一次 `with_structured_output(..., method="json_mode")` 调用;Prompt:
  - 给出用户问题 + 本轮全部证据(`n/section_path/answer/score`);
  - 判断「仅凭这些证据能否完整、准确回答」;
  - 证据不足、只沾边、互相矛盾、需要库外知识 → `useful=false`,并给出简短 `reason`;
  - 只输出 JSON。
- 失败重试 1 次;仍失败 → 保守拒答,`reason="自评调用失败,保守拒答"`。

`stream_chat` 改动点:

1. 工具执行前建 `CitationCollector` 并 `set` 到 contextvar;对 `query_faq` 调 `execute_tool(..., timeout_s=25, retries=0)`。
2. 工具执行后,若 `collector.faq_called`:
   - **先快照** `sources = list(collector.sources)`,再 `yield {"type":"sources","sources": sources}`。超时的 daemon 线程可能还在跑并继续往 collector 写,快照保证本轮自评看到的是稳定结果。
   - **基础设施失败分流**:如果 `query_faq` 返回 `_ERROR_TEXT` 或 `execute_tool` 返回 `"工具执行失败"`,且 `sources` 为空,不跑自评、不写低置信度池;直接返回 `"知识库暂时不可用,请稍后重试或联系人工客服。"`,记 `logger.exception/warning`,本轮结束。超时/宕机不是知识缺口,不能污染 `low_confidence_questions`。
   - 真实无命中(`_MISS_TEXT`)或证据不足:跑 `assess_evidence(self_check_llm, message, sources)`。
   - `useful=false`:
     - 调 `record_low_confidence(session, cid, message, reason)` 写 `low_confidence_questions`(`source="self_check"`);写入失败只记日志,不打断拒答;
     - 用 `other_results_exist = any(tc["name"] != "query_faq" for tc, _ in results)` 判断本轮是否还有其他工具的有效结果;
     - **没有其他结果**:生成拒答文案 `"抱歉,知识库暂未收录足够的信息来确认这个问题。建议您联系人工客服进一步处理。"`,`yield` 并持久化 assistant 消息,`return`;
     - **有其他结果**(混合意图,例如同时查了订单和知识库):不整轮丢弃。在 `final_messages` 末尾追加一条 `SystemMessage`,内容是「本轮知识库证据不足:涉及知识库的问题必须回答『无法确认』,不得猜测;可以回答其他工具已返回的事实;原问题已记入低置信度问题池」,然后走最终生成。这样订单/物流结果不会被知识库拒答连带丢掉。
   - **自评重试后仍失败**:保守拒答,`reason="自评调用失败,保守拒答"` 只写日志,**不写**低置信度池(infra 失败不是知识缺口);有其他工具结果时同样走上面的 SystemMessage 约束路径,没有则返回拒答文案。
3. 自评通过 → 走现有最终生成流程。
4. 未调用 `query_faq` → 不跑自评、不发 sources(订单/物流/工单类问题不受影响)。
5. 生成完成后做一次**引用有效性检查**:用正则抓最终答案里的 `[n]`,合法编号是本轮 `sources` 的 `n` 集合。发现未知编号只 `logger.warning` 并计数(不重生成、不阻断流);前端只把合法编号渲染成可点 chip,未知编号按普通文本显示;评估报告单独统计 `citation_validity`。这是本章对「流式生成无法回滚已发 token」的折中:不假装校验,但把违规变成可观测指标。

- `record_low_confidence` 放在独立 service(如 `app/services/low_confidence.py`),便于单测与异常隔离。
- 自评使用 `temperature=0.0` 的独立 LLM 实例(同 provider/model),通过 `stream_chat` 的新参数 `self_check_llm` 注入;未注入时回退用主 `llm`(测试友好)。`main.py` 新增 `get_self_check_llm_dep`,内部 `build_llm(settings, temperature=0.0)`,并传给 `stream_chat`;`stream_chat` 签名变为 `stream_chat(llm, session, tools, system_prompt, user_id, message, max_tokens, self_check_llm=None)`。
- 拒答消息不写 `sources` 以外的东西;前端正常显示。
- 已知风险与缓解:只有模型实际调用 `query_faq` 才触发自评。System Prompt 强化「知识类问题必须先查」,并由 `app/rag/router.py::is_pure_knowledge_question` 做确定性路由:纯知识问题在首次模型调用上强制 `tool_choice=query_faq`,订单/物流/混合意图保持 auto;强行选择不可用时记录并回退,不伪造 sources。评估集生成段直接走证据,不受此影响;现场验收检查 SSE 里确实出现 `query_faq` 工具帧。

### 9. System Prompt(`app/prompts/customer_service.py`)

新增/调整:

```
【知识库与引用】
- 涉及政策、流程、费用、时效、商品型号参数、售后规则,必须先调用 query_faq 查知识库;query_product 只用于查询某个订单商品的实时价格/库存等交易信息。
- query_faq 返回的证据带 [n] 编号。回答中的每个事实必须能对应到本轮证据的 [n];编号只能来自本轮工具结果,不得编造。
- **每一轮知识类问题都必须使用本轮检索/证据结果**:即使历史对话里有相似答案,也不得复用历史结论或历史 [n];本轮没有 [n] 时不要输出 [n]。
- 证据不足以回答时,明确说「知识库暂未收录/无法确认」,并建议转人工;不得用常识或推测补全。
- 用户明确指定品类时,可把 category/content_type 传给 query_faq;不确定就不要过滤。

【禁止承诺的负面知识】
- 不承诺具体到账时间;只复述政策时效,并说明以银行/支付平台为准。
- 不承诺一定退款、一定赔付、一定换新。
- 不承诺当天到货或具体送达时间。
- 不承诺知识库未列明的优惠、赠品、补贴、赔付标准。
- 不编造订单号、物流单号、金额、处理结果。
```

- 保留既有售后范围、工具使用、转人工等约束。
- Prompt 常量拆成:`_TOOL_ROUTING_RULES`(query_faq/query_product 路由 + 品类过滤)只进 `render_system_prompt()`;`_CITATION_RULES`(本轮 [n]、禁止历史复用、证据不足)与 `_NEGATIVE_RULES` 进两个 renderer。`render_answer_system_prompt()` 用于评估生成,不绑定工具,因此不包含「必须先调用 query_faq」,但保留同样的引用/拒答约束。
- Prompt 单测断言关键约束存在与渲染正常;实际行为用标注样例/评估集验证。

### 10. 评估体系

#### 评估集 `tests/eval/rag_cases.json`

结构:

```json
{
  "version": 1,
  "buckets": {
    "A_policy": "政策条款直问",
    "B_model": "具体型号/专有名词",
    "C_colloquial": "口语/同义改写",
    "D_unanswerable": "知识库没有、应拒答",
    "E_multi": "多跳/多条款组合"
  },
  "cases": [
    {
      "id": "A01",
      "bucket": "A_policy",
      "difficulty": "easy",
      "query": "退款多久到账?",
      "answerable": true,
      "ground_truth": {
        "section_paths": ["退货政策 > 退款说明"],
        "must_contain": ["1-3 个工作日", "原支付路径"]
      },
      "filters": null,
      "notes": "直问政策条款"
    }
  ]
}
```

- 规模:35 条,5 桶 × 7 条;每桶覆盖 easy/medium/hard。
- `ground_truth.section_paths` 必须与 `split_document` 产出的 `section_path` 精确一致;`must_contain` 是答案原文的必备片段,用于人工复核与容错。
- `filters` 可选 `{"category": "...", "content_type": "..."}`,用于验证元数据前置过滤;同一 case 四策略都应用同样过滤。新文档的 `category=商品` 依赖上面的 splitter 调整;过滤 case 优先用它,避免用 policy/manual 的派生 category 造成歧义。
- D 桶 `answerable=false`,`ground_truth` 为空;评估拒答率,不算检索 Recall/MRR。
- 新增样本文档 `knowledge/商品型号与参数.md`(content_type: manual, category: 商品),含 `LX-A100` / `LX-B200` / `LX-C300` 的色温、功率、适用面积、保修期、调光、包装清单与型号对照表;B 桶的「具体型号」必须有原文可命中。

#### 评估任务 `app/jobs/eval_rag.py`

CLI:

```bash
python -m app.jobs.eval_rag \
  --cases tests/eval/rag_cases.json \
  --strategies dense,bm25,hybrid,hybrid_rerank \
  --out docs/superpowers/reports/2026-11-16-ch04-rag-eval.md
```

参数:

- `--limit N`:只跑前 N 条,用于冒烟。
- `--retrieval-only`:只跑检索,不生成、不裁判。
- `--no-faith-cases`:不写 `faith_cases`(默认写)。
- `--out PATH`:写 Markdown 报告;默认路径见上。

检索段:

- 每个 case **先调一次** `analyze(query)`,得到同一个 `QueryAnalysis`;四种策略都把它作为 `analysis=` 注入 `retrieve_hybrid`,避免四次 LLM 改写出四个不同 query。
- 对每个 case、每种策略,调 `retrieve_hybrid(..., analysis=analysis, strategy=...)`,取 Top-10。
- **按 section 去重匹配**,不按 chunk 计数:
  - 把 Top-K 结果按出现顺序转成 `section_path` 序列,同一路径只取首次出现;
  - 某个 ground-truth path 在序列里出现即算命中;
  - `Recall@K` = 命中的 ground-truth path 数 / ground-truth path 总数(多跳题才有分母 > 1);
  - `MRR@10` = 首个命中任一 ground-truth path 的 chunk 排名倒数,未命中为 0。
- 相关判定用 `section_path` **精确相等**;`must_contain` 只用于人工复核,不参与自动指标(避免答案文本变动导致指标漂移)。
- 按桶 + 总体聚合;D 桶单独列「检索返回条数/是否误召回」,不计入 Recall/MRR 均值。
- `RetrievalResult.rerank_failed=True` 的 (case, strategy) 从 `hybrid_rerank` 的 Recall/MRR/Faithfulness 聚合中排除,报告单列 fallback case id 与数量;`analysis.rewritten_by="fallback"` 的数量也单列。

生成段(顺序必须与线上一致:**先自评,再生成**):

- 对每个 case、每种策略:
  1. 用该策略 Top-10,按 `order_for_prompt` 组装证据;
  2. 调线上同款 `assess_evidence(judge_llm, query, sources)`;**所有 case 都跑,包括 D 桶**;
  3. `useful=false` → 记 refusal,不生成、不评 Faithfulness;
  4. `useful=true` → 调生成 Prompt,得到答案 + 引用编号,再交给 Faithfulness 裁判。
- 生成 Prompt 的组装:使用与线上相同的「知识库与引用 / 禁止承诺」约束,但**不绑定 tools**,把证据以与 `query_faq` 完全相同的 `[n] 章节/问/答` 块放进 HumanMessage。不伪造 tool call / ToolMessage,避免模型再发起工具调用;`[n]` 的语义与线上一致。
- 拒答定义:`assess_evidence.useful=false` → `refusal`;`assess_evidence` 重试后仍报错 → `self_check_error`,单独统计,不混入 refusal/faithfulness(与线上 infra 失败不落池一致)。若自评通过但模型自己在答案里说「无法确认」,记 `model_refusal` 并在报告中单列。
- 裁判 Prompt:只依据给出的证据,逐句判断事实性陈述是否有证据支持;返回 JSON `{faithful, unsupported_claims: [...], reason}`。裁判使用 `temperature=0.0` 的独立 LLM 实例。
- 指标:
  - `Faithfulness` = faithful 答案数 / 非拒答且非 judge_error 的答案数;
  - `Refusal rate` = pipeline refusal 数 / 总题数;
  - `Refusal accuracy`(D 桶)= 应拒答且 pipeline refusal 的比例;
  - `Citation validity` = 答案里合法 `[n]` 数 / 全部 `[n]` 数(无引用时不计);
  - `Judge error` 单列,不计入 Faithfulness。
- `faith_cases` 写入:每条 `faithful=false`、非拒答、非 judge_error 的答案,按 `eval_id` upsert:
  - 新题:插入,`seen_count=1`,`status='未解决'`,`first_seen_at=last_seen_at=now`;
  - 旧题:更新 `answer/reason/citations/strategy/judge_model`, `seen_count += 1`,`last_seen_at=now`,`status='未解决'`;保留 `resolved_at`(用于标记复发);
  - `citations` 存本轮 Top-10 证据全集 `[{n,chunk_id,section_path,question,answer}]`。
- 报告内容:总体四策略对比表 + 每桶四策略表 + D 桶拒答准确率 + Citation validity + `faith_cases` 新增/复发计数 + rerank fallback 计数/case id + rewrite fallback 计数 + 运行参数(模型、Top-K、RRF k、commit)。
- 成本估算(35 条 case、4 策略):35 次 query 改写 + 140 次自评 + 最多 140 次生成 + 最多 140 次 Faithfulness 裁判 + 35 次重排(只有 `hybrid_rerank` 调重排)+ 每组策略的向量化/检索调用。冒烟先 `--limit 3`。

### 11. 前端(Vibe Coding,不走 brainstorm/TDD/code review)

#### 数据契约

SSE 新增帧:

```json
{"type":"sources","sources":[
  {"n":1,"chunk_id":12,"section_path":"退货政策 > 退款说明",
   "question":"退款说明","answer":"退货验收通过后……","category":"退货政策",
   "content_type":"policy","score":0.91}
]}
```

- 该帧在工具执行后、最终答案 token 前到达;前端把它挂到接下来创建的 assistant 消息上。
- `Message` 类型扩展:
  ```ts
  export interface CitationSource { n: number; chunk_id: number; section_path: string | null; question: string; answer: string; category: string | null; content_type: string | null; score: number; }
  export type FeedbackValue = "up" | "down";
  // assistant 消息增加 sources?: CitationSource[]; feedback?: FeedbackValue;
  ```

#### 交互

- 引用编号:渲染答案时把 `[n]` 识别为可点 chip(不与 Markdown 链接冲突);点击展开/切换该条来源卡片,显示 `section_path` + 原文 `answer`;无对应 source 的编号按普通文本渲染。
- 来源卡片:答案气泡下方内联展开,一次只展开一条;再次点击同一编号收起。
- 满意度:每条 assistant 回答左下角 👍/👎;点击后所选按钮高亮、另一按钮禁用、显示「已反馈」;再次点击不改变状态。
- 纯前端采集:`localStorage` 写一条结构化记录 `{sessionId, messageId, value, content, createdAt}`(按 `<sessionId>:<messageId>` 去重);保留 `content` 是为了即使页面消息被清空,后续数据飞轮仍能知道用户对哪段回答点了什么。不上传后端。
- 流式期间不显示反馈按钮;完成后显示。

#### 实现落点

- `frontend/src/types.ts`、`frontend/src/api/chat.ts`(解析 sources 帧)、`frontend/src/hooks/useChat.ts`(pending sources、feedback action)、`frontend/src/components/MessageItem.tsx`(citation chip、来源卡片、反馈)、必要时 `MessageList.tsx` 传参。
- 验收:`npm run build`(tsc + vite)通过;浏览器手点引用与反馈正确;无 console error。

### 12. 文档归位(本章 Task 0)

- `dev-notes/ch04.md` 现有内容(知识录入页)整体并入 `dev-notes/ch03.md`,加标题 `## 补记:知识录入页(/kb,原 ch04)`;原 ch04 文件随后重置为本章开发记录。
- `git mv docs/superpowers/specs/2026-11-12-ecommerce-cs-ch04-knowledge-ui-design.md docs/superpowers/specs/2026-11-12-ecommerce-cs-ch03-knowledge-ui-design.md`
- `git mv docs/superpowers/plans/2026-11-13-ecommerce-cs-ch04-knowledge-ui.md docs/superpowers/plans/2026-11-13-ecommerce-cs-ch03-knowledge-ui.md`
- 用 `rg` 更新被移动文档的引用(dev-notes/ch03.md、相关 spec/plan 内的路径);保留 git 历史。
- 本章 spec/plan/dev-notes 一律用 ch04:RAG 进阶编号。

## 失败与降级策略

| 故障 | 行为 |
|---|---|
| Query 改写 LLM 失败/JSON 非法 | 回退原始 query + 静态同义词,`rewritten_by="fallback"`,继续检索 |
| 重排 API 失败/超时 | 记录日志,退回 hybrid RRF 顺序,检索仍返回 Top-10 |
| BM25 集合/Function 不可用 | `query_faq` 返回知识库不可用 → 保守拒答,记日志,不落池(infra 失败) |
| 自评调用失败 | 重试 1 次;仍失败 → 保守拒答,记 `reason="自评调用失败,保守拒答"`,不落池(infra 失败) |
| `low_confidence_questions` 写入失败 | 记日志,拒答照常返回,不打断本轮 |
| MySQL 缺 Milvus 返回的 chunk | 跳过该 id(沿用现有竞态处理) |
| Faithfulness 裁判失败 | 该条标记 `judge_error`,不计入 Faithfulness,报告单列 |
| `faith_cases` upsert 失败 | 记日志,报告继续;该条在报告中标记未落表 |

## 测试与验证策略

### 后端 TDD

按任务先写失败测试,再实现:

1. **DDL/ORM/init**:两表创建、幂等、半初始化报错;ORM 字段/枚举/JSON;FK 可空。
2. **Milvus schema**:fake client 断言 `text` 字段的 `enable_analyzer/analyzer_params`、BM25 Function、sparse 索引、标量字段、`VectorItem` payload。
3. **Reranker**:fake HTTP 断言请求体/`top_n`/响应解析/排序/重试/异常。
4. **Query 理解**:fake LLM 正常 JSON、非法 JSON 回退、同义词扩展、category 透传。
5. **四策略检索**:fake store/embedder/reranker 断言 dense 单路、BM25 单路、hybrid 两请求、rerank 调用、过滤 expr、Top-10、MySQL 回填与竞态跳过。
6. **引用收集**:编号去重、跨多次 query_faq 递增、`execute_tool` 子线程可见、`order_for_prompt` 顺序。
7. **自评/拒答**:useful true/false、自评异常、拒答文案、low-confidence 落库、未调用 query_faq 不触发、infra 失败不落池且返回「知识库不可用」、混合意图有其他工具结果时不整轮拒答、引用未知编号只告警不阻断。
8. **faith_cases upsert**:新增、seen_count、复发退回未解决、resolved_at 保留、citations JSON。
9. **评估打分**:section_path 去重匹配、Recall@K/MRR、多跳分母、桶聚合、D 桶拒答准确率、rerank_failed 排除、citation validity。
10. **Prompt**:关键约束存在、渲染正常。
11. **splitter category**:frontmatter category 对所有 content_type 优先;无 category 时 policy/manual 仍按标题路径;FAQ 行为不变。

全量回归必须保持 ch01–ch03 的 219 个测试绿(允许因 Milvus 集合名/工具签名变化做合理更新)。

### 非可单测任务(替代 TDD)

- **Query 改写 Prompt**:拿 10 条口语标注样例跑真实 LLM,记录 `rewritten/keywords` 与人工期望;结果写 dev-notes。
- **自评 Prompt**:拿 5 条可答 + 5 条不可答案例跑,记录 `useful` 与人工判断;结果写 dev-notes。
- **生成/裁判 Prompt**:由 `eval_rag.py` 的四策略报告验证,不另造单测。

### 前端

- `npm run build` 通过;手动验收引用点击、来源展开、👍/👎 点亮与锁定;不做自动化测试。

### 端到端验收

1. `python -m app.db.init_ch04` 建两表。
2. `python -m app.jobs.reindex_hybrid` 把现有 chunk 全量写入 `knowledge_hybrid`。
3. `python -m app.jobs.eval_rag --out ...` 四策略报告出数字(可先 `--limit 5` 冒烟)。
4. 问「LX-A100 的色温是多少」:BM25 路命中,回答带 `[n]`,引用可点开看到 `商品型号与参数 > LX-A100 落地灯` 原文。
5. 问「你们支持以旧换新吗」:明确拒答,`low_confidence_questions` 新增 `source='self_check'` 行,`conversation_id` 正确。
6. 聊天页 👍/👎:点亮、显示「已反馈」、锁定;刷新后本轮消息不存在属预期。
7. 全量 `pytest -q` 绿。

## 风险与待验证

1. **Milvus 3.0.3 + pymilvus 3.0.3 的 BM25/中文 analyzer 组合**:Context7 文档对应 v3.0.x,但需在 Task 1 先做 live smoke(建临时集合 → 插入两条中文/型号文本 → BM25 搜 `LX-A100` → 删除临时集合),失败则停下报告,不自行换方案。
2. **`hybrid_search` 的 `ranker` 参数名与返回结构**:已按本机 3.0.3 签名设计;Task 1 用真实 Milvus 验证。
3. **DeepSeek JSON 输出稳定性**:沿用 `method="json_mode"` + 重试;解析失败有回退。
4. **评估跑批成本/时长**:提供 `--limit` 与 `--retrieval-only`;完整四策略报告约 35 次改写 + 140 次自评 + 最多 140 次生成 + 最多 140 次裁判 + 35 次重排,跑前先 `--limit 3` 冒烟。
5. **Milvus 集合迁移**:不碰旧 `knowledge`;新集合 upsert 幂等,`--recreate` 仅作用于新集合且需 `--yes`。
6. **引用编号与多工具调用**:用可变 collector + chunk_id 去重保证全局唯一;若模型一轮调用多次 `query_faq`,编号继续递增,不重新从 1 开始。
7. **工具超时与 daemon 线程**:`execute_tool` 超时后 daemon 线程仍在跑,可能继续往 collector 写。已用「快照 sources + query_faq retries=0 + 25s 超时」降低影响;超时按 infra 失败处理,不写低置信度池。
8. **混合意图**:知识部分证据不足但订单/物流工具成功时,只对知识部分说无法确认,不丢弃其他结果;该行为由 `stream_chat` 单测覆盖(SystemMessage 注入路径)。
9. **引用合法性**:流式生成无法回滚已发 token,未知 `[n]` 只能告警 + 前端按纯文本渲染 + 评估统计 `citation_validity`;不重生成、不阻断流。
10. **模型不调用 query_faq**:自评闸门只覆盖实际调用;靠 System Prompt 强化知识类问题必须查,现场验收检查工具帧;若模型绕过,这是 Prompt 层问题,不在代码里做不可靠的关键词路由。

## 交付物

- 设计:`docs/superpowers/specs/2026-11-16-ecommerce-cs-ch04-rag-advanced-design.md`
- 计划:`docs/superpowers/plans/2026-11-16-ecommerce-cs-ch04-rag-advanced.md`
- 开发记录:`dev-notes/ch04.md`(阶段追记,不后补)
- 评估报告:`docs/superpowers/reports/2026-11-16-ch04-rag-eval.md`
- DDL:`sql/ch04_rag_advanced.sql`
- 演示命令、测试结果、验收数字:收尾写入 dev-notes/ch04.md
