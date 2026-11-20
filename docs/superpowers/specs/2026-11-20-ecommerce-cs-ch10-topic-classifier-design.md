# ch10 · 模型微调：多标签主题分类器（RoBERTa-wwm-ext 全参微调）设计

> 对齐参考 spec（课程 ch10 README）与本仓库 ch01–ch09 的既有结构。目标：把飞轮攒下的低置信度问题按 17 类主题批量归类，喂给飞轮后台看各主题分布，决定先补哪块知识。
> 日期：2026-11-20。基线分支：`feature/ch09-observability-flywheel`（未合），本章分支：`feature/ch10-topic-classifier`。

## 1. 意图与为什么

ch09 的飞轮把 bot 没答好的问题攒进 `low_confidence_questions`，再经标准化查重归并到 `review_queue`。人工逐条看池子决定补哪块知识，量一上来就顶不住。本章训一台小的多标签分类器（哈工大讯飞 RoBERTa-wwm-ext，全参微调），挂旁路按批量把池里的问题归进 17 类主题，结果落 `topic_classifications` 表，飞轮后台新增主题分布页，哪类堆得多一眼能看出。

为什么微调而不是提示词：说法无穷无尽、few-shot 举不完；方言、错别字、多诉求句容易判错且答案飘忽；海量逐条喂大模型又贵又慢。量大、类目固定、要便宜要快 → 微调小分类器。

**实时对话主链路不调它。** 意图识别（ch06，会话维度）不动；主题归类是单句维度的独立任务。

## 2. 目标、非目标、反目标

### 目标

1. 17 类权威类目表只有一份，数据处理、预标、训练、推理、评测、前端全部读它。
2. 从飞轮池捞真实问题，量不够由大模型照术语表补模拟问题；清洗、分层抽样、只扩训练集的增强四步可重跑。
3. 大模型预标 + 黄金样例闸 + 人工抽审得到多标签数据；标注遵守「字面提到几个诉求就打几个标签」。
4. RoBERTa-wwm-ext 全参微调，验证集盯分数、正则化 + 早停防过拟合。
5. 测试集按类目出精确率/召回率/F1、micro/macro、每类 2×2 混淆矩阵；错例可导出人工复核。
6. ONNX 导出 + 独立推理服务（:8110）+ 旁路批量归类 job，结果写 `topic_classifications`，幂等。
7. 飞轮后台加主题分布页 + 类目问题列表页；验收页四页可在浏览器里看九项实证并就地重跑。
8. 全程在 `dev-notes/ch10.md` 留痕，每个阶段一段。

### 非目标（本章不做）

- LoRA / QLoRA（大模型才用的 PEFT 手段；本章 102M 全参足够）。
- Embedding 微调、意图识别微调。
- 实时对话主链路接入分类器。
- 类目表治理后台（术语表以 `taxonomy.py` 代码为准）。
- 验收页鉴权、多用户、操作审计（本地开发后台单人使用；作业运行器安全边界靠白名单，不靠权限模型）。
- 作业排队与并发调度（同名作业拒绝重入即够）。

### 反目标（技术上做成了也算失败）

- 测试集指标靠合成题自嗨，却当成真实分布质量对外宣称。
- 训练/验证/测试有文本泄漏，所有分数不作数。
- 为了凑 F1 改测试集或按测试集调阈值/早停。
- 把分类器塞进 `/chat` 主链路或每轮对话同步调用。
- 模拟题生成/增强偷偷改变标签集合（多打或少打诉求）。
- 前端在接口里重算一份指标，和报告 `.json` 出现两个真相。

## 3. 约束与现状（本仓库实测）

- 环境：Windows 11、Git Bash、GNU Make 4.4.1；主 Python 3.13.13；GPU 为 RTX 4060 Laptop 8GB（CUDA 12.4 驱动）；`uv 0.13.0` 可用；uv 管理的 CPython 3.12.15 可用作 ML venv 兜底。
- 本仓库没有 `pyproject.toml` / `uv.lock`，依赖在 `requirements.txt`，主虚拟环境 `.venv`；测试命令 `.venv/Scripts/python.exe -m pytest -q`（ch09 基线 994 passed）。
- 后台不是 React：`app/api/*.py` 直出 `app/static/*.html`；`app/static/admin.js` 提供共享导航，目前 5 项（后台首页/待审队列/观测总览/知识录入/RAG 评估）。
- ORM 在 `app/models/*.py`，`app/models/__init__.py` 统一导出；`app/db/base.py` 是 DeclarativeBase；建表脚本按章分文件（`app/db/init_ch09.py` 是幂等样板）。
- 低置信池实测：`low_confidence_questions` **16 条，去重后 13 条原话 / 12 条标准化问法**；16 条全部已有 `matched_review_id`；`review_queue` 12 条；`knowledge_chunks` 45 条。参考 spec 里的「70 条」与本地库不一致，按本地实际走。
- 现有作业运行器 `app/services/admin_jobs.py` + `/api/admin/jobs` 是**进程内线程**跑 callable，适合主 `.venv` 的轻作业；ch10 的重作业在 `.venv-ml`，不能塞进主进程。
- 端口占用：8000 主应用、8101/8102 MCP、3000 Langfuse；**8110 空闲**，给分类服务。

## 4. 做法与取舍

### 4.1 推理服务：ONNX Runtime（参考 spec 选定）

- 备选 A：FastAPI + transformers，直接加载 `AutoModelForSequenceClassification`。实现短，但推理进程要背 torch/transformers。
- 备选 B：FastAPI + onnxruntime + tokenizers（**选定**）。训练后导出 ONNX，服务轻运行时；代价是导出、数值对齐、tokenizer 边界三块工作，用「ONNX vs torch 标签完全一致 + logits 容差」的导出校验兜住。
- 备选 C：训练完在 job 内加载模型归类。违背「导出成推理服务、挂旁路」，且主进程依赖隔离失效，不选。

### 4.2 数据处理：分阶段 JSONL 流水线（参考 spec 选定）

备选一次性内存脚本更短，但无法人工抽审、无法重跑单步、无法给验收页展示血缘。选定分阶段落盘 + manifest。

### 4.3 依赖隔离：独立 ML venv（本仓库适配，已确认）

参考 spec 写 `pyproject.toml` + `[dependency-groups] ml` + `uv run --group ml`。本仓库没有 pyproject，主 `.venv` 已承载 994 个测试；直接引入 uv 项目会重建/改写 `.venv`，风险高。**已与用户确认**：新增 `requirements-ch10.txt`，用 `uv venv --python 3.12 .venv-ml` 建独立环境（3.12 对 torch/onnxruntime 的 wheel 支持更稳），`uv pip install -r requirements-ch10.txt --python .venv-ml` 安装。主 `.venv` 零污染，不新增 pyproject。

### 4.4 作业运行器：子进程跑 make（参考 spec 选定）

参考 spec 要求页面按钮按的与终端敲的是同一条 make 命令，且重作业要能停止整条进程链。现有进程内 callable 运行器做不到（ML 依赖在主 venv 外）。新增 `app/core/jobs.py` + `app/api/jobs.py`：白名单作业名 → 写死的 make argv；`subprocess` 起独立会话；日志落 `log/acceptance/<job>.log`；同名作业拒绝重入；停止时 Windows 用 `taskkill /T /F`、POSIX 用 `killpg`。ch09 的 `/api/admin/jobs` 保持不动，两套运行器并存，本章不统一（记为已知债务）。

### 4.5 切分：标签组合分层（参考 spec 选定）

按标签组合分层，组合样本数 <10 的按「第一标签」（按 `TOPIC_CLASSES` 顺序取第一个命中类）归层；切完输出每类三份占比报告、跨集文本泄漏自检。自己不引额外分层包，纯 Python 实现 + 单测。

### 4.6 阈值与兜底（参考 spec 选定）

全局阈值在验证集上扫 0.30–0.70（步进 0.05），取 micro-F1 最优；存 `threshold.json`。17 类全部不过线时取最高分那一类兜底，绝不产出空标签。训练早停盯验证集 **micro-F1**，patience=2。

## 5. 权威类目表（17 类，全系统唯一）

落 `app/core/taxonomy.py`：`TOPIC_CLASSES`（17 个类目名的有序元组，顺序即 label id）、每类 `boundary`、`examples`、`LABEL2ID/ID2LABEL`。数据处理、预标、训练、推理、评测、API、前端全部 import 它，不许各自抄一份。

| # | 类目 | 边界说明 | 说法示例 |
|---|------|----------|----------|
| 1 | 退换货 | 退货、换货、退款怎么办；修归保修维修，退归这里 | 退货、退款、退钱、想退了、七天无理由还能退不 |
| 2 | 物流 | 货走到哪了、什么时候送到；运费的钱事归运费 | 快递、发货、到哪了、怎么还不动、海外直邮 |
| 3 | 尺码 | 大小、码数合不合适 | 猫窝买大了、猫别墅尺寸、项圈偏码、适合几斤的猫 |
| 4 | 发票 | 开票、抬头、报销凭证 | 开发票、发票抬头开错了、能开增值税发票吗、合并开票 |
| 5 | 质量问题 | 商品本身的毛病 | 开胶、破了个洞、有瑕疵、猫砂盆电机坏了 |
| 6 | 运费 | 运费谁出、运费险理赔；管的是钱，货走到哪了归物流 | 包邮吗、退货运费谁承担、运费险怎么赔 |
| 7 | 优惠活动 | 券和活动怎么用、能不能叠 | 优惠券、满减、活动价、能叠加用吗、双十一有活动吗 |
| 8 | 价保 | 买完降价了补不补差价 | 刚买就降价了、能补差价吗、保价期多久 |
| 9 | 支付 | 付款环节出的问题 | 付不了款、花呗分期、扣了两次钱、货到付款、数字人民币 |
| 10 | 订单修改 | 下单之后改信息、取消订单 | 改地址、改电话号码、订单还能取消吗 |
| 11 | 库存补货 | 有没有货、什么时候补 | 有货吗、断货了、什么时候补货、有现货吗 |
| 12 | 商品信息 | 材质、功能、用法 | 什么材质、怎么洗、冻干怎么保存、废砂盒多久倒、猫粮怎么选 |
| 13 | 保修维修 | 保修期限、维修换新；修归这里，退归退换货 | 保修多久、坏了能修吗、能换新吗 |
| 14 | 账号 | 登录、绑定、账号安全 | 登录不上、忘了密码、换绑手机号、注销账号 |
| 15 | 会员积分 | 会员权益、积分怎么用 | 积分怎么用、会员几级、积分能抵钱吗 |
| 16 | 评价 | 评价、晒单的规则 | 评价怎么改、追评在哪写、晒单有奖励吗 |
| 17 | 其他 | 上面都对不上的，先兜底 | 闲聊、转人工、客服几点上班 |

**标注铁律（字面规则）**：一句话字面提到几个诉求就打几个标签，一个不多一个不少。「买大了想退」→ 尺码 + 退换货；「这鞋买大了一码」→ 只标尺码，不许脑补退换货。类目间关联留给业务，标注不揽这个活。

**「其他」的语义（我的调用，可纠偏）**：「其他」是互斥兜底——只要命中任何一个具体类目，就不打「其他」；全部类目都不过线时，取最高分那一类，若最高分就是「其他」则兜底为「其他」。这样主题分布页的「其他」只表示真·对不上，不会吞掉具体类目的计数。预标 prompt、黄金样例、单测三处都按这条写。

## 6. 数据处理流水线（`scripts/ch10/`）

### 6.1 捞取与文本选择

- 分类旁路只处理已归并的问题：`low_confidence_questions.matched_review_id IS NOT NULL`，文本取 `review_queue.normalized_question`。未归并的不进分类，留到下一轮归并后再归——保证分类器只吃语义完整的问法。
- 训练语料同样优先标准化问法，未归并时退回 `raw_question`，保留 `question_id` 溯源。本地 16 条全部已归并。
- 真实问题按「用户选定 A」当普通样本混进 80/10/10；报告里明确写「测试指标来自合成主导数据（真实样本约 1%）」，真实问题整批跑归类只作为功能验收，不冒充质量分。

### 6.2 清洗（`build_corpus.py`）

1. **脱敏**：正则抹掉手机号、订单号、微信号/QQ，替换为 `[手机号]` / `[订单号]` / `[微信号]` / `[QQ号]`。
2. **去重**：完全相同文本合并，保留 `source_refs`（question_id / review_id 列表）与 `origin`。
3. **修错别字与格式**：大模型批量清洗，只修字面错误与乱格式，不改语义、不改口语风格；输出 `before`/`after` 对照落盘可复查。LLM 调用走仓库现有 `build_llm(settings)`（`OPENAI_BASE_URL` / `MODEL_NAME`），不引入参考 spec 的 `CHAT_BASE_URL` / `CHAT_MODEL`。

产物：`data/ch10/corpus_raw.jsonl` → `corpus_clean.jsonl`。

### 6.3 归并术语表 + 预标 + 人工抽审（`prelabel.py` / `validate_golden.py`）

- 预标 prompt 写死 17 类术语表（含边界说明与示例）、字面铁律、近邻边界（修/退、运费/物流、价保/优惠）、「其他」互斥兜底。结构化输出 `{"text": ..., "labels": [...]}`；校验 labels 是 17 类子集、无重复、按 `TOPIC_CLASSES` 顺序规范化。
- **黄金样例闸（替代 TDD 的验证步）**：由我起草 30 条覆盖多标签、方言、错别字、近邻类目、边界句的黄金样例，交用户审核/纠正后落 `golden_samples.jsonl`。预标 prompt 在黄金样例上**标签集合完全一致率 ≥ 80%** 才放行批量预标；不一致样本逐条落 `golden_report.json`。先例：ch09 `validate_flywheel_samples`。
- **人工抽审**：预标完按类目分层抽样导出 `data/ch10/sample_review.md`（句子 + 预标标签），**真实问题按 12 条去重后的标准化问法全量进抽审**（对应 16 条池行 / 13 条不同原话），模拟题每类抽 3 条。抽审清单由我起草，用户在对话里指出错标；修正写 `data/ch10/audit_corrections.json`（稳定 id → 纠正后标签），流水线重跑时应用并记录 `audited=true`。抽审一致率进报告。

### 6.4 模拟语料补足（`build_corpus.py` 的 synthetic 段）

- 每类正例目标约 100 个；真实问题预标后先计数，缺口由大模型照术语表造模拟问题补足，**生成时自带标签**（同样过字面铁律）。按类目正例计数配平，达标即停，避免某类超配。
- 刁钻样本按比例掺入：方言口语约 15%、错别字约 10%、多诉求多标签句约 15%。多诉求句必须真的命中两个以上类目（如「催下快递，对了这个有色差还能换不」→ 物流 + 质量问题 + 退换货？按字面：「催快递」物流、「有色差」质量问题、「能换不」退换货）。
- 模拟题生成后用同一预标 prompt 盲标一次做交叉校验；与生成标签不一致的进人工抽审，不直接丢弃（可能是生成标签错，也可能是预标错）。
- 模拟语料同样进人工抽审的分层抽样。

### 6.5 分层抽样划分（`build_dataset.py`）

- 切分前先做全局近重复合并：对清洗后文本做字符 3-gram Jaccard，≥0.85 的视为同一问法，保留一条并合并 `source_refs`；合并数进 manifest。这样跨集的近义改写不会成为「合法泄漏」。
- 合并真实 + 模拟 + 清洗后语料，再按标签组合分层 80/10/10；组合样本数 <10 的按第一标签归层。
- 切完输出 `dataset/manifest.json`：三份条数、每类三份占比、跨集文本重叠数（必须为 0）、每类是否在三份中都出现、近重复合并数。
- **泄漏自检是硬闸**：训练∩验证、训练∩测试、验证∩测试文本重叠必须为 0，不达标直接非零退出。

### 6.6 数据增强（只扩训练集）

- 大模型批量生成同义词替换 + 句式微调变体，**标签继承原文**；每条训练样本扩 1 个变体（训练集约翻倍）。验证集、测试集保持原样。
- 增强后用同一预标 prompt 盲标变体，标签集合与原文不一致的**丢弃并记数**（防止增强偷偷改诉求）；丢弃率进报告。

产物：`data/ch10/corpus_raw.jsonl` → `corpus_clean.jsonl` → `corpus_labeled.jsonl` → `dataset/{train,val,test}.jsonl` + `dataset/train_augmented.jsonl` + `sample_review.md` + `golden_samples.jsonl`。样本格式 `{"text": "...", "labels": ["尺码","退换货"], "id": "...", "origin": "real|synthetic", "source_refs": [...]}`。

### 6.7 生成物冻结、稳定 id 与重跑

- 模拟语料与增强变体生成一次后冻结在 `data/ch10/generated/<run-id>/`，`run-id` 由 taxonomy 版本 + 生成参数 + 时间戳组成；默认 `ch10-corpus` 复用既有冻结产物，不再烧 LLM。
- 样本 id 用内容哈希（`sha1(origin|text)` 前 12 位），同一文本重跑得到同一个 id；`audit_corrections.json` 按这个 id 生效。
- `--force` 重新生成会新建 run-id，并在启动时明确打印「将失效 N 条人工纠正」，同时把旧 corrections 归档到 `data/ch10/audit_corrections.<old-run>.json`；不静默丢纠正。
- LLM 批量调用按 10–20 条/次打包，带 `--max-llm-calls` 上限与断点续跑（已完成的批次落盘，重跑跳过）；上限或失败时非零退出并保留已生成部分。

## 7. 训练（`scripts/ch10/train.py`）

- 基座：`hfl/chinese-roberta-wwm-ext`；网络不通时 `HF_ENDPOINT=https://hf-mirror.com` 兜底。
- 方式：transformers `Trainer`，`problem_type="multi_label_classification"`（BCEWithLogitsLoss），17 维 sigmoid 输出，全参更新，不用 LoRA。
- 设备自适应：cuda → mps → cpu。本机走 CUDA；`fp16`（CUDA）优先，失败回退 `bf16`/fp32。Python 3.12 ML venv。
- 防过拟合：`weight_decay=0.01` + `EarlyStoppingCallback`（盯验证集 micro-F1，patience=2）+ 每 epoch 评验证集 + `load_best_model_at_end`。
- 超参基线：lr=2e-5，batch=16，max_length=128，epochs≤8（早停收手），warmup_ratio=0.1，seed=42。
- 阈值：训完先用 torch 在验证集上扫一版全局最优阈值（0.30–0.70 步进 0.05），写入 `train_report.json` 的 `torch_threshold_scan`；**导出 ONNX 后用 ONNX 在验证集重扫并覆盖 `data/ch10/model/threshold.json`**（记 `source: "onnx"`、阈值、val micro-F1、九档扫描表）。理由：线上服务跑 ONNX，阈值和测试指标必须以 ONNX 为准，torch 那版只作对照。全类无一过线时取分数最高的一类兜底。
- 产物：`data/ch10/model/`（权重 + tokenizer + threshold）、`data/ch10/reports/train_report.json`（每 epoch 验证分数、最佳 checkpoint、早停触发、耗时）。

## 8. 评测（`scripts/ch10/evaluate.py` / `scan_threshold_replay.py`）

- 留出测试集上按类目算精确率、召回率、F1、support，外加 micro/macro 汇总；每类一张 2×2 混淆矩阵（sklearn `multilabel_confusion_matrix`）。
- **容错红线按档位设闸（诊断展示，不是交付硬闸门）**：
  - 严：退换货、物流、尺码、发票、质量问题 → F1 ≥ 0.9。
  - 中：运费、优惠活动、价保、支付、订单修改、库存补货、商品信息、保修维修 → ≥ 0.8。
  - 宽：账号、会员积分、评价、其他 → 不设线，报告画「—」而不是 ✅（避免读者误以为也过了某条线）。
  - 不达标标红提示「先回头搞数据」。九项验收里「测试集评测」只要报告生成、17 类都有数就算 `pass`；红线不达标显示诊断红字，不阻塞交付。若严档超过一半不达标，允许再补一轮数据/调参，**最多两轮**，之后如实报告红项。
- **判错样本导出人工复核**：测试集预测≠标准答案的样本落 `data/ch10/reports/error_samples.md`（句子/标准标签/预测标签/方向）。错误方向记账：**漏打**（该打的没打，记 1 笔放跑）/ **多打**（不该打的多打一个，记 1 笔冤枉）/ **错位**（该打的没打、反而打了别的类，记 2 笔：被抢的类放跑 +1、抢标签的类冤枉 +1）。错例条数与混淆矩阵笔数的差额全在错位上；脚本内置对账断言。
- **判定阈值可复算**：`scan_threshold_replay.py` 重演 ONNX 阈值扫描——验证集经 :8110 打分一次，分数表落 `data/ch10/reports/val_scores.json` 固定；九个候选线套同一张表各算一遍 micro-F1，当选线必须与 `threshold.json`（`source: "onnx"`）**精确一致**。因为阈值本来就是用这版 ONNX 分数扫出来的，重演不依赖浮点漂移；torch 版扫描只作对照，不作验收依据。测试集指标同样走 ONNX，保证「报告 / 页面 / 服务」一个真相。
- **报告一式两份**：`.md` 给人读、`.json` 给验收页读，同一次跑写同一份数；页面只读不重算。落 `data/ch10/reports/`：`eval_report.{md,json}`、`error_samples.md`、`golden_report.json`、`threshold_scan.json`、`export_report.json`、`classify_run.json`。
- 评测集只用自家场景数据，不引公开榜单。

## 9. 部署：ONNX + 独立推理服务 + 旁路批处理

### 9.1 ONNX 导出（`scripts/ch10/export_onnx.py`）

- `torch.onnx.export` 导出（dynamic axes 支持变长 batch），产物 `data/ch10/onnx/model.onnx` + tokenizer 文件。
- 导出后跑一批样本校验 ONNX 与 torch 预测一致（logits 容差内、**用同一 threshold** 阈值后标签完全一致）才算成功；`export_report.json` 记录样本数、最大 logit 误差、标签一致率、opset。
- 导出成功后在验证集上跑 ONNX 打分，重扫九档阈值并覆盖 `threshold.json`（见 §7）；此后评测、服务、重演一律用这份 ONNX 阈值。

### 9.2 推理服务（`scripts/ch10/serve.py`，独立进程 :8110）

- FastAPI + onnxruntime + tokenizers，**进程不导入 torch/transformers**（轻运行时）。
- 接口：`POST /classify` 入 `{"texts": ["...", ...]}`，出逐条 `{"labels": [...], "scores": {类目: 分数}}`（阈值用 `threshold.json`）；`GET /healthz` 探活（模型名、阈值、类别数）。
- 批量上限 64，空/超长入参 400；`texts` 为空 422/400 说清楚。
- 起停：`make classifier-up` / `classifier-down`。Windows 用 `subprocess.Popen(..., creationflags=CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS)` 写 `log/classifier.pid`，日志 `log/classifier.log`；停止用 `taskkill /PID <pid> /T /F`。POSIX 分支用 `start_new_session` + `os.killpg`。
- tokenizer 用 `tokenizers.Tokenizer.from_file(data/ch10/onnx/tokenizer.json)`，手工拼 `input_ids` / `attention_mask` / `token_type_ids`；导出后以 ONNX 实际输入名为准。

### 9.3 旁路批处理（`scripts/ch10/classify_pool.py`）

- 捞 `low_confidence_questions` 中尚未归类的行：`LEFT JOIN topic_classifications ON question_id = id WHERE topic_classifications.id IS NULL AND matched_review_id IS NOT NULL`。
- 文本按 6.1 规则取 `review_queue.normalized_question`；为空才退回 `raw_question`（当前 16 条全有标准化问法）。
- 默认 `--min-batch 10`，不足提示不跑；`--force` 强跑；make `classify-pool` 触发（`FORCE=1` 传 `--force`）。
- 批量调 :8110（每批 ≤32），结果写 `topic_classifications`（question_id / labels JSON / classified_at）。已归类的行不重复归（唯一键 + 先查）；服务失败整批不写，不留半条。
- `data/ch10/reports/classify_run.json` 记录捞取数、实际归类数、跳过数、标签分布、耗时。
- **主链路不碰**：`/chat`、ch06/ch07 图、ch09 飞轮都不调用分类器。

## 10. 后端 API 与飞轮后台前端

### 10.1 数据模型与建表

- `sql/ch10-ddl.sql`：用户定稿的 DDL **原样落仓**（不改列类型、可空性、唯一键、外键、注释）：

```sql
-- =============================================================
-- ch10 · 模型微调 · 建表 DDL
-- 本章新建:topic_classifications(微调分类器旁路批量归类结果)
-- 低置信度问题攒够一批,分类器归一次类,结果落这张表喂给飞轮后台看主题分布
-- 实时对话主链路不写这张表
-- =============================================================

SET NAMES utf8mb4;

CREATE TABLE topic_classifications (
  id            BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '主键',
  question_id   BIGINT UNSIGNED NOT NULL                COMMENT '归类的问题,指向 low_confidence_questions.id',
  labels        JSON            NOT NULL                COMMENT '多标签,17 类权威类目里命中的若干个',
  classified_at DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '归类时间',
  PRIMARY KEY (id),
  UNIQUE KEY uk_question_id (question_id),
  CONSTRAINT fk_topic_question FOREIGN KEY (question_id) REFERENCES low_confidence_questions (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='主题分类结果';
```
- `app/models/topic.py`：`TopicClassification`（id、question_id、labels JSON、classified_at），导出到 `app/models/__init__.py`。
- `app/db/init_ch10.py`：仿 `init_ch09.py` 的幂等建表（表存在即 no-op，部分存在报错；用 `exec_driver_sql` 避开 JSON 注释里的 `:`）。
- 单测：SQLite `Base.metadata.create_all` 能建表；init 幂等/半成品报错；唯一键拒绝重复 question_id。

### 10.2 主题分布 API（`app/api/topics.py`）

- `GET /api/topics/distribution` → 17 类各自问题量（`labels` JSON 在 Python 侧聚合，量级百级）+ 每类最多 3 条样例问题 + 总量 / 未归类数 / 最近归类时间。
- 样例按文本去重：同一句标准化问法对应池里多行时，样例列表只留一条（计数照旧按条算）。
- `GET /api/topics/questions?label=&page=&size=` → `{label,total,page,pages,size,items[]}`。每条给标准化问法（没归并的退回原话并标 `normalized=false`）、用户原话、命中的全部类目、落池入口 source、同义合并条数、审核状态、落池与归类时间。
- 类目名不在权威表里回 **400** 并说清楚，不静静回一页空列表——那会被当成「这类没有问题」。
- 筛选与分页在 Python 侧做（不依赖 MySQL JSON 函数），测试库上结果一致；多标签问题在它命中的每个类目下都出现，与分布图计数口径一致。

### 10.3 页面（原生 HTML/CSS，与 review.html 同款像素风）

- `app/static/topics.html`：`/topics`。17 类横向条形图（纯 HTML/CSS，宽度按计数比例），计数 + 占比；哪类堆得多一眼看出；每类下 3 条去重样例；点类目名进问题列表。
- `app/static/topic-questions.html`：`/topics/questions?label=<类目>&page=N`。分页列出该类下全部问题；类目名与页码在地址栏，刷新回同一页；非法类目显示 API 的 400 错误文案。
- `app/static/acceptance.html` / `acceptance-eval.html` / `acceptance-data.html` / `acceptance-errors.html`：四页验收。
- 共享 `app/static/acceptance.css` + `acceptance.js`（取数、提示、重跑按钮、日志窗口）；条形图、混淆矩阵、分数条仍纯 HTML/CSS，不引外部图表库。
- `app/static/admin.js` 导航加「主题分布」「验收」两项；`/admin` 首页加一张卡（过闸数 N/9 + 分类器在线状态），从 `/api/acceptance/overview` 取数。
- **前端走 Vibe Coding**：用户描述效果、直接改，不套 brainstorm/TDD/code review。

### 10.4 验收 API（`app/api/acceptance.py`，前缀 `/api/acceptance`）

- 只读端点 `overview` / `eval` / `data` / `errors` 读第 7–9 节的 json 产物与文件 stat；`service` 现场探 :8110；`classify` 代理 :8110 做单句试分类（演示「17 类各自独立过线，过几个打几个」）。
- 产物缺失不报错，回 `present=false` + 该跑哪个 make 目标；页面据此长出重跑按钮而不是白屏。
- **闸门三态**：`pass`（跑过且过线）/ `fail`（跑过没过线）/ `missing`（还没跑过）。missing 不显示失败态结论文案——没跑过不等于没过线。
- 九项实证：语料与数据集、黄金样例闸、训练产物、ONNX 导出与服务、测试集评测、阈值扫描、混淆矩阵、错例复核、旁路批量归类。`overview` 给每项 `present` / `pass` / 重跑作业名 / 日志路径。
- **「测试集评测」的 pass 口径**：报告生成、17 类都有 support 即为 `pass`；严/中档红线不达标只显示诊断红字，不把该项判成 `fail`（按用户确认的「诊断展示 + 有界重试」）。
- 页面不重算指标：`eval` 端点原样透传 `eval_report.json`；`overview` 只做「present / pass / missing」与红线诊断标记，不重算第二个真相。

### 10.5 作业运行器（`app/core/jobs.py` + `app/api/jobs.py`）

- 白名单：作业名 → 写死的 make argv（`ch10-golden`、`ch10-corpus`、`ch10-dataset`、`ch10-train`、`ch10-eval`、`ch10-export`、`ch10-threshold-scan`、`classify-pool`、`classifier-up`、`classifier-down`）。前端只能传作业名，传不进命令片段、参数或路径。
- `POST /api/jobs/{name}/run`（202/404/409）、`GET /api/jobs/{name}/status`（状态 + 日志尾 + exit_code）、`POST /api/jobs/{name}/stop`。
- `subprocess.Popen` 独立会话；日志 `log/acceptance/<job>.log`；同名未结束拒绝重入；heavy 作业（corpus/dataset/train/export）由页面二次确认。
- 停止时连 make → python 整条链一起收，不留孤儿进程（Windows `taskkill /T /F`，POSIX `killpg`）。
- **重跑语义**：`classify-pool` 幂等，页面重跑显示「扫到 0 条未归类，已全部归类」而不是假装失败；浏览器不提供删表重归。CLI 另留显式 `scripts/ch10/classify_pool.py --reclassify --yes`，只给终端用。`ch10-threshold-scan` 的 Make 配方先 `classifier-up`（已起则复用）并轮询 `/healthz`，服务没起来直接报错，不让按钮卡在必失败状态。
- 现有 `/api/admin/jobs` 继续服务 ch09 轻作业；本章不迁移、不删除。

## 11. 依赖、文件与落仓

### 11.1 依赖

- `requirements-ch10.txt` 第一行 `-r requirements.txt`（数据脚本要用 `build_llm`/`Settings`/SQLAlchemy/PyMySQL），再加 torch、transformers、accelerate、scikit-learn、onnx、onnxruntime、tokenizers、numpy。具体版本在实现时先查 Context7/官方兼容矩阵再钉。主 `requirements.txt` 不动。
- `.venv-ml` 用 CPython 3.12（uv 管理）。主 `.venv` 继续跑 994 个测试。
- **所有 `scripts/ch10/*` 都在 `.venv-ml` 跑**（含取数、预标、写库；避免两套解释器混用）；`make` 里用 `ML_PYTHON ?= .venv-ml/Scripts/python.exe`。主应用/API/测试仍用 `.venv`，不 import torch/transformers/onnxruntime。

### 11.2 代码与静态文件

- `scripts/ch10/`：`build_corpus.py`、`build_dataset.py`、`corpus_lib.py`、`prelabel.py`、`validate_golden.py`、`train.py`、`evaluate.py`、`export_onnx.py`、`scan_threshold_replay.py`、`serve.py`、`inference_lib.py`、`classify_pool.py`。
- `app/core/taxonomy.py`、`app/core/jobs.py`、`app/api/jobs.py`、`app/api/topics.py`、`app/api/acceptance.py`、`app/models/topic.py`、`app/db/init_ch10.py`。
- `app/static/`：`topics.html`、`topic-questions.html`、`acceptance.html`、`acceptance-eval.html`、`acceptance-data.html`、`acceptance-errors.html`、`acceptance.css`、`acceptance.js`、修改 `admin.js`、`admin.html`。
- `sql/ch10-ddl.sql`。
- `tests/`：taxonomy、脱敏、切分、阈值、topics API、classify_pool、jobs runner、acceptance API、schema init。

### 11.3 产物与 git

- `data/ch10/`：语料、数据集、模型、onnx、报告。**模型与语料不进 git**（`.gitignore` 加 `data/ch10/corpus_*`、`data/ch10/dataset/`、`data/ch10/model/`、`data/ch10/onnx/`）；黄金样例与报告进 git（`data/ch10/golden_samples.jsonl`、`data/ch10/reports/`）。
- 作业日志 `log/acceptance/` 随 `log/` 一并忽略。

### 11.4 Make 目标

`ch10-golden`、`ch10-corpus`、`ch10-dataset`、`ch10-train`、`ch10-eval`、`ch10-export`、`ch10-threshold-scan`、`classifier-up`、`classifier-down`、`classify-pool`（`FORCE=1` 不足一批也跑）。这份清单进 `app/core/jobs.py` 白名单。`ch10-threshold-scan` 依赖 `classifier-up` 并等健康；所有 ch10 配方用 `$(ML_PYTHON)`，主应用配方继续用 `$(PYTHON)`。

## 12. 测试与验证策略

### 可单测（TDD）

- `taxonomy` 模块：17 类唯一、顺序稳定、边界说明/示例齐全、`LABEL2ID/ID2LABEL` 互逆。
- 脱敏函数：手机号、订单号、微信/QQ 替换；不误伤普通数字。
- 分层划分：比例、无重复文本跨集泄漏、每类目三份都有样本、稀有组合归并。
- 阈值应用与兜底：全不过线取最高分、空 labels 不产出。
- `topics` API：分布计数、样例按文本去重、非法类目 400、分页、多标签在每类下出现。
- `classify_pool`：只捞未归类、已归类不重复、服务失败不写半条、min-batch/force。
- 作业运行器：白名单外拒绝、同名未结束 409、停止进程树、日志尾。
- 验收 API：产物缺失回 `present=false` 不抛错；三态判定。
- schema init：幂等、半成品报错、唯一键。

### 验证跑替代 TDD（纯 Prompt / 数据 / 训练类）

- 预标 prompt → 黄金样例 ≥80%（`golden_report.json`）。
- 清洗 → 抽样 before/after 对照（落盘可复查）。
- 模拟造数 → 进人工抽审；生成标签与盲标交叉校验一致率进报告。
- 训练 → 验证集每 epoch 分数曲线 + 早停触发记录。
- 评测 → 测试集报告 + 红线 + 混淆矩阵 + 错例方向对账。
- ONNX → 与 torch 预测一致性校验。
- 阈值 → `scan_threshold_replay.py` 复算与 `threshold.json` 一致。
- 批量归类 → `classify_run.json` + 主题分布页看到统计。

### 库/框架 API

transformers Trainer、torch.onnx、onnxruntime、FastAPI、SQLAlchemy 一律先用 Context7 查最新官方文档和接口定义再写，不凭记忆。

## 13. 验收标准（用户定稿 + 参考 spec 九项）

1. 测试集各类目 F1 和混淆矩阵报告能跑出来，并在 `/acceptance/eval` 上看到每类的红线达标情况；红线是诊断展示，不达标标红，不阻塞交付（严档过半不达标最多补一轮，见 §8）。
2. 拿一批攒下的真实问题跑归类，后台主题分布页能看到各类目统计；点某一类的名字能进到这一类的问题列表，分页看全，每条带来源、同义合并条数与审核状态。
3. 「买大了想退」这类多诉求句子，能同时命中多个类目（尺码 + 退换货）——在 `/acceptance/eval` 的单句试分类里现场打分，17 类各自过线情况一目了然。
4. 九项实证（语料与数据集、黄金样例闸、训练产物、ONNX 导出与服务、测试集评测、阈值扫描、混淆矩阵、错例复核、旁路批量归类）在验收页上全部可看、可重跑，验收过程不必回终端。

## 14. 风险与对策

- **模拟语料占比高（约 96–99%）**：本地真实池仅 16 条（去重 13 条），分布代表性有限。对策：真实样本全量进人工抽审；报告注明语料构成；后续真实问题攒多了可增量重训（效果不行，先搞数据）。
- **Python 3.13 wheel 不全**：ML 环境改用 CPython 3.12；若 3.12 也缺某个轮子，停下来问，不自行换模型/框架。
- **HF 模型下载**：网络不通走 `HF_ENDPOINT=https://hf-mirror.com` 兜底；不动基座选型。
- **ONNX 与 torch 数值/标签不一致**：导出后强制一致性校验；不一致先修 tokenizer/输入名/opset，不带着误差上线。
- **Windows 服务生命周期**：nohup 不可用，用 pid 文件 + `taskkill /T /F`；`classifier-up` 后轮询 `/healthz`，起不来直接报错。
- **标签噪声**：黄金样例闸 + 字面铁律写死进 prompt + 人工抽审三道手段压住；同一说法标出多种组合是第一版翻车主因。
- **测试集自嗨**：报告与页面都标注语料构成、真实样本占比；反目标里写死不按测试集调参。

## 15. 留给 builder 的与待用户确认

- **已确认**：ML 依赖按 §4.3 用 `requirements-ch10.txt` + `.venv-ml`（本仓库适配），不引入 `pyproject.toml`；测试集红线是诊断展示 + 有界重试，不是交付硬闸门；30 条黄金样例和抽审清单由我起草、用户审核，真实问题按 12 条去重标准化问法全量抽审。
- **我的调用（可纠偏）**：「其他」互斥兜底；新增 `app/core/` 包；新增独立子进程作业运行器、ch09 旧运行器保持不动；真实问题按 A 混入 80/10/10 并在报告标注语料构成；本地池按实测 16 条而非参考 spec 的 70 条；ONNX 为阈值与评测的最终口径；模拟语料/增强按内容哈希 id 冻结、默认复用；切分前做全局近重复合并；`classify-pool` 页面重跑只报幂等结果、不做删表重归；所有 ch10 脚本跑 `.venv-ml`。
- **实现时再定**：具体依赖版本、ONNX opset、tokenizer 输入名、Windows 守护进程细节；这些在 Context7/实跑中定，不改变本章边界。
- **DDL**：`sql/ch10-ddl.sql` 用 §10.1 原文，不再改写。
