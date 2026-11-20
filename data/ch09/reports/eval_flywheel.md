# ch04 RAG 四策略评估报告

## 运行参数

- cases: tests/eval/rag_cases.json (21 evaluable, 9 excluded, limit=30)
- strategies: hybrid_rerank
- mode: full
- generation/judge/rewrite model: deepseek-chat (temperature 0)
- embedding model: BAAI/bge-m3
- rerank model: BAAI/bge-reranker-v2-m3
- retrieval: recall_top_k=50, final Top-K=10, RRF k=60
- faith_cases: enabled
- generated at: 2026-11-20 00:25:58
- commit: 1b64339

## 排除用例

- excluded: 9 / 30
- A21 (points_missing)
- A22 (points_missing)
- A24 (points_missing)
- A25 (points_missing)
- A26 (points_missing)
- A27 (points_missing)
- A28 (points_missing)
- A29 (points_missing)
- A30 (points_missing)

口径:语料未覆盖或与语料冲突的用例不参与任何指标聚合;
补语料或修正预期后再将其纳入基准。

## 总体指标

| strategy | Recall@5 | Recall@10 | MRR@10 | Faithfulness | Refusal rate | Citation validity |
| --- | --- | --- | --- | --- | --- | --- |
| hybrid_rerank | 1.000 | 1.000 | 1.000 | 1.000 | 0.048 | 1.000 |

口径:Recall/MRR 只统计可答题(A/B/C/E),并按 section_path 去重;
`hybrid_rerank` 的 rerank 失败样本已从 Recall/MRR、Faithfulness、
Refusal rate、Citation validity 聚合中剔除(D 桶拒答准确率单独统计)。
Faithfulness 只统计裁判成功且未被证据门拒答的回答;
Refusal rate 的分母是证据门完成的运行(含 D 桶预期拒答);
Citation validity 只统计实际带 `[n]` 的生成回答。

## 分桶指标

### A_policy

| strategy | Recall@5 | Recall@10 | MRR@10 | Faithfulness | Refusal rate | Citation validity |
| --- | --- | --- | --- | --- | --- | --- |
| hybrid_rerank | 1.000 | 1.000 | 1.000 | 1.000 | 0.048 | 1.000 |

## D 桶拒答准确率

| strategy | evaluable | refusal | model_refusal | answered/other | accuracy |
| --- | --- | --- | --- | --- | --- |
| hybrid_rerank | 0 | 0 | 0 | 0 | n/a |

口径:证据门拒答(`refusal`)与模型自述无法回答(`model_refusal`)
都算正确拒答;self-check/检索/生成错误不计入分母;
rerank fallback 运行不剔除(本项只判断是否拒答,不评价 rerank 质量)。

## Rerank fallback

- fallback runs: 0
- case ids: (none)

`rerank_failed=True` 的运行没有被当作成功的 rerank,已从 `hybrid_rerank` 的 Recall/MRR、Faithfulness、Refusal rate、Citation validity 聚合中剔除。

## Query rewrite fallback

- cases with fallback rewrite: 0 / 21
- case ids: (none)

## Faith cases 台账

- new: 0
- recurrence: 0
- write errors: 0
- no unfaithful answers
- 口径:台账按 eval_id 一题一行(Task 11);同一次评估中同一题在多个策略下复发会累加 seen_count 并计入 recurrence。

## 排除与错误

- none

`self_check_error` 与 `judge_error` 已从 Faithfulness 中排除,
`retrieval_error` 已从检索聚合中排除,以上均单独列明。
