# ch05 意图分类评估报告

## 运行参数

- command: `python -m app.jobs.eval_intent`
- dataset: ch05_intent_cases.json (42 cases)
- model: deepseek-chat (temperature 0)
- generated at: 2026-11-17 21:25:41
- pass bar: overall accuracy >= 90%, each class >= 80%

## 总体指标

| total | correct | accuracy | overall pass | each class pass | 结论 |
| --- | --- | --- | --- | --- | --- |
| 42 | 42 | 1.000 | PASS | PASS | PASS |

## 每类准确率

| intent | total | correct | accuracy | pass |
| --- | --- | --- | --- | --- |
| 物流 | 6 | 6 | 1.000 | PASS |
| 订单 | 6 | 6 | 1.000 | PASS |
| 商品咨询 | 6 | 6 | 1.000 | PASS |
| 退款退货 | 6 | 6 | 1.000 | PASS |
| 售后 | 6 | 6 | 1.000 | PASS |
| 投诉 | 6 | 6 | 1.000 | PASS |
| 闲聊 | 6 | 6 | 1.000 | PASS |

## 混淆矩阵

行 = 标注(expected),列 = 预测(predicted)。

| expected \ predicted | 物流 | 订单 | 商品咨询 | 退款退货 | 售后 | 投诉 | 闲聊 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 物流 | 6 | 0 | 0 | 0 | 0 | 0 | 0 |
| 订单 | 0 | 6 | 0 | 0 | 0 | 0 | 0 |
| 商品咨询 | 0 | 0 | 6 | 0 | 0 | 0 | 0 |
| 退款退货 | 0 | 0 | 0 | 6 | 0 | 0 | 0 |
| 售后 | 0 | 0 | 0 | 0 | 6 | 0 | 0 |
| 投诉 | 0 | 0 | 0 | 0 | 0 | 6 | 0 |
| 闲聊 | 0 | 0 | 0 | 0 | 0 | 0 | 6 |

## 错误样例

无。

## 分类器异常

无。

## Prompt 调整

无。本次验证未修改 `app/agent/intent.py` 的 prompt 或 few-shot 示例。
