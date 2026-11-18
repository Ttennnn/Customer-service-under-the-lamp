# ch06 多轮验收报告

## 运行参数

- command: `python.exe -m app.jobs.eval_ch06`
- env: `PYTHONIOENCODING=utf-8`
- dialogue: 3 turns
- model: deepseek-chat (temperature 0)
- generated at: 2026-11-18 20:15:42
- pass bar: 每轮 intent 与 expected 一致,resolved_query 含全部必须实体且不含禁止指代词,无 LLM 兜底

## 总体结果

| total | passed | failed | accuracy | 结论 |
| --- | --- | --- | --- | --- |
| 3 | 3 | 0 | 1.000 | PASS |

history 逐轮累积「用户原句 + 消解后的问题」,因此后两轮的「它」只有在 coref 正确补出订单 1001 后才能通过。

## 逐轮结果

| id | message | resolved_query | intent | confidence | expected | 判定 | 原因 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| turn1 | 订单 1001 的物流到哪了 | 订单 1001 的物流到哪了 | 物流 | 0.95 | 物流 | PASS | ok |
| turn2 | 它能退吗 | 订单 1001 能退款吗 | 退款退货 | 0.95 | 退款退货 | PASS | ok |
| turn3 | 那它现在到哪了 | 订单 1001 的物流现在到哪了 | 物流 | 0.97 | 物流 | PASS | ok |

## 失败轮次

无。

## Prompt 调整

无。本次验收未修改 `app/agent/coref.py` 或 `app/agent/intent.py` 的 prompt。
