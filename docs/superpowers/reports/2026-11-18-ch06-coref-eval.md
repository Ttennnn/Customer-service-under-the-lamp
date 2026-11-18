# ch06 指代消解评估报告

## 运行参数

- command: `python.exe -m app.jobs.eval_coref`
- env: `PYTHONIOENCODING=utf-8`
- dataset: ch06_coref_cases.json (13 cases, 13 valid)
- model: deepseek-chat (temperature 0)
- generated at: 2026-11-18 19:05:47
- pass bar: overall accuracy >= 90%, passthrough = 100%, no LLM fallback

## 总体指标

| total | passed | accuracy | passthrough | LLM fallback | 结论 |
| --- | --- | --- | --- | --- | --- |
| 13 | 13 | 1.000 | 3/3 (1.000) | 0 | PASS |

准确率只统计有效模型输出;LLM 兜底样例(回退原文)不参与计分,并直接判 FAIL。透传样例必须与原文逐字符一致。

## 逐条结果

| id | 类型 | message | resolved | 判定 | 原因 |
| --- | --- | --- | --- | --- | --- |
| C01 | 改写 | 它能退吗 | 订单号 1001 的蓝牙耳机能退吗 | PASS | ok |
| C02 | 改写 | 这个还有别的颜色吗 | XM5 头戴式耳机还有别的颜色吗 | PASS | ok |
| C03 | 透传 | 订单 1001 什么时候发货 | 订单 1001 什么时候发货 | PASS | ok |
| C04 | 改写 | 邮费怎么算 | 运费怎么算 | PASS | ok |
| C05 | 改写 | 这个能退吗 | 订单 1001 能退款吗 | PASS | ok |
| C06 | 改写 | 那单能取消吗 | 订单 1002 能取消吗 | PASS | ok |
| C07 | 改写 | 刚才说的那个型号有现货吗 | X200 这款扫地机器人有现货吗 | PASS | ok |
| C08 | 改写 | 退钱要多久到账 | 订单 1003 的退款要多久到账 | PASS | ok |
| C09 | 透传 | XM5 有几种配色 | XM5 有几种配色 | PASS | ok |
| C10 | 改写 | 换个新的要怎么办 | 耳机右耳没有声音了，如果确认是质量问题，换货要怎么办 | PASS | ok |
| C11 | 改写 | 帮我催一下它 | 帮我催一下订单 1003 的发货 | PASS | ok |
| C12 | 透传 | 订单 1002 现在到哪里了 | 订单 1002 现在到哪里了 | PASS | ok |
| C13 | 改写 | 第一个能退吗 | 订单 1001 能退款吗 | PASS | ok |

## 失败样例

无。

## LLM 兜底样例

无。

## Prompt 调整

无。本次验证未修改 `app/agent/coref.py` 的 `COREF_SYSTEM_PROMPT`。
