# ch07 摘要 prompt 评估报告

## 运行参数

- command: `python.exe -m app.jobs.eval_ch07_summary`
- env: `PYTHONIOENCODING=(unset)`
- dataset: ch07_summary_cases.json (6 cases, 6 valid)
- model: deepseek-chat (temperature 0, structured JSON)
- generated at: 2026-11-18 23:09:58
- pass bar: overall accuracy >= 90%, strict JSON on every case

## 总体指标

| total | passed | accuracy | invalid | 结论 |
| --- | --- | --- | --- | --- |
| 6 | 6 | 1.000 | 0 | PASS |

准确率只统计成功返回严格 JSON 的样例;解析失败或模型调用失败的样例不计入分母,并直接判 FAIL。

## 逐条结果

| id | 样例 | 字数 | 摘要 | 判定 | 原因 |
| --- | --- | --- | --- | --- | --- |
| order-refund | 订单号 + 手机号 + 退款诉求；寒暄不得进入摘要 | 54 | 用户就订单1001申请退款，提供手机号13800000000；客服已登记该订单退款申请，用户需留意审核结果。 | PASS | ok |
| product-warranty | 商品型号 + 保修诉求 + 未解决问题（灯不亮） | 100 | 用户询问已购买的 LD-200 是否仍在保修期内，并反映灯不亮，问能否保修。客服答复保修期为购买后 12 个月，需提供订单信息；灯不亮可能属质量问题，可走保修流程，需先寄回检测。用户尚未提供订单信息。 | PASS | ok |
| chitchat-only | 纯寒暄会话，摘要应为空（allow_empty）或极短且不出现业务词 | 0 |  | PASS | ok (empty allowed for chitchat) |
| order-corrected | 用户先报 1001 后更正为 1002，摘要只保留更正后的订单 | 41 | 用户查询订单1002物流，表示尚未收到；客服回复该订单正在运输途中，预计明天送达。 | PASS | ok |
| no-fabrication | 对话里没有订单号，摘要不得编造 4 位以上数字 | 52 | 用户询问店铺发什么快递及到货时间；客服答复默认发顺丰，偏远地区转邮政，一般2到3天送达，具体以物流为准。 | PASS | ok |
| refund-rejected | 退款被拒绝 + 拒绝原因（超过七天） | 82 | 用户询问订单1001的退款为何未通过；客服答复退款申请已被拒绝，原因是超过七天无理由退货期限。用户追问还能怎么办，客服建议联系品牌售后检测，若为质量问题可申请保修。 | PASS | ok |

## 失败样例

无。

## 无效输出

无。

## Prompt 调整

本次验证为 `app/services/summary_job.py` 的 `SUMMARY_SYSTEM_PROMPT` 新增两条红线:(7) 整段只有问候/寒暄且无事实或明确诉求时 summary 留空;(8) 同一事实被用户更正后只保留最新值,不重复被更正的旧值。评估使用 200 字默认模板(`build_summary_prompt` 的默认 `max_chars`)。
