# ch07 摘要 prompt 评估报告

## 运行参数

- command: `python.exe -m app.jobs.eval_ch07_summary`
- env: `PYTHONIOENCODING=(unset)`
- dataset: ch07_summary_cases.json (6 cases, 6 valid)
- model: deepseek-chat (temperature 0, structured JSON)
- generated at: 2026-11-18 23:07:16
- pass bar: overall accuracy >= 90%, strict JSON on every case

## 总体指标

| total | passed | accuracy | invalid | 结论 |
| --- | --- | --- | --- | --- |
| 6 | 4 | 0.667 | 0 | FAIL |

准确率只统计成功返回严格 JSON 的样例;解析失败或模型调用失败的样例不计入分母,并直接判 FAIL。

## 逐条结果

| id | 样例 | 字数 | 摘要 | 判定 | 原因 |
| --- | --- | --- | --- | --- | --- |
| order-refund | 订单号 + 手机号 + 退款诉求；寒暄不得进入摘要 | 55 | 用户提供订单号1001和手机号13800000000，申请退款；客服已登记该订单退款申请，并告知留意审核结果。 | PASS | ok |
| product-warranty | 商品型号 + 保修诉求 + 未解决问题（灯不亮） | 92 | 用户询问LD-200是否仍在保修期内，并反映灯不亮，问能否保修。客服说明保修期为购买后12个月，需提供订单信息；灯不亮可能属质量问题，可走保修流程，需先寄回检测。用户尚未提供订单信息。 | PASS | ok |
| chitchat-only | 纯寒暄会话，摘要应极短且不出现业务词 | 36 | 用户仅打招呼并致谢，未提出任何商品、订单、退款或售后诉求，无待解决问题。 | FAIL | banned: '订单', '退款' |
| order-corrected | 用户先报 1001 后更正为 1002，摘要只保留更正后的订单 | 67 | 用户先查询订单1001物流，客服告知已签收；用户更正为查询订单1002，表示尚未收到，客服回复订单1002正在运输途中，预计明天送达。 | FAIL | banned: '1001' |
| no-fabrication | 对话里没有订单号，摘要不得编造 4 位以上数字 | 62 | 用户询问店铺发什么快递，客服回复默认发顺丰、偏远地区转邮政；用户追问大概几天能到，客服回复一般2到3天送达，具体以物流为准。 | PASS | ok |
| refund-rejected | 退款被拒绝 + 拒绝原因（超过七天） | 79 | 用户询问订单1001的退款为何未通过；客服回复退款申请已因超过七天无理由退货期限被拒绝。用户追问还能怎么办，客服建议联系品牌售后检测，若为质量问题可申请保修。 | PASS | ok |

## 失败样例

| id | 摘要 | 原因 |
| --- | --- | --- |
| chitchat-only | 用户仅打招呼并致谢，未提出任何商品、订单、退款或售后诉求，无待解决问题。 | banned: '订单', '退款' |
| order-corrected | 用户先查询订单1001物流，客服告知已签收；用户更正为查询订单1002，表示尚未收到，客服回复订单1002正在运输途中，预计明天送达。 | banned: '1001' |

## 无效输出

无。

## Prompt 调整

无。本次验证未修改 `app/services/summary_job.py` 的 `SUMMARY_SYSTEM_PROMPT`;评估使用 200 字默认模板(`build_summary_prompt` 的默认 `max_chars`)。
