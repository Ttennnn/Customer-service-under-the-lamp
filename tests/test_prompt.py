from app.prompts.customer_service import (
    render_answer_system_prompt,
    render_system_prompt,
)


def test_render_contains_role_and_constraints():
    p = render_system_prompt(business_name="XX商城")
    assert "XX商城" in p
    assert "客服" in p
    assert "不" in p  # 含约束/拒答措辞


def test_render_default_business_name():
    assert "本店" in render_system_prompt()


def test_render_mentions_tools():
    p = render_system_prompt()
    assert "订单" in p and "物流" in p and "工单" in p
    assert "转人工" in p


def test_prompt_contains_citation_and_negative_constraints():
    p = render_system_prompt()
    assert "[n]" in p and "不得编造" in p
    assert "不承诺具体到账时间" in p
    assert "query_product 只用于" in p


def test_prompt_routes_model_spec_questions_to_query_faq():
    p = render_system_prompt()
    assert "商品型号参数" in p
    assert "必须先调用 query_faq" in p


def test_tool_guidance_routes_model_spec_and_policy_to_faq():
    p = render_system_prompt()
    assert "实时价格、库存或订单快照" in p
    assert "商品型号、规格参数、退换货政策等知识类问题" in p
    assert "必须先查知识库(query_faq)" in p
    assert "查询商品(query_product)" in p  # tool capability retained
    assert "规格、价格、退换货政策时,查询商品" not in p


def test_production_prompt_has_routing_and_citation_rules():
    p = render_system_prompt()
    assert "必须先调用 query_faq" in p
    assert "category/content_type" in p
    assert "不得编造或复用历史编号" in p


def test_answer_prompt_has_citation_rules_without_tool_routing():
    p = render_answer_system_prompt()
    assert "必须先调用 query_faq" not in p
    assert "query_product" not in p
    assert "category/content_type" not in p
    assert "[n]" in p and "不得编造或复用历史编号" in p
    assert "每一轮都必须基于本轮检索/证据重新作答" in p
    assert "本轮没有 [n] 时,回答中不得出现任何 [n]" in p
    assert "明确说「知识库暂未收录/无法确认」" in p


def test_prompt_forbids_reusing_history_citations():
    for p in (render_system_prompt(), render_answer_system_prompt()):
        assert "每一轮都必须基于本轮检索/证据重新作答" in p
        assert "即使历史对话里有相似答案" in p
        assert "不得从历史回答抄结论或 [n] 引用编号" in p
        assert "本轮没有 [n] 时,回答中不得出现任何 [n]" in p


def test_agent_prompt_requires_tracking_no_from_query_order():
    from app.prompts.customer_service import render_agent_system_prompt

    p = render_agent_system_prompt()
    assert "tracking_no 只能来自 query_order 的返回" in p
    assert "不要用订单号或猜测值代替" in p


def test_after_sales_prompt_has_order_and_repair_rules():
    from app.prompts.customer_service import render_after_sales_agent_system_prompt
    prompt = render_after_sales_agent_system_prompt(
        order_data={"order_id": "1001", "status": "已签收"},
        evidence=[],
        intent="售后",
    )
    assert "1001" in prompt
    assert "不调用 submit_refund" in prompt
    assert "退款表单" in prompt


def test_after_sales_prompt_covers_refund_and_warranty_branches():
    from app.prompts.customer_service import render_after_sales_agent_system_prompt

    prompt = render_after_sales_agent_system_prompt(
        order_data={"order_id": "1001", "status": "已签收"},
        evidence=[{"n": 1, "section_path": "售后政策", "question": "保修", "answer": "一年保修"}],
    )
    assert "退款/退货" in prompt or "退款、退货" in prompt
    assert "保修" in prompt and "维修" in prompt and "补发" in prompt
    assert "suggest_handoff" in prompt
    assert "ticket_type" in prompt and "售后" in prompt
    assert "[1] 章节:售后政策" in prompt
    assert "不得编造" in prompt
    assert "自动提交" in prompt


def test_after_sales_prompt_without_order_data_asks_for_order():
    from app.prompts.customer_service import render_after_sales_agent_system_prompt

    prompt = render_after_sales_agent_system_prompt()
    assert "未提供订单数据" in prompt
    assert "不要猜测" in prompt


def test_answer_prompt_reuses_blocks_without_tool_list():
    p = render_answer_system_prompt()
    assert "[n]" in p and "不得编造" in p
    assert "不承诺具体到账时间" in p
    assert "你可以使用以下工具能力" not in p
    assert "本店" in p
