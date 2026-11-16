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


def test_answer_prompt_reuses_blocks_without_tool_list():
    p = render_answer_system_prompt()
    assert "[n]" in p and "不得编造" in p
    assert "不承诺具体到账时间" in p
    assert "query_product 只用于" in p
    assert "你可以使用以下工具能力" not in p
    assert "本店" in p
