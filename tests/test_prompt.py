from app.prompts.customer_service import render_system_prompt


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
