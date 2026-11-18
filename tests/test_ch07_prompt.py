from app.prompts.customer_service import (
    render_after_sales_agent_system_prompt,
    render_agent_system_prompt,
    render_turn_material,
)


def test_agent_system_prompt_is_constant():
    a = render_agent_system_prompt()
    b = render_agent_system_prompt()
    assert a == b
    assert "本轮知识库证据" not in a
    assert "订单号:" not in a


def test_after_sales_system_prompt_is_constant():
    a = render_after_sales_agent_system_prompt()
    assert "订单号:" not in a and "识别意图" not in a


def test_turn_material_only_includes_nonempty_sections():
    assert render_turn_material("", None, None, "") == ""
    text = render_turn_material(
        "订单 1001 退款",
        [{"n": 1, "section_path": "售后", "question": "能退吗", "answer": "可以"}],
        {"order_id": "1001"},
        "退款退货",
    )
    assert "早前对话摘要" in text and "订单 1001 退款" in text
    assert "本轮知识库证据" in text and "[1]" in text
    assert "当前订单" in text and "订单号:1001" in text
    assert "识别意图" in text and "退款退货" in text
