"""Deterministic pure-knowledge router tests."""

import pytest

from app.rag.router import is_pure_knowledge_question


@pytest.mark.parametrize(
    "message",
    [
        "LX-A100 的色温是多少",
        "你们支持以旧换新吗",
        "退款多久到账",
        "可以开发票吗",
        "能不能上门取件",
        "快递费怎么算",
    ],
)
def test_pure_knowledge_question_is_true(message):
    assert is_pure_knowledge_question(message) is True


@pytest.mark.parametrize(
    "message",
    [
        "我的订单到哪了",
        "查一下物流进度",
        "帮我催单",
        "快递单号是多少",
        "能不能帮我转人工",
        "帮我催一下",
        "我要投诉",
        "物流到哪了",
        "你好",
        "",
        "LX-Z900 的色温是多少，另外我的订单1001到哪了",
    ],
)
def test_non_knowledge_or_mixed_question_is_false(message):
    assert is_pure_knowledge_question(message) is False
