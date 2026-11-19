"""参数 JSON Schema 校验的中文错误信息。"""

from app.tools.validation import validate_arguments


def test_valid_arguments_pass():
    schema = {
        "type": "object",
        "properties": {"description": {"type": "string"}},
        "required": ["description"],
    }
    assert validate_arguments(schema, {"description": "耳机坏了"}) == []


def test_missing_required_is_chinese():
    schema = {
        "type": "object",
        "properties": {"description": {"type": "string"}},
        "required": ["description"],
    }
    errors = validate_arguments(schema, {})
    assert errors and "缺少必填参数 description" in errors[0]


def test_type_enum_and_minimum():
    schema = {
        "type": "object",
        "properties": {
            "ticket_type": {"type": "string", "enum": ["售后", "投诉"]},
            "count": {"type": "integer", "minimum": 1},
        },
        "required": ["ticket_type", "count"],
    }
    assert "类型应为 string" in validate_arguments(
        schema, {"ticket_type": 1, "count": 1}
    )[0]
    assert "只能取" in validate_arguments(schema, {"ticket_type": "x", "count": 1})[0]
    assert "不能小于 1" in validate_arguments(
        schema, {"ticket_type": "售后", "count": 0}
    )[0]


def test_maximum_and_nested_path_are_reported():
    schema = {
        "type": "object",
        "properties": {
            "order": {
                "type": "object",
                "properties": {"qty": {"type": "integer", "maximum": 5}},
                "required": ["qty"],
            }
        },
        "required": ["order"],
    }
    errors = validate_arguments(schema, {"order": {"qty": 9}})
    assert any("order.qty" in e for e in errors)
    assert any("不能大于 5" in e for e in errors)


def test_root_error_uses_root_label():
    errors = validate_arguments({"type": "object"}, ["not", "an", "object"])
    assert errors and "(根)" in errors[0]
    assert "类型应为 object" in errors[0]
