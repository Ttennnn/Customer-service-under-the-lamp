"""工具参数 JSON Schema 校验。

用 :class:`jsonschema.Draft202012Validator` 校验模型给出的参数,并把
``ValidationError`` 翻译成中文提示。调用方(执行引擎)只需把返回的
错误列表拼进回灌消息,不需要自己理解 jsonschema 的英文原文。
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from jsonschema import Draft202012Validator

_PREFIX = "参数校验未通过："
_REQUIRED_NAME = re.compile(r"'(.+?)' is a required property")


def _path(error) -> str:
    return ".".join(str(part) for part in error.absolute_path) or "(根)"


def _expected_type(value) -> str:
    if isinstance(value, str):
        return value
    return " / ".join(str(item) for item in value)


def _actual_type(value) -> str:
    if isinstance(value, Mapping):
        return "object"
    if isinstance(value, Sequence) and not isinstance(value, str):
        return "array"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if value is None:
        return "null"
    return type(value).__name__


def _format_required(error) -> list[str]:
    instance = error.instance
    required = error.validator_value
    missing = [
        name
        for name in required
        if not isinstance(instance, dict) or name not in instance
    ]
    if not missing:
        match = _REQUIRED_NAME.search(error.message)
        if match:
            missing = [match.group(1)]
    return [
        f"{_PREFIX}缺少必填参数 {name}" for name in missing
    ] or [f"{_PREFIX}{error.message}"]


def _format_error(error) -> list[str]:
    path = _path(error)
    validator = error.validator

    if validator == "required":
        return _format_required(error)

    if validator == "type":
        expected = _expected_type(error.validator_value)
        actual = _actual_type(error.instance)
        return [f"{_PREFIX}参数 {path} 类型应为 {expected}，实际为 {actual}"]

    if validator == "enum":
        values = "、".join(str(value) for value in error.validator_value)
        return [f"{_PREFIX}参数 {path} 只能取 {values}"]

    if validator == "minimum":
        return [f"{_PREFIX}参数 {path} 不能小于 {error.validator_value}"]

    if validator == "maximum":
        return [f"{_PREFIX}参数 {path} 不能大于 {error.validator_value}"]

    return [f"{_PREFIX}参数 {path} {error.message}"]


def validate_arguments(schema: dict, arguments: object) -> list[str]:
    """Validate *arguments* against *schema*.

    Returns an empty list when valid, otherwise Chinese error strings.
    """
    validator = Draft202012Validator(schema)
    errors: list[str] = []
    for error in sorted(
        validator.iter_errors(arguments),
        key=lambda item: ".".join(str(part) for part in item.absolute_path),
    ):
        errors.extend(_format_error(error))
    return errors
