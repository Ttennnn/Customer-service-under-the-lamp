"""工具结果分类与格式化。

执行引擎拿到工具原始返回值后,先用 :func:`classify_result` 判断成功 /
落空 / 失败,再用 :func:`format_result` 生成回灌给模型的内容与供程序使用的
``data``。格式化按工具名维护字段白名单并翻译内部枚举码,避免把内部字段
或英文状态码暴露给模型。

本模块保持纯函数:不访问数据库、不抛异常;空结果/错误的中文句子由执行
引擎负责(本模块对 JSON ``null`` 提供防御性兜底)。
"""

from __future__ import annotations

import json
import re
from typing import Literal

from app.tools.builtin.faq import ERROR_TEXT, MISS_TEXT
from app.tools.spec import ToolSpec

EMPTY_RESULT_TEXT = "未查询到相关结果"
TRUNCATION_SUFFIX = "…（结果已截断）"

FIELD_ALLOWLISTS: dict[str, tuple[str, ...]] = {
    "query_order": (
        "order_id",
        "items",
        "amount",
        "status",
        "tracking_no",
        "created_at",
        "signed_at",
    ),
    "query_product": ("product_name", "price", "stock", "specs"),
    "query_logistics": ("tracking_no", "carrier", "status_code", "traces"),
    "query_warranty": (
        "order_id",
        "product_name",
        "warranty_status",
        "purchase_date",
        "expire_date",
        "coverage",
    ),
    "query_return_status": (
        "order_id",
        "return_no",
        "status",
        "progress",
        "updated_at",
    ),
    "create_ticket": ("ticket_no",),
}

ENUM_MAPS: dict[str, dict[str, dict[str, str]]] = {
    "query_logistics": {
        "status_code": {
            "PICKED_UP": "已揽收",
            "TRANSIT": "运输中",
            "ARRIVED": "到达转运中心",
            "DELIVERING": "派送中",
            "DELIVERED": "已签收",
        }
    },
    "query_warranty": {
        "warranty_status": {
            "IN_WARRANTY": "在保",
            "OUT_OF_WARRANTY": "已过保",
            "UNKNOWN": "无法确认",
        }
    },
    "query_return_status": {
        "status": {
            "APPLIED": "已申请",
            "PICKING": "待取件",
            "IN_TRANSIT": "退货运输中",
            "RECEIVED": "已收到退货",
            "REFUNDING": "退款中",
            "DONE": "已完成",
        }
    },
}

# Nested enum fields: ``{tool: {list_field: {item_field: mapping}}}``. Only
# the listed subfields are translated; every other item field is preserved.
NESTED_ENUM_MAPS: dict[str, dict[str, dict[str, dict[str, str]]]] = {
    "query_logistics": {
        "traces": {"status_code": ENUM_MAPS["query_logistics"]["status_code"]},
    },
}

_TICKET_NO = re.compile(r"^T\d{11}$")
_UNPARSEABLE = object()


def _parse_json(raw: object) -> object:
    """Return parsed JSON, ``None`` for JSON null, or ``_UNPARSEABLE``."""
    if isinstance(raw, (dict, list)):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            return _UNPARSEABLE
    return _UNPARSEABLE


def _serialize(value: object, *, compact: bool = True) -> str:
    kwargs: dict = {
        "ensure_ascii": False,
        "default": str,
    }
    if compact:
        kwargs["separators"] = (",", ":")
    return json.dumps(value, **kwargs)


def _truncate(content: str, max_chars: int | None) -> str:
    if max_chars is not None and max_chars >= 0 and len(content) > max_chars:
        return content[:max_chars] + TRUNCATION_SUFFIX
    return content


def classify_result(
    spec: ToolSpec, raw: object
) -> Literal["ok", "empty", "error"]:
    """Return ``"ok"`` / ``"empty"`` / ``"error"`` for a tool's raw result."""
    if raw is None:
        return "empty"
    if isinstance(raw, str) and raw == "":
        return "empty"

    name = spec.name

    if name == "query_faq":
        if raw == MISS_TEXT:
            return "empty"
        if raw == ERROR_TEXT:
            return "error"
        return "ok"

    if name == "create_ticket":
        if isinstance(raw, str):
            if _TICKET_NO.match(raw):
                return "ok"
            if raw.startswith("创建工单失败") or "缺少会话上下文" in raw:
                return "error"
        return "error"

    parsed = _parse_json(raw)
    if parsed is None:
        # ``query_order("")`` returns the JSON literal ``null``.
        return "empty"
    if isinstance(parsed, dict):
        if parsed.get("found") is False:
            return "empty"
        if parsed.get("isError") is True:
            return "error"
    return "ok"


def _format_known(name: str, raw: object) -> tuple[str, object]:
    parsed = _parse_json(raw)
    if parsed is None:
        return EMPTY_RESULT_TEXT, None
    if parsed is _UNPARSEABLE or not isinstance(parsed, (dict, list)):
        if isinstance(raw, str):
            return raw, raw
        return _serialize(raw), raw

    value: object = parsed
    if isinstance(value, dict):
        allowed = FIELD_ALLOWLISTS[name]
        value = {field: value[field] for field in allowed if field in value}
        for field, mapping in ENUM_MAPS.get(name, {}).items():
            if field in value and value[field] in mapping:
                value[field] = mapping[value[field]]
        for field, item_maps in NESTED_ENUM_MAPS.get(name, {}).items():
            items = value.get(field)
            if isinstance(items, list):
                translated: list[object] = []
                for item in items:
                    if isinstance(item, dict):
                        item = dict(item)
                        for item_field, mapping in item_maps.items():
                            if item.get(item_field) in mapping:
                                item[item_field] = mapping[item[item_field]]
                    translated.append(item)
                value[field] = translated
    return _serialize(value), value


def _format_unknown(raw: object) -> tuple[str, object]:
    if isinstance(raw, str):
        return raw, raw
    return json.dumps(raw, ensure_ascii=False, default=str), raw


def format_result(
    spec: ToolSpec, raw: object, max_chars: int
) -> tuple[str, object]:
    """Return ``(model_content, data)`` for a tool's raw result."""
    if raw is None or (isinstance(raw, str) and raw == ""):
        return EMPTY_RESULT_TEXT, None

    name = spec.name

    if name == "query_faq":
        # Knowledge-base evidence is returned verbatim: no filtering, no
        # JSON wrapping, and no truncation.
        content = raw if isinstance(raw, str) else _serialize(raw)
        return content, raw

    if name == "create_ticket":
        content = json.dumps({"ticket_no": raw}, ensure_ascii=False)
        return _truncate(content, max_chars), raw

    if name in FIELD_ALLOWLISTS:
        content, data = _format_known(name, raw)
    else:
        content, data = _format_unknown(raw)
    return _truncate(content, max_chars), data
