"""Mock 业务数据纯函数。

提供确定性的订单快照数据,不依赖 LangChain,供内置工具与图节点复用:

- ``DEMO_ORDERS``:演示订单目录,退款子流程的订单选择器与工具共用;
- ``list_user_orders``:返回三张订单卡的深拷贝;
- ``order_snapshot``:按订单号返回订单快照 dict,未知订单号用稳定 RNG 生成;
- ``_days_ago`` / ``_order_rng``:日期与稳定随机数的内部 helper。

``order_snapshot`` 对未知订单号生成与历史 ``query_order`` 完全相同的字段
(``order_id / items / amount / status / tracking_no / created_at /
signed_at``),同一 id 跨进程结果一致。
"""

import copy
import hashlib
import random
from datetime import datetime, timedelta

_NOW = datetime.now()


def _days_ago(days: int) -> str:
    return (_NOW - timedelta(days=days)).strftime("%Y-%m-%d")


def _order_rng(order_id: str) -> random.Random:
    """Return a stable RNG for an order id.

    Python's built-in ``hash()`` is salted per process, so it cannot make the
    mock order stable across restarts. A SHA-256 seed does: the same order id
    yields the same status/items/amount/tracking on every call.
    """
    seed = int.from_bytes(hashlib.sha256(order_id.encode("utf-8")).digest()[:8], "big")
    return random.Random(seed)


# 演示订单目录:退款子流程的订单选择器与 ``order_snapshot`` 共用同一份数据。
# 日期按模块导入时间相对生成,保证「7 天无理由」等政策演示稳定。
DEMO_ORDERS: dict[str, dict] = {
    "1001": {
        "order_id": "1001",
        "items": [{"name": "无线蓝牙耳机", "qty": 1, "price": 299.0}],
        "amount": 299.0,
        "status": "已签收",
        "tracking_no": "SF1234567890",
        "created_at": _days_ago(7),
        "signed_at": _days_ago(3),
    },
    "1002": {
        "order_id": "1002",
        "items": [{"name": "智能手环", "qty": 1, "price": 199.0}],
        "amount": 199.0,
        "status": "运输中",
        "tracking_no": "SF2345678901",
        "created_at": _days_ago(2),
        "signed_at": None,
    },
    "1003": {
        "order_id": "1003",
        "items": [{"name": "移动电源", "qty": 2, "price": 89.0}],
        "amount": 178.0,
        "status": "待发货",
        "tracking_no": "SF3456789012",
        "created_at": _days_ago(1),
        "signed_at": None,
    },
}


def list_user_orders(session_id: str) -> list[dict]:
    """返回演示订单目录里的三张订单卡(深拷贝,调用方可安全修改)。"""
    return [copy.deepcopy(DEMO_ORDERS[order_id]) for order_id in ("1001", "1002", "1003")]


def order_snapshot(order_id: str) -> dict | None:
    """Return the mock order snapshot for *order_id*.

    Demo ids return a deep copy of ``DEMO_ORDERS``; any other non-empty id
    gets stable generated data with the same field shape (including
    ``tracking_no`` and the ``created_at`` / ``signed_at`` dates). A falsy
    id returns ``None``.
    """
    if not order_id:
        return None
    if order_id in DEMO_ORDERS:
        return copy.deepcopy(DEMO_ORDERS[order_id])

    rng = _order_rng(order_id)
    item_names = ["无线蓝牙耳机", "智能手环", "移动电源", "机械键盘", "保温杯"]
    items = [
        {
            "name": rng.choice(item_names),
            "qty": rng.randint(1, 3),
            "price": round(rng.uniform(50, 500), 2),
        }
        for _ in range(rng.randint(1, 3))
    ]
    amount = round(sum(item["price"] * item["qty"] for item in items), 2)
    status = rng.choice(["待付款", "待发货", "运输中", "已签收"])
    tracking_no = "SF" + "".join(str(rng.randint(0, 9)) for _ in range(10))
    created_at = _days_ago(rng.randint(1, 10))
    return {
        "order_id": order_id,
        "items": items,
        "amount": amount,
        "status": status,
        "tracking_no": tracking_no,
        "created_at": created_at,
        "signed_at": created_at if status == "已签收" else None,
    }
