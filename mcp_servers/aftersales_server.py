"""售后 MCP server(端口 8102)。

提供 ``query_warranty`` 与 ``query_return_status``,按 order_id 的 SHA-256
派生稳定随机数;不连接数据库或真实售后系统。

注意:installed ``mcp==1.30.0`` 的 ``FastMCP.run`` 只接受
``transport`` / ``mount_path``;host/port 在构造 ``FastMCP`` 时指定。
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time

from mcp.server.fastmcp import FastMCP

HOST = "127.0.0.1"
PORT = 8102

server = FastMCP("aftersales", host=HOST, port=PORT)

PRODUCTS = ("无线蓝牙耳机", "智能手表", "扫地机器人", "空气净化器", "电热水壶")
WARRANTY_STATUSES = ("IN_WARRANTY", "OUT_OF_WARRANTY", "UNKNOWN")
RETURN_STATUSES = ("APPLIED", "PICKING", "IN_TRANSIT", "RECEIVED", "REFUNDING", "DONE")
COVERAGES = ("整机保修一年", "主要部件保修两年", "仅支持七天无理由退货")


def _rng(order_id: str) -> random.Random:
    digest = hashlib.sha256(order_id.encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


@server.tool()
def query_warranty(order_id: str) -> str:
    """查询订单商品的保修状态。

    Args:
        order_id: 订单号
    """
    time.sleep(float(os.environ.get("MOCK_DELAY_SECONDS", "0")))
    rng = _rng(order_id)
    purchase_month = rng.randint(1, 12)
    purchase_day = rng.randint(1, 28)
    purchase_date = f"2024-{purchase_month:02d}-{purchase_day:02d}"
    expire_date = f"2026-{purchase_month:02d}-{purchase_day:02d}"
    return json.dumps(
        {
            "order_id": order_id,
            "product_name": rng.choice(PRODUCTS),
            "warranty_status": rng.choice(WARRANTY_STATUSES),
            "purchase_date": purchase_date,
            "expire_date": expire_date,
            "coverage": rng.choice(COVERAGES),
        },
        ensure_ascii=False,
    )


@server.tool()
def query_return_status(order_id: str) -> str:
    """查询订单的退货/退款进度。

    Args:
        order_id: 订单号
    """
    time.sleep(float(os.environ.get("MOCK_DELAY_SECONDS", "0")))
    rng = _rng(order_id)
    status_index = rng.randrange(len(RETURN_STATUSES))
    return json.dumps(
        {
            "order_id": order_id,
            "return_no": f"R{10**10 + rng.randrange(10**9)}",
            "status": RETURN_STATUSES[status_index],
            "progress": f"{(status_index + 1) * 100 // len(RETURN_STATUSES)}%",
            "updated_at": f"2026-01-{rng.randint(1, 28):02d} {rng.randint(8, 20):02d}:00:00",
        },
        ensure_ascii=False,
    )


if __name__ == "__main__":
    server.run(transport="streamable-http")
