"""物流 MCP server(端口 8101)。

``query_logistics`` 用 tracking_no 的 SHA-256 派生稳定随机数,同一个单号
永远返回同一份数据。server 仅用于演示/联调,不连接真实物流系统。

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
PORT = 8101

server = FastMCP("logistics", host=HOST, port=PORT)

CARRIERS = ("顺丰速运", "中通快递", "圆通速递", "京东物流")
STATUS_CODES = ("PICKED_UP", "TRANSIT", "ARRIVED", "DELIVERING", "DELIVERED")
_TRACE_DESCRIPTIONS = (
    "快件已揽收",
    "快件已发往转运中心",
    "快件已到达转运中心",
    "快件正在派送中",
    "快件已签收",
)


def _rng(key: str) -> random.Random:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _carrier(tracking_no: str, rng: random.Random) -> str:
    prefixes = {"SF": "顺丰速运", "ZT": "中通快递", "YT": "圆通速递", "JD": "京东物流"}
    for prefix, carrier in prefixes.items():
        if tracking_no.upper().startswith(prefix):
            return carrier
    return rng.choice(CARRIERS)


@server.tool()
def query_logistics(tracking_no: str) -> str:
    """查询包裹物流轨迹。

    Args:
        tracking_no: 快递单号
    """
    time.sleep(float(os.environ.get("MOCK_DELAY_SECONDS", "0")))
    rng = _rng(tracking_no)
    status_index = rng.randrange(len(STATUS_CODES))
    day = rng.randint(1, 28)
    traces = [
        {
            "time": f"2026-01-{day:02d} {9 + i:02d}:{rng.randint(0, 59):02d}:00",
            "status_code": STATUS_CODES[i],
            "description": _TRACE_DESCRIPTIONS[i],
        }
        for i in range(status_index + 1)
    ]
    return json.dumps(
        {
            "tracking_no": tracking_no,
            "carrier": _carrier(tracking_no, rng),
            "status_code": STATUS_CODES[status_index],
            "traces": traces,
        },
        ensure_ascii=False,
    )


if __name__ == "__main__":
    server.run(transport="streamable-http")
