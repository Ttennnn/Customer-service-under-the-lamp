"""裸 ReAct 循环的 CLI demo(第 5 章祛魅热身)。

用法::

    python -m app.agent.raw_demo "订单 1001 的物流到哪了"
"""

import sys

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.raw_loop import run_react_loop
from app.config import build_llm, get_settings
from app.prompts.customer_service import render_system_prompt
from app.tools.registry import ALL_TOOLS


def main() -> int:
    if len(sys.argv) < 2:
        print('用法: python -m app.agent.raw_demo "<question>"', file=sys.stderr)
        return 2

    question = sys.argv[1]
    settings = get_settings()
    events: list[dict] = []
    result = run_react_loop(
        build_llm(settings),
        ALL_TOOLS,
        [
            SystemMessage(content=render_system_prompt()),
            HumanMessage(content=question),
        ],
        max_steps=settings.agent_max_steps,
        token_budget=settings.agent_token_budget,
        emit=events.append,
    )

    # 连续的 token 事件合并成一行,避免流式碎片刷屏;工具事件单独成行。
    token_buffer = ""
    for event in events:
        if event["type"] == "token":
            token_buffer += event["text"]
            continue
        if token_buffer:
            print(f"[token] {token_buffer}")
            token_buffer = ""
        print(f"[tool] {event['label']}")
    if token_buffer:
        print(f"[token] {token_buffer}")

    print(f"steps={result.steps}")
    print(f"token_estimate={result.token_estimate}")
    print(f"exhausted={result.exhausted}")
    print(f"answer={result.answer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
