"""最裸的 ReAct 循环(第 5 章祛魅热身)。

不使用 LangGraph:把模型绑定工具后流式调用,累加 ``AIMessageChunk``;模型要求
工具调用时就执行工具、把结果作为 ``ToolMessage`` 喂回消息列表,直到模型不再调用
工具(正常收敛)或触发 ``max_steps`` / ``token_budget`` 停止条件。
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool

from app.services.context import estimate_tokens
from app.tools.executor import execute_tool
from app.tools.registry import TOOLS_BY_NAME, TOOL_LABELS, tool_timeout


@dataclass(frozen=True)
class AgentLoopResult:
    answer: str
    messages: list[AnyMessage]
    steps: int
    token_estimate: int
    exhausted: bool


def run_react_loop(
    llm,
    tools: Sequence[BaseTool],
    messages: Sequence[BaseMessage],
    *,
    max_steps: int = 6,
    token_budget: int = 12000,
    emit: Callable[[dict], None] | None = None,
) -> AgentLoopResult:
    """Run a framework-free ReAct loop and return its converged result.

    ``emit`` receives ``{"type": "token", "text": ...}`` for streamed model
    content and ``{"type": "tool", "tool": ..., "label": ...}`` before each
    tool execution. Stopping at ``max_steps`` or ``token_budget`` returns an
    empty answer with ``exhausted=True``.
    """
    working: list[AnyMessage] = list(messages)
    steps = 0
    tokens = sum(estimate_tokens(getattr(m, "content", None)) for m in working)

    while True:
        if steps >= max_steps or tokens >= token_budget:
            return AgentLoopResult("", working, steps, tokens, True)

        bound = llm.bind_tools(tools)
        gathered = None
        for chunk in bound.stream(working):
            if chunk.content and emit is not None:
                emit({"type": "token", "text": chunk.content})
            gathered = chunk if gathered is None else gathered + chunk

        steps += 1
        ai_message = AIMessage(
            content=gathered.content or "",
            tool_calls=gathered.tool_calls,
        )
        working.append(ai_message)
        tokens += estimate_tokens(ai_message.content)

        if not ai_message.tool_calls:
            return AgentLoopResult(
                ai_message.content or "", working, steps, tokens, False
            )

        for tool_call in ai_message.tool_calls:
            name = tool_call["name"]
            if emit is not None:
                emit(
                    {
                        "type": "tool",
                        "tool": name,
                        "label": TOOL_LABELS.get(name, name),
                    }
                )

            tool = TOOLS_BY_NAME.get(name)
            if tool is None:
                result = f"未知工具:{name}"
            else:
                result = execute_tool(
                    tool,
                    tool_call["args"],
                    timeout_s=tool_timeout(name),
                )

            working.append(
                ToolMessage(content=result, tool_call_id=tool_call["id"])
            )
            tokens += estimate_tokens(result)
