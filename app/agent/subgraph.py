"""Agent ReAct 子图(设计文档 §6.5)。

子图在父图路由到 ``knowledge`` / ``business`` / ``refund_flow`` 时运行,负责:

- ``agent``:按 ``route`` 组装 system prompt 与工具子集(``refund_flow``
  用售后 prompt + 退款控制工具,其余用通用 prompt + 业务工具),流式调用
  绑定工具的模型,累加 ``AIMessageChunk`` 并把完整回复写回 ``messages``;
- ``tools``:执行业务工具并经 ``current_conversation_id`` 传播会话上下文;
  ``suggest_handoff`` 与 ``submit_refund`` 是控制工具,只把建议合并进
  ``suggested_actions``,绝不调用 ``execute_tool``;
- ``finalize``:步数/token 预算耗尽或答案为空时,发出固定兜底话术、
  追加 ``AIMessage``,并在没有选项时补一个「转人工」。

流式帧只允许两种:``{"type": "token", "text": ...}`` 和
``{"type": "tool", "tool": ..., "label": ...}``。编译时不带 checkpointer,
由父图编译时自动传播。
"""

from collections.abc import Sequence
from typing import Any

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    AnyMessage,
    BaseMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from app.agent.state import AgentState
from app.agent.tools import (
    AGENT_TOOLS_BY_NAME,
    BUSINESS_TOOL_NAMES,
    suggested_actions_from_handoff,
    suggested_actions_from_refund,
    tools_for_route,
)
from app.prompts.customer_service import (
    render_after_sales_agent_system_prompt,
    render_agent_system_prompt,
)
from app.services.context import build_agent_messages, estimate_tokens
from app.tools.context import current_conversation_id
from app.tools.executor import execute_tool
from app.tools.registry import TOOL_LABELS, tool_timeout

FALLBACK_ANSWER = (
    "抱歉,这个问题需要更多信息才能继续处理。"
    "您可以补充订单号或具体诉求,也可以转人工客服进一步处理。"
)
TRANSFER_ACTION: dict[str, str] = {"action": "transfer", "label": "转人工"}

# 控制工具不在业务工具注册表里,单独给一个流式状态标签。
_CONTROL_TOOL_LABELS = {
    "suggest_handoff": "转人工建议",
    "submit_refund": "准备退款表单",
}
_INVALID_HANDOFF_OPTIONS = "参数无效: options 只能是 transfer/ticket。"
_INVALID_REFUND_ORDER = "订单号与当前订单不一致,无法准备退款表单。"
_VALID_HANDOFF_OPTIONS = frozenset({"transfer", "ticket"})
_VALID_TICKET_TYPES = frozenset({"售后", "投诉", "咨询"})


def _tool_label(name: str) -> str:
    return TOOL_LABELS.get(name) or _CONTROL_TOOL_LABELS.get(name, name)


def _merge_actions(
    existing: Sequence[dict],
    incoming: Sequence[dict],
) -> list[dict]:
    """Merge handoff actions by ``action``.

    Existing actions are not replaced; for a repeated ``ticket`` action the
    later ``ticket_type`` wins. Order is preserved.
    """
    merged = [dict(action) for action in existing]
    positions = {action["action"]: index for index, action in enumerate(merged)}
    for action in incoming:
        name = action["action"]
        if name not in positions:
            positions[name] = len(merged)
            merged.append(dict(action))
        elif name == "ticket" and "ticket_type" in action:
            merged[positions[name]]["ticket_type"] = action["ticket_type"]
    return merged


def _handoff_payload(args: dict) -> tuple[list[str], str] | None:
    """Return ``(options, ticket_type)`` or ``None`` for invalid arguments.

    An empty option list or any option outside ``transfer`` / ``ticket``
    invalidates the whole call, so nothing is written to State.
    """
    options = args.get("options")
    if not isinstance(options, list) or not options:
        return None
    if any(option not in _VALID_HANDOFF_OPTIONS for option in options):
        return None
    ticket_type = args.get("ticket_type", "咨询")
    if ticket_type not in _VALID_TICKET_TYPES:
        ticket_type = "咨询"
    return options, ticket_type


def _handoff_result(args: dict) -> str:
    """Acknowledge a ``suggest_handoff`` call without executing any action."""
    payload = _handoff_payload(args)
    if payload is None:
        return _INVALID_HANDOFF_OPTIONS
    options, ticket_type = payload
    try:
        return AGENT_TOOLS_BY_NAME["suggest_handoff"].invoke(
            {
                "options": options,
                "ticket_type": ticket_type,
                "reason": str(args.get("reason") or ""),
            }
        )
    except Exception:  # noqa: BLE001 - 控制工具只回确认,不向外抛
        return "已记录转人工/建工单建议。"


def _refund_result(args: dict, order_data: dict) -> tuple[str, dict | None]:
    """Validate a ``submit_refund`` call; return ``(ack, action or None)``.

    Only the order currently carried by the parent graph may produce a
    refund form. Any other order number is rejected without writing State.
    """
    order_id = args.get("order_id")
    current = order_data.get("order_id") if isinstance(order_data, dict) else None
    if not order_id or current is None or str(order_id) != str(current):
        return _INVALID_REFUND_ORDER, None
    try:
        result = AGENT_TOOLS_BY_NAME["submit_refund"].invoke(
            {"order_id": str(order_id)}
        )
    except Exception:  # noqa: BLE001 - 控制工具只回确认,不向外抛
        result = "已准备退款表单。"
    return result, suggested_actions_from_refund(str(order_id))[0]


def build_agent_subgraph(
    llm,
    *,
    max_steps: int = 6,
    token_budget: int = 12000,
    tool_result_max_chars: int = 2000,
    max_context_tokens: int = 4096,
    tools: Sequence[BaseTool] | None = None,
):
    """Return a compiled Agent subgraph (no checkpointer; parent propagates it).

    When ``tools`` is omitted, each route binds its own subset: ``refund_flow``
    gets the refund control tool plus handoff, other routes get the business
    tools plus handoff. An explicit ``tools`` list overrides route selection
    (used by tests and callers with a fixed tool set).
    """
    explicit_tools: list[BaseTool] | None = (
        list(tools) if tools is not None else None
    )

    def agent_node(state: AgentState) -> dict[str, Any]:
        step_count = state.get("step_count", 0) + 1
        token_estimate = state.get("token_estimate", 0)
        if step_count > max_steps or token_estimate >= token_budget:
            return {"step_count": step_count, "exhausted": True}

        route = state.get("route") or "business"
        if explicit_tools is not None:
            bound_tools = explicit_tools
        else:
            bound_tools = tools_for_route(route)
        if route == "refund_flow":
            system_content = render_after_sales_agent_system_prompt(
                order_data=state.get("order_data"),
                evidence=state.get("evidence"),
            )
        else:
            system_content = render_agent_system_prompt(
                evidence=state.get("evidence")
            )
        system = SystemMessage(content=system_content)
        prompt = build_agent_messages(
            system,
            state.get("messages", []),
            max_context_tokens,
        )

        writer = get_stream_writer()
        gathered: AIMessageChunk | None = None
        for chunk in llm.bind_tools(bound_tools).stream(prompt):
            if chunk.content:
                writer({"type": "token", "text": chunk.content})
            gathered = chunk if gathered is None else gathered + chunk

        content = (gathered.content or "") if gathered is not None else ""
        tool_calls = gathered.tool_calls if gathered is not None else []
        ai_message = AIMessage(content=content, tool_calls=tool_calls)

        prompt_tokens = sum(estimate_tokens(m.content) for m in prompt)
        update: dict[str, Any] = {
            "messages": [ai_message],
            "step_count": step_count,
            "token_estimate": token_estimate
            + prompt_tokens
            + estimate_tokens(content),
            "exhausted": False,
        }
        if not tool_calls:
            update["answer"] = content
            if not content:
                update["exhausted"] = True
        return update

    def tools_node(state: AgentState) -> dict[str, Any]:
        messages = state.get("messages", [])
        last = messages[-1] if messages else None
        if not isinstance(last, AIMessage) or not last.tool_calls:
            return {}

        writer = get_stream_writer()
        suggested_actions = list(state.get("suggested_actions") or [])
        token_delta = 0
        new_messages: list[AnyMessage] = []

        for tool_call in last.tool_calls:
            name = tool_call["name"]
            args = tool_call.get("args") or {}
            writer({"type": "tool", "tool": name, "label": _tool_label(name)})

            if name == "suggest_handoff":
                result = _handoff_result(args)
                payload = _handoff_payload(args)
                if payload is not None:
                    options, ticket_type = payload
                    suggested_actions = _merge_actions(
                        suggested_actions,
                        suggested_actions_from_handoff(options, ticket_type),
                    )
            elif name == "submit_refund":
                result, refund_action = _refund_result(
                    args, state.get("order_data") or {}
                )
                if refund_action is not None:
                    suggested_actions = _merge_actions(
                        suggested_actions, [refund_action]
                    )
            elif name in BUSINESS_TOOL_NAMES:
                tool = AGENT_TOOLS_BY_NAME[name]
                current_conversation_id.set(state["conversation_id"])
                result = execute_tool(
                    tool,
                    args,
                    timeout_s=tool_timeout(name),
                )
                result = str(result)[:tool_result_max_chars]
            else:
                result = f"未知工具:{name}"

            token_delta += estimate_tokens(result)
            new_messages.append(
                ToolMessage(content=result, tool_call_id=tool_call.get("id") or "")
            )

        return {
            "messages": new_messages,
            "suggested_actions": suggested_actions,
            "token_estimate": state.get("token_estimate", 0) + token_delta,
        }

    def finalize_node(state: AgentState) -> dict[str, Any]:
        writer = get_stream_writer()
        writer({"type": "token", "text": FALLBACK_ANSWER})
        suggested_actions = list(state.get("suggested_actions") or [])
        if not suggested_actions:
            suggested_actions = [dict(TRANSFER_ACTION)]
        return {
            "messages": [AIMessage(content=FALLBACK_ANSWER)],
            "answer": FALLBACK_ANSWER,
            "suggested_actions": suggested_actions,
            "exhausted": True,
        }

    def should_continue(state: AgentState) -> str:
        if state.get("exhausted"):
            return "finalize"
        messages = state.get("messages", [])
        last = messages[-1] if messages else None
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        if not state.get("answer"):
            return "finalize"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_node("finalize", finalize_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "finalize": "finalize", END: END},
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("finalize", END)
    return graph.compile()
