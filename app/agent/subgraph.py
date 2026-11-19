"""Agent ReAct 子图(设计文档 §6.5)。

子图在父图路由到 ``knowledge`` / ``business`` / ``refund_flow`` 时运行,负责:

- ``agent``:按 ``route`` 从注册中心绑定工具子集(``refund_flow`` 用售后
  prompt + 退款控制工具,其余用通用 prompt + 业务工具),流式调用绑定
  工具的模型,累加 ``AIMessageChunk`` 并把完整回复写回 ``messages``;
- ``tools``:所有工具调用统一走 :mod:`app.tools.engine`;``suggest_handoff``
  与 ``submit_refund`` 先做参数/订单校验,再把建议合并进
  ``suggested_actions``;
- ``finalize``:步数/token 预算耗尽或答案为空时,发出固定兜底话术、
  追加 ``AIMessage``,并在没有选项时补一个「转人工」。

流式帧只允许两种:``{"type": "token", "text": ...}`` 和
``{"type": "tool", "tool": ..., "label": ...}``。编译时不带 checkpointer,
由父图编译时自动传播。
"""

import logging
import math
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    AnyMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from app.agent.state import AgentState
from app.agent.tools import (
    suggested_actions_from_handoff,
    suggested_actions_from_refund,
    tools_for_route,
)
from app.config import Settings, get_settings
from app.prompts.customer_service import (
    render_after_sales_agent_system_prompt,
    render_agent_system_prompt,
    render_turn_material,
)
from app.services.context import (
    assemble_agent_prompt,
    estimate_tokens,
    log_context,
    message_tokens,
)
from app.tools.engine import deny_validation, execute, execute_tool_call
from app.tools.registry import TOOL_LABELS, get_all_specs
from app.tools.spec import ToolSpec

_logger = logging.getLogger(__name__)

FALLBACK_ANSWER = (
    "抱歉,这个问题需要更多信息才能继续处理。"
    "您可以补充订单号或具体诉求,也可以转人工客服进一步处理。"
)
TRANSFER_ACTION: dict[str, str] = {"action": "transfer", "label": "转人工"}

# 控制工具的 spec 没有 label,单独给一个流式状态标签。
_CONTROL_TOOL_LABELS = {
    "suggest_handoff": "转人工建议",
    "submit_refund": "准备退款表单",
}
_INVALID_HANDOFF_OPTIONS = "参数无效: options 只能是 transfer/ticket。"
_INVALID_REFUND_ORDER = "订单号与当前订单不一致,无法准备退款表单。"
_VALID_HANDOFF_OPTIONS = frozenset({"transfer", "ticket"})
_VALID_TICKET_TYPES = frozenset({"售后", "投诉", "咨询"})


def _tool_label(name: str, spec: ToolSpec | None = None) -> str:
    if spec is not None and spec.label:
        return spec.label
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


def _refund_order_matches(args: dict, order_data: dict) -> bool:
    """Only the order currently carried by the parent graph may be refunded."""
    order_id = args.get("order_id")
    current = order_data.get("order_id") if isinstance(order_data, dict) else None
    return bool(order_id) and current is not None and str(order_id) == str(current)


def build_agent_subgraph(
    llm,
    *,
    max_steps: int = 6,
    tool_result_max_tokens: int = 2000,
    settings: Settings | None = None,
    tools: Sequence[BaseTool] | None = None,
):
    """Return a compiled Agent subgraph (no checkpointer; parent propagates it).

    The system prompt is constant per route, so it stays cacheable. The current
    turn is ``messages[turn_start_index:]``; the summary, evidence, order data
    and intent become one labeled material message inserted after the current
    user message. ``prompt_history`` is trimmed only when the assembled prompt
    exceeds the model window minus the configured safety margin.

    When ``tools`` is omitted, each route binds its own subset from the tool
    registry: ``refund_flow`` gets the refund/handoff/ticket control tools plus
    aftersales MCP tools; other routes get every exposed builtin tool that is
    route-agnostic or scoped to the route. An explicit ``tools`` list overrides
    route selection (used by tests and callers with a fixed tool set).
    """
    settings = settings or get_settings()
    explicit_tools: list[BaseTool] | None = (
        list(tools) if tools is not None else None
    )

    def agent_node(state: AgentState) -> dict[str, Any]:
        step_count = state.get("step_count", 0) + 1
        if step_count > max_steps:
            return {"step_count": step_count, "exhausted": True}

        route = state.get("route") or "business"
        if explicit_tools is not None:
            bound_tools = explicit_tools
        else:
            bound_tools = tools_for_route(route, get_all_specs(settings))
        if route == "refund_flow":
            system_content = render_after_sales_agent_system_prompt()
        else:
            system_content = render_agent_system_prompt()
        system = SystemMessage(content=system_content)

        history = list(state.get("prompt_history") or [])
        turn_start = state.get("turn_start_index", 0) or 0
        current = list(state.get("messages") or [])[turn_start:]
        summary = state.get("summary") or ""
        material = render_turn_material(
            summary,
            state.get("evidence"),
            state.get("order_data"),
            state.get("intent") or "",
        )
        window = settings.model_context_window
        max_prompt_tokens = window - math.ceil(
            window * settings.safety_margin_ratio
        )
        prompt = assemble_agent_prompt(
            system, history, current, material, max_prompt_tokens
        )
        log_context(
            _logger,
            "model_ctx",
            summary,
            prompt,
            len(history),
            preview_chars=settings.context_log_preview_chars,
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

        update: dict[str, Any] = {
            "messages": [ai_message],
            "step_count": step_count,
            "token_estimate": state.get("token_estimate", 0)
            + message_tokens(prompt)
            + estimate_tokens(content),
            "exhausted": False,
        }
        if not tool_calls:
            update["answer"] = content
            if not content:
                update["exhausted"] = True
        return update

    def tools_node(state: AgentState) -> dict[str, Any]:
        specs = get_all_specs(settings)
        messages = state.get("messages", [])
        last = messages[-1] if messages else None
        if not isinstance(last, AIMessage) or not last.tool_calls:
            return {}

        writer = get_stream_writer()
        conversation_id = state["conversation_id"]
        suggested_actions = list(state.get("suggested_actions") or [])
        token_delta = 0
        new_messages: list[AnyMessage] = []

        for tool_call in last.tool_calls:
            name = tool_call["name"]
            args = tool_call.get("args") or {}
            tool_call_id = tool_call.get("id")
            spec = specs.get(name)
            writer(
                {
                    "type": "tool",
                    "tool": name,
                    "label": _tool_label(name, spec),
                }
            )

            if name == "suggest_handoff" and spec is not None:
                payload = _handoff_payload(args)
                if payload is None:
                    run = deny_validation(
                        spec,
                        args,
                        conversation_id=conversation_id,
                        tool_call_id=tool_call_id,
                        error_message=_INVALID_HANDOFF_OPTIONS,
                    )
                else:
                    options, ticket_type = payload
                    run = execute(
                        spec,
                        args,
                        conversation_id=conversation_id,
                        tool_call_id=tool_call_id,
                        settings=settings,
                    )
                    if run.status == "ok":
                        suggested_actions = _merge_actions(
                            suggested_actions,
                            suggested_actions_from_handoff(options, ticket_type),
                        )
            elif name == "submit_refund" and spec is not None:
                if not _refund_order_matches(args, state.get("order_data") or {}):
                    run = deny_validation(
                        spec,
                        args,
                        conversation_id=conversation_id,
                        tool_call_id=tool_call_id,
                        error_message=_INVALID_REFUND_ORDER,
                    )
                else:
                    order_id = str(args["order_id"])
                    run = execute(
                        spec,
                        args,
                        conversation_id=conversation_id,
                        tool_call_id=tool_call_id,
                        settings=settings,
                    )
                    if run.status == "ok":
                        suggested_actions = _merge_actions(
                            suggested_actions,
                            suggested_actions_from_refund(order_id),
                        )
            else:
                run = execute_tool_call(
                    tool_call,
                    conversation_id,
                    specs=specs,
                    settings=settings,
                )

            # One character is one token; keep the raw result out of the window.
            result = str(run.content)[:tool_result_max_tokens]
            token_delta += estimate_tokens(result)
            new_messages.append(
                ToolMessage(content=result, tool_call_id=tool_call_id or "")
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
