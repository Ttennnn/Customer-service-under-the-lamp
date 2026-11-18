"""第 5 章客服工作流的共享 State 定义。

七类意图、四个出口,以及父图(``CustomerServiceState``)与 Agent 子图
(``AgentState``)的 TypedDict。字段与设计文档 §6.3 / §6.5 一致,子图与父图
共享同名键。
"""

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

Intent = Literal["物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "闲聊"]
Route = Literal["knowledge", "business", "complaint", "chitchat"]


class CustomerServiceState(TypedDict, total=False):
    """父图 State:一轮会话的输入、路由、检索证据与输出。"""

    messages: Annotated[list[AnyMessage], add_messages]
    user_message: str
    resolved_message: str
    intent: str
    route: str
    evidence: list[dict]
    sources: list[dict]
    confidence: float | None
    gate_passed: bool
    gate_reason: str
    rerank_failed: bool
    answer: str
    suggested_actions: list[dict]
    step_count: int
    token_estimate: int
    token_budget: int
    max_steps: int
    conversation_id: int
    exhausted: bool


class AgentState(TypedDict, total=False):
    """Agent 子图 State:与父图共享消息、证据、预算与输出键。"""

    messages: Annotated[list[AnyMessage], add_messages]
    evidence: list[dict]
    route: str
    step_count: int
    token_estimate: int
    token_budget: int
    max_steps: int
    answer: str
    suggested_actions: list[dict]
    exhausted: bool
    conversation_id: int
