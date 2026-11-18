from typing_extensions import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class CounterState(TypedDict, total=False):
    selected: str


def test_interrupt_surfaces_and_resumes_from_node_start():
    calls: list[str] = []

    def gate(state: CounterState) -> dict:
        calls.append("pre")
        selected = interrupt({"kind": "select_order", "orders": ["1001"]})
        calls.append("post")
        return {"selected": selected}

    graph = StateGraph(CounterState)
    graph.add_node("gate", gate)
    graph.add_edge(START, "gate")
    graph.add_edge("gate", END)
    compiled = graph.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "smoke"}}

    list(compiled.stream({}, config, stream_mode="custom"))
    snap = compiled.get_state(config)
    assert snap.interrupts
    assert snap.interrupts[0].value["kind"] == "select_order"

    list(compiled.stream(Command(resume="1001"), config, stream_mode="custom"))
    assert compiled.get_state(config).values["selected"] == "1001"
    assert calls == ["pre", "pre", "post"]
