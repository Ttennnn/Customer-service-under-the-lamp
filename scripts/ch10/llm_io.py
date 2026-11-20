"""Small LLM-call helpers: a call budget and structured output coercion."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage


class BudgetExceeded(RuntimeError):
    """Raised when the configured LLM call cap is reached."""


class CallBudget:
    def __init__(self, max_calls: int | None = None) -> None:
        self.max_calls = max_calls
        self.calls = 0

    def spend(self) -> None:
        if self.max_calls is not None and self.calls >= self.max_calls:
            raise BudgetExceeded(
                f"LLM call budget exhausted ({self.calls}/{self.max_calls})"
            )
        self.calls += 1


def _as_dict(result: Any) -> dict:
    if isinstance(result, dict):
        return result
    dump = getattr(result, "model_dump", None)
    if callable(dump):
        return dump()
    return dict(result)


def invoke_structured(
    llm,
    system: str,
    human: str,
    schema,
    budget: CallBudget,
) -> dict:
    """Spend one budgeted call and return the structured result as a dict."""
    budget.spend()
    structured = llm.with_structured_output(schema, method="json_mode")
    result = structured.invoke(
        [SystemMessage(content=system), HumanMessage(content=human)]
    )
    return _as_dict(result)
