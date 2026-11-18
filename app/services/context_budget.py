import logging
import math
from dataclasses import dataclass

from app.config import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ContextBudget:
    window: int
    fixed: int
    transient: int
    window_surplus: int
    target: int
    history: int
    layer1: int
    layer2: int
    safety: int
    summary_tokens: int


def compute_budget(settings: Settings, summary_tokens: int = 0) -> ContextBudget:
    safety = math.ceil(settings.model_context_window * settings.safety_margin_ratio)
    fixed = (
        settings.system_prompt_reserve_tokens
        + settings.effective_rerank_top_k * settings.evidence_tokens_per_chunk
        + max(settings.summary_reserve_tokens, summary_tokens)
        + settings.max_output_tokens
        + safety
    )
    transient = (
        settings.max_user_input_tokens
        + settings.effective_max_agent_steps * settings.effective_tool_result_max_tokens
    )
    surplus = settings.model_context_window - fixed - transient
    target = settings.history_target_turns * settings.history_tokens_per_turn
    history = max(0, min(target, surplus))
    layer1 = history * 7 // 10
    layer2 = history - layer1
    return ContextBudget(
        window=settings.model_context_window,
        fixed=fixed,
        transient=transient,
        window_surplus=surplus,
        target=target,
        history=history,
        layer1=layer1,
        layer2=layer2,
        safety=safety,
        summary_tokens=summary_tokens,
    )


def validate_budget(settings: Settings) -> ContextBudget:
    budget = compute_budget(settings)
    logger.info(
        "context budget window=%d fixed=%d transient=%d history=%d layer1=%d layer2=%d",
        budget.window,
        budget.fixed,
        budget.transient,
        budget.history,
        budget.layer1,
        budget.layer2,
    )
    if budget.history < settings.history_tokens_per_turn or budget.layer1 <= 0:
        logger.warning(
            "上下文预算不足 window=%d fixed=%d transient=%d history=%d layer1=%d",
            budget.window,
            budget.fixed,
            budget.transient,
            budget.history,
            budget.layer1,
        )
    return budget
