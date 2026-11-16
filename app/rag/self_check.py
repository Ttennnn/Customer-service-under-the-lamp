"""Evidence sufficiency self-check run before answer generation.

The check asks the model whether the retrieved knowledge evidence is enough to
answer the user's question completely and accurately. It is intentionally
independent from retrieval and generation so that an infrastructure failure is
visible to the caller: :func:`assess_evidence` re-raises the last error after
exhausting retries, and the chat layer decides on a conservative refusal
without polluting the low-confidence pool.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel


class EvidenceAssessment(BaseModel):
    """Whether the supplied evidence can answer the question."""

    useful: bool
    reason: str = ""


SELF_CHECK_SYSTEM_PROMPT = """\
你是电商客服知识库的证据充分性审核员。请仅依据本轮提供的知识库证据,判断能否完整、准确地回答用户问题。

判定规则:
- 仅凭这些证据就能完整、准确回答 → useful=true。
- 证据不足、只沾边、互相矛盾,或回答需要库外知识 → useful=false,并给出简短 reason。
- 不要脑补证据里没有的事实;证据为空时一律 useful=false。

只输出一个合法的 JSON 对象,字段为 useful(布尔值)和 reason(字符串),不要输出任何解释、前后缀或 Markdown 代码块。输出示例:
{"useful": false, "reason": "证据只涉及退货政策,未提以旧换新"}"""


def _build_user_prompt(question: str, sources: list[dict]) -> str:
    lines = [f"用户问题(question):{question}", "", f"本轮证据({len(sources)} 条):"]
    if not sources:
        lines.append("(无可用证据)")
    for source in sources:
        score = source.get("score")
        lines.extend(
            [
                f"- n={source.get('n')}",
                f"  section_path:{source.get('section_path') or '(未知)'}",
                f"  answer:{source.get('answer') or ''}",
                f"  score:{score if score is not None else '(无)'}",
            ]
        )
    return "\n".join(lines)


def assess_evidence(
    llm, question: str, sources: list[dict], retries: int = 1
) -> EvidenceAssessment:
    """Assess whether ``sources`` are enough to answer ``question``.

    ``retries`` is the number of retries *after* the first attempt. Any
    exception is retried; once attempts are exhausted the last exception is
    re-raised rather than swallowed, so the caller can distinguish an
    infrastructure failure from ``useful=false``.
    """
    messages = [
        SystemMessage(content=SELF_CHECK_SYSTEM_PROMPT),
        HumanMessage(content=_build_user_prompt(question, sources)),
    ]
    attempts = max(0, retries) + 1
    for attempt in range(attempts):
        try:
            structured = llm.with_structured_output(
                EvidenceAssessment, method="json_mode"
            )
            return structured.invoke(messages)
        except Exception:
            if attempt == attempts - 1:
                raise
    raise AssertionError("unreachable")  # pragma: no cover
