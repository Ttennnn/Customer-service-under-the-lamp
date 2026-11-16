"""Faithfulness judging for generated answers.

The judge asks a temperature-0 model whether every factual claim in a generated
answer is supported by the retrieved evidence snapshot. It is deliberately
independent from generation so the evaluation job (Task 12) can reuse the same
verdict for every strategy and record unfaithful cases in the ``faith_cases``
ledger.

``llm`` is supplied by the caller and is expected to be a temperature-0
``ChatOpenAI``/DeepSeek-compatible model. DeepSeek only supports JSON output via
``response_format={"type": "json_object"}``, which LangChain exposes as
``method="json_mode"`` (see ``app/services/extract.py``).
"""

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field


class FaithfulnessVerdict(BaseModel):
    """Whether an answer is grounded in the supplied evidence."""

    faithful: bool
    unsupported_claims: list[str] = Field(default_factory=list)
    reason: str = ""


FAITHFULNESS_SYSTEM_PROMPT = """\
你是电商客服答案的忠实性裁判。请只依据本轮提供的知识库证据,逐句判断生成回答中的事实性陈述是否都有证据支持。

判定规则:
- 对回答中的每一条事实性陈述(事实、数字、时限、费用、条件、型号参数、承诺等)逐句核对,判断能否在证据中找到直接支持。
- 只要有一条事实性陈述没有证据支持,即 faithful=false,并把该陈述的原文逐条列入 unsupported_claims。
- 所有事实性陈述都能被证据支持 → faithful=true,unsupported_claims 为空列表。
- 不要使用证据之外的知识为回答补台;不要因为措辞不同而误判,但证据没有写出的具体承诺一律视为无支持。
- 证据为空时,任何事实性陈述都视为无支持,faithful=false。
- reason 用一句话说明判定依据。

只输出一个合法的 JSON 对象,字段为 faithful(布尔值)、unsupported_claims(字符串数组)和 reason(字符串),不要输出任何解释、前后缀或 Markdown 代码块。输出示例:
{"faithful": false, "unsupported_claims": ["退款当天到账"], "reason": "证据只写明 1-3 个工作日,没有当天到账的承诺"}"""


def _build_user_prompt(question: str, answer: str, citations: list[dict]) -> str:
    lines = [
        f"用户问题(question):{question}",
        f"生成回答(answer):{answer}",
        "",
        f"本轮证据(citations,{len(citations)} 条):",
    ]
    if not citations:
        lines.append("(无可用证据)")
    for citation in citations:
        lines.extend(
            [
                f"- n={citation.get('n')}",
                f"  chunk_id:{citation.get('chunk_id') or '(未知)'}",
                f"  section_path:{citation.get('section_path') or '(未知)'}",
                f"  question:{citation.get('question') or '(无)'}",
                f"  answer:{citation.get('answer') or '(无)'}",
            ]
        )
    return "\n".join(lines)


def judge_faithfulness(
    llm,
    question: str,
    answer: str,
    citations: list[dict],
    retries: int = 1,
) -> FaithfulnessVerdict:
    """Judge whether ``answer`` is fully supported by ``citations``.

    ``retries`` is the number of retries *after* the first attempt. Any
    exception is retried; once attempts are exhausted the last exception is
    re-raised rather than swallowed, so an infrastructure failure stays
    distinguishable from an unfaithful answer.

    A verdict that lists unsupported claims is normalised to
    ``faithful=False`` even if the model contradicts itself, keeping the
    ledger/aggregation rule "unfaithful if any claim is unsupported".
    """
    messages = [
        SystemMessage(content=FAITHFULNESS_SYSTEM_PROMPT),
        HumanMessage(content=_build_user_prompt(question, answer, citations)),
    ]
    attempts = max(0, retries) + 1
    for attempt in range(attempts):
        try:
            structured = llm.with_structured_output(
                FaithfulnessVerdict, method="json_mode"
            )
            verdict = structured.invoke(messages)
            if verdict.unsupported_claims and verdict.faithful:
                verdict = verdict.model_copy(update={"faithful": False})
            return verdict
        except Exception:
            if attempt == attempts - 1:
                raise
    raise AssertionError("unreachable")  # pragma: no cover
