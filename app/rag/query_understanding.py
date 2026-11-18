"""Query understanding for retrieval.

User questions in a customer-service setting are often colloquial and use
everyday words that never appear in the knowledge corpus (``邮费`` vs the
documented ``运费``).  :func:`analyze` asks an LLM for a standard rewritten
query plus keywords, and :func:`expand_synonyms` provides a deterministic
synonym fallback when the LLM is unavailable.  Synonym fan-out happens only
on the retrieval side; stored chunks are never duplicated per synonym.

:func:`expand_queries` is the stricter ch06 fan-out used only by the
refund/after-sales retrieval path: exactly three differently focused queries,
otherwise a ``[query]`` fallback.
"""

import json
import logging
from dataclasses import dataclass
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Colloquial term -> canonical terms used by the knowledge corpus.
SYNONYMS: dict[str, tuple[str, ...]] = {
    "邮费": ("运费", "配送费"),
    "快递费": ("运费",),
    "到账": ("退款", "原路退回"),
    "灯不亮": ("故障", "维修"),
    "换货": ("换新",),
}

_QUERY_REWRITE_SYSTEM_PROMPT = """\
你是电商客服检索助手。请把用户的口语化问题改写成适合知识库检索的标准问法,
并抽取用于关键词检索的词。要求:

- rewritten:改写后的标准问法,保持原意,不要添加用户没有提到的条件或事实;必须把口语词和故障描述替换为知识库的规范说法,改写结果中不要保留口语词本身(如「邮费」→「运费」、「退钱」→「退款」、「灯不亮」→「故障/维修」、「换个新的」→「换货」)。
- keywords:2-5 个检索关键词,优先使用知识库里的规范说法(如用「运费」而不是「邮费」)。
- 只输出一个合法的 JSON 对象,包含 rewritten 和 keywords 两个字段,不要输出解释或 Markdown 代码块。

输出示例:
{"rewritten": "运费怎么计算", "keywords": ["运费", "配送费"]}"""


class QueryRewrite(BaseModel):
    """Structured output schema for the rewrite LLM call."""

    rewritten: str
    keywords: list[str]


class QueryExpansion(BaseModel):
    """Structured output schema for the refund/after-sales expansion call."""

    queries: list[str]


QUERY_EXPANSION_SYSTEM_PROMPT = """\
你是电商客服的检索查询扩写器,只服务退款/退货、售后这类容错低的核心场景。
请把用户问题扩写成 3 条侧重点不同的检索查询,用于检索相关政策和条款。

要求:
1. 必须正好 3 条,每条一句,侧重点各不相同(例如:政策条款、时间或条件限制、流程与所需材料);不要用三条同义改写凑数。
2. 保留用户问题里的订单号、商品名、型号、金额等实体,原样写入,不要编造。
3. 不要引入用户没有提到的条件或参数,不要假设退款原因、日期、地址。
4. 每条查询都能独立检索,不要互相引用(如「上述」「该情况」)。
5. 只输出一个合法的 JSON 对象,字段只有 queries,值是 3 个字符串组成的数组。
6. 不要输出解释、理由或 Markdown 代码块,只输出 JSON。

输出示例:
{"queries": ["退款政策 七天无理由", "已签收商品 退款条件 时间限制", "退款流程 所需材料"]}"""


@dataclass(frozen=True)
class QueryAnalysis:
    """Result of :func:`analyze`.

    ``rewritten_by`` says whether the rewrite came from the LLM or from the
    deterministic fallback.  Filters (category/content_type) are deliberately
    not part of this object: they must be explicit retrieval parameters.
    """

    original: str
    rewritten: str
    keywords: list[str]
    rewritten_by: Literal["llm", "fallback"]


def expand_synonyms(text: str) -> list[str]:
    """Return synonym expansions for keys found in *text*.

    Terms already present in *text* are skipped, and duplicates are removed
    while preserving the order in which they first appear.
    """
    terms: list[str] = []
    seen: set[str] = set()
    for key, expansions in SYNONYMS.items():
        if key not in text:
            continue
        for term in expansions:
            if term in text or term in seen:
                continue
            seen.add(term)
            terms.append(term)
    return terms


def _fallback(query: str) -> QueryAnalysis:
    return QueryAnalysis(
        original=query,
        rewritten=query,
        keywords=expand_synonyms(query),
        rewritten_by="fallback",
    )


def analyze(query: str, llm=None) -> QueryAnalysis:
    """Rewrite *query* and extract keywords, falling back on any failure.

    The default LLM is built lazily so tests can inject a fake and so this
    module does not import application config at import time.
    """
    try:
        if llm is None:
            from app.config import build_llm, get_settings

            llm = build_llm(get_settings(), temperature=0.0)
        structured = llm.with_structured_output(QueryRewrite, method="json_mode")
        result = structured.invoke(
            [
                SystemMessage(content=_QUERY_REWRITE_SYSTEM_PROMPT),
                HumanMessage(content=query),
            ]
        )
    except Exception:
        return _fallback(query)

    rewritten = (result.rewritten or "").strip() or query
    keywords: list[str] = []
    for keyword in result.keywords:
        keyword = (keyword or "").strip()
        if keyword and keyword not in keywords:
            keywords.append(keyword)
    if not keywords:
        keywords = expand_synonyms(query)
    return QueryAnalysis(
        original=query,
        rewritten=rewritten,
        keywords=keywords,
        rewritten_by="llm",
    )


def _render_expansion_input(query: str, order_data: dict | None) -> str:
    """把订单数据渲染进 HumanMessage,让扩写保留订单号/状态等实体。"""
    if not order_data:
        return query
    try:
        data = json.dumps(order_data, ensure_ascii=False, default=str, sort_keys=True)
    except (TypeError, ValueError):
        data = str(order_data)
    return f"订单数据：{data}\n\n用户问题：{query}"


def expand_queries(
    llm,
    query: str,
    order_data: dict | None = None,
) -> list[str]:
    """把 *query* 扩写成 3 条侧重不同的检索查询。

    清洗空白项并按首次出现顺序去重;清洗后不足或超过 3 条、输出为空、
    调用异常,一律回退 ``[query]``,不把不合规的扩写交给检索。
    """
    try:
        structured = llm.with_structured_output(QueryExpansion, method="json_mode")
        result = structured.invoke(
            [
                SystemMessage(content=QUERY_EXPANSION_SYSTEM_PROMPT),
                HumanMessage(content=_render_expansion_input(query, order_data)),
            ]
        )
        raw_queries = getattr(result, "queries", None) or []
        if not isinstance(raw_queries, (list, tuple)):
            raise ValueError(f"queries is not a list: {type(raw_queries).__name__}")
    except Exception as exc:
        logger.warning("ch06 expand_queries fallback to original: %s", exc)
        return [query]

    queries: list[str] = []
    for item in raw_queries:
        if not isinstance(item, str):
            continue
        item = item.strip()
        if item and item not in queries:
            queries.append(item)

    if len(queries) != 3:
        logger.warning(
            "ch06 expand_queries fallback to original: got %d unique queries",
            len(queries),
        )
        return [query]

    logger.info("ch06 expand_queries produced 3 queries")
    return queries
