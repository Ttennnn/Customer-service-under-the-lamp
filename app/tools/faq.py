"""FAQ semantic retrieval tool.

Embeds the user's keyword with BGE-M3 and searches the Milvus ``knowledge``
collection, returning the matching questions and answers formatted for the
model.  Unlike the previous SQL ``LIKE`` implementation this is semantic
similarity search, so paraphrases and synonyms can match.

Infrastructure failures (embedding API or Milvus down) are caught and turned
into a safe apology so a retrieval problem never breaks the chat round.
"""

import logging

from langchain.tools import tool

from app.rag.embedding import get_embedder
from app.rag.milvus_store import get_vector_store
from app.rag.retrieval import retrieve

logger = logging.getLogger(__name__)

_MISS_TEXT = "未找到相关问题,请换个关键词再试试。"
_ERROR_TEXT = "知识库暂时不可用,请稍后重试或联系人工客服。"


@tool
def query_faq(keyword: str) -> str:
    """在知识库中语义检索常见问题(FAQ)。

    将关键词向量化后在 Milvus 知识库中做相似度检索,返回语义最接近的
    问题与答案;若没有足够相近的内容,返回未找到提示。

    Args:
        keyword: 用于语义检索的问题或关键词
    """
    try:
        results = retrieve(
            keyword, embedder=get_embedder(), store=get_vector_store()
        )
    except Exception:  # noqa: BLE001 - 工具必须返回字符串,不能打断对话
        logger.exception("query_faq retrieval failed")
        return _ERROR_TEXT

    if not results:
        return _MISS_TEXT

    return "\n\n".join(f"问:{r.questions}\n答:{r.answer}" for r in results)
