"""FAQ 检索工具。

用 SQL LIKE 在 ``faq`` 表的问题(question)与答案(answer)字段中做模糊
检索,命中时返回格式化问答文本,未命中返回「未找到相关问题」提示。

工具内部自行通过 ``get_session_factory()()`` 开启新会话(工具在子线程执行,
不能复用调用方的会话),查询完成后关闭会话。
"""

from sqlalchemy import or_

from langchain.tools import tool

from app.db import session as db_session
from app.models import Faq


@tool
def query_faq(keyword: str) -> str:
    """查询常见问题(FAQ)。

    在 faq 表的问题(question)与答案(answer)字段中做模糊匹配,
    返回匹配到的问题与答案文本;若没有匹配,返回未找到提示。

    Args:
        keyword: 用于模糊检索的关键词
    """
    s = db_session.get_session_factory()()
    try:
        rows = (
            s.query(Faq)
            .filter(
                or_(
                    Faq.question.like(f"%{keyword}%"),
                    Faq.answer.like(f"%{keyword}%"),
                )
            )
            .all()
        )
    finally:
        s.close()

    if not rows:
        return "未找到相关问题,请换个关键词再试试。"

    return "\n\n".join(f"问:{row.question}\n答:{row.answer}" for row in rows)
