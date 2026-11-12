"""工单创建工具。

根据当前会话上下文(``app.tools.context.current_conversation_id``)创建工单:
生成当日递增的工单号,写入 ``Ticket``(status 默认「待处理」),并把关联
``Conversation.status`` 置为「已转人工」,返回生成的工单号。

工具内部自行通过 ``get_session_factory()()`` 开启新会话(工具可能在子线程
执行,不能复用调用方的会话),提交后关闭;对工单号主键冲突做少量重试。
"""

from datetime import datetime
from typing import Literal

from sqlalchemy.exc import IntegrityError

from langchain.tools import tool

from app.db import session as db_session
from app.models import Conversation, Ticket
from app.tools.context import current_conversation_id


@tool
def create_ticket(description: str, ticket_type: Literal["售后", "投诉", "咨询"]) -> str:
    """创建工单。

    为当前会话创建一张工单:工单号按「T + 当日日期(YYYYMMDD) + 三位递增
    序号」生成;工单状态默认「待处理」,并把关联会话状态置为「已转人工」。

    Args:
        description: 工单描述(问题详情)
        ticket_type: 工单类型,取值「售后」「投诉」「咨询」之一
    """
    cid = current_conversation_id.get()
    if cid is None:
        return "创建工单失败:缺少会话上下文。"

    date = datetime.now().strftime("%Y%m%d")
    prefix = f"T{date}"

    s = db_session.get_session_factory()()
    try:
        ticket_no = None
        for _ in range(5):
            count = s.query(Ticket).filter(Ticket.ticket_no.startswith(prefix)).count()
            ticket_no = f"{prefix}{count + 1:03d}"
            s.add(
                Ticket(
                    ticket_no=ticket_no,
                    conversation_id=cid,
                    description=description,
                    ticket_type=ticket_type,
                )
            )
            conversation = s.get(Conversation, cid)
            if conversation is not None:
                conversation.status = "已转人工"
            try:
                s.commit()
                break
            except IntegrityError:
                s.rollback()
        else:
            return "创建工单失败:工单号生成冲突,请重试。"
    finally:
        s.close()

    return ticket_no
