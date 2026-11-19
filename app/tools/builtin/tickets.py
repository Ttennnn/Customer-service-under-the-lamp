"""工单创建工具(内置插件)。

根据当前会话上下文(``app.tools.context.current_conversation_id``)创建工单:
生成当日递增的工单号,写入 ``Ticket``(status 默认「待处理」),返回生成的
工单号。只写 tickets 表,不改会话状态;转人工由前端独立模拟。

工具内部自行通过 ``get_session_factory()()`` 开启新会话(工具可能在子线程
执行,不能复用调用方的会话),提交后关闭;对工单号主键冲突做少量重试。

演示模式下 ``Settings.demo_ticket_delay_seconds`` 会让工具在 **提交之后**、
返回工单号之前休眠,用于前端展示「正在创建工单」的过渡态。
"""

import re
import time
from datetime import datetime
from typing import Annotated, Literal

from langchain.tools import tool
from pydantic import Field
from sqlalchemy.exc import IntegrityError

from app.config import get_settings
from app.db import session as db_session
from app.models import Ticket
from app.tools.context import current_conversation_id
from app.tools.spec import spec_from_tool

TICKET_NO_PATTERN = re.compile(r"^T\d{11}$")


@tool
def create_ticket(
    description: Annotated[str, Field(min_length=1, pattern=r"\S")],
    ticket_type: Literal["售后", "投诉", "咨询", "退款"],
) -> str:
    """创建工单。

    为当前会话创建一张工单:工单号按「T + 当日日期(YYYYMMDD) + 三位递增
    序号」生成;工单状态默认「待处理」。只写 tickets 表,不改会话状态;
    转人工由前端独立模拟。

    Args:
        description: 工单描述(问题详情),不能为空或只包含空白字符
        ticket_type: 工单类型,取值「售后」「投诉」「咨询」「退款」之一
    """
    # Defense in depth: the JSON Schema rejects blank input before the tool
    # runs; this guard keeps a direct in-process call from writing an empty
    # ticket description.
    if not description.strip():
        return "创建工单失败:工单描述不能为空。"

    cid = current_conversation_id.get()
    if cid is None:
        return "创建工单失败:缺少会话上下文。"

    settings = get_settings()
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
            try:
                s.commit()
                break
            except IntegrityError:
                s.rollback()
        else:
            return "创建工单失败:工单号生成冲突,请重试。"
    finally:
        s.close()

    # Demo only: pause after the commit so the front end can show progress.
    if settings.demo_ticket_delay_seconds > 0:
        time.sleep(settings.demo_ticket_delay_seconds)

    return ticket_no


SPECS = [
    spec_from_tool(
        create_ticket,
        permission="write",
        inject_conversation=True,
        label="创建工单",
    )
]
