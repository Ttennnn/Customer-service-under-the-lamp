from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class AppealType(str, Enum):
    REFUND = "退款"
    RETURN_REFUND = "退货退款"
    EXCHANGE = "换货"
    REFUND_ONLY = "仅退款"
    RESHIPMENT = "补发"
    REPAIR = "维修"
    LOGISTICS = "物流催单"
    COMPLAINT = "投诉"
    OTHER = "其他"


class AfterSaleExtraction(BaseModel):
    order_id: str | None = Field(default=None, description="订单号")
    appeal_type: AppealType = Field(description="诉求类型")
    expected_solution: str | None = Field(default=None, description="期望解决方案")


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, description="会话 ID")
    message: str = Field(min_length=1, description="用户消息")
    conversation_id: int | None = Field(
        default=None, gt=0, description="目标会话 ID,省略时用最近一个进行中的会话"
    )
    new_conversation: bool = Field(
        default=False, description="强制新建会话"
    )

    @field_validator("session_id", "message", mode="before")
    @classmethod
    def _strip_not_blank(cls, v: str) -> str:
        if isinstance(v, str):
            v = v.strip()
            if not v:
                raise ValueError("must not be blank")
        return v


class TicketRequest(BaseModel):
    session_id: str = Field(min_length=1, description="会话 ID")
    description: str = Field(min_length=1, description="工单描述")
    ticket_type: Literal["售后", "投诉", "咨询", "退款"] = "咨询"
    conversation_id: int | None = Field(
        default=None, gt=0, description="工单归属会话 ID,省略时用最近一个进行中的会话"
    )

    @field_validator("session_id", "description", mode="before")
    @classmethod
    def _strip_not_blank(cls, v: str) -> str:
        if isinstance(v, str):
            v = v.strip()
            if not v:
                raise ValueError("must not be blank")
        return v


class ExtractRequest(BaseModel):
    text: str = Field(min_length=1, description="待抽取的文本")

    @field_validator("text", mode="before")
    @classmethod
    def _strip_not_blank(cls, v: str) -> str:
        if isinstance(v, str):
            v = v.strip()
            if not v:
                raise ValueError("must not be blank")
        return v


# 退款原因固定类目（前后端同一份），由前端下拉选择，后端校验。
REFUND_REASONS: tuple[str, ...] = (
    "七天无理由",
    "质量问题",
    "发错货",
    "不想要了",
    "其他",
)


class ResumeRequest(BaseModel):
    session_id: str = Field(min_length=1, description="会话 ID")
    order_id: str | None = Field(
        default=None, description="用户选择的订单号(订单选择器)"
    )
    confirmed: bool | None = Field(
        default=None, description="建工单确认结果(工单确认框)"
    )
    conversation_id: int | None = Field(
        default=None, gt=0, description="待恢复会话 ID,省略时用最近一个进行中的会话"
    )

    @field_validator("session_id", "order_id", mode="before")
    @classmethod
    def _strip_not_blank(cls, v: str) -> str:
        if isinstance(v, str):
            v = v.strip()
            if not v:
                raise ValueError("must not be blank")
        return v

    @model_validator(mode="after")
    def _exactly_one_resume_value(self) -> "ResumeRequest":
        if (self.order_id is None) == (self.confirmed is None):
            raise ValueError("order_id 与 confirmed 必须且只能提供一个")
        return self


class RefundRequest(BaseModel):
    session_id: str = Field(min_length=1, description="会话 ID")
    order_id: str = Field(min_length=1, description="退款订单号")
    reason: Literal[*REFUND_REASONS] = Field(description="退款原因")
    conversation_id: int | None = Field(
        default=None, gt=0, description="退款归属会话 ID,省略时用最近一个进行中的会话"
    )

    @field_validator("session_id", "order_id", mode="before")
    @classmethod
    def _strip_not_blank(cls, v: str) -> str:
        if isinstance(v, str):
            v = v.strip()
            if not v:
                raise ValueError("must not be blank")
        return v
