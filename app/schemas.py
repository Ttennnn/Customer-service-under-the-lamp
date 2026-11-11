from enum import Enum

from pydantic import BaseModel, Field, field_validator


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

    @field_validator("session_id", "message", mode="before")
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
