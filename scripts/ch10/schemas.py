"""Structured-output schemas shared by the ch10 LLM steps."""

from __future__ import annotations

from pydantic import BaseModel, field_validator

from app.core.taxonomy import normalize_labels


class PrelabelItem(BaseModel):
    id: str
    labels: list[str]

    @field_validator("labels")
    @classmethod
    def _normalize(cls, value: list[str]) -> list[str]:
        return normalize_labels(value)


class PrelabelBatch(BaseModel):
    items: list[PrelabelItem]


class CleanItem(BaseModel):
    id: str
    text: str


class CleanBatch(BaseModel):
    items: list[CleanItem]


class SyntheticItem(BaseModel):
    text: str
    labels: list[str]

    @field_validator("labels")
    @classmethod
    def _normalize(cls, value: list[str]) -> list[str]:
        return normalize_labels(value)


class SyntheticBatch(BaseModel):
    items: list[SyntheticItem]


class AugmentItem(BaseModel):
    id: str
    text: str


class AugmentBatch(BaseModel):
    items: list[AugmentItem]
