from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Citation(BaseModel):
    source_id: str
    filename: str
    page: int | None = None
    title: str | None = None
    excerpt: str | None = None


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=12000)
    session_id: str | None = Field(default=None, max_length=64)


class ChatResponse(BaseModel):
    session_id: str
    request_id: str
    answer: str
    citations: list[Citation] = []
    route: Literal["GENERAL_CONVERSATION", "BUSINESS_KNOWLEDGE", "MIXED", "CLARIFICATION_REQUIRED"]
    evidence_status: str | None = None


class ConversationSummary(BaseModel):
    session_id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int


class MessageOut(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    created_at: str
    citations: list[Citation] = []


class ConversationOut(BaseModel):
    session_id: str
    title: str
    messages: list[MessageOut]


class ReindexRequest(BaseModel):
    full: bool = False


class KnowledgeStatus(BaseModel):
    ready: bool
    directory: str
    empty: bool
    document_count: int
    chunk_count: int
    indexed_files: int
    errors: list[dict[str, str]] = []
    message: str
