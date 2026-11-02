from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


class RouteName(StrEnum):
    GENERAL_CONVERSATION = "GENERAL_CONVERSATION"
    BUSINESS_KNOWLEDGE = "BUSINESS_KNOWLEDGE"
    MIXED = "MIXED"
    CLARIFICATION_REQUIRED = "CLARIFICATION_REQUIRED"


class RouteDecision(BaseModel):
    route: RouteName
    confidence: float = Field(ge=0.0, le=1.0)
    requires_rag: bool
    requires_grounding: bool
    reason: str = Field(min_length=1, max_length=500)
    retrieval_query: str | None = Field(default=None, max_length=1500)
    clarifying_question: str | None = Field(default=None, max_length=800)
    user_language: str = Field(default="English", max_length=60)

    @field_validator("retrieval_query", "clarifying_question")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        return value.strip() if value and value.strip() else None

    def enforce_invariants(self) -> "RouteDecision":
        expected_rag = self.route in {RouteName.BUSINESS_KNOWLEDGE, RouteName.MIXED}
        self.requires_rag = expected_rag
        self.requires_grounding = expected_rag
        if expected_rag and not self.retrieval_query:
            self.retrieval_query = self.reason
        if self.route == RouteName.CLARIFICATION_REQUIRED and not self.clarifying_question:
            self.clarifying_question = "Which service, product, or topic are you referring to?"
        if self.route != RouteName.CLARIFICATION_REQUIRED:
            self.clarifying_question = None
        return self
