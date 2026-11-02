from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class EvidenceStatus(StrEnum):
    SUFFICIENT = "SUFFICIENT"
    PARTIAL = "PARTIAL"
    INSUFFICIENT = "INSUFFICIENT"
    CONFLICTING = "CONFLICTING"


class EvidenceEvaluation(BaseModel):
    status: EvidenceStatus
    supported_aspects: list[str] = Field(default_factory=list)
    missing_aspects: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    reason: str = Field(default="", max_length=800)


class GroundingStatus(StrEnum):
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    CONTRADICTED = "CONTRADICTED"


class GroundingDecision(BaseModel):
    status: GroundingStatus
    supported_source_ids: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    contradicted_claims: list[str] = Field(default_factory=list)
    reason: str = Field(default="", max_length=1000)
