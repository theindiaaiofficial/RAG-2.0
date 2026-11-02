from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.config import Settings
from app.llm import LLMClient, LLMConfigurationError, LLMProviderError
from app.json_utils import extract_json_object
from app.router.prompts import ROUTER_SYSTEM_PROMPT
from app.router.schemas import RouteDecision, RouteName

logger = logging.getLogger(__name__)


def _looks_business_related(text: str) -> bool:
    cues = (
        "your company",
        "your business",
        "your services",
        "your price",
        "your pricing",
        "services do you offer",
        "what services",
        "refund",
        "pricing",
        "price",
        "business hours",
        "who founded",
        "founder",
        "contact",
        "do you offer",
        "company policy",
        "company's",
        "aapki company",
        "aapki services",
        "kitna charge",
        "price kya",
        "सेवाएं",
        "कीमत",
        "रिफंड नीति",
    )
    lowered = text.casefold()
    return any(cue in lowered for cue in cues)


def deterministic_route(message: str, history: list[dict[str, str]]) -> RouteDecision | None:
    """Narrow fast path for obvious general requests and unresolved references."""
    text = message.strip().lower()
    generic_reference = bool(
        re.match(
            r"^(how much|what about (it|that|them|him|her)|can you explain that|explain (it|that|this)|and the price|what does it cost)\b",
            text,
        )
    )
    if generic_reference and not history:
        return RouteDecision(
            route=RouteName.CLARIFICATION_REQUIRED,
            confidence=0.98,
            requires_rag=False,
            requires_grounding=False,
            reason="The referenced subject is not present in the conversation.",
            clarifying_question="Which product, service, or topic are you asking about?",
        )
    if generic_reference and history:
        return None  # Let LLM 1 resolve the reference from the preceding turns.

    # A request that asks for an explanation AND checks whether a business offers
    # the explained item must use both general generation and business retrieval.
    has_general_part = bool(
        re.search(
            r"\b(explain|define|describe|what is|what are|what's|how does|meaning of|"
            r"kya hota hai|kya hai|samjhao|samjhaiye)\b",
            text,
        )
    )
    has_business_availability_check = bool(
        re.search(
            r"\b(whether|if)\b[\s\S]{0,180}\b"
            r"(offer(?:s)?|provide(?:s)?|support(?:s)?|sell(?:s)?|have|has)\b",
            text,
        )
        or re.search(
            r"\b(company|business|provider|organization|organisation)\b"
            r"[\s\S]{0,120}\b"
            r"(offer(?:s)?|provide(?:s)?|support(?:s)?|sell(?:s)?|have|has|deti|deta|karti|karta)\b",
            text,
        )
    )
    if has_general_part and has_business_availability_check:
        return RouteDecision(
            route=RouteName.MIXED,
            confidence=0.98,
            requires_rag=True,
            requires_grounding=True,
            reason="The request combines a general explanation with a business availability check.",
            retrieval_query=message,
        ).enforce_invariants()

    if re.fullmatch(
        r"(?:hi|hello|hey|hii+|good morning|good afternoon|good evening|namaste|नमस्ते)[!.\s]*", text
    ):
        return RouteDecision(
            route=RouteName.GENERAL_CONVERSATION,
            confidence=0.99,
            requires_rag=False,
            requires_grounding=False,
            reason="A standalone greeting does not require business knowledge.",
            user_language="Hindi" if "नमस्ते" in text else "English",
        )
    if not _looks_business_related(text) and re.match(
        r"^(explain|define|what is|what are|what's|what are the|difference between|compare|"
        r"help me write|write (?:an?|the) |tell me a joke|translate|how does .+ work)\b",
        text,
    ):
        return RouteDecision(
            route=RouteName.GENERAL_CONVERSATION,
            confidence=0.96,
            requires_rag=False,
            requires_grounding=False,
            reason="The request is a recognizable general-knowledge or writing question.",
        )
    return None


class IntelligenceRouter:
    def __init__(self, llm: LLMClient, settings: Settings):
        self.llm = llm
        self.settings = settings

    async def route(
        self,
        message: str,
        history: list[dict[str, str]],
        session_context: dict[str, Any] | None = None,
        business_context: dict[str, Any] | None = None,
    ) -> RouteDecision:
        fast = deterministic_route(message, history)
        if fast:
            return fast.enforce_invariants()

        prompt_context = {
            "current_message": message,
            "recent_conversation": history[-self.settings.max_history_messages :],
            "session_context": session_context or {},
            "business_context_metadata": business_context or {},
        }
        try:
            raw = await self.llm.complete(
                "router",
                [
                    {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(prompt_context, ensure_ascii=False)},
                ],
                temperature=0,
                response_format={"type": "json_object"},
                max_tokens=450,
            )
            parsed = RouteDecision.model_validate(extract_json_object(raw)).enforce_invariants()
            if parsed.confidence < self.settings.router_confidence_threshold:
                # Low-confidence business-adjacent intent takes the retrieval path;
                # a truly unresolved reference remains a clarification.
                if parsed.route == RouteName.CLARIFICATION_REQUIRED:
                    return parsed
                if parsed.route == RouteName.GENERAL_CONVERSATION:
                    if _looks_business_related(message):
                        parsed.route = RouteName.BUSINESS_KNOWLEDGE
                        parsed.retrieval_query = parsed.retrieval_query or message
                        return parsed.enforce_invariants()
                    return RouteDecision(
                        route=RouteName.CLARIFICATION_REQUIRED,
                        confidence=parsed.confidence,
                        requires_rag=False,
                        requires_grounding=False,
                        reason="The intent classifier was uncertain.",
                        clarifying_question="Are you asking about this business, or would you like a general explanation?",
                        user_language=parsed.user_language,
                    )
                if parsed.route != RouteName.MIXED:
                    parsed.route = RouteName.BUSINESS_KNOWLEDGE
                parsed.enforce_invariants()
            return parsed
        except (LLMConfigurationError, LLMProviderError, ValueError, TypeError) as exc:
            logger.warning("Router fallback activated error_type=%s", type(exc).__name__)
            # Conservative: preserve resolvable business follow-ups even if LLM 1 is unavailable.
            text = message.casefold()
            generic_reference = bool(
                re.match(
                    r"^(how much|what about (it|that|them|him|her)|can you explain that|and the price|what does it cost)\b",
                    text,
                )
            )
            recent_text = " ".join(item.get("content", "") for item in history[-6:]).casefold()
            business_history_cues = (
                "your company",
                "your business",
                "your services",
                "services do you offer",
                "do you offer",
                "our company",
                "our services",
                "we offer",
                "business hours",
                "refund policy",
                "founder of",
                "company founder",
                "company pricing",
                "our pricing",
                "aapki company",
                "aapki services",
            )
            if generic_reference and any(cue in recent_text for cue in business_history_cues):
                context = " ".join(item.get("content", "") for item in history[-4:])[:900]
                return RouteDecision(
                    route=RouteName.BUSINESS_KNOWLEDGE,
                    confidence=0.55,
                    requires_rag=True,
                    requires_grounding=True,
                    reason="Conservative fallback for a business-specific follow-up resolved from history.",
                    retrieval_query=f"Find the requested price, terms, or details for the business service discussed in this conversation. Context: {context}",
                ).enforce_invariants()
            if _looks_business_related(text):
                return RouteDecision(
                    route=RouteName.BUSINESS_KNOWLEDGE,
                    confidence=0.52,
                    requires_rag=True,
                    requires_grounding=True,
                    reason="Conservative fallback for a likely business-specific request.",
                    retrieval_query=message,
                ).enforce_invariants()
            if re.search(r"\b(it|that|them|him|her|same service|second option)\b", text):
                return RouteDecision(
                    route=RouteName.CLARIFICATION_REQUIRED,
                    confidence=0.4,
                    requires_rag=False,
                    requires_grounding=False,
                    reason="The router could not safely resolve the reference.",
                    clarifying_question="What are you referring to?",
                ).enforce_invariants()
            # No uncertain message is silently treated as general conversation.
            return RouteDecision(
                route=RouteName.CLARIFICATION_REQUIRED,
                confidence=0.0,
                requires_rag=False,
                requires_grounding=False,
                reason="The router is unavailable and intent could not be established safely.",
                clarifying_question="Is your question about this business, or is it a general question?",
            ).enforce_invariants()
