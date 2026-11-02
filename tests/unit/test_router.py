from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError

from app.json_utils import extract_json_object
from app.llm import LLMProviderError
from app.router.intelligence_router import IntelligenceRouter, deterministic_route
from app.router.schemas import RouteDecision, RouteName


def test_parse_valid_json_with_surrounding_text():
    result = extract_json_object('Here is the route:\n```json\n{"route":"GENERAL_CONVERSATION"}\n```')
    assert result["route"] == "GENERAL_CONVERSATION"


def test_invalid_json_rejected():
    with pytest.raises(ValueError):
        extract_json_object("not json")


def test_schema_rejects_missing_fields():
    with pytest.raises(ValidationError):
        RouteDecision.model_validate({"route": "BUSINESS_KNOWLEDGE"})


def test_schema_rejects_invalid_route():
    with pytest.raises(ValidationError):
        RouteDecision.model_validate(
            {
                "route": "OTHER",
                "confidence": 0.9,
                "requires_rag": False,
                "requires_grounding": False,
                "reason": "bad route",
            }
        )


def test_schema_rejects_confidence_out_of_range():
    with pytest.raises(ValidationError):
        RouteDecision.model_validate(
            {
                "route": "GENERAL_CONVERSATION",
                "confidence": 1.5,
                "requires_rag": False,
                "requires_grounding": False,
                "reason": "invalid confidence",
            }
        )


def test_route_invariants_override_model_flags():
    decision = RouteDecision(
        route=RouteName.BUSINESS_KNOWLEDGE,
        confidence=0.9,
        requires_rag=False,
        requires_grounding=False,
        reason="Business pricing",
        retrieval_query="Company pricing",
    ).enforce_invariants()
    assert decision.requires_rag is True
    assert decision.requires_grounding is True


def test_clarification_invariant_adds_question():
    decision = RouteDecision(
        route=RouteName.CLARIFICATION_REQUIRED,
        confidence=0.8,
        requires_rag=True,
        requires_grounding=True,
        reason="Missing context",
    ).enforce_invariants()
    assert decision.requires_rag is False
    assert decision.clarifying_question


def test_greeting_fast_path_is_general():
    decision = deterministic_route("Hello!", [])
    assert decision is not None
    assert decision.route == RouteName.GENERAL_CONVERSATION
    assert not decision.requires_rag


def test_context_free_price_question_clarifies():
    decision = deterministic_route("How much does it cost?", [])
    assert decision is not None
    assert decision.route == RouteName.CLARIFICATION_REQUIRED


class FailingLLM:
    async def complete(self, *args, **kwargs):
        raise LLMProviderError("provider unavailable")


@pytest.mark.parametrize(
    "message", ["What services do you offer?", "Tell me your pricing", "What is your refund policy?"]
)
def test_router_failure_uses_conservative_business_fallback(settings, message):
    decision = asyncio.run(IntelligenceRouter(FailingLLM(), settings).route(message, []))
    assert decision.route == RouteName.BUSINESS_KNOWLEDGE
    assert decision.requires_rag and decision.requires_grounding


def test_router_failure_does_not_default_unknown_request_to_general(settings):
    decision = asyncio.run(IntelligenceRouter(FailingLLM(), settings).route("Please tell me about that", []))
    assert decision.route == RouteName.CLARIFICATION_REQUIRED


def test_deterministic_general_explanation_bypasses_router_llm(settings):
    class ExplodingLLM:
        async def complete(self, *args, **kwargs):
            raise AssertionError("Router LLM should not run for recognized general questions")

    decision = asyncio.run(IntelligenceRouter(ExplodingLLM(), settings).route("Explain machine learning", []))
    assert decision.route == RouteName.GENERAL_CONVERSATION
    assert not decision.requires_rag


class JSONLLM:
    def __init__(self, payload):
        self.payload = payload
        self.messages = None

    async def complete(self, role, messages, **kwargs):
        import json

        self.messages = messages
        return json.dumps(self.payload)


def test_low_confidence_business_adjacent_general_is_routed_to_retrieval(settings):
    llm = JSONLLM(
        {
            "route": "GENERAL_CONVERSATION",
            "confidence": 0.41,
            "requires_rag": False,
            "requires_grounding": False,
            "reason": "uncertain",
            "retrieval_query": None,
            "clarifying_question": None,
            "user_language": "English",
        }
    )
    decision = asyncio.run(
        IntelligenceRouter(llm, settings).route("Could you provide your company pricing?", [])
    )
    assert decision.route == RouteName.BUSINESS_KNOWLEDGE
    assert decision.requires_rag


def test_low_confidence_mixed_route_is_preserved(settings):
    llm = JSONLLM(
        {
            "route": "MIXED",
            "confidence": 0.58,
            "requires_rag": True,
            "requires_grounding": True,
            "reason": "general plus company details",
            "retrieval_query": "company AI automation services",
            "clarifying_question": None,
            "user_language": "English",
        }
    )
    decision = asyncio.run(IntelligenceRouter(llm, settings).route("Explain AI and your services", []))
    assert decision.route == RouteName.MIXED
    assert decision.requires_rag


def test_router_sends_recent_history_to_model(settings):
    llm = JSONLLM(
        {
            "route": "BUSINESS_KNOWLEDGE",
            "confidence": 0.95,
            "requires_rag": True,
            "requires_grounding": True,
            "reason": "Price follow-up",
            "retrieval_query": "price of web development",
            "clarifying_question": None,
            "user_language": "English",
        }
    )
    history = [
        {"role": "user", "content": "What services do you offer?"},
        {"role": "assistant", "content": "Web design."},
    ]
    decision = asyncio.run(IntelligenceRouter(llm, settings).route("How much does it cost?", history))
    assert decision.retrieval_query == "price of web development"
    assert "What services do you offer?" in llm.messages[1]["content"]


def test_general_and_business_message_deterministic_precedence(settings):
    general = deterministic_route("What is machine learning?", [])
    business = deterministic_route("What is your refund policy?", [])
    assert general and general.route == RouteName.GENERAL_CONVERSATION
    assert business is None  # Business-specific message must reach router/fallback classification.


def test_business_followup_uses_history_even_when_router_provider_fails(settings):
    history = [
        {"role": "user", "content": "What services do you offer?"},
        {"role": "assistant", "content": "We offer web design and AI automation."},
    ]
    decision = asyncio.run(
        IntelligenceRouter(FailingLLM(), settings).route("How much does it cost?", history)
    )
    assert decision.route == RouteName.BUSINESS_KNOWLEDGE
    assert "web design" in decision.retrieval_query


def test_context_free_explain_that_question_clarifies():
    decision = deterministic_route("Can you explain that?", [])
    assert decision is not None
    assert decision.route == RouteName.CLARIFICATION_REQUIRED
