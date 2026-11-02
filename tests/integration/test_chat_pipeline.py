from __future__ import annotations

import asyncio

from app.memory.conversation_store import ConversationStore
from app.router.intelligence_router import deterministic_route
from app.router.schemas import RouteDecision, RouteName
from app.services.chat_service import ChatService
from app.verification.schemas import (
    EvidenceEvaluation,
    EvidenceStatus,
    GroundingDecision,
    GroundingStatus,
)


class RouterStub:
    def __init__(self, route):
        self.route_value = route
        self.seen_history = None

    async def route(self, message, history, **kwargs):
        self.seen_history = history
        return self.route_value


class RetrieverSpy:
    def __init__(self, chunks=None):
        self.calls = 0
        self.chunks = chunks or []

    def retrieve(self, query):
        self.calls += 1
        for index, chunk in enumerate(self.chunks, 1):
            chunk["source_id"] = f"S{index}"
        return type(
            "Retrieval",
            (),
            {
                "chunks": self.chunks,
                "context": "\n".join(c["text"] for c in self.chunks),
            },
        )()


class EvaluatorStub:
    def __init__(self):
        self.calls = 0

    async def evaluate(self, *args):
        self.calls += 1
        return EvidenceEvaluation(status=EvidenceStatus.SUFFICIENT)


class GeneratorStub:
    def __init__(self):
        self.general_calls = 0
        self.evidence_calls = 0
        self.correction_calls = 0

    async def generate_general(self, message, history):
        self.general_calls += 1
        return "A direct general answer."

    async def generate_with_evidence(self, *args):
        self.evidence_calls += 1
        return "North Star offers web design. [S1]"

    async def correct_with_evidence(self, *args):
        self.correction_calls += 1
        return "North Star offers web design. [S1]"


class VerifierStub:
    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.calls = 0

    async def verify(self, *args):
        self.calls += 1
        return self.decisions.pop(0)


def route(name):
    return RouteDecision(
        route=name,
        confidence=0.99,
        requires_rag=name in {RouteName.BUSINESS_KNOWLEDGE, RouteName.MIXED},
        requires_grounding=name in {RouteName.BUSINESS_KNOWLEDGE, RouteName.MIXED},
        reason="test route",
        retrieval_query=(
            "North Star services"
            if name in {RouteName.BUSINESS_KNOWLEDGE, RouteName.MIXED}
            else None
        ),
    ).enforce_invariants()


def make_service(tmp_path, settings, route_value, chunks=None, verifier_decisions=None):
    store = ConversationStore(tmp_path / "conversations.db")
    router = RouterStub(route_value)
    retriever = RetrieverSpy(chunks)
    evaluator = EvaluatorStub()
    generator = GeneratorStub()
    verifier = VerifierStub(
        verifier_decisions
        or [
            GroundingDecision(
                status=GroundingStatus.SUPPORTED, supported_source_ids=["S1"]
            )
        ]
    )
    service = ChatService(
        settings, store, router, retriever, evaluator, generator, verifier
    )
    return service, store, router, retriever, evaluator, generator, verifier


def test_general_route_bypasses_retriever_completely(tmp_path, settings):
    service, store, _, retriever, evaluator, generator, verifier = make_service(
        tmp_path, settings, route(RouteName.GENERAL_CONVERSATION)
    )
    result = asyncio.run(service.handle("Explain photosynthesis", None, "req-1"))
    assert result["route"] == "GENERAL_CONVERSATION"
    assert generator.general_calls == 1
    assert retriever.calls == 0 and evaluator.calls == 0 and verifier.calls == 0
    assert (
        store.get_conversation(result["session_id"])["messages"][-1]["content"]
        == "A direct general answer."
    )


def test_business_route_retrieves_and_verifies_with_real_source_id(tmp_path, settings):
    chunks = [
        {
            "text": "North Star offers web design.",
            "metadata": {
                "source_filename": "services.md",
                "page_number": -1,
                "document_title": "Services",
            },
            "similarity": 0.9,
        }
    ]
    service, _, _, retriever, evaluator, _, verifier = make_service(
        tmp_path, settings, route(RouteName.BUSINESS_KNOWLEDGE), chunks
    )
    result = asyncio.run(
        service.handle("What services does the business offer?", None, "req-2")
    )
    assert result["route"] == "BUSINESS_KNOWLEDGE"
    assert retriever.calls == evaluator.calls == verifier.calls == 1
    assert result["citations"][0].filename == "services.md"
    assert result["answer"].endswith("[S1]")


def test_mixed_route_uses_retrieval_and_answer_model(tmp_path, settings):
    chunks = [
        {
            "text": "North Star offers website development.",
            "metadata": {
                "source_filename": "services.md",
                "page_number": -1,
                "document_title": "Services",
            },
            "similarity": 0.9,
        }
    ]
    service, *_ = make_service(tmp_path, settings, route(RouteName.MIXED), chunks)
    result = asyncio.run(
        service.handle(
            "Explain website development and whether the business offers it.",
            None,
            "req-3",
        )
    )
    assert result["route"] == "MIXED"
    assert result["evidence_status"] == "SUFFICIENT"
    assert result["citations"]


def test_followup_history_is_passed_to_router(tmp_path, settings):
    service, store, router, *_ = make_service(
        tmp_path, settings, route(RouteName.GENERAL_CONVERSATION)
    )
    session = store.create_session()
    store.save_message(session, "user", "What services do you offer?")
    store.save_message(session, "assistant", "We offer web design.")
    asyncio.run(service.handle("How much does it cost?", session, "req-4"))
    assert [item["role"] for item in router.seen_history] == ["user", "assistant"]


def test_one_correction_retry_then_keep_verified_answer(tmp_path, settings):
    chunks = [
        {
            "text": "North Star offers web design.",
            "metadata": {
                "source_filename": "services.md",
                "page_number": -1,
                "document_title": "Services",
            },
            "similarity": 0.9,
        }
    ]
    decisions = [
        GroundingDecision(
            status=GroundingStatus.UNSUPPORTED,
            unsupported_claims=["An unsupported price claim."],
        ),
        GroundingDecision(
            status=GroundingStatus.SUPPORTED, supported_source_ids=["S1"]
        ),
    ]
    service, _, _, _, _, generator, verifier = make_service(
        tmp_path, settings, route(RouteName.BUSINESS_KNOWLEDGE), chunks, decisions
    )
    result = asyncio.run(service.handle("What services?", None, "req-5"))
    assert generator.correction_calls == 1
    assert verifier.calls == 2
    assert result["citations"]


def test_mixed_request_without_knowledge_still_answers_general_part(tmp_path, settings):
    service, _, _, _, evaluator, generator, verifier = make_service(
        tmp_path, settings, route(RouteName.MIXED), chunks=[]
    )
    result = asyncio.run(
        service.handle(
            "Explain AI automation and whether the business offers it.", None, "req-6"
        )
    )
    assert result["route"] == "MIXED"
    assert result["evidence_status"] == "INSUFFICIENT"
    assert generator.general_calls == 1
    assert evaluator.calls == 0 and verifier.calls == 0
    assert result["citations"] == []


def test_deterministic_router_classifies_explanation_plus_business_offer_check():
    message = "Explain AI automation and tell me whether Northwind Workshop offers it."

    decision = deterministic_route(message, [])

    assert decision is not None
    assert decision.route == RouteName.MIXED
    assert decision.requires_rag is True
    assert decision.requires_grounding is True
    assert decision.retrieval_query == message


def test_business_request_without_knowledge_does_not_invent_details(tmp_path, settings):
    service, _, _, _, _, generator, _ = make_service(
        tmp_path, settings, route(RouteName.BUSINESS_KNOWLEDGE), chunks=[]
    )
    result = asyncio.run(service.handle("What are your prices?", None, "req-7"))
    assert result["evidence_status"] == "INSUFFICIENT"
    assert "don’t have relevant business information indexed" in result["answer"]
    assert generator.evidence_calls == 0 and generator.general_calls == 0
