from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any

from app.config import Settings
from app.generation.answer_generator import AnswerGenerator
from app.memory.conversation_store import ConversationStore
from app.models import Citation
from app.rag.retriever import RAGRetriever
from app.router.intelligence_router import IntelligenceRouter
from app.router.schemas import RouteName
from app.verification.evidence import EvidenceEvaluator
from app.verification.grounding import GroundingVerifier
from app.verification.schemas import EvidenceStatus, GroundingStatus

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(
        self,
        settings: Settings,
        store: ConversationStore,
        router: IntelligenceRouter,
        retriever: RAGRetriever,
        evaluator: EvidenceEvaluator,
        generator: AnswerGenerator,
        verifier: GroundingVerifier,
    ):
        self.settings = settings
        self.store = store
        self.router = router
        self.retriever = retriever
        self.evaluator = evaluator
        self.generator = generator
        self.verifier = verifier

    @staticmethod
    def _citations(chunks: list[dict], allowed: set[str] | None = None) -> list[Citation]:
        citations = []
        for chunk in chunks:
            source_id = chunk.get("source_id")
            if allowed is not None and source_id not in allowed:
                continue
            metadata = chunk.get("metadata", {})
            page = metadata.get("page_number", -1)
            citations.append(
                Citation(
                    source_id=source_id,
                    filename=str(metadata.get("source_filename", "Unknown source")),
                    page=page if isinstance(page, int) and page > 0 else None,
                    title=metadata.get("document_title"),
                    excerpt=chunk.get("text", "")[:360],
                )
            )
        return citations

    @staticmethod
    def _safe_evidence_fallback(
        question: str, chunks: list[dict], conflict: bool = False
    ) -> tuple[str, list[Citation]]:
        if not chunks:
            return (
                "I don’t have relevant business information indexed for that yet. Add the relevant document to the `knowledge/` folder, then reindex the knowledge base.",
                [],
            )
        selected = chunks[:2]
        parts = ["Here’s what I can confirm from the available material:"]
        for chunk in selected:
            source_id = chunk.get("source_id", "")
            text = " ".join(chunk.get("text", "").split())
            if len(text) > 450:
                text = text[:447].rstrip() + "..."
            parts.append(f"{text} [{source_id}]")
        if conflict:
            parts.append(
                "The retrieved material appears to contain conflicting details, so I can’t confidently choose between them."
            )
        else:
            parts.append("The documents may not cover every part of your question.")
        return "\n\n".join(parts), ChatService._citations(selected)

    async def handle(self, message: str, session_id: str | None, request_id: str) -> dict[str, Any]:
        pipeline_started = time.perf_counter()
        if not message.strip():
            raise ValueError("Message cannot be empty")
        if session_id:
            if not self.store.session_exists(session_id):
                raise KeyError("Conversation not found")
        else:
            session_id = self.store.create_session()

        history = self.store.get_history(session_id, self.settings.max_history_messages)
        self.store.save_message(session_id, "user", message)
        timings: dict[str, float] = {}

        started = time.perf_counter()
        decision = await self.router.route(message, history, session_context={"session_id": session_id})
        timings["router_ms"] = (time.perf_counter() - started) * 1000

        citations: list[Citation] = []
        evidence_status: str | None = None

        if decision.route == RouteName.CLARIFICATION_REQUIRED:
            answer = decision.clarifying_question or "What are you referring to?"

        elif decision.route == RouteName.GENERAL_CONVERSATION:
            # Architectural invariant: this branch never calls the retriever or embeddings.
            started = time.perf_counter()
            answer = await self.generator.generate_general(message, history)
            timings["answer_ms"] = (time.perf_counter() - started) * 1000

        else:
            started = time.perf_counter()
            retrieval = await asyncio.to_thread(self.retriever.retrieve, decision.retrieval_query or message)
            timings["retrieval_ms"] = (time.perf_counter() - started) * 1000
            timings["embedding_ms"] = float(getattr(retrieval, "embedding_ms", 0.0))
            timings["vector_search_ms"] = float(getattr(retrieval, "vector_search_ms", 0.0))

            if not retrieval.chunks and decision.route == RouteName.BUSINESS_KNOWLEDGE:
                answer = (
                    "I don’t have relevant business information indexed for that yet. "
                    "Add the relevant file to the `knowledge/` folder and trigger a reindex from the Knowledge panel."
                )
                evidence_status = EvidenceStatus.INSUFFICIENT.value
            elif not retrieval.chunks and decision.route == RouteName.MIXED:
                evidence_status = EvidenceStatus.INSUFFICIENT.value
                started = time.perf_counter()
                answer = await self.generator.generate_general(
                    message + "\n\nPlease answer only the general or educational part of this request. "
                    "Do not guess any business-specific fact; explain naturally that the business-specific part "
                    "cannot be confirmed from the currently indexed knowledge.",
                    history,
                )
                timings["answer_ms"] = (time.perf_counter() - started) * 1000
            else:
                started = time.perf_counter()
                evaluation = await self.evaluator.evaluate(
                    message, retrieval.context, retrieval.chunks, history
                )
                timings["evidence_evaluation_ms"] = (time.perf_counter() - started) * 1000
                evidence_status = evaluation.status.value

                started = time.perf_counter()
                answer = await self.generator.generate_with_evidence(
                    message, history, retrieval.context, evaluation.status.value
                )
                timings["answer_ms"] = (time.perf_counter() - started) * 1000

                started = time.perf_counter()
                valid_ids = {chunk["source_id"] for chunk in retrieval.chunks}
                verification = await self.verifier.verify(
                    message, answer, retrieval.context, valid_ids, history
                )
                timings["grounding_ms"] = (time.perf_counter() - started) * 1000

                if (
                    verification.status in {GroundingStatus.UNSUPPORTED, GroundingStatus.CONTRADICTED}
                    or verification.unsupported_claims
                    or verification.contradicted_claims
                    or not verification.supported_source_ids
                ):
                    started = time.perf_counter()
                    try:
                        corrected = await self.generator.correct_with_evidence(
                            message,
                            answer,
                            verification.unsupported_claims,
                            verification.contradicted_claims,
                            retrieval.context,
                            history,
                        )
                        timings["correction_ms"] = (time.perf_counter() - started) * 1000
                        started = time.perf_counter()
                        second = await self.verifier.verify(
                            message, corrected, retrieval.context, valid_ids, history
                        )
                        timings["grounding_retry_ms"] = (time.perf_counter() - started) * 1000
                    except Exception as exc:
                        logger.warning("Grounding correction unavailable error_type=%s", type(exc).__name__)
                        second = None
                    if (
                        second
                        and second.status == GroundingStatus.SUPPORTED
                        and second.supported_source_ids
                        and not second.unsupported_claims
                        and not second.contradicted_claims
                    ):
                        answer = corrected
                        verification = second
                    elif (
                        second
                        and second.status == GroundingStatus.PARTIALLY_SUPPORTED
                        and second.supported_source_ids
                        and not second.unsupported_claims
                        and not second.contradicted_claims
                    ):
                        answer = corrected
                        verification = second
                    else:
                        answer, citations = self._safe_evidence_fallback(
                            message,
                            retrieval.chunks,
                            conflict=(evaluation.status == EvidenceStatus.CONFLICTING),
                        )
                if not citations:
                    cited_ids = set(re.findall(r"\[(S\d+)\]", answer))
                    allowed_ids = set(verification.supported_source_ids) & valid_ids
                    if cited_ids:
                        allowed_ids &= cited_ids
                    citations = self._citations(retrieval.chunks, allowed=allowed_ids)
                    if not citations and retrieval.chunks:
                        answer, citations = self._safe_evidence_fallback(
                            message,
                            retrieval.chunks,
                            conflict=(evaluation.status == EvidenceStatus.CONFLICTING),
                        )

        self.store.save_message(
            session_id,
            "assistant",
            answer,
            citations=[citation.model_dump() for citation in citations],
        )
        timings["total_ms"] = (time.perf_counter() - pipeline_started) * 1000
        logger.info(
            "chat_pipeline_complete route=%s evidence=%s timings_ms=%s",
            decision.route.value,
            evidence_status or "n/a",
            {key: round(value, 1) for key, value in timings.items()},
            extra={"request_id": request_id},
        )
        return {
            "session_id": session_id,
            "request_id": request_id,
            "answer": answer,
            "citations": citations,
            "route": decision.route.value,
            "evidence_status": evidence_status,
        }
