from __future__ import annotations

import asyncio
import json
import logging
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    from app.config import get_settings
    from report_utils import update_report_section
    from app.generation.answer_generator import AnswerGenerator
    from app.knowledge.ingestion import KnowledgeIngestor
    from app.llm import LLMClient
    from app.memory.conversation_store import ConversationStore
    from app.rag.embeddings import RemoteEmbeddingModel
    from app.rag.retriever import RAGRetriever
    from app.rag.vector_store import ChromaVectorStore
    from app.router.intelligence_router import IntelligenceRouter
    from app.services.chat_service import ChatService
    from app.verification.evidence import EvidenceEvaluator
    from app.verification.grounding import GroundingVerifier

    settings = get_settings()
    print("LIVE LLM VERIFICATION REPORT")
    import importlib.util

    missing_packages = []
    for module, name in (
        ("litellm", "LiteLLM"),
        ("chromadb", "ChromaDB"),
        ("google.genai", "Google GenAI SDK"),
    ):
        try:
            installed = importlib.util.find_spec(module) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            installed = False
        if not installed:
            missing_packages.append(name)
    issues = settings.readiness_issues()
    if missing_packages or issues:
        reasons = []
        if missing_packages:
            reasons.append("Missing runtime packages: " + ", ".join(missing_packages))
        if issues:
            reasons.append("Missing provider configuration: " + "; ".join(issues))
        detail = "; ".join(reasons)
        print("[SKIP] Live verification blocked: " + detail)
        update_report_section(
            "Live LLM verification",
            "**Status:** SKIPPED / BLOCKED\n\n"
            + detail
            + "\n\nNo provider or end-to-end model tests were run.",
        )
        print(
            "Install the pinned dependencies and configure both model roles in .env, then rerun this command."
        )
        return 2

    async def verify() -> int:
        llm = LLMClient(settings)
        outcomes: list[tuple[str, bool, str]] = []
        for role in ("router", "answer"):
            ok, detail = await llm.check_connection(role)
            outcomes.append((f"{role.title()} model connection", ok, detail))
            print(
                f"[{'PASS' if ok else 'FAIL'}] {role.title()} model connection: {detail}"
            )
        if not all(item[1] for item in outcomes):
            return 1

        with tempfile.TemporaryDirectory(
            prefix="orbit-live-verify-",
            ignore_cleanup_errors=True,
        ) as temp:
            root = Path(temp)
            knowledge = root / "knowledge"
            knowledge.mkdir()
            (knowledge / "company.txt").write_text(
                "Northwind Workshop is a fictional test company. It offers website development and AI workflow automation. "
                "Website projects start at 1200 USD. Support hours are Monday to Friday, 9 AM to 5 PM. "
                "The founder is Asha Rao. The company does not offer mobile app maintenance.\n",
                encoding="utf-8",
            )
            from app.config import Settings

            live_settings = Settings(
                router_provider=settings.router_provider,
                router_model=settings.router_model,
                router_api_key=settings.router_api_key,
                router_api_base=settings.router_api_base,
                router_litellm_params=settings.router_litellm_params,
                answer_provider=settings.answer_provider,
                answer_model=settings.answer_model,
                answer_api_key=settings.answer_api_key,
                answer_api_base=settings.answer_api_base,
                answer_litellm_params=settings.answer_litellm_params,
                llm_timeout_seconds=settings.llm_timeout_seconds,
                llm_drop_unsupported_params=settings.llm_drop_unsupported_params,
                app_host=settings.app_host,
                app_port=settings.app_port,
                max_history_messages=settings.max_history_messages,
                rag_top_k=settings.rag_top_k,
                rag_chunk_size=settings.rag_chunk_size,
                rag_chunk_overlap=settings.rag_chunk_overlap,
                rag_min_similarity=settings.rag_min_similarity,
                rag_max_context_chars=settings.rag_max_context_chars,
                embedding_api_key=settings.embedding_api_key,
                embedding_model=settings.embedding_model,
                embedding_output_dimensionality=settings.embedding_output_dimensionality,
                embedding_batch_size=settings.embedding_batch_size,
                knowledge_dir=knowledge,
                data_dir=root / "data",
                chroma_dir=root / "data" / "chroma",
                sqlite_path=root / "data" / "assistant.db",
                router_confidence_threshold=settings.router_confidence_threshold,
                log_level="WARNING",
            )
            store = ConversationStore(live_settings.sqlite_path)
            vector = ChromaVectorStore(
                live_settings.chroma_dir, collection_name="live_verification_only"
            )
            vector.initialize()
            embedding_api_key = (
                live_settings.embedding_api_key.get_secret_value()
                if live_settings.embedding_api_key
                else None
            )
            embedding = RemoteEmbeddingModel(
                live_settings.embedding_model,
                api_key=embedding_api_key,
                output_dimensionality=live_settings.embedding_output_dimensionality,
                batch_size=live_settings.embedding_batch_size,
            )
            embedding.initialize()
            ingestor = KnowledgeIngestor(live_settings, embedding, vector, store)
            print(
                "[RUN] Calling remote Gemini embeddings and indexing isolated test knowledge…"
            )
            start = time.perf_counter()
            stats = await asyncio.to_thread(ingestor.scan_and_ingest, True)
            outcomes.append(
                (
                    "Isolated test knowledge ingestion",
                    stats["chunk_count"] > 0,
                    json.dumps(stats),
                )
            )
            print(
                f"[{'PASS' if stats['chunk_count'] > 0 else 'FAIL'}] Isolated test knowledge ingestion: {stats['chunk_count']} chunks in {(time.perf_counter() - start) * 1000:.0f} ms"
            )

            router = IntelligenceRouter(llm, live_settings)
            retriever = RAGRetriever(live_settings, embedding, vector)
            evaluator = EvidenceEvaluator(llm)
            generator = AnswerGenerator(llm, live_settings)
            verifier = GroundingVerifier(llm)
            service = ChatService(
                live_settings, store, router, retriever, evaluator, generator, verifier
            )
            scenarios = [
                (
                    "General conversation",
                    "Explain the idea of a feedback loop in simple terms.",
                    "GENERAL_CONVERSATION",
                ),
                (
                    "Business knowledge",
                    "What services does Northwind Workshop offer?",
                    "BUSINESS_KNOWLEDGE",
                ),
                (
                    "Mixed request",
                    "Explain AI automation and tell me whether Northwind Workshop offers it.",
                    "MIXED",
                ),
                ("Clarification", "How much does it cost?", "CLARIFICATION_REQUIRED"),
                (
                    "Hindi response",
                    "मशीन लर्निंग क्या है? सरल हिंदी में समझाइए।",
                    "GENERAL_CONVERSATION",
                ),
                (
                    "Hinglish response",
                    "AI automation kya hota hai? Simple Hinglish mein samjhao.",
                    "GENERAL_CONVERSATION",
                ),
                (
                    "Hindi business retrieval",
                    "Northwind Workshop कौन सी सेवाएँ देती है?",
                    "BUSINESS_KNOWLEDGE",
                ),
                (
                    "Hinglish business retrieval",
                    "Northwind Workshop ka pricing kya hai?",
                    "BUSINESS_KNOWLEDGE",
                ),
            ]
            business_session = None
            for label, message, expected in scenarios:
                start = time.perf_counter()
                try:
                    # Each scenario starts a separate session so the clarification case has no context.
                    result = await service.handle(
                        message, None, f"live-{label.lower().replace(' ', '-')}"
                    )
                    if label == "Business knowledge":
                        business_session = result["session_id"]
                    ok = result["route"] == expected
                    detail = f"route={result['route']}; expected={expected}; elapsed_ms={(time.perf_counter() - start) * 1000:.0f}"
                    outcomes.append((label, ok, detail))
                    print(f"[{'PASS' if ok else 'FAIL'}] {label}: {detail}")
                    if label == "Business knowledge":
                        citations_ok = bool(result["citations"])
                        outcomes.append(
                            (
                                "Business citations",
                                citations_ok,
                                f"citations={len(result['citations'])}",
                            )
                        )
                        print(
                            f"[{'PASS' if citations_ok else 'FAIL'}] Business citations: {len(result['citations'])}"
                        )
                except Exception as exc:
                    outcomes.append((label, False, type(exc).__name__))
                    print(f"[FAIL] {label}: {type(exc).__name__}")

            # Follow-up check uses the same isolated session and expects context-aware routing.
            try:
                start = time.perf_counter()
                result = await service.handle(
                    "How much does the website service cost?",
                    business_session,
                    "live-follow-up",
                )
                ok = result["route"] == "BUSINESS_KNOWLEDGE"
                detail = f"route={result['route']}; elapsed_ms={(time.perf_counter() - start) * 1000:.0f}"
                outcomes.append(("Business follow-up", ok, detail))
                print(f"[{'PASS' if ok else 'FAIL'}] Business follow-up: {detail}")
            except Exception as exc:
                outcomes.append(("Business follow-up", False, type(exc).__name__))
                print(f"[FAIL] Business follow-up: {type(exc).__name__}")

            # Release Chroma's persistent file handles before the temporary
            # directory is removed on Windows. Chroma PersistentClient exposes close().
            chroma_client = getattr(vector, "client", None)
            if chroma_client is None:
                chroma_client = getattr(vector, "_client", None)
            close_client = getattr(chroma_client, "close", None)
            if callable(close_client):
                close_client()

        failed = [item for item in outcomes if not item[1]]
        print(f"\nLive checks passed: {len(outcomes) - len(failed)}/{len(outcomes)}")
        print(
            "Measured latencies above include full end-to-end request time; router/retrieval/generation/verification stage timings are also logged by the service."
        )
        status = "PASS" if not failed else "FAIL"
        report_body = [
            f"**Status:** {status}",
            f"**Checks:** {len(outcomes) - len(failed)} passed / {len(outcomes)} total",
            "",
            "### Results",
        ]
        for label, ok, detail in outcomes:
            report_body.append(f"- [{'PASS' if ok else 'FAIL'}] {label} — {detail}")
        report_body += [
            "",
            "Stage-level latency values are emitted in structured server logs. No values are estimated or fabricated.",
        ]
        update_report_section("Live LLM verification", "\n".join(report_body))
        return 1 if failed else 0

    try:
        return asyncio.run(verify())
    except Exception as exc:
        logging.exception("Live verification failed error_type=%s", type(exc).__name__)
        print(
            f"[FAIL] Live verification aborted: {type(exc).__name__}. Check logs; secrets are not printed."
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
