from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.config import get_settings
from app.generation.answer_generator import AnswerGenerator
from app.llm import LLMClient
from app.logging_config import configure_logging
from app.knowledge.ingestion import KnowledgeIngestor
from app.memory.conversation_store import ConversationStore
from app.rag.embeddings import RemoteEmbeddingModel
from app.rag.retriever import RAGRetriever
from app.rag.vector_store import ChromaVectorStore
from app.router.intelligence_router import IntelligenceRouter
from app.services.chat_service import ChatService
from app.verification.evidence import EvidenceEvaluator
from app.verification.grounding import GroundingVerifier

logger = logging.getLogger(__name__)


def create_app(settings_override=None) -> FastAPI:
    active_settings = settings_override or get_settings()
    configure_logging(active_settings.log_level)
    app = FastAPI(title="Orbit Intelligent Assistant", version="0.1.0")

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        active_settings.data_dir.mkdir(parents=True, exist_ok=True)
        active_settings.knowledge_dir.mkdir(parents=True, exist_ok=True)
        application.state.settings = active_settings
        application.state.startup_issues = []
        try:
            import asyncio

            store = ConversationStore(active_settings.sqlite_path)
            application.state.store = store
            vector_store = ChromaVectorStore(active_settings.chroma_dir)
            vector_store.initialize()
            application.state.vector_store = vector_store
            embedding_api_key = (
                active_settings.embedding_api_key.get_secret_value()
                if active_settings.embedding_api_key
                else None
            )
            embeddings = RemoteEmbeddingModel(
                active_settings.embedding_model,
                api_key=embedding_api_key,
                output_dimensionality=active_settings.embedding_output_dimensionality,
                batch_size=active_settings.embedding_batch_size,
            )
            application.state.embeddings = embeddings
            # Initializes a lightweight HTTP client only; no local ML model is loaded.
            await asyncio.to_thread(embeddings.initialize)
            llm = LLMClient(active_settings)
            intelligence_router = IntelligenceRouter(llm, active_settings)
            retriever = RAGRetriever(active_settings, embeddings, vector_store)
            evaluator = EvidenceEvaluator(llm)
            answer_generator = AnswerGenerator(llm, active_settings)
            verifier = GroundingVerifier(llm)
            ingestor = KnowledgeIngestor(active_settings, embeddings, vector_store, store)
            application.state.ingestor = ingestor
            application.state.chat_service = ChatService(
                active_settings, store, intelligence_router, retriever, evaluator, answer_generator, verifier
            )
            if ingestor._files():
                logger.info("Knowledge files detected; beginning startup ingestion")
                await asyncio.to_thread(ingestor.scan_and_ingest, False)
            else:
                logger.info("Knowledge directory is empty; remote embedding client is ready")
            if active_settings.readiness_issues():
                logger.warning("Provider configuration incomplete: %s", active_settings.readiness_issues())
        except Exception as exc:
            logger.exception("Application startup initialization failed error_type=%s", type(exc).__name__)
            application.state.startup_issues.append(
                f"Initialization failed: {type(exc).__name__}. Check server logs."
            )
            if not hasattr(application.state, "store"):
                application.state.store = ConversationStore(active_settings.sqlite_path)
            if not hasattr(application.state, "vector_store"):
                vector_store = ChromaVectorStore(active_settings.chroma_dir)
                try:
                    vector_store.initialize()
                except Exception:
                    pass
                application.state.vector_store = vector_store
            if not hasattr(application.state, "embeddings"):
                embedding_api_key = (
                    active_settings.embedding_api_key.get_secret_value()
                    if active_settings.embedding_api_key
                    else None
                )
                application.state.embeddings = RemoteEmbeddingModel(
                    active_settings.embedding_model,
                    api_key=embedding_api_key,
                    output_dimensionality=active_settings.embedding_output_dimensionality,
                    batch_size=active_settings.embedding_batch_size,
                )
            if not hasattr(application.state, "ingestor"):
                application.state.ingestor = KnowledgeIngestor(
                    active_settings,
                    application.state.embeddings,
                    application.state.vector_store,
                    application.state.store,
                )
            llm = LLMClient(active_settings)
            application.state.chat_service = ChatService(
                active_settings,
                application.state.store,
                IntelligenceRouter(llm, active_settings),
                RAGRetriever(active_settings, application.state.embeddings, application.state.vector_store),
                EvidenceEvaluator(llm),
                AnswerGenerator(llm, active_settings),
                GroundingVerifier(llm),
            )
        yield

    app.router.lifespan_context = lifespan
    static_dir = Path(__file__).resolve().parent / "frontend" / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    app.include_router(router)

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        return response

    return app


app = create_app()
