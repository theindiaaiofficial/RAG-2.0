from __future__ import annotations

import asyncio
import importlib.util
import logging
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.models import (
    ChatRequest,
    ChatResponse,
    ConversationOut,
    ConversationSummary,
    KnowledgeStatus,
    ReindexRequest,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def _module_installed(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False
TEMPLATES = Jinja2Templates(directory=str(Path(__file__).resolve().parents[1] / "frontend" / "templates"))


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def home(request: Request):
    return TEMPLATES.TemplateResponse(request=request, name="index.html", context={})


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
async def ready(request: Request) -> dict[str, Any]:
    state = request.app.state
    issues = list(getattr(state, "startup_issues", []))
    settings = state.settings
    issues.extend(settings.readiness_issues())
    components = {
        "sqlite": bool(getattr(state, "store", None)),
        "chroma": bool(getattr(getattr(state, "vector_store", None), "_collection", None)),
        "embeddings_configured": bool(
            settings.embedding_model
            and settings.embedding_api_key
            and settings.embedding_api_key.get_secret_value().strip()
        ),
        "embeddings_client_initialized": bool(
            getattr(getattr(state, "embeddings", None), "is_loaded", False)
        ),
        "litellm_installed": _module_installed("litellm"),
        "google_genai_installed": _module_installed("google.genai"),
        "router_model_configured": settings.provider_ready("router"),
        "answer_model_configured": settings.provider_ready("answer"),
    }
    is_ready = not issues and all(components.values())
    if not is_ready:
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "components": components, "issues": issues},
        )
    return {"status": "ready", "components": components, "issues": []}


@router.post("/api/conversations", response_model=ConversationSummary)
async def create_conversation(request: Request):
    store = request.app.state.store
    session_id = store.create_session()
    conversations = store.list_conversations(limit=100)
    result = next(item for item in conversations if item["session_id"] == session_id)
    return result


@router.get("/api/conversations", response_model=list[ConversationSummary])
async def list_conversations(request: Request):
    return request.app.state.store.list_conversations(limit=100)


@router.get("/api/conversations/{session_id}", response_model=ConversationOut)
async def get_conversation(session_id: str, request: Request):
    data = request.app.state.store.get_conversation(session_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return data


@router.post("/api/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, request: Request):
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    if not payload.message.strip():
        raise HTTPException(status_code=422, detail="Message cannot be empty")
    try:
        result = await request.app.state.chat_service.handle(
            payload.message.strip(), payload.session_id, request_id
        )
        return result
    except KeyError:
        raise HTTPException(status_code=404, detail="Conversation not found") from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    except Exception as exc:
        logger.exception(
            "Chat request failed error_type=%s", type(exc).__name__, extra={"request_id": request_id}
        )
        message = str(exc)
        if "not configured" in message.lower() or "not installed" in message.lower():
            detail = message
        elif "timed out" in message.lower():
            detail = "The language model request timed out. Please try again."
        else:
            detail = "The assistant could not complete this request. Check the server setup and logs."
        raise HTTPException(status_code=503, detail=detail) from None


@router.get("/api/knowledge/status", response_model=KnowledgeStatus)
async def knowledge_status(request: Request):
    ingestor = request.app.state.ingestor
    try:
        result = ingestor.status()
    except Exception:
        result = {
            "ready": False,
            "directory": str(request.app.state.settings.knowledge_dir),
            "empty": True,
            "document_count": 0,
            "chunk_count": 0,
            "indexed_files": 0,
            "errors": [
                {
                    "file": "startup",
                    "error": "Knowledge components are unavailable; check dependencies and logs.",
                }
            ],
            "message": "Knowledge components are unavailable. Install requirements and restart.",
        }
    if request.app.state.startup_issues:
        result["ready"] = False
        result["errors"] = result["errors"] + [
            {"file": "startup", "error": issue} for issue in request.app.state.startup_issues
        ]
    return result


@router.post("/api/knowledge/reindex")
async def reindex(payload: ReindexRequest, request: Request):
    ingestor = request.app.state.ingestor
    try:
        stats = await asyncio.to_thread(ingestor.scan_and_ingest, payload.full)
        return {"status": "completed" if not stats["failed_files"] else "completed_with_errors", **stats}
    except Exception as exc:
        logger.exception("Knowledge reindex failed error_type=%s", type(exc).__name__)
        raise HTTPException(
            status_code=503, detail="Knowledge reindex failed. Check server logs and model setup."
        ) from None
