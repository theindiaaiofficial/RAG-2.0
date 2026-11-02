from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import create_app


def test_health_home_and_knowledge_status(settings):
    app = create_app(settings)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert "Orbit Assistant" in client.get("/").text
        status = client.get("/api/knowledge/status")
        assert status.status_code == 200
        assert "document_count" in status.json()
        ready = client.get("/ready")
        # The test Settings contain placeholder keys but Chroma may be unavailable in minimal test env.
        assert ready.status_code in (200, 503)


def test_conversation_api_and_missing_session(settings):
    app = create_app(settings)
    with TestClient(app) as client:
        created = client.post("/api/conversations", json={})
        assert created.status_code == 200
        session_id = created.json()["session_id"]
        assert client.get(f"/api/conversations/{session_id}").status_code == 200
        assert client.get("/api/conversations/not-a-session").status_code == 404
