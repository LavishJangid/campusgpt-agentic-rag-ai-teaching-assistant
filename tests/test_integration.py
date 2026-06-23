"""Integration tests for full RAG + auth flow."""

from unittest.mock import patch


def test_full_chat_flow_with_auth(auth_client):
    """Register, ingest (mocked), chat, feedback — end-to-end."""
    with patch("backend.main.ingestion_pipeline") as mock_pipe:
        mock_pipe.save_upload.return_value = "/tmp/test.pdf"
        mock_pipe.ingest_file.return_value = {
            "status": "success",
            "message": "OK",
            "source": "test.pdf",
            "chunks": 5,
            "pages": 2,
            "file_type": "pdf",
            "time_seconds": 0.5,
        }
        ingest = auth_client.post(
            "/ingest",
            files={"file": ("test.pdf", b"content", "application/pdf")},
            data={"subject": "ML", "semester": "6", "document_type": "notes"},
        )
        assert ingest.status_code == 200

    chat = auth_client.post(
        "/chat",
        json={"question": "What is machine learning?", "subject": "ML", "session_id": "int-test"},
    )
    assert chat.status_code == 200
    data = chat.json()
    assert "answer" in data

    fb = auth_client.post(
        "/feedback",
        json={"question": "What is ML?", "answer": data["answer"], "feedback": "helpful"},
    )
    assert fb.status_code == 200

    metrics = auth_client.get("/metrics")
    assert metrics.status_code == 200
    assert metrics.json()["feedback"]["total"] >= 1


def test_protected_endpoint_requires_auth(monkeypatch):
    """Without token, protected endpoints return 401 when auth enabled."""
    monkeypatch.setenv("AUTH_DISABLED", "false")
    from backend.config import get_settings

    get_settings.cache_clear()
    from backend.main import app
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        r = client.get("/metrics")
        assert r.status_code == 401
    get_settings.cache_clear()
