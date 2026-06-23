"""Tests for FastAPI backend REST API endpoints."""

import pytest
from unittest.mock import MagicMock, patch


def test_api_root(test_app_client):
    """Test root API endpoint response structure."""
    response = test_app_client.get("/")
    assert response.status_code == 200
    assert "CampusGPT" in response.json()["message"]


def test_api_health(test_app_client):
    """Test health check status."""
    response = test_app_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_api_metrics(test_app_client):
    """Test metrics calculation retrieval."""
    response = test_app_client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_queries" in data
    assert "avg_response_time" in data
    assert "total_chunks" in data


def test_api_chat_default(test_app_client):
    """Test general RAG Q&A chat endpoint."""
    payload = {
        "question": "What is overfitting?",
        "subject": "Machine Learning",
        "semester": "6",
        "top_k": 3,
        "session_id": "test_session"
    }
    response = test_app_client.post("/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert data["session_id"] == "test_session"


def test_api_chat_exam_prep(test_app_client):
    """Test RAG exam preparation guide generation."""
    payload = {
        "question": "Generate study guide",
        "subject": "Data Science",
        "mode": "exam_prep",
        "session_id": "test_session_prep"
    }
    response = test_app_client.post("/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert data["session_id"] == "test_session_prep"


def test_api_chat_quiz(test_app_client):
    """Test RAG quiz generation endpoint."""
    payload = {
        "question": "Generate quiz",
        "subject": "AI",
        "mode": "quiz",
        "difficulty": "hard",
        "num_questions": 5,
        "session_id": "test_session_quiz"
    }
    response = test_app_client.post("/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data


def test_api_chat_viva(test_app_client):
    """Test viva questions generator."""
    payload = {
        "question": "viva prep",
        "subject": "OOPs",
        "mode": "viva",
        "num_questions": 8
    }
    response = test_app_client.post("/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data


def test_api_chat_history(test_app_client):
    """Test chat history retrieval and deletion."""
    # First write a query to establish session
    test_app_client.post("/chat", json={"question": "Hello", "session_id": "session-123"})
    
    # Get history
    history_res = test_app_client.get("/chat/history/session-123")
    assert history_res.status_code == 200
    assert len(history_res.json()["messages"]) >= 2  # Question and Answer
    
    # Clear history
    del_res = test_app_client.delete("/chat/history/session-123")
    assert del_res.status_code == 200
    
    # Verify empty history
    history_res = test_app_client.get("/chat/history/session-123")
    assert len(history_res.json()["messages"]) == 0


def test_api_chat_export(test_app_client):
    """Test formatting and exporting chat session."""
    test_app_client.post("/chat", json={"question": "Explain RAG", "session_id": "session-export"})
    
    response = test_app_client.get("/chat/export/session-export")
    assert response.status_code == 200
    assert "session-export" in response.json()["export"]
    assert "[USER]" in response.json()["export"]
    assert "[ASSISTANT]" in response.json()["export"]


def test_api_list_documents(test_app_client):
    """Test active documents listing."""
    response = test_app_client.get("/documents")
    assert response.status_code == 200
    data = response.json()
    assert "documents" in data
    assert "total_documents" in data


@patch("backend.main.ingestion_pipeline")
def test_api_ingest_document(mock_pipeline, test_app_client):
    """Test document ingestion uploading endpoint."""
    mock_pipeline.save_upload.return_value = "/path/to/uploaded/test.pdf"
    mock_pipeline.ingest_file.return_value = {
        "status": "success",
        "message": "Ingested 10 chunks",
        "source": "test.pdf",
        "chunks": 10,
        "pages": 2,
        "file_type": "pdf",
        "time_seconds": 1.2
    }
    
    file_payload = {
        "file": ("test.pdf", b"pdf binary content", "application/pdf")
    }
    form_data = {
        "subject": "Data Mining",
        "semester": "7",
        "document_type": "lecture"
    }
    
    response = test_app_client.post("/ingest", files=file_payload, data=form_data)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["chunks"] == 10
    assert data["source"] == "test.pdf"


def test_api_delete_document(test_app_client):
    """Test document deletion by name."""
    with patch("backend.main.rag_engine") as mock_engine:
        mock_engine.vector_store.delete_by_source.return_value = 5
        
        response = test_app_client.request(
            "DELETE",
            "/documents",
            json={"source": "lecture2.pdf"}
        )
        assert response.status_code == 200
        assert "Deleted 5 chunks" in response.json()["message"]


def test_api_delete_document_not_found(test_app_client):
    """Test deleting document that doesn't exist."""
    with patch("backend.main.rag_engine") as mock_engine:
        mock_engine.vector_store.delete_by_source.return_value = 0
        
        response = test_app_client.request(
            "DELETE",
            "/documents",
            json={"source": "missing.pdf"}
        )
        assert response.status_code == 404
