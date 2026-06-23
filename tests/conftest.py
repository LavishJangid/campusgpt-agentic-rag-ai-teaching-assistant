"""Pytest configuration and global fixtures."""

import os
import sys
import types
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["APP_ENV"] = "test"
os.environ["GEMINI_API_KEY"] = "mock_key"
os.environ["CHROMA_PERSIST_DIR"] = "./test_chroma_db"
os.environ["DATABASE_URL"] = "sqlite:///./test_campusgpt.db"
os.environ["LOG_LEVEL"] = "WARNING"
os.environ["REDIS_ENABLED"] = "false"
os.environ["RERANKER_DISABLED"] = "true"
os.environ["RAGAS_DISABLED"] = "true"
os.environ["USE_LANGGRAPH_AGENT"] = "false"
os.environ["JWT_SECRET_KEY"] = "test-secret-key-for-pytest-only"


def _ensure_module(name: str) -> types.ModuleType:
    if name not in sys.modules:
        sys.modules[name] = types.ModuleType(name)
    return sys.modules[name]


def _setup_mock_llm_stack():
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "Mocked LLM Response text with code context."
    mock_llm.invoke.return_value = mock_response

    genai = _ensure_module("langchain_google_genai")
    genai.ChatGoogleGenerativeAI = MagicMock(return_value=mock_llm)
    return mock_llm


_mock_llm = _setup_mock_llm_stack()


@pytest.fixture(autouse=True)
def mock_dependencies(monkeypatch):
    monkeypatch.setattr(
        "langchain_google_genai.ChatGoogleGenerativeAI",
        MagicMock(return_value=_mock_llm),
        raising=False,
    )

    mock_embeddings = MagicMock()
    mock_embeddings.embed_query.return_value = [0.1] * 384
    mock_embeddings.embed_documents.return_value = [[0.1] * 384]
    monkeypatch.setattr(
        "langchain_community.embeddings.HuggingFaceEmbeddings",
        MagicMock(return_value=mock_embeddings),
        raising=False,
    )

    mock_collection = MagicMock()
    mock_collection.count.return_value = 10
    mock_collection.get.return_value = {
        "ids": ["1"],
        "documents": ["sample doc chunk"],
        "metadatas": [
            {
                "source": "test.pdf",
                "page_number": 1,
                "document_type": "notes",
                "subject": "ML",
                "semester": "6",
                "topic": "Overfitting",
            }
        ],
    }
    mock_collection.query.return_value = {
        "ids": [["1"]],
        "documents": [["sample doc chunk"]],
        "metadatas": [
            [
                {
                    "source": "test.pdf",
                    "page_number": 1,
                    "document_type": "notes",
                    "subject": "ML",
                    "semester": "6",
                    "topic": "Overfitting",
                }
            ]
        ],
        "distances": [[0.1]],
    }

    mock_chroma_client = MagicMock()
    mock_chroma_client.get_or_create_collection.return_value = mock_collection
    monkeypatch.setattr("chromadb.PersistentClient", MagicMock(return_value=mock_chroma_client))


@pytest.fixture
def test_app_client(monkeypatch):
    monkeypatch.setenv("AUTH_DISABLED", "true")
    from backend.config import get_settings

    get_settings.cache_clear()
    from backend.main import app

    with TestClient(app) as client:
        yield client
    get_settings.cache_clear()


@pytest.fixture
def auth_client(monkeypatch):
    """Register and return client with real JWT auth."""
    monkeypatch.setenv("AUTH_DISABLED", "false")
    from backend.config import get_settings

    get_settings.cache_clear()
    from backend.main import app

    with TestClient(app) as client:
        reg = client.post(
            "/auth/register",
            json={
                "email": "test@rgpv.edu",
                "username": "testuser",
                "password": "testpass123",
                "full_name": "Test User",
            },
        )
        if reg.status_code != 201:
            # User may already exist from prior test run
            pass
        login = client.post(
            "/auth/login",
            json={"email": "test@rgpv.edu", "password": "testpass123"},
        )
        if login.status_code == 200:
            token = login.json()["access_token"]
            client.headers.update({"Authorization": f"Bearer {token}"})
        yield client
    get_settings.cache_clear()
