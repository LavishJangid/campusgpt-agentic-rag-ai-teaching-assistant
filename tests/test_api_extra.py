"""Additional API endpoint tests."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from backend.config import Settings

def test_search_endpoints(auth_client):
    for path in [
        "/search/topic/Normalization",
        "/search/subject/DBMS?query=SQL",
        "/search/semester/5?query=units",
    ]:
        r = auth_client.get(path)
        assert r.status_code == 200
        assert "count" in r.json()


def test_chat_validation_errors(auth_client):
    r = auth_client.post("/chat", json={"question": "   ", "session_id": "x"})
    assert r.status_code == 400

    r = auth_client.post("/chat", json={"question": "x" * 5001, "session_id": "x"})
    assert r.status_code == 422  # Pydantic max_length validation


def test_settings_reads_production_environment_and_low_memory_flags(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("DEBUG", "false")
    monkeypatch.setenv("REDIS_ENABLED", "false")
    monkeypatch.setenv("RERANKER_DISABLED", "true")

    settings = Settings(_env_file=None)

    assert settings.app_env == "production"
    assert settings.debug is False
    assert settings.redis_enabled is False
    assert settings.reranker_disabled is True


def test_health_reports_active_collection_count_without_initializing_rag(monkeypatch):
    import backend.main as main

    settings = Settings(_env_file=None, APP_ENV="production", DEBUG="false")
    monkeypatch.setattr(main, "settings", settings)
    monkeypatch.setattr(main, "rag_engine", None)
    monkeypatch.setattr(main, "ingestion_pipeline", None)
    monkeypatch.setattr(main, "ragas_evaluator", None)
    monkeypatch.setattr(
        main,
        "get_active_vector_collection_count",
        lambda: ("teaching_assistant__gemini_models_gemini_embedding_001", 7),
    )
    monkeypatch.setattr(
        main,
        "get_rag_engine",
        lambda: (_ for _ in ()).throw(AssertionError("RAG must remain lazy")),
    )

    response = main.health_check()

    assert response["status"] == "healthy"
    assert response["environment"] == "production"
    assert response["vector_store_docs"] == 7
    assert response["vector_store_collection"].endswith("gemini_embedding_001")
    assert main.rag_engine is None
    assert main.ingestion_pipeline is None
    assert main.ragas_evaluator is None


def test_chroma_diagnostic_returns_real_count_without_creating_collection(monkeypatch):
    from backend.vectorstore import chroma_store

    collection = MagicMock()
    collection.count.return_value = 13
    client = MagicMock()
    client.list_collections.return_value = [SimpleNamespace(name="active_collection")]
    client.get_collection.return_value = collection
    monkeypatch.setattr(chroma_store, "get_chroma_client", lambda _: client)

    count = chroma_store.get_existing_collection_count("unused", "active_collection")

    assert count == 13
    client.get_collection.assert_called_once_with(name="active_collection")
    client.get_or_create_collection.assert_not_called()


def test_chroma_diagnostic_reports_zero_when_active_collection_is_absent(monkeypatch):
    from backend.vectorstore import chroma_store

    client = MagicMock()
    client.list_collections.return_value = [SimpleNamespace(name="another_collection")]
    monkeypatch.setattr(chroma_store, "get_chroma_client", lambda _: client)

    assert chroma_store.get_existing_collection_count("unused", "active_collection") == 0
    client.get_collection.assert_not_called()
    client.get_or_create_collection.assert_not_called()


def test_chroma_diagnostic_counts_records_from_a_persisted_collection(tmp_path, monkeypatch):
    import conftest
    from backend.vectorstore import chroma_store

    client = conftest._REAL_CHROMA_PERSISTENT_CLIENT(path=str(tmp_path))
    collection = client.get_or_create_collection(name="active_collection")
    collection.add(
        ids=["chunk-1", "chunk-2"],
        documents=["first chunk", "second chunk"],
        embeddings=[[0.1, 0.2], [0.3, 0.4]],
        metadatas=[{"source": "notes.txt"}, {"source": "notes.txt"}],
    )
    monkeypatch.setattr(chroma_store, "get_chroma_client", lambda _: client)

    assert chroma_store.get_existing_collection_count("unused", "active_collection") == 2


def test_rag_and_ingestion_use_the_same_embedding_and_vector_store(monkeypatch):
    import backend.main as main

    monkeypatch.setattr(main, "rag_engine", None)
    monkeypatch.setattr(main, "ingestion_pipeline", None)
    with (
        patch("backend.rag.engine.ChatGoogleGenerativeAI"),
        patch("backend.rag.engine.ChromaStore") as store_class,
        patch("backend.ingestion.pipeline.IngestionPipeline") as pipeline_class,
    ):
        engine = main.get_rag_engine()
        main.get_ingestion_pipeline()

    kwargs = pipeline_class.call_args.kwargs
    assert kwargs["embedding_generator"] is engine.embedding_generator
    assert kwargs["vector_store"] is engine.vector_store
    assert store_class.call_count == 1
    assert pipeline_class.call_count == 1


def test_redis_and_cross_encoder_are_disabled_by_settings(monkeypatch):
    from backend.memory.redis_memory import ConversationMemory
    from backend.rag.reranker import CrossEncoderReranker
    import backend.memory.redis_memory as redis_memory
    import backend.rag.reranker as reranker_module

    settings = Settings(
        _env_file=None,
        APP_ENV="production",
        DEBUG="false",
        REDIS_ENABLED="false",
        RERANKER_DISABLED="true",
    )
    monkeypatch.setattr(redis_memory, "get_settings", lambda: settings)
    monkeypatch.setattr(reranker_module, "get_settings", lambda: settings)

    memory = ConversationMemory()
    reranker = CrossEncoderReranker()

    assert memory._use_redis is False
    assert reranker.model is None


def test_duplicate_ingestion_is_idempotent_and_changed_file_is_rejected(tmp_path):
    from backend.ingestion.pipeline import IngestionPipeline

    class SourceStore:
        def __init__(self):
            self.source = None
            self.add_calls = 0

        def get_source_info(self, source):
            return self.source

        def add_documents(self, texts, metadatas, ids):
            self.add_calls += 1
            self.source = {
                "content_hash": metadatas[0]["content_hash"],
                "chunks": len(ids),
            }
            return ids

    source_store = SourceStore()
    file_path = tmp_path / "lecture.txt"
    file_path.write_text("A short lecture about vector search.", encoding="utf-8")
    pipeline = IngestionPipeline(
        embedding_generator=MagicMock(),
        vector_store=source_store,
    )

    first = pipeline.ingest_file(str(file_path), subject="AI")
    duplicate = pipeline.ingest_file(str(file_path), subject="AI")
    file_path.write_text("A changed lecture with different content.", encoding="utf-8")
    changed = pipeline.ingest_file(str(file_path), subject="AI")

    assert first["status"] == "success"
    assert duplicate["status"] == "skipped"
    assert changed["status"] == "conflict"
    assert source_store.add_calls == 1
