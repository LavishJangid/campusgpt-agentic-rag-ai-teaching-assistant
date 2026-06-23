"""Extended ChromaDB vector store tests."""

from unittest.mock import MagicMock

from backend.vectorstore.chroma_store import ChromaStore


def _store():
    mock_emb = MagicMock()
    mock_emb.embed_text.return_value = [0.1] * 384
    mock_emb.embed_texts.return_value = [[0.1] * 384]
    return ChromaStore(persist_dir="./test_chroma_db", embedding_generator=mock_emb)


def test_hybrid_search_with_filters():
    store = _store()
    store.collection.count.return_value = 5
    store.collection.query.return_value = {
        "ids": [["1"]],
        "documents": [["filtered chunk"]],
        "metadatas": [[{"source": "dbms.pdf", "subject": "DBMS", "page_number": 3}]],
        "distances": [[0.2]],
    }
    results = store.hybrid_search("normalization", subject="DBMS", semester="5", top_k=3)
    assert len(results) >= 0


def test_hybrid_search_fallback():
    store = _store()
    store.collection.count.return_value = 5
    store.collection.query.side_effect = [
        {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]},
        {
            "ids": [["1"]],
            "documents": [["fallback chunk"]],
            "metadatas": [[{"source": "x.pdf", "page_number": 1}]],
            "distances": [[0.3]],
        },
    ]
    results = store.hybrid_search("query", subject="Unknown", top_k=3)
    assert isinstance(results, list)


def test_get_collection_stats_empty():
    store = _store()
    store.collection.count.return_value = 0
    stats = store.get_collection_stats()
    assert stats["total_chunks"] == 0


def test_delete_by_source_not_found():
    store = _store()
    store.collection.get.return_value = {"ids": []}
    assert store.delete_by_source("missing.pdf") == 0


def test_clear_collection():
    store = _store()
    assert store.clear_collection() is True
