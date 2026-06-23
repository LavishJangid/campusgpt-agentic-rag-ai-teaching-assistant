"""Tests for ChromaDB Vector Store."""

from unittest.mock import MagicMock
import pytest
from backend.vectorstore.chroma_store import ChromaStore


def test_chroma_store_add_documents():
    """Test adding documents with metadata to ChromaDB."""
    mock_emb = MagicMock()
    mock_emb.embed_texts.return_value = [[0.1, 0.2] * 192]
    
    store = ChromaStore(persist_dir="./test_chroma_db", embedding_generator=mock_emb)
    
    texts = ["Artificial intelligence is revolutionizing education."]
    metadatas = [{"source": "lecture1.pdf", "subject": "AI", "semester": "6"}]
    
    ids = store.add_documents(texts, metadatas)
    
    assert len(ids) == 1
    assert isinstance(ids[0], str)
    assert store.collection.add.called


def test_chroma_store_similarity_search():
    """Test semantic search retrieval and scoring."""
    mock_emb = MagicMock()
    mock_emb.embed_text.return_value = [0.1] * 384
    
    store = ChromaStore(persist_dir="./test_chroma_db", embedding_generator=mock_emb)
    
    # Configure mock response from chroma query
    store.collection.query.return_value = {
        "ids": [["doc-1"]],
        "documents": [["Data Science covers stats and machine learning."]],
        "metadatas": [[{"source": "lecture1.pdf", "page_number": 3}]],
        "distances": [[0.25]]
    }
    
    results = store.similarity_search("Data Science introduction", top_k=1)
    
    assert len(results) == 1
    assert results[0]["id"] == "doc-1"
    assert results[0]["similarity_score"] == 0.75  # 1 - 0.25
    assert results[0]["source"] == "lecture1.pdf"
    assert results[0]["page_number"] == 3


def test_chroma_store_delete():
    """Test deleting documents from collection by source name."""
    store = ChromaStore(persist_dir="./test_chroma_db")
    
    # Configure mock response for delete lookup
    store.collection.get.return_value = {"ids": ["doc-1", "doc-2"]}
    
    deleted_count = store.delete_by_source("lecture1.pdf")
    
    assert deleted_count == 2
    assert store.collection.delete.called
