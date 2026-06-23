"""Tests for document chunking module."""

from backend.ingestion.chunker import DocumentChunker


def test_chunker_basic():
    """Test text splitting logic and overlap settings."""
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
    
    raw_document = {
        "text": "This is a very long sentence designed to test text chunking. Let's see if the splitter cuts it into chunks accurately.",
        "pages": [
            {
                "text": "This is a very long sentence designed to test text chunking. Let's see if the splitter cuts it into chunks accurately.",
                "page_number": 1,
            }
        ],
        "metadata": {
            "source": "test.txt",
            "file_type": "txt",
            "total_pages": 1,
            "subject": "Data Science",
            "semester": "5"
        }
    }
    
    chunks = chunker.chunk_document(raw_document)
    
    assert len(chunks) > 0
    assert chunks[0]["metadata"]["source"] == "test.txt"
    assert chunks[0]["metadata"]["subject"] == "Data Science"
    assert chunks[0]["metadata"]["semester"] == "5"
    assert chunks[0]["metadata"]["page_number"] == 1
    assert chunks[0]["metadata"]["chunk_size"] <= 100


def test_chunk_raw_text():
    """Test text chunking with standalone strings."""
    chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)
    text = "Machine learning is the study of computer algorithms that improve automatically through experience."
    
    chunks = chunker.chunk_text(text, metadata={"subject": "ML"})
    
    assert len(chunks) > 0
    assert all(len(c["text"]) <= 50 for c in chunks)
    assert chunks[0]["metadata"]["subject"] == "ML"
