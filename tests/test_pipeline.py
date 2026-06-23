"""Tests for ingestion pipeline (mocked)."""

from unittest.mock import MagicMock, patch


@patch("backend.ingestion.pipeline.ChromaStore")
@patch("backend.ingestion.pipeline.EmbeddingGenerator")
@patch("backend.ingestion.pipeline.DocumentChunker")
@patch("backend.ingestion.pipeline.DocumentLoader")
def test_pipeline_ingest(mock_loader_cls, mock_chunker_cls, mock_emb_cls, mock_store_cls, tmp_path):
    from backend.ingestion.pipeline import IngestionPipeline

    mock_loader = mock_loader_cls.return_value
    mock_loader.load.return_value = {
        "text": "Sample content for testing pipeline.",
        "pages": [{"text": "Sample content for testing pipeline.", "page_number": 1}],
        "metadata": {"source": "test.txt", "file_type": "txt", "total_pages": 1},
    }

    mock_chunker = mock_chunker_cls.return_value
    mock_chunker.chunk_document.return_value = [
        {"text": "chunk1", "metadata": {"source": "test.txt", "page_number": 1}},
    ]

    mock_emb = mock_emb_cls.return_value
    mock_emb.embed_texts.return_value = [[0.1] * 384]

    mock_store = mock_store_cls.return_value
    mock_store.add_documents.return_value = ["id-1"]

    pipeline = IngestionPipeline()
    f = tmp_path / "test.txt"
    f.write_text("hello", encoding="utf-8")

    result = pipeline.ingest_file(str(f), subject="DS", semester="5")
    assert result["status"] == "success"
    assert result["chunks"] == 1
