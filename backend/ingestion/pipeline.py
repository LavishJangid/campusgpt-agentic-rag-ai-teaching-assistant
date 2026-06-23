"""Complete ingestion pipeline orchestrator."""

import os
import time
import shutil
from pathlib import Path
from typing import Optional
from loguru import logger

from backend.config import get_settings
from backend.ingestion.loader import DocumentLoader
from backend.ingestion.chunker import DocumentChunker
from backend.ingestion.embeddings import EmbeddingGenerator
from backend.vectorstore.chroma_store import ChromaStore


class IngestionPipeline:
    """
    End-to-end document ingestion pipeline.

    Flow: Upload → Load → Clean → Chunk → Embed → Store
    """

    def __init__(self):
        self.settings = get_settings()
        self.loader = DocumentLoader()
        self.chunker = DocumentChunker(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        self.embedding_generator = EmbeddingGenerator(
            model_name=self.settings.embedding_model
        )
        self.vector_store = ChromaStore(
            persist_dir=self.settings.chroma_persist_dir,
            collection_name=self.settings.chroma_collection_name,
            embedding_generator=self.embedding_generator,
        )

        # Ensure directories exist
        os.makedirs(self.settings.upload_dir, exist_ok=True)
        os.makedirs(self.settings.processed_dir, exist_ok=True)

        logger.info("Ingestion pipeline initialized")

    def ingest_file(
        self,
        file_path: str,
        subject: str = "",
        semester: str = "",
        unit: str = "",
        topic: str = "",
        document_type: str = "",
        course: str = "",
    ) -> dict:
        """
        Process a single file through the complete ingestion pipeline.

        Returns:
            dict with ingestion results and statistics.
        """
        start_time = time.time()

        try:
            # Step 1: Load document
            logger.info(f"Step 1/4: Loading document '{file_path}'")
            doc_data = self.loader.load(file_path)

            # Inject educational metadata
            doc_data["metadata"]["subject"] = subject
            doc_data["metadata"]["semester"] = semester
            doc_data["metadata"]["unit"] = unit
            doc_data["metadata"]["topic"] = topic
            doc_data["metadata"]["document_type"] = document_type
            doc_data["metadata"]["course"] = course

            # Step 2: Chunk document
            logger.info("Step 2/4: Chunking document")
            chunks = self.chunker.chunk_document(doc_data)

            if not chunks:
                return {
                    "status": "warning",
                    "message": "No text extracted from document",
                    "source": os.path.basename(file_path),
                    "chunks": 0,
                    "time_seconds": round(time.time() - start_time, 2),
                }

            # Step 3: Generate embeddings & store
            logger.info(f"Step 3/4: Generating embeddings for {len(chunks)} chunks")
            texts = [c["text"] for c in chunks]
            metadatas = [c["metadata"] for c in chunks]

            # Step 4: Store in vector database
            logger.info("Step 4/4: Storing in vector database")
            doc_ids = self.vector_store.add_documents(texts, metadatas)

            elapsed = round(time.time() - start_time, 2)
            result = {
                "status": "success",
                "message": f"Successfully ingested {len(chunks)} chunks",
                "source": os.path.basename(file_path),
                "chunks": len(chunks),
                "pages": doc_data["metadata"].get("extracted_pages", 0),
                "file_type": doc_data["metadata"].get("file_type", "unknown"),
                "time_seconds": elapsed,
                "document_ids": doc_ids[:5],  # Return first 5 IDs
            }

            logger.info(
                f"Ingestion complete: {result['source']} | "
                f"{result['chunks']} chunks | {elapsed}s"
            )
            return result

        except Exception as e:
            elapsed = round(time.time() - start_time, 2)
            logger.error(f"Ingestion failed for '{file_path}': {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "source": os.path.basename(file_path),
                "chunks": 0,
                "time_seconds": elapsed,
            }

    def ingest_directory(
        self,
        directory: str,
        subject: str = "",
        semester: str = "",
        course: str = "",
    ) -> list[dict]:
        """Process all supported files in a directory."""
        results = []
        path = Path(directory)

        if not path.exists():
            logger.error(f"Directory not found: {directory}")
            return [{"status": "error", "message": f"Directory not found: {directory}"}]

        supported = DocumentLoader.SUPPORTED_EXTENSIONS
        files = [f for f in path.iterdir() if f.suffix.lower() in supported]

        logger.info(f"Found {len(files)} files to ingest in '{directory}'")

        for file in files:
            result = self.ingest_file(
                str(file),
                subject=subject,
                semester=semester,
                course=course,
            )
            results.append(result)

        return results

    def save_upload(self, file_content: bytes, filename: str) -> str:
        """Save an uploaded file to the uploads directory."""
        upload_path = os.path.join(self.settings.upload_dir, filename)
        with open(upload_path, "wb") as f:
            f.write(file_content)
        logger.info(f"Saved upload: {filename} ({len(file_content)} bytes)")
        return upload_path

    def get_stats(self) -> dict:
        """Get ingestion statistics."""
        vs_stats = self.vector_store.get_collection_stats()
        return {
            "loader_stats": self.loader.stats,
            "vector_store": vs_stats,
        }
