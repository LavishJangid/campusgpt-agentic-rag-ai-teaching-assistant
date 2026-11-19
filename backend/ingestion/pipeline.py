"""Complete ingestion pipeline orchestrator."""

import hashlib
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

    def __init__(self, embedding_generator=None, vector_store=None):
        self.settings = get_settings()
        self.loader = DocumentLoader()
        self.chunker = DocumentChunker(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        self.embedding_generator = (
            embedding_generator if embedding_generator is not None else EmbeddingGenerator()
        )
        self.vector_store = (
            vector_store
            if vector_store is not None
            else ChromaStore(
                persist_dir=self.settings.chroma_persist_dir,
                collection_name=self.embedding_generator.collection_name(
                    self.settings.chroma_collection_name
                ),
                embedding_generator=self.embedding_generator,
            )
        )

        # Ensure directories exist
        os.makedirs(self.settings.upload_dir, exist_ok=True)
        os.makedirs(self.settings.processed_dir, exist_ok=True)

        logger.info("Ingestion pipeline initialized")

    def check_existing_content(self, source: str, content: bytes) -> Optional[dict]:
        """Return an idempotent-skip or conflict result before saving an upload."""
        content_hash = hashlib.sha256(content).hexdigest()
        return self._existing_source_result(Path(source).name, content_hash)

    def _existing_source_result(self, source: str, content_hash: str) -> Optional[dict]:
        existing = self.vector_store.get_source_info(Path(source).name)
        if not isinstance(existing, dict):
            return None
        if existing.get("content_hash") == content_hash:
            return {
                "status": "skipped",
                "message": "This exact document is already indexed.",
                "source": Path(source).name,
                "chunks": existing.get("chunks", 0),
                "time_seconds": 0,
            }
        return {
            "status": "conflict",
            "message": (
                "A different document with this filename is already indexed. "
                "Delete that document explicitly or upload this file with a new filename."
            ),
            "source": Path(source).name,
            "chunks": 0,
            "time_seconds": 0,
        }

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
            source = Path(file_path).name
            digest = hashlib.sha256()
            with open(file_path, "rb") as document_file:
                for block in iter(lambda: document_file.read(1024 * 1024), b""):
                    digest.update(block)
            content_hash = digest.hexdigest()

            # Repeated uploads of the same file are idempotent. A changed file
            # with the same source is rejected so old chunks are never silently
            # mixed with the new revision or deleted without an explicit action.
            existing_result = self._existing_source_result(source, content_hash)
            if existing_result:
                existing_result["time_seconds"] = round(time.time() - start_time, 2)
                if existing_result["status"] == "skipped":
                    logger.info("Skipping already indexed document '{}'", source)
                return existing_result

            # Step 1: Load document
            logger.info(f"Step 1/4: Loading document '{file_path}'")
            doc_data = self.loader.load(file_path)
            doc_data["metadata"]["source"] = source
            doc_data["metadata"]["content_hash"] = content_hash

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
            doc_ids = [
                hashlib.sha256(f"{source}\0{content_hash}\0{index}".encode()).hexdigest()
                for index in range(len(texts))
            ]
            doc_ids = self.vector_store.add_documents(texts, metadatas, ids=doc_ids)

            elapsed = round(time.time() - start_time, 2)
            result = {
                "status": "success",
                "message": f"Successfully ingested {len(chunks)} chunks",
                "source": source,
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
            logger.exception("Ingestion failed for '{}'", file_path)
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
