"""Document chunking module with intelligent text splitting strategies."""

from typing import Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger


class DocumentChunker:
    """
    Intelligent document chunking with metadata preservation.
    Uses recursive character splitting with overlap for context continuity.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[list[str]] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
            is_separator_regex=False,
        )
        logger.info(
            f"Chunker initialized | size={chunk_size}, overlap={chunk_overlap}"
        )

    def chunk_document(self, document_data: dict) -> list[dict]:
        """
        Split a document into chunks with full metadata.

        Args:
            document_data: Output from DocumentLoader.load()

        Returns:
            List of chunk dicts with text and metadata.
        """
        pages = document_data.get("pages", [])
        metadata = document_data.get("metadata", {})
        chunks = []

        for page in pages:
            page_text = page["text"]
            page_number = page["page_number"]

            if len(page_text) <= self.chunk_size:
                # Page fits in a single chunk
                chunks.append(
                    self._create_chunk(
                        text=page_text,
                        page_number=page_number,
                        chunk_index=len(chunks),
                        metadata=metadata,
                    )
                )
            else:
                # Split page into multiple chunks
                splits = self.text_splitter.split_text(page_text)
                for split_text in splits:
                    chunks.append(
                        self._create_chunk(
                            text=split_text,
                            page_number=page_number,
                            chunk_index=len(chunks),
                            metadata=metadata,
                        )
                    )

        logger.info(
            f"Chunked '{metadata.get('source', 'unknown')}' into {len(chunks)} chunks"
        )
        return chunks

    def _create_chunk(
        self, text: str, page_number: int, chunk_index: int, metadata: dict
    ) -> dict:
        """Create a standardized chunk dictionary with metadata."""
        return {
            "text": text,
            "metadata": {
                "source": metadata.get("source", "unknown"),
                "file_type": metadata.get("file_type", "unknown"),
                "page_number": page_number,
                "chunk_index": chunk_index,
                "chunk_size": len(text),
                "total_pages": metadata.get("total_pages", 0),
                # Educational metadata (populated during ingestion)
                "subject": metadata.get("subject", ""),
                "semester": metadata.get("semester", ""),
                "unit": metadata.get("unit", ""),
                "topic": metadata.get("topic", ""),
                "document_type": metadata.get("document_type", ""),
                "course": metadata.get("course", ""),
            },
        }

    def chunk_text(self, text: str, metadata: Optional[dict] = None) -> list[dict]:
        """Chunk raw text directly."""
        metadata = metadata or {}
        splits = self.text_splitter.split_text(text)

        return [
            self._create_chunk(
                text=split,
                page_number=1,
                chunk_index=i,
                metadata=metadata,
            )
            for i, split in enumerate(splits)
        ]
