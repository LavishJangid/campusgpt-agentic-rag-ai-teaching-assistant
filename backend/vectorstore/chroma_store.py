"""ChromaDB vector store implementation with semantic search and metadata filtering."""

import uuid
from typing import Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from loguru import logger


class ChromaStore:
    """
    ChromaDB vector store for document storage and retrieval.

    Features:
    - Persistent storage
    - Semantic similarity search
    - MMR (Maximal Marginal Relevance) retrieval
    - Metadata filtering (subject, semester, unit, topic, document_type)
    - Hybrid retrieval combining semantic + metadata filters
    """

    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        collection_name: str = "teaching_assistant",
        embedding_generator=None,
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedding_generator = embedding_generator

        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(
            f"ChromaDB initialized | dir={persist_dir} | "
            f"collection={collection_name} | "
            f"docs={self.collection.count()}"
        )

    def add_documents(
        self,
        texts: list[str],
        metadatas: list[dict],
        ids: Optional[list[str]] = None,
    ) -> list[str]:
        """
        Add documents with embeddings to the vector store.

        Args:
            texts: List of text chunks
            metadatas: List of metadata dicts
            ids: Optional list of document IDs

        Returns:
            List of document IDs
        """
        if not texts:
            return []

        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]

        # Clean metadata (ChromaDB only supports str, int, float, bool)
        clean_metadatas = []
        for meta in metadatas:
            clean = {}
            for k, v in meta.items():
                if v is None:
                    clean[k] = ""
                elif isinstance(v, (str, int, float, bool)):
                    clean[k] = v
                else:
                    clean[k] = str(v)
            clean_metadatas.append(clean)

        # Generate embeddings
        embeddings = None
        if self.embedding_generator:
            embeddings = self.embedding_generator.embed_texts(texts)

        # Add to ChromaDB
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch_end = min(i + batch_size, len(texts))
            add_kwargs = {
                "ids": ids[i:batch_end],
                "documents": texts[i:batch_end],
                "metadatas": clean_metadatas[i:batch_end],
            }
            if embeddings:
                add_kwargs["embeddings"] = embeddings[i:batch_end]

            self.collection.add(**add_kwargs)

        logger.info(f"Added {len(texts)} documents to ChromaDB")
        return ids

    def similarity_search(
        self,
        query: str,
        top_k: int = 5,
        where_filter: Optional[dict] = None,
    ) -> list[dict]:
        """
        Perform semantic similarity search.

        Args:
            query: Search query text
            top_k: Number of results to return
            where_filter: Optional metadata filter (ChromaDB where clause)

        Returns:
            List of result dicts with text, metadata, and score.
        """
        query_embedding = None
        if self.embedding_generator:
            query_embedding = self.embedding_generator.embed_text(query)

        query_kwargs = {
            "n_results": min(top_k, self.collection.count() or 1),
        }

        if query_embedding:
            query_kwargs["query_embeddings"] = [query_embedding]
        else:
            query_kwargs["query_texts"] = [query]

        if where_filter:
            query_kwargs["where"] = where_filter

        query_kwargs["include"] = ["documents", "metadatas", "distances"]

        try:
            results = self.collection.query(**query_kwargs)
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []

        return self._format_results(results)

    def mmr_search(
        self,
        query: str,
        top_k: int = 5,
        diversity: float = 0.3,
        fetch_k: int = 20,
        where_filter: Optional[dict] = None,
    ) -> list[dict]:
        """
        Maximal Marginal Relevance search for diverse results.

        Args:
            query: Search query
            top_k: Number of results to return
            diversity: Balance between relevance and diversity (0-1)
            fetch_k: Number of initial candidates to fetch
            where_filter: Optional metadata filter
        """
        # Fetch more candidates, then re-rank for diversity
        candidates = self.similarity_search(
            query, top_k=fetch_k, where_filter=where_filter
        )

        if len(candidates) <= top_k:
            return candidates

        # Simple MMR: greedily select diverse documents
        if not self.embedding_generator:
            return candidates[:top_k]

        import numpy as np

        query_emb = np.array(self.embedding_generator.embed_text(query))
        candidate_embs = np.array(
            self.embedding_generator.embed_texts(
                [c["text"] for c in candidates]
            )
        )

        selected = []
        remaining = list(range(len(candidates)))

        for _ in range(min(top_k, len(candidates))):
            if not remaining:
                break

            mmr_scores = []
            for idx in remaining:
                relevance = float(np.dot(query_emb, candidate_embs[idx]))

                max_sim = 0.0
                for sel_idx in selected:
                    sim = float(np.dot(candidate_embs[idx], candidate_embs[sel_idx]))
                    max_sim = max(max_sim, sim)

                mmr = (1 - diversity) * relevance - diversity * max_sim
                mmr_scores.append((idx, mmr))

            best_idx = max(mmr_scores, key=lambda x: x[1])[0]
            selected.append(best_idx)
            remaining.remove(best_idx)

        return [candidates[i] for i in selected]

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        subject: str = "",
        semester: str = "",
        unit: str = "",
        topic: str = "",
        document_type: str = "",
        course: str = "",
    ) -> list[dict]:
        """
        Hybrid retrieval combining semantic search with metadata filtering.
        """
        # Build filter
        conditions = []
        if subject:
            conditions.append({"subject": {"$eq": subject}})
        if semester:
            conditions.append({"semester": {"$eq": semester}})
        if unit:
            conditions.append({"unit": {"$eq": unit}})
        if topic:
            conditions.append({"topic": {"$eq": topic}})
        if document_type:
            conditions.append({"document_type": {"$eq": document_type}})
        if course:
            conditions.append({"course": {"$eq": course}})

        where_filter = None
        if len(conditions) == 1:
            where_filter = conditions[0]
        elif len(conditions) > 1:
            where_filter = {"$and": conditions}

        # Try filtered search first
        results = self.similarity_search(query, top_k=top_k, where_filter=where_filter)

        # Fallback to unfiltered if no results
        if not results and where_filter:
            logger.info("Filtered search returned 0 results, falling back to unfiltered")
            results = self.similarity_search(query, top_k=top_k)

        return results

    def delete_by_source(self, source: str) -> int:
        """Delete all documents from a specific source file."""
        try:
            # Get documents with matching source
            results = self.collection.get(
                where={"source": {"$eq": source}},
                include=["metadatas"],
            )
            if results["ids"]:
                self.collection.delete(ids=results["ids"])
                logger.info(f"Deleted {len(results['ids'])} documents from source: {source}")
                return len(results["ids"])
            return 0
        except Exception as e:
            logger.error(f"Delete failed: {str(e)}")
            return 0

    def get_all_documents(self) -> list[dict]:
        """Get summary of all stored documents."""
        try:
            count = self.collection.count()
            if count == 0:
                return []

            results = self.collection.get(
                include=["metadatas"],
                limit=min(count, 10000),
            )

            # Group by source
            sources = {}
            for meta in results["metadatas"]:
                source = meta.get("source", "unknown")
                if source not in sources:
                    sources[source] = {
                        "source": source,
                        "file_type": meta.get("file_type", "unknown"),
                        "subject": meta.get("subject", ""),
                        "semester": meta.get("semester", ""),
                        "document_type": meta.get("document_type", ""),
                        "course": meta.get("course", ""),
                        "chunks": 0,
                        "total_pages": meta.get("total_pages", 0),
                    }
                sources[source]["chunks"] += 1

            return list(sources.values())
        except Exception as e:
            logger.error(f"Get all documents failed: {str(e)}")
            return []

    def get_collection_stats(self) -> dict:
        """Get collection statistics."""
        count = self.collection.count()
        docs = self.get_all_documents()

        subjects = set()
        semesters = set()
        doc_types = set()

        for doc in docs:
            if doc.get("subject"):
                subjects.add(doc["subject"])
            if doc.get("semester"):
                semesters.add(doc["semester"])
            if doc.get("document_type"):
                doc_types.add(doc["document_type"])

        return {
            "total_chunks": count,
            "total_documents": len(docs),
            "subjects": list(subjects),
            "semesters": list(semesters),
            "document_types": list(doc_types),
            "documents": docs,
        }

    def _format_results(self, raw_results: dict) -> list[dict]:
        """Format ChromaDB results into standardized dicts."""
        results = []

        if not raw_results or not raw_results.get("ids"):
            return results

        ids = raw_results["ids"][0] if raw_results["ids"] else []
        documents = raw_results.get("documents", [[]])[0]
        metadatas = raw_results.get("metadatas", [[]])[0]
        distances = raw_results.get("distances", [[]])[0]

        for i, doc_id in enumerate(ids):
            text = documents[i] if i < len(documents) else ""
            meta = metadatas[i] if i < len(metadatas) else {}
            dist = distances[i] if i < len(distances) else 0

            # Convert cosine distance to similarity score
            similarity = round(1 - dist, 4) if dist else 0

            results.append({
                "id": doc_id,
                "text": text,
                "metadata": meta,
                "similarity_score": similarity,
                "source": meta.get("source", "unknown"),
                "page_number": meta.get("page_number", 0),
            })

        # Sort by similarity (highest first)
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results

    def clear_collection(self) -> bool:
        """Delete all documents from the collection."""
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("Collection cleared")
            return True
        except Exception as e:
            logger.error(f"Clear collection failed: {str(e)}")
            return False
