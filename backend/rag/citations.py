"""Source citation formatting for RAG responses."""


def format_source_citation(doc: dict) -> dict:
    """Build rich citation metadata from a retrieved document."""
    meta = doc.get("metadata", {})
    return {
        "source": doc.get("source") or meta.get("source", "unknown"),
        "page_number": doc.get("page_number") or meta.get("page_number", 0),
        "subject": meta.get("subject", ""),
        "semester": meta.get("semester", ""),
        "unit": meta.get("unit", ""),
        "topic": meta.get("topic", ""),
        "document_type": meta.get("document_type", ""),
        "similarity_score": doc.get("similarity_score", 0),
        "rerank_score": doc.get("rerank_score"),
        "text_preview": (doc.get("text", "")[:200] + "...") if doc.get("text") else "",
        "citation_label": _build_label(doc, meta),
    }


def _build_label(doc: dict, meta: dict) -> str:
    source = doc.get("source") or meta.get("source", "unknown")
    page = doc.get("page_number") or meta.get("page_number", "N/A")
    parts = [f"{source}", f"Page {page}"]
    if meta.get("subject"):
        parts.append(meta["subject"])
    if meta.get("semester"):
        parts.append(f"Sem {meta['semester']}")
    if meta.get("topic"):
        parts.append(meta["topic"])
    return " | ".join(parts)


def format_sources_list(documents: list[dict]) -> list[dict]:
    return [format_source_citation(doc) for doc in documents]
