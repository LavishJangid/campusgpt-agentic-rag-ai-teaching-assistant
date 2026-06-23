"""
CampusGPT — RAG AI Teaching Assistant (Production)
==================================================
FastAPI backend with JWT auth, SQLite, Redis memory, reranking, LangGraph, RAGAS.
"""

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from loguru import logger
from sqlalchemy.orm import Session

from backend.auth.dependencies import get_current_user
from backend.auth.router import router as auth_router
from backend.config import get_settings, BASE_DIR
from backend.database.models import EvaluationRun, User
from backend.database.services import (
    get_feedback_stats,
    log_analytics,
    save_chat_message,
    save_feedback,
    save_uploaded_document,
)
from backend.database.session import get_db, init_db
from backend.evaluation import RAGASEvaluator
from backend.ingestion.pipeline import IngestionPipeline
from backend.memory import ConversationMemory
from backend.models import (
    ChatRequest,
    ChatResponse,
    DeleteRequest,
    DocumentListResponse,
    EvaluationRequest,
    EvaluationResponse,
    FeedbackRequest,
    FeedbackStatsResponse,
    HealthResponse,
    IngestResponse,
    MetricsResponse,
)
from backend.rag.engine import RAGEngine

settings = get_settings()

rag_engine: RAGEngine | None = None
ingestion_pipeline: IngestionPipeline | None = None
conversation_memory: ConversationMemory | None = None
ragas_evaluator: RAGASEvaluator | None = None

limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_engine, ingestion_pipeline, conversation_memory, ragas_evaluator

    logger.info("Starting CampusGPT...")

    os.makedirs(settings.upload_dir, exist_ok=True)
    os.makedirs(settings.processed_dir, exist_ok=True)
    os.makedirs(BASE_DIR / "data", exist_ok=True)
    os.makedirs(Path(settings.log_file).parent, exist_ok=True)

    init_db()
    logger.info("Database initialized")

    rag_engine = RAGEngine()
    ingestion_pipeline = IngestionPipeline()
    conversation_memory = ConversationMemory(max_messages=50)
    ragas_evaluator = RAGASEvaluator()

    logger.info("All systems initialized")
    yield
    logger.info("Shutting down CampusGPT...")


app = FastAPI(
    title="CampusGPT — RAG AI Teaching Assistant",
    description="Production-grade RAG teaching assistant for RGPV students.",
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(auth_router)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    elapsed = round(time.time() - start_time, 3)
    logger.info(f"{request.method} {request.url.path} | {response.status_code} | {elapsed}s")
    return response


@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "CampusGPT — RAG AI Teaching Assistant API",
        "version": settings.app_version,
        "docs": "/docs",
        "auth": {"register": "POST /auth/register", "login": "POST /auth/login"},
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    try:
        doc_count = rag_engine.vector_store.get_collection_stats().get("total_chunks", 0) if rag_engine else 0
        return HealthResponse(
            status="healthy",
            version=settings.app_version,
            vector_store_docs=doc_count,
            environment=settings.app_env,
        )
    except Exception as e:
        return HealthResponse(status=f"degraded: {e}", version=settings.app_version)


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
@limiter.limit(settings.rate_limit)
async def chat(
    request: Request,
    chat_request: ChatRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG Engine not initialized")

    question = chat_request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    if len(question) > 5000:
        raise HTTPException(status_code=400, detail="Question too long (max 5000 chars)")

    history = conversation_memory.get_history(chat_request.session_id)
    mode = chat_request.mode.lower()

    if mode == "exam_prep":
        result = rag_engine.exam_preparation(
            subject=chat_request.subject, topic=chat_request.topic, unit=chat_request.unit
        )
        answer, sources, response_time = result["content"], [
            {"source": s, "page_number": 0, "similarity_score": 0, "citation_label": s}
            for s in result.get("sources", [])
        ], result["response_time"]
    elif mode == "quiz":
        result = rag_engine.generate_quiz(
            subject=chat_request.subject,
            topic=chat_request.topic,
            difficulty=chat_request.difficulty,
            num_questions=chat_request.num_questions,
        )
        answer = result["quiz"]
        sources = [{"source": s, "page_number": 0, "similarity_score": 0, "citation_label": s} for s in result.get("sources", [])]
        response_time = result["response_time"]
    elif mode == "viva":
        result = rag_engine.generate_viva_questions(
            subject=chat_request.subject,
            topic=chat_request.topic,
            num_questions=chat_request.num_questions,
        )
        answer = result["questions"]
        sources = [{"source": s, "page_number": 0, "similarity_score": 0, "citation_label": s} for s in result.get("sources", [])]
        response_time = result["response_time"]
    elif mode == "assignment":
        result = rag_engine.assignment_help(question=question, subject=chat_request.subject)
        answer = result["answer"]
        sources = result.get("sources", [])
        response_time = result["response_time"]
    elif mode == "important_questions":
        result = rag_engine.find_important_questions(
            subject=chat_request.subject, unit=chat_request.unit
        )
        answer = result["questions"]
        sources = [{"source": s, "page_number": 0, "similarity_score": 0, "citation_label": s} for s in result.get("sources", [])]
        response_time = result["response_time"]
    else:
        result = rag_engine.chat(
            question=question,
            chat_history=history,
            subject=chat_request.subject,
            semester=chat_request.semester,
            unit=chat_request.unit,
            topic=chat_request.topic,
            document_type=chat_request.document_type,
            course=chat_request.course,
            top_k=chat_request.top_k,
            use_mmr=chat_request.use_mmr,
        )
        answer = result["answer"]
        sources = result["sources"]
        response_time = result["response_time"]

    conversation_memory.add_exchange(chat_request.session_id, question, answer)
    save_chat_message(db, chat_request.session_id, "user", question, user.id, mode)
    save_chat_message(db, chat_request.session_id, "assistant", answer, user.id, mode)
    log_analytics(db, "chat", user.id, {"mode": mode, "session_id": chat_request.session_id})

    return ChatResponse(
        answer=answer,
        sources=sources,
        confidence_score=result.get("confidence_score", 0),
        response_time=response_time,
        follow_up_questions=result.get("follow_up_questions", []),
        context_used=result.get("context_used", False),
        num_sources=len(sources),
        session_id=chat_request.session_id,
    )


@app.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_document(
    file: UploadFile = File(...),
    subject: str = Form(default=""),
    semester: str = Form(default=""),
    unit: str = Form(default=""),
    topic: str = Form(default=""),
    document_type: str = Form(default=""),
    course: str = Form(default=""),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not ingestion_pipeline:
        raise HTTPException(status_code=503, detail="Ingestion pipeline not initialized")
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    ext = Path(file.filename).suffix.lower().lstrip(".")
    if ext not in settings.allowed_extensions_list:
        raise HTTPException(status_code=400, detail=f"Unsupported format: .{ext}")

    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > settings.max_upload_size_mb:
        raise HTTPException(status_code=400, detail=f"File too large: {size_mb:.1f}MB")

    try:
        file_path = ingestion_pipeline.save_upload(content, file.filename)
        result = ingestion_pipeline.ingest_file(
            file_path=file_path,
            subject=subject,
            semester=semester,
            unit=unit,
            topic=topic,
            document_type=document_type,
            course=course,
        )
        save_uploaded_document(
            db,
            source=result.get("source", file.filename),
            user_id=user.id,
            subject=subject,
            semester=semester,
            unit=unit,
            topic=topic,
            document_type=document_type,
            course=course,
            chunks=result.get("chunks", 0),
            pages=result.get("pages", 0),
            file_size_kb=round(len(content) / 1024, 2),
        )
        log_analytics(db, "ingest", user.id, {"source": file.filename})
        return IngestResponse(**result)
    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")


@app.get("/documents", response_model=DocumentListResponse, tags=["Documents"])
async def list_documents(user: User = Depends(get_current_user)):
    if not rag_engine:
        raise HTTPException(status_code=503, detail="System not initialized")
    stats = rag_engine.vector_store.get_collection_stats()
    return DocumentListResponse(
        documents=stats.get("documents", []),
        total_documents=stats.get("total_documents", 0),
        total_chunks=stats.get("total_chunks", 0),
    )


@app.delete("/documents", tags=["Documents"])
async def delete_document(
    request: DeleteRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not rag_engine:
        raise HTTPException(status_code=503, detail="System not initialized")
    deleted_count = rag_engine.vector_store.delete_by_source(request.source)
    if deleted_count == 0:
        raise HTTPException(status_code=404, detail=f"Document not found: {request.source}")
    upload_path = os.path.join(settings.upload_dir, request.source)
    if os.path.exists(upload_path):
        os.remove(upload_path)
    log_analytics(db, "delete_document", user.id, {"source": request.source})
    return {
        "status": "success",
        "message": f"Deleted {deleted_count} chunks from '{request.source}'",
        "deleted_chunks": deleted_count,
    }


@app.get("/metrics", response_model=MetricsResponse, tags=["Analytics"])
async def get_metrics(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not rag_engine:
        raise HTTPException(status_code=503, detail="System not initialized")
    engine_metrics = rag_engine.get_metrics()
    vs_stats = engine_metrics.get("vector_store_stats", {})
    return MetricsResponse(
        total_queries=engine_metrics.get("total_queries", 0),
        avg_response_time=engine_metrics.get("avg_response_time", 0),
        total_documents=vs_stats.get("total_documents", 0),
        total_chunks=vs_stats.get("total_chunks", 0),
        topics_searched=engine_metrics.get("topics_searched", {}),
        subjects=vs_stats.get("subjects", []),
        semesters=vs_stats.get("semesters", []),
        feedback=get_feedback_stats(db),
    )


@app.post("/feedback", tags=["Feedback"])
async def submit_feedback(
    payload: FeedbackRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    fb = save_feedback(
        db,
        question=payload.question,
        answer=payload.answer,
        feedback=payload.feedback,
        user_id=user.id,
        session_id=payload.session_id,
    )
    return {"status": "success", "id": fb.id, "feedback": fb.feedback}


@app.get("/feedback/stats", response_model=FeedbackStatsResponse, tags=["Feedback"])
async def feedback_stats(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    stats = get_feedback_stats(db)
    return FeedbackStatsResponse(**stats)


@app.post("/evaluation", response_model=EvaluationResponse, tags=["Evaluation"])
async def run_evaluation(
    payload: EvaluationRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not rag_engine or not ragas_evaluator:
        raise HTTPException(status_code=503, detail="Evaluation not available")

    start = time.time()
    result = rag_engine.chat(question=payload.question, chat_history=[])
    contexts = [s.get("text_preview", "") for s in result.get("sources", [])]
    eval_scores = ragas_evaluator.evaluate(
        question=payload.question,
        answer=result["answer"],
        contexts=contexts or ["No context retrieved"],
        ground_truth=payload.ground_truth or None,
    )

    run = EvaluationRun(
        user_id=user.id,
        question=payload.question,
        answer=result["answer"],
        context="\n".join(contexts),
        faithfulness=eval_scores["faithfulness"],
        context_precision=eval_scores["context_precision"],
        answer_relevancy=eval_scores["answer_relevancy"],
        latency_seconds=eval_scores["latency_seconds"],
    )
    db.add(run)
    db.commit()

    return EvaluationResponse(
        question=payload.question,
        answer=result["answer"],
        sources=result.get("sources", []),
        **eval_scores,
    )


@app.get("/evaluation/history", tags=["Evaluation"])
async def evaluation_history(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    limit: int = 20,
):
    runs = (
        db.query(EvaluationRun)
        .order_by(EvaluationRun.created_at.desc())
        .limit(limit)
        .all()
    )
    return {
        "runs": [
            {
                "id": r.id,
                "question": r.question,
                "faithfulness": r.faithfulness,
                "context_precision": r.context_precision,
                "answer_relevancy": r.answer_relevancy,
                "latency_seconds": r.latency_seconds,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in runs
        ]
    }


@app.get("/chat/history/{session_id}", tags=["Chat"])
async def get_chat_history(session_id: str, user: User = Depends(get_current_user)):
    if not conversation_memory:
        raise HTTPException(status_code=503, detail="System not initialized")
    history = conversation_memory.get_history(session_id)
    return {"session_id": session_id, "messages": history, "count": len(history)}


@app.delete("/chat/history/{session_id}", tags=["Chat"])
async def clear_chat_history(session_id: str, user: User = Depends(get_current_user)):
    if not conversation_memory:
        raise HTTPException(status_code=503, detail="System not initialized")
    conversation_memory.clear_session(session_id)
    return {"status": "success", "message": f"Cleared history for session: {session_id}"}


@app.get("/chat/export/{session_id}", tags=["Chat"])
async def export_chat(session_id: str, user: User = Depends(get_current_user)):
    if not conversation_memory:
        raise HTTPException(status_code=503, detail="System not initialized")
    return {"session_id": session_id, "export": conversation_memory.export_chat(session_id)}


@app.get("/search/topic/{topic}", tags=["Search"])
async def search_by_topic(topic: str, top_k: int = 10, user: User = Depends(get_current_user)):
    if not rag_engine:
        raise HTTPException(status_code=503, detail="System not initialized")
    results = rag_engine.search_by_topic(topic, top_k=top_k)
    return {"topic": topic, "results": results, "count": len(results)}


@app.get("/search/subject/{subject}", tags=["Search"])
async def search_by_subject(
    subject: str, query: str = "", top_k: int = 10, user: User = Depends(get_current_user)
):
    if not rag_engine:
        raise HTTPException(status_code=503, detail="System not initialized")
    results = rag_engine.search_by_subject(subject, query=query, top_k=top_k)
    return {"subject": subject, "results": results, "count": len(results)}


@app.get("/search/semester/{semester}", tags=["Search"])
async def search_by_semester(
    semester: str, query: str = "", top_k: int = 10, user: User = Depends(get_current_user)
):
    if not rag_engine:
        raise HTTPException(status_code=503, detail="System not initialized")
    results = rag_engine.search_by_semester(semester, query=query, top_k=top_k)
    return {"semester": semester, "results": results, "count": len(results)}


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An internal error occurred. Please try again.",
            "error": str(exc) if settings.debug else "Internal server error",
        },
    )
