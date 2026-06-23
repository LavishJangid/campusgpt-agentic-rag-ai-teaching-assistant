"""Pydantic models for API request/response schemas."""

from typing import Optional
from pydantic import BaseModel, Field


# ============ Request Models ============

class ChatRequest(BaseModel):
    """Chat endpoint request."""
    question: str = Field(..., min_length=1, max_length=5000, description="Student question")
    subject: str = Field(default="", description="Subject filter")
    semester: str = Field(default="", description="Semester filter")
    unit: str = Field(default="", description="Unit filter")
    topic: str = Field(default="", description="Topic filter")
    document_type: str = Field(default="", description="Document type filter")
    course: str = Field(default="", description="Course filter")
    session_id: str = Field(default="default", description="Session ID for conversation memory")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of documents to retrieve")
    use_mmr: bool = Field(default=False, description="Use MMR for diverse retrieval")
    mode: str = Field(
        default="chat",
        description="Mode: chat, exam_prep, quiz, viva, assignment, important_questions"
    )
    difficulty: str = Field(default="medium", description="Quiz difficulty: easy, medium, hard")
    num_questions: int = Field(default=10, ge=1, le=50, description="Number of quiz/viva questions")


class IngestRequest(BaseModel):
    """Document ingestion metadata."""
    subject: str = Field(default="", description="Subject name")
    semester: str = Field(default="", description="Semester number")
    unit: str = Field(default="", description="Unit number")
    topic: str = Field(default="", description="Topic name")
    document_type: str = Field(
        default="",
        description="Type: notes, lecture, assignment, lab_manual, question_paper, ppt"
    )
    course: str = Field(default="", description="Course name")


class DeleteRequest(BaseModel):
    """Document deletion request."""
    source: str = Field(..., description="Source filename to delete")


# ============ Auth Models ============

class RegisterRequest(BaseModel):
    email: str = Field(..., min_length=5, max_length=255)
    username: str = Field(..., min_length=3, max_length=100)
    password: str = Field(..., min_length=6, max_length=128)
    full_name: str = Field(default="")


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_at: str


class UserResponse(BaseModel):
    id: str
    email: str
    username: str
    full_name: str = ""


# ============ Feedback Models ============

class FeedbackRequest(BaseModel):
    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    feedback: str = Field(..., pattern="^(helpful|not_helpful)$")
    session_id: str = Field(default="")


class FeedbackStatsResponse(BaseModel):
    helpful: int = 0
    not_helpful: int = 0
    total: int = 0
    satisfaction_rate: float = 0.0


# ============ Evaluation Models ============

class EvaluationRequest(BaseModel):
    question: str = Field(..., min_length=1)
    ground_truth: str = Field(default="")


class EvaluationResponse(BaseModel):
    question: str
    answer: str
    faithfulness: float
    context_precision: float
    answer_relevancy: float
    latency_seconds: float
    method: str = "heuristic"
    sources: list[dict] = []


# ============ Response Models ============

class ChatResponse(BaseModel):
    """Chat endpoint response."""
    answer: str
    sources: list[dict] = []
    confidence_score: float = 0.0
    response_time: float = 0.0
    follow_up_questions: list[str] = []
    context_used: bool = False
    num_sources: int = 0
    session_id: str = "default"


class IngestResponse(BaseModel):
    """Ingestion endpoint response."""
    status: str
    message: str
    source: str = ""
    chunks: int = 0
    pages: int = 0
    file_type: str = ""
    time_seconds: float = 0.0


class DocumentListResponse(BaseModel):
    """Document listing response."""
    documents: list[dict] = []
    total_documents: int = 0
    total_chunks: int = 0


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    version: str = "1.0.0"
    vector_store_docs: int = 0
    environment: str = "development"


class MetricsResponse(BaseModel):
    """Metrics endpoint response."""
    total_queries: int = 0
    avg_response_time: float = 0.0
    total_documents: int = 0
    total_chunks: int = 0
    topics_searched: dict = {}
    subjects: list[str] = []
    semesters: list[str] = []
    feedback: dict = {}
