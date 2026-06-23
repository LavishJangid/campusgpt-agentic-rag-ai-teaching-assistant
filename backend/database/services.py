"""Database service helpers."""

import json
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from backend.database.models import (
    AnalyticsEvent,
    ChatMessage,
    Feedback,
    UploadedDocument,
    User,
    UserSession,
)


def create_user(db: Session, email: str, username: str, password_hash: str, full_name: str = "") -> User:
    user = User(email=email, username=username, hashed_password=password_hash, full_name=full_name)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def get_user_by_email(db: Session, email: str) -> User | None:
    return db.query(User).filter(User.email == email).first()


def get_user_by_username(db: Session, username: str) -> User | None:
    return db.query(User).filter(User.username == username).first()


def create_session_record(db: Session, user_id: str, jti: str, expires_at: datetime) -> UserSession:
    session = UserSession(user_id=user_id, token_jti=jti, expires_at=expires_at)
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def revoke_session(db: Session, jti: str) -> bool:
    session = db.query(UserSession).filter(UserSession.token_jti == jti).first()
    if session:
        session.is_revoked = True
        db.commit()
        return True
    return False


def save_chat_message(
    db: Session,
    session_id: str,
    role: str,
    content: str,
    user_id: str | None = None,
    mode: str = "chat",
) -> ChatMessage:
    msg = ChatMessage(
        user_id=user_id,
        session_id=session_id,
        role=role,
        content=content,
        mode=mode,
    )
    db.add(msg)
    db.commit()
    db.refresh(msg)
    return msg


def save_uploaded_document(
    db: Session,
    source: str,
    user_id: str | None,
    subject: str = "",
    semester: str = "",
    unit: str = "",
    topic: str = "",
    document_type: str = "",
    course: str = "",
    chunks: int = 0,
    pages: int = 0,
    file_size_kb: float = 0.0,
) -> UploadedDocument:
    doc = UploadedDocument(
        user_id=user_id,
        source=source,
        subject=subject,
        semester=semester,
        unit=unit,
        topic=topic,
        document_type=document_type,
        course=course,
        chunks=chunks,
        pages=pages,
        file_size_kb=file_size_kb,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return doc


def log_analytics(db: Session, event_type: str, user_id: str | None = None, data: dict | None = None):
    event = AnalyticsEvent(
        user_id=user_id,
        event_type=event_type,
        event_data=json.dumps(data or {}),
    )
    db.add(event)
    db.commit()


def save_feedback(
    db: Session,
    question: str,
    answer: str,
    feedback: str,
    user_id: str | None = None,
    session_id: str = "",
) -> Feedback:
    fb = Feedback(
        user_id=user_id,
        session_id=session_id,
        question=question,
        answer=answer,
        feedback=feedback,
    )
    db.add(fb)
    db.commit()
    db.refresh(fb)
    return fb


def get_feedback_stats(db: Session) -> dict:
    helpful = db.query(Feedback).filter(Feedback.feedback == "helpful").count()
    not_helpful = db.query(Feedback).filter(Feedback.feedback == "not_helpful").count()
    total = helpful + not_helpful
    return {
        "helpful": helpful,
        "not_helpful": not_helpful,
        "total": total,
        "satisfaction_rate": round(helpful / total, 4) if total else 0.0,
    }
