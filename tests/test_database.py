"""Tests for database services."""

from backend.database.services import (
    create_user,
    get_feedback_stats,
    get_user_by_email,
    save_feedback,
)
from backend.auth.security import hash_password


def test_create_user_and_lookup():
    import uuid
    from backend.database.session import SessionLocal, init_db

    init_db()
    db = SessionLocal()
    try:
        uid = uuid.uuid4().hex[:8]
        email = f"dbtest_{uid}@rgpv.edu"
        user = create_user(
            db,
            email=email,
            username=f"dbtest_{uid}",
            password_hash=hash_password("pass1234"),
        )
        found = get_user_by_email(db, email)
        assert found is not None
        assert found.id == user.id
    finally:
        db.close()


def test_feedback_stats_empty():
    from backend.database.session import SessionLocal, init_db

    init_db()
    db = SessionLocal()
    try:
        stats = get_feedback_stats(db)
        assert "satisfaction_rate" in stats
    finally:
        db.close()
