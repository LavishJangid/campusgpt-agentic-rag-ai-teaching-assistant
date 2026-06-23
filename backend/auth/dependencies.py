"""FastAPI auth dependencies."""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session
import jwt

from backend.auth.security import decode_access_token
from backend.config import get_settings
from backend.database.models import User, UserSession
from backend.database.session import get_db

security = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
    db: Session = Depends(get_db),
) -> User:
    settings = get_settings()
    if settings.auth_disabled:
        # Test / dev bypass: return or create a system user
        user = db.query(User).filter(User.username == "system").first()
        if not user:
            from backend.auth.security import hash_password

            user = User(
                email="system@campusgpt.local",
                username="system",
                hashed_password=hash_password("system"),
                full_name="System User",
            )
            db.add(user)
            db.commit()
            db.refresh(user)
        return user

    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        payload = decode_access_token(credentials.credentials)
        user_id = payload.get("sub")
        jti = payload.get("jti")
        if not user_id or not jti:
            raise HTTPException(status_code=401, detail="Invalid token")

        session = (
            db.query(UserSession)
            .filter(UserSession.token_jti == jti, UserSession.is_revoked.is_(False))
            .first()
        )
        if not session:
            raise HTTPException(status_code=401, detail="Session revoked or expired")

        user = db.query(User).filter(User.id == user_id, User.is_active.is_(True)).first()
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return user
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
