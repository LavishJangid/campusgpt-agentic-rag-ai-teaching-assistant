"""Authentication API routes."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.auth.dependencies import get_current_user, security
from backend.auth.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from backend.database.models import User
from backend.database.services import (
    create_session_record,
    create_user,
    get_user_by_email,
    get_user_by_username,
    revoke_session,
)
from backend.database.session import get_db
from backend.models import (
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    UserResponse,
)
from fastapi.security import HTTPAuthorizationCredentials

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register(payload: RegisterRequest, db: Session = Depends(get_db)):
    if get_user_by_email(db, payload.email):
        raise HTTPException(status_code=400, detail="Email already registered")
    if get_user_by_username(db, payload.username):
        raise HTTPException(status_code=400, detail="Username already taken")

    user = create_user(
        db,
        email=payload.email,
        username=payload.username,
        password_hash=hash_password(payload.password),
        full_name=payload.full_name,
    )
    return UserResponse(
        id=user.id,
        email=user.email,
        username=user.username,
        full_name=user.full_name,
    )


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    user = get_user_by_email(db, payload.email)
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account disabled")

    token, jti, expires = create_access_token(user.id, user.username)
    create_session_record(db, user.id, jti, expires)
    return TokenResponse(access_token=token, token_type="bearer", expires_at=expires.isoformat())


@router.post("/logout")
def logout(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    if credentials:
        try:
            payload = decode_access_token(credentials.credentials)
            jti = payload.get("jti")
            if jti:
                revoke_session(db, jti)
        except Exception:
            pass
    return {"status": "success", "message": "Logged out successfully"}


@router.get("/me", response_model=UserResponse)
def me(user: User = Depends(get_current_user)):
    return UserResponse(
        id=user.id,
        email=user.email,
        username=user.username,
        full_name=user.full_name,
    )
