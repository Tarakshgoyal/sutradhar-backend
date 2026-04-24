import uuid
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, EmailStr, field_validator
from sqlalchemy.orm import Session

from api.auth_utils import (
    hash_password,
    verify_password,
    create_access_token,
    get_current_user,
    send_verification_email,
)
from config import FRONTEND_URL
from models.database import get_db
from models.user import User

router = APIRouter(prefix="/api/auth", tags=["auth"])


# ── Schemas ──────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    name: str
    email: EmailStr
    password: str

    @field_validator("password")
    @classmethod
    def password_min_length(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserOut(BaseModel):
    id: int
    email: str
    name: str
    is_verified: bool
    created_at: datetime

    class Config:
        from_attributes = True


# ── Routes ───────────────────────────────────────────────────────────────────

@router.post("/register", status_code=201)
def register(payload: RegisterRequest, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == payload.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    token = str(uuid.uuid4())
    expiry = datetime.utcnow() + timedelta(hours=24)

    user = User(
        email=payload.email,
        name=payload.name,
        password_hash=hash_password(payload.password),
        verification_token=token,
        verification_token_expiry=expiry,
    )
    db.add(user)
    db.commit()

    try:
        send_verification_email(payload.email, payload.name, token)
    except Exception as exc:
        # Don't block registration if email fails — log it
        print(f"[auth] Failed to send verification email: {exc}")

    return {"message": "Registration successful. Check your email to verify your account."}


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email).first()
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if not user.is_verified:
        raise HTTPException(
            status_code=403,
            detail="Please verify your email first. Check your inbox for the verification link.",
        )

    access_token = create_access_token(data={"sub": str(user.id)})
    return TokenResponse(access_token=access_token)


@router.get("/verify/{token}")
def verify_email(token: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.verification_token == token).first()
    if not user:
        raise HTTPException(status_code=404, detail="Invalid or expired verification link")

    if user.verification_token_expiry and datetime.utcnow() > user.verification_token_expiry:
        raise HTTPException(status_code=400, detail="Verification link has expired")

    user.is_verified = True
    user.verification_token = None
    user.verification_token_expiry = None
    db.commit()

    return RedirectResponse(url=f"{FRONTEND_URL}/auth?verified=1")


@router.get("/me", response_model=UserOut)
def me(current_user: User = Depends(get_current_user)):
    return current_user
