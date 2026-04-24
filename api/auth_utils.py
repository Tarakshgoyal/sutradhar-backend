import smtplib
import ssl
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import bcrypt
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from config import (
    JWT_SECRET,
    JWT_ALGORITHM,
    JWT_EXPIRE_MINUTES,
    SMTP_HOST,
    SMTP_PORT,
    SMTP_USER,
    SMTP_PASSWORD,
    SMTP_FROM,
    APP_BASE_URL,
)
from models.database import get_db

bearer_scheme = HTTPBearer()


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (
        expires_delta if expires_delta else timedelta(minutes=JWT_EXPIRE_MINUTES)
    )
    to_encode["exp"] = expire
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    db: Session = Depends(get_db),
):
    from models.user import User  # local import to avoid circular

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id: int = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = db.query(User).filter(User.id == int(user_id)).first()
    if user is None:
        raise credentials_exception
    return user


def send_verification_email(email: str, name: str, token: str) -> None:
    verify_url = f"{APP_BASE_URL}/api/auth/verify/{token}"

    html_body = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8" />
  <style>
    body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #0a0a0a; color: #e5e7eb; margin: 0; padding: 0; }}
    .container {{ max-width: 600px; margin: 40px auto; background: #111827; border-radius: 12px; overflow: hidden; border: 1px solid #1f2937; }}
    .header {{ background: linear-gradient(135deg, #0ea5e9, #6366f1); padding: 40px 32px; text-align: center; }}
    .header h1 {{ margin: 0; font-size: 28px; color: #fff; letter-spacing: -0.5px; }}
    .header p {{ margin: 8px 0 0; color: rgba(255,255,255,0.8); font-size: 15px; }}
    .body {{ padding: 40px 32px; }}
    .body p {{ font-size: 15px; line-height: 1.6; color: #9ca3af; margin: 0 0 20px; }}
    .btn {{ display: inline-block; background: #0ea5e9; color: #fff !important; text-decoration: none;
            padding: 14px 32px; border-radius: 8px; font-size: 16px; font-weight: 600; margin: 8px 0 24px; }}
    .footer {{ padding: 24px 32px; border-top: 1px solid #1f2937; font-size: 13px; color: #6b7280; }}
    .link {{ word-break: break-all; color: #0ea5e9; }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>Sutradhar</h1>
      <p>The Third Eye</p>
    </div>
    <div class="body">
      <p>Hi {name},</p>
      <p>Welcome to India Innovates! Please verify your email address to activate your account.</p>
      <p style="text-align:center;">
        <a class="btn" href="{verify_url}">Verify Email</a>
      </p>
      <p>This link expires in <strong>24 hours</strong>. If you didn't sign up, you can safely ignore this email.</p>
    </div>
    <div class="footer">
      Or copy this link into your browser:<br/>
      <span class="link">{verify_url}</span>
    </div>
  </div>
</body>
</html>
"""

    message = (
        f"From: Sutradhar <{SMTP_FROM}>\r\n"
        f"To: {email}\r\n"
        f"Subject: Verify your email — Sutradhar\r\n"
        f"MIME-Version: 1.0\r\n"
        f"Content-Type: text/html; charset=UTF-8\r\n"
        f"\r\n"
        f"{html_body}"
    )

    # Try STARTTLS on port 587 first (more reliable), fall back to SSL on 465
    try:
        with smtplib.SMTP(SMTP_HOST, 587, timeout=15) as server:
            server.ehlo()
            server.starttls(context=ssl.create_default_context())
            server.ehlo()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_FROM, email, message.encode("utf-8"))
    except Exception:
        # Fall back to SSL with relaxed verification (handles cert chain issues)
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        with smtplib.SMTP_SSL(SMTP_HOST, int(SMTP_PORT), context=context) as server:
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_FROM, email, message.encode("utf-8"))
