# app/auth/token_utils.py
import jwt
from datetime import datetime, timedelta, timezone
from typing import Optional

SECRET_KEY = "supersecretkey"  # <--- move to env var in prod
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 1 day

def create_access_token(sub: str, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create JWT whose payload is minimal: {"sub": <filename>, "exp": ...}
    """
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    payload = {"sub": sub, "exp": expire}
    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    # PyJWT may return bytes on older versions; ensure str
    if isinstance(token, bytes):
        token = token.decode("utf-8")
    return token

def decode_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload  # e.g. {"sub": "S001_48s.edf", "exp": 169xxx}
    except jwt.ExpiredSignatureError:
        print("❌ Token expired")
        return None
    except jwt.InvalidTokenError as e:
        print("❌ Invalid token:", e)
        return None
