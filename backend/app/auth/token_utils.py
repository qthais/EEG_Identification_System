# app/auth/token_utils.py
import jwt
from datetime import datetime, timedelta, timezone
from typing import Optional

SECRET_KEY = "supersecretkey"  # <--- move to env var in prod
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 1 day

def create_access_token(
    user_id: str,
    user_name: Optional[str] = None,
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create JWT with both user_id and user_name.
    Payload includes:
      - sub: unique user_id (standard JWT subject)
      - name: user_name (optional, human-readable)
      - exp: expiration time
    """
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))

    payload = {
        "sub": user_id,         # ✅ standard JWT claim for subject
        "name": user_name,      # ✅ custom field
        "exp": expire
    }

    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

    # Ensure return is string (older PyJWT versions may return bytes)
    if isinstance(token, bytes):
        token = token.decode("utf-8")

    return token


def decode_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload 
    except jwt.ExpiredSignatureError:
        print("❌ Token expired")
        return None
    except jwt.InvalidTokenError as e:
        print("❌ Invalid token:", e)
        return None
