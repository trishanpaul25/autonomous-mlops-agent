#this file is responsible for: creating JWT tokens, decoding JWT tokens

from datetime import datetime, timedelta, timezone
from typing import Any

from jose import JWTError, jwt

from server.core.config import settings


def create_access_token(
    data: dict[str, Any],
    expires_delta: timedelta | None = None,
) -> str:
    """
    Create a signed JWT access token.
    """

    payload = data.copy()

    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )

    payload.update({"exp": expire})

    return jwt.encode(
        payload,
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM,
    )


def decode_access_token(token: str) -> dict[str, Any] | None:
    """
    Decode and validate a JWT token.

    Returns:
        Payload dictionary if valid.
        None if invalid or expired.
    """

    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM],
        )

        return payload

    except JWTError:
        return None