"""Security utilities for password hashing and JWT token management."""

from datetime import UTC, datetime, timedelta

import jwt
from pwdlib import PasswordHash

from app.fastapi.config import get_settings

# JWT configuration
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hasher using Argon2
password_hash = PasswordHash.recommended()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plain password against a hashed password.

    Parameters
    ----------
    plain_password : str
        The plain text password to verify.
    hashed_password : str
        The hashed password to verify against.

    Returns
    -------
    bool
        True if the password matches, False otherwise.
    """
    return password_hash.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """
    Hash a password using Argon2.

    Parameters
    ----------
    password : str
        The plain text password to hash.

    Returns
    -------
    str
        The hashed password.
    """
    return password_hash.hash(password)


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    """
    Create a JWT access token.

    Parameters
    ----------
    data : dict
        The data to encode in the token (typically {"sub": username}).
    expires_delta : timedelta | None, optional
        Custom expiration time. If None, uses ACCESS_TOKEN_EXPIRE_MINUTES.

    Returns
    -------
    str
        The encoded JWT token.
    """
    settings = get_settings()

    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(UTC) + expires_delta
    else:
        expire = datetime.now(UTC) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=ALGORITHM)

    return encoded_jwt
