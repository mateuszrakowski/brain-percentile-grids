"""Authentication dependencies for FastAPI endpoints."""

from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jwt.exceptions import InvalidTokenError
from sqlmodel import Session, select

from app.fastapi.config import get_settings
from app.fastapi.db.database import get_session
from app.fastapi.db.models import User

from .schemas import TokenData
from .security import ALGORITHM

# OAuth2 scheme - extracts token from Authorization header
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")


async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    session: Session = Depends(get_session),
) -> User:
    """
    Validate JWT token and return the current user.

    This dependency:
    1. Extracts the token from the Authorization header
    2. Decodes and validates the JWT
    3. Retrieves the user from the database

    Parameters
    ----------
    token : str
        The JWT token extracted from the Authorization header.
    session : Session
        The database session.

    Returns
    -------
    User
        The authenticated user.

    Raises
    ------
    HTTPException
        401 Unauthorized if the token is invalid or user not found.
    """
    settings = get_settings()

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        # Decode and validate the token
        payload = jwt.decode(token, settings.secret_key, algorithms=[ALGORITHM])
        username: str | None = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except InvalidTokenError:
        raise credentials_exception from None

    # Get user from database
    statement = select(User).where(User.username == token_data.username)
    user = session.exec(statement).first()

    if user is None:
        raise credentials_exception

    return user
