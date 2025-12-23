"""Authentication endpoints for user registration and login."""

from datetime import timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlmodel import Session, select

from app.fastapi.auth.dependencies import get_current_user
from app.fastapi.auth.schemas import Token, UserCreate, UserResponse
from app.fastapi.auth.security import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    create_access_token,
    get_password_hash,
    verify_password,
)
from app.fastapi.db.database import get_session
from app.fastapi.db.models import User

router = APIRouter(prefix="/api/auth", tags=["authentication"])


@router.post(
    "/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED
)
async def register(
    user_data: UserCreate,
    session: Session = Depends(get_session),
) -> User:
    """
    Register a new user.

    Parameters
    ----------
    user_data : UserCreate
        The user registration data (username, password).
    session : Session
        The database session.

    Returns
    -------
    User
        The created user.

    Raises
    ------
    HTTPException
        400 Bad Request if username already exists.
    """
    # Check if username already exists
    statement = select(User).where(User.username == user_data.username)
    existing_user = session.exec(statement).first()

    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered",
        )

    # Create new user with hashed password
    user = User(
        username=user_data.username,
        hashed_password=get_password_hash(user_data.password),
    )

    session.add(user)
    session.commit()
    session.refresh(user)

    return user


@router.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    session: Session = Depends(get_session),
) -> Token:
    """
    Authenticate user and return JWT access token.

    This endpoint follows OAuth2 password flow:
    - Accepts form data with username and password
    - Returns JWT token if credentials are valid

    Parameters
    ----------
    form_data : OAuth2PasswordRequestForm
        The OAuth2 form data containing username and password.
    session : Session
        The database session.

    Returns
    -------
    Token
        The access token and token type.

    Raises
    ------
    HTTPException
        401 Unauthorized if credentials are invalid.
    """
    # Find user by username
    statement = select(User).where(User.username == form_data.username)
    user = session.exec(statement).first()

    # Verify user exists and password is correct
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=access_token_expires,
    )

    return Token(access_token=access_token, token_type="bearer")


@router.get("/me", response_model=UserResponse)
async def read_users_me(
    current_user: Annotated[User, Depends(get_current_user)],
) -> User:
    """
    Get current authenticated user's information.

    Parameters
    ----------
    current_user : User
        The authenticated user (injected by dependency).

    Returns
    -------
    User
        The current user's information.
    """
    return current_user
