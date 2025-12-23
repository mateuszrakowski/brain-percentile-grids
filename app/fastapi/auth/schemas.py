"""Pydantic schemas for authentication."""

from datetime import datetime

from pydantic import BaseModel, Field


class Token(BaseModel):
    """
    JWT token response schema.

    Attributes
    ----------
    access_token : str
        The JWT access token.
    token_type : str
        The type of token (always "bearer").
    """

    access_token: str
    token_type: str


class TokenData(BaseModel):
    """
    Data extracted from JWT token.

    Attributes
    ----------
    username : str | None
        The username from the token's "sub" claim.
    """

    username: str | None = None


class UserCreate(BaseModel):
    """
    Schema for user registration.

    Attributes
    ----------
    username : str
        The desired username (3-50 characters).
    password : str
        The password (minimum 8 characters).
    """

    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)


class UserResponse(BaseModel):
    """
    Schema for user data in responses.

    Attributes
    ----------
    id : int
        The user's ID.
    username : str
        The user's username.
    created_at : datetime
        When the user was created.
    """

    id: int
    username: str
    created_at: datetime

    model_config = {"from_attributes": True}
