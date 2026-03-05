"""Unit tests for authentication utilities.

Tests create_access_token, verify_password, get_password_hash, and
get_current_user. Token creation is tested by mocking datetime and
settings. Password functions are tested via roundtrip verification.
get_current_user is tested in isolation against all branches.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

import jwt
import pytest
from fastapi import HTTPException
from sqlmodel import Session

from app.fastapi.auth.dependencies import get_current_user
from app.fastapi.auth.security import (
    ALGORITHM,
    create_access_token,
    get_password_hash,
    verify_password,
)
from app.fastapi.db.models import User

FIXED_TIME = datetime(2035, 1, 1, 12, 0, 0, tzinfo=UTC)


class MockDatetime:
    @classmethod
    def now(cls, timezone):
        return FIXED_TIME


class TestCreateAccessToken:
    def test_valid_token(self, test_settings, monkeypatch):
        """Verify JWT contains correct subject claim and expiration time."""
        monkeypatch.setattr("app.fastapi.auth.security.datetime", MockDatetime)
        monkeypatch.setattr(
            "app.fastapi.auth.security.get_settings", lambda: test_settings
        )
        username = {"sub": "test_username"}

        access_token = create_access_token(username)

        decoded_token = jwt.decode(
            access_token, test_settings.secret_key, algorithms=[ALGORITHM]
        )

        expiration_time = (
            MockDatetime.now(UTC)
            + timedelta(minutes=test_settings.access_token_expire_minutes)
        ).timestamp()

        assert int(expiration_time) == decoded_token["exp"]
        assert decoded_token["sub"] == username["sub"]


class TestVerifyPassword:
    """Tests for password hashing and verification roundtrip."""

    def test_correct_password_verifies(self):
        """Verify hashing then verifying the same password succeeds."""
        hashed = get_password_hash("test-password-123")
        assert verify_password("test-password-123", hashed) is True

    def test_wrong_password_fails(self):
        """Verify a wrong password does not pass verification."""
        hashed = get_password_hash("test-password-123")
        assert verify_password("wrong-password", hashed) is False

    def test_hash_is_not_deterministic(self):
        """Verify two hashes of the same password differ (salted)."""
        hash1 = get_password_hash("same-password")
        hash2 = get_password_hash("same-password")
        assert hash1 != hash2


class TestGetCurrentUser:
    """Tests for the get_current_user authentication dependency."""

    def _make_token(self, payload: dict, secret: str) -> str:
        """Create a JWT token with the given payload."""
        return jwt.encode(payload, secret, algorithm=ALGORITHM)

    async def test_valid_token_returns_user(
        self, test_settings, test_session, monkeypatch
    ):
        """Verify a valid token with an existing user returns the User object."""
        monkeypatch.setattr(
            "app.fastapi.auth.dependencies.get_settings", lambda: test_settings
        )

        user = User(username="auth-test-user", hashed_password="hashed")
        test_session.add(user)
        test_session.commit()
        test_session.refresh(user)

        token = self._make_token(
            {"sub": "auth-test-user", "exp": datetime(2099, 1, 1, tzinfo=UTC)},
            test_settings.secret_key,
        )

        result = await get_current_user(token=token, session=test_session)
        assert result.username == "auth-test-user"
        assert result.id == user.id

    async def test_expired_token_raises_401(self, test_settings, test_session, monkeypatch):
        """Verify an expired token raises 401 Unauthorized."""
        monkeypatch.setattr(
            "app.fastapi.auth.dependencies.get_settings", lambda: test_settings
        )

        token = self._make_token(
            {"sub": "user", "exp": datetime(2000, 1, 1, tzinfo=UTC)},
            test_settings.secret_key,
        )

        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(token=token, session=test_session)
        assert exc_info.value.status_code == 401

    async def test_malformed_token_raises_401(self, test_settings, test_session, monkeypatch):
        """Verify a malformed token string raises 401 Unauthorized."""
        monkeypatch.setattr(
            "app.fastapi.auth.dependencies.get_settings", lambda: test_settings
        )

        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(token="not-a-valid-jwt", session=test_session)
        assert exc_info.value.status_code == 401

    async def test_missing_sub_claim_raises_401(
        self, test_settings, test_session, monkeypatch
    ):
        """Verify a token without the 'sub' claim raises 401 Unauthorized."""
        monkeypatch.setattr(
            "app.fastapi.auth.dependencies.get_settings", lambda: test_settings
        )

        token = self._make_token(
            {"exp": datetime(2099, 1, 1, tzinfo=UTC)},
            test_settings.secret_key,
        )

        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(token=token, session=test_session)
        assert exc_info.value.status_code == 401

    async def test_nonexistent_user_raises_401(
        self, test_settings, test_session, monkeypatch
    ):
        """Verify a token for a user not in the database raises 401."""
        monkeypatch.setattr(
            "app.fastapi.auth.dependencies.get_settings", lambda: test_settings
        )

        token = self._make_token(
            {"sub": "ghost-user", "exp": datetime(2099, 1, 1, tzinfo=UTC)},
            test_settings.secret_key,
        )

        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(token=token, session=test_session)
        assert exc_info.value.status_code == 401
