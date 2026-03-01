"""Unit tests for JWT token creation.

Tests the create_access_token function in isolation by mocking
datetime and settings. Verifies that the token contains the correct
subject claim and expiration time.
"""

from datetime import UTC, datetime, timedelta

import jwt

from app.fastapi.auth.security import ALGORITHM, create_access_token

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
