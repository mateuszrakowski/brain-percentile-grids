from datetime import UTC, datetime, timedelta

import jwt
import pytest

from app.fastapi.auth.security import create_access_token
from app.fastapi.config import get_settings

ALGORITHM = "HS256"
SETTINGS = get_settings()
FIXED_TIME = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
def user_data() -> dict:
    return {"sub": "test_username"}


class MockDatetime:
    @classmethod
    def now(cls, timezone):
        return FIXED_TIME


def test_create_access_token(monkeypatch, user_data):
    monkeypatch.setattr("app.fastapi.auth.security.datetime", MockDatetime)
    access_token = create_access_token(user_data)

    decoded_token = jwt.decode(
        access_token, SETTINGS.secret_key, algorithms=[ALGORITHM]
    )

    expiration_time = (
        datetime.now(UTC) + timedelta(minutes=SETTINGS.access_token_expire_minutes)
    ).timestamp()

    assert int(expiration_time) == decoded_token["exp"]
    assert decoded_token["sub"] == user_data["sub"]
