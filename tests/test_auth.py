from datetime import UTC, datetime, timedelta

import jwt

from app.fastapi.auth.schemas import UserCreate
from app.fastapi.auth.security import create_access_token

ALGORITHM = "HS256"
FIXED_TIME = datetime(2035, 1, 1, 12, 0, 0, tzinfo=UTC)


class MockDatetime:
    @classmethod
    def now(cls, timezone):
        return FIXED_TIME


class TestCreateAccessToken:
    def test_valid_token(self, test_settings, monkeypatch):
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


class TestRegisterUser:
    def test_success(self, client):
        user = UserCreate(username="username", password="password")
        response = client.post("/api/auth/register", json=user.model_dump())
        assert response.status_code == 201

    def test_duplicated(self, client, test_user):
        user = UserCreate(username="test-username", password="test-password")
        response = client.post("/api/auth/register", json=user.model_dump())
        assert response.status_code == 400


class TestLogin:
    def test_success(self, client, test_user):
        login_response = client.post(
            "/api/auth/token",
            data={"username": test_user.username, "password": test_user.password},
        )

        assert login_response.status_code == 200

    def test_invalid_user(self, client, test_user):
        login_response = client.post(
            "/api/auth/token",
            data={"username": "invalid-username", "password": test_user.password},
        )

        assert login_response.status_code == 401

    def test_invalid_password(self, client, test_user):
        login_response = client.post(
            "/api/auth/token",
            data={"username": test_user.username, "password": "invalid-password"},
        )

        assert login_response.status_code == 401

    def test_missing_user(self, client):
        login_response = client.post(
            "/api/auth/token",
            data={"username": "not-in-database", "password": "not-in-database"},
        )

        assert login_response.status_code == 401


class TestGetCurrentUser:
    def test_valid_token(self, client, test_user):
        login_response = client.post(
            "/api/auth/token",
            data={"username": test_user.username, "password": test_user.password},
        )

        assert login_response.status_code == 200

        response = client.get(
            "/api/auth/me",
            headers={
                "Authorization": f"Bearer {login_response.json()['access_token']}"
            },
        )

        assert response.status_code == 200
        assert response.json()["username"] == test_user.username

    def test_invalid_token(self, client):
        response = client.get(
            "/api/auth/me",
            headers={"Authorization": "Bearer InvalidToken"},
        )

        assert response.status_code == 401

    def test_missing_header(self, client):
        response = client.get("/api/auth/me")

        assert response.status_code == 401
