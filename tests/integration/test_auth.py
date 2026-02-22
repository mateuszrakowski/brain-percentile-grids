"""Integration tests for authentication endpoints.

Tests the full auth flow through the HTTP client: registration,
login (OAuth2 password flow), and the /me endpoint. Verifies
correct status codes for valid and invalid credentials.
"""

from app.fastapi.auth.schemas import UserCreate


class TestRegisterUser:
    """Tests for POST /api/auth/register."""
    def test_success(self, client):
        user = UserCreate(username="username", password="password")
        response = client.post("/api/auth/register", json=user.model_dump())
        assert response.status_code == 201

    def test_duplicated(self, client, test_user):
        user = UserCreate(username="test-username", password="test-password")
        response = client.post("/api/auth/register", json=user.model_dump())
        assert response.status_code == 400


class TestLogin:
    """Tests for POST /api/auth/token (OAuth2 password flow)."""
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
    """Tests for GET /api/auth/me (token-protected endpoint)."""
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
