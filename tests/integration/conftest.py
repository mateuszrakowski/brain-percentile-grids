import pytest
from fastapi.testclient import TestClient
from sqlmodel import select

from app.fastapi.auth.schemas import UserCreate
from app.fastapi.config import get_settings
from app.fastapi.db.database import get_session
from app.fastapi.db.models import User
from app.fastapi.main import app


@pytest.fixture
def client(test_settings, test_session):
    app.dependency_overrides[get_settings] = lambda: test_settings
    app.dependency_overrides[get_session] = lambda: test_session
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def test_user(client):
    user = UserCreate(username="test-username", password="test-password")
    response = client.post("/api/auth/register", json=user.model_dump())
    if response.status_code != 201:
        raise RuntimeError(
            f"Fixture setup failed: registration returned {response.status_code}: "
            f"{response.text}"
        )

    return user


@pytest.fixture
def test_user_db(test_session, test_user):
    return test_session.exec(
        select(User).where(User.username == test_user.username)
    ).first()


@pytest.fixture
def test_user_token(client, test_user):
    response = client.post(
        "/api/auth/token",
        data={"username": test_user.username, "password": test_user.password},
    )
    return response.json()["access_token"]


@pytest.fixture
def test_dataset(client, test_user_token):
    response = client.post(
        "/api/datasets",
        json={"name": "test-dataset"},
        headers={"Authorization": f"Bearer {test_user_token}"},
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"Fixture setup failed: dataset creation returned "
            f"{response.status_code}: {response.text}"
        )

    return response.json()
