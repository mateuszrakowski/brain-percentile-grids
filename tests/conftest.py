import pytest
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

from app.fastapi.auth.schemas import UserCreate
from app.fastapi.config import Settings, get_settings
from app.fastapi.db.database import get_session
from app.fastapi.db.models import User
from app.fastapi.main import app


class TestSettings(Settings):
    debug: bool = True
    environment: str = "test"
    db_url: str = "sqlite:///:memory:"
    secret_key: str = "test-secret"
    access_token_expire_minutes: int = 30


@pytest.fixture(scope="session")
def test_settings(tmp_path_factory):
    settings = TestSettings()

    settings.upload_folder = str(tmp_path_factory.mktemp("upload"))
    settings.models_dir = str(tmp_path_factory.mktemp("models"))

    return settings


@pytest.fixture
def test_engine(test_settings):
    engine = create_engine(
        test_settings.db_url,
        echo=test_settings.debug,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )  # StaticPool is needed for single table connection
    SQLModel.metadata.create_all(engine)

    return engine


@pytest.fixture
def test_session(test_engine):
    with Session(test_engine) as session:
        yield session


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
    assert response.status_code == 201

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

    assert response.json()["name"] == "test-dataset"
    assert response.status_code == 200

    return response.json()
