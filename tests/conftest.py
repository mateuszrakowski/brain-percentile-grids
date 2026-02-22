import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from app.fastapi.config import Settings


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
