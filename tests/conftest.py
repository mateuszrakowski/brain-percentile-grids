import pandas as pd
import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from app.fastapi.config import Settings
from app.fastapi.db.models import ReferenceDataset, User
from app.fastapi.services.reference_data import ReferenceDataService


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
def test_dataset_db(test_session):
    user = User(username="test", hashed_password="hashed")
    test_session.add(user)
    test_session.flush()

    dataset = ReferenceDataset(user_id=user.id, name="test_dataset")
    test_session.add(dataset)
    test_session.flush()

    return dataset


@pytest.fixture
def test_reference_dataset(test_dataset_db, test_session):
    service = ReferenceDataService(test_session)
    service.save_reference_data(
        dataset_id=test_dataset_db.id,
        dataframes=[
            pd.DataFrame(
                {
                    "PatientID": ["p1", "p2"],
                    "AgeYears": [25, 35],
                    "StudyDate": ["2024-01-01", "2024-01-02"],
                    "StudyDescription": ["scan1", "scan2"],
                    "hippo": [0.5, 0.6],
                }
            )
        ],
    )

    return test_dataset_db
