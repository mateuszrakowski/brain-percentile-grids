from collections.abc import Generator

from sqlmodel import Session, SQLModel, create_engine

from app.fastapi.config import get_settings

settings = get_settings()

engine = create_engine(settings.db_url, echo=settings.debug)


def init_db() -> None:
    """Create all tables. Safe to call multiple times - only creates if not exists."""
    SQLModel.metadata.create_all(engine)


def get_session() -> Generator[Session, None, None]:
    """Dependency for getting DB session in endpoints."""
    with Session(engine) as session:
        yield session
