from collections.abc import Iterator

from sqlmodel import Session, SQLModel, create_engine

from app.fastapi.config import get_settings

_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_engine(settings.db_url, echo=settings.debug)
    return _engine


def init_db() -> None:
    """Create all tables. Safe to call multiple times - only creates if not exists."""
    SQLModel.metadata.create_all(_get_engine())


def get_session() -> Iterator[Session]:
    """Dependency for getting DB session in endpoints."""
    with Session(_get_engine()) as session:
        yield session
