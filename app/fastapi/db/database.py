import logging
from collections.abc import Iterator

from sqlalchemy import text
from sqlmodel import Session, SQLModel, create_engine

from app.fastapi.config import get_settings

logger = logging.getLogger(__name__)

_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_engine(settings.db_url, echo=settings.debug)
    return _engine


def _migrate_add_column(
    session: Session, table: str, column: str, col_type: str
) -> None:
    """
    Add a column to an existing SQLite table if it doesn't exist.

    Parameters
    ----------
    session : Session
        Active database session.
    table : str
        Table name.
    column : str
        Column name to add.
    col_type : str
        SQLite column type (e.g. 'TEXT').
    """
    result = session.exec(text(f"PRAGMA table_info({table})"))
    columns = {row[1] for row in result}
    if column not in columns:
        session.exec(text(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}"))
        session.commit()
        logger.info(f"Migration: added {column} to {table}")


def init_db() -> None:
    """Create all tables and run lightweight migrations."""
    engine = _get_engine()
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        _migrate_add_column(session, "fittedmodel", "reference_plot_path", "TEXT")


def get_session() -> Iterator[Session]:
    """Dependency for getting DB session in endpoints."""
    with Session(_get_engine()) as session:
        yield session
