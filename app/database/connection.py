"""
Database connection management for JobComparisonAI.
"""

from collections.abc import Generator

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings


class DatabaseConfigurationError(RuntimeError):
    """Raised when required database configuration is missing."""


def _get_database_url() -> str:
    """
    Retrieve and validate DATABASE_URL from application settings.

    Returns:
        The configured PostgreSQL DATABASE_URL.

    Raises:
        DatabaseConfigurationError:
            If DATABASE_URL is missing or empty.
    """
    database_url = getattr(settings, "DATABASE_URL", None)

    if not database_url:
        raise DatabaseConfigurationError(
            "DATABASE_URL is not configured. "
            "Please define DATABASE_URL in your .env file."
        )

    return database_url

engine: Engine = create_engine(
    _get_database_url(),
    pool_pre_ping=True,
)


class Base(DeclarativeBase):
    """
    Base class for all SQLAlchemy ORM models.
    """

    pass

SessionLocal: sessionmaker[Session] = sessionmaker(
    bind=engine,
    autoflush=False,
    expire_on_commit=False,
)

def get_db() -> Generator[Session, None, None]:
    """
    Provide a database session and guarantee that it is closed.

    Yields:
        Session: Active SQLAlchemy database session.
    """
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()