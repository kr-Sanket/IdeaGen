"""SQLAlchemy database engine and session management.

Provides an application-level engine, session factory, and a request-scoped
FastAPI dependency with automatic rollback and closing.
"""

from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from ideagen.core.config import get_settings

settings = get_settings()

if not settings.database_url:
    raise ValueError(
        "IDEAGEN_DATABASE_URL environment variable is not configured. "
        "A valid PostgreSQL connection string is required."
    )

# Application-level engine
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)

# Application-level session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


def get_db() -> Generator[Session, None, None]:
    """Request-scoped database session dependency for FastAPI.

    Yields a SQLAlchemy Session, rolling back uncommitted changes
    if an unhandled exception occurs, and always closing the session.
    """
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
