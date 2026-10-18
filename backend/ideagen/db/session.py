"""SQLAlchemy database engine and session management.

Provides an application-level engine, session factory, and a request-scoped
FastAPI dependency with automatic rollback and closing.
"""

from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from ideagen.core.config import get_settings

settings = get_settings()

_engine = None
_SessionLocal = None


def get_engine():
    """Return the application-level SQLAlchemy engine, initialized lazily."""
    global _engine
    if _engine is None:
        cfg = get_settings()
        if not cfg.database_url:
            raise ValueError(
                "IDEAGEN_DATABASE_URL environment variable is not configured. "
                "A valid PostgreSQL connection string is required."
            )
        _engine = create_engine(
            cfg.database_url,
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20,
        )
    return _engine


def get_session_factory():
    """Return the application-level sessionmaker factory, initialized lazily."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=get_engine(),
        )
    return _SessionLocal


def __getattr__(name: str):
    """Provide lazy attribute access for engine and SessionLocal."""
    if name == "engine":
        return get_engine()
    if name == "SessionLocal":
        return get_session_factory()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_db() -> Generator[Session, None, None]:
    """Request-scoped database session dependency for FastAPI.

    Yields a SQLAlchemy Session, rolling back uncommitted changes
    if an unhandled exception occurs, and always closing the session.
    """
    session_factory = get_session_factory()
    db = session_factory()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
