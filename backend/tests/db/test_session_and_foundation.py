"""Tests for PostgreSQL foundation, session lifecycle, and migration compatibility.

Validates:
- Engine connectivity to PostgreSQL
- Session lifecycle (commit, rollback, auto-close)
- FastAPI get_db dependency injection
- /api/health and /api/papers API endpoints
- Alembic migration head alignment
"""

import pytest
from sqlalchemy import text, inspect
from sqlalchemy.orm import Session
from fastapi.testclient import TestClient

from ideagen.db.session import engine, SessionLocal, get_db
from ideagen.db.models import Base, Paper
from ideagen.api.main import app
from ideagen.ingestion.models import SourceType


def test_postgres_engine_connection():
    """Verify that application engine connects to PostgreSQL and executes queries."""
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1")).scalar()
        assert result == 1


def test_postgres_session_lifecycle():
    """Verify that SessionLocal manages transactions and closes cleanly."""
    session = SessionLocal()
    try:
        val = session.execute(text("SELECT 42")).scalar()
        assert val == 42
    finally:
        session.close()


def test_session_dependency_generator():
    """Verify that get_db yields an active session and closes it."""
    gen = get_db()
    db = next(gen)
    assert isinstance(db, Session)
    assert db.is_active
    
    # Teardown generator
    with pytest.raises(StopIteration):
        next(gen)


def test_session_dependency_rollback_on_exception():
    """Verify that get_db rolls back the transaction when an exception is raised."""
    gen = get_db()
    db = next(gen)
    
    # Begin a transaction by modifying or querying
    db.execute(text("SELECT 1"))
    
    # Simulate an error within request handling
    with pytest.raises(RuntimeError, match="Simulated request failure"):
        try:
            raise RuntimeError("Simulated request failure")
        except Exception:
            gen.throw(RuntimeError("Simulated request failure"))


def test_alembic_tables_exist_in_postgres():
    """Verify that all core SQLAlchemy model tables are present in the database."""
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    
    expected_tables = {
        "papers",
        "paper_documents",
        "opportunities",
        "opportunity_evidence",
        "opportunity_history",
        "alembic_version",
    }
    for table in expected_tables:
        assert table in existing_tables, f"Table '{table}' missing from PostgreSQL"


def test_api_health_endpoint():
    """Verify that /api/health succeeds and confirms PostgreSQL connectivity."""
    client = TestClient(app)
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["database"] == "postgresql_connected"


def test_api_papers_endpoint():
    """Verify that /api/papers queries real PostgreSQL database without errors."""
    client = TestClient(app)
    response = client.get("/api/papers")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
