import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from ideagen.core.config import get_settings
from ideagen.db.models import Base
import os

@pytest.fixture(scope="session")
def engine():
    db_url = os.getenv("IDEAGEN_DATABASE_URL", get_settings().database_url)
    if not db_url:
        pytest.skip("IDEAGEN_DATABASE_URL is not configured; skipping PostgreSQL integration tests.")
    eng = create_engine(db_url)
    Base.metadata.create_all(eng)
    yield eng

@pytest.fixture
def session(engine):
    connection = engine.connect()
    transaction = connection.begin()
    session_maker = sessionmaker(bind=connection, join_transaction_mode="create_savepoint")
    session = session_maker()
    
    yield session
    
    session.close()
    transaction.rollback()
    connection.close()
