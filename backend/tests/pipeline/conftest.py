import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from ideagen.core.config import get_settings
from ideagen.db.models import Base
import os

DB_URL = os.getenv("IDEAGEN_DATABASE_URL", get_settings().database_url)
if not DB_URL:
    raise ValueError("IDEAGEN_DATABASE_URL environment variable is required to run database integration tests.")

@pytest.fixture(scope="session")
def engine():
    engine = create_engine(DB_URL)
    Base.metadata.create_all(engine)
    yield engine

@pytest.fixture
def session(engine):
    connection = engine.connect()
    transaction = connection.begin()
    session_maker = sessionmaker(bind=connection)
    session = session_maker()
    
    yield session
    
    session.close()
    transaction.rollback()
    connection.close()
