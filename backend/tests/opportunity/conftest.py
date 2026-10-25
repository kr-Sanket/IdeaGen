import os
import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from ideagen.core.config import get_settings
from ideagen.db.models import Base
from ideagen.db.session import configure_sqlite_foreign_keys


@pytest.fixture(scope="session")
def engine():
    db_url = os.getenv("IDEAGEN_DATABASE_URL", get_settings().database_url)
    if db_url:
        eng = create_engine(db_url)
        configure_sqlite_foreign_keys(eng)
    else:
        eng = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        configure_sqlite_foreign_keys(eng)

        @event.listens_for(eng, "connect")
        def do_connect(dbapi_connection, connection_record):
            dbapi_connection.isolation_level = None

        @event.listens_for(eng, "begin")
        def do_begin(conn):
            conn.exec_driver_sql("BEGIN")

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
