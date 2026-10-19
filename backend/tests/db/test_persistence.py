import pytest
import os
from datetime import date
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from ideagen.db.models import Base
from ideagen.db.persistence import SQLAlchemyPersistenceLayer
from ideagen.ingestion.models import PaperCreateDTO, SourceType

from ideagen.core.config import get_settings

settings = get_settings()

@pytest.fixture(scope="session")
def engine():
    db_url = settings.database_url
    if not db_url:
        pytest.skip("IDEAGEN_DATABASE_URL is not configured; skipping persistence tests.")
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

def test_insert_valid_paper(session):
    layer = SQLAlchemyPersistenceLayer(session)
    
    paper = PaperCreateDTO(
        source_type=SourceType.OPENALEX_API,
        source_id="W123",
        doi="10.123/test",
        title="Valid Paper",
        publication_date=date(2020, 1, 1),
        batch_id="Batch 0",
        abstract="Test",
        raw_metadata={}
    )
    
    layer.save_paper(paper)
    
    assert layer.paper_exists_by_source_id(SourceType.OPENALEX_API, "W123")
    assert layer.paper_exists_by_doi("10.123/test")

def test_duplicate_openalex_identity(session):
    layer = SQLAlchemyPersistenceLayer(session)
    
    paper1 = PaperCreateDTO(
        source_type=SourceType.OPENALEX_API,
        source_id="W123",
        title="Valid Paper",
        publication_date=date(2020, 1, 1),
        batch_id="Batch 0",
        abstract="Test",
        raw_metadata={}
    )
    layer.save_paper(paper1)
    
    paper2 = PaperCreateDTO(
        source_type=SourceType.OPENALEX_SNAPSHOT,
        source_id="W123",
        title="Valid Paper",
        publication_date=date(2020, 1, 1),
        batch_id="Batch 0",
        abstract="Test",
        raw_metadata={}
    )
    
    with pytest.raises(ValueError, match="IntegrityError"):
        layer.save_paper(paper2)

def test_transaction_rollback_on_failure(session):
    layer = SQLAlchemyPersistenceLayer(session)
    
    paper = PaperCreateDTO(
        source_type=SourceType.OPENALEX_API,
        source_id="W123", # Duplicate ID
        title="Valid Paper",
        publication_date=date(2020, 1, 1),
        batch_id="Batch 0",
        abstract="Test",
        raw_metadata={}
    )
    layer.save_paper(paper)
    
    with pytest.raises(ValueError):
        layer.save_paper(paper)
        
    # Session should be rolled back and usable again
    paper3 = PaperCreateDTO(
        source_type=SourceType.OPENALEX_API,
        source_id="W456",
        title="Valid Paper",
        publication_date=date(2020, 1, 1),
        batch_id="Batch 0",
        abstract="Test",
        raw_metadata={}
    )
    layer.save_paper(paper3)
    assert layer.paper_exists_by_source_id(SourceType.OPENALEX_API, "W456")

def test_insert_paper_document(session):
    layer = SQLAlchemyPersistenceLayer(session)
    from datetime import datetime
    
    # 1. create Paper
    paper = PaperCreateDTO(
        source_type=SourceType.MANUAL_UPLOAD,
        source_id="TEST_DOC_1",
        title="Test Document",
        publication_date=date(2023, 1, 1),
        batch_id="Batch 1",
        abstract="Test Abstract",
        raw_metadata={}
    )
    layer.save_paper(paper)
    
    # Need the UUID of the inserted paper
    from ideagen.db.models import Paper, PaperDocument
    db_paper = session.query(Paper).filter_by(source_id="TEST_DOC_1").first()
    assert db_paper is not None
    
    # 2. create PaperDocument linked to Paper
    from ideagen.ingestion.models import PaperDocumentCreateDTO
    doc_dto = PaperDocumentCreateDTO(
        paper_id=db_paper.id,
        full_text="This is the full text of the document.\n" * 100,
        extraction_method="pdf_test",
        extraction_version="v1.0",
        extracted_at=datetime.utcnow()
    )
    layer.save_paper_document(doc_dto)
    
    # 3. retrieve PaperDocument and verify
    db_doc = session.query(PaperDocument).filter_by(paper_id=db_paper.id).first()
    assert db_doc is not None
    assert db_doc.full_text == "This is the full text of the document.\n" * 100
    assert db_doc.extraction_method == "pdf_test"
    assert db_doc.extraction_version == "v1.0"
    
    # 6. foreign-key relationship works
    assert len(db_paper.documents) == 1
    assert db_paper.documents[0].id == db_doc.id

def test_invalid_paper_id_rejected(session):
    layer = SQLAlchemyPersistenceLayer(session)
    from datetime import datetime
    import uuid
    from ideagen.ingestion.models import PaperDocumentCreateDTO
    
    doc_dto = PaperDocumentCreateDTO(
        paper_id=uuid.uuid4(),  # Does not exist
        full_text="Test",
        extraction_method="pdf_test",
        extraction_version="v1",
        extracted_at=datetime.utcnow()
    )
    
    with pytest.raises(ValueError, match="IntegrityError"):
        layer.save_paper_document(doc_dto)


def test_persistence_type_hints_resolvable():
    """Regression test ensuring get_type_hints() resolves cleanly on Python 3.11-3.14."""
    import typing
    hints_passages = typing.get_type_hints(SQLAlchemyPersistenceLayer.get_all_passages)
    assert "cutoff_date" in hints_passages
    assert "return" in hints_passages

    hints_statements = typing.get_type_hints(SQLAlchemyPersistenceLayer.get_all_research_statements)
    assert "cutoff_date" in hints_statements
    assert "return" in hints_statements

