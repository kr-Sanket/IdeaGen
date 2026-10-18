import pytest
from pathlib import Path
from datetime import date
import pymupdf

from ideagen.ingestion.pdf import ManualPDFIngestionService, PDFExtractionError
from ideagen.db.persistence import SQLAlchemyPersistenceLayer
from ideagen.ingestion.models import SourceType
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from ideagen.core.config import get_settings
from ideagen.db.models import Base

settings = get_settings()
DB_URL = settings.database_url

@pytest.fixture(scope="session")
def engine():
    if not DB_URL:
        pytest.skip("IDEAGEN_DATABASE_URL is not configured; skipping PDF persistence tests.")
    eng = create_engine(DB_URL)
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

@pytest.fixture
def test_pdfs_dir(tmp_path):
    # 1. Normal text PDF
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 50), "This is a normal scientific paper test.\n" * 50)
    normal_pdf = tmp_path / "normal.pdf"
    doc.save(str(normal_pdf))
    doc.close()

    # 2. Multi-page PDF
    doc = pymupdf.open()
    page1 = doc.new_page()
    page1.insert_text((50, 50), "Page 1 content.\n" * 20)
    page2 = doc.new_page()
    page2.insert_text((50, 50), "Page 2 content.\n" * 20)
    multi_pdf = tmp_path / "multi.pdf"
    doc.save(str(multi_pdf))
    doc.close()

    # 3. Empty PDF (too short for threshold)
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Short.")
    empty_pdf = tmp_path / "empty.pdf"
    doc.save(str(empty_pdf))
    doc.close()

    # 4. Malformed PDF (not a PDF)
    malformed_pdf = tmp_path / "malformed.pdf"
    malformed_pdf.write_text("This is not a PDF file format.", encoding="utf-8")

    return {
        "normal": normal_pdf,
        "multi": multi_pdf,
        "empty": empty_pdf,
        "malformed": malformed_pdf,
    }

def test_valid_pdf_extracts_text(session, test_pdfs_dir):
    layer = SQLAlchemyPersistenceLayer(session)
    service = ManualPDFIngestionService(persistence=layer)
    
    paper_dto, doc_dto = service.ingest_pdf(
        pdf_path=test_pdfs_dir["normal"],
        title="Normal Paper",
        source_id="normal123"
    )
    
    assert paper_dto.title == "Normal Paper"
    assert paper_dto.source_type == SourceType.MANUAL_UPLOAD
    assert paper_dto.source_id == "normal123"
    
    assert doc_dto is not None
    assert "This is a normal scientific paper test." in doc_dto.full_text
    assert len(doc_dto.full_text) >= 500

def test_multipage_pdf_extracts_text(session, test_pdfs_dir):
    layer = SQLAlchemyPersistenceLayer(session)
    service = ManualPDFIngestionService(persistence=layer)
    
    paper_dto, doc_dto = service.ingest_pdf(
        pdf_path=test_pdfs_dir["multi"],
        title="Multi Paper",
        source_id="multi123"
    )
    
    assert "Page 1 content." in doc_dto.full_text
    assert "Page 2 content." in doc_dto.full_text

def test_malformed_non_pdf_rejected(session, test_pdfs_dir):
    layer = SQLAlchemyPersistenceLayer(session)
    service = ManualPDFIngestionService(persistence=layer)
    
    with pytest.raises(PDFExtractionError, match="Failed to open PDF file"):
        service.ingest_pdf(
            pdf_path=test_pdfs_dir["malformed"],
            title="Malformed Paper"
        )

def test_empty_pdf_rejected(session, test_pdfs_dir):
    layer = SQLAlchemyPersistenceLayer(session)
    service = ManualPDFIngestionService(persistence=layer)
    
    with pytest.raises(PDFExtractionError, match="below the minimum threshold"):
        service.ingest_pdf(
            pdf_path=test_pdfs_dir["empty"],
            title="Empty Paper"
        )

def test_duplicate_handling_follows_rules(session, test_pdfs_dir):
    layer = SQLAlchemyPersistenceLayer(session)
    service = ManualPDFIngestionService(persistence=layer)
    
    # Ingest once
    service.ingest_pdf(
        pdf_path=test_pdfs_dir["normal"],
        title="Normal Paper",
        source_id="dup123"
    )
    
    # Ingest again with same source_id (which falls to IntegrityError due to constraint)
    with pytest.raises(ValueError, match="IntegrityError"):
        service.ingest_pdf(
            pdf_path=test_pdfs_dir["normal"],
            title="Normal Paper Dup",
            source_id="dup123"
        )

def test_doi_metadata_preserved(session, test_pdfs_dir):
    layer = SQLAlchemyPersistenceLayer(session)
    service = ManualPDFIngestionService(persistence=layer)
    
    paper_dto, doc_dto = service.ingest_pdf(
        pdf_path=test_pdfs_dir["normal"],
        title="DOI Paper",
        source_id="doi123",
        doi="10.1234/test"
    )
    
    assert paper_dto.doi == "10.1234/test"
    
    # Missing DOI remains NULL
    paper_dto2, doc_dto2 = service.ingest_pdf(
        pdf_path=test_pdfs_dir["normal"],
        title="No DOI Paper",
        source_id="nodoi123",
        doi=None
    )
    assert paper_dto2.doi is None

def test_publication_date_handling(session, test_pdfs_dir):
    layer = SQLAlchemyPersistenceLayer(session)
    service = ManualPDFIngestionService(persistence=layer)
    
    # Supplied date preserved
    test_date = date(2025, 1, 1)
    paper_dto, doc_dto = service.ingest_pdf(
        pdf_path=test_pdfs_dir["normal"],
        title="Date Paper",
        source_id="date123",
        publication_date=test_date
    )
    assert paper_dto.publication_date == test_date
    
    from ideagen.db.models import Paper, PaperDocument
    db_paper1 = session.query(Paper).filter_by(source_id="date123").first()
    assert db_paper1.publication_date == test_date
    
    # Omitted date remains NULL
    paper_dto2, doc_dto2 = service.ingest_pdf(
        pdf_path=test_pdfs_dir["normal"],
        title="No Date Paper",
        source_id="nodate123",
        publication_date=None
    )
    assert paper_dto2.publication_date is None
    
    db_paper2 = session.query(Paper).filter_by(source_id="nodate123").first()
    assert db_paper2.publication_date is None
    
    # Extracted_at is populated
    db_doc2 = session.query(PaperDocument).filter_by(paper_id=db_paper2.id).first()
    assert db_doc2.extracted_at is not None

def test_persistence_verification(session, test_pdfs_dir):
    layer = SQLAlchemyPersistenceLayer(session)
    service = ManualPDFIngestionService(persistence=layer)
    
    paper_dto, doc_dto = service.ingest_pdf(
        pdf_path=test_pdfs_dir["normal"],
        title="Persistence Paper",
        source_id="pers123"
    )
    
    from ideagen.db.models import Paper, PaperDocument
    db_paper = session.query(Paper).filter_by(source_id="pers123").first()
    assert db_paper is not None
    assert db_paper.title == "Persistence Paper"
    
    db_doc = session.query(PaperDocument).filter_by(paper_id=db_paper.id).first()
    assert db_doc is not None
    assert db_doc.extraction_method == "pymupdf"
    assert db_doc.extraction_version == pymupdf.__version__
    assert db_doc.extracted_at is not None
    assert len(db_doc.full_text) >= 500
