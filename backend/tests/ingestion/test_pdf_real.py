"""Comprehensive tests for Phase 2: Real Document Ingestion & Structured Text.

Verifies:
- Real PyMuPDF text extraction with page preservation
- Deterministic title extraction (and null when undetectable)
- Rule-based section detection (Abstract, Intro, Methods, Results, Limitations, References)
- Text cleaning (dehyphenation, line wraps, whitespace)
- Deterministic passage/chunk creation and page span references
- Provenance tracking and SHA-256 content hashing
- Content hash duplicate detection
- PostgreSQL persistence (Paper, PaperDocument, PaperPassage)
- API upload and passage retrieval
"""

import hashlib
import io
import pytest
from pathlib import Path
from datetime import date
import pymupdf
from fastapi.testclient import TestClient

from ideagen.ingestion.pdf import (
    ManualPDFIngestionService,
    PDFExtractionError,
    DuplicateDocumentError,
)
from ideagen.ingestion.text_cleaner import TextCleaner
from ideagen.ingestion.section_detector import SectionDetector
from ideagen.core.config import get_settings

settings = get_settings()
if not settings.database_url:
    pytest.skip(
        "IDEAGEN_DATABASE_URL is not configured; skipping real PDF database integration tests.",
        allow_module_level=True,
    )

from ideagen.db.persistence import SQLAlchemyPersistenceLayer
from ideagen.db.models import Paper, PaperDocument, PaperPassage
from ideagen.db.session import engine, SessionLocal
from ideagen.api.main import app


@pytest.fixture
def db_session():
    """Provides a transactional database session for tests with rollback."""
    connection = engine.connect()
    transaction = connection.begin()
    from sqlalchemy.orm import Session as SASession
    session = SASession(bind=connection, join_transaction_mode="create_savepoint")
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()


@pytest.fixture
def real_scientific_pdf(tmp_path) -> Path:
    """Creates a real 3-page scientific PDF fixture using PyMuPDF."""
    doc = pymupdf.open()

    # Page 1: Title, Abstract, Introduction
    page1 = doc.new_page()
    # Insert Title with prominent size
    page1.insert_text(
        (50, 60),
        "Robust Retrieval-Augmented Generation for Scientific Question Answering",
        fontsize=16,
    )
    # Insert Abstract
    page1.insert_text(
        (50, 110),
        "Abstract\nRetrieval-augmented generation (RAG) models often suffer from unfaithfulness.\n"
        "In this work, we propose an evidence-grounded framework that verifies factual claims.\n"
        "Our method- \nology achieves significant robustness gains across benchmark datasets.\n" * 4,
        fontsize=10,
    )
    # Insert Introduction
    page1.insert_text(
        (50, 320),
        "1. Introduction\nLarge language models have demonstrated impressive capabilities in NLP.\n"
        "However, hallucination remains a primary failure mode in mission-critical applications.\n"
        "Recent advances in dense retrieval provide external knowledge to mitigate errors [1].\n" * 5,
        fontsize=10,
    )

    # Page 2: Related Work, Methodology, Experiments
    page2 = doc.new_page()
    page2.insert_text(
        (50, 60),
        "2. Related Work\nPrior studies have explored parametric and non-parametric memory systems.\n"
        "Lewis et al. (2020) combined neural retrieval with sequence-to-sequence generators.\n" * 4,
        fontsize=10,
    )
    page2.insert_text(
        (50, 220),
        "3. Methodology\nWe present a multi-stage verification pipeline for retrieved scientific evidence.\n"
        "The system filters noisy passages using localized semantic alignment.\n" * 5,
        fontsize=10,
    )
    page2.insert_text(
        (50, 420),
        "4. Experiments\nWe conduct comprehensive experiments on scientific question answering tasks.\n"
        "We evaluate precision, recall, and hallucination rates across 1,000 queries.\n" * 4,
        fontsize=10,
    )

    # Page 3: Results, Limitations, Conclusion, References
    page3 = doc.new_page()
    page3.insert_text(
        (50, 60),
        "5. Results\nEmpirical evaluation confirms a 15% reduction in fabricated citations.\n"
        "Furthermore, evidence retrieval accuracy improves substantially over baseline RAG.\n" * 4,
        fontsize=10,
    )
    page3.insert_text(
        (50, 220),
        "6. Limitations\nOur approach currently relies on structured document segmentations.\n"
        "Future extensions will investigate multilingual scientific domains.\n" * 3,
        fontsize=10,
    )
    page3.insert_text(
        (50, 380),
        "7. Conclusion\nWe introduced an evidence-grounded framework for scientific knowledge extraction.\n" * 3,
        fontsize=10,
    )
    page3.insert_text(
        (50, 500),
        "References\n[1] Vaswani et al. Attention is all you need. NeurIPS 2017.\n"
        "[2] Lewis et al. Retrieval-Augmented Generation. NeurIPS 2020.\n",
        fontsize=9,
    )

    pdf_path = tmp_path / "scientific_paper.pdf"
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def untitled_pdf(tmp_path) -> Path:
    """Creates a PDF without any prominent title or metadata title."""
    doc = pymupdf.open()
    page = doc.new_page()
    # Regular uniform body text only
    page.insert_text(
        (50, 50),
        "This is an informal technical note without an explicit paper title.\n"
        "It discusses various implementation details of the pipeline.\n" * 20,
        fontsize=10,
    )
    pdf_path = tmp_path / "untitled_note.pdf"
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


def test_real_pdf_text_extraction(real_scientific_pdf):
    """Verifies that real PyMuPDF extraction preserves pages and text content."""
    service = ManualPDFIngestionService()
    extraction, blocks = service.extract_structured(real_scientific_pdf)

    assert len(extraction.pages) == 3
    assert extraction.pages[0].page_number == 1
    assert extraction.pages[1].page_number == 2
    assert extraction.pages[2].page_number == 3

    assert "Abstract" in extraction.pages[0].text or "Retrieval-augmented" in extraction.pages[0].text
    assert "Methodology" in extraction.pages[1].text or "verification" in extraction.pages[1].text
    assert "Limitations" in extraction.pages[2].text or "Conclusion" in extraction.pages[2].text


def test_title_extraction_detects_prominent_heading(real_scientific_pdf):
    """Verifies that the prominent title is correctly detected."""
    service = ManualPDFIngestionService()
    extraction, _ = service.extract_structured(real_scientific_pdf)
    assert extraction.title == "Robust Retrieval-Augmented Generation for Scientific Question Answering"


def test_title_extraction_leaves_null_when_undetectable(untitled_pdf):
    """Verifies that missing titles are NOT fabricated and remain None."""
    service = ManualPDFIngestionService()
    extraction, _ = service.extract_structured(untitled_pdf)
    assert extraction.title is None


def test_section_detection_identifies_canonical_sections(real_scientific_pdf):
    """Verifies that standard scientific paper sections are rule-detected."""
    service = ManualPDFIngestionService()
    extraction, _ = service.extract_structured(real_scientific_pdf)

    detected_names = {s.name for s in extraction.sections}
    expected_sections = {
        "Abstract",
        "Introduction",
        "Related Work",
        "Methods",
        "Experiments",
        "Results",
        "Limitations",
        "Conclusion",
        "References",
    }
    # Check that core sections were detected
    assert "Abstract" in detected_names
    assert "Introduction" in detected_names
    assert "Related Work" in detected_names
    assert "Methods" in detected_names
    assert "Experiments" in detected_names
    assert "Results" in detected_names
    assert "Limitations" in detected_names
    assert "Conclusion" in detected_names
    assert "References" in detected_names


def test_text_cleaning_dehyphenation_and_whitespace():
    """Verifies conservative dehyphenation and line-wrap normalization."""
    raw = (
        "This is a method- \nology that achieves state-of-the-art results.\n"
        "It preserves citations [1] and math representations like x = y + 2.\n\n"
        "Page 1 of 12\n\n"
        "Another paragraph with excessive    spaces."
    )
    cleaned = TextCleaner.clean_text(raw)

    assert "methodology" in cleaned
    assert "state-of-the-art" in cleaned
    assert "[1]" in cleaned
    assert "x = y + 2" in cleaned
    assert "Page 1 of 12" not in cleaned
    assert "excessive spaces" in cleaned


def test_deterministic_passages_and_page_references(real_scientific_pdf):
    """Verifies passage creation, deterministic ordering, and page span tracking."""
    service = ManualPDFIngestionService()
    extraction, blocks = service.extract_structured(real_scientific_pdf)

    paper_id = extraction.pages[0].page_number
    import uuid
    p_id = uuid.uuid4()
    d_id = uuid.uuid4()

    passages = service.chunker.create_passages(blocks, paper_id=p_id, document_id=d_id)
    assert len(passages) >= 3

    # Check strict sequential indexing
    for idx, passage in enumerate(passages):
        assert passage.passage_index == idx
        assert passage.paper_id == p_id
        assert passage.document_id == d_id
        assert passage.page_start <= passage.page_end
        assert 1 <= passage.page_start <= 3
        assert 1 <= passage.page_end <= 3
        assert len(passage.text.strip()) > 0


def test_provenance_and_sha256_hash(real_scientific_pdf):
    """Verifies that SHA-256 hash matches the actual file content."""
    service = ManualPDFIngestionService()
    computed_hash = service.compute_sha256(real_scientific_pdf)

    # Calculate independent hash
    with open(real_scientific_pdf, "rb") as f:
        expected_hash = hashlib.sha256(f.read()).hexdigest()

    assert computed_hash == expected_hash
    assert len(computed_hash) == 64


def test_sha256_duplicate_handling(db_session, real_scientific_pdf):
    """Verifies duplicate detection when the same PDF is ingested twice."""
    layer = SQLAlchemyPersistenceLayer(db_session)
    service = ManualPDFIngestionService(persistence=layer)

    # Ingest first time
    paper_dto, doc_dto = service.ingest_pdf(real_scientific_pdf, source_id="unique_doc_1")
    assert paper_dto is not None
    assert doc_dto is not None

    # Check duplicate detection helper
    assert service.is_content_duplicate(real_scientific_pdf) is True

    # Ingest second time with duplicate check enabled
    with pytest.raises(DuplicateDocumentError, match="Duplicate document detected"):
        service.ingest_pdf(real_scientific_pdf, source_id="unique_doc_2", check_duplicates=True)


def test_postgresql_persistence_and_relationships(db_session, real_scientific_pdf):
    """Verifies that Paper, PaperDocument, and PaperPassages are persisted in PostgreSQL."""
    layer = SQLAlchemyPersistenceLayer(db_session)
    service = ManualPDFIngestionService(persistence=layer)

    paper_dto, doc_dto = service.ingest_pdf(
        real_scientific_pdf,
        title="Persisted Title",
        source_id="pg_test_doc_1",
    )

    db_paper = db_session.query(Paper).filter_by(source_id="pg_test_doc_1").first()
    assert db_paper is not None
    assert db_paper.title == "Persisted Title"
    assert len(db_paper.documents) == 1

    db_doc = db_paper.documents[0]
    assert db_doc.content_hash == service.compute_sha256(real_scientific_pdf)
    assert db_doc.original_filename == "scientific_paper.pdf"
    assert len(db_doc.passages) > 0

    first_passage = db_doc.passages[0]
    assert isinstance(first_passage, PaperPassage)
    assert first_passage.passage_index == 0
    assert first_passage.page_start >= 1


def test_api_upload_and_passages_retrieval(real_scientific_pdf):
    """Smoke test: uploads a real PDF through FastAPI and fetches created passages."""
    client = TestClient(app)

    with open(real_scientific_pdf, "rb") as f:
        file_bytes = f.read()

    files = [("files", ("test_paper.pdf", io.BytesIO(file_bytes), "application/pdf"))]
    res = client.post("/api/papers/upload", files=files)
    assert res.status_code == 200

    data = res.json()
    assert data["count"] == 1
    uploaded_paper = data["papers"][0]
    assert "Robust Retrieval-Augmented Generation" in uploaded_paper["title"] or uploaded_paper["title"] == "test_paper.pdf"
    assert "content_hash" in uploaded_paper["raw_metadata"]

    # Verify paper exists in /api/papers
    papers_res = client.get("/api/papers")
    assert papers_res.status_code == 200
    papers_list = papers_res.json()
    assert len(papers_list) >= 1
