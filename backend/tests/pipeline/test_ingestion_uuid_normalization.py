"""Regression tests for paper ID normalization across pipeline boundaries.

Verifies:
1. RunPaperRecord model stores paper_id canonically as string even when assigned uuid.UUID.
2. ResearchWorkflowEngine._normalize_paper_ids handles UUID objects, string UUIDs, non-UUID strings, and None.
3. ResearchWorkflowEngine._execute_stage_ingestion correctly handles PaperIngestionResult with UUID paper_id.
4. ResearchWorkflowEngine._execute_stage_ingestion correctly handles PaperIngestionResult with string paper_id.
5. Ingestion failures record cleanly to SQLite without secondary UUID-binding or session exceptions.
6. Existing corpus papers are cleanly linked and skipped without re-download on retry.
"""

from __future__ import annotations

import datetime
import uuid
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from ideagen.db.models import (
    Base,
    Paper,
    PaperDocument,
    PaperPassage,
    ProjectPaperRecord,
    ResearchKnowledge,
    ResearchProjectRecord,
    ResearchRunRecord,
    ResearchStatement,
    RunPaperRecord,
)
from ideagen.ingestion.openalex_ingestion import PaperIngestionResult
from ideagen.pipeline.workflow_engine import ResearchWorkflowEngine


@pytest.fixture
def db_session():
    """Sets up an in-memory SQLite database session."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()


def test_run_paper_record_paper_id_normalization_model(db_session):
    """Verifies RunPaperRecord.paper_id is always converted to string or None in SQLite."""
    test_uuid = uuid.uuid4()
    run = ResearchRunRecord(id="run_test", project_id="proj_test", status="PENDING")
    db_session.add(run)
    db_session.commit()

    # 1. Initialized with uuid.UUID
    r1 = RunPaperRecord(
        id="rp_1",
        run_id="run_test",
        openalex_id="W1",
        title="Paper 1",
        paper_id=test_uuid,
    )
    assert r1.paper_id == str(test_uuid)
    assert isinstance(r1.paper_id, str)

    db_session.add(r1)
    db_session.commit()
    db_session.refresh(r1)
    assert r1.paper_id == str(test_uuid)

    # 2. Reassigned with uuid.UUID
    new_uuid = uuid.uuid4()
    r1.paper_id = new_uuid
    assert r1.paper_id == str(new_uuid)
    assert isinstance(r1.paper_id, str)
    db_session.commit()
    db_session.refresh(r1)
    assert r1.paper_id == str(new_uuid)

    # 3. Assigned via corpus_paper_id setter
    newer_uuid = uuid.uuid4()
    r1.corpus_paper_id = newer_uuid
    assert r1.paper_id == str(newer_uuid)
    assert isinstance(r1.paper_id, str)
    db_session.commit()

    # 4. Assigned None
    r1.paper_id = None
    assert r1.paper_id is None
    db_session.commit()
    db_session.refresh(r1)
    assert r1.paper_id is None


def test_normalize_paper_ids_helper():
    """Verifies ResearchWorkflowEngine._normalize_paper_ids behavior for all inputs."""
    # 1. uuid.UUID instance
    u = uuid.uuid4()
    str_id, uuid_obj = ResearchWorkflowEngine._normalize_paper_ids(u)
    assert str_id == str(u)
    assert uuid_obj == u
    assert isinstance(uuid_obj, uuid.UUID)

    # 2. Hyphenated string UUID
    u_str = str(u)
    str_id, uuid_obj = ResearchWorkflowEngine._normalize_paper_ids(u_str)
    assert str_id == u_str
    assert uuid_obj == u

    # 3. Hex string without hyphens
    hex_str = u.hex
    str_id, uuid_obj = ResearchWorkflowEngine._normalize_paper_ids(hex_str)
    assert str_id == str(u)
    assert uuid_obj == u

    # 4. Non-UUID string
    str_id, uuid_obj = ResearchWorkflowEngine._normalize_paper_ids("custom-string-identifier")
    assert str_id == "custom-string-identifier"
    assert uuid_obj is None

    # 5. None
    str_id, uuid_obj = ResearchWorkflowEngine._normalize_paper_ids(None)
    assert str_id is None
    assert uuid_obj is None

    # 6. Empty / whitespace string
    str_id, uuid_obj = ResearchWorkflowEngine._normalize_paper_ids("   ")
    assert str_id is None
    assert uuid_obj is None


def test_stage_ingestion_with_uuid_paper_id(db_session):
    """Verifies that ingestion result with a UUID paper_id is cleanly persisted in SQLite."""
    project = ResearchProjectRecord(id="proj_alpha", name="Alpha Project")
    db_session.add(project)
    db_session.commit()

    run = ResearchRunRecord(
        id="run_alpha",
        project_id="proj_alpha",
        status="RUNNING",
        planned_stages=["INGESTION"],
        stage_states={},
    )
    db_session.add(run)

    paper = RunPaperRecord(
        id="rp_alpha_1",
        run_id="run_alpha",
        openalex_id="W9999",
        title="Test Paper with UUID",
        status="PENDING",
    )
    db_session.add(paper)
    db_session.commit()

    test_paper_uuid = uuid.uuid4()
    mock_search_item = MagicMock()
    mock_search_item.raw_metadata = {"id": "https://openalex.org/W9999", "title": "Test Paper with UUID"}

    mock_ingestion_result = PaperIngestionResult(
        paper_id=test_paper_uuid,
        source_id="W9999",
        status="INGESTED",
        has_pdf=True,
        passage_count=42,
        batch_id="manual",
    )

    with patch("ideagen.pipeline.workflow_engine.OpenAlexSearchProvider") as MockSearch, \
         patch("ideagen.pipeline.workflow_engine.OpenAlexPaperIngestionService") as MockIngest:

        MockSearch.return_value.search.return_value = [mock_search_item]
        MockIngest.return_value.ingest_work.return_value = mock_ingestion_result

        states = {}
        ResearchWorkflowEngine._execute_stage_ingestion(db_session, run, states)

    db_session.refresh(paper)
    db_session.refresh(run)

    # Paper status and ID
    assert paper.status == "INGESTED"
    assert paper.paper_id == str(test_paper_uuid)
    assert isinstance(paper.paper_id, str)
    assert paper.passage_count == 42
    assert paper.error_message is None

    # Project paper association
    assoc = db_session.query(ProjectPaperRecord).filter(
        (ProjectPaperRecord.project_id == "proj_alpha") & (ProjectPaperRecord.paper_id == test_paper_uuid)
    ).first()
    assert assoc is not None
    assert assoc.paper_id == test_paper_uuid

    # Stage state
    assert states["INGESTION"]["status"] == "COMPLETED"
    assert states["INGESTION"]["details"]["ingested"] == 1
    assert states["INGESTION"]["details"]["failed"] == 0


def test_stage_ingestion_with_string_paper_id(db_session):
    """Verifies that ingestion result with a string paper_id is cleanly accepted and linked."""
    project = ResearchProjectRecord(id="proj_beta", name="Beta Project")
    db_session.add(project)
    db_session.commit()

    run = ResearchRunRecord(
        id="run_beta",
        project_id="proj_beta",
        status="RUNNING",
        planned_stages=["INGESTION"],
        stage_states={},
    )
    db_session.add(run)

    paper = RunPaperRecord(
        id="rp_beta_1",
        run_id="run_beta",
        openalex_id="W8888",
        title="Test Paper with String UUID",
        status="PENDING",
    )
    db_session.add(paper)
    db_session.commit()

    test_paper_uuid = uuid.uuid4()
    mock_search_item = MagicMock()
    mock_search_item.raw_metadata = {"id": "https://openalex.org/W8888", "title": "Test Paper with String UUID"}

    # Mock returns a result where paper_id is a string
    mock_ingestion_result = MagicMock()
    mock_ingestion_result.paper_id = str(test_paper_uuid)
    mock_ingestion_result.status = "INGESTED"
    mock_ingestion_result.passage_count = 15

    with patch("ideagen.pipeline.workflow_engine.OpenAlexSearchProvider") as MockSearch, \
         patch("ideagen.pipeline.workflow_engine.OpenAlexPaperIngestionService") as MockIngest:

        MockSearch.return_value.search.return_value = [mock_search_item]
        MockIngest.return_value.ingest_work.return_value = mock_ingestion_result

        states = {}
        ResearchWorkflowEngine._execute_stage_ingestion(db_session, run, states)

    db_session.refresh(paper)
    assert paper.status == "INGESTED"
    assert paper.paper_id == str(test_paper_uuid)
    assert isinstance(paper.paper_id, str)
    assert paper.passage_count == 15

    assoc = db_session.query(ProjectPaperRecord).filter(
        (ProjectPaperRecord.project_id == "proj_beta") & (ProjectPaperRecord.paper_id == test_paper_uuid)
    ).first()
    assert assoc is not None


def test_stage_ingestion_failure_persists_without_secondary_uuid_exception(db_session):
    """Proves that unexpected exceptions during ingestion record FAILED status without parameter binding crash."""
    project = ResearchProjectRecord(id="proj_gamma", name="Gamma Project")
    db_session.add(project)
    db_session.commit()

    run = ResearchRunRecord(
        id="run_gamma",
        project_id="proj_gamma",
        status="RUNNING",
        planned_stages=["INGESTION"],
        stage_states={},
    )
    db_session.add(run)

    paper = RunPaperRecord(
        id="rp_gamma_1",
        run_id="run_gamma",
        openalex_id="W7777",
        title="Failing Paper",
        status="PENDING",
    )
    db_session.add(paper)
    db_session.commit()

    mock_search_item = MagicMock()
    mock_search_item.raw_metadata = {"id": "https://openalex.org/W7777", "title": "Failing Paper"}

    with patch("ideagen.pipeline.workflow_engine.OpenAlexSearchProvider") as MockSearch, \
         patch("ideagen.pipeline.workflow_engine.OpenAlexPaperIngestionService") as MockIngest:

        MockSearch.return_value.search.return_value = [mock_search_item]
        # Simulate unexpected crash during ingestion
        MockIngest.return_value.ingest_work.side_effect = RuntimeError("PDF storage volume full")

        states = {}
        # Must NOT raise sqlite3.ProgrammingError or bubble unhandled exception
        ResearchWorkflowEngine._execute_stage_ingestion(db_session, run, states)

    db_session.refresh(paper)
    assert paper.status == "FAILED"
    assert "PDF storage volume full" in paper.error_message
    assert states["INGESTION"]["status"] == "FAILED"
    assert states["INGESTION"]["details"]["failed"] == 1


def test_stage_ingestion_skips_existing_corpus_paper(db_session):
    """Verifies that retry on an existing corpus paper skips re-download and links properly."""
    project = ResearchProjectRecord(id="proj_delta", name="Delta Project")
    db_session.add(project)
    db_session.commit()

    # Pre-existing paper in the corpus
    existing_paper = Paper(
        id=uuid.uuid4(),
        source_type="OPENALEX_API",
        source_id="W3196371845",
        title="Federated learning on non-IID data: A survey",
        abstract="Survey abstract",
        authors=[],
        referenced_works=[],
        raw_metadata={},
        batch_id="manual",
    )
    db_session.add(existing_paper)
    db_session.flush()

    doc = PaperDocument(
        paper_id=existing_paper.id,
        full_text="Sample survey full text",
        extraction_method="pymupdf",
        extraction_version="1.0",
        extracted_at=datetime.datetime.now(datetime.timezone.utc),
        content_hash="hash123",
    )
    db_session.add(doc)
    db_session.flush()

    for idx in range(136):
        db_session.add(PaperPassage(
            paper_id=existing_paper.id,
            document_id=doc.id,
            passage_index=idx,
            page_start=1,
            page_end=1,
            text=f"Passage text {idx}",
        ))
    db_session.commit()

    run = ResearchRunRecord(
        id="run_delta",
        project_id="proj_delta",
        status="RUNNING",
        planned_stages=["INGESTION"],
        stage_states={},
    )
    db_session.add(run)

    paper = RunPaperRecord(
        id="rp_delta_1",
        run_id="run_delta",
        openalex_id="W3196371845",
        title="Federated learning on non-IID data: A survey",
        status="PENDING",
    )
    db_session.add(paper)
    db_session.commit()

    with patch("ideagen.pipeline.workflow_engine.OpenAlexSearchProvider") as MockSearch, \
         patch("ideagen.pipeline.workflow_engine.OpenAlexPaperIngestionService") as MockIngest:

        states = {}
        ResearchWorkflowEngine._execute_stage_ingestion(db_session, run, states)

        # Neither search nor ingestion should have been called because it was in corpus!
        MockSearch.return_value.search.assert_not_called()
        MockIngest.return_value.ingest_work.assert_not_called()

    db_session.refresh(paper)
    assert paper.status == "SKIPPED_DUPLICATE"
    assert paper.paper_id == str(existing_paper.id)
    assert isinstance(paper.paper_id, str)
    assert paper.passage_count == 136

    # Verify association was created
    assoc = db_session.query(ProjectPaperRecord).filter(
        (ProjectPaperRecord.project_id == "proj_delta") & (ProjectPaperRecord.paper_id == existing_paper.id)
    ).first()
    assert assoc is not None
    assert assoc.paper_id == existing_paper.id


def test_stage_understanding_corpus_paper_with_zero_statements_is_re_extracted(db_session):
    """Verifies that an unassociated corpus paper with extraction_status='SUCCESS' but 0 statements is re-extracted."""
    project = ResearchProjectRecord(id="proj_corpus_retry", name="Corpus Retry Project")
    db_session.add(project)
    db_session.commit()

    run = ResearchRunRecord(
        id="run_corpus_retry",
        project_id="proj_corpus_retry",
        status="RUNNING",
        planned_stages=["UNDERSTANDING"],
        stage_states={},
    )
    db_session.add(run)

    corpus_paper = Paper(
        id=uuid.uuid4(),
        source_type="OPENALEX_API",
        source_id="W12345",
        title="Zero-statement Paper",
        abstract="Abstract",
        authors=[],
        referenced_works=[],
        raw_metadata={},
        batch_id="manual",
    )
    db_session.add(corpus_paper)
    db_session.flush()

    # Associate with project
    assoc = ProjectPaperRecord(project_id="proj_corpus_retry", paper_id=corpus_paper.id)
    db_session.add(assoc)

    # Add passages for this paper
    doc = PaperDocument(
        paper_id=corpus_paper.id,
        full_text="Some full text",
        extraction_method="pymupdf",
        extraction_version="1.0",
        extracted_at=datetime.datetime.now(datetime.timezone.utc),
        content_hash="hash_c1",
    )
    db_session.add(doc)
    db_session.flush()

    db_session.add(PaperPassage(
        paper_id=corpus_paper.id,
        document_id=doc.id,
        passage_index=0,
        page_start=1,
        page_end=1,
        text="Sample passage text",
    ))

    # Existing knowledge has status SUCCESS but ZERO statements!
    empty_knowledge = ResearchKnowledge(
        paper_id=corpus_paper.id,
        extraction_status="SUCCESS",
        model_name="mock-model",
        extracted_at=datetime.datetime.now(datetime.timezone.utc),
    )
    db_session.add(empty_knowledge)
    db_session.commit()

    with patch("ideagen.pipeline.workflow_engine.PaperUnderstandingPipeline") as MockPipeline:
        mock_pipeline_instance = MockPipeline.return_value
        mock_dto = MagicMock()
        mock_dto.statement_count = 2
        mock_pipeline_instance.understand_paper.return_value = mock_dto

        states = {}
        ResearchWorkflowEngine._execute_stage_understanding(db_session, run, states)

        # Because it had 0 statements, it must NOT be skipped!
        mock_pipeline_instance.understand_paper.assert_called_once_with(corpus_paper.id)


def test_stage_understanding_corpus_paper_with_statements_is_skipped(db_session):
    """Verifies that an unassociated corpus paper with extraction_status='SUCCESS' and statements > 0 is skipped."""
    project = ResearchProjectRecord(id="proj_corpus_skip", name="Corpus Skip Project")
    db_session.add(project)
    db_session.commit()

    run = ResearchRunRecord(
        id="run_corpus_skip",
        project_id="proj_corpus_skip",
        status="RUNNING",
        planned_stages=["UNDERSTANDING"],
        stage_states={},
    )
    db_session.add(run)

    corpus_paper = Paper(
        id=uuid.uuid4(),
        source_type="OPENALEX_API",
        source_id="W67890",
        title="Valid-statements Paper",
        abstract="Abstract",
        authors=[],
        referenced_works=[],
        raw_metadata={},
        batch_id="manual",
    )
    db_session.add(corpus_paper)
    db_session.flush()

    assoc = ProjectPaperRecord(project_id="proj_corpus_skip", paper_id=corpus_paper.id)
    db_session.add(assoc)

    doc = PaperDocument(
        paper_id=corpus_paper.id,
        full_text="Some full text",
        extraction_method="pymupdf",
        extraction_version="1.0",
        extracted_at=datetime.datetime.now(datetime.timezone.utc),
        content_hash="hash_c2",
    )
    db_session.add(doc)
    db_session.flush()

    passage = PaperPassage(
        paper_id=corpus_paper.id,
        document_id=doc.id,
        passage_index=0,
        page_start=1,
        page_end=1,
        text="Sample passage text",
    )
    db_session.add(passage)
    db_session.flush()

    # Knowledge has status SUCCESS AND 1 statement
    valid_knowledge = ResearchKnowledge(
        paper_id=corpus_paper.id,
        extraction_status="SUCCESS",
        model_name="mock-model",
        extracted_at=datetime.datetime.now(datetime.timezone.utc),
    )
    db_session.add(valid_knowledge)
    db_session.flush()

    stmt = ResearchStatement(
        knowledge=valid_knowledge,
        paper_id=corpus_paper.id,
        passage_id=passage.id,
        statement_text="Grounded statement",
        statement_type="METHOD",
    )
    db_session.add(stmt)
    db_session.commit()

    with patch("ideagen.pipeline.workflow_engine.PaperUnderstandingPipeline") as MockPipeline:
        mock_pipeline_instance = MockPipeline.return_value

        states = {}
        ResearchWorkflowEngine._execute_stage_understanding(db_session, run, states)

        # Because it already had statements, it should be skipped!
        mock_pipeline_instance.understand_paper.assert_not_called()

