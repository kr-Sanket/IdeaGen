"""Stage 3C Regression Test Suite: Evolution Persistence & Transaction Boundaries.

Verifies:
1. Standalone successful evolution: close original session and verify from a fresh session that opportunity status, evidence, and history persist.
2. Standalone partial success: verify successful candidate changes persist while failed candidates leave no partial evidence/history.
3. Standalone evolution exception: verify paper/document/passages remain persisted, evolution reports failure, and no uncommitted partial evolution changes leak.
4. Caller-owned transaction success: verify the hook does not commit prematurely and the outer transaction can still roll back all changes.
5. Caller-owned transaction rollback: verify no paper, document, passage, opportunity transition, evidence, or history rows leak after rollback.
6. Commit failure: verify the result does not claim evolution success and diagnostics are retained.
7. Verify disabled-hook and manual PDF ingestion behavior remain unchanged.
"""

from __future__ import annotations

import datetime
import io
from pathlib import Path
from typing import Dict, Any, Optional
from unittest.mock import MagicMock, patch
import uuid
import pytest
import pymupdf
from sqlalchemy import create_engine, select, event
from sqlalchemy.orm import Session, sessionmaker

from ideagen.db.models import (
    Base,
    Opportunity,
    Paper,
    PaperDocument,
    PaperPassage,
    OpportunityEvidence,
    OpportunityHistory,
)
from ideagen.db.persistence import SQLAlchemyPersistenceLayer
from ideagen.ingestion.openalex_ingestion import (
    OpenAlexPaperIngestionService,
    IngestionStatus,
    EvolutionHookStatus,
)
from ideagen.ingestion.pdf import ManualPDFIngestionService
from ideagen.opportunity.models import OpportunityRelationship, OpportunityStatus
from ideagen.opportunity.service import OpportunityTransitionService
from ideagen.opportunity.classifier import (
    RelationshipClassifier,
    RelationshipClassification,
)
from ideagen.opportunity.integration import OpportunityClassificationIntegration
from ideagen.opportunity.incremental import (
    IncrementalOpportunityEvolutionService,
    IncrementalEvolutionResult,
    IncrementalEvolutionRunStatus,
    OpportunityEvaluationOutcome,
    OpportunityEvaluationStatus,
)
from ideagen.llm.provider import LLMProvider
from ideagen.llm.exceptions import ProviderFailureError


# ── TEST FIXTURES & HELPERS ───────────────────────────────────────────────────

def create_sample_pdf_bytes(title: str = "Test Title", body: str = "Detailed scientific research text. ") -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    lines = [title, "", "Abstract"]
    for i in range(25):
        lines.append(f"Abstract sentence {i}: {body}")
    lines.extend(["", "1. Introduction"])
    for i in range(25):
        lines.append(f"Introduction sentence {i}: {body}")
    page.insert_text((50, 50), "\n".join(lines))
    data = doc.tobytes()
    doc.close()
    return data


class DummyMockResponse:
    def __init__(self, data: bytes, headers: Optional[Dict[str, str]] = None, status: int = 200):
        self.bio = io.BytesIO(data)
        self.headers = headers if headers is not None else {"Content-Length": str(len(data))}
        self.status = status

    def read(self, amt: int = -1) -> bytes:
        return self.bio.read(amt)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.bio.close()


def make_sample_work(
    source_id: str = "https://openalex.org/W3000000001",
    doi: str = "https://doi.org/10.1000/3000000001",
    title: str = "Decentralized State Machine Durability",
    pdf_url: Optional[str] = "https://example.com/paper.pdf",
    pub_date: str = "2023-01-01",
) -> Dict[str, Any]:
    return {
        "id": source_id,
        "doi": doi,
        "title": title,
        "publication_date": pub_date,
        "abstract_inverted_index": {
            "We": [0], "verify": [1], "state": [2], "machine": [3], "durability.": [4]
        },
        "authorships": [{"author": {"display_name": "Alice Author"}}],
        "primary_location": {
            "source": {"type": "journal"},
            "pdf_url": pdf_url,
            "landing_page_url": "https://example.com/landing",
        },
        "best_oa_location": {
            "pdf_url": pdf_url,
            "landing_page_url": "https://example.com/landing",
        },
    }


@pytest.fixture
def test_db_factory(tmp_path: Path):
    """File-backed SQLite database to strictly verify cross-session persistence and durability."""
    db_file = tmp_path / "test_boundary.db"
    engine = create_engine(f"sqlite:///{db_file}")

    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    return session_factory


class ConfigurableMockLLM(LLMProvider):
    def __init__(self, classification: Optional[RelationshipClassification] = None):
        self.classification = classification
        self.should_fail = False

    def generate_structured(self, prompt: str, response_schema):
        if self.should_fail:
            raise ProviderFailureError("LLM cluster 503 unavailable")
        if self.classification is not None:
            return self.classification
        raise ValueError("No mock classification configured")


# ── TEST 1: STANDALONE SUCCESSFUL EVOLUTION PERSISTS IN FRESH SESSION ─────────

def test_1_standalone_successful_evolution_persists_in_fresh_session(test_db_factory):
    """Verifies that in standalone mode, closing the original session preserves evolution records in a fresh session."""
    session1 = test_db_factory()
    persistence1 = SQLAlchemyPersistenceLayer(session1)

    opp_id = uuid.uuid4()
    opp = Opportunity(
        id=opp_id,
        title="Durability Opportunity",
        problem_statement="Test durability across session boundaries.",
        description="Verify fresh session sees committed evolution changes.",
        status=OpportunityStatus.OPEN.value,
        confidence=0.9,
        created_at=datetime.datetime(2022, 1, 1, tzinfo=datetime.timezone.utc),
    )
    session1.add(opp)
    session1.commit()

    passage_text = "Proving transaction boundary durability across sessions."
    mock_llm = ConfigurableMockLLM(
        classification=RelationshipClassification(
            relationship=OpportunityRelationship.ADDRESSES,
            confidence=0.95,
            rationale="Addresses durability requirements.",
            evidence_excerpt=passage_text,
        )
    )
    classifier = RelationshipClassifier(provider=mock_llm)
    trans_svc = OpportunityTransitionService(session=session1)
    integration = OpportunityClassificationIntegration(classifier=classifier, transition_service=trans_svc, session=session1)
    evolution_svc = IncrementalOpportunityEvolutionService(integration=integration, session=session1)

    service = OpenAlexPaperIngestionService(
        persistence=persistence1,
        auto_evolve=True,
        evolution_service=evolution_svc,
    )

    pdf_bytes = create_sample_pdf_bytes(title="Durability Paper", body=passage_text)
    work = make_sample_work(
        source_id="https://openalex.org/W_STAGE3C_1",
        doi="https://doi.org/10.1000/s3c_1",
        pub_date="2023-01-01",
    )

    with patch.object(service.downloader, "_open_request", return_value=DummyMockResponse(pdf_bytes)):
        res = service.ingest_work(work)

    assert res.status == IngestionStatus.INGESTED
    assert res.evolution_status == EvolutionHookStatus.SUCCESS
    paper_id = res.paper_id

    # Crucial step: CLOSE and DISPOSE the original session completely
    session1.close()

    # Query strictly from a FRESH session connected to the database
    fresh_session = test_db_factory()
    try:
        # 1. Paper and documents exist
        db_paper = fresh_session.execute(select(Paper).where(Paper.id == paper_id)).scalar_one_or_none()
        assert db_paper is not None
        assert db_paper.source_id == "W_STAGE3C_1"

        # 2. Opportunity status was durably updated
        db_opp = fresh_session.execute(select(Opportunity).where(Opportunity.id == opp_id)).scalar_one_or_none()
        assert db_opp is not None
        assert db_opp.status == OpportunityStatus.PARTIALLY_ADDRESSED.value

        # 3. OpportunityEvidence was durably committed
        db_ev = fresh_session.execute(
            select(OpportunityEvidence).where(OpportunityEvidence.opportunity_id == opp_id)
        ).scalar_one_or_none()
        assert db_ev is not None
        assert db_ev.paper_id == paper_id
        assert db_ev.evidence_type == OpportunityRelationship.ADDRESSES.value

        # 4. OpportunityHistory was durably committed
        db_hist = fresh_session.execute(
            select(OpportunityHistory).where(OpportunityHistory.opportunity_id == opp_id)
        ).scalars().all()
        assert len(db_hist) == 1
        assert db_hist[0].previous_status == OpportunityStatus.OPEN.value
        assert db_hist[0].new_status == OpportunityStatus.PARTIALLY_ADDRESSED.value
    finally:
        fresh_session.close()


# ── TEST 2: STANDALONE PARTIAL SUCCESS PERSISTS VALID CANDIDATES ──────────────

def test_2_standalone_partial_success_persists_valid_candidates(test_db_factory):
    """Verifies that in partial success, successful candidate changes persist while failed candidates leave no partial rows."""
    session1 = test_db_factory()
    persistence1 = SQLAlchemyPersistenceLayer(session1)

    opp_success_id = uuid.uuid4()
    opp_success = Opportunity(
        id=opp_success_id,
        title="Candidate Success",
        problem_statement="Will be successfully evaluated.",
        description="First candidate.",
        status=OpportunityStatus.OPEN.value,
        confidence=0.9,
        created_at=datetime.datetime(2022, 1, 1, tzinfo=datetime.timezone.utc),
    )
    opp_fail_id = uuid.uuid4()
    opp_fail = Opportunity(
        id=opp_fail_id,
        title="Candidate Fail",
        problem_statement="Will fail evaluation.",
        description="Second candidate.",
        status=OpportunityStatus.OPEN.value,
        confidence=0.9,
        created_at=datetime.datetime(2022, 2, 1, tzinfo=datetime.timezone.utc),
    )
    session1.add_all([opp_success, opp_fail])
    session1.commit()

    passage_text = "Scientific text for partial success."
    mock_llm = ConfigurableMockLLM()
    classifier = RelationshipClassifier(provider=mock_llm)
    trans_svc = OpportunityTransitionService(session=session1)
    integration = OpportunityClassificationIntegration(classifier=classifier, transition_service=trans_svc, session=session1)
    evolution_svc = IncrementalOpportunityEvolutionService(integration=integration, session=session1)

    # Perform evaluation callback that creates evidence for opp_success using the real paper_id
    def mock_eval_paper(paper_id):
        opp_success.status = OpportunityStatus.PARTIALLY_ADDRESSED.value
        ev = OpportunityEvidence(
            opportunity_id=opp_success_id,
            paper_id=paper_id,
            evidence_type=OpportunityRelationship.ADDRESSES.value,
            supporting_text=passage_text,
        )
        session1.add(ev)
        return IncrementalEvolutionResult(
            paper_id=paper_id,
            run_status=IncrementalEvolutionRunStatus.PARTIAL_SUCCESS,
            candidate_count=2,
            evaluated_count=1,
            failed_count=1,
            outcomes=[
                OpportunityEvaluationOutcome(
                    opportunity_id=opp_success_id,
                    status=OpportunityEvaluationStatus.EVALUATED,
                    initial_status=OpportunityStatus.OPEN,
                    final_status=OpportunityStatus.PARTIALLY_ADDRESSED,
                    relationship=OpportunityRelationship.ADDRESSES,
                    state_changed=True,
                ),
                OpportunityEvaluationOutcome(
                    opportunity_id=opp_fail_id,
                    status=OpportunityEvaluationStatus.FAILED,
                    initial_status=OpportunityStatus.OPEN,
                    final_status=OpportunityStatus.OPEN,
                    error_message="LLM timeout on candidate 2",
                ),
            ],
        )

    service = OpenAlexPaperIngestionService(
        persistence=persistence1,
        auto_evolve=True,
        evolution_service=evolution_svc,
    )

    pdf_bytes = create_sample_pdf_bytes(title="Partial Success Paper", body=passage_text)
    work = make_sample_work(
        source_id="https://openalex.org/W_STAGE3C_2",
        doi="https://doi.org/10.1000/s3c_2",
        pub_date="2023-01-01",
    )

    with patch.object(service.downloader, "_open_request", return_value=DummyMockResponse(pdf_bytes)):
        with patch.object(evolution_svc, "evaluate_paper", side_effect=mock_eval_paper):
            res = service.ingest_work(work)

    assert res.status == IngestionStatus.INGESTED
    assert res.evolution_status == EvolutionHookStatus.PARTIAL_SUCCESS

    # Close session and query from fresh session
    session1.close()

    fresh_session = test_db_factory()
    try:
        # Successful candidate is committed
        db_opp_success = fresh_session.execute(select(Opportunity).where(Opportunity.id == opp_success_id)).scalar_one()
        assert db_opp_success.status == OpportunityStatus.PARTIALLY_ADDRESSED.value

        db_ev_success = fresh_session.execute(
            select(OpportunityEvidence).where(OpportunityEvidence.opportunity_id == opp_success_id)
        ).scalar_one_or_none()
        assert db_ev_success is not None

        # Failed candidate left no evidence
        db_opp_fail = fresh_session.execute(select(Opportunity).where(Opportunity.id == opp_fail_id)).scalar_one()
        assert db_opp_fail.status == OpportunityStatus.OPEN.value

        db_ev_fail = fresh_session.execute(
            select(OpportunityEvidence).where(OpportunityEvidence.opportunity_id == opp_fail_id)
        ).scalars().all()
        assert len(db_ev_fail) == 0
    finally:
        fresh_session.close()


# ── TEST 3: STANDALONE EVOLUTION EXCEPTION LEAVES NO PARTIAL CHANGES ──────────

def test_3_standalone_evolution_exception_leaves_paper_intact_and_rolls_back_evolution(test_db_factory):
    """Verifies that on evolution exception, paper remains committed, hook reports FAILED, and session is rolled back."""
    session1 = test_db_factory()
    persistence1 = SQLAlchemyPersistenceLayer(session1)

    opp_id = uuid.uuid4()
    opp = Opportunity(
        id=opp_id,
        title="Exception Opportunity",
        problem_statement="Test exception boundary.",
        description="Verify rollback on unhandled evolution error.",
        status=OpportunityStatus.OPEN.value,
        confidence=0.9,
        created_at=datetime.datetime(2022, 1, 1, tzinfo=datetime.timezone.utc),
    )
    session1.add(opp)
    session1.commit()

    mock_llm = ConfigurableMockLLM()
    mock_llm.should_fail = True  # Throws ProviderFailureError

    classifier = RelationshipClassifier(provider=mock_llm)
    trans_svc = OpportunityTransitionService(session=session1)
    integration = OpportunityClassificationIntegration(classifier=classifier, transition_service=trans_svc, session=session1)
    evolution_svc = IncrementalOpportunityEvolutionService(integration=integration, session=session1)

    service = OpenAlexPaperIngestionService(
        persistence=persistence1,
        auto_evolve=True,
        evolution_service=evolution_svc,
    )

    pdf_bytes = create_sample_pdf_bytes(title="Exception Paper", body="Scientific text.")
    work = make_sample_work(
        source_id="https://openalex.org/W_STAGE3C_3",
        doi="https://doi.org/10.1000/s3c_3",
        pub_date="2023-01-01",
    )

    with patch.object(service.downloader, "_open_request", return_value=DummyMockResponse(pdf_bytes)):
        res = service.ingest_work(work)

    # Ingestion status MUST be INGESTED; evolution status MUST be FAILED
    assert res.status == IngestionStatus.INGESTED
    assert res.paper_id is not None
    assert res.evolution_status == EvolutionHookStatus.FAILED

    session1.close()

    # Query from fresh session
    fresh_session = test_db_factory()
    try:
        # Paper, doc, passages MUST remain committed
        db_paper = fresh_session.execute(select(Paper).where(Paper.id == res.paper_id)).scalar_one_or_none()
        assert db_paper is not None
        assert db_paper.source_id == "W_STAGE3C_3"

        db_doc = fresh_session.execute(select(PaperDocument).where(PaperDocument.paper_id == res.paper_id)).scalar_one_or_none()
        assert db_doc is not None

        db_passages = fresh_session.execute(select(PaperPassage).where(PaperPassage.paper_id == res.paper_id)).scalars().all()
        assert len(db_passages) > 0

        # Opportunity MUST remain OPEN with 0 evidence
        db_opp = fresh_session.execute(select(Opportunity).where(Opportunity.id == opp_id)).scalar_one()
        assert db_opp.status == OpportunityStatus.OPEN.value

        db_ev = fresh_session.execute(select(OpportunityEvidence).where(OpportunityEvidence.opportunity_id == opp_id)).scalars().all()
        assert len(db_ev) == 0
    finally:
        fresh_session.close()


# ── TEST 4: CALLER-OWNED TRANSACTION SUCCESS DOES NOT COMMIT PREMATURELY ──────

def test_4_caller_owned_transaction_does_not_commit_prematurely(test_db_factory):
    """Verifies that when caller owns outer transaction, the hook does not commit prematurely and caller can still rollback."""
    session1 = test_db_factory()
    persistence1 = SQLAlchemyPersistenceLayer(session1)

    opp_id = uuid.uuid4()
    opp = Opportunity(
        id=opp_id,
        title="Caller Owned Opportunity",
        problem_statement="Test caller-owned boundary.",
        description="Verify caller controls commit.",
        status=OpportunityStatus.OPEN.value,
        confidence=0.9,
        created_at=datetime.datetime(2022, 1, 1, tzinfo=datetime.timezone.utc),
    )
    session1.add(opp)
    session1.commit()

    passage_text = "Caller transaction boundary text."
    mock_llm = ConfigurableMockLLM(
        classification=RelationshipClassification(
            relationship=OpportunityRelationship.ADDRESSES,
            confidence=0.95,
            rationale="Addresses caller boundary.",
            evidence_excerpt=passage_text,
        )
    )
    classifier = RelationshipClassifier(provider=mock_llm)
    trans_svc = OpportunityTransitionService(session=session1)
    integration = OpportunityClassificationIntegration(classifier=classifier, transition_service=trans_svc, session=session1)
    evolution_svc = IncrementalOpportunityEvolutionService(integration=integration, session=session1)

    service = OpenAlexPaperIngestionService(
        persistence=persistence1,
        auto_evolve=True,
        evolution_service=evolution_svc,
    )

    pdf_bytes = create_sample_pdf_bytes(title="Caller Paper", body=passage_text)
    work = make_sample_work(
        source_id="https://openalex.org/W_STAGE3C_4",
        doi="https://doi.org/10.1000/s3c_4",
        pub_date="2023-01-01",
    )

    # Caller wraps ingestion in outer transaction and decides to abort after ingestion returns
    try:
        with persistence1.transaction():
            with patch.object(service.downloader, "_open_request", return_value=DummyMockResponse(pdf_bytes)):
                res = service.ingest_work(work)
            # Verify hook completed within the active transaction
            assert res.status == IngestionStatus.INGESTED
            assert res.evolution_status == EvolutionHookStatus.SUCCESS
            # Caller-side business logic aborts
            raise RuntimeError("Caller decided to abort outer unit of work")
    except RuntimeError:
        pass

    session1.close()

    # Fresh session must prove that the hook DID NOT commit prematurely; everything rolled back!
    fresh_session = test_db_factory()
    try:
        db_paper = fresh_session.execute(select(Paper).where(Paper.source_id == "W_STAGE3C_4")).scalar_one_or_none()
        assert db_paper is None  # Paper was NOT committed prematurely!

        db_opp = fresh_session.execute(select(Opportunity).where(Opportunity.id == opp_id)).scalar_one()
        assert db_opp.status == OpportunityStatus.OPEN.value  # Opportunity status was NOT committed prematurely!

        db_ev = fresh_session.execute(select(OpportunityEvidence).where(OpportunityEvidence.opportunity_id == opp_id)).scalars().all()
        assert len(db_ev) == 0  # Evidence was NOT committed prematurely!
    finally:
        fresh_session.close()


# ── TEST 5: CALLER-OWNED TRANSACTION ROLLBACK LEAKS ZERO ROWS ─────────────────

def test_5_caller_owned_transaction_rollback_leaks_zero_rows(test_db_factory):
    """Verifies that an aborted outer transaction leaves exactly 0 paper, document, passage, evidence, or history rows."""
    session1 = test_db_factory()
    persistence1 = SQLAlchemyPersistenceLayer(session1)

    opp_id = uuid.uuid4()
    opp = Opportunity(
        id=opp_id,
        title="Zero Leakage Opportunity",
        problem_statement="Test complete atomicity.",
        description="Verify 0 leaked rows.",
        status=OpportunityStatus.OPEN.value,
        confidence=0.9,
        created_at=datetime.datetime(2022, 1, 1, tzinfo=datetime.timezone.utc),
    )
    session1.add(opp)
    session1.commit()

    passage_text = "Zero leakage verification text."
    mock_llm = ConfigurableMockLLM(
        classification=RelationshipClassification(
            relationship=OpportunityRelationship.ADDRESSES,
            confidence=0.95,
            rationale="Addresses bounds.",
            evidence_excerpt=passage_text,
        )
    )
    classifier = RelationshipClassifier(provider=mock_llm)
    trans_svc = OpportunityTransitionService(session=session1)
    integration = OpportunityClassificationIntegration(classifier=classifier, transition_service=trans_svc, session=session1)
    evolution_svc = IncrementalOpportunityEvolutionService(integration=integration, session=session1)

    service = OpenAlexPaperIngestionService(
        persistence=persistence1,
        auto_evolve=True,
        evolution_service=evolution_svc,
    )

    pdf_bytes = create_sample_pdf_bytes(title="Leak Test Paper", body=passage_text)
    work = make_sample_work(
        source_id="https://openalex.org/W_STAGE3C_5",
        doi="https://doi.org/10.1000/s3c_5",
        pub_date="2023-01-01",
    )

    try:
        with persistence1.transaction():
            with patch.object(service.downloader, "_open_request", return_value=DummyMockResponse(pdf_bytes)):
                res = service.ingest_work(work)
            raise ValueError("Aborting transaction immediately")
    except ValueError:
        pass

    session1.close()

    fresh_session = test_db_factory()
    try:
        assert fresh_session.execute(select(Paper).where(Paper.source_id == "W_STAGE3C_5")).scalar_one_or_none() is None
        assert fresh_session.execute(select(PaperDocument)).scalar_one_or_none() is None
        assert len(fresh_session.execute(select(PaperPassage)).scalars().all()) == 0
        assert len(fresh_session.execute(select(OpportunityEvidence)).scalars().all()) == 0
        assert len(fresh_session.execute(select(OpportunityHistory)).scalars().all()) == 0
    finally:
        fresh_session.close()


# ── TEST 6: COMMIT FAILURE REPORTS EVOLUTION FAILED AND PRESERVES DIAGNOSTIC ──

def test_6_commit_failure_reports_evolution_failed_and_preserves_paper(test_db_factory):
    """Verifies that if session.commit() fails during standalone evolution, evolution reports FAILED, paper remains INGESTED."""
    session1 = test_db_factory()
    persistence1 = SQLAlchemyPersistenceLayer(session1)

    opp_id = uuid.uuid4()
    opp = Opportunity(
        id=opp_id,
        title="Commit Failure Opportunity",
        problem_statement="Test commit failure handling.",
        description="Verify failure handling on commit exception.",
        status=OpportunityStatus.OPEN.value,
        confidence=0.9,
        created_at=datetime.datetime(2022, 1, 1, tzinfo=datetime.timezone.utc),
    )
    session1.add(opp)
    session1.commit()

    passage_text = "Commit failure test text."
    mock_llm = ConfigurableMockLLM(
        classification=RelationshipClassification(
            relationship=OpportunityRelationship.ADDRESSES,
            confidence=0.95,
            rationale="Addresses bounds.",
            evidence_excerpt=passage_text,
        )
    )
    classifier = RelationshipClassifier(provider=mock_llm)
    trans_svc = OpportunityTransitionService(session=session1)
    integration = OpportunityClassificationIntegration(classifier=classifier, transition_service=trans_svc, session=session1)
    evolution_svc = IncrementalOpportunityEvolutionService(integration=integration, session=session1)

    service = OpenAlexPaperIngestionService(
        persistence=persistence1,
        auto_evolve=True,
        evolution_service=evolution_svc,
    )

    pdf_bytes = create_sample_pdf_bytes(title="Commit Failure Paper", body=passage_text)
    work = make_sample_work(
        source_id="https://openalex.org/W_STAGE3C_6",
        doi="https://doi.org/10.1000/s3c_6",
        pub_date="2023-01-01",
    )

    # Force the standalone commit at the end of evolution hook to fail
    orig_commit = session1.commit
    def failing_commit():
        # First commit during paper ingestion succeeds; subsequent commit during evolution fails
        if not getattr(session1, "_paper_committed", False):
            session1._paper_committed = True
            return orig_commit()
        raise RuntimeError("Disk I/O error during final evolution commit")

    with patch.object(service.downloader, "_open_request", return_value=DummyMockResponse(pdf_bytes)):
        with patch.object(session1, "commit", side_effect=failing_commit):
            res = service.ingest_work(work)

    # Paper ingestion status MUST still be INGESTED!
    assert res.status == IngestionStatus.INGESTED
    assert res.paper_id is not None
    # Evolution outcome MUST report FAILED with accurate diagnostics
    assert res.evolution_status == EvolutionHookStatus.FAILED
    assert "Evolution commit failed" in str(res.evolution_result)
    assert "Disk I/O error during final evolution commit" in str(res.evolution_result)

    session1.close()

    # Verify that the paper remains intact in a fresh session!
    fresh_session = test_db_factory()
    try:
        db_paper = fresh_session.execute(select(Paper).where(Paper.id == res.paper_id)).scalar_one_or_none()
        assert db_paper is not None
        assert db_paper.source_id == "W_STAGE3C_6"
    finally:
        fresh_session.close()


# ── TEST 7: DISABLED-HOOK AND MANUAL PDF INGESTION UNCHANGED ──────────────────

def test_7_disabled_hook_and_manual_pdf_ingestion_unchanged(test_db_factory, tmp_path: Path):
    """Verifies that disabled hook and ManualPDFIngestionService remain completely unchanged."""
    session = test_db_factory()
    persistence = SQLAlchemyPersistenceLayer(session)

    # 7a. Disabled hook by default
    service = OpenAlexPaperIngestionService(
        persistence=persistence,
        auto_evolve=False,
    )
    pdf_bytes = create_sample_pdf_bytes(title="Disabled Hook Paper")
    work = make_sample_work(
        source_id="https://openalex.org/W_STAGE3C_7",
        doi="https://doi.org/10.1000/s3c_7",
    )
    with patch.object(service.downloader, "_open_request", return_value=DummyMockResponse(pdf_bytes)):
        res = service.ingest_work(work)

    assert res.status == IngestionStatus.INGESTED
    assert res.evolution_status == EvolutionHookStatus.DISABLED
    assert res.evolution_result is None

    # 7b. Manual PDF Ingestion
    manual_service = ManualPDFIngestionService(persistence=persistence)
    pdf_path = tmp_path / "manual_test.pdf"
    pdf_path.write_bytes(pdf_bytes)

    paper_dto, doc_dto = manual_service.ingest_pdf(pdf_path, title="Manual Title", batch_id="batch_manual")
    assert paper_dto.title == "Manual Title"
    assert doc_dto.full_text is not None

    session.close()
