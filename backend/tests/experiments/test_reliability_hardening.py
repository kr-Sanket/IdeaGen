"""Comprehensive test suite for Checkpoint 13: Experiment Reliability Hardening.

Covers:
1. SQLite foreign key enforcement on new connections and rejection of invalid FK inserts.
2. Dialect isolation (PostgreSQL unaffected).
3. LLM timeout, transient error retry, and exponential backoff.
4. Permanent error fail-closed behavior (no indiscriminate retries).
5. Retry exhaustion reporting as explicit failure.
6. Zero duplicate evidence or history records on retries.
7. Batch chunking with deterministic order and cutoff boundaries.
8. Offline corpus-date preflight (missing, malformed, duplicates, date range, impact).
9. Stage 5 remains disabled by default.
"""

from __future__ import annotations

import datetime
import json
import uuid
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest
from sqlalchemy import create_engine, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from ideagen.db.models import (
    Base,
    Paper,
    Opportunity,
    OpportunityEvidence,
    OpportunityHistory,
    ResearchOpportunityRecord,
)
from ideagen.db.session import configure_sqlite_foreign_keys
from ideagen.ingestion.preflight import run_corpus_preflight, CorpusPreflightReport
from ideagen.llm.exceptions import (
    ProviderFailureError,
    MalformedResponseError,
    MissingInputError,
)
from ideagen.opportunity.classifier import (
    RelationshipClassifier,
    RelationshipClassificationInput,
    RelationshipClassification,
)
from ideagen.opportunity.models import OpportunityRelationship, OpportunityStatus
from ideagen.opportunity.service import OpportunityTransitionService
from ideagen.opportunity.integration import OpportunityClassificationIntegration
from ideagen.opportunity.temporal import (
    TemporalBatch,
    OpportunityEvolutionService,
)
from experiments.runner import (
    ControlledExperimentRunner,
    compute_configuration_fingerprint,
    compare_fingerprints,
)


# ─── TASK A: SQLite Foreign-Key Enforcement ───────────────────────────────────

def test_sqlite_foreign_keys_pragma_enabled():
    """Verify configure_sqlite_foreign_keys enables PRAGMA foreign_keys on SQLite."""
    engine = create_engine("sqlite:///:memory:")
    configure_sqlite_foreign_keys(engine)

    with engine.connect() as conn:
        res = conn.execute(text("PRAGMA foreign_keys;")).scalar()
        assert res == 1, "PRAGMA foreign_keys must be ON (1)"


def test_sqlite_invalid_foreign_key_insert_fails():
    """Verify that with foreign keys enabled, inserting an orphan FK raises IntegrityError."""
    engine = create_engine("sqlite:///:memory:")
    configure_sqlite_foreign_keys(engine)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Create a valid Opportunity
    opp = Opportunity(
        id=uuid.uuid4(),
        title="Opp",
        problem_statement="Prob",
        description="Desc",
        status="OPEN",
    )
    session.add(opp)
    session.commit()

    # Attempt to insert OpportunityEvidence referencing valid opportunity but non-existent paper
    orphan_evidence = OpportunityEvidence(
        opportunity_id=opp.id,
        paper_id=uuid.uuid4(),  # Does not exist in papers table
        evidence_type="ADDRESSES",
        supporting_text="Non-existent paper excerpt",
    )
    session.add(orphan_evidence)

    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()
    session.close()


def test_postgresql_dialect_check_unaffected():
    """Verify non-SQLite engines (e.g. mock PostgreSQL) do not register SQLite pragma."""
    mock_engine = MagicMock()
    mock_engine.dialect.name = "postgresql"

    configure_sqlite_foreign_keys(mock_engine)
    # No connect listeners registered on non-sqlite engine
    assert not mock_engine.connect.called


# ─── TASK B: LLM Reliability, Timeouts, Retries, and Chunking ────────────────

def _create_test_paper(
    session,
    pid: Optional[uuid.UUID] = None,
    title: str = "Paper",
    pub_date: datetime.date = datetime.date(2022, 1, 1),
    abstract: str = "Abstract text",
) -> Paper:
    paper_id = pid or uuid.uuid4()
    p = Paper(
        id=paper_id,
        source_type="local",
        source_id=str(paper_id),
        doi=f"10.1000/{paper_id}",
        title=title,
        publication_date=pub_date,
        batch_id="test-batch",
        abstract=abstract,
        authors=["Author A"],
        referenced_works=[],
        raw_metadata={},
    )
    session.add(p)
    return p


class FlakyLLMProvider:
    """Mock provider that fails with transient errors N times before succeeding."""

    def __init__(self, failures_before_success: int = 1, relationship: str = "SUPPORTS", excerpt: str = "Valid evidence"):
        self.failures_before_success = failures_before_success
        self.attempts = 0
        self.relationship = relationship
        self.excerpt = excerpt
        self.received_timeouts = []

    def generate_structured(self, prompt: str, schema: Any, timeout: Optional[float] = None) -> Any:
        self.attempts += 1
        self.received_timeouts.append(timeout)
        if self.attempts <= self.failures_before_success:
            raise ProviderFailureError(f"Transient connection failure attempt {self.attempts}")
        return schema(
            relationship=self.relationship,
            confidence=0.85,
            rationale="Transient recovery test",
            evidence_excerpt=self.excerpt,
        )


class PermanentFailingLLMProvider:
    """Mock provider that raises permanent errors."""

    def __init__(self, error_type: str = "malformed"):
        self.attempts = 0
        self.error_type = error_type

    def generate_structured(self, prompt: str, schema: Any, timeout: Optional[float] = None) -> Any:
        self.attempts += 1
        if self.error_type == "malformed":
            raise MalformedResponseError("Unparseable output")
        raise ProviderFailureError("Permanent provider down")


def test_relationship_classifier_retries_transient_failure_and_succeeds():
    """Verify classifier retries transient failure up to max_retries with backoff."""
    provider = FlakyLLMProvider(failures_before_success=2)
    classifier = RelationshipClassifier(
        provider=provider,
        timeout=15.0,
        max_retries=3,
        backoff_factor=0.01,  # Fast in tests
    )

    input_data = RelationshipClassificationInput(
        opportunity_title="Opp A",
        opportunity_problem_statement="Prob A",
        opportunity_description="Desc A",
        paper_title="Paper B",
        paper_abstract="Valid evidence in abstract",
    )

    res = classifier.classify(input_data)
    assert res.relationship == OpportunityRelationship.SUPPORTS
    assert provider.attempts == 3
    assert all(t == 15.0 for t in provider.received_timeouts)


def test_relationship_classifier_does_not_retry_permanent_error():
    """Verify permanent errors (MalformedResponseError, MissingInputError) are NOT retried."""
    provider = PermanentFailingLLMProvider(error_type="malformed")
    classifier = RelationshipClassifier(
        provider=provider,
        max_retries=3,
        backoff_factor=0.01,
    )

    input_data = RelationshipClassificationInput(
        opportunity_title="Opp A",
        opportunity_problem_statement="Prob A",
        opportunity_description="Desc A",
        paper_title="Paper B",
        paper_abstract="Abstract",
    )

    with pytest.raises(MalformedResponseError):
        classifier.classify(input_data)
    # Permanent error fails immediately on attempt 1
    assert provider.attempts == 1


def test_retry_exhaustion_raises_provider_failure():
    """Verify that when retries are exhausted, explicit failure is raised."""
    provider = FlakyLLMProvider(failures_before_success=5)  # Needs 5, allowed 2
    classifier = RelationshipClassifier(
        provider=provider,
        max_retries=2,
        backoff_factor=0.01,
    )

    input_data = RelationshipClassificationInput(
        opportunity_title="Opp A",
        opportunity_problem_statement="Prob A",
        opportunity_description="Desc A",
        paper_title="Paper B",
        paper_abstract="Abstract",
    )

    with pytest.raises(ProviderFailureError) as exc_info:
        classifier.classify(input_data)
    assert "Classification failed after 3 attempts" in str(exc_info.value)
    assert provider.attempts == 3


def test_retrying_does_not_duplicate_evidence_or_history():
    """Verify that retrying transient failures does not create multiple evidence or history rows."""
    engine = create_engine("sqlite:///:memory:")
    configure_sqlite_foreign_keys(engine)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Seed paper and opportunity with matching excerpt
    p = _create_test_paper(
        session,
        title="Paper 1",
        pub_date=datetime.date(2022, 1, 1),
        abstract="Valid evidence in the abstract for grounding test.",
    )
    opp = Opportunity(id=uuid.uuid4(), title="Opp 1", problem_statement="Prob 1", description="Desc 1", status="OPEN")
    session.add(opp)
    session.commit()

    # Flaky provider fails on attempt 1, succeeds on attempt 2
    provider = FlakyLLMProvider(
        failures_before_success=1,
        relationship="ADDRESSES",
        excerpt="Valid evidence in the abstract",
    )
    classifier = RelationshipClassifier(provider=provider, max_retries=2, backoff_factor=0.01)
    transition_svc = OpportunityTransitionService(session)
    integration = OpportunityClassificationIntegration(classifier, transition_svc, session)

    res = integration.classify_and_apply(opportunity_id=opp.id, paper_id=p.id)
    session.commit()

    assert res.state_changed is True
    # Count rows in DB
    ev_count = session.query(OpportunityEvidence).count()
    hist_count = session.query(OpportunityHistory).count()
    assert ev_count == 1, "Exactly one evidence row must exist"
    assert hist_count == 1, "Exactly one history row must exist"
    session.close()


def test_batch_chunking_preserves_deterministic_order_and_results():
    """Verify chunked processing produces exact same transitions and events as unchunked."""
    fixed_paper_ids = [
        uuid.UUID(f"00000000-0000-0000-0000-00000000000{i}") for i in range(1, 5)
    ]
    fixed_opp_id = uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")

    def run_with_chunk_size(chunk_size: Optional[int]):
        engine = create_engine("sqlite:///:memory:")
        configure_sqlite_foreign_keys(engine)
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        session = Session()

        opp = Opportunity(id=fixed_opp_id, title="Opp", problem_statement="Prob", description="Desc", status="OPEN")
        session.add(opp)

        papers = [
            _create_test_paper(
                session,
                pid=fixed_paper_ids[i - 1],
                title=f"Paper {i}",
                pub_date=datetime.date(2022, i, 1),
                abstract=f"Abstract {i}",
            )
            for i in range(1, 5)
        ]
        session.commit()

        # Deterministic mock provider returning SUPPORTS
        mock_provider = MagicMock()
        mock_provider.generate_structured.return_value = RelationshipClassification(
            relationship=OpportunityRelationship.SUPPORTS,
            confidence=0.9,
            rationale="Test supports",
            evidence_excerpt=None,
        )
        classifier = RelationshipClassifier(provider=mock_provider, max_retries=0)
        transition_svc = OpportunityTransitionService(session)
        integration = OpportunityClassificationIntegration(classifier, transition_svc, session)
        svc = OpportunityEvolutionService(integration)

        batch = TemporalBatch(
            batch_id="b1",
            start_date=datetime.date(2022, 1, 1),
            end_date=datetime.date(2022, 12, 31),
            sequence_number=0,
        )

        res = svc.process_batch(
            db=session,
            opportunity_id=opp.id,
            batch=batch,
            papers=papers,
            chunk_size=chunk_size,
        )
        session.commit()
        session.close()
        return res

    res_unchunked = run_with_chunk_size(None)
    res_chunked = run_with_chunk_size(2)

    assert len(res_unchunked.events) == 4
    assert len(res_chunked.events) == 4
    assert [e.paper_id for e in res_unchunked.events] == [e.paper_id for e in res_chunked.events]
    assert [e.publication_date for e in res_unchunked.events] == [e.publication_date for e in res_chunked.events]
    assert res_unchunked.final_status == res_chunked.final_status


# ─── TASK C: Corpus-Date Preflight ────────────────────────────────────────────

def test_corpus_preflight_detects_dated_undated_malformed_and_duplicates(tmp_path: Path):
    """Verify offline preflight correctly audits corpus dates and duplicate identities."""
    # Create mock PDFs
    p1 = tmp_path / "paper1.pdf"
    p2 = tmp_path / "paper2.pdf"
    p3 = tmp_path / "paper3.pdf"
    p4 = tmp_path / "paper4.pdf"
    for p in [p1, p2, p3, p4]:
        p.write_bytes(b"%PDF-1.4 mock content")

    manifest = [
        {"filename": "paper1.pdf", "title": "Alpha Research", "publication_date": "2021-05-10", "doi": "10.1000/182"},
        {"filename": "paper2.pdf", "title": "Beta Study", "publication_date": None, "doi": "10.1000/183"},  # Pure undated
        {"filename": "paper3.pdf", "title": "Gamma Paper", "publication_date": "invalid-date-string", "doi": "10.1000/184"},  # Malformed
        {"filename": "paper4.pdf", "title": "Alpha Research", "publication_date": "2023-11-20", "doi": "10.1000/182"},  # Duplicate title & DOI
    ]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    report = run_corpus_preflight(tmp_path)

    assert report.total_papers == 4
    assert report.dated_papers_count == 2  # paper1 and paper4
    assert report.undated_papers_count == 1  # paper2
    assert report.malformed_dates_count == 1  # paper3
    assert report.malformed_dates[0]["filename"] == "paper3.pdf"
    assert "doi:10.1000/182" in report.duplicate_identities
    assert "title:alpha research" in report.duplicate_identities
    assert report.date_range["earliest"] == "2021-05-10"
    assert report.date_range["latest"] == "2023-11-20"
    assert report.distribution_by_year == {"2021": 1, "2023": 1}
    assert report.has_warnings is True
    assert "Under strict temporal isolation" in report.temporal_evaluation_impact


# ─── Runner Integration & Default Contracts ──────────────────────────────────

def test_runner_stage_5_disabled_by_default():
    """Verify ControlledExperimentRunner keeps Stage 5 disabled by default."""
    runner = ControlledExperimentRunner()
    assert runner.enable_stage_5 is False
    assert runner.stage_5_max_retries == 2
    assert runner.stage_5_backoff_factor == 0.5
    assert runner.stage_5_chunk_size is None


def test_fingerprint_includes_stage_5_reliability_settings():
    """Verify configuration fingerprint includes stage 5 chunking and retry settings."""
    fp1 = compute_configuration_fingerprint(
        llm_provider=None,
        llm_model="qwen3:8b",
        embedding_provider=None,
        embedding_model="nomic",
        embedding_dimension=768,
        max_passages=10,
        max_total_words=2000,
        extraction_schema_version="v1.0",
        similarity_threshold=0.65,
        min_supporting_evidence=1,
        enable_stage_5=True,
        stage_5_chunk_size=50,
        stage_5_max_retries=3,
    )
    assert fp1["stage_5_chunk_size"] == 50
    assert fp1["stage_5_max_retries"] == 3

    fp2 = compute_configuration_fingerprint(
        llm_provider=None,
        llm_model="qwen3:8b",
        embedding_provider=None,
        embedding_model="nomic",
        embedding_dimension=768,
        max_passages=10,
        max_total_words=2000,
        extraction_schema_version="v1.0",
        similarity_threshold=0.65,
        min_supporting_evidence=1,
        enable_stage_5=True,
        stage_5_chunk_size=100,  # Changed
        stage_5_max_retries=3,
    )
    mismatches = compare_fingerprints(fp1, fp2)
    assert any("Stage 5 chunk size mismatch" in m for m in mismatches)
