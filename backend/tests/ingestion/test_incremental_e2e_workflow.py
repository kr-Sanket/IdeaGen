"""End-to-End Incremental Workflow Validation Suite (Stage 3B, Part 3).

Comprehensive offline integration validation across:
1. Successful ingestion and post-ingestion evolution (committed records, evidence, history, statuses).
2. Duplicate and idempotent processing (source ID duplicate, content duplicate, no duplicate evidence/history).
3. Incremental multi-paper lifecycle transitions & contradiction reopening (OPEN -> PARTIALLY_ADDRESSED -> RESOLVED -> OPEN).
4. Evolution failure isolation (provider failure leaves paper/document/passages intact, no partial evidence).
5. Temporal consistency and eligibility enforcement (opp_date > pub_date skipped, undated skipped, NO_PDF skipped, no future-data leakage).
"""

from __future__ import annotations

import datetime
import io
from pathlib import Path
from typing import Dict, Any, Optional, List
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
    ResearchOpportunityRecord,
)
from ideagen.db.persistence import SQLAlchemyPersistenceLayer
from ideagen.ingestion.models import SourceType
from ideagen.ingestion.openalex_ingestion import (
    OpenAlexPaperIngestionService,
    IngestionStatus,
    EvolutionHookStatus,
    PaperIngestionResult,
)
from ideagen.ingestion.pdf import DuplicateDocumentError
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
    OpportunityEvaluationStatus,
)
from ideagen.llm.provider import LLMProvider
from ideagen.llm.exceptions import ProviderFailureError


# ── TEST FIXTURES & HELPERS ───────────────────────────────────────────────────

def create_sample_pdf_bytes(title: str = "Test Title", body: str = "Detailed scientific research text. ") -> bytes:
    """Generates valid in-memory PDF bytes with > 500 characters of text."""
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
    """Simulates an urllib response stream for offline download mocking."""
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
    source_id: str = "https://openalex.org/W1000000001",
    doi: str = "https://doi.org/10.1000/1000000001",
    title: str = "Novel Dynamic State Machine Consensus",
    pdf_url: Optional[str] = "https://example.com/paper.pdf",
    pub_date: Optional[str] = "2023-01-01",
) -> Dict[str, Any]:
    """Generates an OpenAlex work dictionary."""
    return {
        "id": source_id,
        "doi": doi,
        "title": title,
        "publication_date": pub_date,
        "abstract_inverted_index": {
            "We": [0], "prove": [1], "state": [2], "machine": [3], "consensus": [4], "safety.": [5]
        },
        "authorships": [{"author": {"display_name": "Dr. Researcher"}}],
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
def isolated_db():
    """In-memory SQLite database with foreign keys and unique constraints enforced."""
    engine = create_engine("sqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    session = session_factory()
    yield session
    session.close()


class ConfigurableMockLLMProvider(LLMProvider):
    """Deterministic LLM Provider that can dynamically route responses or simulate failures."""
    def __init__(self, default_classification: Optional[RelationshipClassification] = None):
        self.default_classification = default_classification
        self.routes: Dict[str, RelationshipClassification] = {}
        self.should_fail: bool = False
        self.failure_error: Exception = ProviderFailureError("Mock LLM Provider outage 503")

    def set_route(self, key_fragment: str, classification: RelationshipClassification):
        self.routes[key_fragment] = classification

    def generate_structured(self, prompt: str, response_schema):
        if self.should_fail:
            raise self.failure_error

        for key, classification in self.routes.items():
            if key in prompt:
                return classification

        if self.default_classification is not None:
            return self.default_classification

        raise ValueError(f"No mock response configured for prompt: {prompt[:100]}")


# ── SCENARIO 1: SUCCESSFUL INGESTION AND EVOLUTION ─────────────────────────────

def test_scenario_1_successful_ingestion_and_evolution(isolated_db: Session):
    """Ingests a valid paper, commits paper/doc/passages, and evolves an eligible opportunity."""
    session = isolated_db
    persistence = SQLAlchemyPersistenceLayer(session)

    # 1. Pre-seed Opportunity created on 2022-01-01
    opp_id = uuid.uuid4()
    opp = Opportunity(
        id=opp_id,
        title="Partition-Tolerant Consensus Verification",
        problem_statement="High latency degrades partition tolerance.",
        description="Formal bounds for partition tolerant consensus.",
        status=OpportunityStatus.OPEN.value,
        confidence=0.9,
        created_at=datetime.datetime(2022, 1, 1, tzinfo=datetime.timezone.utc),
    )
    session.add(opp)
    session.commit()

    # 2. Build live incremental evolution service with deterministic mock LLM
    passage_body = "We prove Byzantine fault tolerant safety under network delay."
    mock_llm = ConfigurableMockLLMProvider(
        default_classification=RelationshipClassification(
            relationship=OpportunityRelationship.ADDRESSES,
            confidence=0.96,
            rationale="Addresses partition-tolerant consensus safety.",
            evidence_excerpt=passage_body,
        )
    )
    classifier = RelationshipClassifier(provider=mock_llm)
    trans_svc = OpportunityTransitionService(session=session)
    integration = OpportunityClassificationIntegration(classifier=classifier, transition_service=trans_svc, session=session)
    evolution_svc = IncrementalOpportunityEvolutionService(integration=integration, session=session)

    # 3. Create ingestion service with auto_evolve=True
    service = OpenAlexPaperIngestionService(
        persistence=persistence,
        auto_evolve=True,
        evolution_service=evolution_svc,
    )

    pdf_bytes = create_sample_pdf_bytes(title="Byzantine Consensus Under Delay", body=passage_body)
    work = make_sample_work(
        source_id="https://openalex.org/W_SCENARIO_1",
        doi="https://doi.org/10.1000/scen1",
        pub_date="2023-01-01",
    )

    with patch.object(service.downloader, "_open_request", return_value=DummyMockResponse(pdf_bytes)):
        res = service.ingest_work(work)

    # Ingestion checks
    assert res.status == IngestionStatus.INGESTED
    assert res.has_pdf is True
    assert res.paper_id is not None
    assert res.evolution_status == EvolutionHookStatus.SUCCESS
    assert res.evolution_result.evaluated_count == 1
    assert res.evolution_result.run_status == IncrementalEvolutionRunStatus.SUCCESS

    # Database records checks
    paper = session.execute(select(Paper).where(Paper.id == res.paper_id)).scalar_one_or_none()
    assert paper is not None
    assert paper.publication_date == datetime.date(2023, 1, 1)

    doc = session.execute(select(PaperDocument).where(PaperDocument.paper_id == res.paper_id)).scalar_one_or_none()
    assert doc is not None

    passages = session.execute(select(PaperPassage).where(PaperPassage.paper_id == res.paper_id)).scalars().all()
    assert len(passages) > 0

    # Opportunity lifecycle check: OPEN -> PARTIALLY_ADDRESSED
    session.refresh(opp)
    assert opp.status == OpportunityStatus.PARTIALLY_ADDRESSED.value

    # Evidence & History checks
    ev = session.execute(select(OpportunityEvidence).where(OpportunityEvidence.opportunity_id == opp_id)).scalar_one_or_none()
    assert ev is not None
    assert ev.paper_id == res.paper_id
    assert ev.evidence_type == OpportunityRelationship.ADDRESSES.value

    hist = session.execute(select(OpportunityHistory).where(OpportunityHistory.opportunity_id == opp_id)).scalars().all()
    assert len(hist) == 1
    assert hist[0].previous_status == OpportunityStatus.OPEN.value
    assert hist[0].new_status == OpportunityStatus.PARTIALLY_ADDRESSED.value


# ── SCENARIO 2: DUPLICATE / IDEMPOTENT PROCESSING ─────────────────────────────

def test_scenario_2_duplicate_and_idempotent_processing(isolated_db: Session):
    """Processes duplicate papers and verifies evolution is skipped with no duplicate evidence."""
    session = isolated_db
    persistence = SQLAlchemyPersistenceLayer(session)

    opp_id = uuid.uuid4()
    opp = Opportunity(
        id=opp_id,
        title="Deduplication Robustness Test",
        problem_statement="Test duplicate ingestion.",
        description="Verify idempotency constraints.",
        status=OpportunityStatus.OPEN.value,
        confidence=0.85,
        created_at=datetime.datetime(2022, 1, 1, tzinfo=datetime.timezone.utc),
    )
    session.add(opp)
    session.commit()

    passage_body = "Identical research content for idempotency testing."
    mock_llm = ConfigurableMockLLMProvider(
        default_classification=RelationshipClassification(
            relationship=OpportunityRelationship.ADDRESSES,
            confidence=0.92,
            rationale="Addresses idempotency.",
            evidence_excerpt=passage_body,
        )
    )
    classifier = RelationshipClassifier(provider=mock_llm)
    trans_svc = OpportunityTransitionService(session=session)
    integration = OpportunityClassificationIntegration(classifier=classifier, transition_service=trans_svc, session=session)
    evolution_svc = IncrementalOpportunityEvolutionService(integration=integration, session=session)

    service = OpenAlexPaperIngestionService(
        persistence=persistence,
        auto_evolve=True,
        evolution_service=evolution_svc,
    )

    pdf_bytes = create_sample_pdf_bytes(title="Idempotency Test Paper", body=passage_body)
    work = make_sample_work(
        source_id="https://openalex.org/W_DUP_1",
        doi="https://doi.org/10.1000/dup1",
        pub_date="2023-01-01",
    )

    # First ingestion: successful
    with patch.object(service.downloader, "_open_request", return_value=DummyMockResponse(pdf_bytes)):
        res1 = service.ingest_work(work, check_duplicates=True)

    assert res1.status == IngestionStatus.INGESTED
    assert res1.evolution_status == EvolutionHookStatus.SUCCESS
    orig_paper_id = res1.paper_id

    # Verify initial evidence count = 1
    ev_count_1 = len(session.execute(select(OpportunityEvidence).where(OpportunityEvidence.opportunity_id == opp_id)).scalars().all())
    assert ev_count_1 == 1

    # Second ingestion: duplicate work record
    with patch.object(service.downloader, "_open_request", return_value=DummyMockResponse(pdf_bytes)):
        res2 = service.ingest_work(work, check_duplicates=True)

    assert res2.status == IngestionStatus.SKIPPED_DUPLICATE
    assert res2.evolution_status == EvolutionHookStatus.SKIPPED
    assert res2.paper_id == orig_paper_id

    # Verify evidence and history counts are unchanged
    ev_count_2 = len(session.execute(select(OpportunityEvidence).where(OpportunityEvidence.opportunity_id == opp_id)).scalars().all())
    assert ev_count_2 == 1

    hist_count_2 = len(session.execute(select(OpportunityHistory).where(OpportunityHistory.opportunity_id == opp_id)).scalars().all())
    assert hist_count_2 == 1

    # Third ingestion: different source ID & DOI but identical PDF content hash
    work_content_dup = make_sample_work(
        source_id="https://openalex.org/W_DUP_DIFFERENT_METADATA",
        doi="https://doi.org/10.1000/different_doi_same_pdf",
        pub_date="2023-02-01",
    )
    with patch.object(service.downloader, "_open_request", return_value=DummyMockResponse(pdf_bytes)):
        with pytest.raises(DuplicateDocumentError):
            service.ingest_work(work_content_dup, check_duplicates=True)

    # Ensure still no duplicate evidence or papers leaked
    ev_count_3 = len(session.execute(select(OpportunityEvidence).where(OpportunityEvidence.opportunity_id == opp_id)).scalars().all())
    assert ev_count_3 == 1


# ── SCENARIO 3: INCREMENTAL LIFECYCLE TRANSITION & REOPENING ───────────────────

def test_scenario_3_incremental_lifecycle_transition_and_reopening(isolated_db: Session):
    """Progresses opportunity across multi-paper lifecycle: OPEN -> PARTIALLY_ADDRESSED -> RESOLVED -> OPEN."""
    session = isolated_db
    persistence = SQLAlchemyPersistenceLayer(session)

    # Seed opportunity created in 2021
    opp_id = uuid.uuid4()
    opp = Opportunity(
        id=opp_id,
        title="Scalable Multi-Agent State Synchronization",
        problem_statement="Multi-agent synchronization lacks sub-second bounds.",
        description="Investigating sub-second state synchronization.",
        status=OpportunityStatus.OPEN.value,
        confidence=0.9,
        created_at=datetime.datetime(2021, 1, 1, tzinfo=datetime.timezone.utc),
    )
    session.add(opp)
    session.commit()

    mock_llm = ConfigurableMockLLMProvider()
    classifier = RelationshipClassifier(provider=mock_llm)
    trans_svc = OpportunityTransitionService(session=session)
    integration = OpportunityClassificationIntegration(classifier=classifier, transition_service=trans_svc, session=session)
    evolution_svc = IncrementalOpportunityEvolutionService(integration=integration, session=session)

    service = OpenAlexPaperIngestionService(
        persistence=persistence,
        auto_evolve=True,
        evolution_service=evolution_svc,
    )

    # ── Step 3A: Paper 1 (ADDRESSES) -> Opportunity transitions OPEN -> PARTIALLY_ADDRESSED
    excerpt_1 = "Paper 1 proves sub-second bounds for 100 agents."
    mock_llm.set_route(
        "Paper 1",
        RelationshipClassification(
            relationship=OpportunityRelationship.ADDRESSES,
            confidence=0.95,
            rationale="Addresses first half of sub-second synchronization.",
            evidence_excerpt=excerpt_1,
        ),
    )
    pdf_bytes_1 = create_sample_pdf_bytes(title="Paper 1 Title", body=excerpt_1)
    work_1 = make_sample_work(
        source_id="https://openalex.org/W_STEP_1",
        doi="https://doi.org/10.1000/step1",
        pub_date="2022-01-01",
    )
    with patch.object(service.downloader, "_open_request", return_value=DummyMockResponse(pdf_bytes_1)):
        res1 = service.ingest_work(work_1)

    assert res1.status == IngestionStatus.INGESTED
    assert res1.evolution_status == EvolutionHookStatus.SUCCESS
    session.refresh(opp)
    assert opp.status == OpportunityStatus.PARTIALLY_ADDRESSED.value

    # ── Step 3B: Paper 2 (ADDRESSES) -> 2 distinct papers transition PARTIALLY_ADDRESSED -> RESOLVED
    excerpt_2 = "Paper 2 generalizes synchronization bounds to 10,000 agents."
    mock_llm.set_route(
        "Paper 2",
        RelationshipClassification(
            relationship=OpportunityRelationship.ADDRESSES,
            confidence=0.97,
            rationale="Conclusively addresses synchronization scalability.",
            evidence_excerpt=excerpt_2,
        ),
    )
    pdf_bytes_2 = create_sample_pdf_bytes(title="Paper 2 Title", body=excerpt_2)
    work_2 = make_sample_work(
        source_id="https://openalex.org/W_STEP_2",
        doi="https://doi.org/10.1000/step2",
        pub_date="2022-06-01",
    )
    with patch.object(service.downloader, "_open_request", return_value=DummyMockResponse(pdf_bytes_2)):
        res2 = service.ingest_work(work_2)

    assert res2.status == IngestionStatus.INGESTED
    assert res2.evolution_status == EvolutionHookStatus.SUCCESS
    session.refresh(opp)
    assert opp.status == OpportunityStatus.RESOLVED.value

    # Verify 2 distinct evidence records and 2 history records
    ev_records = session.execute(
        select(OpportunityEvidence).where(OpportunityEvidence.opportunity_id == opp_id).order_by(OpportunityEvidence.created_at.asc())
    ).scalars().all()
    assert len(ev_records) == 2
    assert {ev.paper_id for ev in ev_records} == {res1.paper_id, res2.paper_id}

    hist_records = session.execute(
        select(OpportunityHistory).where(OpportunityHistory.opportunity_id == opp_id).order_by(OpportunityHistory.created_at.asc())
    ).scalars().all()
    assert len(hist_records) == 2
    assert (hist_records[0].previous_status, hist_records[0].new_status) == (OpportunityStatus.OPEN.value, OpportunityStatus.PARTIALLY_ADDRESSED.value)
    assert (hist_records[1].previous_status, hist_records[1].new_status) == (OpportunityStatus.PARTIALLY_ADDRESSED.value, OpportunityStatus.RESOLVED.value)

    # ── Step 3C: Paper 3 (CONTRADICTS) -> Re-opens RESOLVED -> OPEN
    excerpt_3 = "Paper 3 refutes the synchronization claims under Byzantine partitions."
    mock_llm.set_route(
        "Paper 3",
        RelationshipClassification(
            relationship=OpportunityRelationship.CONTRADICTS,
            confidence=0.98,
            rationale="Contradicts earlier proofs under adversarial network models.",
            evidence_excerpt=excerpt_3,
        ),
    )
    pdf_bytes_3 = create_sample_pdf_bytes(title="Paper 3 Title", body=excerpt_3)
    work_3 = make_sample_work(
        source_id="https://openalex.org/W_STEP_3",
        doi="https://doi.org/10.1000/step3",
        pub_date="2023-01-01",
    )
    with patch.object(service.downloader, "_open_request", return_value=DummyMockResponse(pdf_bytes_3)):
        res3 = service.ingest_work(work_3)

    assert res3.status == IngestionStatus.INGESTED
    assert res3.evolution_status == EvolutionHookStatus.SUCCESS
    session.refresh(opp)
    assert opp.status == OpportunityStatus.OPEN.value

    # Verify 3 distinct evidence records and 3 history records
    all_ev = session.execute(
        select(OpportunityEvidence).where(OpportunityEvidence.opportunity_id == opp_id)
    ).scalars().all()
    assert len(all_ev) == 3

    all_hist = session.execute(
        select(OpportunityHistory).where(OpportunityHistory.opportunity_id == opp_id).order_by(OpportunityHistory.created_at.asc())
    ).scalars().all()
    assert len(all_hist) == 3
    assert (all_hist[2].previous_status, all_hist[2].new_status) == (OpportunityStatus.RESOLVED.value, OpportunityStatus.OPEN.value)


# ── SCENARIO 4: EVOLUTION FAILURE ISOLATION ────────────────────────────────────

def test_scenario_4_evolution_failure_isolation(isolated_db: Session):
    """Simulates provider crash during evolution: paper remains intact, status=FAILED, no partial writes."""
    session = isolated_db
    persistence = SQLAlchemyPersistenceLayer(session)

    opp_id = uuid.uuid4()
    opp = Opportunity(
        id=opp_id,
        title="Failure Isolation Opportunity",
        problem_statement="Test error boundary robustness.",
        description="Verify database integrity on LLM crash.",
        status=OpportunityStatus.OPEN.value,
        confidence=0.88,
        created_at=datetime.datetime(2022, 1, 1, tzinfo=datetime.timezone.utc),
    )
    session.add(opp)
    session.commit()

    # Configure mock LLM to throw provider outage exception
    mock_llm = ConfigurableMockLLMProvider()
    mock_llm.should_fail = True
    mock_llm.failure_error = ProviderFailureError("HTTP 503 LLM cluster overloaded")

    classifier = RelationshipClassifier(provider=mock_llm)
    trans_svc = OpportunityTransitionService(session=session)
    integration = OpportunityClassificationIntegration(classifier=classifier, transition_service=trans_svc, session=session)
    evolution_svc = IncrementalOpportunityEvolutionService(integration=integration, session=session)

    service = OpenAlexPaperIngestionService(
        persistence=persistence,
        auto_evolve=True,
        evolution_service=evolution_svc,
    )

    pdf_bytes = create_sample_pdf_bytes(title="Failure Isolation Paper", body="Normal scientific text.")
    work = make_sample_work(
        source_id="https://openalex.org/W_FAIL_ISO_1",
        doi="https://doi.org/10.1000/failiso1",
        pub_date="2023-01-01",
    )

    with patch.object(service.downloader, "_open_request", return_value=DummyMockResponse(pdf_bytes)):
        res = service.ingest_work(work)

    # Ingestion MUST NOT fail
    assert res.status == IngestionStatus.INGESTED
    assert res.has_pdf is True
    assert res.paper_id is not None
    assert res.evolution_status == EvolutionHookStatus.FAILED
    assert res.evolution_result.failed_count == 1
    assert "HTTP 503 LLM cluster overloaded" in str(res.evolution_result)

    # Paper, document, and passages MUST still exist in database
    db_paper = session.execute(select(Paper).where(Paper.id == res.paper_id)).scalar_one_or_none()
    assert db_paper is not None
    assert db_paper.source_id == "W_FAIL_ISO_1"

    db_doc = session.execute(select(PaperDocument).where(PaperDocument.paper_id == res.paper_id)).scalar_one_or_none()
    assert db_doc is not None

    db_passages = session.execute(select(PaperPassage).where(PaperPassage.paper_id == res.paper_id)).scalars().all()
    assert len(db_passages) > 0

    # Opportunity must remain completely OPEN with 0 evidence and 0 history records
    session.refresh(opp)
    assert opp.status == OpportunityStatus.OPEN.value

    ev = session.execute(select(OpportunityEvidence).where(OpportunityEvidence.opportunity_id == opp_id)).scalars().all()
    assert len(ev) == 0

    hist = session.execute(select(OpportunityHistory).where(OpportunityHistory.opportunity_id == opp_id)).scalars().all()
    assert len(hist) == 0


# ── SCENARIO 5: TEMPORAL AND ELIGIBILITY CORRECTNESS ──────────────────────────

def test_scenario_5_temporal_and_eligibility_correctness(isolated_db: Session):
    """Verifies temporal boundary (no future leakage), missing content skips, and bridge eligibility."""
    session = isolated_db
    persistence = SQLAlchemyPersistenceLayer(session)

    # 5a. Opportunity created in future (2024-01-01) relative to paper (2022-01-01)
    opp_future_id = uuid.uuid4()
    opp_future = Opportunity(
        id=opp_future_id,
        title="Future Opportunity",
        problem_statement="Created after paper was published.",
        description="Temporal boundary check.",
        status=OpportunityStatus.OPEN.value,
        confidence=0.8,
        created_at=datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc),
    )
    session.add(opp_future)
    session.commit()

    mock_llm = ConfigurableMockLLMProvider(
        default_classification=RelationshipClassification(
            relationship=OpportunityRelationship.ADDRESSES,
            confidence=0.9,
            rationale="Would address if temporal rules allowed.",
            evidence_excerpt="Common ground truth.",
        )
    )
    classifier = RelationshipClassifier(provider=mock_llm)
    trans_svc = OpportunityTransitionService(session=session)
    integration = OpportunityClassificationIntegration(classifier=classifier, transition_service=trans_svc, session=session)
    evolution_svc = IncrementalOpportunityEvolutionService(integration=integration, session=session)

    service = OpenAlexPaperIngestionService(
        persistence=persistence,
        auto_evolve=True,
        evolution_service=evolution_svc,
    )

    pdf_bytes = create_sample_pdf_bytes(title="Past Paper", body="Common ground truth.")
    work_past = make_sample_work(
        source_id="https://openalex.org/W_PAST_1",
        doi="https://doi.org/10.1000/past1",
        pub_date="2022-01-01",  # Strictly BEFORE opp_future.created_at (2024-01-01)
    )

    with patch.object(service.downloader, "_open_request", return_value=DummyMockResponse(pdf_bytes)):
        res_past = service.ingest_work(work_past)

    # Paper is ingested, but hook outcome is SKIPPED due to temporal boundary
    assert res_past.status == IngestionStatus.INGESTED
    assert res_past.evolution_status == EvolutionHookStatus.SKIPPED
    assert res_past.evolution_result.evaluated_count == 0
    assert res_past.evolution_result.skipped_count == 1
    assert "is after paper publication date" in res_past.evolution_result.outcomes[0].skip_reason

    # Zero evidence or history for the future opportunity
    ev_future = session.execute(select(OpportunityEvidence).where(OpportunityEvidence.opportunity_id == opp_future_id)).scalars().all()
    assert len(ev_future) == 0

    # 5b. Parser rejects OpenAlex work with missing publication date
    work_undated = make_sample_work(
        source_id="https://openalex.org/W_UNDATED_1",
        doi="https://doi.org/10.1000/undated1",
        pub_date=None,
    )
    with pytest.raises(ValueError, match="Missing publication date"):
        service.ingest_work(work_undated)

    # 5c. Evolution service directly skips undated paper in database
    undated_paper = Paper(
        id=uuid.uuid4(),
        source_type=SourceType.OPENALEX_API.value,
        source_id="W_DB_UNDATED",
        title="Undated Database Paper",
        publication_date=None,  # Missing publication date in DB
        batch_id="batch_0",
        abstract="Valid abstract but missing publication date.",
        authors=[],
        referenced_works=[],
        raw_metadata={},
    )
    session.add(undated_paper)
    session.commit()

    evo_res_undated = evolution_svc.evaluate_paper(paper_id=undated_paper.id)
    assert evo_res_undated.run_status == IncrementalEvolutionRunStatus.SKIPPED_UNDATED
    assert "undated" in evo_res_undated.error

    # 5d. NO_PDF record: metadata-only skips evolution cleanly
    work_no_pdf = make_sample_work(
        source_id="https://openalex.org/W_NOPDF_1",
        doi="https://doi.org/10.1000/nopdf1",
        pdf_url=None,
    )
    res_no_pdf = service.ingest_work(work_no_pdf)
    assert res_no_pdf.status == IngestionStatus.NO_PDF
    assert res_no_pdf.evolution_status == EvolutionHookStatus.SKIPPED
    assert res_no_pdf.evolution_result is None
