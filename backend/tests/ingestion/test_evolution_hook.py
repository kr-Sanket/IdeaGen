"""Focused offline tests for Stage 3B, Part 2: Configurable Post-Ingestion Evolution Hook.

Verifies:
1. Hook disabled by default preserves current ingestion behavior.
2. Hook enabled invokes evolution exactly once after successful persistence.
3. Evolution observes committed paper/document/passage records.
4. NO_PDF, missing/undated content, failed ingestion, and duplicate ingestion do not trigger inappropriate evolution.
5. Evolution provider failure leaves persisted paper records intact and produces an accurate outcome.
6. Partial failure across opportunities is reported accurately.
7. Existing manual PDF ingestion remains unchanged unless explicitly configured.
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
)
from ideagen.db.persistence import SQLAlchemyPersistenceLayer
from ideagen.ingestion.models import (
    PaperCreateDTO,
    PaperDocumentCreateDTO,
    PassageCreateDTO,
    SourceType,
)
from ideagen.ingestion.openalex_ingestion import (
    OpenAlexPaperIngestionService,
    IngestionStatus,
    EvolutionHookStatus,
    PaperIngestionResult,
)
from ideagen.ingestion.downloader import OpenAccessPDFDownloader
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
from ideagen.opportunity.ranking import TFIDFCandidateRanker, FIFOCandidateRanker
from ideagen.llm.provider import LLMProvider
from ideagen.llm.exceptions import ProviderFailureError


def create_sample_pdf_bytes(title: str = "Test Title", body: str = "Detailed scientific research text. ") -> bytes:
    """Generates valid, parseable in-memory PDF bytes with > 500 characters of text."""
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
    """Simulates an urllib response stream."""
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
    source_id: str = "https://openalex.org/W1234567890",
    doi: str = "https://doi.org/10.1000/123456",
    pdf_url: Optional[str] = "https://example.com/paper.pdf",
    pub_date: str = "2023-01-01",
) -> Dict[str, Any]:
    return {
        "id": source_id,
        "doi": doi,
        "title": "Novel Consensus Bounds Under Asymmetric Network Delay",
        "publication_date": pub_date,
        "abstract_inverted_index": {
            "We": [0], "prove": [1], "state": [2], "machine": [3], "consensus": [4], "safety.": [5]
        },
        "authorships": [{"author": {"display_name": "Author Alice"}}],
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


class MockLLMProvider(LLMProvider):
    def __init__(self, classification: RelationshipClassification):
        self.classification = classification
        self.call_count: int = 0

    def generate_structured(self, prompt: str, response_schema):
        self.call_count += 1
        return self.classification


# ── TEST 1: Hook disabled by default preserves current behavior ───────────────

def test_hook_disabled_by_default(isolated_db: Session):
    session = isolated_db
    persistence = SQLAlchemyPersistenceLayer(session)
    mock_evolution_svc = MagicMock(spec=IncrementalOpportunityEvolutionService)

    service = OpenAlexPaperIngestionService(
        persistence=persistence,
        auto_evolve=None,  # Defaults to settings (False)
        evolution_service=mock_evolution_svc,
    )

    pdf_bytes = create_sample_pdf_bytes()
    work = make_sample_work()

    with patch.object(service.downloader, "_open_request", return_value=DummyMockResponse(pdf_bytes)):
        result = service.ingest_work(work)

    assert result.status == IngestionStatus.INGESTED
    assert result.has_pdf is True
    assert result.evolution_status == EvolutionHookStatus.DISABLED
    assert result.evolution_result is None

    # Evolution service must NOT have been called
    mock_evolution_svc.evaluate_paper.assert_not_called()


# ── TEST 2: Hook enabled invokes evolution exactly once after persistence ─────

def test_hook_enabled_invokes_evolution_once_after_persistence(isolated_db: Session):
    session = isolated_db
    persistence = SQLAlchemyPersistenceLayer(session)

    mock_evolution_svc = MagicMock(spec=IncrementalOpportunityEvolutionService)
    mock_evolution_svc.evaluate_paper.return_value = IncrementalEvolutionResult(
        paper_id=uuid.uuid4(),
        run_status=IncrementalEvolutionRunStatus.SUCCESS,
        candidate_count=1,
        evaluated_count=1,
    )

    service = OpenAlexPaperIngestionService(
        persistence=persistence,
        auto_evolve=True,
        evolution_service=mock_evolution_svc,
    )

    pdf_bytes = create_sample_pdf_bytes()
    work = make_sample_work()

    with patch.object(service.downloader, "_open_request", return_value=DummyMockResponse(pdf_bytes)):
        result = service.ingest_work(work)

    assert result.status == IngestionStatus.INGESTED
    assert result.has_pdf is True
    assert result.evolution_status == EvolutionHookStatus.SUCCESS
    assert result.evolution_result is not None

    # Evolution service must have been called exactly once with the persisted paper_id
    mock_evolution_svc.evaluate_paper.assert_called_once_with(paper_id=result.paper_id)


# ── TEST 3: Evolution observes committed records ──────────────────────────────

def test_evolution_observes_committed_records_end_to_end(isolated_db: Session):
    session = isolated_db
    persistence = SQLAlchemyPersistenceLayer(session)

    # 1. Create pre-existing Opportunity in DB (created before the paper)
    opp = Opportunity(
        id=uuid.uuid4(),
        title="Scalable Consensus Under Partitions",
        problem_statement="Consensus degradation under delay.",
        description="State machine replication under partitions.",
        status=OpportunityStatus.OPEN.value,
        confidence=0.9,
        created_at=datetime.datetime(2022, 1, 1, tzinfo=datetime.timezone.utc),
    )
    session.add(opp)
    session.commit()

    # 2. Build real IncrementalOpportunityEvolutionService backed by the session
    passage_excerpt = "Detailed scientific research text."
    mock_llm = MockLLMProvider(
        classification=RelationshipClassification(
            relationship=OpportunityRelationship.ADDRESSES,
            confidence=0.95,
            rationale="Addresses consensus latency under partitions.",
            evidence_excerpt=passage_excerpt,
        )
    )
    classifier = RelationshipClassifier(provider=mock_llm)
    trans_svc = OpportunityTransitionService(session=session)
    integration = OpportunityClassificationIntegration(classifier, trans_svc, session=session)
    evolution_svc = IncrementalOpportunityEvolutionService(integration=integration, session=session)

    # 3. Create Ingestion Service with auto_evolve=True
    service = OpenAlexPaperIngestionService(
        persistence=persistence,
        auto_evolve=True,
        evolution_service=evolution_svc,
    )

    pdf_bytes = create_sample_pdf_bytes(body=passage_excerpt)
    work = make_sample_work(pub_date="2023-01-01")

    with patch.object(service.downloader, "_open_request", return_value=DummyMockResponse(pdf_bytes)):
        result = service.ingest_work(work)

    assert result.status == IngestionStatus.INGESTED
    assert result.has_pdf is True
    assert result.evolution_status == EvolutionHookStatus.SUCCESS
    assert result.evolution_result.evaluated_count == 1

    # Verify that Opportunity was transitioned to PARTIALLY_ADDRESSED
    session.refresh(opp)
    assert opp.status == OpportunityStatus.PARTIALLY_ADDRESSED.value

    # Verify Evidence and History records exist in DB
    ev = session.execute(
        select(OpportunityEvidence).where(OpportunityEvidence.opportunity_id == opp.id)
    ).scalar_one_or_none()
    assert ev is not None
    assert ev.paper_id == result.paper_id
    assert ev.evidence_type == OpportunityRelationship.ADDRESSES.value

    hist = session.execute(
        select(OpportunityHistory).where(OpportunityHistory.opportunity_id == opp.id)
    ).scalars().all()
    assert len(hist) == 1
    assert hist[0].previous_status == OpportunityStatus.OPEN.value
    assert hist[0].new_status == OpportunityStatus.PARTIALLY_ADDRESSED.value


# ── TEST 4: NO_PDF, duplicate, and failed ingestion do not trigger evolution ──

def test_no_pdf_duplicate_and_failed_ingestion_do_not_trigger_evolution(isolated_db: Session):
    session = isolated_db
    persistence = SQLAlchemyPersistenceLayer(session)
    mock_evolution_svc = MagicMock(spec=IncrementalOpportunityEvolutionService)

    service = OpenAlexPaperIngestionService(
        persistence=persistence,
        auto_evolve=True,
        evolution_service=mock_evolution_svc,
    )

    # 4a. NO_PDF work (no pdf_url in metadata)
    work_no_pdf = make_sample_work(
        source_id="https://openalex.org/W_NO_PDF_1",
        doi="https://doi.org/10.1000/nopdf1",
        pdf_url=None,
    )
    res_no_pdf = service.ingest_work(work_no_pdf)
    assert res_no_pdf.status == IngestionStatus.NO_PDF
    assert res_no_pdf.evolution_status == EvolutionHookStatus.SKIPPED
    mock_evolution_svc.evaluate_paper.assert_not_called()

    # 4b. Duplicate work
    res_duplicate = service.ingest_work(work_no_pdf, check_duplicates=True)
    assert res_duplicate.status == IngestionStatus.SKIPPED_DUPLICATE
    assert res_duplicate.evolution_status == EvolutionHookStatus.SKIPPED
    mock_evolution_svc.evaluate_paper.assert_not_called()

    # 4c. Ingestion persistence error
    with patch.object(persistence, "save_paper", side_effect=RuntimeError("Disk full")):
        work_err = make_sample_work(
            source_id="https://openalex.org/W_ERR_1",
            doi="https://doi.org/10.1000/err1",
            pdf_url=None,
        )
        with pytest.raises(RuntimeError):
            service.ingest_work(work_err)
    mock_evolution_svc.evaluate_paper.assert_not_called()


# ── TEST 5: Evolution failure leaves persisted paper intact ───────────────────

def test_evolution_failure_leaves_persisted_paper_intact(isolated_db: Session):
    session = isolated_db
    persistence = SQLAlchemyPersistenceLayer(session)

    # Evolution service that simulates LLM provider outage or unhandled error
    mock_evolution_svc = MagicMock(spec=IncrementalOpportunityEvolutionService)
    mock_evolution_svc.evaluate_paper.side_effect = ProviderFailureError("LLM API gateway 503 Service Unavailable")

    service = OpenAlexPaperIngestionService(
        persistence=persistence,
        auto_evolve=True,
        evolution_service=mock_evolution_svc,
    )

    pdf_bytes = create_sample_pdf_bytes()
    work = make_sample_work(
        source_id="https://openalex.org/W_FAIL_EVO_1",
        doi="https://doi.org/10.1000/failevo1",
    )

    with patch.object(service.downloader, "_open_request", return_value=DummyMockResponse(pdf_bytes)):
        result = service.ingest_work(work)

    # Ingestion MUST NOT fail!
    assert result.status == IngestionStatus.INGESTED
    assert result.has_pdf is True
    assert result.paper_id is not None
    assert result.evolution_status == EvolutionHookStatus.FAILED
    assert "ProviderFailureError" in str(result.evolution_result)

    # Verify that the paper, document, and passages STILL EXIST in the database!
    db_paper = session.execute(select(Paper).where(Paper.id == result.paper_id)).scalar_one_or_none()
    assert db_paper is not None
    assert "W_FAIL_EVO_1" in db_paper.source_id or db_paper.source_id == "W1234567890"

    db_doc = session.execute(select(PaperDocument).where(PaperDocument.paper_id == result.paper_id)).scalar_one_or_none()
    assert db_doc is not None

    db_passages = session.execute(select(PaperPassage).where(PaperPassage.paper_id == result.paper_id)).scalars().all()
    assert len(db_passages) > 0


# ── TEST 6: Partial failure across opportunities is reported accurately ───────

def test_partial_failure_reported_accurately(isolated_db: Session):
    session = isolated_db
    persistence = SQLAlchemyPersistenceLayer(session)

    mock_evolution_svc = MagicMock(spec=IncrementalOpportunityEvolutionService)
    mock_evolution_svc.evaluate_paper.return_value = IncrementalEvolutionResult(
        paper_id=uuid.uuid4(),
        run_status=IncrementalEvolutionRunStatus.PARTIAL_SUCCESS,
        candidate_count=2,
        evaluated_count=1,
        failed_count=1,
        outcomes=[
            OpportunityEvaluationOutcome(
                opportunity_id=uuid.uuid4(),
                status=OpportunityEvaluationStatus.EVALUATED,
                state_changed=True,
            ),
            OpportunityEvaluationOutcome(
                opportunity_id=uuid.uuid4(),
                status=OpportunityEvaluationStatus.FAILED,
                error_message="LLM timeout on candidate 2",
            ),
        ],
    )

    service = OpenAlexPaperIngestionService(
        persistence=persistence,
        auto_evolve=True,
        evolution_service=mock_evolution_svc,
    )

    pdf_bytes = create_sample_pdf_bytes()
    work = make_sample_work(
        source_id="https://openalex.org/W_PARTIAL_1",
        doi="https://doi.org/10.1000/partial1",
    )

    with patch.object(service.downloader, "_open_request", return_value=DummyMockResponse(pdf_bytes)):
        result = service.ingest_work(work)

    assert result.status == IngestionStatus.INGESTED
    assert result.evolution_status == EvolutionHookStatus.PARTIAL_SUCCESS
    assert result.evolution_result.evaluated_count == 1
    assert result.evolution_result.failed_count == 1


# ── TEST 7: Manual PDF ingestion service remains unchanged ────────────────────

def test_manual_pdf_ingestion_service_remains_unchanged(isolated_db: Session, tmp_path: Path):
    session = isolated_db
    persistence = SQLAlchemyPersistenceLayer(session)

    manual_service = ManualPDFIngestionService(persistence=persistence)
    # ManualPDFIngestionService has no auto_evolve attribute or evolution hook
    assert not hasattr(manual_service, "auto_evolve")
    assert not hasattr(manual_service, "evolution_service")

    pdf_bytes = create_sample_pdf_bytes()
    pdf_path = tmp_path / "manual_test.pdf"
    pdf_path.write_bytes(pdf_bytes)

    paper_dto, doc_dto = manual_service.ingest_pdf(pdf_path, title="Manual Title", batch_id="batch_manual_1")
    assert paper_dto.title == "Manual Title"
    assert doc_dto.full_text is not None


# ── TEST 8: PAPER_NOT_FOUND is mapped to FAILED with diagnostics ──────────────

def test_paper_not_found_mapped_to_failed(isolated_db: Session):
    session = isolated_db
    persistence = SQLAlchemyPersistenceLayer(session)

    mock_evolution_svc = MagicMock(spec=IncrementalOpportunityEvolutionService)
    mock_evolution_svc.evaluate_paper.return_value = IncrementalEvolutionResult(
        paper_id=uuid.uuid4(),
        run_status=IncrementalEvolutionRunStatus.PAPER_NOT_FOUND,
        error="Paper was not found in the persistence store.",
    )

    service = OpenAlexPaperIngestionService(
        persistence=persistence,
        auto_evolve=True,
        evolution_service=mock_evolution_svc,
    )

    pdf_bytes = create_sample_pdf_bytes()
    work = make_sample_work(
        source_id="https://openalex.org/W_PNF_1",
        doi="https://doi.org/10.1000/pnf1",
    )

    with patch.object(service.downloader, "_open_request", return_value=DummyMockResponse(pdf_bytes)):
        result = service.ingest_work(work)

    # Ingestion status MUST remain INGESTED
    assert result.status == IngestionStatus.INGESTED
    assert result.paper_id is not None
    # Hook outcome must be FAILED with error diagnostic preserved
    assert result.evolution_status == EvolutionHookStatus.FAILED
    assert result.evolution_result is not None
    assert result.evolution_result.run_status == IncrementalEvolutionRunStatus.PAPER_NOT_FOUND
    assert "not found" in result.evolution_result.error


# ── TEST 9: SUCCESS with evaluated_count == 0 is mapped to SKIPPED ─────────────

def test_success_with_zero_evaluated_mapped_to_skipped(isolated_db: Session):
    session = isolated_db
    persistence = SQLAlchemyPersistenceLayer(session)

    mock_evolution_svc = MagicMock(spec=IncrementalOpportunityEvolutionService)
    # 2 candidates existed, but both were skipped (e.g. pub_date < opp created_at)
    mock_evolution_svc.evaluate_paper.return_value = IncrementalEvolutionResult(
        paper_id=uuid.uuid4(),
        run_status=IncrementalEvolutionRunStatus.SUCCESS,
        candidate_count=2,
        evaluated_count=0,
        skipped_count=2,
        failed_count=0,
    )

    service = OpenAlexPaperIngestionService(
        persistence=persistence,
        auto_evolve=True,
        evolution_service=mock_evolution_svc,
    )

    pdf_bytes = create_sample_pdf_bytes()
    work = make_sample_work(
        source_id="https://openalex.org/W_SKIPPED_CANDS_1",
        doi="https://doi.org/10.1000/skipcand1",
    )

    with patch.object(service.downloader, "_open_request", return_value=DummyMockResponse(pdf_bytes)):
        result = service.ingest_work(work)

    # Ingestion status MUST remain INGESTED
    assert result.status == IngestionStatus.INGESTED
    assert result.paper_id is not None
    # Hook outcome must be SKIPPED because no opportunities were actually evaluated
    assert result.evolution_status == EvolutionHookStatus.SKIPPED
    assert result.evolution_result is not None
    assert result.evolution_result.evaluated_count == 0
    assert result.evolution_result.skipped_count == 2


# ── TEST 10: Ingestion triggers configured candidate ranking under candidate cap ──

@pytest.mark.parametrize("ranking_strategy", ["tfidf", "fifo"])
def test_ingestion_triggers_candidate_ranking_strategy_under_cap(
    isolated_db: Session,
    monkeypatch: pytest.MonkeyPatch,
    ranking_strategy: str,
):
    """Verifies that OpenAlex ingestion triggers the real evolution service with the
    configured candidate ranking strategy (TF-IDF vs FIFO), properly consuming the
    environment setting, evaluating the appropriate opportunity under cap=1, and
    persisting evidence and lifecycle history.
    """
    session = isolated_db
    persistence = SQLAlchemyPersistenceLayer(session)

    # 1. Configure settings: enable post-ingestion evolution, candidate cap = 1, strategy from parameter
    monkeypatch.setenv("IDEAGEN_ENABLE_POST_INGESTION_EVOLUTION", "true")
    monkeypatch.setenv("IDEAGEN_CANDIDATE_RANKING_STRATEGY", ranking_strategy)
    monkeypatch.setenv("IDEAGEN_POST_INGESTION_EVOLUTION_MAX_CANDIDATES", "1")

    # 2. Seed 3 older unrelated distractor opportunities (created in 2020-2021)
    distractor_1 = Opportunity(
        id=uuid.uuid4(),
        title="Deep Vision Transformers for Medical Image Segmentation",
        problem_statement="Vision transformer attention heads miss fine-grained boundary features in MRI scans.",
        description="Study convolutional hybrid backbones for biomedical MRI slice segmentation.",
        status=OpportunityStatus.OPEN.value,
        confidence=0.85,
        created_at=datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc),
    )
    distractor_2 = Opportunity(
        id=uuid.uuid4(),
        title="High-Throughput Genomic Sequencing Pipelines",
        problem_statement="Sequence alignment bottlenecks on long-read sequencing datasets.",
        description="Design GPU-accelerated Smith-Waterman aligners for genomic variant calling.",
        status=OpportunityStatus.OPEN.value,
        confidence=0.85,
        created_at=datetime.datetime(2020, 6, 1, tzinfo=datetime.timezone.utc),
    )
    distractor_3 = Opportunity(
        id=uuid.uuid4(),
        title="Autonomous Mobile Robotics SLAM Trajectory Optimization",
        problem_statement="Lidar odometry drifts significantly in feature-sparse underground environments.",
        description="Investigate factor graph smoothing for autonomous underground mobile robotics.",
        status=OpportunityStatus.OPEN.value,
        confidence=0.85,
        created_at=datetime.datetime(2021, 1, 1, tzinfo=datetime.timezone.utc),
    )

    # 3. Seed 1 newer relevant opportunity (created in 2022) matching consensus partition topic
    opp_relevant = Opportunity(
        id=uuid.uuid4(),
        title="Byzantine Consensus Under Asymmetric Network Partitions",
        problem_statement="Distributed consensus protocols deadlock during asymmetric network partition and high latency.",
        description="Design partition-tolerant Byzantine state machine replication protocols with guaranteed liveness.",
        status=OpportunityStatus.OPEN.value,
        confidence=0.90,
        created_at=datetime.datetime(2022, 1, 1, tzinfo=datetime.timezone.utc),
    )

    session.add_all([distractor_1, distractor_2, distractor_3, opp_relevant])
    session.commit()

    # Determine expected ranker class and targets based on ranking strategy:
    # TF-IDF evaluates the lexically relevant opportunity and skips older distractors.
    # FIFO evaluates the oldest opportunity (distractor_1) and skips the newer relevant opportunity.
    if ranking_strategy == "tfidf":
        expected_ranker_cls = TFIDFCandidateRanker
        expected_evaluated = opp_relevant
        expected_skipped = [distractor_1, distractor_2, distractor_3]
    elif ranking_strategy == "fifo":
        expected_ranker_cls = FIFOCandidateRanker
        expected_evaluated = distractor_1
        expected_skipped = [distractor_2, distractor_3, opp_relevant]
    else:
        pytest.fail(f"Unexpected ranking strategy: {ranking_strategy}")

    # 4. Prepare deterministic mock LLM provider with grounded classification and call tracking
    passage_excerpt = "Byzantine consensus bounds under asymmetric network partition delay."
    mock_llm = MockLLMProvider(
        classification=RelationshipClassification(
            relationship=OpportunityRelationship.ADDRESSES,
            confidence=0.95,
            rationale="Addresses evaluated opportunity under candidate ranking.",
            evidence_excerpt=passage_excerpt,
        )
    )

    # 5. Prepare paper and PDF content (published in 2023, after all opportunities)
    paper_title = "Byzantine Consensus Protocols Under Asymmetric Network Delay"
    abstract_text = "We analyze partition tolerance and state machine replication in distributed systems under Byzantine network faults."
    abstract_words = abstract_text.split()
    abstract_index = {word: [i] for i, word in enumerate(abstract_words)}

    work = {
        "id": "https://openalex.org/W99990001",
        "doi": "https://doi.org/10.1000/w99990001",
        "title": paper_title,
        "publication_date": "2023-01-01",
        "abstract_inverted_index": abstract_index,
        "authorships": [{"author": {"display_name": "Distributed Systems Researcher"}}],
        "primary_location": {
            "source": {"type": "journal"},
            "pdf_url": "https://example.com/consensus.pdf",
            "landing_page_url": "https://example.com/consensus",
        },
        "best_oa_location": {
            "pdf_url": "https://example.com/consensus.pdf",
            "landing_page_url": "https://example.com/consensus",
        },
    }

    pdf_bytes = create_sample_pdf_bytes(title=paper_title, body=passage_excerpt)

    # 6. Instantiate OpenAlexPaperIngestionService with auto_evolve=None (reads settings)
    # and evolution_service=None (triggers _get_or_create_evolution_service)
    service = OpenAlexPaperIngestionService(
        persistence=persistence,
        auto_evolve=None,
        evolution_service=None,
    )

    # 7. Execute ingestion and evolution path (patching LLM provider lookup and PDF download stream)
    with patch("ideagen.llm.get_default_llm_provider", return_value=mock_llm), \
         patch.object(service.downloader, "_open_request", return_value=DummyMockResponse(pdf_bytes)):
        result = service.ingest_work(work)

    # 8. Verify ingestion outcome and evolution hook status
    assert result.status == IngestionStatus.INGESTED
    assert result.has_pdf is True
    assert result.evolution_status == EvolutionHookStatus.SUCCESS
    assert result.evolution_result is not None
    assert result.evolution_result.run_status == IncrementalEvolutionRunStatus.SUCCESS

    # Verify candidate count accounting: 4 candidates found, 1 evaluated under cap=1, 3 skipped
    assert result.evolution_result.candidate_count == 4
    assert result.evolution_result.evaluated_count == 1
    assert result.evolution_result.skipped_count == 3
    assert result.evolution_result.grounding_rejected_count == 0
    assert result.evolution_result.failed_count == 0

    # Verify exactly one classification call occurred under candidate cap 1
    assert mock_llm.call_count == 1

    # 9. Verify that the real evolution service used the configured candidate ranker and cap
    assert service.evolution_service is not None
    assert isinstance(service.evolution_service.candidate_ranker, expected_ranker_cls)
    assert service.evolution_service.max_candidates == 1

    # 10. Verify candidate evaluation outcomes:
    # Expected candidate was evaluated; other 3 were skipped by candidate limit
    outcomes = {o.opportunity_id: o for o in result.evolution_result.outcomes}
    assert outcomes[expected_evaluated.id].status == OpportunityEvaluationStatus.EVALUATED
    for skipped_opp in expected_skipped:
        assert outcomes[skipped_opp.id].status == OpportunityEvaluationStatus.SKIPPED
        assert "Candidate limit (1) reached" in outcomes[skipped_opp.id].skip_reason

    # 11. Verify lifecycle and persistence:
    # Evaluated opportunity transitioned to PARTIALLY_ADDRESSED
    session.refresh(expected_evaluated)
    assert expected_evaluated.status == OpportunityStatus.PARTIALLY_ADDRESSED.value

    # Skipped opportunities remained in OPEN status
    for skipped_opp in expected_skipped:
        session.refresh(skipped_opp)
        assert skipped_opp.status == OpportunityStatus.OPEN.value

    # Exactly one Evidence record exists for the evaluated opportunity
    evidence_rows = session.execute(
        select(OpportunityEvidence).where(OpportunityEvidence.opportunity_id == expected_evaluated.id)
    ).scalars().all()
    assert len(evidence_rows) == 1
    evidence = evidence_rows[0]
    assert evidence.paper_id == result.paper_id
    assert evidence.evidence_type == OpportunityRelationship.ADDRESSES.value
    assert evidence.supporting_text == passage_excerpt

    # No evidence records exist for skipped opportunities
    for skipped_opp in expected_skipped:
        skipped_evidence = session.execute(
            select(OpportunityEvidence).where(OpportunityEvidence.opportunity_id == skipped_opp.id)
        ).scalars().all()
        assert len(skipped_evidence) == 0

    # Exactly one History record was persisted for the evaluated opportunity, with matching transition fields
    history_rows = session.execute(
        select(OpportunityHistory).where(OpportunityHistory.opportunity_id == expected_evaluated.id)
    ).scalars().all()
    assert len(history_rows) == 1
    history_record = history_rows[0]
    assert history_record.previous_status == OpportunityStatus.OPEN.value
    assert history_record.new_status == OpportunityStatus.PARTIALLY_ADDRESSED.value
    assert history_record.trigger_type == OpportunityRelationship.ADDRESSES.value
    assert history_record.evidence_id == evidence.id

    # No history records exist for skipped opportunities
    for skipped_opp in expected_skipped:
        skipped_history = session.execute(
            select(OpportunityHistory).where(OpportunityHistory.opportunity_id == skipped_opp.id)
        ).scalars().all()
        assert len(skipped_history) == 0
