"""Focused test suite for TF-IDF and FIFO candidate ranking in IncrementalOpportunityEvolutionService.

Verifies:
1. A highly relevant opportunity ranks ahead of older unrelated opportunities under a small cap.
2. FIFO mode preserves the existing chronological behavior.
3. Equal TF-IDF scores use deterministic chronological/ID tie-breaking.
4. Empty query text, empty opportunity text, and all-zero similarity safely fall back to deterministic FIFO ordering.
5. Temporal eligibility remains enforced: an opportunity created after the incoming paper's publication date is never evaluated.
6. Lifecycle-status filtering and existing idempotency behavior remain intact.
7. Candidate-cap exclusions retain explicit, accurate SKIPPED logging.
8. An UngroundedEvidenceError still rolls back only the affected nested savepoint and does not corrupt other candidate evaluations.
9. Existing tests that assert chronological ordering remain valid in FIFO mode.
"""

from __future__ import annotations

import datetime
from typing import Optional, List
import uuid
import pytest
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
from ideagen.opportunity.models import OpportunityRelationship, OpportunityStatus
from ideagen.opportunity.service import OpportunityTransitionService
from ideagen.opportunity.classifier import (
    RelationshipClassifier,
    RelationshipClassification,
)
from ideagen.opportunity.integration import OpportunityClassificationIntegration
from ideagen.opportunity.grounding import UngroundedEvidenceError
from ideagen.opportunity.incremental import (
    IncrementalOpportunityEvolutionService,
    OpportunityEvaluationStatus,
    IncrementalEvolutionRunStatus,
)
from ideagen.opportunity.ranking import (
    CandidateRanker,
    TFIDFCandidateRanker,
    FIFOCandidateRanker,
)
from ideagen.llm.provider import LLMProvider


class MockLLMProvider(LLMProvider):
    """Deterministic offline mock LLM provider for relationship classification."""

    def __init__(self, responses: Optional[dict[str, RelationshipClassification]] = None, default: Optional[RelationshipClassification] = None):
        self.responses = responses or {}
        self.default = default
        self.call_history: List[str] = []

    def generate_structured(self, prompt: str, response_schema):
        self.call_history.append(prompt)
        for key, resp in self.responses.items():
            if key in prompt:
                if isinstance(resp, Exception):
                    raise resp
                return resp
        if self.default is not None:
            if isinstance(self.default, Exception):
                raise self.default
            return self.default
        return RelationshipClassification(
            relationship=OpportunityRelationship.UNRELATED,
            confidence=0.9,
            rationale="Default mock unrelated",
            evidence_excerpt=None,
        )


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


def _create_test_paper(
    session: Session,
    title: str = "Test Paper on Optimizing Distributed Consensus",
    abstract: str = "We investigate consensus latency and network partition bounds.",
    doi: str = "10.1000/test.123",
    pub_date: Optional[datetime.date] = datetime.date(2023, 1, 1),
    passage_text: str = "Consensus algorithms provide state machine replication guarantees.",
) -> Paper:
    paper = Paper(
        id=uuid.uuid4(),
        source_type="TEST",
        source_id=f"src_{uuid.uuid4().hex[:8]}",
        doi=doi,
        title=title,
        publication_date=pub_date,
        batch_id="batch-1",
        abstract=abstract,
        authors=["Alice", "Bob"],
        referenced_works=[],
        raw_metadata={},
    )
    session.add(paper)
    session.flush()

    if passage_text is not None:
        doc_id = uuid.uuid4()
        doc = PaperDocument(
            id=doc_id,
            paper_id=paper.id,
            full_text=passage_text,
            extraction_method="pdf_test",
            extraction_version="v1.0",
            extracted_at=datetime.datetime.now(datetime.timezone.utc),
        )
        session.add(doc)
        session.flush()

        passage = PaperPassage(
            id=uuid.uuid4(),
            paper_id=paper.id,
            document_id=doc_id,
            passage_index=0,
            page_start=1,
            page_end=1,
            section="Discussion",
            text=passage_text,
        )
        session.add(passage)
        session.flush()

    return paper


def _create_test_opportunity(
    session: Session,
    title: str = "Scalable Consensus Under Network Partitions",
    problem_statement: str = "Consensus protocols suffer severe throughput degradation under asymmetric network delay.",
    description: str = "Investigate partition-tolerant state machine replication algorithms.",
    status: OpportunityStatus = OpportunityStatus.OPEN,
    created_at: Optional[datetime.datetime] = None,
) -> Opportunity:
    opp = Opportunity(
        id=uuid.uuid4(),
        title=title,
        problem_statement=problem_statement,
        description=description,
        status=status.value,
        confidence=0.85,
        created_at=created_at or datetime.datetime(2022, 1, 1, 0, 0, tzinfo=datetime.timezone.utc),
    )
    session.add(opp)
    session.flush()
    return opp


def _build_service(
    session: Session,
    mock_llm: MockLLMProvider,
    max_candidates: Optional[int] = None,
    candidate_ranker: Optional[CandidateRanker] = None,
) -> IncrementalOpportunityEvolutionService:
    classifier = RelationshipClassifier(provider=mock_llm)
    trans_service = OpportunityTransitionService(session)
    integration = OpportunityClassificationIntegration(classifier, trans_service, session)
    return IncrementalOpportunityEvolutionService(
        integration=integration,
        session=session,
        max_candidates=max_candidates,
        candidate_ranker=candidate_ranker,
    )


# ── TEST 1: Relevant opportunity ranks ahead of older unrelated opportunities under cap ────

def test_relevant_opportunity_ranks_ahead_of_older_unrelated_under_cap(isolated_db: Session):
    session = isolated_db
    passage_text = "Consensus algorithm bounds under network partitions."
    paper = _create_test_paper(
        session,
        title="Byzantine Consensus Under Severe Network Partitions",
        abstract="We analyze partition tolerance and state machine replication in distributed systems.",
        pub_date=datetime.date(2023, 6, 1),
        passage_text=passage_text,
    )

    # 2 older unrelated distractor opportunities
    opp_distractor_1 = _create_test_opportunity(
        session,
        title="Deep Vision Transformers for Medical Image Segmentation",
        problem_statement="Vision transformer attention heads miss fine-grained boundary features.",
        description="Study convolutional hybrid backbones for MRI scans.",
        created_at=datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc),
    )
    opp_distractor_2 = _create_test_opportunity(
        session,
        title="High-Throughput Genomic Sequencing Pipelines",
        problem_statement="Sequence alignment bottlenecks on long-read datasets.",
        description="Design GPU-accelerated Smith-Waterman aligners.",
        created_at=datetime.datetime(2021, 1, 1, tzinfo=datetime.timezone.utc),
    )
    # 1 newer highly relevant opportunity matching the consensus partition problem
    opp_relevant = _create_test_opportunity(
        session,
        title="Byzantine Consensus Under Severe Network Partitions",
        problem_statement="Distributed consensus protocols deadlock during asymmetric network partition.",
        description="Design partition-tolerant state machine replication protocols.",
        created_at=datetime.datetime(2022, 1, 1, tzinfo=datetime.timezone.utc),
    )

    mock_llm = MockLLMProvider(
        default=RelationshipClassification(
            relationship=OpportunityRelationship.ADDRESSES,
            confidence=0.95,
            rationale="Addresses the partition consensus problem.",
            evidence_excerpt=passage_text,
        )
    )

    # With cap = 1 and default TF-IDF ranking:
    # opp_relevant should rank #1 and be evaluated, while distractors 1 and 2 are skipped by cap.
    service_tfidf = _build_service(session, mock_llm, max_candidates=1, candidate_ranker=TFIDFCandidateRanker())
    res_tfidf = service_tfidf.evaluate_paper(paper_id=paper.id)

    outcome_map = {o.opportunity_id: o for o in res_tfidf.outcomes}
    assert res_tfidf.evaluated_count == 1
    assert outcome_map[opp_relevant.id].status == OpportunityEvaluationStatus.EVALUATED
    assert outcome_map[opp_distractor_1.id].status == OpportunityEvaluationStatus.SKIPPED
    assert outcome_map[opp_distractor_2.id].status == OpportunityEvaluationStatus.SKIPPED
    assert "Candidate limit (1) reached" in outcome_map[opp_distractor_1.id].skip_reason
    assert "Candidate limit (1) reached" in outcome_map[opp_distractor_2.id].skip_reason


# ── TEST 2: FIFO mode preserves existing chronological behavior ───────────────

def test_fifo_mode_preserves_existing_chronological_behavior(isolated_db: Session):
    session = isolated_db
    passage_text = "Consensus algorithm bounds under network partitions."
    paper = _create_test_paper(
        session,
        title="Byzantine Consensus Under Severe Network Partitions",
        abstract="We analyze partition tolerance and state machine replication in distributed systems.",
        pub_date=datetime.date(2023, 6, 1),
        passage_text=passage_text,
    )

    opp_old_unrelated = _create_test_opportunity(
        session,
        title="Deep Vision Transformers",
        problem_statement="Attention heads miss boundary features.",
        description="Study hybrid backbones.",
        created_at=datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc),
    )
    opp_new_relevant = _create_test_opportunity(
        session,
        title="Byzantine Consensus Under Severe Network Partitions",
        problem_statement="Distributed consensus protocols deadlock during partition.",
        description="Partition-tolerant state machine replication protocols.",
        created_at=datetime.datetime(2022, 1, 1, tzinfo=datetime.timezone.utc),
    )

    mock_llm = MockLLMProvider(
        default=RelationshipClassification(
            relationship=OpportunityRelationship.UNRELATED,
            confidence=0.9,
            rationale="Unrelated topic.",
            evidence_excerpt=None,
        )
    )

    # In FIFO mode with cap = 1: the oldest opportunity (opp_old_unrelated) is evaluated
    # even though it is unrelated, and opp_new_relevant is skipped due to candidate cap.
    service_fifo = _build_service(session, mock_llm, max_candidates=1, candidate_ranker=FIFOCandidateRanker())
    res_fifo = service_fifo.evaluate_paper(paper_id=paper.id)

    outcome_map = {o.opportunity_id: o for o in res_fifo.outcomes}
    assert res_fifo.evaluated_count == 1
    assert outcome_map[opp_old_unrelated.id].status == OpportunityEvaluationStatus.EVALUATED
    assert outcome_map[opp_new_relevant.id].status == OpportunityEvaluationStatus.SKIPPED
    assert "Candidate limit (1) reached" in outcome_map[opp_new_relevant.id].skip_reason


# ── TEST 3: Equal TF-IDF scores use deterministic tie-breaking ─────────────────

def test_equal_tfidf_scores_use_deterministic_tie_breaking(isolated_db: Session):
    session = isolated_db
    paper = _create_test_paper(session, title="Generic Paper Title", abstract="Generic paper abstract.")

    # 3 opportunities with identical text (so TF-IDF scores are identical)
    opp_1 = _create_test_opportunity(session, title="Common Title", created_at=datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc))
    opp_2 = _create_test_opportunity(session, title="Common Title", created_at=datetime.datetime(2021, 1, 1, tzinfo=datetime.timezone.utc))
    opp_3 = _create_test_opportunity(session, title="Common Title", created_at=datetime.datetime(2022, 1, 1, tzinfo=datetime.timezone.utc))

    mock_llm = MockLLMProvider()
    service = _build_service(session, mock_llm, max_candidates=2, candidate_ranker=TFIDFCandidateRanker())
    res = service.evaluate_paper(paper_id=paper.id)

    outcome_map = {o.opportunity_id: o for o in res.outcomes}
    assert res.evaluated_count == 2
    # The two older opportunities must be evaluated first by deterministic tie-breaking
    assert outcome_map[opp_1.id].status == OpportunityEvaluationStatus.EVALUATED
    assert outcome_map[opp_2.id].status == OpportunityEvaluationStatus.EVALUATED
    assert outcome_map[opp_3.id].status == OpportunityEvaluationStatus.SKIPPED
    assert "Candidate limit (2) reached" in outcome_map[opp_3.id].skip_reason


# ── TEST 4: Empty query, empty opportunity text, and zero similarity fallback ──

def test_empty_query_or_empty_opportunity_text_falls_back_to_fifo(isolated_db: Session):
    session = isolated_db
    # Stop-word-only or minimal text paper
    paper = _create_test_paper(session, title="the and of", abstract="in at on", passage_text="to for with")

    opp_1 = _create_test_opportunity(session, title="the and of", problem_statement="", description="", created_at=datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc))
    opp_2 = _create_test_opportunity(session, title="the and of", problem_statement="", description="", created_at=datetime.datetime(2021, 1, 1, tzinfo=datetime.timezone.utc))

    mock_llm = MockLLMProvider()
    service = _build_service(session, mock_llm, max_candidates=1, candidate_ranker=TFIDFCandidateRanker())
    # Should not crash on empty/stop-word vocabulary; falls back to chronological ordering
    res = service.evaluate_paper(paper_id=paper.id)

    outcome_map = {o.opportunity_id: o for o in res.outcomes}
    assert res.evaluated_count == 1
    assert outcome_map[opp_1.id].status == OpportunityEvaluationStatus.EVALUATED
    assert outcome_map[opp_2.id].status == OpportunityEvaluationStatus.SKIPPED


# ── TEST 5: Temporal eligibility: future opportunity never evaluated ───────────

def test_temporal_eligibility_enforced_future_opportunity_never_evaluated(isolated_db: Session):
    session = isolated_db
    passage_text = "Consensus algorithms provide partition tolerance guarantees."
    paper = _create_test_paper(
        session,
        title="Consensus Partitioning",
        pub_date=datetime.date(2023, 1, 1),
        passage_text=passage_text,
    )

    # Future opportunity (created 2024, after paper 2023) with EXACT keyword match
    opp_future = _create_test_opportunity(
        session,
        title="Consensus Partitioning",
        problem_statement="Consensus Partitioning",
        description="Consensus Partitioning",
        created_at=datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc),
    )
    # Past opportunity (created 2022) with lower match
    opp_past = _create_test_opportunity(
        session,
        title="General Distributed Computing",
        problem_statement="Network protocols",
        description="General description",
        created_at=datetime.datetime(2022, 1, 1, tzinfo=datetime.timezone.utc),
    )

    mock_llm = MockLLMProvider()
    service = _build_service(session, mock_llm, max_candidates=1, candidate_ranker=TFIDFCandidateRanker())
    res = service.evaluate_paper(paper_id=paper.id)

    outcome_map = {o.opportunity_id: o for o in res.outcomes}
    # Future opportunity MUST be skipped for temporal causality despite higher TF-IDF score
    assert outcome_map[opp_future.id].status == OpportunityEvaluationStatus.SKIPPED
    assert "after paper publication date" in outcome_map[opp_future.id].skip_reason
    # Past opportunity is evaluated
    assert outcome_map[opp_past.id].status == OpportunityEvaluationStatus.EVALUATED
    assert res.evaluated_count == 1


# ── TEST 6: Lifecycle-status and idempotency preserved ─────────────────────────

def test_lifecycle_status_and_idempotency_preserved(isolated_db: Session):
    session = isolated_db
    paper = _create_test_paper(session, title="Consensus Paper")

    # Already evaluated opportunity with high match
    opp_already_eval = _create_test_opportunity(
        session,
        title="Consensus Paper",
        created_at=datetime.datetime(2021, 1, 1, tzinfo=datetime.timezone.utc),
    )
    session.add(OpportunityEvidence(
        opportunity_id=opp_already_eval.id,
        paper_id=paper.id,
        evidence_type=OpportunityRelationship.UNRELATED.value,
        supporting_text=None,
    ))
    session.flush()

    opp_new = _create_test_opportunity(
        session,
        title="Consensus Paper New",
        created_at=datetime.datetime(2022, 1, 1, tzinfo=datetime.timezone.utc),
    )

    mock_llm = MockLLMProvider()
    service = _build_service(session, mock_llm, max_candidates=1, candidate_ranker=TFIDFCandidateRanker())
    res = service.evaluate_paper(paper_id=paper.id)

    outcome_map = {o.opportunity_id: o for o in res.outcomes}
    # Already evaluated opportunity is SKIPPED with "already recorded" and does not consume the cap
    assert outcome_map[opp_already_eval.id].status == OpportunityEvaluationStatus.SKIPPED
    assert "already recorded" in outcome_map[opp_already_eval.id].skip_reason
    # opp_new receives the available slot and is evaluated
    assert outcome_map[opp_new.id].status == OpportunityEvaluationStatus.EVALUATED
    assert res.evaluated_count == 1


# ── TEST 7: Candidate-cap exclusions retain explicit SKIPPED logging ────────────

def test_candidate_cap_exclusions_retain_explicit_skipped_logging(isolated_db: Session):
    session = isolated_db
    paper = _create_test_paper(session, title="Consensus Paper")

    opp_1 = _create_test_opportunity(session, title="Consensus Paper High", created_at=datetime.datetime(2021, 1, 1, tzinfo=datetime.timezone.utc))
    opp_2 = _create_test_opportunity(session, title="Consensus Paper Medium", created_at=datetime.datetime(2021, 2, 1, tzinfo=datetime.timezone.utc))
    opp_3 = _create_test_opportunity(session, title="Consensus Paper Low", created_at=datetime.datetime(2021, 3, 1, tzinfo=datetime.timezone.utc))

    mock_llm = MockLLMProvider()
    service = _build_service(session, mock_llm, max_candidates=1, candidate_ranker=TFIDFCandidateRanker())
    res = service.evaluate_paper(paper_id=paper.id)

    outcome_map = {o.opportunity_id: o for o in res.outcomes}
    assert res.evaluated_count == 1
    assert res.skipped_count == 2
    for opp_id in [opp_2.id, opp_3.id]:
        assert outcome_map[opp_id].status == OpportunityEvaluationStatus.SKIPPED
        assert "Candidate limit (1) reached" in outcome_map[opp_id].skip_reason


# ── TEST 8: UngroundedEvidenceError rolls back savepoint cleanly ───────────────

def test_ungrounded_evidence_error_rolls_back_savepoint_cleanly(isolated_db: Session):
    session = isolated_db
    passage_text = "Verified passage text."
    paper = _create_test_paper(session, passage_text=passage_text)

    # Opportunity 1 will trigger an ungrounded hallucination error
    opp_1 = _create_test_opportunity(
        session,
        title="Opportunity 1 Consensus",
        status=OpportunityStatus.OPEN,
        created_at=datetime.datetime(2021, 1, 1, tzinfo=datetime.timezone.utc),
    )
    # Opportunity 2 will succeed
    opp_2 = _create_test_opportunity(
        session,
        title="Opportunity 2 Consensus",
        status=OpportunityStatus.OPEN,
        created_at=datetime.datetime(2021, 2, 1, tzinfo=datetime.timezone.utc),
    )

    mock_llm = MockLLMProvider(
        responses={
            "Opportunity 1 Consensus": RelationshipClassification(
                relationship=OpportunityRelationship.ADDRESSES,
                confidence=0.9,
                rationale="Hallucinated evidence",
                evidence_excerpt="Completely fabricated excerpt that does not exist in the paper text.",
            ),
            "Opportunity 2 Consensus": RelationshipClassification(
                relationship=OpportunityRelationship.SUPPORTS,
                confidence=0.9,
                rationale="Grounded evidence",
                evidence_excerpt=passage_text,
            ),
        }
    )

    service = _build_service(session, mock_llm, max_candidates=2, candidate_ranker=TFIDFCandidateRanker())
    res = service.evaluate_paper(paper_id=paper.id)

    outcome_map = {o.opportunity_id: o for o in res.outcomes}
    # Opp 1 was rejected by grounding gatekeeper
    assert outcome_map[opp_1.id].status == OpportunityEvaluationStatus.GROUNDING_REJECTED
    # Opp 2 evaluated successfully
    assert outcome_map[opp_2.id].status == OpportunityEvaluationStatus.EVALUATED
    assert outcome_map[opp_2.id].final_status == OpportunityStatus.STRENGTHENED

    session.refresh(opp_1)
    session.refresh(opp_2)
    assert opp_1.status == OpportunityStatus.OPEN.value
    assert opp_2.status == OpportunityStatus.STRENGTHENED.value


# ── TEST 9: Existing chronological ordering remains valid in FIFO mode ─────────

def test_existing_chronological_ordering_tests_pass_in_fifo_mode(isolated_db: Session):
    session = isolated_db
    passage_text = "Consensus algorithm bounds."
    paper = _create_test_paper(session, pub_date=datetime.date(2023, 1, 1), passage_text=passage_text)

    # Create opportunities out of chronological order
    opp_c = _create_test_opportunity(session, title="Opportunity C", created_at=datetime.datetime(2022, 6, 1, tzinfo=datetime.timezone.utc))
    opp_a = _create_test_opportunity(session, title="Opportunity A", created_at=datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc))
    opp_b = _create_test_opportunity(session, title="Opportunity B", created_at=datetime.datetime(2021, 1, 1, tzinfo=datetime.timezone.utc))

    mock_llm = MockLLMProvider(
        default=RelationshipClassification(
            relationship=OpportunityRelationship.SUPPORTS,
            confidence=0.8,
            rationale="Deterministic test",
            evidence_excerpt=passage_text,
        )
    )

    service = _build_service(session, mock_llm, candidate_ranker=FIFOCandidateRanker())
    result = service.evaluate_paper(paper_id=paper.id)

    assert result.evaluated_count == 3
    evaluated_ids = [o.opportunity_id for o in result.outcomes]
    assert evaluated_ids == [opp_a.id, opp_b.id, opp_c.id]
