"""Focused offline test suite for IncrementalOpportunityEvolutionService (Stage 3B, Part 1).

Verifies:
1. Eligible opportunity evaluated successfully.
2. Opportunity created after the paper's publication date skipped.
3. Missing or undated paper handled explicitly.
4. Duplicate processing does not create duplicate evidence/history.
5. Ungrounded classification fails closed.
6. Failure on one opportunity does not corrupt other evaluations.
7. Deterministic candidate ordering.
8. Atomic evidence/status/history behavior.
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
    ResearchOpportunityRecord,
)
from ideagen.opportunity.models import OpportunityRelationship, OpportunityStatus
from ideagen.opportunity.service import OpportunityTransitionService
from ideagen.opportunity.classifier import (
    RelationshipClassifier,
    RelationshipClassification,
    RelationshipClassificationInput,
)
from ideagen.opportunity.integration import OpportunityClassificationIntegration
from ideagen.opportunity.incremental import (
    IncrementalOpportunityEvolutionService,
    IncrementalEvolutionResult,
    IncrementalEvolutionRunStatus,
    OpportunityEvaluationStatus,
    get_opportunity_effective_creation_date,
)
from ideagen.llm.provider import LLMProvider
from ideagen.llm.exceptions import ProviderFailureError


class MockLLMProvider(LLMProvider):
    """Deterministic offline mock LLM provider for opportunity relationship classification."""

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
    doi: str = "10.1000/test.123",
    abstract: str = "This paper presents a novel consensus algorithm that achieves lower latency under high contention.",
    passage_text: Optional[str] = "We prove that the state machine replication protocol guarantees safety under crash faults.",
    pub_date: Optional[datetime.date] = datetime.date(2023, 1, 1),
) -> Paper:
    paper_id = uuid.uuid4()
    p = Paper(
        id=paper_id,
        source_type="TEST",
        source_id=f"src_{paper_id.hex[:8]}",
        doi=doi,
        title=title,
        publication_date=pub_date,
        batch_id="batch-1",
        abstract=abstract,
        authors=["Author Alice"],
        referenced_works=[],
        raw_metadata={},
    )
    session.add(p)
    session.flush()

    if passage_text is not None:
        doc_id = uuid.uuid4()
        doc = PaperDocument(
            id=doc_id,
            paper_id=paper_id,
            full_text=passage_text,
            extraction_method="pdf_test",
            extraction_version="v1.0",
            extracted_at=datetime.datetime.now(datetime.timezone.utc),
        )
        session.add(doc)
        session.flush()

        passage = PaperPassage(
            id=uuid.uuid4(),
            paper_id=paper_id,
            document_id=doc_id,
            passage_index=0,
            page_start=1,
            page_end=1,
            section="Methods",
            text=passage_text,
        )
        session.add(passage)
        session.flush()

    return p


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


# ── TEST 1: Eligible opportunity evaluated successfully ───────────────────────

def test_eligible_opportunity_evaluated_successfully(isolated_db: Session):
    session = isolated_db

    passage_text = "We prove that the state machine replication protocol guarantees safety under crash faults."
    paper = _create_test_paper(
        session,
        pub_date=datetime.date(2023, 1, 1),
        passage_text=passage_text,
    )
    opp = _create_test_opportunity(
        session,
        status=OpportunityStatus.OPEN,
        created_at=datetime.datetime(2022, 1, 1, tzinfo=datetime.timezone.utc),
    )

    mock_llm = MockLLMProvider(
        default=RelationshipClassification(
            relationship=OpportunityRelationship.ADDRESSES,
            confidence=0.95,
            rationale="Paper addresses the consensus problem directly.",
            evidence_excerpt=passage_text,
        )
    )
    classifier = RelationshipClassifier(provider=mock_llm)
    trans_service = OpportunityTransitionService(session)
    integration = OpportunityClassificationIntegration(classifier, trans_service, session)
    service = IncrementalOpportunityEvolutionService(integration, session)

    result = service.evaluate_paper(paper_id=paper.id)

    assert result.run_status == IncrementalEvolutionRunStatus.SUCCESS
    assert result.paper_id == paper.id
    assert result.evaluated_count == 1
    assert result.skipped_count == 0
    assert result.failed_count == 0
    assert len(result.outcomes) == 1

    outcome = result.outcomes[0]
    assert outcome.opportunity_id == opp.id
    assert outcome.status == OpportunityEvaluationStatus.EVALUATED
    assert outcome.initial_status == OpportunityStatus.OPEN
    assert outcome.final_status == OpportunityStatus.PARTIALLY_ADDRESSED
    assert outcome.relationship == OpportunityRelationship.ADDRESSES
    assert outcome.state_changed is True
    assert outcome.evidence_id is not None
    assert outcome.history_id is not None

    # Verify DB state
    session.refresh(opp)
    assert opp.status == OpportunityStatus.PARTIALLY_ADDRESSED.value

    evidence = session.execute(
        select(OpportunityEvidence).where(
            OpportunityEvidence.opportunity_id == opp.id,
            OpportunityEvidence.paper_id == paper.id,
        )
    ).scalar_one_or_none()
    assert evidence is not None
    assert evidence.evidence_type == OpportunityRelationship.ADDRESSES.value

    history = session.execute(
        select(OpportunityHistory).where(OpportunityHistory.opportunity_id == opp.id)
    ).scalars().all()
    assert len(history) == 1
    assert history[0].previous_status == OpportunityStatus.OPEN.value
    assert history[0].new_status == OpportunityStatus.PARTIALLY_ADDRESSED.value


# ── TEST 2: Opportunity created after paper publication date skipped ──────────

def test_opportunity_created_after_paper_pub_date_skipped(isolated_db: Session):
    session = isolated_db

    passage_text = "We introduce verifiable fault detection for distributed consensus."
    paper = _create_test_paper(
        session,
        pub_date=datetime.date(2022, 6, 1),
        passage_text=passage_text,
    )

    # Opp A was created before paper -> eligible
    opp_a = _create_test_opportunity(
        session,
        title="Opportunity A",
        created_at=datetime.datetime(2022, 1, 1, tzinfo=datetime.timezone.utc),
    )
    # Opp B was created after paper -> ineligible (future opportunity)
    opp_b = _create_test_opportunity(
        session,
        title="Opportunity B",
        created_at=datetime.datetime(2023, 1, 1, tzinfo=datetime.timezone.utc),
    )

    mock_llm = MockLLMProvider(
        default=RelationshipClassification(
            relationship=OpportunityRelationship.SUPPORTS,
            confidence=0.88,
            rationale="Strengthens consensus importance.",
            evidence_excerpt=passage_text,
        )
    )
    classifier = RelationshipClassifier(provider=mock_llm)
    trans_service = OpportunityTransitionService(session)
    integration = OpportunityClassificationIntegration(classifier, trans_service, session)
    service = IncrementalOpportunityEvolutionService(integration, session)

    result = service.evaluate_paper(paper_id=paper.id)

    assert result.run_status == IncrementalEvolutionRunStatus.SUCCESS
    assert result.evaluated_count == 1
    assert result.skipped_count == 1

    outcome_map = {o.opportunity_id: o for o in result.outcomes}
    assert outcome_map[opp_a.id].status == OpportunityEvaluationStatus.EVALUATED
    assert outcome_map[opp_a.id].final_status == OpportunityStatus.STRENGTHENED

    assert outcome_map[opp_b.id].status == OpportunityEvaluationStatus.SKIPPED
    assert "after paper publication date" in outcome_map[opp_b.id].skip_reason

    # Opp B must have 0 evidence and 0 history in DB
    ev_b = session.execute(
        select(OpportunityEvidence).where(OpportunityEvidence.opportunity_id == opp_b.id)
    ).scalars().all()
    assert len(ev_b) == 0


# ── TEST 3: Missing, undated, and insufficient content paper handled explicitly ─

def test_missing_or_undated_or_insufficient_content_paper(isolated_db: Session):
    session = isolated_db
    mock_llm = MockLLMProvider()
    classifier = RelationshipClassifier(provider=mock_llm)
    trans_service = OpportunityTransitionService(session)
    integration = OpportunityClassificationIntegration(classifier, trans_service, session)
    service = IncrementalOpportunityEvolutionService(integration, session)

    # 3a. Paper does not exist
    missing_id = uuid.uuid4()
    res_missing = service.evaluate_paper(paper_id=missing_id)
    assert res_missing.run_status == IncrementalEvolutionRunStatus.PAPER_NOT_FOUND
    assert "not found in database" in str(res_missing.error)

    # 3b. Paper has publication_date = None (undated)
    undated_paper = _create_test_paper(session, doi="10.1000/undated.1", pub_date=None)
    res_undated = service.evaluate_paper(paper_id=undated_paper.id)
    assert res_undated.run_status == IncrementalEvolutionRunStatus.SKIPPED_UNDATED
    assert "undated" in str(res_undated.error).lower()

    # 3c. Paper lacks sufficient usable content (empty title, no abstract, no passages)
    empty_paper = Paper(
        id=uuid.uuid4(),
        source_type="TEST",
        source_id="src_empty",
        doi="10.1000/empty.1",
        title="",
        publication_date=datetime.date(2023, 1, 1),
        batch_id="batch-1",
        abstract="",
        authors=[],
        referenced_works=[],
        raw_metadata={},
    )
    session.add(empty_paper)
    session.flush()

    res_empty = service.evaluate_paper(paper_id=empty_paper.id)
    assert res_empty.run_status == IncrementalEvolutionRunStatus.SKIPPED_INSUFFICIENT_CONTENT
    assert "lacks sufficient usable textual content" in str(res_empty.error)


# ── TEST 4: Duplicate processing does not create duplicate evidence/history ───

def test_duplicate_processing_idempotent(isolated_db: Session):
    session = isolated_db
    passage_text = "We demonstrate fault-tolerant protocols under partition."
    paper = _create_test_paper(session, pub_date=datetime.date(2023, 1, 1), passage_text=passage_text)
    opp = _create_test_opportunity(session, status=OpportunityStatus.OPEN)

    mock_llm = MockLLMProvider(
        default=RelationshipClassification(
            relationship=OpportunityRelationship.SUPPORTS,
            confidence=0.9,
            rationale="Supports opportunity premise.",
            evidence_excerpt=passage_text,
        )
    )
    classifier = RelationshipClassifier(provider=mock_llm)
    trans_service = OpportunityTransitionService(session)
    integration = OpportunityClassificationIntegration(classifier, trans_service, session)
    service = IncrementalOpportunityEvolutionService(integration, session)

    # First run: evaluated
    res1 = service.evaluate_paper(paper_id=paper.id)
    assert res1.evaluated_count == 1
    assert res1.skipped_count == 0

    ev_count_1 = session.execute(
        select(OpportunityEvidence).where(
            OpportunityEvidence.opportunity_id == opp.id,
            OpportunityEvidence.paper_id == paper.id,
        )
    ).scalars().all()
    assert len(ev_count_1) == 1

    hist_count_1 = session.execute(
        select(OpportunityHistory).where(OpportunityHistory.opportunity_id == opp.id)
    ).scalars().all()
    assert len(hist_count_1) == 1

    # Second run: must be idempotently skipped without creating duplicates
    res2 = service.evaluate_paper(paper_id=paper.id)
    assert res2.evaluated_count == 0
    assert res2.skipped_count == 1
    assert res2.outcomes[0].status == OpportunityEvaluationStatus.SKIPPED
    assert "already recorded" in str(res2.outcomes[0].skip_reason)

    ev_count_2 = session.execute(
        select(OpportunityEvidence).where(
            OpportunityEvidence.opportunity_id == opp.id,
            OpportunityEvidence.paper_id == paper.id,
        )
    ).scalars().all()
    assert len(ev_count_2) == 1  # Still exactly 1

    hist_count_2 = session.execute(
        select(OpportunityHistory).where(OpportunityHistory.opportunity_id == opp.id)
    ).scalars().all()
    assert len(hist_count_2) == 1  # Still exactly 1


# ── TEST 5: Ungrounded classification fails closed ────────────────────────────

def test_ungrounded_classification_fails_closed(isolated_db: Session):
    session = isolated_db
    paper = _create_test_paper(
        session,
        abstract="This paper discusses basic linear programming techniques.",
        passage_text="Simplex method solves bounded convex polytopes efficiently.",
        pub_date=datetime.date(2023, 1, 1),
    )
    opp = _create_test_opportunity(session, status=OpportunityStatus.OPEN)

    # Fabricated hallucinated excerpt that does not exist in the paper
    mock_llm = MockLLMProvider(
        default=RelationshipClassification(
            relationship=OpportunityRelationship.ADDRESSES,
            confidence=0.99,
            rationale="Hallucinated addressing excerpt",
            evidence_excerpt="We conclusively solved quantum gravity and cured all known diseases.",
        )
    )
    classifier = RelationshipClassifier(provider=mock_llm)
    trans_service = OpportunityTransitionService(session)
    integration = OpportunityClassificationIntegration(classifier, trans_service, session)
    service = IncrementalOpportunityEvolutionService(integration, session)

    result = service.evaluate_paper(paper_id=paper.id)

    assert result.run_status == IncrementalEvolutionRunStatus.SUCCESS
    assert result.evaluated_count == 0
    assert result.grounding_rejected_count == 1
    assert result.outcomes[0].status == OpportunityEvaluationStatus.GROUNDING_REJECTED
    assert "Ungrounded evidence rejected" in str(result.outcomes[0].skip_reason)

    # Opportunity must remain OPEN
    session.refresh(opp)
    assert opp.status == OpportunityStatus.OPEN.value

    # No evidence or history in DB
    ev = session.execute(
        select(OpportunityEvidence).where(OpportunityEvidence.opportunity_id == opp.id)
    ).scalars().all()
    assert len(ev) == 0

    hist = session.execute(
        select(OpportunityHistory).where(OpportunityHistory.opportunity_id == opp.id)
    ).scalars().all()
    assert len(hist) == 0


# ── TEST 6: Failure on one opportunity does not corrupt other evaluations ──────

def test_failure_on_one_opportunity_does_not_corrupt_others(isolated_db: Session):
    session = isolated_db
    passage_text = "We introduce verifiable fault detection for distributed consensus."
    paper = _create_test_paper(session, pub_date=datetime.date(2023, 1, 1), passage_text=passage_text)

    opp_1 = _create_test_opportunity(
        session,
        title="Opportunity One",
        created_at=datetime.datetime(2022, 1, 1, tzinfo=datetime.timezone.utc),
    )
    opp_2 = _create_test_opportunity(
        session,
        title="Opportunity Two",
        created_at=datetime.datetime(2022, 2, 1, tzinfo=datetime.timezone.utc),
    )

    # Simulate provider failure on Opportunity One, success on Opportunity Two
    mock_llm = MockLLMProvider(
        responses={
            "Opportunity One": ProviderFailureError("LLM Provider Timeout or Outage"),
            "Opportunity Two": RelationshipClassification(
                relationship=OpportunityRelationship.SUPPORTS,
                confidence=0.92,
                rationale="Supports opportunity two",
                evidence_excerpt=passage_text,
            ),
        }
    )
    classifier = RelationshipClassifier(provider=mock_llm)
    trans_service = OpportunityTransitionService(session)
    integration = OpportunityClassificationIntegration(classifier, trans_service, session)
    service = IncrementalOpportunityEvolutionService(integration, session)

    result = service.evaluate_paper(paper_id=paper.id)

    assert result.run_status == IncrementalEvolutionRunStatus.PARTIAL_SUCCESS
    assert result.failed_count == 1
    assert result.evaluated_count == 1

    outcome_map = {o.opportunity_id: o for o in result.outcomes}
    # Opp 1 failed gracefully
    assert outcome_map[opp_1.id].status == OpportunityEvaluationStatus.FAILED
    assert "LLM Provider Timeout or Outage" in str(outcome_map[opp_1.id].error_message)

    # Opp 2 succeeded despite Opp 1 failure
    assert outcome_map[opp_2.id].status == OpportunityEvaluationStatus.EVALUATED
    assert outcome_map[opp_2.id].final_status == OpportunityStatus.STRENGTHENED

    session.refresh(opp_1)
    session.refresh(opp_2)
    assert opp_1.status == OpportunityStatus.OPEN.value
    assert opp_2.status == OpportunityStatus.STRENGTHENED.value

    # DB session remains usable and unpoisoned
    db_test = session.execute(select(Paper).where(Paper.id == paper.id)).scalar_one_or_none()
    assert db_test is not None


# ── TEST 7: Deterministic candidate ordering ──────────────────────────────────

def test_deterministic_candidate_ordering(isolated_db: Session):
    session = isolated_db
    passage_text = "Consensus algorithm bounds."
    paper = _create_test_paper(session, pub_date=datetime.date(2023, 1, 1), passage_text=passage_text)

    # Create opportunities out of order
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
    classifier = RelationshipClassifier(provider=mock_llm)
    trans_service = OpportunityTransitionService(session)
    integration = OpportunityClassificationIntegration(classifier, trans_service, session)
    service = IncrementalOpportunityEvolutionService(integration, session)

    result = service.evaluate_paper(paper_id=paper.id)

    assert result.evaluated_count == 3
    # Candidate order in outcomes must strictly be [opp_a, opp_b, opp_c]
    evaluated_ids = [o.opportunity_id for o in result.outcomes]
    assert evaluated_ids == [opp_a.id, opp_b.id, opp_c.id]


# ── TEST 8: Atomic evidence/status/history behavior ───────────────────────────

def test_atomic_evidence_status_history_behavior(isolated_db: Session):
    session = isolated_db
    passage_text = "Atomic transaction testing passage."
    paper = _create_test_paper(session, pub_date=datetime.date(2023, 1, 1), passage_text=passage_text)
    opp = _create_test_opportunity(session, status=OpportunityStatus.OPEN)

    mock_llm = MockLLMProvider(
        default=RelationshipClassification(
            relationship=OpportunityRelationship.ADDRESSES,
            confidence=0.9,
            rationale="Atomic test transition",
            evidence_excerpt=passage_text,
        )
    )
    classifier = RelationshipClassifier(provider=mock_llm)
    trans_service = OpportunityTransitionService(session)
    integration = OpportunityClassificationIntegration(classifier, trans_service, session)
    service = IncrementalOpportunityEvolutionService(integration, session)

    # Savepoint inside outer transaction
    outer_sp = session.begin_nested()

    result = service.evaluate_paper(paper_id=paper.id)
    assert result.evaluated_count == 1

    # Changes exist inside transaction
    session.refresh(opp)
    assert opp.status == OpportunityStatus.PARTIALLY_ADDRESSED.value

    # Caller rolls back outer transaction
    outer_sp.rollback()

    # Opportunity status and tables are rolled back cleanly
    session.refresh(opp)
    assert opp.status == OpportunityStatus.OPEN.value

    ev = session.execute(
        select(OpportunityEvidence).where(OpportunityEvidence.opportunity_id == opp.id)
    ).scalars().all()
    assert len(ev) == 0

    hist = session.execute(
        select(OpportunityHistory).where(OpportunityHistory.opportunity_id == opp.id)
    ).scalars().all()
    assert len(hist) == 0


def _unrelated_service(session: Session, mock_llm: "MockLLMProvider", max_candidates=None):
    classifier = RelationshipClassifier(provider=mock_llm)
    trans_service = OpportunityTransitionService(session)
    integration = OpportunityClassificationIntegration(classifier, trans_service, session)
    return IncrementalOpportunityEvolutionService(integration, session, max_candidates=max_candidates)


def test_candidate_limit_applies_after_eligibility_filtering(isolated_db: Session):
    """Ineligible early candidates must not consume the cap; eligible candidates beyond the
    cap are recorded as SKIPPED with an explicit reason rather than dropped."""
    session = isolated_db
    paper = _create_test_paper(session, pub_date=datetime.date(2023, 1, 1))

    already_evaluated = [
        _create_test_opportunity(session, title=f"Old {i}",
                                 created_at=datetime.datetime(2021, 1, i + 1, tzinfo=datetime.timezone.utc))
        for i in range(2)
    ]
    for opp in already_evaluated:
        session.add(OpportunityEvidence(
            opportunity_id=opp.id, paper_id=paper.id,
            evidence_type=OpportunityRelationship.UNRELATED.value, supporting_text=None,
        ))
    eligible = [
        _create_test_opportunity(session, title=f"Eligible {i}",
                                 created_at=datetime.datetime(2022, 1, i + 1, tzinfo=datetime.timezone.utc))
        for i in range(3)
    ]
    session.flush()

    mock_llm = MockLLMProvider()
    result = _unrelated_service(session, mock_llm, max_candidates=2).evaluate_paper(paper_id=paper.id)

    outcome = {o.opportunity_id: o for o in result.outcomes}
    assert len(mock_llm.call_history) == 2
    assert result.evaluated_count == 2
    assert [outcome[o.id].status for o in eligible[:2]] == [OpportunityEvaluationStatus.EVALUATED] * 2
    assert outcome[eligible[2].id].status == OpportunityEvaluationStatus.SKIPPED
    assert "Candidate limit (2) reached" in outcome[eligible[2].id].skip_reason
    for opp in already_evaluated:
        assert "already recorded" in outcome[opp.id].skip_reason


def test_candidate_limit_per_call_override_and_unbounded_default(isolated_db: Session):
    session = isolated_db
    paper = _create_test_paper(session, pub_date=datetime.date(2023, 1, 1))
    for i in range(3):
        _create_test_opportunity(session, title=f"Eligible {i}",
                                 created_at=datetime.datetime(2022, 1, i + 1, tzinfo=datetime.timezone.utc))
    session.flush()

    mock_llm = MockLLMProvider()
    result = _unrelated_service(session, mock_llm, max_candidates=3).evaluate_paper(paper_id=paper.id, max_candidates=1)
    assert len(mock_llm.call_history) == 1
    assert result.skipped_count == 2

    session.rollback()
    paper = _create_test_paper(session, pub_date=datetime.date(2023, 1, 1))
    for i in range(3):
        _create_test_opportunity(session, title=f"Eligible {i}",
                                 created_at=datetime.datetime(2022, 1, i + 1, tzinfo=datetime.timezone.utc))
    session.flush()
    mock_llm = MockLLMProvider()
    _unrelated_service(session, mock_llm).evaluate_paper(paper_id=paper.id)
    assert len(mock_llm.call_history) == 3


def test_post_ingestion_evolution_service_is_built_with_configured_cap(monkeypatch, isolated_db: Session):
    from ideagen.db.persistence import SQLAlchemyPersistenceLayer
    from ideagen.ingestion.openalex_ingestion import OpenAlexPaperIngestionService

    monkeypatch.setenv("IDEAGEN_POST_INGESTION_EVOLUTION_MAX_CANDIDATES", "7")
    service = OpenAlexPaperIngestionService(persistence=SQLAlchemyPersistenceLayer(isolated_db), auto_evolve=True)
    evo = service._get_or_create_evolution_service()
    assert evo is not None
    assert evo.max_candidates == 7
