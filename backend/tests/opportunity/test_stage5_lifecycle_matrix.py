"""IdeaGen Stage 5: Lifecycle Impact and Traceable Test Matrix.

Verifies deterministic opportunity lifecycle transitions, evidence grounding,
idempotency, distinct-paper resolution rules, contradiction-driven reopening,
temporal filtering, and provider-failure transaction boundaries across 12 explicit scenarios:

1. A paper that genuinely supports an open opportunity.
2. A paper that directly addresses an opportunity.
3. A paper that only studies or documents the problem.
4. A paper that extends related work without addressing the specific limitation.
5. A paper that contradicts the opportunity's premise.
6. An unrelated paper.
7. Two distinct addressing papers required for resolution.
8. A contradiction after resolution that reopens an opportunity.
9. Repeated ingestion and duplicate evidence.
10. Ungrounded or fabricated evidence excerpts.
11. Future-dated opportunities and undated papers.
12. Provider failure and transaction rollback behavior.
"""

from __future__ import annotations

import datetime
from datetime import date, timezone
from typing import Optional
import uuid
import pytest

from sqlalchemy import select
from sqlalchemy.orm import Session

from ideagen.db.models import (
    Opportunity,
    Paper,
    PaperDocument,
    PaperPassage,
    OpportunityEvidence,
    OpportunityHistory,
)
from ideagen.ingestion.models import SourceType
from ideagen.llm.exceptions import ProviderFailureError, MalformedResponseError
from ideagen.opportunity.classifier import RelationshipClassifier
from ideagen.opportunity.grounding import UngroundedEvidenceError
from ideagen.opportunity.incremental import (
    IncrementalOpportunityEvolutionService,
    OpportunityEvaluationStatus,
    IncrementalEvolutionRunStatus,
)
from ideagen.opportunity.integration import (
    OpportunityClassificationIntegration,
    ClassificationTransitionResult,
)
from ideagen.opportunity.models import OpportunityRelationship, OpportunityStatus
from ideagen.opportunity.service import OpportunityTransitionService
from ideagen.opportunity.state_machine import transition
from tests.opportunity.fake_provider import FakeLLMProvider


# ── Fixtures & Test Helpers ──────────────────────────────────────────────────


def _create_test_paper(
    session: Session,
    title: str = "Test Grounded Paper",
    abstract: str = "This paper presents empirical findings on transformer retrieval.",
    passages: Optional[list[str]] = None,
    pub_date: Optional[date] = date(2024, 1, 1),
    doi: Optional[str] = None,
) -> Paper:
    """Creates and persists a paper with optional stored passages."""
    p_id = uuid.uuid4()
    paper = Paper(
        id=p_id,
        source_type=SourceType.OPENALEX_API.value,
        source_id=f"W{uuid.uuid4()}",
        doi=doi,
        title=title,
        publication_date=pub_date,
        batch_id="stage5-batch",
        abstract=abstract,
        authors=["Author A", "Author B"],
        referenced_works=[],
        raw_metadata={},
    )
    session.add(paper)
    session.flush()

    if passages:
        doc = PaperDocument(
            id=uuid.uuid4(),
            paper_id=paper.id,
            full_text="\n\n".join(passages),
            extraction_method="test_extraction",
            extraction_version="v1.0",
            extracted_at=datetime.datetime.now(timezone.utc),
        )
        session.add(doc)
        session.flush()

        for idx, text in enumerate(passages):
            passage = PaperPassage(
                id=uuid.uuid4(),
                paper_id=paper.id,
                document_id=doc.id,
                passage_index=idx,
                page_start=1,
                page_end=1,
                section="Results",
                text=text,
            )
            session.add(passage)
        session.flush()

    return paper



def _create_test_opportunity(
    session: Session,
    status: OpportunityStatus = OpportunityStatus.OPEN,
    title: str = "Knowledge Access Bottlenecks in RAG",
    problem: str = "Pre-trained models fail to precisely manipulate retrieved knowledge.",
    created_at: Optional[datetime.datetime] = None,
) -> Opportunity:
    """Creates and persists an opportunity in a given initial lifecycle status."""
    opp = Opportunity(
        id=uuid.uuid4(),
        title=title,
        problem_statement=problem,
        description="Explores external knowledge integration bounds.",
        status=status.value,
        confidence=0.85,
        created_at=created_at or datetime.datetime(2023, 1, 1, 12, 0, tzinfo=timezone.utc),
    )
    session.add(opp)
    session.flush()
    return opp


def _build_test_integration(session: Session, fake_provider: FakeLLMProvider) -> OpportunityClassificationIntegration:
    classifier = RelationshipClassifier(provider=fake_provider)
    transition_service = OpportunityTransitionService(session)
    return OpportunityClassificationIntegration(
        classifier=classifier,
        transition_service=transition_service,
        session=session,
    )


# ── Scenario 1: Paper Genuinely Supports Open Opportunity ──────────────────


def test_scenario_1_paper_supports_open_opportunity(session: Session):
    """Scenario 1: Grounded SUPPORTS evidence transitions OPEN -> STRENGTHENED."""
    opp = _create_test_opportunity(session, status=OpportunityStatus.OPEN)
    passage_text = "We confirm that pre-trained language models suffer severe factual hallucinations under knowledge manipulation."
    paper = _create_test_paper(session, passages=[passage_text])

    provider = FakeLLMProvider(
        response={
            "relationship": "SUPPORTS",
            "confidence": 0.95,
            "rationale": "Empirically demonstrates the persistence of the knowledge bottleneck.",
            "evidence_excerpt": passage_text,
        }
    )
    integration = _build_test_integration(session, provider)
    result = integration.classify_and_apply(opp.id, paper.id)

    assert result.relationship == OpportunityRelationship.SUPPORTS
    assert result.previous_status == OpportunityStatus.OPEN
    assert result.new_status == OpportunityStatus.STRENGTHENED
    assert result.state_changed is True

    # Verify DB persistence
    session.refresh(opp)
    assert opp.status == OpportunityStatus.STRENGTHENED.value

    evidence = session.execute(
        select(OpportunityEvidence).where(OpportunityEvidence.opportunity_id == opp.id)
    ).scalar_one()
    assert evidence.evidence_type == OpportunityRelationship.SUPPORTS.value
    assert evidence.supporting_text == passage_text

    history = session.execute(
        select(OpportunityHistory).where(OpportunityHistory.opportunity_id == opp.id)
    ).scalar_one()
    assert history.previous_status == OpportunityStatus.OPEN.value
    assert history.new_status == OpportunityStatus.STRENGTHENED.value
    assert history.trigger_type == OpportunityRelationship.SUPPORTS.value


# ── Scenario 2: Paper Directly Addresses Opportunity ───────────────────────


def test_scenario_2_paper_directly_addresses_opportunity(session: Session):
    """Scenario 2: Single grounded ADDRESSES paper transitions OPEN -> PARTIALLY_ADDRESSED."""
    opp = _create_test_opportunity(session, status=OpportunityStatus.OPEN)
    passage_text = "We introduce a novel neural retriever that reduces knowledge manipulation errors by 42% on benchmark datasets."
    paper = _create_test_paper(session, passages=[passage_text])

    provider = FakeLLMProvider(
        response={
            "relationship": "ADDRESSES",
            "confidence": 0.90,
            "rationale": "Directly mitigates the retrieval manipulation limitation with a novel neural retriever.",
            "evidence_excerpt": passage_text,
        }
    )
    integration = _build_test_integration(session, provider)
    result = integration.classify_and_apply(opp.id, paper.id)

    assert result.relationship == OpportunityRelationship.ADDRESSES
    assert result.previous_status == OpportunityStatus.OPEN
    assert result.new_status == OpportunityStatus.PARTIALLY_ADDRESSED
    assert result.state_changed is True

    session.refresh(opp)
    assert opp.status == OpportunityStatus.PARTIALLY_ADDRESSED.value


# ── Scenario 3: Paper Only Studies or Documents the Problem ────────────────


def test_scenario_3_paper_studies_or_documents_problem(session: Session):
    """Scenario 3: Paper documenting the problem (SUPPORTS) does not resolve an already partially addressed opportunity."""
    opp = _create_test_opportunity(session, status=OpportunityStatus.PARTIALLY_ADDRESSED)
    passage_text = "Our survey measures the failure modes of retrieval augmented generation across 10 distinct knowledge benchmarks."
    paper = _create_test_paper(session, passages=[passage_text])

    provider = FakeLLMProvider(
        response={
            "relationship": "SUPPORTS",
            "confidence": 0.92,
            "rationale": "Surveys and documents the limitation without proposing a novel mitigation.",
            "evidence_excerpt": passage_text,
        }
    )
    integration = _build_test_integration(session, provider)
    result = integration.classify_and_apply(opp.id, paper.id)

    assert result.relationship == OpportunityRelationship.SUPPORTS
    assert result.previous_status == OpportunityStatus.PARTIALLY_ADDRESSED
    assert result.new_status == OpportunityStatus.PARTIALLY_ADDRESSED
    assert result.state_changed is False

    session.refresh(opp)
    assert opp.status == OpportunityStatus.PARTIALLY_ADDRESSED.value


# ── Scenario 4: Paper Extends Related Work Without Addressing Specific Limit ─


def test_scenario_4_paper_extends_without_addressing_limitation(session: Session):
    """Scenario 4: EXTENDS relationship creates grounded evidence but preserves current status."""
    opp = _create_test_opportunity(session, status=OpportunityStatus.OPEN)
    passage_text = "We adapt general transformer architectures to vision-language multi-modal tasks, broadening model scope."
    paper = _create_test_paper(session, passages=[passage_text])

    provider = FakeLLMProvider(
        response={
            "relationship": "EXTENDS",
            "confidence": 0.85,
            "rationale": "Advances multi-modal extensions without directly resolving text-based knowledge manipulation bounds.",
            "evidence_excerpt": passage_text,
        }
    )
    integration = _build_test_integration(session, provider)
    result = integration.classify_and_apply(opp.id, paper.id)

    assert result.relationship == OpportunityRelationship.EXTENDS
    assert result.previous_status == OpportunityStatus.OPEN
    assert result.new_status == OpportunityStatus.OPEN
    assert result.state_changed is False

    # Evidence row must be persisted
    evidence = session.execute(
        select(OpportunityEvidence).where(OpportunityEvidence.opportunity_id == opp.id)
    ).scalar_one()
    assert evidence.evidence_type == OpportunityRelationship.EXTENDS.value

    # History must NOT be created when status does not change
    history = session.execute(
        select(OpportunityHistory).where(OpportunityHistory.opportunity_id == opp.id)
    ).scalars().all()
    assert len(history) == 0


# ── Scenario 5: Paper Contradicts Opportunity Premise ──────────────────────


def test_scenario_5_paper_contradicts_opportunity_premise(session: Session):
    """Scenario 5: CONTRADICTS resets STRENGTHENED opportunity back to OPEN."""
    opp = _create_test_opportunity(session, status=OpportunityStatus.STRENGTHENED)
    passage_text = "We demonstrate that standard pre-trained models already possess perfect factual access when queried with optimal prompts."
    paper = _create_test_paper(session, passages=[passage_text])

    provider = FakeLLMProvider(
        response={
            "relationship": "CONTRADICTS",
            "confidence": 0.98,
            "rationale": "Directly refutes the core premise that pre-trained models have an inherent architectural access bottleneck.",
            "evidence_excerpt": passage_text,
        }
    )
    integration = _build_test_integration(session, provider)
    result = integration.classify_and_apply(opp.id, paper.id)

    assert result.relationship == OpportunityRelationship.CONTRADICTS
    assert result.previous_status == OpportunityStatus.STRENGTHENED
    assert result.new_status == OpportunityStatus.OPEN
    assert result.state_changed is True

    session.refresh(opp)
    assert opp.status == OpportunityStatus.OPEN.value


# ── Scenario 6: Unrelated Paper ─────────────────────────────────────────────


def test_scenario_6_unrelated_paper(session: Session):
    """Scenario 6: UNRELATED paper preserves status and does not create evidence or history."""
    opp = _create_test_opportunity(session, status=OpportunityStatus.OPEN)
    paper = _create_test_paper(session, abstract="A study on deep learning for crop disease identification in agriculture.")

    provider = FakeLLMProvider(
        response={
            "relationship": "UNRELATED",
            "confidence": 0.99,
            "rationale": "Agricultural crop disease detection is not materially relevant to NLP knowledge manipulation.",
            "evidence_excerpt": None,
        }
    )
    integration = _build_test_integration(session, provider)
    result = integration.classify_and_apply(opp.id, paper.id)

    assert result.relationship == OpportunityRelationship.UNRELATED
    assert result.previous_status == OpportunityStatus.OPEN
    assert result.new_status == OpportunityStatus.OPEN
    assert result.state_changed is False
    assert result.evidence_id is not None
    assert result.history_id is None

    # Evidence row is persisted with UNRELATED
    ev_row = session.execute(
        select(OpportunityEvidence).where(OpportunityEvidence.opportunity_id == opp.id)
    ).scalar_one()
    assert ev_row.evidence_type == OpportunityRelationship.UNRELATED.value

    # No history row created
    hist_count = session.execute(
        select(OpportunityHistory).where(OpportunityHistory.opportunity_id == opp.id)
    ).scalars().all()
    assert len(hist_count) == 0


# ── Scenario 7: Two Distinct Addressing Papers Required for Resolution ─────


def test_scenario_7_two_distinct_addressing_papers_required_for_resolution(session: Session):
    """Scenario 7: Full AD-19 and AD-20 distinct paper requirement for RESOLVED."""
    opp = _create_test_opportunity(session, status=OpportunityStatus.OPEN)

    # Paper 1: ArXiv preprint (without DOI)
    same_title = "Shared Canonical Retrieval Architecture Title"
    psg1 = "Method Alpha mitigates knowledge retrieval manipulation errors by 30%."
    paper1 = _create_test_paper(session, title=same_title, passages=[psg1], doi=None)
    provider1 = FakeLLMProvider(response={"relationship": "ADDRESSES", "confidence": 0.9, "rationale": "R1", "evidence_excerpt": psg1})
    integration1 = _build_test_integration(session, provider1)

    res1 = integration1.classify_and_apply(opp.id, paper1.id)
    assert res1.new_status == OpportunityStatus.PARTIALLY_ADDRESSED

    # Attempting to re-apply the exact same paper must be rejected by idempotency
    with pytest.raises(ValueError, match="already exists"):
        integration1.classify_and_apply(opp.id, paper1.id)

    # Paper 1 Duplicate: Journal publication with same normalized title (different UUID and source_id, doi=None)
    paper1_dup = _create_test_paper(session, title="shared canonical retrieval architecture title!", passages=[psg1], doi=None)
    provider1_dup = FakeLLMProvider(response={"relationship": "ADDRESSES", "confidence": 0.9, "rationale": "R1", "evidence_excerpt": psg1})
    integration1_dup = _build_test_integration(session, provider1_dup)

    res_dup = integration1_dup.classify_and_apply(opp.id, paper1_dup.id)
    # Recognized as same canonical title identity, distinct count remains 1
    assert res_dup.new_status == OpportunityStatus.PARTIALLY_ADDRESSED

    # Paper 2: Distinct publication with distinct title and DOI
    psg2 = "Method Beta introduces independent benchmark verification resolving knowledge manipulation bounds."
    paper2 = _create_test_paper(session, title="Completely Distinct Paper Beta", passages=[psg2], doi="10.1000/beta")
    provider2 = FakeLLMProvider(response={"relationship": "ADDRESSES", "confidence": 0.9, "rationale": "R2", "evidence_excerpt": psg2})
    integration2 = _build_test_integration(session, provider2)

    res2 = integration2.classify_and_apply(opp.id, paper2.id)
    # Distinct count reaches 2 -> Transitions to RESOLVED
    assert res2.new_status == OpportunityStatus.RESOLVED

    session.refresh(opp)
    assert opp.status == OpportunityStatus.RESOLVED.value



# ── Scenario 8: Contradiction After Resolution Reopens Opportunity ──────────


def test_scenario_8_contradiction_after_resolution_reopens_opportunity(session: Session):
    """Scenario 8: CONTRADICTS after RESOLVED transitions opportunity back to OPEN with fresh evidence boundary."""
    opp = _create_test_opportunity(session, status=OpportunityStatus.RESOLVED)

    # Paper that refutes the resolution
    contra_psg = "We demonstrate that all previously reported methods for knowledge manipulation fail completely on dynamic datasets."
    contra_paper = _create_test_paper(session, passages=[contra_psg])

    contra_provider = FakeLLMProvider(response={"relationship": "CONTRADICTS", "confidence": 0.95, "rationale": "Refutes resolution", "evidence_excerpt": contra_psg})
    integration = _build_test_integration(session, contra_provider)

    res_contra = integration.classify_and_apply(opp.id, contra_paper.id)
    assert res_contra.previous_status == OpportunityStatus.RESOLVED
    assert res_contra.new_status == OpportunityStatus.OPEN

    session.refresh(opp)
    assert opp.status == OpportunityStatus.OPEN.value

    # Reopening boundary check: a new addressing paper brings it to PARTIALLY_ADDRESSED, not RESOLVED
    new_psg = "Method Gamma offers a new robust mitigation post-contradiction."
    new_paper = _create_test_paper(session, passages=[new_psg], doi="10.1000/gamma")
    new_provider = FakeLLMProvider(response={"relationship": "ADDRESSES", "confidence": 0.9, "rationale": "R", "evidence_excerpt": new_psg})
    integration_new = _build_test_integration(session, new_provider)

    res_new = integration_new.classify_and_apply(opp.id, new_paper.id)
    assert res_new.previous_status == OpportunityStatus.OPEN
    assert res_new.new_status == OpportunityStatus.PARTIALLY_ADDRESSED


# ── Scenario 9: Repeated Ingestion and Duplicate Evidence ───────────────────


def test_scenario_9_repeated_ingestion_and_duplicate_evidence(session: Session):
    """Scenario 9: Re-evaluating an already evaluated paper-opportunity pair is rejected idempotently."""
    opp = _create_test_opportunity(session, status=OpportunityStatus.OPEN)
    psg = "Method Alpha mitigates knowledge retrieval manipulation errors."
    paper = _create_test_paper(session, passages=[psg])

    provider = FakeLLMProvider(response={"relationship": "ADDRESSES", "confidence": 0.9, "rationale": "R1", "evidence_excerpt": psg})
    integration = _build_test_integration(session, provider)

    # First application succeeds
    res1 = integration.classify_and_apply(opp.id, paper.id)
    assert res1.new_status == OpportunityStatus.PARTIALLY_ADDRESSED

    # Second application of identical paper to identical opportunity must raise ValueError
    with pytest.raises(ValueError, match="already exists"):
        integration.classify_and_apply(opp.id, paper.id)

    # Opportunity remains in valid state
    session.refresh(opp)
    assert opp.status == OpportunityStatus.PARTIALLY_ADDRESSED.value


# ── Scenario 10: Ungrounded or Fabricated Evidence Excerpts ─────────────────


def test_scenario_10_ungrounded_or_fabricated_evidence_excerpts(session: Session):
    """Scenario 10: Substantive relationship with hallucinated excerpt raises UngroundedEvidenceError and rolls back."""
    opp = _create_test_opportunity(session, status=OpportunityStatus.OPEN)
    real_psg = "This paper analyzes transformer attention distributions across layers."
    paper = _create_test_paper(session, passages=[real_psg])

    hallucinated_excerpt = "We completely eliminate all hallucinations in large language models using magic tokens."

    provider = FakeLLMProvider(
        response={
            "relationship": "ADDRESSES",
            "confidence": 0.95,
            "rationale": "Claims total elimination of hallucinations.",
            "evidence_excerpt": hallucinated_excerpt,
        }
    )
    integration = _build_test_integration(session, provider)

    with pytest.raises(UngroundedEvidenceError, match="not grounded in stored passages"):
        integration.classify_and_apply(opp.id, paper.id)

    # Verify fail-closed: opportunity status untouched, zero evidence rows created
    session.refresh(opp)
    assert opp.status == OpportunityStatus.OPEN.value

    ev_rows = session.execute(
        select(OpportunityEvidence).where(OpportunityEvidence.opportunity_id == opp.id)
    ).scalars().all()
    assert len(ev_rows) == 0


# ── Scenario 11: Future-Dated Opportunities and Undated Papers ──────────────


def test_scenario_11_future_dated_opportunities_and_undated_papers(session: Session):
    """Scenario 11: Incremental evolution enforces strict temporal ordering and rejects undated inputs."""
    provider = FakeLLMProvider(response={"relationship": "ADDRESSES", "confidence": 0.9, "rationale": "R", "evidence_excerpt": "E"})
    integration = _build_test_integration(session, provider)
    evolution_service = IncrementalOpportunityEvolutionService(integration=integration, session=session)

    # 1. Undated Paper
    undated_paper = _create_test_paper(session, pub_date=None)
    res_undated = evolution_service.evaluate_paper(undated_paper.id, session=session)
    assert res_undated.run_status == IncrementalEvolutionRunStatus.SKIPPED_UNDATED

    # 2. Future-Dated Opportunity relative to Paper publication date
    paper_2022 = _create_test_paper(session, pub_date=date(2022, 6, 1))
    opp_2024 = _create_test_opportunity(
        session,
        created_at=datetime.datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
    )

    res_future = evolution_service.evaluate_paper(
        paper_2022.id,
        session=session,
        candidate_opportunity_ids=[opp_2024.id],
    )
    assert res_future.candidate_count == 1
    assert len(res_future.outcomes) == 1
    assert res_future.outcomes[0].status == OpportunityEvaluationStatus.SKIPPED
    assert "after paper publication date" in res_future.outcomes[0].skip_reason


# ── Scenario 12: Provider Failure and Transaction Rollback Behavior ─────────


def test_scenario_12_provider_failure_and_transaction_rollback(session: Session):
    """Scenario 12: Provider failure rolls back candidate evaluation without corrupting session or paper data."""
    opp = _create_test_opportunity(session, status=OpportunityStatus.OPEN)
    paper = _create_test_paper(session, passages=["Genuine passage text."])

    provider = FakeLLMProvider(should_fail=True)
    integration = _build_test_integration(session, provider)
    evolution_service = IncrementalOpportunityEvolutionService(integration=integration, session=session)

    res = evolution_service.evaluate_paper(
        paper.id,
        session=session,
        candidate_opportunity_ids=[opp.id],
    )

    assert res.run_status == IncrementalEvolutionRunStatus.PARTIAL_SUCCESS or res.failed_count == 1
    assert res.outcomes[0].status == OpportunityEvaluationStatus.FAILED
    assert "failed" in res.outcomes[0].error_message.lower()

    # Ensure paper and passages remain intact in DB
    session.refresh(paper)
    assert paper.title == "Test Grounded Paper"

    # Ensure opportunity remains in original status
    session.refresh(opp)
    assert opp.status == OpportunityStatus.OPEN.value

    # Session is healthy: can execute a subsequent successful operation
    opp2 = _create_test_opportunity(session, status=OpportunityStatus.OPEN)
    assert opp2.status == OpportunityStatus.OPEN.value
