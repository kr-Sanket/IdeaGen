"""Tests for Stage 0 Read-Only API Contract Preparation.

Verifies:
1. GET /api/v1/run-metadata returns run fingerprint, temporal cutoffs, screening decisions, and counts.
2. GET /api/v1/opportunities exposes lifecycle_status, lifecycle_opportunity_id, and evidence_count.
3. GET /api/v1/opportunities/{id}/evidence exposes paper_publication_date, batch_id, and evaluation_cutoff.
4. GET /api/v1/papers exposes temporal_role (BASELINE vs LATER_EVALUATION) and is_baseline.
5. GET /api/v1/statements exposes passage_text provenance.
6. GET requests do NOT write, bridge, or flush Opportunity records to the database as a side effect.
7. POST /api/v1/gaps/{id}/opportunity does not fabricate fallback results on LLM failure.
"""

import datetime
import json
import uuid
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from ideagen.api.main import app
from ideagen.db import (
    Base,
    get_db,
    Paper,
    PaperDocument,
    PaperPassage,
    ResearchStatement,
    ResearchGap,
    GapEvidence,
    Opportunity,
    OpportunityEvidence,
    OpportunityHistory,
    ResearchOpportunityRecord,
    ExperimentMetadataRecord,
)
from ideagen.opportunity.models import OpportunityRelationship, OpportunityStatus as LifecycleStatus


class FakeEmbeddingProvider:
    @property
    def dimension(self) -> int:
        return 768

    @property
    def model_name(self) -> str:
        return "fake-nomic-embed"

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        return np.zeros((len(texts), self.dimension), dtype=np.float32)


@pytest.fixture(autouse=True)
def mock_pipeline_dependencies():
    fake_provider = FakeEmbeddingProvider()
    with patch(
        "ideagen.api.main.get_default_embedding_provider",
        return_value=fake_provider,
    ), patch(
        "ideagen.discovery.embeddings.get_default_embedding_provider",
        return_value=fake_provider,
    ):
        yield fake_provider


@pytest.fixture
def test_db():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    SessionMaker = sessionmaker(bind=engine)
    session = SessionMaker()

    # 1. Experiment metadata
    fp = {
        "llm_provider": "OllamaLLMProvider",
        "llm_model": "qwen3:8b",
        "embedding_provider": "OllamaEmbeddingProvider",
        "embedding_model": "nomic-embed-text",
        "embedding_dimension": 768,
        "temporal_cutoff": "2020-12-31",
        "enable_stage_5": True,
        "stage_5_cutoffs": ["2023-01-31", "2023-05-31", "2023-10-31"],
    }
    meta_fp = ExperimentMetadataRecord(
        key="configuration_fingerprint",
        value=json.dumps(fp),
    )
    session.add(meta_fp)

    screening_data = {
        "topic": "Retrieval Augmented Generation",
        "threshold": 0.5,
        "total_eligible_papers": 5,
        "selected_count": 4,
        "excluded_count": 1,
    }
    meta_screen = ExperimentMetadataRecord(
        key="relevance_screening_report",
        value=json.dumps(screening_data),
    )
    session.add(meta_screen)

    # 2. Papers: Baseline (2020) and Later Evaluation (2023)
    p_baseline = Paper(
        id=uuid.uuid4(),
        batch_id="batch_0",
        source_type="MANUAL_UPLOAD",
        source_id="lewis2020",
        title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        publication_date=datetime.date(2020, 5, 22),
        abstract="We explore RAG models combining pre-trained parametric memory with non-parametric memory.",
        authors=["Patrick Lewis", "Ethan Perez"],
        referenced_works=[],
        raw_metadata={},
    )
    session.add(p_baseline)

    p_later = Paper(
        id=uuid.uuid4(),
        batch_id="batch_2_2023-10-31",
        source_type="MANUAL_UPLOAD",
        source_id="asai2023",
        title="Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection",
        publication_date=datetime.date(2023, 10, 17),
        abstract="SELF-RAG trains an arbitrary LM to generate text with reflection tokens.",
        authors=["Akari Asai", "Zeqiu Wu"],
        referenced_works=[],
        raw_metadata={},
    )
    session.add(p_later)

    p_undated = Paper(
        id=uuid.uuid4(),
        batch_id="batch_0",
        source_type="MANUAL_UPLOAD",
        source_id="undated_paper",
        title="Undated Methodology Paper",
        publication_date=None,
        abstract="Undated paper abstract.",
        authors=[],
        referenced_works=[],
        raw_metadata={},
    )
    session.add(p_undated)

    # 3. Documents and Passages
    doc1 = PaperDocument(
        id=uuid.uuid4(),
        paper_id=p_baseline.id,
        full_text="Full baseline text.",
        extraction_method="PDFMiner",
        extraction_version="1.0",
        extracted_at=datetime.datetime.now(datetime.timezone.utc),
    )
    session.add(doc1)
    pass1 = PaperPassage(
        id=uuid.uuid4(),
        document_id=doc1.id,
        paper_id=p_baseline.id,
        passage_index=0,
        page_start=1,
        page_end=1,
        section="Abstract",
        text="RAG models combine parametric and non-parametric memory.",
    )
    session.add(pass1)


    # 4. Statements
    stmt1 = ResearchStatement(
        id=uuid.uuid4(),
        knowledge_id=uuid.uuid4(),
        paper_id=p_baseline.id,
        passage_id=pass1.id,
        statement_text="Parametric memory alone struggles with factual precision.",
        statement_type="LIMITATION",
        confidence=0.92,
        section="Abstract",
    )
    session.add(stmt1)

    # 5. Gap
    gap = ResearchGap(
        id=uuid.uuid4(),
        gap_statement="Unquantified hallucination accumulation in multi-hop RAG.",
        research_problem="Multi-hop retrieval error propagation",
        status="VALIDATED",
        evidence_count=1,
    )
    session.add(gap)

    # 6. Bridged Opportunity
    rec_bridged = ResearchOpportunityRecord(
        id=uuid.uuid4(),
        gap_id=gap.id,
        research_problem=gap.research_problem,
        identified_gap=gap.gap_statement,
        proposed_research_direction="Investigate differentiable memory integration for RAG.",
        reasoning="Recurring factual precision bottlenecks.",
        technical_approach="Differentiable key-value index augmentation.",
        expected_contribution="Bounded hallucination bounds on multi-hop benchmarks.",
        feasibility={"feasibility_score": 0.85},
        novelty_assessment={"assessment": "Potentially novel within the analyzed corpus"},
        confidence=0.88,
        supporting_statement_ids=[str(stmt1.id)],
        supporting_paper_ids=[str(p_baseline.id)],
        status="VALIDATED",
        generation_model="qwen3:8b",
        validation_model="qwen3:8b",
        embedding_model="nomic-embed-text",
    )
    session.add(rec_bridged)

    lifecycle_opp = Opportunity(
        id=uuid.uuid4(),
        research_opportunity_id=rec_bridged.id,
        gap_id=gap.id,
        title="Differentiable Memory Integration",
        problem_statement=gap.research_problem,
        description="Lifecycle description",
        status=LifecycleStatus.OPEN.value,
        confidence=0.88,
    )
    session.add(lifecycle_opp)

    ev1 = OpportunityEvidence(
        id=uuid.uuid4(),
        opportunity_id=lifecycle_opp.id,
        paper_id=p_later.id,
        evidence_type=OpportunityRelationship.EXTENDS.value,
        supporting_text="SELF-RAG trains an arbitrary LM to generate text with reflection tokens.",
    )
    session.add(ev1)

    # 7. Unbridged Opportunity (has ResearchOpportunityRecord but NO Opportunity row)
    rec_unbridged = ResearchOpportunityRecord(
        id=uuid.uuid4(),
        gap_id=gap.id,
        research_problem="Secondary Research Problem",
        identified_gap="Secondary Gap",
        proposed_research_direction="Secondary Direction",
        reasoning="Secondary Reasoning",
        technical_approach="Secondary Approach",
        expected_contribution="Secondary Contribution",
        feasibility={"feasibility_score": 0.8},
        novelty_assessment={"assessment": "Potentially novel"},
        confidence=0.75,
        supporting_statement_ids=[],
        supporting_paper_ids=[],
        status="VALIDATED",
        generation_model="qwen3:8b",
        validation_model="qwen3:8b",
        embedding_model="nomic-embed-text",
    )
    session.add(rec_unbridged)

    session.commit()

    def override_get_db():
        db = SessionMaker()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)

    yield {
        "client": client,
        "session_maker": SessionMaker,
        "p_baseline": p_baseline,
        "p_later": p_later,
        "p_undated": p_undated,
        "rec_bridged": rec_bridged,
        "rec_unbridged": rec_unbridged,
        "lifecycle_opp": lifecycle_opp,
        "gap": gap,
        "stmt1": stmt1,
    }

    app.dependency_overrides.clear()


def test_get_run_metadata_endpoint(test_db):
    """Verify GET /api/v1/run-metadata returns fingerprint, temporal cutoffs, and metrics."""
    client = test_db["client"]
    res = client.get("/api/v1/run-metadata")
    assert res.status_code == 200
    data = res.json()

    assert data["baseline_cutoff"] == "2020-12-31"
    assert data["evaluation_cutoffs"] == ["2023-01-31", "2023-05-31", "2023-10-31"]
    assert data["stage_5_enabled"] is True
    assert data["run_fingerprint"]["llm_model"] == "qwen3:8b"
    assert data["relevance_screening"]["selected_count"] == 4
    assert data["total_papers"] == 3
    assert data["total_statements"] == 1
    assert data["total_gaps"] == 1
    assert data["total_opportunities"] == 2
    assert "counts" in data
    assert data["counts"]["papers"] == 3
    assert data["counts"]["baseline_papers"] == 1
    assert data["counts"]["later_evaluation_papers"] == 1
    assert data["counts"]["statements"] == 1
    assert data["counts"]["research_gaps"] == 1
    assert data["counts"]["opportunities"] == 2
    assert data["counts"]["lifecycle_opportunities"] == 1
    assert data["counts"]["evidence_records"] == 1
    assert data["counts"]["history_records"] == 0
    assert "Stage 5 measures descriptive temporal evolution counts" in data["metric_interpretation_note"]


def test_list_opportunities_exposes_lifecycle_mappings(test_db):
    """Verify GET /api/v1/opportunities returns lifecycle_status, lifecycle_opportunity_id, and evidence_count."""
    client = test_db["client"]
    bridged_id = str(test_db["rec_bridged"].id)
    unbridged_id = str(test_db["rec_unbridged"].id)

    res = client.get("/api/v1/opportunities")
    assert res.status_code == 200
    opps = res.json()
    assert len(opps) == 2

    opp_map = {o["id"]: o for o in opps}

    # Bridged opportunity has lifecycle fields populated
    bridged = opp_map[bridged_id]
    assert bridged["lifecycle_opportunity_id"] == str(test_db["lifecycle_opp"].id)
    assert bridged["lifecycle_status"] == "OPEN"
    assert bridged["lifecycle_confidence"] == 0.88
    assert bridged["evidence_count"] == 1

    # Unbridged opportunity has None for lifecycle fields and evidence_count 0
    unbridged = opp_map[unbridged_id]
    assert unbridged["lifecycle_opportunity_id"] is None
    assert unbridged["lifecycle_status"] is None
    assert unbridged["evidence_count"] == 0


def test_get_opportunity_evidence_exposes_temporal_fields(test_db):
    """Verify GET /api/v1/opportunities/{id}/evidence exposes paper_publication_date, batch_id, and cutoff."""
    client = test_db["client"]
    opp_id = str(test_db["rec_bridged"].id)

    res = client.get(f"/api/v1/opportunities/{opp_id}/evidence")
    assert res.status_code == 200
    evs = res.json()
    assert len(evs) == 1

    ev = evs[0]
    assert ev["relationship"] == "EXTENDS"
    assert ev["paper_publication_date"] == "2023-10-17"
    assert ev["evaluation_cutoff"] == "2023-10-31"
    assert ev["batch_id"] == "batch_2_2023-10-31"
    assert "SELF-RAG trains an arbitrary LM" in ev["supporting_text"]


def test_get_papers_exposes_temporal_role(test_db):
    """Verify GET /api/v1/papers classifies baseline vs later-evaluation papers using temporal_cutoff."""
    client = test_db["client"]
    res = client.get("/api/v1/papers")
    assert res.status_code == 200
    papers = res.json()
    assert len(papers) == 3

    paper_map = {p["source_id"]: p for p in papers}

    # 2020-05-22 <= 2020-12-31 -> BASELINE
    p_base = paper_map["lewis2020"]
    assert p_base["temporal_role"] == "BASELINE"
    assert p_base["is_baseline"] is True

    # 2023-10-17 > 2020-12-31 -> LATER_EVALUATION
    p_later = paper_map["asai2023"]
    assert p_later["temporal_role"] == "LATER_EVALUATION"
    assert p_later["is_baseline"] is False

    # Undated paper -> UNDATED
    p_und = paper_map["undated_paper"]
    assert p_und["temporal_role"] == "UNDATED"
    assert p_und["is_baseline"] is None


def test_get_statements_exposes_passage_text(test_db):
    """Verify GET /api/v1/statements includes source passage_text."""
    client = test_db["client"]
    res = client.get("/api/v1/statements")
    assert res.status_code == 200
    stmts = res.json()
    assert len(stmts) == 1
    s = stmts[0]
    assert s["passage_text"] == "RAG models combine parametric and non-parametric memory."


def test_get_endpoints_do_not_write_or_flush_to_db(test_db):
    """Verify GET operations on unbridged opportunities do NOT insert or flush Opportunity rows."""
    client = test_db["client"]
    SessionMaker = test_db["session_maker"]
    unbridged_id = str(test_db["rec_unbridged"].id)

    with SessionMaker() as db:
        initial_opp_count = db.query(Opportunity).count()
        assert initial_opp_count == 1  # only lifecycle_opp exists

    # Calling GET /history on unbridged opportunity
    res_hist = client.get(f"/api/v1/opportunities/{unbridged_id}/history")
    assert res_hist.status_code == 200
    assert res_hist.json() == []

    # Calling GET /evidence on unbridged opportunity
    res_ev = client.get(f"/api/v1/opportunities/{unbridged_id}/evidence")
    assert res_ev.status_code == 200
    assert res_ev.json() == []

    # Verify no new Opportunity rows were inserted or flushed
    with SessionMaker() as db:
        final_opp_count = db.query(Opportunity).count()
        assert final_opp_count == initial_opp_count


def test_generate_opportunity_does_not_fabricate_fallback_on_failure(test_db):
    """Verify POST /api/v1/gaps/{id}/opportunity returns HTTP 500 when pipeline fails, with NO fabricated fallback."""
    client = test_db["client"]
    gap_id = str(test_db["gap"].id)
    SessionMaker = test_db["session_maker"]

    with SessionMaker() as db:
        opp_count_before = db.query(ResearchOpportunityRecord).count()

    from ideagen.generation.pipeline import ResearchOpportunityPipeline

    with patch.object(
        ResearchOpportunityPipeline,
        "process_validated_gap",
        side_effect=RuntimeError("Simulated LLM pipeline failure"),
    ):
        res = client.post(f"/api/v1/gaps/{gap_id}/opportunity")
        assert res.status_code == 500
        assert "Opportunity generation failed" in res.json()["detail"]

    # Verify NO fabricated record was persisted
    with SessionMaker() as db:
        opp_count_after = db.query(ResearchOpportunityRecord).count()
        assert opp_count_after == opp_count_before


def test_report_returns_null_for_unrecorded_assessments_instead_of_placeholders(test_db):
    """Only values stored on the opportunity are reported; nothing is filled with invented
    numbers or template text."""
    client = test_db["client"]
    rec = test_db["rec_bridged"]  # stores feasibility_score=0.85 and a novelty assessment only
    res = client.get(f"/api/v1/opportunities/{rec.id}/report")
    assert res.status_code == 200
    data = res.json()

    assert data["feasibility"]["feasibility_score"] == 0.85
    for key in ("required_data", "required_compute", "implementation_complexity", "experimental_setup", "summary"):
        assert data["feasibility"][key] is None
    assert data["novelty_assessment"]["assessment"] == "Potentially novel within the analyzed corpus"
    assert data["novelty_assessment"]["confidence"] is None
    assert data["prior_art_assessment"]["confidence"] is None
    assert data["prior_art_assessment"]["reasoning"] is None
    assert data["differentiation"] is None
    # The only supporting statement is a LIMITATION: no method statements -> no approaches.
    assert data["existing_approaches"] == []
    assert data["confidence"] == 0.88


def test_report_why_insufficient_is_null_without_limitation_evidence(test_db):
    client = test_db["client"]
    rec = test_db["rec_unbridged"]  # no supporting statements, no gap limitations
    data = client.get(f"/api/v1/opportunities/{rec.id}/report").json()
    assert data["why_existing_work_is_insufficient"] is None
    assert data["feasibility"]["feasibility_score"] == 0.8  # stored value, passed through


def test_run_metadata_database_status_is_observed_with_dialect(test_db):
    data = test_db["client"].get("/api/v1/run-metadata").json()
    assert data["database_status"] == "CONNECTED (sqlite)"
