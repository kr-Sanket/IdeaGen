"""Tests for Non-Destructive Research-Gap Refinement API (POST /api/v1/gaps/{gap_id}/refine).

Verifies safe researcher-facing workflow:
- Successful child candidate creation and parent linkage.
- Parent preservation and evidence provenance copying.
- Rejection of invalid parent statuses (REJECTED, VALIDATED, 404 missing).
- Rejection of missing, whitespace, or malformed fields (422).
- Idempotent duplicate submission prevention.
- Child status initialized to CANDIDATE with needs_refinement=True.
- Opportunity generation remaining strictly blocked for child candidate.
- Cross-project isolation and no evidence leakage.
"""

import uuid
import datetime
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from ideagen.db.models import (
    Base,
    Paper,
    PaperDocument,
    PaperPassage,
    ResearchKnowledge,
    ResearchStatement,
    ResearchGap,
    GapEvidence,
    ResearchProjectRecord,
)
from ideagen.db import get_db
from ideagen.api.main import app


@pytest.fixture
def test_setup():
    """Sets up an isolated, in-memory SQLite database with seed data."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    SessionMaker = sessionmaker(bind=engine)
    session = SessionMaker()

    proj_id = "proj_test_refinement"
    foreign_proj_id = "proj_foreign_123"

    p1 = ResearchProjectRecord(id=proj_id, name="Solar Microgrid Net Load", description="Testing Net Load Refinement")
    p2 = ResearchProjectRecord(id=foreign_proj_id, name="Foreign Project", description="Isolated Corpus")
    session.add_all([p1, p2])

    paper_id = uuid.uuid4()
    doc_id = uuid.uuid4()
    passage_id = uuid.uuid4()
    stmt_id = uuid.uuid4()

    paper = Paper(
        id=paper_id,
        source_type="MANUAL_UPLOAD",
        source_id="solar_paper_01",
        title="Direct Net-Load Forecasting for Microgrids",
        publication_date=datetime.date(2025, 6, 1),
        batch_id="batch_solar",
        abstract="Forecasting net-load under aggregated distribution systems.",
        authors=["Researcher A", "Researcher B"],
        referenced_works=[],
        raw_metadata={},
    )
    session.add(paper)

    doc = PaperDocument(
        id=doc_id,
        paper_id=paper_id,
        full_text="Passage text discussing direct net-load forecasting constraints.",
        extraction_method="PyMuPDF",
        extraction_version="v1.0",
        extracted_at=datetime.datetime.now(datetime.timezone.utc),
    )
    session.add(doc)

    passage = PaperPassage(
        id=passage_id,
        paper_id=paper_id,
        document_id=doc_id,
        passage_index=0,
        page_start=1,
        page_end=2,
        section="Introduction",
        text="STNLF performance degrades significantly under volatile residential PV.",
    )
    session.add(passage)

    knowledge = ResearchKnowledge(
        id=uuid.uuid4(),
        paper_id=paper_id,
        research_problem="Short-term net-load forecasting volatility",
        methodology="Direct STNLF neural network",
        model_name="qwen2.5:7b",
        extracted_at=datetime.datetime.now(datetime.timezone.utc),
    )
    session.add(knowledge)

    stmt = ResearchStatement(
        id=stmt_id,
        knowledge_id=knowledge.id,
        paper_id=paper_id,
        passage_id=passage_id,
        statement_text="STNLF performance degrades significantly under volatile residential PV.",
        statement_type="LIMITATION",
        section="Introduction",
        confidence=0.9,
    )
    session.add(stmt)

    # 1. Eligible parent candidate gap
    candidate_gap_id = uuid.uuid4()
    candidate_gap = ResearchGap(
        id=candidate_gap_id,
        project_id=proj_id,
        gap_statement="STNLF methods require evaluation under distributed rooftop PV.",
        research_problem="Transferability of STNLF to residential microgrids",
        status="CANDIDATE",
        evidence_count=1,
        limitations=["Aggregated feeder smoothing assumption"],
        future_work=["Validate transferability to individual residential nodes"],
        validation_reasoning="Single passage introductory limitation flagged for refinement",
        validation_confidence=None,
        metadata_json={"needs_refinement": True, "single_passage_evidence": True},
    )
    session.add(candidate_gap)

    evidence = GapEvidence(
        id=uuid.uuid4(),
        gap_id=candidate_gap_id,
        statement_id=stmt_id,
        paper_id=paper_id,
        evidence_role="LIMITATION",
        metadata_json={"source": "extraction"},
    )
    session.add(evidence)

    # 2. Rejected gap (ineligible)
    rejected_gap_id = uuid.uuid4()
    rejected_gap = ResearchGap(
        id=rejected_gap_id,
        project_id=proj_id,
        gap_statement="Net load is defined as gross load minus behind-the-meter generation.",
        research_problem="Net load background definition",
        status="REJECTED",
        evidence_count=1,
        validation_reasoning="Background definition or established taxonomy",
        metadata_json={"category": "BACKGROUND_MOTIVATION"},
    )
    session.add(rejected_gap)

    # 3. Validated gap (ineligible for candidate refinement workflow)
    validated_gap_id = uuid.uuid4()
    validated_gap = ResearchGap(
        id=validated_gap_id,
        project_id=proj_id,
        gap_statement="Unquantified compound forecasting error in multi-stage inverters.",
        research_problem="Multi-stage inverter error propagation",
        status="VALIDATED",
        evidence_count=2,
        validation_reasoning="Cross-paper evidence confirms technical limitation",
        metadata_json={"category": "EVIDENCE_GROUNDED_GAP"},
    )
    session.add(validated_gap)

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
        "proj_id": proj_id,
        "foreign_proj_id": foreign_proj_id,
        "candidate_gap_id": candidate_gap_id,
        "rejected_gap_id": rejected_gap_id,
        "validated_gap_id": validated_gap_id,
        "paper_id": paper_id,
        "stmt_id": stmt_id,
    }

    app.dependency_overrides.clear()


def test_successful_child_creation_and_parent_linkage(test_setup):
    client = test_setup["client"]
    proj_id = test_setup["proj_id"]
    parent_id = test_setup["candidate_gap_id"]

    payload = {
        "refined_statement": "To what extent do direct short-term net-load forecasting methods developed on aggregated microgrids transfer to residential microgrids with volatile behind-the-meter rooftop PV and household demand?",
        "refined_problem": "Evaluating domain and scale transferability of short-term net-load forecasting algorithms from aggregated distribution networks to residential microgrids with unobserved behind-the-meter PV generation.",
        "refinement_rationale": "Utility-scale models assume aggregate smoothing across hundreds of feeders. Testing transferability to individual residential nodes with volatile rooftop generation addresses the unverified scale assumption without asserting unproven performance degradation as an established fact.",
        "why_insufficient": "Prior work benchmarked only aggregated substation feeders.",
    }

    resp = client.post(f"/api/v1/gaps/{parent_id}/refine?project_id={proj_id}", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()

    child_id = data["id"]
    assert child_id != str(parent_id)
    assert data["gap_statement"] == payload["refined_statement"]
    assert data["research_problem"] == payload["refined_problem"]
    assert data["status"] == "CANDIDATE"
    assert data["evidence_count"] == 1
    assert str(test_setup["stmt_id"]) in [str(s) for s in data["supporting_statement_ids"]]
    assert str(test_setup["paper_id"]) in [str(p) for p in data["supporting_paper_ids"]]

    # Lineage metadata
    metadata = data["metadata"]
    assert metadata["parent_gap_id"] == str(parent_id)
    assert metadata["refinement_type"] == "RESEARCHER_REFINED"
    assert metadata["refinement_rationale"] == payload["refinement_rationale"]
    assert metadata["why_insufficient"] == payload["why_insufficient"]
    assert metadata["needs_refinement"] is True
    assert "provenance" in metadata
    assert metadata["provenance"]["parent_gap_id"] == str(parent_id)
    assert metadata["provenance"]["source"] == "RESEARCHER_REFINED"

    # Verify parent gap updated with child reference
    parent_resp = client.get(f"/api/v1/gaps/{parent_id}?project_id={proj_id}")
    assert parent_resp.status_code == 200
    parent_data = parent_resp.json()
    assert child_id in parent_data["metadata"].get("child_gap_ids", [])


def test_parent_preservation_and_evidence_provenance(test_setup):
    client = test_setup["client"]
    session_maker = test_setup["session_maker"]
    proj_id = test_setup["proj_id"]
    parent_id = test_setup["candidate_gap_id"]

    # Record parent state before
    with session_maker() as session:
        parent_before = session.query(ResearchGap).filter(ResearchGap.id == parent_id).first()
        before_stmt = parent_before.gap_statement
        before_prob = parent_before.research_problem
        before_status = parent_before.status
        before_reasoning = parent_before.validation_reasoning

    payload = {
        "refined_statement": "Refined test statement with at least ten characters.",
        "refined_problem": "Refined problem formulation with at least ten characters.",
        "refinement_rationale": "Clear rationale for researcher refinement with at least ten characters.",
    }

    resp = client.post(f"/api/v1/gaps/{parent_id}/refine?project_id={proj_id}", json=payload)
    assert resp.status_code == 200
    child_id = uuid.UUID(resp.json()["id"])

    # Verify parent in DB is preserved intact
    with session_maker() as session:
        parent_after = session.query(ResearchGap).filter(ResearchGap.id == parent_id).first()
        assert parent_after.gap_statement == before_stmt
        assert parent_after.research_problem == before_prob
        assert parent_after.status == before_status
        assert parent_after.validation_reasoning == before_reasoning

        # Verify child evidence was copied from parent without fabrication
        child_evidence = session.query(GapEvidence).filter(GapEvidence.gap_id == child_id).all()
        assert len(child_evidence) == 1
        ce = child_evidence[0]
        assert ce.statement_id == test_setup["stmt_id"]
        assert ce.paper_id == test_setup["paper_id"]
        assert ce.evidence_role == "LIMITATION"
        assert ce.metadata_json.get("inherited_from_parent_gap_id") == str(parent_id)
        assert ce.metadata_json.get("independent_corroboration") is False


def test_rejection_of_invalid_parent_statuses(test_setup):
    client = test_setup["client"]
    proj_id = test_setup["proj_id"]
    payload = {
        "refined_statement": "Valid statement with sufficient length.",
        "refined_problem": "Valid problem with sufficient length.",
        "refinement_rationale": "Valid rationale with sufficient length.",
    }

    # 1. Non-existent parent gap -> 404
    non_existent = uuid.uuid4()
    resp_404 = client.post(f"/api/v1/gaps/{non_existent}/refine?project_id={proj_id}", json=payload)
    assert resp_404.status_code == 404
    assert "not found" in resp_404.json()["detail"].lower()

    # 2. REJECTED gap -> 400
    rejected_id = test_setup["rejected_gap_id"]
    resp_rej = client.post(f"/api/v1/gaps/{rejected_id}/refine?project_id={proj_id}", json=payload)
    assert resp_rej.status_code == 400
    assert "cannot refine a rejected" in resp_rej.json()["detail"].lower()

    # 3. VALIDATED gap -> 400
    validated_id = test_setup["validated_gap_id"]
    resp_val = client.post(f"/api/v1/gaps/{validated_id}/refine?project_id={proj_id}", json=payload)
    assert resp_val.status_code == 400
    assert "only candidate research gaps are eligible" in resp_val.json()["detail"].lower()


def test_missing_or_malformed_fields(test_setup):
    client = test_setup["client"]
    proj_id = test_setup["proj_id"]
    parent_id = test_setup["candidate_gap_id"]

    # 1. Missing refined_statement -> 422
    resp1 = client.post(
        f"/api/v1/gaps/{parent_id}/refine?project_id={proj_id}",
        json={
            "refined_problem": "Valid problem domain explanation.",
            "refinement_rationale": "Valid rationale for refining candidate.",
        },
    )
    assert resp1.status_code == 422

    # 2. Whitespace-only statement -> 422
    resp2 = client.post(
        f"/api/v1/gaps/{parent_id}/refine?project_id={proj_id}",
        json={
            "refined_statement": "          ",
            "refined_problem": "Valid problem domain explanation.",
            "refinement_rationale": "Valid rationale for refining candidate.",
        },
    )
    assert resp2.status_code == 422

    # 3. Too short statement (< 10 chars) -> 422
    resp3 = client.post(
        f"/api/v1/gaps/{parent_id}/refine?project_id={proj_id}",
        json={
            "refined_statement": "Short",
            "refined_problem": "Valid problem domain explanation.",
            "refinement_rationale": "Valid rationale for refining candidate.",
        },
    )
    assert resp3.status_code == 422


def test_duplicate_submission_behavior(test_setup):
    client = test_setup["client"]
    session_maker = test_setup["session_maker"]
    proj_id = test_setup["proj_id"]
    parent_id = test_setup["candidate_gap_id"]

    payload = {
        "refined_statement": "Testing idempotent duplicate submission for refinement.",
        "refined_problem": "Testing underlying research problem for duplicate detection.",
        "refinement_rationale": "Ensuring safe repeated requests do not create duplicate records.",
    }

    # Initial submission
    resp1 = client.post(f"/api/v1/gaps/{parent_id}/refine?project_id={proj_id}", json=payload)
    assert resp1.status_code == 200
    first_child_id = resp1.json()["id"]

    with session_maker() as session:
        initial_count = session.query(ResearchGap).count()

    # Repeated identical submission
    resp2 = client.post(f"/api/v1/gaps/{parent_id}/refine?project_id={proj_id}", json=payload)
    assert resp2.status_code == 200
    second_child_id = resp2.json()["id"]

    # Same child gap returned, preventing duplicate child proliferation
    assert first_child_id == second_child_id

    with session_maker() as session:
        final_count = session.query(ResearchGap).count()
        assert final_count == initial_count


def test_child_status_remaining_candidate_and_opportunity_blocked(test_setup):
    client = test_setup["client"]
    proj_id = test_setup["proj_id"]
    parent_id = test_setup["candidate_gap_id"]

    payload = {
        "refined_statement": "Child candidate gap that must not bypass the validator.",
        "refined_problem": "Problem domain for unvalidated child candidate.",
        "refinement_rationale": "Scientific safeguard: child candidate must remain CANDIDATE.",
    }

    resp = client.post(f"/api/v1/gaps/{parent_id}/refine?project_id={proj_id}", json=payload)
    assert resp.status_code == 200
    child_id = resp.json()["id"]

    # Verify status is CANDIDATE
    assert resp.json()["status"] == "CANDIDATE"

    # Attempt opportunity generation from this child candidate
    opp_resp = client.post(f"/api/v1/gaps/{child_id}/opportunity?project_id={proj_id}")
    assert opp_resp.status_code == 400
    assert "must be validated before opportunity generation" in opp_resp.json()["detail"].lower()


def test_no_cross_project_evidence_leakage(test_setup):
    client = test_setup["client"]
    parent_id = test_setup["candidate_gap_id"]
    foreign_proj_id = test_setup["foreign_proj_id"]

    payload = {
        "refined_statement": "Refinement across project boundary must fail.",
        "refined_problem": "Problem description across project boundary.",
        "refinement_rationale": "Testing project scoping protection.",
    }

    # Attempting to refine parent gap using a different project ID must return 404
    resp = client.post(f"/api/v1/gaps/{parent_id}/refine?project_id={foreign_proj_id}", json=payload)
    assert resp.status_code == 404
    assert "not found in project" in resp.json()["detail"].lower()
