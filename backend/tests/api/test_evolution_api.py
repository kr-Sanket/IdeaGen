"""Tests for Opportunity Evolution REST Endpoints (Checkpoint 10).

Verifies:
- Evolve a bridged opportunity using mocked classification.
- Reject unknown opportunity/paper IDs with appropriate status codes (404, 400).
- Enforce temporal cutoff, including undated papers.
- Duplicate requests do not create duplicate evidence or history.
- History endpoint returns chronological records with trigger reason and causal evidence ID.
- Evidence endpoint returns persisted records with verification status.
- Compatibility with both /api/v1/ and /api/ route prefixes.
- Existing API contracts remain intact.
"""

from unittest.mock import patch
import pytest
import uuid
import datetime
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from ideagen.db.models import (
    Base,
    Paper,
    PaperDocument,
    PaperPassage,
    Opportunity,
    OpportunityEvidence,
    OpportunityHistory,
    ResearchOpportunityRecord,
    ResearchGap,
)
from ideagen.db import get_db
from ideagen.api.main import app, get_relationship_classifier
from ideagen.opportunity.models import OpportunityRelationship, OpportunityStatus
from ideagen.opportunity.classifier import (
    RelationshipClassifier,
    RelationshipClassification,
)


@pytest.fixture
def api_test_env():
    """Isolated in-memory SQLite database and TestClient for evolution API tests."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    SessionMaker = sessionmaker(bind=engine)
    session = SessionMaker()

    # Create dummy Gap and ResearchOpportunityRecord (bridged)
    gap_id = uuid.uuid4()
    gap = ResearchGap(
        id=gap_id,
        gap_statement="Unaddressed constraint propagation bottleneck.",
        research_problem="Error accumulation in multi-hop models",
        status="VALIDATED",
        validation_confidence=0.85,
    )
    session.add(gap)

    opp_record_id = uuid.uuid4()
    opp_record = ResearchOpportunityRecord(
        id=opp_record_id,
        gap_id=gap_id,
        research_problem="Error accumulation in multi-hop models",
        identified_gap="Unaddressed constraint propagation bottleneck",
        proposed_research_direction="Investigate certified step-wise verification",
        reasoning="Soundness proofs for constraint verification",
        technical_approach="Formal inductive invariants",
        expected_contribution="Provable reduction in false inferences",
        feasibility={"required_data": "HotpotQA", "required_compute": "1x GPU", "implementation_complexity": "Moderate", "experimental_setup": "Benchmarking", "feasibility_score": 0.85, "summary": "Feasible"},
        novelty_assessment={"assessment": "Potentially novel within the analyzed corpus", "reasoning": "Distinct focus", "differentiation": "Step-wise constraint focus", "confidence": 0.85},
        confidence=0.9,
        status="VALIDATED",
        generation_model="gemini-2.5-flash",
        validation_model="gemini-2.5-flash",
        embedding_model="gemini-embedding-001",
        pipeline_version="v1.0",
        # Historical knowledge date (as the runner backdates to the baseline cutoff);
        # evolution evidence must be published on or after it.
        created_at=datetime.datetime(2023, 1, 1, tzinfo=datetime.timezone.utc),
    )
    session.add(opp_record)

    # Create native Opportunity
    native_opp_id = uuid.uuid4()
    native_opp = Opportunity(
        id=native_opp_id,
        title="Native Test Opportunity",
        problem_statement="Native problem statement",
        description="Native opportunity description",
        status=OpportunityStatus.OPEN.value,
        confidence=0.8,
        created_at=datetime.datetime(2023, 1, 1, tzinfo=datetime.timezone.utc),
    )
    session.add(native_opp)

    # Create Papers with passages
    p1_id = uuid.uuid4()
    p1 = Paper(
        id=p1_id,
        source_type="MANUAL_UPLOAD",
        source_id="p1_src",
        title="Paper Alpha On Alignment",
        publication_date=datetime.date(2023, 5, 10),
        batch_id="batch1",
        abstract="Abstract of Paper Alpha",
        authors=["Author One"],
        referenced_works=[],
        raw_metadata={},
    )
    session.add(p1)

    doc1_id = uuid.uuid4()
    doc1 = PaperDocument(
        id=doc1_id,
        paper_id=p1_id,
        full_text="Paper 1 full text",
        extraction_method="pdf",
        extraction_version="v1",
        extracted_at=datetime.datetime.now(datetime.timezone.utc),
    )
    session.add(doc1)

    psg1 = PaperPassage(
        id=uuid.uuid4(),
        paper_id=p1_id,
        document_id=doc1_id,
        passage_index=0,
        page_start=1,
        page_end=1,
        section="RESULTS",
        text="We present certified verification that directly addresses constraint propagation errors.",
    )
    session.add(psg1)

    p2_id = uuid.uuid4()
    p2 = Paper(
        id=p2_id,
        source_type="MANUAL_UPLOAD",
        source_id="p2_src",
        title="Paper Beta On Alignment",
        publication_date=datetime.date(2023, 8, 15),
        batch_id="batch1",
        abstract="Abstract of Paper Beta",
        authors=["Author Two"],
        referenced_works=[],
        raw_metadata={},
    )
    session.add(p2)

    doc2_id = uuid.uuid4()
    doc2 = PaperDocument(
        id=doc2_id,
        paper_id=p2_id,
        full_text="Paper 2 full text",
        extraction_method="pdf",
        extraction_version="v1",
        extracted_at=datetime.datetime.now(datetime.timezone.utc),
    )
    session.add(doc2)

    psg2 = PaperPassage(
        id=uuid.uuid4(),
        paper_id=p2_id,
        document_id=doc2_id,
        passage_index=0,
        page_start=1,
        page_end=1,
        section="RESULTS",
        text="We independently verify and address constraint propagation using inductive proof checkers.",
    )
    session.add(psg2)

    # Future paper (after cutoff 2023-06-01)
    p_future_id = uuid.uuid4()
    p_future = Paper(
        id=p_future_id,
        source_type="MANUAL_UPLOAD",
        source_id="p_future_src",
        title="Future Paper",
        publication_date=datetime.date(2024, 1, 1),
        batch_id="batch1",
        abstract="Future abstract",
        authors=["Author Future"],
        referenced_works=[],
        raw_metadata={},
    )
    session.add(p_future)

    # Undated paper
    p_undated_id = uuid.uuid4()
    p_undated = Paper(
        id=p_undated_id,
        source_type="MANUAL_UPLOAD",
        source_id="p_undated_src",
        title="Undated Paper",
        publication_date=None,
        batch_id="batch1",
        abstract="Undated abstract",
        authors=["Author Undated"],
        referenced_works=[],
        raw_metadata={},
    )
    session.add(p_undated)

    session.commit()

    def override_get_db():
        db = SessionMaker()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)

    meta = {
        "gap_id": gap_id,
        "opp_record_id": opp_record_id,
        "native_opp_id": native_opp_id,
        "p1_id": p1_id,
        "p2_id": p2_id,
        "p_future_id": p_future_id,
        "p_undated_id": p_undated_id,
    }

    yield client, meta, session

    app.dependency_overrides.clear()
    session.close()


def _mock_classifier_response(client_app, responses_by_title: dict):
    class MockProvider:
        def generate_structured(self, prompt, schema):
            for title, classification in responses_by_title.items():
                if title in prompt:
                    return classification
            raise ValueError(f"No mock response found for prompt: {prompt[:80]}...")

    classifier = RelationshipClassifier(provider=MockProvider())
    client_app.dependency_overrides[get_relationship_classifier] = lambda: classifier


# ── Test Cases ─────────────────────────────────────────────────────────────


def test_evolve_bridged_opportunity_single_addresses(api_test_env):
    """Evolve a bridged ResearchOpportunityRecord to PARTIALLY_ADDRESSED via single ADDRESSES paper."""
    client, meta, _ = api_test_env

    _mock_classifier_response(app, {
        "Paper Alpha": RelationshipClassification(
            relationship=OpportunityRelationship.ADDRESSES,
            confidence=0.92,
            rationale="Directly addresses constraint propagation.",
            evidence_excerpt="directly addresses constraint propagation errors",
            prompt_version="1.0.0",
        )
    })

    # Call evolve using the bridged ResearchOpportunityRecord ID
    resp = client.post(
        f"/api/v1/opportunities/{meta['opp_record_id']}/evolve",
        json={"paper_ids": [str(meta["p1_id"])]},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert data["initial_status"] == "OPEN"
    assert data["final_status"] == "PARTIALLY_ADDRESSED"
    assert data["processed_count"] == 1
    assert data["skipped_count"] == 0
    assert data["failed_count"] == 0
    assert len(data["events"]) == 1
    assert data["events"][0]["relationship"] == "ADDRESSES"
    assert data["events"][0]["new_status"] == "PARTIALLY_ADDRESSED"
    assert data["events"][0]["state_changed"] is True


def test_evolve_to_resolved_via_two_distinct_papers(api_test_env):
    """Two distinct ADDRESSES papers transition opportunity to RESOLVED."""
    client, meta, _ = api_test_env

    _mock_classifier_response(app, {
        "Paper Alpha": RelationshipClassification(
            relationship=OpportunityRelationship.ADDRESSES,
            confidence=0.92,
            rationale="Addresses constraint propagation.",
            evidence_excerpt="directly addresses constraint propagation errors",
            prompt_version="1.0.0",
        ),
        "Paper Beta": RelationshipClassification(
            relationship=OpportunityRelationship.ADDRESSES,
            confidence=0.95,
            rationale="Independently verifies and addresses constraint propagation.",
            evidence_excerpt="independently verify and address constraint propagation",
            prompt_version="1.0.0",
        ),
    })

    # Evolve with Paper 1 -> PARTIALLY_ADDRESSED
    resp1 = client.post(
        f"/api/v1/opportunities/{meta['native_opp_id']}/evolve",
        json={"paper_ids": [str(meta["p1_id"])]},
    )
    assert resp1.status_code == 200
    assert resp1.json()["final_status"] == "PARTIALLY_ADDRESSED"

    # Evolve with Paper 2 -> RESOLVED
    resp2 = client.post(
        f"/api/v1/opportunities/{meta['native_opp_id']}/evolve",
        json={"paper_ids": [str(meta["p2_id"])]},
    )
    assert resp2.status_code == 200
    assert resp2.json()["final_status"] == "RESOLVED"
    assert resp2.json()["processed_count"] == 1


def test_duplicate_requests_do_not_reprocess_or_duplicate_history(api_test_env):
    """Replaying the same paper ID is safely skipped and does not inflate evidence/history."""
    client, meta, _ = api_test_env

    _mock_classifier_response(app, {
        "Paper Alpha": RelationshipClassification(
            relationship=OpportunityRelationship.ADDRESSES,
            confidence=0.92,
            rationale="Addresses constraint propagation.",
            evidence_excerpt="directly addresses constraint propagation errors",
            prompt_version="1.0.0",
        )
    })

    # First request
    resp1 = client.post(
        f"/api/v1/opportunities/{meta['native_opp_id']}/evolve",
        json={"paper_ids": [str(meta["p1_id"])]},
    )
    assert resp1.status_code == 200
    assert resp1.json()["processed_count"] == 1

    # Second duplicate request
    resp2 = client.post(
        f"/api/v1/opportunities/{meta['native_opp_id']}/evolve",
        json={"paper_ids": [str(meta["p1_id"])]},
    )
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["processed_count"] == 0
    assert data2["skipped_count"] == 1
    assert data2["final_status"] == "PARTIALLY_ADDRESSED"

    # Check history endpoint: exactly 1 history event
    hist_resp = client.get(f"/api/v1/opportunities/{meta['native_opp_id']}/history")
    assert hist_resp.status_code == 200
    assert len(hist_resp.json()) == 1


def test_reject_unknown_opportunity_and_paper_ids(api_test_env):
    """Endpoints return 404 for nonexistent opportunity or paper IDs."""
    client, meta, _ = api_test_env

    random_opp_id = uuid.uuid4()
    random_paper_id = uuid.uuid4()

    # Nonexistent opportunity
    resp1 = client.post(
        f"/api/v1/opportunities/{random_opp_id}/evolve",
        json={"paper_ids": [str(meta["p1_id"])]},
    )
    assert resp1.status_code == 404

    # Nonexistent paper
    resp2 = client.post(
        f"/api/v1/opportunities/{meta['native_opp_id']}/evolve",
        json={"paper_ids": [str(random_paper_id)]},
    )
    assert resp2.status_code == 404
    assert "Papers not found" in resp2.json()["detail"]

    # Empty paper_ids list
    resp3 = client.post(
        f"/api/v1/opportunities/{meta['native_opp_id']}/evolve",
        json={"paper_ids": []},
    )
    assert resp3.status_code == 422 or resp3.status_code == 400


def test_enforce_temporal_cutoff_and_undated_papers(api_test_env):
    """Cutoff excludes future and undated papers before classification."""
    client, meta, _ = api_test_env

    _mock_classifier_response(app, {
        "Paper Alpha": RelationshipClassification(
            relationship=OpportunityRelationship.ADDRESSES,
            confidence=0.9,
            rationale="Grounded.",
            evidence_excerpt="directly addresses constraint propagation errors",
            prompt_version="1.0.0",
        )
    })

    # Cutoff set to 2023-06-01:
    # - p1 (2023-05-10) is eligible (processed)
    # - p_future (2024-01-01) is after cutoff (skipped)
    # - p_undated (None) is undated (skipped)
    resp = client.post(
        f"/api/v1/opportunities/{meta['native_opp_id']}/evolve",
        json={
            "paper_ids": [
                str(meta["p1_id"]),
                str(meta["p_future_id"]),
                str(meta["p_undated_id"]),
            ],
            "cutoff_date": "2023-06-01",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["processed_count"] == 1
    assert data["skipped_count"] == 2
    assert data["final_status"] == "PARTIALLY_ADDRESSED"


def test_get_history_endpoint_chronological(api_test_env):
    """GET /history returns persisted lifecycle history in chronological order."""
    client, meta, _ = api_test_env

    _mock_classifier_response(app, {
        "Paper Alpha": RelationshipClassification(
            relationship=OpportunityRelationship.ADDRESSES,
            confidence=0.9,
            rationale="Addresses.",
            evidence_excerpt="directly addresses constraint propagation errors",
            prompt_version="1.0.0",
        ),
        "Paper Beta": RelationshipClassification(
            relationship=OpportunityRelationship.ADDRESSES,
            confidence=0.95,
            rationale="Addresses independently.",
            evidence_excerpt="independently verify and address constraint propagation",
            prompt_version="1.0.0",
        ),
    })

    client.post(f"/api/v1/opportunities/{meta['native_opp_id']}/evolve", json={"paper_ids": [str(meta["p1_id"])]})
    client.post(f"/api/v1/opportunities/{meta['native_opp_id']}/evolve", json={"paper_ids": [str(meta["p2_id"])]})

    resp = client.get(f"/api/v1/opportunities/{meta['native_opp_id']}/history")
    assert resp.status_code == 200
    history = resp.json()

    assert len(history) == 2
    # Chronological progression
    assert history[0]["previous_status"] == "OPEN"
    assert history[0]["new_status"] == "PARTIALLY_ADDRESSED"
    assert history[0]["trigger_type"] == "ADDRESSES"
    assert history[0]["evidence_id"] is not None

    assert history[1]["previous_status"] == "PARTIALLY_ADDRESSED"
    assert history[1]["new_status"] == "RESOLVED"
    assert history[1]["trigger_type"] == "ADDRESSES"
    assert history[1]["evidence_id"] is not None


def test_get_evidence_endpoint_verified_status(api_test_env):
    """GET /evidence returns persisted records and correctly indicates verification."""
    client, meta, _ = api_test_env

    _mock_classifier_response(app, {
        "Paper Alpha": RelationshipClassification(
            relationship=OpportunityRelationship.ADDRESSES,
            confidence=0.9,
            rationale="Addresses.",
            evidence_excerpt="directly addresses constraint propagation errors",
            prompt_version="1.0.0",
        )
    })

    client.post(f"/api/v1/opportunities/{meta['native_opp_id']}/evolve", json={"paper_ids": [str(meta["p1_id"])]})

    resp = client.get(f"/api/v1/opportunities/{meta['native_opp_id']}/evidence")
    assert resp.status_code == 200
    evidence_list = resp.json()

    assert len(evidence_list) == 1
    ev = evidence_list[0]
    assert ev["paper_id"] == str(meta["p1_id"])
    assert ev["paper_title"] == "Paper Alpha On Alignment"
    assert ev["relationship"] == "ADDRESSES"
    assert ev["verified"] is True
    assert "directly addresses constraint propagation errors" in ev["supporting_text"]


def test_endpoint_aliases_and_404_handling(api_test_env):
    """Endpoints work with both /api/v1/ and /api/ route prefixes."""
    client, meta, _ = api_test_env

    # /api/opportunities/{id}/history
    r1 = client.get(f"/api/opportunities/{meta['native_opp_id']}/history")
    assert r1.status_code == 200

    # /api/opportunities/{id}/evidence
    r2 = client.get(f"/api/opportunities/{meta['native_opp_id']}/evidence")
    assert r2.status_code == 200

    # Unknown ID on both
    unknown_id = uuid.uuid4()
    assert client.get(f"/api/v1/opportunities/{unknown_id}/history").status_code == 404
    assert client.get(f"/api/v1/opportunities/{unknown_id}/evidence").status_code == 404
    assert client.get(f"/api/opportunities/{unknown_id}/history").status_code == 404
    assert client.get(f"/api/opportunities/{unknown_id}/evidence").status_code == 404


def test_reject_opportunity_created_after_cutoff(api_test_env):
    """Evolution endpoint rejects an opportunity created strictly after the cutoff date."""
    client, meta, session = api_test_env

    # Create an opportunity created in 2024
    future_opp_id = uuid.uuid4()
    future_opp = Opportunity(
        id=future_opp_id,
        title="Future Opportunity 2024",
        problem_statement="Problem statement from 2024",
        description="Opportunity description from 2024",
        status=OpportunityStatus.OPEN.value,
        confidence=0.8,
        created_at=datetime.datetime(2024, 1, 15, tzinfo=datetime.timezone.utc),
    )
    session.add(future_opp)
    session.commit()

    # Call /evolve with cutoff_date in 2023 (before opportunity creation)
    resp = client.post(
        f"/api/v1/opportunities/{future_opp_id}/evolve",
        json={
            "paper_ids": [str(meta["p1_id"])],
            "cutoff_date": "2023-06-01",
        },
    )
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "after the requested cutoff date 2023-06-01" in detail


def test_accept_opportunity_created_on_cutoff(api_test_env):
    """Evolution endpoint accepts an opportunity created exactly on the cutoff date."""
    client, meta, session = api_test_env

    _mock_classifier_response(app, {
        "Paper Alpha": RelationshipClassification(
            relationship=OpportunityRelationship.ADDRESSES,
            confidence=0.9,
            rationale="Addresses on cutoff.",
            evidence_excerpt="directly addresses constraint propagation errors",
            prompt_version="1.0.0",
        )
    })

    # Opportunity created on 2023-06-01
    cutoff_opp_id = uuid.uuid4()
    cutoff_opp = Opportunity(
        id=cutoff_opp_id,
        title="Cutoff Boundary Opportunity",
        problem_statement="Problem statement on cutoff",
        description="Opportunity description on cutoff",
        status=OpportunityStatus.OPEN.value,
        confidence=0.8,
        created_at=datetime.datetime(2023, 6, 1, 12, 0, 0, tzinfo=datetime.timezone.utc),
    )
    session.add(cutoff_opp)
    session.commit()

    resp = client.post(
        f"/api/v1/opportunities/{cutoff_opp_id}/evolve",
        json={
            "paper_ids": [str(meta["p1_id"])],
            "cutoff_date": "2023-06-01",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    # The opportunity is accepted on the cutoff boundary, but Paper Alpha (2023-05-10) predates
    # the opportunity (2023-06-01), so it cannot be evolution evidence.
    assert data["processed_count"] == 0
    assert data["skipped_count"] == 1
    assert data["final_status"] == "OPEN"



def test_papers_published_before_opportunity_are_not_evolution_evidence(api_test_env):
    """Lower temporal bound: without any cutoff, a paper published before the opportunity's
    effective creation date is skipped and never sent to the classifier."""
    client, meta, session = api_test_env
    calls = []

    class _RecordingProvider:
        def generate_structured(self, prompt, response_schema):
            calls.append(prompt)
            raise AssertionError("classifier must not be called for pre-opportunity papers")

    app.dependency_overrides[get_relationship_classifier] = lambda: RelationshipClassifier(provider=_RecordingProvider())

    late_opp_id = uuid.uuid4()
    session.add(Opportunity(
        id=late_opp_id,
        title="Late Opportunity",
        problem_statement="Problem identified after Paper Alpha was published",
        description="Created 2023-07-01",
        status=OpportunityStatus.OPEN.value,
        confidence=0.8,
        created_at=datetime.datetime(2023, 7, 1, tzinfo=datetime.timezone.utc),
    ))
    session.commit()

    resp = client.post(
        f"/api/v1/opportunities/{late_opp_id}/evolve",
        json={"paper_ids": [str(meta["p1_id"]), str(meta["p_undated_id"])]},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["processed_count"] == 0
    assert data["skipped_count"] == 2
    assert data["events"] == []
    assert data["final_status"] == "OPEN"
    assert calls == []
