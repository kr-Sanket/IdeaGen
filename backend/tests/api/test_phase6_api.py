"""Tests for Phase 6 API Endpoints, Workspace Stats, and Opportunity Reports."""

import pytest
import uuid
import datetime
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, Index
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
    ResearchOpportunityRecord,
)
from ideagen.db import get_db
from ideagen.api.main import app


@pytest.fixture
def test_db():
    # Use StaticPool so all threads share the exact same in-memory SQLite database
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)

    SessionMaker = sessionmaker(bind=engine)
    session = SessionMaker()

    # Seed test data
    paper_id = uuid.uuid4()
    doc_id = uuid.uuid4()
    passage_id = uuid.uuid4()
    stmt_id = uuid.uuid4()
    gap_id = uuid.uuid4()
    opp_id = uuid.uuid4()

    paper = Paper(
        id=paper_id,
        source_type="MANUAL_UPLOAD",
        source_id="test_p1",
        title="Test Scientific Paper on RAG",
        publication_date=datetime.date(2025, 1, 1),
        batch_id="batch1",
        abstract="Test abstract about retrieval-augmented generation and hallucination mitigation.",
        authors=["Author A", "Author B"],
        referenced_works=[],
        raw_metadata={},
    )
    session.add(paper)

    doc = PaperDocument(
        id=doc_id,
        paper_id=paper_id,
        full_text="Full text of the test paper.",
        extraction_method="PyMuPDF",
        extraction_version="v1.0",
        extracted_at=datetime.datetime.utcnow(),
    )
    session.add(doc)

    passage = PaperPassage(
        id=passage_id,
        paper_id=paper_id,
        document_id=doc_id,
        passage_index=0,
        page_start=1,
        page_end=2,
        section="Limitations",
        text="Dense retrieval models suffer from unquantified hallucination accumulation across multi-hop reasoning steps.",
    )
    session.add(passage)

    knowledge = ResearchKnowledge(
        id=uuid.uuid4(),
        paper_id=paper_id,
        research_problem="Multi-hop reasoning latency and hallucination",
        methodology="Flat FAISS vector index retrieval",
        model_name="test-model",
        extracted_at=datetime.datetime.utcnow(),
    )
    session.add(knowledge)

    statement = ResearchStatement(
        id=stmt_id,
        knowledge_id=knowledge.id,
        paper_id=paper_id,
        passage_id=passage_id,
        statement_text="Dense retrieval models suffer from unquantified hallucination accumulation across multi-hop reasoning steps.",
        statement_type="LIMITATION",
        page_start=1,
        page_end=2,
        section="Limitations",
        confidence=0.9,
    )
    session.add(statement)

    gap = ResearchGap(
        id=gap_id,
        gap_statement="Unquantified hallucination accumulation in multi-hop RAG",
        research_problem="Multi-hop retrieval error propagation",
        status="VALIDATED",
        evidence_count=1,
        limitations=["Lack of step-wise verification"],
        future_work=["Develop dynamic pruning constraints"],
        validation_reasoning="Consistently identified across multiple papers",
        validation_confidence=0.88,
    )
    session.add(gap)

    evidence = GapEvidence(
        id=uuid.uuid4(),
        gap_id=gap_id,
        statement_id=stmt_id,
        paper_id=paper_id,
        evidence_role="SUPPORTING",
    )
    session.add(evidence)

    opp = ResearchOpportunityRecord(
        id=opp_id,
        gap_id=gap_id,
        research_problem="Multi-hop retrieval error propagation",
        identified_gap="Unquantified hallucination accumulation in multi-hop RAG",
        proposed_research_direction="Investigate step-wise verification constraints for multi-hop RAG",
        reasoning="Targeting unaddressed step-wise verification bottlenecks",
        technical_approach="Chain-of-thought verification with sub-query pruning",
        expected_contribution="Reduces hallucination rate by 35% in multi-hop benchmarks",
        feasibility={"required_data": "HotpotQA", "required_compute": "1x GPU", "implementation_complexity": "Moderate", "experimental_setup": "Benchmarking", "feasibility_score": 0.85, "summary": "Feasible"},
        novelty_assessment={"assessment": "Potentially novel within the analyzed corpus", "reasoning": "Distinct focus", "differentiation": "Step-wise constraint focus", "confidence": 0.85},
        confidence=0.88,
        supporting_statement_ids=[str(stmt_id)],
        supporting_paper_ids=[str(paper_id)],
        prior_art_statement_ids=[],
        status="VALIDATED",
        generation_model="gemini-2.5-flash",
        validation_model="gemini-2.5-flash",
        embedding_model="gemini-embedding-001",
        refinement_iteration_count=0,
        pipeline_version="v1.0",
    )
    session.add(opp)
    session.commit()

    def override_get_db():
        db = SessionMaker()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)

    yield client, {
        "paper_id": paper_id,
        "gap_id": gap_id,
        "opp_id": opp_id,
        "stmt_id": stmt_id,
    }

    app.dependency_overrides.clear()
    session.close()


def test_workspace_stats_endpoint(test_db):
    client, ids = test_db
    res = client.get("/api/v1/stats")
    assert res.status_code == 200
    data = res.json()
    assert data["total_papers"] == 1
    assert data["analyzed_papers"] == 1
    assert data["total_statements"] == 1
    assert data["total_gaps"] == 1
    assert data["validated_gaps"] == 1
    assert data["total_opportunities"] == 1
    assert data["validated_opportunities"] == 1


def test_list_papers_endpoint(test_db):
    client, ids = test_db
    res = client.get("/api/v1/papers")
    assert res.status_code == 200
    data = res.json()
    assert len(data) == 1
    assert data[0]["title"] == "Test Scientific Paper on RAG"
    assert data[0]["knowledge_status"] == "ANALYZED"


def test_get_paper_statements_endpoint(test_db):
    client, ids = test_db
    res = client.get(f"/api/papers/{ids['paper_id']}/statements")
    assert res.status_code == 200
    data = res.json()
    assert len(data) == 1
    assert data[0]["statement_type"] == "LIMITATION"
    assert "hallucination accumulation" in data[0]["statement_text"]


def test_list_gaps_endpoint(test_db):
    client, ids = test_db
    res = client.get("/api/v1/gaps?status=VALIDATED")
    assert res.status_code == 200
    data = res.json()
    assert len(data) == 1
    assert data[0]["status"] == "VALIDATED"


def test_list_opportunities_endpoint(test_db):
    client, ids = test_db
    res = client.get("/api/v1/opportunities")
    assert res.status_code == 200
    data = res.json()
    assert len(data) == 1
    assert data[0]["status"] == "VALIDATED"


def test_get_opportunity_report_endpoint(test_db):
    client, ids = test_db
    res = client.get(f"/api/v1/opportunities/{ids['opp_id']}/report")
    assert res.status_code == 200
    report = res.json()

    # Verify 16 required sections
    assert report["research_problem"] == "Multi-hop retrieval error propagation"
    assert report["identified_gap"] == "Unquantified hallucination accumulation in multi-hop RAG"
    assert "proposed_research_direction" in report
    assert "reasoning" in report
    assert "technical_approach" in report
    assert "expected_contribution" in report
    assert "feasibility" in report
    assert "prior_art_assessment" in report
    assert "differentiation" in report
    assert "supporting_papers" in report
    assert len(report["supporting_papers"]) == 1
    assert report["supporting_papers"][0]["title"] == "Test Scientific Paper on RAG"
    assert len(report["supporting_evidence"]) >= 1
    assert report["supporting_evidence"][0]["paper_title"] == "Test Scientific Paper on RAG"
    assert report["supporting_evidence"][0]["section"] == "Limitations"
    assert report["novelty_assessment"]["assessment"] == "Potentially novel within the analyzed corpus"


def test_get_evidence_detail_endpoint(test_db):
    client, ids = test_db
    res = client.get(f"/api/v1/evidence/{ids['stmt_id']}")
    assert res.status_code == 200
    evidence = res.json()
    assert evidence["paper_title"] == "Test Scientific Paper on RAG"
    assert evidence["section"] == "Limitations"
    assert evidence["page_start"] == 1
    assert evidence["page_end"] == 2
