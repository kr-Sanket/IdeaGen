from unittest.mock import patch, MagicMock
import pytest
import uuid
import datetime
import numpy as np
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
from ideagen.generation.pipeline import ResearchOpportunityPipeline


class FakeEmbeddingProvider:
    """Deterministic in-memory embedding provider for offline API testing."""

    def __init__(self, model_name: str = "fake-all-MiniLM-L6-v2"):
        self._model_name = model_name
        self.call_history: list[list[str]] = []

    @property
    def dimension(self) -> int:
        return 384

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        self.call_history.append(list(texts))
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        vecs = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for i in range(len(texts)):
            vecs[i, (i + 1) % self.dimension] = 1.0
        return vecs


from tests.opportunity.fake_provider import FakeLLMProvider


@pytest.fixture(autouse=True)
def mock_pipeline_dependencies():
    """Ensure API endpoints do not make network calls or require live LLM/embedding servers offline."""
    fake_provider = FakeEmbeddingProvider()
    valid_mock_response = {
        "proposed_research_direction": "Investigate dynamic context constraints for multi-hop RAG.",
        "reasoning": "Multi-hop RAG suffers from compounding retrieval error propagation.",
        "technical_approach": "Step-wise consistency filtering and selective context pruning.",
        "expected_contribution": "Bounded hallucination rate in multi-hop question answering.",
        "required_data": "Multi-hop QA benchmark corpora",
        "required_compute": "1x GPU workstation",
        "implementation_complexity": "Moderate",
        "experimental_setup": "Compare against standard 2-hop baseline models",
        "feasibility_score": 0.85,
        "confidence": 0.88,
        "assessment": "Potentially novel within the analyzed corpus",
        "differentiation": "Prior methods do not prune intermediate context errors.",
        "overlapping_statement_indices": [],
    }
    fake_llm = FakeLLMProvider(response=valid_mock_response)

    with patch(
        "ideagen.discovery.pipeline.get_default_embedding_provider",
        return_value=fake_provider,
    ), patch(
        "ideagen.discovery.embeddings.get_default_embedding_provider",
        return_value=fake_provider,
    ), patch(
        "ideagen.api.main.get_default_embedding_provider",
        return_value=fake_provider,
    ), patch(
        "ideagen.discovery.pipeline.get_default_llm_provider",
        return_value=fake_llm,
    ), patch(
        "ideagen.api.main.get_default_llm_provider",
        return_value=fake_llm,
    ), patch(
        "ideagen.llm.get_default_llm_provider",
        return_value=fake_llm,
    ):
        yield fake_provider


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
        "knowledge_id": knowledge.id,
        "session_maker": SessionMaker,
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


def test_generate_opportunity_endpoint_uses_configured_provider(test_db):
    client, ids = test_db
    res = client.post(f"/api/v1/gaps/{ids['gap_id']}/opportunity")
    assert res.status_code == 200
    report = res.json()
    assert report["research_problem"] == "Multi-hop retrieval error propagation"
    assert report["identified_gap"] == "Unquantified hallucination accumulation in multi-hop RAG"
    assert "proposed_research_direction" in report
    assert len(report["supporting_papers"]) == 1


def test_opportunity_endpoint_never_uses_random_vectors(test_db, mock_pipeline_dependencies):
    """Requirement 1: Verify the opportunity endpoint never uses random vectors for prior-art search."""
    client, ids = test_db
    with patch("numpy.random.randn", side_effect=AssertionError("np.random.randn should never be called")):
        res = client.post(f"/api/v1/gaps/{ids['gap_id']}/opportunity")
        assert res.status_code == 200


def test_actual_statement_text_passed_to_embedding_provider(test_db, mock_pipeline_dependencies):
    """Requirement 2 & 3: Actual statement text is passed to the configured embedding provider,
    preserving vector dimension and statement order."""
    client, ids = test_db
    SessionMaker = ids["session_maker"]
    with SessionMaker() as db:
        # Add additional statements to verify multi-statement ordering and dimension
        stmt2 = ResearchStatement(
            id=uuid.uuid4(),
            knowledge_id=uuid.uuid4(),
            paper_id=ids["paper_id"],
            passage_id=ids["stmt_id"],
            statement_text="Quantized key-value cache decreases activation memory by 4x.",
            statement_type="FINDING",
            confidence=0.95,
        )
        db.add(stmt2)
        db.commit()

        all_stmts = db.query(ResearchStatement).all()
        expected_texts = [s.statement_text for s in all_stmts]

    mock_pipeline_dependencies.call_history.clear()
    res = client.post(f"/api/v1/gaps/{ids['gap_id']}/opportunity")
    assert res.status_code == 200

    # Verify that the configured embedding provider was called with the actual statement texts in DB order
    assert expected_texts in mock_pipeline_dependencies.call_history


def test_empty_statement_collection_handled_safely(test_db, mock_pipeline_dependencies):
    """Requirement 4: Empty statement collections do not cause an error."""
    client, ids = test_db
    SessionMaker = ids["session_maker"]
    with SessionMaker() as db:
        # Delete evidence and statements
        db.query(GapEvidence).delete()
        db.query(ResearchStatement).delete()
        db.commit()

    mock_pipeline_dependencies.call_history.clear()
    res = client.post(f"/api/v1/gaps/{ids['gap_id']}/opportunity")
    assert res.status_code == 200
    report = res.json()
    assert report["identified_gap"] == "Unquantified hallucination accumulation in multi-hop RAG"


def test_method_statements_included_in_opportunity_inputs(test_db):
    """Requirement 5: METHOD, METHODOLOGY, and BASELINE statements are included in opportunity-generation inputs."""
    client, ids = test_db
    SessionMaker = ids["session_maker"]
    with SessionMaker() as db:
        method_stmt = ResearchStatement(
            id=uuid.uuid4(),
            knowledge_id=ids["knowledge_id"],
            paper_id=ids["paper_id"],
            statement_text="Adaptive semantic beam search with step-level entropy thresholding.",
            statement_type="METHOD",
            confidence=0.9,
        )
        methodology_stmt = ResearchStatement(
            id=uuid.uuid4(),
            knowledge_id=ids["knowledge_id"],
            paper_id=ids["paper_id"],
            statement_text="Dual-encoder retrieval with dense passage cross-attention.",
            statement_type="METHODOLOGY",
            confidence=0.9,
        )
        baseline_stmt = ResearchStatement(
            id=uuid.uuid4(),
            knowledge_id=ids["knowledge_id"],
            paper_id=ids["paper_id"],
            statement_text="BM25 sparse inverted index baseline.",
            statement_type="BASELINE",
            confidence=0.85,
        )
        limitation_stmt = ResearchStatement(
            id=uuid.uuid4(),
            knowledge_id=ids["knowledge_id"],
            paper_id=ids["paper_id"],
            statement_text="High latency during multi-hop iterative querying.",
            statement_type="LIMITATION",
            confidence=0.9,
        )
        db.add_all([method_stmt, methodology_stmt, baseline_stmt, limitation_stmt])

        # Link as supporting evidence for gap
        for s in [method_stmt, methodology_stmt, baseline_stmt, limitation_stmt]:
            db.add(GapEvidence(
                id=uuid.uuid4(),
                gap_id=ids["gap_id"],
                statement_id=s.id,
                paper_id=ids["paper_id"],
                evidence_role="SUPPORTING",
            ))
        db.commit()

    captured_inputs = []
    original_process = ResearchOpportunityPipeline.process_validated_gap

    def spy_process(self, input_dto, **kwargs):
        captured_inputs.append(input_dto)
        return original_process(self, input_dto, **kwargs)

    with patch.object(ResearchOpportunityPipeline, "process_validated_gap", side_effect=spy_process, autospec=True):
        res = client.post(f"/api/v1/gaps/{ids['gap_id']}/opportunity")
        assert res.status_code == 200

    assert len(captured_inputs) == 1
    inp = captured_inputs[0]
    assert "Adaptive semantic beam search with step-level entropy thresholding." in inp.existing_methods
    assert "Dual-encoder retrieval with dense passage cross-attention." in inp.existing_methods
    assert "BM25 sparse inverted index baseline." in inp.existing_methods
    assert "High latency during multi-hop iterative querying." not in inp.existing_methods


def test_method_statements_included_in_reports(test_db):
    """Requirement 6: METHOD, METHODOLOGY, and BASELINE statements are included in reports where applicable."""
    client, ids = test_db
    SessionMaker = ids["session_maker"]
    with SessionMaker() as db:
        m_stmt = ResearchStatement(
            id=uuid.uuid4(),
            knowledge_id=ids["knowledge_id"],
            paper_id=ids["paper_id"],
            statement_text="Hierarchical graph clustering with contrastive edge pruning.",
            statement_type="METHOD",
            confidence=0.91,
        )
        db.add(m_stmt)
        # Link statement to existing opportunity
        opp = db.query(ResearchOpportunityRecord).filter(ResearchOpportunityRecord.id == ids["opp_id"]).first()
        opp.supporting_statement_ids = [str(ids["stmt_id"]), str(m_stmt.id)]
        db.commit()

    res = client.get(f"/api/v1/opportunities/{ids['opp_id']}/report")
    assert res.status_code == 200
    report = res.json()
    assert "Hierarchical graph clustering with contrastive edge pruning." in report["existing_approaches"]


def test_existing_endpoint_contracts_and_gap_status_checks(test_db):
    """Requirement 7: Existing endpoint contracts and validated-gap status checks remain intact."""
    client, ids = test_db
    SessionMaker = ids["session_maker"]

    # 1. Non-existent gap returns 404
    non_existent_id = uuid.uuid4()
    res = client.post(f"/api/v1/gaps/{non_existent_id}/opportunity")
    assert res.status_code == 404
    assert f"Research gap {non_existent_id} not found" in res.json()["detail"]

    # 2. Non-VALIDATED gap returns 400
    with SessionMaker() as db:
        candidate_gap = ResearchGap(
            id=uuid.uuid4(),
            gap_statement="Unvalidated hypothesis regarding model pruning",
            research_problem="Model efficiency",
            status="CANDIDATE",
            evidence_count=0,
        )
        db.add(candidate_gap)
        db.commit()
        candidate_gap_id = candidate_gap.id

    res = client.post(f"/api/v1/gaps/{candidate_gap_id}/opportunity")
    assert res.status_code == 400
    assert "must be VALIDATED before opportunity generation" in res.json()["detail"]

    # 3. Validated gap returns 200 and conforms to OpportunityReportDTO contract
    res = client.post(f"/api/v1/gaps/{ids['gap_id']}/opportunity")
    assert res.status_code == 200
    data = res.json()
    required_keys = [
        "opportunity_id", "gap_id", "research_problem", "evidence_of_problem",
        "existing_approaches", "identified_gap", "why_existing_work_is_insufficient",
        "proposed_research_direction", "reasoning", "technical_approach",
        "expected_contribution", "feasibility", "prior_art_assessment",
        "differentiation", "supporting_papers", "supporting_evidence",
        "confidence", "novelty_assessment", "status", "created_at",
        "generation_model", "validation_model", "embedding_model",
        "refinement_iteration_count", "pipeline_version"
    ]
    for key in required_keys:
        assert key in data, f"Missing required contract key: {key}"
    assert data["status"] == "VALIDATED"


def test_discover_gaps_endpoint_accepts_cutoff_date(test_db):
    """Regression test for Priority 5: discover endpoint accepts cutoff_date query param
    and passes it to SemanticGapDiscoveryPipeline."""
    client, ids = test_db
    with patch("ideagen.api.main.SemanticGapDiscoveryPipeline.discover_gaps", return_value=[]) as mock_discover:
        res = client.post("/api/gaps/discover?cutoff_date=2024-06-01")
        assert res.status_code == 200
        mock_discover.assert_called_once_with(cutoff_date="2024-06-01", paper_ids=None)


def test_generate_opportunity_endpoint_respects_cutoff_date(test_db, mock_pipeline_dependencies):
    """Regression test for Priority 5: opportunity endpoint restricts prior-art FAISS index
    to statements strictly eligible at cutoff_date, excluding future papers."""
    client, ids = test_db
    SessionMaker = ids["session_maker"]

    future_paper_id = uuid.uuid4()
    future_stmt_id = uuid.uuid4()

    with SessionMaker() as db:
        # Paper from 2026 (future relative to cutoff 2025-06-01)
        future_paper = Paper(
            id=future_paper_id,
            source_type="MANUAL_UPLOAD",
            source_id="future_paper_1",
            title="Future Advances in LLM Reasoning (2026)",
            publication_date=datetime.date(2026, 1, 1),
            batch_id="batch_future",
            abstract="Future methods solving multi-hop reasoning completely.",
            authors=["Future Author"],
            referenced_works=[],
            raw_metadata={},
        )
        db.add(future_paper)

        future_doc = PaperDocument(
            id=uuid.uuid4(),
            paper_id=future_paper_id,
            full_text="Future paper full text.",
            extraction_method="PyMuPDF",
            extraction_version="v1.0",
            extracted_at=datetime.datetime.utcnow(),
        )
        db.add(future_doc)

        future_knowledge = ResearchKnowledge(
            id=uuid.uuid4(),
            paper_id=future_paper_id,
            research_problem="Future Multi-hop",
            methodology="Future Method",
            model_name="future-model",
            extracted_at=datetime.datetime.utcnow(),
        )
        db.add(future_knowledge)

        future_stmt = ResearchStatement(
            id=future_stmt_id,
            knowledge_id=future_knowledge.id,
            paper_id=future_paper_id,
            statement_text="Future breakthrough eliminates multi-hop hallucinations completely.",
            statement_type="FINDING",
            confidence=0.99,
        )
        db.add(future_stmt)
        db.commit()

    captured_indexes = []
    original_init = ResearchOpportunityPipeline.__init__

    def spy_init(self, *args, **kwargs):
        captured_indexes.append(kwargs.get("vector_index"))
        return original_init(self, *args, **kwargs)

    with patch.object(ResearchOpportunityPipeline, "__init__", side_effect=spy_init, autospec=True):
        # 1. When cutoff_date="2025-06-01" is passed, future paper statement must NOT be in vector index
        res = client.post(f"/api/v1/gaps/{ids['gap_id']}/opportunity?cutoff_date=2025-06-01")
        assert res.status_code == 200

        assert len(captured_indexes) == 1
        v_idx = captured_indexes[0]
        assert v_idx is not None
        # Historical statement (paper_id published 2025-01-01) should be indexed
        assert ids["stmt_id"] in v_idx.id_list
        # Future statement (paper_id published 2026-01-01) must NOT be indexed
        assert future_stmt_id not in v_idx.id_list

    # 2. When gap has metadata_json temporal_cutoff and no query param passed, it should also isolate
    captured_indexes.clear()
    with SessionMaker() as db:
        gap = db.query(ResearchGap).filter(ResearchGap.id == ids["gap_id"]).first()
        gap.metadata_json = {"temporal_cutoff": "2025-06-01"}
        db.commit()

    with patch.object(ResearchOpportunityPipeline, "__init__", side_effect=spy_init, autospec=True):
        res = client.post(f"/api/v1/gaps/{ids['gap_id']}/opportunity")
        assert res.status_code == 200

        assert len(captured_indexes) == 1
        v_idx = captured_indexes[0]
        assert ids["stmt_id"] in v_idx.id_list
        assert future_stmt_id not in v_idx.id_list



def test_generate_opportunity_bridge_db_failure_saves_nothing_and_hides_sql(test_db):
    """A database failure while bridging the generated record must not commit an unbridged
    record, must return an actionable 500 without SQL details, and must not break later requests."""
    from sqlalchemy.exc import IntegrityError
    from ideagen.db.models import Opportunity
    from ideagen.opportunity.bridge import OpportunityBridgeService

    client, ids = test_db
    SessionMaker = ids["session_maker"]
    with SessionMaker() as db:
        records_before = db.query(ResearchOpportunityRecord).count()
        lifecycle_before = db.query(Opportunity).count()

    db_error = IntegrityError(
        "INSERT INTO opportunities (id, research_opportunity_id) VALUES (?, ?)",
        ("secret-param",),
        Exception("UNIQUE constraint failed: opportunities.id"),
    )
    with patch.object(OpportunityBridgeService, "bridge_opportunity", side_effect=db_error):
        res = client.post(f"/api/v1/gaps/{ids['gap_id']}/opportunity")

    assert res.status_code == 500
    detail = res.json()["detail"]
    assert detail.startswith("Opportunity generation failed")
    assert "nothing was saved" in detail
    for leaked in ("INSERT", "opportunities.id", "secret-param", "UNIQUE"):
        assert leaked not in detail

    with SessionMaker() as db:
        assert db.query(ResearchOpportunityRecord).count() == records_before
        assert db.query(Opportunity).count() == lifecycle_before

    # The same client/database recovers: a retry succeeds and is bridged.
    res_ok = client.post(f"/api/v1/gaps/{ids['gap_id']}/opportunity")
    assert res_ok.status_code == 200
    new_id = uuid.UUID(res_ok.json()["opportunity_id"])
    with SessionMaker() as db:
        assert db.query(ResearchOpportunityRecord).count() == records_before + 1
        lifecycle = db.query(Opportunity).filter(Opportunity.research_opportunity_id == new_id).one()
        assert lifecycle.status == "OPEN"
