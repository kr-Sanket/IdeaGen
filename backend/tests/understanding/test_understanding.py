"""Tests for AI Paper Understanding (Phase 3).

Verifies ResearchKnowledge schemas, structured model output parsing,
malformed output handling, passage selection, statement provenance mapping,
PostgreSQL persistence, and real model integration.
"""

import os
from datetime import datetime, timezone
import uuid
import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ideagen.core.config import get_settings
from ideagen.db.models import Base, Paper, PaperDocument, PaperPassage
from ideagen.db.persistence import SQLAlchemyPersistenceLayer
from ideagen.ingestion.models import PaperCreateDTO, SourceType
from ideagen.llm.exceptions import ProviderFailureError, MalformedResponseError
from ideagen.llm.gemini_provider import GeminiLLMProvider
from ideagen.llm.provider import LLMProvider
from ideagen.understanding.models import (
    StatementType,
    ExtractionStatus,
    RawStatementExtraction,
    StructuredPaperExtractionOutput,
    ResearchStatementCreateDTO,
    ResearchKnowledgeCreateDTO,
    ResearchKnowledgeResponseDTO,
)
from ideagen.understanding.passage_selector import PassageSelector, SelectedPassage
from ideagen.understanding.extractor import ScientificPaperExtractor
from ideagen.understanding.pipeline import PaperUnderstandingPipeline


settings = get_settings()
DB_URL = settings.database_url
if not DB_URL:
    raise ValueError("IDEAGEN_DATABASE_URL environment variable is required to run database integration tests.")


@pytest.fixture(scope="session")
def engine():
    engine = create_engine(DB_URL)
    Base.metadata.create_all(engine)
    yield engine


@pytest.fixture
def session(engine):
    connection = engine.connect()
    transaction = connection.begin()
    session_maker = sessionmaker(bind=connection, join_transaction_mode="create_savepoint")
    session = session_maker()
    yield session
    session.close()
    transaction.rollback()
    connection.close()


class StubRealContractProvider(LLMProvider):
    """Deterministic provider implementing the exact LLMProvider protocol for boundary verification."""
    def __init__(self, response_data: dict = None, should_fail: bool = False, fail_type: str = "provider"):
        self.response_data = response_data or {}
        self.should_fail = should_fail
        self.fail_type = fail_type
        self.last_prompt = None

    def generate_structured(self, prompt: str, response_schema):
        self.last_prompt = prompt
        if self.should_fail:
            if self.fail_type == "malformed":
                raise MalformedResponseError("Invalid JSON received from model.")
            raise ProviderFailureError("Model connection timed out.")
        return response_schema.model_validate(self.response_data)


def test_research_knowledge_schema_validation():
    """Verifies Pydantic validation rules on research knowledge schemas."""
    paper_id = uuid.uuid4()
    passage_id = uuid.uuid4()

    # Valid statement
    statement = ResearchStatementCreateDTO(
        statement_text="RAG hallucination rates increase with distractor documents.",
        statement_type=StatementType.FINDING,
        source_passage_id=passage_id,
        source_page_start=1,
        source_page_end=2,
        section="Results",
        confidence=0.95,
    )
    assert statement.statement_type == StatementType.FINDING
    assert statement.confidence == 0.95

    # Valid knowledge
    knowledge = ResearchKnowledgeCreateDTO(
        paper_id=paper_id,
        research_problem="Hallucinations in Retrieval-Augmented Generation.",
        research_questions=["How does distractor noise affect generation faithfulness?"],
        methodology="Controlled injection of irrelevant documents into context.",
        datasets=["HotpotQA", "NQ-Open"],
        findings=["Noise reduces faithfulness by 34%."],
        limitations=["Only evaluated on extractive QA datasets."],
        future_work=["Evaluate on multi-step reasoning corpora."],
        research_statements=[statement],
        model_name="test-model",
        model_version="1.0",
        extraction_schema_version="v1.0",
    )
    assert knowledge.paper_id == paper_id
    assert len(knowledge.datasets) == 2
    assert len(knowledge.research_statements) == 1

    # Empty optional fields explicitly supported (not fabricated)
    empty_knowledge = ResearchKnowledgeCreateDTO(
        paper_id=paper_id,
        research_problem=None,
        methodology=None,
        model_name="test-model",
    )
    assert empty_knowledge.research_problem is None
    assert empty_knowledge.research_questions == []


def test_structured_model_output_parsing():
    """Verifies StructuredPaperExtractionOutput parsing and validation."""
    raw_json = {
        "research_problem": "Lack of evidence grounding in LLM opportunities.",
        "research_questions": ["Can graph verification improve gap precision?"],
        "methodology": "Graph-based constraint validation.",
        "datasets": ["PubMed-QA"],
        "findings": ["Precision increased from 0.62 to 0.88."],
        "limitations": ["Evaluated only in English."],
        "future_work": ["Multilingual graph validation."],
        "statements": [
            {
                "statement_text": "Graph verification reduces spurious opportunities.",
                "statement_type": "CONTRIBUTION",
                "passage_index": 2,
                "confidence": 0.92,
            },
            {
                "statement_text": "Evaluating on non-English papers remains open.",
                "statement_type": "FUTURE_WORK",
                "passage_index": 5,
                "confidence": 0.85,
            }
        ]
    }
    output = StructuredPaperExtractionOutput.model_validate(raw_json)
    assert output.research_problem == "Lack of evidence grounding in LLM opportunities."
    assert len(output.statements) == 2
    assert output.statements[0].statement_type == StatementType.CONTRIBUTION
    assert output.statements[0].passage_index == 2


def test_malformed_output_handling():
    """Verifies that model failures and malformed outputs are handled safely without fabricating data."""
    paper_id = uuid.uuid4()
    passages = [
        SelectedPassage(
            passage_id=uuid.uuid4(),
            passage_index=0,
            page_start=1,
            page_end=1,
            section="Abstract",
            text="This is an abstract about LLM reliability.",
        )
    ]

    # 1. Provider failure error
    failing_provider = StubRealContractProvider(should_fail=True, fail_type="provider")
    extractor = ScientificPaperExtractor(provider=failing_provider, model_name="test-failing")
    res = extractor.extract(paper_id=paper_id, selected_passages=passages)

    assert res.extraction_status == ExtractionStatus.FAILED
    assert "Model connection timed out" in res.error_message
    assert res.research_problem is None
    assert res.research_statements == []

    # 2. Malformed JSON error
    malformed_provider = StubRealContractProvider(should_fail=True, fail_type="malformed")
    extractor_malformed = ScientificPaperExtractor(provider=malformed_provider, model_name="test-malformed")
    res2 = extractor_malformed.extract(paper_id=paper_id, selected_passages=passages)

    assert res2.extraction_status == ExtractionStatus.FAILED
    assert "Invalid JSON" in res2.error_message
    assert res2.research_problem is None


def test_passage_selector_deterministic():
    """Verifies deterministic passage selection prioritizing critical scientific sections."""
    doc_id = uuid.uuid4()
    paper_id = uuid.uuid4()

    passages = [
        PaperPassage(id=uuid.uuid4(), paper_id=paper_id, document_id=doc_id, passage_index=0, page_start=1, page_end=1, section="Abstract", text="Abstract passage describing the problem."),
        PaperPassage(id=uuid.uuid4(), paper_id=paper_id, document_id=doc_id, passage_index=1, page_start=1, page_end=1, section="Introduction", text="Introduction passage."),
        PaperPassage(id=uuid.uuid4(), paper_id=paper_id, document_id=doc_id, passage_index=2, page_start=2, page_end=2, section="Related Work", text="Related work background."),
        PaperPassage(id=uuid.uuid4(), paper_id=paper_id, document_id=doc_id, passage_index=3, page_start=3, page_end=3, section="Methodology", text="Detailed methodology description."),
        PaperPassage(id=uuid.uuid4(), paper_id=paper_id, document_id=doc_id, passage_index=4, page_start=4, page_end=4, section="Results", text="Empirical evaluation results."),
        PaperPassage(id=uuid.uuid4(), paper_id=paper_id, document_id=doc_id, passage_index=5, page_start=5, page_end=5, section="Limitations", text="Key limitations of our method."),
        PaperPassage(id=uuid.uuid4(), paper_id=paper_id, document_id=doc_id, passage_index=6, page_start=5, page_end=5, section="Conclusion", text="Conclusion and future summary."),
    ]

    selector = PassageSelector(max_passages=4)
    selected = selector.select_passages(passages)

    assert len(selected) <= 4
    # Must preserve ordering by passage_index
    assert [s.passage_index for s in selected] == sorted([s.passage_index for s in selected])
    # Priority sections (Abstract, Limitations) must be present
    sections_present = [s.section for s in selected]
    assert "Abstract" in sections_present
    assert "Limitations" in sections_present


def test_passage_provenance_mapping():
    """Verifies statements map accurately to their source passage ID and page bounds."""
    paper_id = uuid.uuid4()
    p0_id = uuid.uuid4()
    p1_id = uuid.uuid4()

    passages = [
        SelectedPassage(passage_id=p0_id, passage_index=0, page_start=1, page_end=1, section="Abstract", text="We introduce RAG-Guard to mitigate hallucination."),
        SelectedPassage(passage_id=p1_id, passage_index=1, page_start=4, page_end=4, section="Limitations", text="Our benchmark does not include multimodal tokens."),
    ]

    provider = StubRealContractProvider(response_data={
        "research_problem": "Hallucination in RAG systems.",
        "research_questions": [],
        "methodology": "RAG-Guard framework.",
        "datasets": [],
        "findings": [],
        "limitations": ["No multimodal support."],
        "future_work": [],
        "statements": [
            {
                "statement_text": "We introduce RAG-Guard to mitigate hallucination.",
                "statement_type": "CONTRIBUTION",
                "passage_index": 0,
                "confidence": 0.98,
            },
            {
                "statement_text": "Our benchmark does not include multimodal tokens.",
                "statement_type": "LIMITATION",
                "passage_index": 1,
                "confidence": 0.91,
            }
        ]
    })

    extractor = ScientificPaperExtractor(provider=provider, model_name="contract-model")
    result = extractor.extract(paper_id=paper_id, selected_passages=passages)

    assert result.extraction_status == ExtractionStatus.SUCCESS
    assert len(result.research_statements) == 2

    # Check statement 0 provenance
    stmt0 = result.research_statements[0]
    assert stmt0.source_passage_id == p0_id
    assert stmt0.source_page_start == 1
    assert stmt0.source_page_end == 1
    assert stmt0.section == "Abstract"
    assert stmt0.statement_type == StatementType.CONTRIBUTION

    # Check statement 1 provenance
    stmt1 = result.research_statements[1]
    assert stmt1.source_passage_id == p1_id
    assert stmt1.source_page_start == 4
    assert stmt1.source_page_end == 4
    assert stmt1.section == "Limitations"
    assert stmt1.statement_type == StatementType.LIMITATION


def test_statement_provenance_and_confidence():
    """Verifies statement confidence retention and unknown passage handling."""
    paper_id = uuid.uuid4()
    p0_id = uuid.uuid4()
    passages = [
        SelectedPassage(passage_id=p0_id, passage_index=0, page_start=2, page_end=3, section="Method", text="Gradient masking prevents gradient leakage."),
    ]

    provider = StubRealContractProvider(response_data={
        "research_problem": "Privacy in distributed learning.",
        "statements": [
            {
                "statement_text": "Gradient masking prevents gradient leakage.",
                "statement_type": "METHOD",
                "passage_index": 0,
                "confidence": 0.88,
            },
            {
                "statement_text": "An unreferenced statement without passage index.",
                "statement_type": "ASSUMPTION",
                "passage_index": None,
                "confidence": 0.70,
            }
        ]
    })
    extractor = ScientificPaperExtractor(provider=provider, model_name="confidence-test")
    result = extractor.extract(paper_id=paper_id, selected_passages=passages)

    assert result.research_statements[0].confidence == 0.88
    assert result.research_statements[0].source_passage_id == p0_id
    assert result.research_statements[1].source_passage_id is None
    assert result.research_statements[1].statement_type == StatementType.ASSUMPTION


def test_extraction_metadata_and_versioning():
    """Verifies that model versioning, extraction schema, and timestamps are captured."""
    paper_id = uuid.uuid4()
    passages = [
        SelectedPassage(passage_id=uuid.uuid4(), passage_index=0, page_start=1, page_end=1, section="Abstract", text="Text"),
    ]
    provider = StubRealContractProvider(response_data={"research_problem": "Test Problem", "statements": []})
    extractor = ScientificPaperExtractor(
        provider=provider,
        model_name="scibert-v2",
        model_version="2.1.0",
        schema_version="v1.0"
    )
    result = extractor.extract(paper_id=paper_id, selected_passages=passages)

    assert result.model_name == "scibert-v2"
    assert result.model_version == "2.1.0"
    assert result.extraction_schema_version == "v1.0"
    assert isinstance(result.extracted_at, datetime)


def test_research_knowledge_persistence_and_retrieval(session):
    """Verifies PostgreSQL persistence and retrieval of ResearchKnowledge and ResearchStatements."""
    layer = SQLAlchemyPersistenceLayer(session)

    # 1. Create and persist paper and passage
    paper_dto = PaperCreateDTO(
        source_type=SourceType.MANUAL_UPLOAD,
        source_id="TEST_KNOWLEDGE_PAPER",
        title="Knowledge Testing Paper",
        batch_id="Batch 3",
        abstract="Abstract testing knowledge persistence.",
        authors=[{"name": "Alice"}, {"name": "Bob"}],
        referenced_works=[],
        raw_metadata={},
    )
    paper_id = layer.save_paper(paper_dto)

    doc = PaperDocument(
        paper_id=paper_id,
        full_text="Sample text",
        extraction_method="pymupdf",
        extraction_version="v2.0",
        extracted_at=datetime.now(timezone.utc),
    )
    session.add(doc)
    session.commit()

    passage = PaperPassage(
        paper_id=paper_id,
        document_id=doc.id,
        passage_index=0,
        page_start=1,
        page_end=1,
        section="Abstract",
        text="Sample passage text.",
    )
    session.add(passage)
    session.commit()

    # 2. Persist ResearchKnowledge with statements
    stmt_dto = ResearchStatementCreateDTO(
        statement_text="Verified finding in test paper.",
        statement_type=StatementType.FINDING,
        source_passage_id=passage.id,
        source_page_start=1,
        source_page_end=1,
        section="Abstract",
        confidence=0.96,
    )

    knowledge_dto = ResearchKnowledgeCreateDTO(
        paper_id=paper_id,
        research_problem="Reliability in scientific extraction.",
        research_questions=["Can provenance be persisted faithfully?"],
        methodology="SQLAlchemy and PostgreSQL persistence.",
        datasets=["IdeaGen-Bench"],
        findings=["All assertions succeed."],
        limitations=["Single paper test scope."],
        future_work=["Expand to entire corpus."],
        research_statements=[stmt_dto],
        model_name="postgres-test-model",
        model_version="1.0",
        extraction_schema_version="v1.0",
    )

    k_id = layer.save_research_knowledge(knowledge_dto)
    assert k_id is not None

    # 3. Retrieve and assert
    retrieved = layer.get_research_knowledge_by_paper(paper_id)
    assert retrieved is not None
    assert retrieved.id == k_id
    assert retrieved.research_problem == "Reliability in scientific extraction."
    assert retrieved.datasets == ["IdeaGen-Bench"]
    assert retrieved.model_name == "postgres-test-model"

    statements = layer.get_statements_by_paper(paper_id)
    assert len(statements) == 1
    assert statements[0].statement_text == "Verified finding in test paper."
    assert statements[0].statement_type == "FINDING"
    assert statements[0].passage_id == passage.id
    assert statements[0].confidence == 0.96


def test_real_model_inference_or_integration(session):
    """Verifies end-to-end extraction using real pretrained Gemini 2.5 Flash model when credentials are configured."""
    api_key = settings.gemini_api_key
    if not api_key:
        pytest.skip("GEMINI_API_KEY not configured; skipping live model inference test.")

    layer = SQLAlchemyPersistenceLayer(session)

    # 1. Create a Paper with real scientific passages
    paper_dto = PaperCreateDTO(
        source_type=SourceType.MANUAL_UPLOAD,
        source_id="REAL_AI_TEST_PAPER",
        title="Mitigating Hallucinations in Large Language Models via Grounded Reasoning",
        batch_id="Batch 3",
        abstract="Hallucinations remain a critical obstacle to the deployment of LLMs in mission-critical applications.",
        authors=[{"name": "Dr. Researcher"}],
        referenced_works=[],
        raw_metadata={},
    )
    paper_id = layer.save_paper(paper_dto)

    doc = PaperDocument(
        paper_id=paper_id,
        full_text="Full text of research paper",
        extraction_method="pymupdf",
        extraction_version="v2.0",
        extracted_at=datetime.now(timezone.utc),
    )
    session.add(doc)
    session.commit()

    passages = [
        PaperPassage(
            paper_id=paper_id,
            document_id=doc.id,
            passage_index=0,
            page_start=1,
            page_end=1,
            section="Abstract",
            text=(
                "Hallucination in Retrieval-Augmented Generation (RAG) significantly limits reliability in scientific discovery. "
                "We propose FaithGuard, a graph-grounded verification procedure that validates scientific claims against citation graphs. "
                "Our evaluation demonstrates a 28% reduction in ungrounded claims on the SciFact benchmark."
            ),
        ),
        PaperPassage(
            paper_id=paper_id,
            document_id=doc.id,
            passage_index=1,
            page_start=5,
            page_end=5,
            section="Limitations",
            text=(
                "A limitation of FaithGuard is the inference overhead introduced by graph traversal, "
                "which adds approximately 140ms latency per generation. Furthermore, evaluations were restricted to English scientific text."
            ),
        ),
    ]
    session.add_all(passages)
    session.commit()

    # 2. Run real PaperUnderstandingPipeline with Gemini 2.5 Flash
    provider = GeminiLLMProvider(api_key=api_key, model_name="gemini-2.5-flash")
    pipeline = PaperUnderstandingPipeline(persistence=layer, provider=provider, model_name="gemini-2.5-flash")

    response = pipeline.understand_paper(paper_id)

    # 3. Assertions on real AI structured extraction
    assert response.extraction_status == "SUCCESS"
    assert response.research_problem is not None
    assert len(response.research_problem) > 5
    assert response.methodology is not None
    assert response.statement_count > 0

    # Verify statements persisted in PostgreSQL with provenance
    persisted_statements = layer.get_statements_by_paper(paper_id)
    assert len(persisted_statements) == response.statement_count
    for stmt in persisted_statements:
        assert stmt.statement_text
        assert stmt.statement_type in [st.value for st in StatementType]
        # Must link to either passage 0 or passage 1
        assert stmt.passage_id in [p.id for p in passages]
