"""Comprehensive offline tests for Temporal Filtering and Leakage Prevention (Phase 3).

Verifies that:
1. Papers published before the cutoff are included.
2. Papers published exactly on the cutoff are included.
3. Papers published after the cutoff are excluded.
4. Papers with no publication date are excluded under a temporal cutoff.
5. Papers with invalid date formats are handled safely without crashing.
6. Statements from excluded papers never reach clustering or candidate generation.
7. Excluded statements cannot appear in candidate evidence.
8. Excluded papers' passages cannot leak into validation evidence.
9. Existing no-cutoff behavior remains backward compatible.
10. Invalid cutoff input is rejected clearly.
11. Existing Phase 1 limitation-eligibility behavior remains intact.
12. Existing Phase 2 fail-closed citation validation remains intact.
"""

from datetime import date, datetime, timezone
from pathlib import Path
import uuid
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from ideagen.db.models import Base, Paper, PaperDocument, PaperPassage, ResearchKnowledge, ResearchStatement, ResearchGap, GapEvidence
from ideagen.db.persistence import SQLAlchemyPersistenceLayer
from ideagen.discovery.models import (
    GapStatus,
    ValidationVerdict,
    GapCandidate,
    GapValidationResult,
)
from ideagen.discovery.pipeline import SemanticGapDiscoveryPipeline
from ideagen.discovery.temporal import (
    normalize_cutoff_date,
    is_paper_eligible,
    filter_statements_by_cutoff,
    extract_paper_publication_date,
)
from tests.discovery.test_gap_discovery import (
    DeterministicTestEmbeddingProvider,
    StubValidationProvider,
)


@pytest.fixture
def sqlite_session():
    """Provides an isolated in-memory SQLite database session with all schemas created."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_maker = sessionmaker(bind=engine)
    session = session_maker()
    yield session
    session.close()
    engine.dispose()


def test_normalize_cutoff_date_validation():
    """Requirement 10: Valid cutoffs are parsed to date; invalid or unparseable inputs are rejected clearly."""
    # Valid date object
    d = date(2024, 12, 31)
    assert normalize_cutoff_date(d) == d

    # Valid datetime object
    dt = datetime(2024, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
    assert normalize_cutoff_date(dt) == date(2024, 12, 31)

    # Valid ISO string
    assert normalize_cutoff_date("2024-12-31") == date(2024, 12, 31)
    assert normalize_cutoff_date("2024-12-31T15:30:00Z") == date(2024, 12, 31)
    assert normalize_cutoff_date("  2024-12-31  ") == date(2024, 12, 31)

    # None returns None (backward compatibility)
    assert normalize_cutoff_date(None) is None

    # Invalid string formats raise ValueError
    with pytest.raises(ValueError, match="Invalid cutoff date format"):
        normalize_cutoff_date("not-a-date")

    with pytest.raises(ValueError, match="Invalid cutoff date format"):
        normalize_cutoff_date("2024/12/31")

    with pytest.raises(ValueError, match="empty or whitespace"):
        normalize_cutoff_date("   ")

    # Unsupported types raise TypeError
    with pytest.raises(TypeError, match="Unsupported cutoff date type"):
        normalize_cutoff_date(12345)


def test_extract_paper_publication_date_safety():
    """Requirement 5: Missing or invalid paper publication dates are handled safely without fabricating dates."""
    # Date object
    assert extract_paper_publication_date({"publication_date": date(2024, 5, 1)}) == date(2024, 5, 1)

    # ISO string
    assert extract_paper_publication_date({"publication_date": "2024-05-01"}) == date(2024, 5, 1)

    # None / missing
    assert extract_paper_publication_date({"publication_date": None}) is None
    assert extract_paper_publication_date({}) is None
    assert extract_paper_publication_date(None) is None

    # Invalid string does not crash and returns None
    assert extract_paper_publication_date({"publication_date": "corrupted-date-value"}) is None
    assert extract_paper_publication_date({"publication_date": ""}) is None


def test_paper_eligibility_rules():
    """Requirements 1, 2, 3, 4: Tests cutoff boundary rules on paper models."""
    cutoff = date(2024, 6, 30)

    # 1. Before cutoff: eligible
    p_before = {"publication_date": date(2024, 6, 29)}
    assert is_paper_eligible(p_before, cutoff) is True

    # 2. Exactly on cutoff: eligible
    p_exact = {"publication_date": date(2024, 6, 30)}
    assert is_paper_eligible(p_exact, cutoff) is True

    # 3. After cutoff: excluded
    p_after = {"publication_date": date(2024, 7, 1)}
    assert is_paper_eligible(p_after, cutoff) is False

    # 4. Undated: excluded under temporal cutoff
    p_undated = {"publication_date": None}
    assert is_paper_eligible(p_undated, cutoff) is False

    # 5. Invalid date: excluded
    p_invalid = {"publication_date": "unparseable"}
    assert is_paper_eligible(p_invalid, cutoff) is False


def test_persistence_statement_and_passage_temporal_filtering(sqlite_session):
    """Verifies that persistence queries at the database layer exclude future and undated records."""
    persistence = SQLAlchemyPersistenceLayer(sqlite_session)
    cutoff = date(2024, 6, 30)

    # 1. Setup 4 papers: past, exact, future, undated
    p_past = Paper(source_type="MANUAL_UPLOAD", source_id="P_PAST", batch_id="b1", abstract="a", publication_date=date(2024, 1, 15), authors=[], referenced_works=[], raw_metadata={})
    p_exact = Paper(source_type="MANUAL_UPLOAD", source_id="P_EXACT", batch_id="b1", abstract="a", publication_date=date(2024, 6, 30), authors=[], referenced_works=[], raw_metadata={})
    p_future = Paper(source_type="MANUAL_UPLOAD", source_id="P_FUTURE", batch_id="b1", abstract="a", publication_date=date(2024, 7, 1), authors=[], referenced_works=[], raw_metadata={})
    p_undated = Paper(source_type="MANUAL_UPLOAD", source_id="P_UNDATED", batch_id="b1", abstract="a", publication_date=None, authors=[], referenced_works=[], raw_metadata={})

    sqlite_session.add_all([p_past, p_exact, p_future, p_undated])
    sqlite_session.commit()

    # Create documents & passages
    passages = []
    for p in [p_past, p_exact, p_future, p_undated]:
        doc = PaperDocument(paper_id=p.id, full_text="Text", extraction_method="test", extraction_version="v1", extracted_at=datetime.now(timezone.utc))
        sqlite_session.add(doc)
        sqlite_session.commit()

        pas = PaperPassage(paper_id=p.id, document_id=doc.id, passage_index=0, page_start=1, page_end=1, section="Intro", text=f"Passage for {p.source_id}")
        sqlite_session.add(pas)
        sqlite_session.commit()
        passages.append(pas)

    # Create knowledge & statements
    for p, pas in zip([p_past, p_exact, p_future, p_undated], passages):
        k = ResearchKnowledge(paper_id=p.id, model_name="test", extracted_at=datetime.now(timezone.utc), extraction_status="SUCCESS")
        sqlite_session.add(k)
        sqlite_session.commit()

        stmt = ResearchStatement(
            knowledge_id=k.id,
            paper_id=p.id,
            passage_id=pas.id,
            statement_text=f"Limitation from {p.source_id}: Severe latency under stress.",
            statement_type="LIMITATION",
            section="Limitations",
        )
        sqlite_session.add(stmt)
    sqlite_session.commit()

    # Test A: Temporally filtered statements
    eligible_stmts = persistence.get_all_research_statements(cutoff_date=cutoff)
    eligible_paper_ids = {s.paper_id for s in eligible_stmts}

    assert len(eligible_stmts) == 2
    assert p_past.id in eligible_paper_ids
    assert p_exact.id in eligible_paper_ids
    assert p_future.id not in eligible_paper_ids
    assert p_undated.id not in eligible_paper_ids

    # Test B: Temporally filtered passages
    eligible_passages = persistence.get_all_passages(cutoff_date=cutoff)
    eligible_pass_paper_ids = {pas.paper_id for pas in eligible_passages}

    assert len(eligible_passages) == 2
    assert p_past.id in eligible_pass_paper_ids
    assert p_exact.id in eligible_pass_paper_ids
    assert p_future.id not in eligible_pass_paper_ids
    assert p_undated.id not in eligible_pass_paper_ids

    # Test C: No cutoff returns all 4 (backward compatible)
    all_stmts = persistence.get_all_research_statements(cutoff_date=None)
    assert len(all_stmts) == 4

    all_pass = persistence.get_all_passages(cutoff_date=None)
    assert len(all_pass) == 4


def test_end_to_end_discovery_pipeline_prevents_future_leakage(sqlite_session, tmp_path):
    """Requirements 6, 7, 8: Verifies future statements and passages never enter clustering, candidates, or validation."""
    persistence = SQLAlchemyPersistenceLayer(sqlite_session)
    cutoff = date(2024, 6, 30)

    # 1. Historical paper (eligible)
    p_past = Paper(
        source_type="MANUAL_UPLOAD",
        source_id="HISTORICAL_PAPER",
        batch_id="b1",
        abstract="a",
        publication_date=date(2024, 3, 15),
        authors=[],
        referenced_works=[],
        raw_metadata={},
    )
    # 2. Future paper (excluded)
    p_future = Paper(
        source_type="MANUAL_UPLOAD",
        source_id="FUTURE_PAPER_2025",
        batch_id="b1",
        abstract="a",
        publication_date=date(2025, 2, 10),
        authors=[],
        referenced_works=[],
        raw_metadata={},
    )
    sqlite_session.add_all([p_past, p_future])
    sqlite_session.commit()

    # Passages
    doc_past = PaperDocument(paper_id=p_past.id, full_text="Past text", extraction_method="t", extraction_version="v1", extracted_at=datetime.now(timezone.utc))
    doc_future = PaperDocument(paper_id=p_future.id, full_text="Future text", extraction_method="t", extraction_version="v1", extracted_at=datetime.now(timezone.utc))
    sqlite_session.add_all([doc_past, doc_future])
    sqlite_session.commit()

    pas_past = PaperPassage(paper_id=p_past.id, document_id=doc_past.id, passage_index=0, page_start=1, page_end=1, section="Exp", text="Historical empirical passage text on RAG degradation.")
    pas_future = PaperPassage(paper_id=p_future.id, document_id=doc_future.id, passage_index=0, page_start=1, page_end=1, section="Exp", text="Future 2025 empirical passage text describing breakthrough.")
    sqlite_session.add_all([pas_past, pas_future])
    sqlite_session.commit()

    # Statements
    k_past = ResearchKnowledge(paper_id=p_past.id, model_name="t", extracted_at=datetime.now(timezone.utc), extraction_status="SUCCESS")
    k_future = ResearchKnowledge(paper_id=p_future.id, model_name="t", extracted_at=datetime.now(timezone.utc), extraction_status="SUCCESS")
    sqlite_session.add_all([k_past, k_future])
    sqlite_session.commit()

    s_past = ResearchStatement(
        knowledge_id=k_past.id,
        paper_id=p_past.id,
        passage_id=pas_past.id,
        statement_text="Context pruning fails to filter distractor passages in scientific QA.",
        statement_type="LIMITATION",
        section="Limitations",
    )
    s_future = ResearchStatement(
        knowledge_id=k_future.id,
        paper_id=p_future.id,
        passage_id=pas_future.id,
        statement_text="Future 2025 finding: Advanced transformer distillation solves context pruning.",
        statement_type="LIMITATION",
        section="Limitations",
    )
    sqlite_session.add_all([s_past, s_future])
    sqlite_session.commit()

    # Setup pipeline with mock LLM that cites historical statement
    val_provider = StubValidationProvider(
        verdict=ValidationVerdict.VALID,
        reasoning="Validated based strictly on historical evidence.",
        cited_ids=[str(s_past.id)],
    )
    embed_provider = DeterministicTestEmbeddingProvider(dimension=64)

    pipeline = SemanticGapDiscoveryPipeline(
        persistence=persistence,
        embedding_provider=embed_provider,
        llm_provider=val_provider,
        index_dir=tmp_path / "indexes",
        similarity_threshold=0.50,
        min_supporting_evidence=1,
    )

    # Run discovery with cutoff
    gaps = pipeline.discover_gaps(persist_index=True, cutoff_date=cutoff)

    assert len(gaps) >= 1
    for gap in gaps:
        # Requirement 6 & 7: Excluded future statements cannot appear in supporting evidence
        assert s_future.id not in gap.supporting_statement_ids
        assert p_future.id not in gap.supporting_paper_ids
        # Historical statement is present
        assert s_past.id in gap.supporting_statement_ids
        assert p_past.id in gap.supporting_paper_ids
        # Temporal cutoff is auditable in gap metadata
        assert gap.metadata.get("temporal_cutoff") == "2024-06-30"

    # Requirement 8: Future paper passage was never supplied in validation prompt
    assert val_provider.last_prompt is not None
    assert "Historical empirical passage text on RAG degradation." in val_provider.last_prompt
    assert "Future 2025 empirical passage text describing breakthrough." not in val_provider.last_prompt
    assert str(s_future.id) not in val_provider.last_prompt
    assert str(p_future.id) not in val_provider.last_prompt


def test_discovery_pipeline_backward_compatible_without_cutoff(sqlite_session, tmp_path):
    """Requirement 9: Without a cutoff, all statements are eligible as before."""
    persistence = SQLAlchemyPersistenceLayer(sqlite_session)

    p1 = Paper(source_type="MANUAL_UPLOAD", source_id="P1", batch_id="b1", abstract="a", publication_date=date(2024, 1, 1), authors=[], referenced_works=[], raw_metadata={})
    p2 = Paper(source_type="MANUAL_UPLOAD", source_id="P2", batch_id="b1", abstract="a", publication_date=date(2025, 1, 1), authors=[], referenced_works=[], raw_metadata={})
    sqlite_session.add_all([p1, p2])
    sqlite_session.commit()

    k1 = ResearchKnowledge(paper_id=p1.id, model_name="t", extracted_at=datetime.now(timezone.utc), extraction_status="SUCCESS")
    k2 = ResearchKnowledge(paper_id=p2.id, model_name="t", extracted_at=datetime.now(timezone.utc), extraction_status="SUCCESS")
    sqlite_session.add_all([k1, k2])
    sqlite_session.commit()

    s1 = ResearchStatement(knowledge_id=k1.id, paper_id=p1.id, statement_text="Limitation 1: Severe latency in search.", statement_type="LIMITATION", section="Limitations")
    s2 = ResearchStatement(knowledge_id=k2.id, paper_id=p2.id, statement_text="Limitation 2: Search latency increases exponentially.", statement_type="LIMITATION", section="Limitations")
    sqlite_session.add_all([s1, s2])
    sqlite_session.commit()

    val_provider = StubValidationProvider(
        verdict=ValidationVerdict.VALID,
        reasoning="Validated across all papers.",
        cited_ids=[str(s1.id), str(s2.id)],
    )
    embed_provider = DeterministicTestEmbeddingProvider(dimension=64)

    pipeline = SemanticGapDiscoveryPipeline(
        persistence=persistence,
        embedding_provider=embed_provider,
        llm_provider=val_provider,
        index_dir=tmp_path / "indexes",
        similarity_threshold=0.50,
        min_supporting_evidence=1,
    )

    # Calling without cutoff
    gaps = pipeline.discover_gaps(persist_index=False, cutoff_date=None)

    assert len(gaps) >= 1
    # Both statements are present in unconstrained run
    all_supporting = {sid for g in gaps for sid in g.supporting_statement_ids}
    assert s1.id in all_supporting
    assert s2.id in all_supporting
    assert "temporal_cutoff" not in gaps[0].metadata


def test_pipeline_temporal_cutoff_does_not_persist_shared_faiss_index(sqlite_session, tmp_path: Path):
    """Priority 3: Verifies that a temporal cutoff run never persists its filtered index to shared disk directory."""
    persistence = SQLAlchemyPersistenceLayer(sqlite_session)

    p1 = Paper(source_type="MANUAL_UPLOAD", source_id="p1", title="Paper 2023", publication_date=date(2023, 1, 1), batch_id="b1", abstract="A1", authors=[], referenced_works=[], raw_metadata={})
    sqlite_session.add(p1)
    sqlite_session.commit()

    k1 = ResearchKnowledge(paper_id=p1.id, model_name="m", extracted_at=datetime.now(timezone.utc), extraction_status="SUCCESS")
    sqlite_session.add(k1)
    sqlite_session.commit()

    s1 = ResearchStatement(knowledge_id=k1.id, paper_id=p1.id, statement_text="Historical limitation text.", statement_type="LIMITATION", section="Limitations")
    sqlite_session.add(s1)
    sqlite_session.commit()

    val_provider = StubValidationProvider(
        verdict=ValidationVerdict.VALID,
        reasoning="Validated.",
        cited_ids=[str(s1.id)],
    )
    embed_provider = DeterministicTestEmbeddingProvider(dimension=64)

    index_dir = tmp_path / "protected_shared_indexes"
    pipeline = SemanticGapDiscoveryPipeline(
        persistence=persistence,
        embedding_provider=embed_provider,
        llm_provider=val_provider,
        index_dir=index_dir,
        similarity_threshold=0.50,
        min_supporting_evidence=1,
    )

    # 1. Run with persist_index=True BUT cutoff_date is active -> must NOT persist to index_dir
    gaps = pipeline.discover_gaps(persist_index=True, cutoff_date="2023-06-30")
    assert len(gaps) >= 1
    assert not (index_dir / "index.faiss").exists()
    assert not (index_dir / "meta.json").exists()

    # 2. Run without cutoff_date and persist_index=True -> SHOULD persist to index_dir
    gaps_full = pipeline.discover_gaps(persist_index=True, cutoff_date=None)
    assert len(gaps_full) >= 1
    assert (index_dir / "index.faiss").exists()
    assert (index_dir / "meta.json").exists()

