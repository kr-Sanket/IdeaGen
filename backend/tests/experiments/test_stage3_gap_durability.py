"""Deterministic tests for Checkpoint 5: Prevent Stale or Partial Gap Reuse.

Verifies:
1. Unchanged complete discovery inputs -> reuse permitted.
2. Adding a newly understood paper -> stale reuse prevented.
3. Retrying a failed paper -> stale reuse prevented.
4. Interrupted discovery after partial gap persistence -> partial set is not treated as complete.
5. Changed cutoff or statement-ID set -> reuse prevented.
6. Successful full discovery -> completion metadata recorded.
7. Legacy database without completion metadata -> handled safely.
8. Existing resume and fingerprint protections still work.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import uuid

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_BACKEND_DIR = _REPO_ROOT / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

import pymupdf
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

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
from ideagen.discovery.models import GapStatus
from experiments.runner import (
    ControlledExperimentRunner,
    ExperimentMetadataRecord,
    STAGE_3_DISCOVERY_COMPLETION_KEY,
    compute_discovery_input_signature,
)
from tests.experiments.test_controlled_experiment_runner import (
    MockExperimentLLMProvider,
    DeterministicExperimentEmbeddingProvider,
    _create_synthetic_pdf,
)


@pytest.fixture
def two_paper_corpus(tmp_path: Path) -> Path:
    """Creates a two-paper test corpus with explicit publication dates."""
    corpus_dir = tmp_path / "corpus_stage3"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    pdf1 = corpus_dir / "paper_a.pdf"
    pdf2 = corpus_dir / "paper_b.pdf"

    _create_synthetic_pdf(
        pdf1,
        "Paper A: Analysis of Distractor Injection in Scientific Retrieval Augmented Generation. "
        "We demonstrate that RAG systems experience severe reasoning drift when context contains irrelevant passages."
    )
    _create_synthetic_pdf(
        pdf2,
        "Paper B: Context Pruning and Faithfulness Evaluation Under Context Noise in Scientific QA. "
        "Our experiments confirm that standard dense retrievers introduce distractor documents that corrupt generation."
    )

    manifest = [
        {
            "filename": "paper_a.pdf",
            "title": "Analysis of Distractor Injection in Scientific RAG",
            "publication_date": "2024-03-15",
            "doi": "10.1234/scirag.2024.01",
            "authors": [{"name": "Alice Researcher", "affiliation": "AI Lab"}],
        },
        {
            "filename": "paper_b.pdf",
            "title": "Context Pruning and Faithfulness Evaluation",
            "publication_date": "2024-06-20",
            "doi": "10.1234/context.2024.02",
            "authors": [{"name": "Bob Scientist", "affiliation": "University"}],
        },
    ]
    with open(corpus_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    return corpus_dir


def test_unchanged_complete_discovery_inputs_reuse_permitted(two_paper_corpus: Path, tmp_path: Path):
    """Verifies that running again against the same database reuses Stage 3 gaps when inputs are identical."""
    artifacts_dir = tmp_path / "artifacts_unchanged"
    db_file = artifacts_dir / "experiment_state.db"

    llm = MockExperimentLLMProvider()
    embedder = DeterministicExperimentEmbeddingProvider(64)
    runner = ControlledExperimentRunner(
        llm_provider=llm,
        embedding_provider=embedder,
        artifacts_dir=artifacts_dir,
    )

    # 1. Initial run: Stage 3 discovery executes and records completion metadata
    res1 = runner.run(two_paper_corpus)
    assert res1.success is True
    assert res1.metrics["discovery"]["validated_gaps_count"] >= 1
    assert all(not g["resumed_from_cache"] for g in res1.metrics["discovery"]["validated_gaps"])

    # 2. Re-run: Same corpus and DB
    runner2 = ControlledExperimentRunner(
        llm_provider=llm,
        embedding_provider=embedder,
        artifacts_dir=artifacts_dir,
    )
    res2 = runner2.run(two_paper_corpus)
    assert res2.success is True
    # Reused from cache!
    assert res2.metrics["discovery"]["validated_gaps_count"] == res1.metrics["discovery"]["validated_gaps_count"]
    assert all(g["resumed_from_cache"] for g in res2.metrics["discovery"]["validated_gaps"])


def test_adding_newly_understood_paper_prevents_stale_reuse(two_paper_corpus: Path, tmp_path: Path):
    """Verifies that adding a new paper prevents stale reuse of previously discovered gaps."""
    artifacts_dir = tmp_path / "artifacts_new_paper"
    db_file = artifacts_dir / "experiment_state.db"

    llm = MockExperimentLLMProvider()
    embedder = DeterministicExperimentEmbeddingProvider(64)
    runner = ControlledExperimentRunner(
        llm_provider=llm,
        embedding_provider=embedder,
        artifacts_dir=artifacts_dir,
    )

    # 1. Run with initial 2 papers
    res1 = runner.run(two_paper_corpus)
    assert res1.success is True

    # 2. Add a third paper to the corpus and manifest
    pdf3 = two_paper_corpus / "paper_c.pdf"
    _create_synthetic_pdf(
        pdf3,
        "Paper C: Graph-based distractor pruning in scientific search. "
        "We show that attention graphs effectively prune irrelevant documents from scientific contexts."
    )
    manifest_path = two_paper_corpus / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    manifest.append({
        "filename": "paper_c.pdf",
        "title": "Graph-based Distractor Pruning",
        "publication_date": "2024-08-10",
        "doi": "10.1234/graph.2024.03",
        "authors": [{"name": "Charlie Engineer", "affiliation": "Tech Corp"}],
    })
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    # 3. Second run must fail-closed rather than reusing stale gaps
    runner2 = ControlledExperimentRunner(
        llm_provider=llm,
        embedding_provider=embedder,
        artifacts_dir=artifacts_dir,
    )
    res2 = runner2.run(two_paper_corpus)
    assert res2.success is False
    assert "Cannot safely reuse existing research gaps" in res2.error


def test_retrying_failed_paper_prevents_stale_reuse(two_paper_corpus: Path, tmp_path: Path):
    """Verifies that retrying a previously failed paper prevents stale reuse of partial-corpus gaps."""
    artifacts_dir = tmp_path / "artifacts_retry_stale"

    # LLM fails on paper 1, succeeds on paper 2
    llm1 = MockExperimentLLMProvider(fail_on_understanding_call=1)
    embedder = DeterministicExperimentEmbeddingProvider(64)
    runner1 = ControlledExperimentRunner(
        llm_provider=llm1,
        embedding_provider=embedder,
        artifacts_dir=artifacts_dir,
    )
    res1 = runner1.run(two_paper_corpus)

    # Manually populate completion marker and gaps for paper 2 to simulate gaps existing from 1 paper
    db_file = artifacts_dir / "experiment_state.db"
    engine = create_engine(f"sqlite:///{db_file}")
    with Session(engine) as session:
        p2 = session.query(Paper).filter(Paper.title.like("%Context Pruning%")).first()
        assert p2 is not None
        p2_stmts = session.query(ResearchStatement).filter(ResearchStatement.paper_id == p2.id).all()
        assert len(p2_stmts) > 0

        # Create a gap linked to p2
        gap = ResearchGap(
            id=uuid.uuid4(),
            gap_statement="Gap based solely on Paper B",
            research_problem="Problem B",
            status=GapStatus.VALIDATED.value,
            evidence_count=len(p2_stmts),
        )
        session.add(gap)
        for s in p2_stmts:
            session.add(GapEvidence(
                id=uuid.uuid4(),
                gap=gap,
                statement_id=s.id,
                paper_id=p2.id,
                evidence_role="SUPPORTING",
            ))

        # Record completion signature for Paper B only
        sig = compute_discovery_input_signature({s.id for s in p2_stmts}, cutoff_date=None)
        comp_row = session.query(ExperimentMetadataRecord).filter(
            ExperimentMetadataRecord.key == STAGE_3_DISCOVERY_COMPLETION_KEY
        ).first()
        if comp_row:
            comp_row.value = json.dumps({"status": "COMPLETED", "signature": sig, "candidate_gaps_count": 1})
        else:
            session.add(ExperimentMetadataRecord(
                key=STAGE_3_DISCOVERY_COMPLETION_KEY,
                value=json.dumps({"status": "COMPLETED", "signature": sig, "candidate_gaps_count": 1}),
            ))
        session.commit()

    # Now re-run with retry_failed=True so paper A is understood
    llm2 = MockExperimentLLMProvider()
    runner2 = ControlledExperimentRunner(
        llm_provider=llm2,
        embedding_provider=embedder,
        artifacts_dir=artifacts_dir,
        retry_failed=True,
    )
    res2 = runner2.run(two_paper_corpus)
    # Stale reuse must be prevented because new statements were extracted!
    assert res2.success is False
    assert "Cannot safely reuse existing research gaps" in res2.error


def test_interrupted_discovery_partial_gap_persistence_not_treated_as_complete(two_paper_corpus: Path, tmp_path: Path):
    """Verifies that partial gaps from an interrupted discovery run are rejected fail-closed."""
    artifacts_dir = tmp_path / "artifacts_interrupted_gaps"
    db_file = artifacts_dir / "experiment_state.db"

    llm = MockExperimentLLMProvider()
    embedder = DeterministicExperimentEmbeddingProvider(64)
    runner = ControlledExperimentRunner(
        llm_provider=llm,
        embedding_provider=embedder,
        artifacts_dir=artifacts_dir,
    )

    # Ingest and understand papers
    res1 = runner.run(two_paper_corpus)
    assert res1.success is True

    # Simulate an interrupted run: delete completion marker, leaving gaps in DB
    engine = create_engine(f"sqlite:///{db_file}")
    with Session(engine) as session:
        comp_row = session.query(ExperimentMetadataRecord).filter(
            ExperimentMetadataRecord.key == STAGE_3_DISCOVERY_COMPLETION_KEY
        ).first()
        assert comp_row is not None
        session.delete(comp_row)
        session.commit()

    # Second run against this state must refuse to reuse the unverified/partial gaps
    runner2 = ControlledExperimentRunner(
        llm_provider=llm,
        embedding_provider=embedder,
        artifacts_dir=artifacts_dir,
    )
    res2 = runner2.run(two_paper_corpus)
    assert res2.success is False
    assert "Stage 3 discovery was not marked complete" in res2.error


def test_changed_cutoff_or_statement_set_prevents_reuse(two_paper_corpus: Path, tmp_path: Path):
    """Verifies that altering the statement set or cutoff prevents stale reuse via signature mismatch."""
    artifacts_dir = tmp_path / "artifacts_changed_cutoff"
    db_file = artifacts_dir / "experiment_state.db"

    llm = MockExperimentLLMProvider()
    embedder = DeterministicExperimentEmbeddingProvider(64)

    # Run 1: full pipeline discovery
    runner1 = ControlledExperimentRunner(
        llm_provider=llm,
        embedding_provider=embedder,
        artifacts_dir=artifacts_dir,
    )
    res1 = runner1.run(two_paper_corpus)
    assert res1.success is True

    # Alter the recorded signature in the database to simulate a different statement set or cutoff
    engine = create_engine(f"sqlite:///{db_file}")
    with Session(engine) as session:
        comp_row = session.query(ExperimentMetadataRecord).filter(
            ExperimentMetadataRecord.key == STAGE_3_DISCOVERY_COMPLETION_KEY
        ).first()
        assert comp_row is not None
        data = json.loads(comp_row.value)
        # Point signature to a different cutoff and statement count
        data["signature"] = compute_discovery_input_signature(
            [uuid.uuid4(), uuid.uuid4()],
            cutoff_date="2020-12-31"
        )
        comp_row.value = json.dumps(data)
        session.commit()

    # Re-run with matching runner settings: Stage 3 detects signature mismatch and fails closed
    runner2 = ControlledExperimentRunner(
        llm_provider=llm,
        embedding_provider=embedder,
        artifacts_dir=artifacts_dir,
    )
    res2 = runner2.run(two_paper_corpus)
    assert res2.success is False
    assert "Stage 3 discovery input signature mismatch" in res2.error


def test_successful_full_discovery_records_completion_metadata(two_paper_corpus: Path, tmp_path: Path):
    """Verifies that completing Stage 3 discovery persists completion metadata with valid signature."""
    artifacts_dir = tmp_path / "artifacts_metadata"
    db_file = artifacts_dir / "experiment_state.db"

    llm = MockExperimentLLMProvider()
    embedder = DeterministicExperimentEmbeddingProvider(64)
    runner = ControlledExperimentRunner(
        llm_provider=llm,
        embedding_provider=embedder,
        artifacts_dir=artifacts_dir,
    )

    res = runner.run(two_paper_corpus)
    assert res.success is True

    # Inspect SQLite database directly
    engine = create_engine(f"sqlite:///{db_file}")
    with Session(engine) as session:
        comp_row = session.query(ExperimentMetadataRecord).filter(
            ExperimentMetadataRecord.key == STAGE_3_DISCOVERY_COMPLETION_KEY
        ).first()
        assert comp_row is not None
        data = json.loads(comp_row.value)
        assert data["status"] == "COMPLETED"
        assert "completed_at" in data
        assert "signature" in data
        sig = data["signature"]
        assert "statement_count" in sig
        assert "statement_ids_hash" in sig
        assert "signature_hash" in sig
        assert data["candidate_gaps_count"] >= 0


def test_legacy_database_without_completion_metadata_handled_safely(two_paper_corpus: Path, tmp_path: Path):
    """Verifies that a legacy database with research gaps but no completion marker is safely rejected."""
    artifacts_dir = tmp_path / "artifacts_legacy"
    db_file = artifacts_dir / "experiment_state.db"

    llm = MockExperimentLLMProvider()
    embedder = DeterministicExperimentEmbeddingProvider(64)
    runner = ControlledExperimentRunner(
        llm_provider=llm,
        embedding_provider=embedder,
        artifacts_dir=artifacts_dir,
    )

    # Initial run to generate papers and statements
    res = runner.run(two_paper_corpus)
    assert res.success is True

    # Strip completion record to simulate legacy database
    engine = create_engine(f"sqlite:///{db_file}")
    with Session(engine) as session:
        comp_row = session.query(ExperimentMetadataRecord).filter(
            ExperimentMetadataRecord.key == STAGE_3_DISCOVERY_COMPLETION_KEY
        ).first()
        if comp_row:
            session.delete(comp_row)
            session.commit()

    # Re-run must safely stop rather than assuming legacy gaps are complete
    runner2 = ControlledExperimentRunner(
        llm_provider=llm,
        embedding_provider=embedder,
        artifacts_dir=artifacts_dir,
    )
    res2 = runner2.run(two_paper_corpus)
    assert res2.success is False
    assert "Stage 3 discovery was not marked complete" in res2.error


def test_existing_resume_and_fingerprint_protections_still_work(two_paper_corpus: Path, tmp_path: Path):
    """Verifies that changing pipeline parameters still triggers fingerprint mismatch rejection."""
    artifacts_dir = tmp_path / "artifacts_fingerprint"
    db_file = artifacts_dir / "experiment_state.db"

    llm = MockExperimentLLMProvider()
    embedder = DeterministicExperimentEmbeddingProvider(64)
    runner1 = ControlledExperimentRunner(
        llm_provider=llm,
        embedding_provider=embedder,
        artifacts_dir=artifacts_dir,
        similarity_threshold=0.75,
    )
    res1 = runner1.run(two_paper_corpus)
    assert res1.success is True

    # Re-run with modified similarity_threshold
    runner2 = ControlledExperimentRunner(
        llm_provider=llm,
        embedding_provider=embedder,
        artifacts_dir=artifacts_dir,
        similarity_threshold=0.88,
    )
    res2 = runner2.run(two_paper_corpus)
    assert res2.success is False
    assert "Configuration mismatch detected" in res2.error
