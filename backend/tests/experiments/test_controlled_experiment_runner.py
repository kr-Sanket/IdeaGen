"""Deterministic offline tests for ControlledExperimentRunner.

Verifies:
1. Input directory & manifest validation.
2. Complete SQLite database isolation and absence of PostgreSQL connection.
3. Provenance preservation across all pipeline stages (paper, passages, statements, gaps, evidence).
4. Honest recording of candidate, validated, and rejected gaps.
5. Suppression of opportunity generation when candidate gaps are rejected.
6. Safe database and resource cleanup on success and upon fatal exceptions.
7. Accurate metrics and markdown summary generation without faked outcomes.
8. Proper rejection/failure behavior when input is empty or invalid.
"""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
import sys
import tempfile
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pymupdf
import pytest

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.runner import (
    ControlledExperimentRunner,
    PaperInputMetadata,
    ExperimentResult,
    ExperimentMetadataRecord,
)
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
from ideagen.discovery.embeddings import EmbeddingProvider
from ideagen.discovery.models import (
    GapStatus,
    ValidationVerdict,
    GapValidationResult,
)
from ideagen.generation.generator import LLMCandidateOpportunitySchema
from ideagen.generation.prior_art import LLMPriorArtValidationSchema
from ideagen.llm.provider import LLMProvider
from ideagen.understanding.models import (
    StructuredPaperExtractionOutput,
    RawStatementExtraction,
    StatementType,
)


class DeterministicExperimentEmbeddingProvider(EmbeddingProvider):
    """Reproducible embedding provider for offline runner testing."""

    def __init__(self, dimension: int = 64, model_name: str = "deterministic-test-embedder-64d"):
        self._dim = dimension
        self._model_name = model_name

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)

        vecs = []
        for text in texts:
            seed = sum(ord(c) for c in text) % 10000
            rng = np.random.RandomState(seed)
            base = rng.randn(self._dim).astype(np.float32)
            # Give semantic direction for clustering test statements
            if "hallucination" in text.lower() or "rag" in text.lower():
                base[:10] += 5.0
            norm = np.linalg.norm(base)
            vecs.append(base / (norm if norm > 0 else 1.0))

        return np.array(vecs, dtype=np.float32)


class MockExperimentLLMProvider(LLMProvider):
    """Deterministic LLM Provider dispatching schemas across all 4 pipeline stages."""

    def __init__(
        self,
        validation_verdict: ValidationVerdict = ValidationVerdict.VALID,
        validation_reasoning: str = "Evidence verified across papers.",
        raise_on_understanding: bool = False,
        interrupt_on_understanding_call: Optional[int] = None,
        fail_on_understanding_call: Optional[Union[int, List[int], Tuple[int, ...]]] = None,
        model_name: str = "mock-experiment-qwen3:8b",
        timeout: Optional[float] = None,
        empty_statements: bool = False,
    ):
        self.validation_verdict = validation_verdict
        self.validation_reasoning = validation_reasoning
        self.raise_on_understanding = raise_on_understanding
        self.interrupt_on_understanding_call = interrupt_on_understanding_call
        self.fail_on_understanding_call = fail_on_understanding_call
        self._model_name = model_name
        self.timeout = timeout
        self.empty_statements = empty_statements
        self.opportunity_generation_called = False
        self.understanding_calls = 0

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate_structured(self, prompt: str, response_schema):
        # 1. Understanding stage
        if response_schema == StructuredPaperExtractionOutput:
            self.understanding_calls += 1
            if self.interrupt_on_understanding_call == self.understanding_calls:
                raise KeyboardInterrupt("Simulated SIGINT during paper understanding")
            if self.fail_on_understanding_call is not None:
                if isinstance(self.fail_on_understanding_call, (list, tuple, set)):
                    if self.understanding_calls in self.fail_on_understanding_call:
                        raise RuntimeError("Simulated transient failure on paper understanding")
                elif self.fail_on_understanding_call == self.understanding_calls:
                    raise RuntimeError("Simulated transient failure on paper understanding")
            if self.raise_on_understanding:
                raise RuntimeError("Simulated LLM extraction timeout")
            if self.empty_statements:
                return StructuredPaperExtractionOutput(
                    research_problem="No statements extracted.",
                    research_questions=[],
                    methodology="Empty extraction test.",
                    datasets=[],
                    findings=[],
                    limitations=[],
                    future_work=[],
                    statements=[],
                )
            return StructuredPaperExtractionOutput(
                research_problem="Hallucination accumulation in scientific retrieval-augmented generation.",
                research_questions=["How do distractor documents influence reasoning drift?"],
                methodology="Controlled adversarial context evaluation on synthetic benchmarks.",
                datasets=["SciFact", "PubMedQA"],
                findings=["Distractors increase hallucination rates monotonically beyond 3 passages."],
                limitations=["Only evaluated on biomedical corpus."],
                future_work=["Investigate multi-hop cross-domain generalization."],
                statements=[
                    RawStatementExtraction(
                        statement_text="RAG hallucination rates degrade sharply when distractors exceed 3 documents.",
                        statement_type=StatementType.LIMITATION,
                        passage_index=0,
                        confidence=0.92,
                    ),
                    RawStatementExtraction(
                        statement_text="Context pruning fails to filter distractor passages in complex scientific QA.",
                        statement_type=StatementType.LIMITATION,
                        passage_index=0,
                        confidence=0.88,
                    ),
                ],
            )

        # 2. Gap validation stage (Phase 4)
        if response_schema == GapValidationResult:
            import re
            cited_ids = re.findall(r"\[ID:\s*([0-9a-fA-F-]{36})", prompt) if self.validation_verdict == ValidationVerdict.VALID else []
            return GapValidationResult(
                verdict=self.validation_verdict,
                reasoning=self.validation_reasoning,
                supporting_evidence_statement_ids=cited_ids,
                confidence=0.90 if self.validation_verdict == ValidationVerdict.VALID else 0.20,
            )

        # 3. Opportunity generation (Phase 5)
        if response_schema == LLMCandidateOpportunitySchema:
            self.opportunity_generation_called = True
            return LLMCandidateOpportunitySchema(
                proposed_research_direction="Self-supervised distractor verification for scientific RAG",
                reasoning="Addresses hallucination drift across multi-document contexts.",
                technical_approach="Graph-constrained attention filters for distractor elimination.",
                expected_contribution="Reduces false assertions by over 40% on biomedical benchmarks.",
                required_data="SciFact and OpenAlex corpora",
                required_compute="2x A100 GPUs",
                implementation_complexity="Moderate",
                experimental_setup="Standard RAG evaluation under varied distractor densities.",
                feasibility_score=0.85,
                confidence=0.88,
            )

        # 4. Prior-art assessment (Phase 5)
        if response_schema == LLMPriorArtValidationSchema:
            return LLMPriorArtValidationSchema(
                assessment="LOW_OVERLAP",
                reasoning="Existing approaches do not filter multi-passage distractor graphs.",
                differentiation="Introduces graph topological constraints.",
                overlapping_statement_indices=[],
                confidence=0.91,
            )

        raise ValueError(f"Unexpected response schema in test provider: {response_schema}")


def _create_synthetic_pdf(filepath: Path, content: str) -> Path:
    """Creates a minimal valid PDF with at least 500 characters for ManualPDFIngestionService."""
    doc = pymupdf.open()
    page = doc.new_page()
    # Repeat text to ensure > 500 characters threshold
    full_text = f"{content}\n" * 15
    page.insert_text((50, 50), full_text)
    doc.save(str(filepath))
    doc.close()
    return filepath


@pytest.fixture
def sample_pdf_corpus(tmp_path: Path) -> Path:
    """Creates an input corpus directory with two valid scientific PDFs and a manifest."""
    corpus_dir = tmp_path / "test_corpus"
    corpus_dir.mkdir()

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


# ── TEST SUITE ─────────────────────────────────────────────────────────────


def test_input_validation_nonexistent_directory():
    """Fails cleanly and honestly when input directory does not exist."""
    runner = ControlledExperimentRunner()
    result = runner.run(Path("/nonexistent/directory/path/12345"))
    assert not result.success
    assert "Input directory does not exist" in result.error
    assert result.metrics["error"]


def test_input_validation_empty_directory(tmp_path: Path):
    """Fails cleanly when directory contains no PDFs."""
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    runner = ControlledExperimentRunner()
    result = runner.run(empty_dir)
    assert not result.success
    assert "No PDF files found" in result.error


def test_manifest_parsing_explicit_vs_optional(sample_pdf_corpus: Path):
    """Validates that explicit metadata from manifest is accurately extracted and distinguished."""
    # Add a third PDF without manifest entry
    pdf3 = sample_pdf_corpus / "paper_c.pdf"
    _create_synthetic_pdf(pdf3, "Paper C without explicit manifest metadata for fallback testing.")

    runner = ControlledExperimentRunner()
    papers = runner._resolve_manifest(sample_pdf_corpus)
    assert len(papers) == 3

    meta_by_name = {p.filepath.name: p for p in papers}

    # Explicit
    assert meta_by_name["paper_a.pdf"].is_explicit
    assert meta_by_name["paper_a.pdf"].title == "Analysis of Distractor Injection in Scientific RAG"
    assert meta_by_name["paper_a.pdf"].publication_date == datetime.date(2024, 3, 15)
    assert meta_by_name["paper_a.pdf"].doi == "10.1234/scirag.2024.01"
    assert len(meta_by_name["paper_a.pdf"].authors) == 1

    # Optional / missing
    assert not meta_by_name["paper_c.pdf"].is_explicit
    assert meta_by_name["paper_c.pdf"].title is None
    assert meta_by_name["paper_c.pdf"].publication_date is None
    assert meta_by_name["paper_c.pdf"].doi is None
    assert len(meta_by_name["paper_c.pdf"].authors) == 0


def test_end_to_end_runner_with_mock_providers(sample_pdf_corpus: Path, tmp_path: Path):
    """Verifies end-to-end execution, database isolation, provenance, and metric output."""
    artifacts_dir = tmp_path / "artifacts"
    llm = MockExperimentLLMProvider(validation_verdict=ValidationVerdict.VALID)
    embedder = DeterministicExperimentEmbeddingProvider(dimension=64)

    runner = ControlledExperimentRunner(
        llm_provider=llm,
        embedding_provider=embedder,
        artifacts_dir=artifacts_dir,
        min_supporting_evidence=1,
    )

    result = runner.run(sample_pdf_corpus)

    assert result.success is True, f"Runner failed with error: {result.error}"
    assert result.error is None
    assert artifacts_dir.exists()
    assert (artifacts_dir / "experiment_metrics.json").exists()
    assert (artifacts_dir / "experiment_summary.md").exists()

    # Check metrics structure
    metrics = result.metrics
    assert metrics["metadata"]["database_isolated"] is True
    assert metrics["metadata"]["production_postgresql_connected"] is False
    assert metrics["metadata"]["llm_model"] == "mock-experiment-qwen3:8b"
    assert metrics["metadata"]["embedding_model"] == "deterministic-test-embedder-64d"
    assert metrics["metadata"]["embedding_dimension"] == 64

    # Ingestion metrics
    assert metrics["ingestion"]["total_papers"] == 2
    assert metrics["ingestion"]["success_count"] == 2
    assert metrics["ingestion"]["failed_count"] == 0
    assert metrics["ingestion"]["total_passages_extracted"] >= 2

    # Understanding metrics
    assert metrics["understanding"]["understood_papers_count"] == 2
    assert metrics["understanding"]["total_statements_extracted"] == 4
    assert metrics["understanding"]["corpus_statements_by_type"]["LIMITATION"] == 4
    assert metrics["understanding"]["overall_provenance_validity_rate"] == 1.0

    # Discovery metrics
    assert metrics["discovery"]["candidate_gaps_total"] >= 1
    assert metrics["discovery"]["validated_gaps_count"] >= 1
    assert metrics["discovery"]["rejected_gaps_count"] == 0

    # Opportunity metrics
    assert metrics["opportunities"]["opportunities_count"] >= 1
    assert llm.opportunity_generation_called is True

    # Check markdown summary content
    summary_text = (artifacts_dir / "experiment_summary.md").read_text(encoding="utf-8")
    assert "# IdeaGen Controlled Multi-Paper Experiment Summary" in summary_text
    assert "mock-experiment-qwen3:8b" in summary_text
    assert "**Status**: SUCCESS" in summary_text


def test_rejected_gaps_prevent_opportunity_generation(sample_pdf_corpus: Path, tmp_path: Path):
    """Verifies honest recording of rejected gaps and suppression of opportunity generation."""
    artifacts_dir = tmp_path / "artifacts_rejected"
    # Provide an LLM that rejects candidate gaps
    llm = MockExperimentLLMProvider(
        validation_verdict=ValidationVerdict.REJECT,
        validation_reasoning="Candidate claim is not supported by cited passages.",
    )
    embedder = DeterministicExperimentEmbeddingProvider(dimension=64)

    runner = ControlledExperimentRunner(
        llm_provider=llm,
        embedding_provider=embedder,
        artifacts_dir=artifacts_dir,
        min_supporting_evidence=1,
    )

    result = runner.run(sample_pdf_corpus)

    assert result.success is True
    metrics = result.metrics

    # Discovery must record candidate and rejected gaps honestly
    assert metrics["discovery"]["candidate_gaps_total"] >= 1
    assert metrics["discovery"]["validated_gaps_count"] == 0
    assert metrics["discovery"]["rejected_gaps_count"] >= 1
    assert len(metrics["discovery"]["rejected_gaps"]) >= 1
    assert metrics["discovery"]["rejected_gaps"][0]["status"] == "REJECTED"

    # Opportunity generation MUST NOT be called for rejected gaps
    assert llm.opportunity_generation_called is False
    assert metrics["opportunities"]["opportunities_count"] == 0
    assert metrics["opportunities"]["status"] == "NO_VALIDATED_GAPS"


def test_sqlite_cleanup_on_success(sample_pdf_corpus: Path, tmp_path: Path):
    """Verifies that the temporary SQLite database file is unlinked after successful run."""
    runner = ControlledExperimentRunner(
        llm_provider=MockExperimentLLMProvider(),
        embedding_provider=DeterministicExperimentEmbeddingProvider(64),
    )

    # Spy on temporary DB creation by passing a custom path or letting runner make tempfile
    result = runner.run(sample_pdf_corpus)
    assert result.success is True

    # Check that no lingering ideagen_experiment_*.db files exist in system temp
    temp_dir = Path(tempfile.gettempdir())
    lingering = list(temp_dir.glob("ideagen_experiment_*.db"))
    # None of them should belong to this runner execution
    assert len(lingering) == 0


def test_sqlite_cleanup_on_exception(sample_pdf_corpus: Path, monkeypatch):
    """Verifies that the temporary SQLite database file is cleaned up even if a fatal error occurs."""
    runner = ControlledExperimentRunner(
        llm_provider=MockExperimentLLMProvider(),
        embedding_provider=DeterministicExperimentEmbeddingProvider(64),
    )

    # Force a fatal exception during discovery stage
    from ideagen.discovery.pipeline import SemanticGapDiscoveryPipeline

    def exploding_discover_gaps(self, *args, **kwargs):
        raise RuntimeError("Fatal explosion during gap discovery")

    monkeypatch.setattr(SemanticGapDiscoveryPipeline, "discover_gaps", exploding_discover_gaps)

    result = runner.run(sample_pdf_corpus)
    assert result.success is False
    assert "Fatal explosion during gap discovery" in result.error

    # Ensure no lingering SQLite files
    temp_dir = Path(tempfile.gettempdir())
    lingering = list(temp_dir.glob("ideagen_experiment_*.db"))
    assert len(lingering) == 0


def test_in_memory_sqlite_mode(sample_pdf_corpus: Path):
    """Verifies runner runs cleanly with explicit :memory: SQLite database."""
    runner = ControlledExperimentRunner(
        llm_provider=MockExperimentLLMProvider(),
        embedding_provider=DeterministicExperimentEmbeddingProvider(64),
        db_path=":memory:",
    )

    result = runner.run(sample_pdf_corpus)
    assert result.success is True
    assert result.metrics["metadata"]["database_mode"] == "in-memory"


def test_reusing_already_ingested_pdfs_without_duplicate_records(sample_pdf_corpus: Path, tmp_path: Path):
    """Verifies that running again against a persistent DB reuses existing papers and passages."""
    artifacts_dir = tmp_path / "artifacts_reuse"
    runner = ControlledExperimentRunner(
        llm_provider=MockExperimentLLMProvider(),
        embedding_provider=DeterministicExperimentEmbeddingProvider(64),
        artifacts_dir=artifacts_dir,
    )

    res1 = runner.run(sample_pdf_corpus)
    assert res1.success is True
    assert res1.metrics["ingestion"]["reused_count"] == 0

    db_path = artifacts_dir / "experiment_state.db"
    assert db_path.exists()

    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        paper_count_1 = session.query(Paper).count()
        doc_count_1 = session.query(PaperDocument).count()
        passage_count_1 = session.query(PaperPassage).count()
        assert paper_count_1 == 2
        assert doc_count_1 == 2

    # Second run against the same artifacts directory / DB
    res2 = runner.run(sample_pdf_corpus)
    assert res2.success is True
    assert res2.metrics["ingestion"]["reused_count"] == 2

    with Session(engine) as session:
        assert session.query(Paper).count() == paper_count_1
        assert session.query(PaperDocument).count() == doc_count_1
        assert session.query(PaperPassage).count() == passage_count_1


def test_skipping_successful_paper_understanding_without_llm_call(sample_pdf_corpus: Path, tmp_path: Path):
    """Verifies that completed paper understanding is loaded from cache without calling LLM."""
    artifacts_dir = tmp_path / "artifacts_skip_llm"
    llm1 = MockExperimentLLMProvider()
    runner1 = ControlledExperimentRunner(
        llm_provider=llm1,
        embedding_provider=DeterministicExperimentEmbeddingProvider(64),
        artifacts_dir=artifacts_dir,
    )
    res1 = runner1.run(sample_pdf_corpus)
    assert res1.success is True
    assert llm1.understanding_calls == 2

    llm2 = MockExperimentLLMProvider()
    runner2 = ControlledExperimentRunner(
        llm_provider=llm2,
        embedding_provider=DeterministicExperimentEmbeddingProvider(64),
        artifacts_dir=artifacts_dir,
    )
    res2 = runner2.run(sample_pdf_corpus)
    assert res2.success is True
    # Zero LLM understanding calls made!
    assert llm2.understanding_calls == 0
    assert res2.metrics["understanding"]["resumed_count"] == 2
    assert all(p["resumed_from_cache"] for p in res2.metrics["understanding"]["understood_papers"])


def test_retrying_failed_extraction_safely(sample_pdf_corpus: Path, tmp_path: Path):
    """Verifies that previously failed paper extraction is cleanly retried without constraint violations."""
    artifacts_dir = tmp_path / "artifacts_retry_failed"
    # LLM fails on paper 1, succeeds on paper 2
    llm1 = MockExperimentLLMProvider(fail_on_understanding_call=1)
    runner1 = ControlledExperimentRunner(
        llm_provider=llm1,
        embedding_provider=DeterministicExperimentEmbeddingProvider(64),
        artifacts_dir=artifacts_dir,
    )
    res1 = runner1.run(sample_pdf_corpus)
    assert res1.metrics["understanding"]["failed_count"] == 1
    assert res1.metrics["understanding"]["understood_count"] == 1

    # Second run with healthy LLM provider retries paper 1 and skips paper 2
    llm2 = MockExperimentLLMProvider()
    runner2 = ControlledExperimentRunner(
        llm_provider=llm2,
        embedding_provider=DeterministicExperimentEmbeddingProvider(64),
        artifacts_dir=artifacts_dir,
        retry_failed=True,
    )
    res2 = runner2.run(sample_pdf_corpus)
    assert res2.success is True
    assert llm2.understanding_calls == 1  # Only paper 1 was retried!
    assert res2.metrics["understanding"]["understood_count"] == 2
    assert res2.metrics["understanding"]["failed_count"] == 0


def test_keyboard_interrupt_exports_partial_metrics_and_summary(sample_pdf_corpus: Path, tmp_path: Path):
    """Verifies that KeyboardInterrupt triggers partial metrics export and marks interrupted status."""
    artifacts_dir = tmp_path / "artifacts_interrupt"
    # Interrupt on second paper understanding
    llm = MockExperimentLLMProvider(interrupt_on_understanding_call=2)
    runner = ControlledExperimentRunner(
        llm_provider=llm,
        embedding_provider=DeterministicExperimentEmbeddingProvider(64),
        artifacts_dir=artifacts_dir,
    )

    res = runner.run(sample_pdf_corpus)
    assert res.success is False
    assert res.metrics["metadata"]["interrupted"] is True
    assert "Interrupted by user (SIGINT)" in res.error

    metrics_file = artifacts_dir / "experiment_metrics.json"
    summary_file = artifacts_dir / "experiment_summary.md"
    assert metrics_file.exists()
    assert summary_file.exists()

    summary_text = summary_file.read_text(encoding="utf-8")
    assert "INTERRUPTED" in summary_text

    # First paper succeeded, second paper interrupted
    assert res.metrics["paper_statuses"]["paper_a.pdf"] == "COMPLETED"
    assert res.metrics["paper_statuses"]["paper_b.pdf"] == "INTERRUPTED"


def test_persistent_database_surviving_interruption(sample_pdf_corpus: Path, tmp_path: Path):
    """Verifies that the SQLite database file survives an interruption with partial data intact."""
    artifacts_dir = tmp_path / "artifacts_db_survive"
    llm = MockExperimentLLMProvider(interrupt_on_understanding_call=2)
    runner = ControlledExperimentRunner(
        llm_provider=llm,
        embedding_provider=DeterministicExperimentEmbeddingProvider(64),
        artifacts_dir=artifacts_dir,
    )
    runner.run(sample_pdf_corpus)

    db_file = artifacts_dir / "experiment_state.db"
    assert db_file.exists(), "Persistent SQLite database must not be deleted on interruption"

    engine = create_engine(f"sqlite:///{db_file}")
    with Session(engine) as session:
        # Both papers were ingested into SQLite
        assert session.query(Paper).count() == 2
        # Paper A knowledge was committed
        assert session.query(ResearchKnowledge).count() == 1
        # Paper A statements were committed
        assert session.query(ResearchStatement).count() == 2


def test_resuming_partially_completed_run(sample_pdf_corpus: Path, tmp_path: Path):
    """Verifies that resuming an interrupted run completes all stages without re-running completed papers."""
    artifacts_dir = tmp_path / "artifacts_resume_run"

    # Run 1: Interrupted on paper 2
    llm1 = MockExperimentLLMProvider(interrupt_on_understanding_call=2)
    runner1 = ControlledExperimentRunner(
        llm_provider=llm1,
        embedding_provider=DeterministicExperimentEmbeddingProvider(64),
        artifacts_dir=artifacts_dir,
    )
    res1 = runner1.run(sample_pdf_corpus)
    assert res1.success is False

    # Run 2: Resume with clean provider
    llm2 = MockExperimentLLMProvider()
    runner2 = ControlledExperimentRunner(
        llm_provider=llm2,
        embedding_provider=DeterministicExperimentEmbeddingProvider(64),
        artifacts_dir=artifacts_dir,
    )
    res2 = runner2.run(sample_pdf_corpus)
    assert res2.success is True
    # LLM inference only called for paper 2!
    assert llm2.understanding_calls == 1
    assert res2.metrics["understanding"]["resumed_count"] == 1
    assert res2.metrics["understanding"]["understood_count"] == 2
    assert res2.metrics["opportunities"]["opportunities_count"] >= 1


def test_no_duplicate_statements_gaps_evidence_opportunities(sample_pdf_corpus: Path, tmp_path: Path):
    """Verifies that re-running on a completed database creates zero duplicate rows across all tables."""
    artifacts_dir = tmp_path / "artifacts_dedup"
    runner = ControlledExperimentRunner(
        llm_provider=MockExperimentLLMProvider(),
        embedding_provider=DeterministicExperimentEmbeddingProvider(64),
        artifacts_dir=artifacts_dir,
    )
    res1 = runner.run(sample_pdf_corpus)
    assert res1.success is True

    db_path = artifacts_dir / "experiment_state.db"
    engine = create_engine(f"sqlite:///{db_path}")

    with Session(engine) as session:
        papers_1 = session.query(Paper).count()
        statements_1 = session.query(ResearchStatement).count()
        gaps_1 = session.query(ResearchGap).count()
        evidence_1 = session.query(GapEvidence).count()
        opportunities_1 = session.query(ResearchOpportunityRecord).count()

    # Re-run
    res2 = runner.run(sample_pdf_corpus)
    assert res2.success is True

    with Session(engine) as session:
        assert session.query(Paper).count() == papers_1
        assert session.query(ResearchStatement).count() == statements_1
        assert session.query(ResearchGap).count() == gaps_1
        assert session.query(GapEvidence).count() == evidence_1
        assert session.query(ResearchOpportunityRecord).count() == opportunities_1


def test_corpus_mismatch_detection(sample_pdf_corpus: Path, tmp_path: Path):
    """Verifies that altering the corpus against an existing DB raises a clear CorpusMismatch error."""
    artifacts_dir = tmp_path / "artifacts_mismatch"
    runner = ControlledExperimentRunner(
        llm_provider=MockExperimentLLMProvider(),
        embedding_provider=DeterministicExperimentEmbeddingProvider(64),
        artifacts_dir=artifacts_dir,
    )
    res1 = runner.run(sample_pdf_corpus)
    assert res1.success is True

    # Create a different corpus with a modified paper_a
    mismatched_corpus = tmp_path / "mismatched_corpus"
    mismatched_corpus.mkdir()
    _create_synthetic_pdf(
        mismatched_corpus / "paper_a.pdf",
        "Different content entirely. This should produce a different SHA-256 hash and trigger corpus mismatch."
    )

    res2 = runner.run(mismatched_corpus)
    assert res2.success is False
    assert "Corpus mismatch detected" in res2.error


def test_temporary_database_cleanup_where_applicable(sample_pdf_corpus: Path):
    """Verifies that transient tempfile databases are unlinked when no artifacts_dir or db_path is passed."""
    runner = ControlledExperimentRunner(
        llm_provider=MockExperimentLLMProvider(),
        embedding_provider=DeterministicExperimentEmbeddingProvider(64),
    )
    res = runner.run(sample_pdf_corpus)
    assert res.success is True
    temp_dir = Path(tempfile.gettempdir())
    lingering = list(temp_dir.glob("ideagen_experiment_*.db"))
    assert len(lingering) == 0


def test_matching_configuration_permits_resume(sample_pdf_corpus: Path, tmp_path: Path):
    """Verifies that an identical configuration permits resuming without errors."""
    artifacts_dir = tmp_path / "artifacts_config_match"
    llm = MockExperimentLLMProvider()
    embedder = DeterministicExperimentEmbeddingProvider(64)

    runner1 = ControlledExperimentRunner(
        llm_provider=llm,
        embedding_provider=embedder,
        artifacts_dir=artifacts_dir,
        min_supporting_evidence=1,
        max_passages=12,
        max_total_words=2500,
    )
    res1 = runner1.run(sample_pdf_corpus)
    assert res1.success is True

    runner2 = ControlledExperimentRunner(
        llm_provider=llm,
        embedding_provider=embedder,
        artifacts_dir=artifacts_dir,
        min_supporting_evidence=1,
        max_passages=12,
        max_total_words=2500,
    )
    res2 = runner2.run(sample_pdf_corpus)
    assert res2.success is True
    assert res2.metrics["understanding"]["resumed_count"] == 2


def test_changed_llm_model_rejects_resume(sample_pdf_corpus: Path, tmp_path: Path):
    """Verifies that changing the LLM model rejects resume against an existing database."""
    artifacts_dir = tmp_path / "artifacts_llm_mismatch"
    runner1 = ControlledExperimentRunner(
        llm_provider=MockExperimentLLMProvider(model_name="mock-experiment-qwen3:8b"),
        embedding_provider=DeterministicExperimentEmbeddingProvider(64),
        artifacts_dir=artifacts_dir,
    )
    res1 = runner1.run(sample_pdf_corpus)
    assert res1.success is True

    runner2 = ControlledExperimentRunner(
        llm_provider=MockExperimentLLMProvider(model_name="mock-experiment-qwen3:14b"),
        embedding_provider=DeterministicExperimentEmbeddingProvider(64),
        artifacts_dir=artifacts_dir,
    )
    res2 = runner2.run(sample_pdf_corpus)
    assert res2.success is False
    assert "Configuration mismatch detected" in res2.error
    assert "LLM model mismatch" in res2.error


def test_changed_embedding_model_rejects_resume(sample_pdf_corpus: Path, tmp_path: Path):
    """Verifies that changing the embedding model rejects resume against an existing database."""
    artifacts_dir = tmp_path / "artifacts_embed_mismatch"
    runner1 = ControlledExperimentRunner(
        llm_provider=MockExperimentLLMProvider(),
        embedding_provider=DeterministicExperimentEmbeddingProvider(64, model_name="test-embed-v1"),
        artifacts_dir=artifacts_dir,
    )
    res1 = runner1.run(sample_pdf_corpus)
    assert res1.success is True

    runner2 = ControlledExperimentRunner(
        llm_provider=MockExperimentLLMProvider(),
        embedding_provider=DeterministicExperimentEmbeddingProvider(64, model_name="test-embed-v2"),
        artifacts_dir=artifacts_dir,
    )
    res2 = runner2.run(sample_pdf_corpus)
    assert res2.success is False
    assert "Configuration mismatch detected" in res2.error
    assert "Embedding model mismatch" in res2.error


def test_changed_passage_budget_rejects_resume(sample_pdf_corpus: Path, tmp_path: Path):
    """Verifies that changing the passage word budget rejects resume against an existing database."""
    artifacts_dir = tmp_path / "artifacts_budget_mismatch"
    runner1 = ControlledExperimentRunner(
        llm_provider=MockExperimentLLMProvider(),
        embedding_provider=DeterministicExperimentEmbeddingProvider(64),
        artifacts_dir=artifacts_dir,
        max_total_words=2500,
    )
    res1 = runner1.run(sample_pdf_corpus)
    assert res1.success is True

    runner2 = ControlledExperimentRunner(
        llm_provider=MockExperimentLLMProvider(),
        embedding_provider=DeterministicExperimentEmbeddingProvider(64),
        artifacts_dir=artifacts_dir,
        max_total_words=1500,
    )
    res2 = runner2.run(sample_pdf_corpus)
    assert res2.success is False
    assert "Configuration mismatch detected" in res2.error
    assert "Passage budget max_total_words mismatch" in res2.error


def test_changed_evidence_threshold_rejects_resume(sample_pdf_corpus: Path, tmp_path: Path):
    """Verifies that changing the gap evidence threshold rejects resume against an existing database."""
    artifacts_dir = tmp_path / "artifacts_evidence_mismatch"
    runner1 = ControlledExperimentRunner(
        llm_provider=MockExperimentLLMProvider(),
        embedding_provider=DeterministicExperimentEmbeddingProvider(64),
        artifacts_dir=artifacts_dir,
        min_supporting_evidence=1,
    )
    res1 = runner1.run(sample_pdf_corpus)
    assert res1.success is True

    runner2 = ControlledExperimentRunner(
        llm_provider=MockExperimentLLMProvider(),
        embedding_provider=DeterministicExperimentEmbeddingProvider(64),
        artifacts_dir=artifacts_dir,
        min_supporting_evidence=2,
    )
    res2 = runner2.run(sample_pdf_corpus)
    assert res2.success is False
    assert "Configuration mismatch detected" in res2.error
    assert "Evidence threshold min_supporting_evidence mismatch" in res2.error


def test_missing_or_malformed_stored_metadata_fails_safely(sample_pdf_corpus: Path, tmp_path: Path):
    """Verifies that missing or malformed configuration metadata in an existing DB fails safely."""
    artifacts_dir = tmp_path / "artifacts_metadata_fail"
    runner1 = ControlledExperimentRunner(
        llm_provider=MockExperimentLLMProvider(),
        embedding_provider=DeterministicExperimentEmbeddingProvider(64),
        artifacts_dir=artifacts_dir,
    )
    res1 = runner1.run(sample_pdf_corpus)
    assert res1.success is True

    db_path = artifacts_dir / "experiment_state.db"
    engine = create_engine(f"sqlite:///{db_path}")

    # Part A: Missing stored metadata
    with Session(engine) as session:
        meta_rec = session.query(ExperimentMetadataRecord).filter(ExperimentMetadataRecord.key == "configuration_fingerprint").first()
        if meta_rec:
            session.delete(meta_rec)
            session.commit()

    runner2 = ControlledExperimentRunner(
        llm_provider=MockExperimentLLMProvider(),
        embedding_provider=DeterministicExperimentEmbeddingProvider(64),
        artifacts_dir=artifacts_dir,
    )
    res2 = runner2.run(sample_pdf_corpus)
    assert res2.success is False
    assert "stored configuration fingerprint is missing" in res2.error

    # Part B: Malformed stored metadata
    with Session(engine) as session:
        session.add(ExperimentMetadataRecord(key="configuration_fingerprint", value="{invalid_json: corrupt"))
        session.commit()

    runner3 = ControlledExperimentRunner(
        llm_provider=MockExperimentLLMProvider(),
        embedding_provider=DeterministicExperimentEmbeddingProvider(64),
        artifacts_dir=artifacts_dir,
    )
    res3 = runner3.run(sample_pdf_corpus)
    assert res3.success is False
    assert "malformed" in res3.error


def test_fingerprint_includes_timeout_and_refinement_iterations(sample_pdf_corpus: Path, tmp_path: Path):
    """Verifies that ollama_timeout and max_refinement_iterations are recorded in configuration fingerprint."""
    artifacts_dir = tmp_path / "artifacts_fingerprint_fields"
    llm = MockExperimentLLMProvider(timeout=180.0)
    embedder = DeterministicExperimentEmbeddingProvider(64)

    runner = ControlledExperimentRunner(
        llm_provider=llm,
        embedding_provider=embedder,
        artifacts_dir=artifacts_dir,
        max_refinement_iterations=3,
    )
    res = runner.run(sample_pdf_corpus)
    assert res.success is True

    fp = res.metrics["metadata"]["configuration_fingerprint"]
    assert "ollama_timeout" in fp
    assert fp["ollama_timeout"] == 180.0
    assert "max_refinement_iterations" in fp
    assert fp["max_refinement_iterations"] == 3


def test_changed_ollama_timeout_rejects_resume(sample_pdf_corpus: Path, tmp_path: Path):
    """Verifies that changing the Ollama request timeout rejects resume against an existing database."""
    artifacts_dir = tmp_path / "artifacts_timeout_mismatch"
    runner1 = ControlledExperimentRunner(
        llm_provider=MockExperimentLLMProvider(),
        embedding_provider=DeterministicExperimentEmbeddingProvider(64),
        artifacts_dir=artifacts_dir,
        ollama_timeout=300.0,
    )
    res1 = runner1.run(sample_pdf_corpus)
    assert res1.success is True

    runner2 = ControlledExperimentRunner(
        llm_provider=MockExperimentLLMProvider(),
        embedding_provider=DeterministicExperimentEmbeddingProvider(64),
        artifacts_dir=artifacts_dir,
        ollama_timeout=60.0,
    )
    res2 = runner2.run(sample_pdf_corpus)
    assert res2.success is False
    assert "Configuration mismatch detected" in res2.error
    assert "Ollama timeout mismatch" in res2.error


def test_changed_max_refinement_iterations_rejects_resume(sample_pdf_corpus: Path, tmp_path: Path):
    """Verifies that changing max_refinement_iterations rejects resume against an existing database."""
    artifacts_dir = tmp_path / "artifacts_refine_mismatch"
    runner1 = ControlledExperimentRunner(
        llm_provider=MockExperimentLLMProvider(),
        embedding_provider=DeterministicExperimentEmbeddingProvider(64),
        artifacts_dir=artifacts_dir,
        max_refinement_iterations=2,
    )
    res1 = runner1.run(sample_pdf_corpus)
    assert res1.success is True

    runner2 = ControlledExperimentRunner(
        llm_provider=MockExperimentLLMProvider(),
        embedding_provider=DeterministicExperimentEmbeddingProvider(64),
        artifacts_dir=artifacts_dir,
        max_refinement_iterations=4,
    )
    res2 = runner2.run(sample_pdf_corpus)
    assert res2.success is False
    assert "Configuration mismatch detected" in res2.error
    assert "Opportunity max refinement iterations mismatch" in res2.error


def test_changed_temporal_cutoff_rejects_resume(sample_pdf_corpus: Path, tmp_path: Path):
    """Verifies that changing temporal_cutoff rejects resume against an existing database."""
    artifacts_dir = tmp_path / "artifacts_cutoff_mismatch"
    runner1 = ControlledExperimentRunner(
        llm_provider=MockExperimentLLMProvider(),
        embedding_provider=DeterministicExperimentEmbeddingProvider(64),
        artifacts_dir=artifacts_dir,
        temporal_cutoff="2024-12-31",
    )
    res1 = runner1.run(sample_pdf_corpus)
    assert res1.success is True

    runner2 = ControlledExperimentRunner(
        llm_provider=MockExperimentLLMProvider(),
        embedding_provider=DeterministicExperimentEmbeddingProvider(64),
        artifacts_dir=artifacts_dir,
        temporal_cutoff="2023-12-31",
    )
    res2 = runner2.run(sample_pdf_corpus)
    assert res2.success is False
    assert "Configuration mismatch detected" in res2.error
    assert "Temporal cutoff mismatch" in res2.error


def test_explicit_runner_timeout_updates_provider_timeout(sample_pdf_corpus: Path, tmp_path: Path):
    """Verifies that an explicit runner timeout updates a mock provider's writable timeout."""
    artifacts_dir = tmp_path / "artifacts_provider_timeout_sync"
    llm = MockExperimentLLMProvider(timeout=300.0)
    embedder = DeterministicExperimentEmbeddingProvider(64)

    runner = ControlledExperimentRunner(
        llm_provider=llm,
        embedding_provider=embedder,
        artifacts_dir=artifacts_dir,
        ollama_timeout=45.0,
    )
    res = runner.run(sample_pdf_corpus)
    assert res.success is True
    assert llm.timeout == 45.0
    assert res.metrics["metadata"]["configuration_fingerprint"]["ollama_timeout"] == 45.0


def test_provider_existing_timeout_respected_when_no_runner_timeout(sample_pdf_corpus: Path, tmp_path: Path):
    """Verifies that if no explicit runner timeout is supplied, provider's timeout is preserved and recorded."""
    artifacts_dir = tmp_path / "artifacts_provider_existing_timeout"
    llm = MockExperimentLLMProvider(timeout=150.0)
    embedder = DeterministicExperimentEmbeddingProvider(64)

    runner = ControlledExperimentRunner(
        llm_provider=llm,
        embedding_provider=embedder,
        artifacts_dir=artifacts_dir,
        ollama_timeout=None,
    )
    res = runner.run(sample_pdf_corpus)
    assert res.success is True
    assert llm.timeout == 150.0
    assert res.metrics["metadata"]["configuration_fingerprint"]["ollama_timeout"] == 150.0


def test_provider_without_timeout_attribute_works_safely(sample_pdf_corpus: Path, tmp_path: Path):
    """Verifies that providers without a timeout attribute continue to execute safely."""
    artifacts_dir = tmp_path / "artifacts_no_timeout_attr"
    llm = MockExperimentLLMProvider()
    # Deliberately remove timeout attribute to simulate provider without timeout
    delattr(llm, "timeout")
    assert not hasattr(llm, "timeout")

    embedder = DeterministicExperimentEmbeddingProvider(64)
    runner = ControlledExperimentRunner(
        llm_provider=llm,
        embedding_provider=embedder,
        artifacts_dir=artifacts_dir,
    )
    res = runner.run(sample_pdf_corpus)
    assert res.success is True
    assert "ollama_timeout" in res.metrics["metadata"]["configuration_fingerprint"]
    assert res.metrics["metadata"]["configuration_fingerprint"]["ollama_timeout"] == 300.0


def test_zero_statements_produce_failed_run(sample_pdf_corpus: Path, tmp_path: Path):
    """TASK 3.1: Zero statements extracted across all papers must produce a failed run."""
    artifacts_dir = tmp_path / "artifacts_zero_stmts_fail"
    llm = MockExperimentLLMProvider(empty_statements=True)
    embedder = DeterministicExperimentEmbeddingProvider(64)

    runner = ControlledExperimentRunner(
        llm_provider=llm,
        embedding_provider=embedder,
        artifacts_dir=artifacts_dir,
    )
    res = runner.run(sample_pdf_corpus)
    assert res.success is False
    assert res.error == "Zero statements extracted across all papers."


def test_zero_statements_do_not_produce_100_percent_provenance_validity(sample_pdf_corpus: Path, tmp_path: Path):
    """TASK 3.2: Zero statements extracted must not report 100% provenance validity (must report 0.0)."""
    artifacts_dir = tmp_path / "artifacts_zero_stmts_prov"
    llm = MockExperimentLLMProvider(empty_statements=True)
    embedder = DeterministicExperimentEmbeddingProvider(64)

    runner = ControlledExperimentRunner(
        llm_provider=llm,
        embedding_provider=embedder,
        artifacts_dir=artifacts_dir,
    )
    res = runner.run(sample_pdf_corpus)
    understanding_metrics = res.metrics["understanding"]
    assert understanding_metrics["total_statements_extracted"] == 0
    assert understanding_metrics["overall_provenance_validity_rate"] == 0.0
    assert understanding_metrics["overall_provenance_validity_rate"] != 1.0


def test_zero_statements_failure_reason_appears_in_summary(sample_pdf_corpus: Path, tmp_path: Path):
    """TASK 3.3: Failure reason appears explicitly in metrics and markdown summary."""
    artifacts_dir = tmp_path / "artifacts_zero_stmts_summary"
    llm = MockExperimentLLMProvider(empty_statements=True)
    embedder = DeterministicExperimentEmbeddingProvider(64)

    runner = ControlledExperimentRunner(
        llm_provider=llm,
        embedding_provider=embedder,
        artifacts_dir=artifacts_dir,
    )
    res = runner.run(sample_pdf_corpus)
    assert res.metrics.get("fatal_error") == "Zero statements extracted across all papers."
    assert res.summary_file is not None
    assert res.summary_file.exists()

    summary_text = res.summary_file.read_text(encoding="utf-8")
    assert "- **Status**: FAILED (Zero statements extracted across all papers.)" in summary_text
    assert "- Statements Extracted: **0**" in summary_text
    assert "- Provenance Validity Rate: **0.0%**" in summary_text


def test_successful_non_empty_run_retains_existing_behavior(sample_pdf_corpus: Path, tmp_path: Path):
    """TASK 3.4: A successful non-empty run retains its existing behavior."""
    artifacts_dir = tmp_path / "artifacts_non_empty_success"
    llm = MockExperimentLLMProvider(empty_statements=False)
    embedder = DeterministicExperimentEmbeddingProvider(64)

    runner = ControlledExperimentRunner(
        llm_provider=llm,
        embedding_provider=embedder,
        artifacts_dir=artifacts_dir,
    )
    res = runner.run(sample_pdf_corpus)
    assert res.success is True
    assert res.error is None
    assert res.metrics.get("fatal_error") is None
    assert res.metrics["understanding"]["total_statements_extracted"] > 0
    assert res.metrics["understanding"]["overall_provenance_validity_rate"] > 0.0
    assert res.summary_file is not None
    assert res.summary_file.exists()

    summary_text = res.summary_file.read_text(encoding="utf-8")
    assert "- **Status**: SUCCESS" in summary_text


def test_runner_prior_art_vector_index_temporal_isolation(sample_pdf_corpus: Path, tmp_path: Path):
    """Priority 1 & 4: Verifies prior-art vector index only contains eligible statements and metrics distinguish counts."""
    # Add a third future paper (paper_c, 2024-09-01) to sample_pdf_corpus
    pdf3 = sample_pdf_corpus / "paper_c.pdf"
    _create_synthetic_pdf(
        pdf3,
        "Paper C: Future 2024 Innovations in Retrieval Augmented Generation and Pruning. "
        "Here we show that future methods from late 2024 solve earlier limitations completely."
    )
    manifest_path = sample_pdf_corpus / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.append({
        "filename": "paper_c.pdf",
        "title": "Future 2024 Innovations in RAG",
        "publication_date": "2024-09-01",
        "doi": "10.1234/future.2024.03",
        "authors": [{"name": "Carol Future", "affiliation": "NextGen AI"}],
    })
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    artifacts_dir = tmp_path / "artifacts_temporal_isolation"
    db_file = tmp_path / "temporal_isolation.db"
    llm = MockExperimentLLMProvider(empty_statements=False)
    embedder = DeterministicExperimentEmbeddingProvider(64)

    # Cutoff at 2024-07-01: paper_a (2024-03-15) & paper_b (2024-06-20) are eligible; paper_c (2024-09-01) is future
    runner = ControlledExperimentRunner(
        llm_provider=llm,
        embedding_provider=embedder,
        db_path=db_file,
        artifacts_dir=artifacts_dir,
        temporal_cutoff="2024-07-01",
    )
    res = runner.run(sample_pdf_corpus)
    assert res.success is True

    # 1. Understanding metrics distinguish total from eligible
    under = res.metrics["understanding"]
    assert under["total_statements_extracted"] == 6
    assert under["eligible_statements_extracted"] == 4
    assert under["temporal_cutoff"] == "2024-07-01"

    # 2. Markdown summary reports both
    assert res.summary_file is not None
    summary_text = res.summary_file.read_text(encoding="utf-8")
    assert "- Statements Extracted: **6**" in summary_text
    assert "- Statements Eligible for Cutoff (2024-07-01): **4**" in summary_text

    # 3. Verify in database: paper_c statements NEVER appear in generated opportunity prior art
    engine = create_engine(f"sqlite:///{db_file}")
    with Session(engine) as session:
        paper_c = session.query(Paper).filter(Paper.title.like("%Future 2024 Innovations%")).first()
        assert paper_c is not None
        paper_c_stmt_ids = {s.id for s in session.query(ResearchStatement).filter(ResearchStatement.paper_id == paper_c.id).all()}
        assert len(paper_c_stmt_ids) == 2

        opp_records = session.query(ResearchOpportunityRecord).all()
        assert len(opp_records) > 0
        for opp in opp_records:
            opp_prior_ids = set(opp.prior_art_statement_ids or [])
            # Assert complete absence of future statement IDs in opportunity prior art
            assert not (opp_prior_ids & paper_c_stmt_ids), "Future paper statements leaked into opportunity prior art!"


def test_runner_zero_eligible_statements_fails_cleanly(sample_pdf_corpus: Path, tmp_path: Path):
    """Priority 1 & 4: Zero eligible statements for cutoff fails cleanly and reports in summary."""
    artifacts_dir = tmp_path / "artifacts_zero_eligible"
    llm = MockExperimentLLMProvider(empty_statements=False)
    embedder = DeterministicExperimentEmbeddingProvider(64)

    # All papers in sample corpus are in 2024; cutoff at 2023-12-31 leaves 0 eligible statements
    runner = ControlledExperimentRunner(
        llm_provider=llm,
        embedding_provider=embedder,
        artifacts_dir=artifacts_dir,
        temporal_cutoff="2023-12-31",
    )
    res = runner.run(sample_pdf_corpus)
    assert res.success is False
    assert "Zero eligible statements found for temporal cutoff 2023-12-31." in res.error
    assert res.metrics.get("fatal_error") == "Zero eligible statements found for temporal cutoff 2023-12-31."

    assert res.summary_file is not None
    summary_text = res.summary_file.read_text(encoding="utf-8")
    assert "- **Status**: FAILED (Zero eligible statements found for temporal cutoff 2023-12-31.)" in summary_text
    assert "- Statements Extracted: **4**" in summary_text
    assert "- Statements Eligible for Cutoff (2023-12-31): **0**" in summary_text





