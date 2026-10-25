"""Deterministic offline tests for Stage 5: Temporal Opportunity Evolution in ControlledExperimentRunner.

Verifies:
1. Stage 5 is disabled by default, and existing Stage 1–4 behavior remains unchanged.
2. Chronological batch processing with deterministic ordering.
3. Strict future-data isolation: papers published after a cutoff cannot affect that cutoff.
4. Undated papers are excluded whenever cutoffs are active.
5. Opportunities generated after a cutoff are unavailable to that historical step.
6. Repeated batches do not duplicate evidence or transition history (replay safety).
7. Resumption rejects mismatched configuration fingerprints (e.g. Stage 5 enabled flag).
8. Failed Stage 5 runs are marked as failed (success=False) in manifest and metrics.
9. Stage 5 metrics distinguish descriptive event counts from quality/accuracy claims.
10. Entire evaluation runs offline without live models, Ollama, or external networks.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
import sys
import tempfile
import uuid
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pymupdf
import pytest

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.runner import (
    ControlledExperimentRunner,
    PaperInputMetadata,
    ExperimentResult,
    is_opportunity_available_at_cutoff,
    build_temporal_batches,
)
from ideagen.db.models import (
    Base,
    Paper,
    PaperPassage,
    Opportunity,
    OpportunityEvidence,
    OpportunityHistory,
    ResearchOpportunityRecord,
)
from ideagen.discovery.models import ValidationVerdict, GapValidationResult
from ideagen.generation.generator import LLMCandidateOpportunitySchema
from ideagen.generation.prior_art import LLMPriorArtValidationSchema
from ideagen.llm.provider import LLMProvider
from ideagen.opportunity.classifier import RelationshipClassification
from ideagen.opportunity.models import OpportunityRelationship, OpportunityStatus
from ideagen.opportunity.temporal import TemporalBatch
from ideagen.understanding.models import (
    StructuredPaperExtractionOutput,
    RawStatementExtraction,
    StatementType,
)
from tests.experiments.test_controlled_experiment_runner import (
    DeterministicExperimentEmbeddingProvider,
    _create_synthetic_pdf,
)


class Stage5MockLLMProvider(LLMProvider):
    """Deterministic LLM Provider supporting Stages 1–5 offline."""

    def __init__(
        self,
        relationship: OpportunityRelationship = OpportunityRelationship.SUPPORTS,
        evidence_excerpt: str = "Retrieval-augmented generation enables large language models to ground their reasoning in external knowledge",
        fail_stage_5: bool = False,
        confidence: float = 0.88,
        model_name: str = "mock-stage5-llm:8b",
        custom_stage_5_handler: Optional[Callable[[str, int], RelationshipClassification]] = None,
    ):
        self.relationship = relationship
        self.evidence_excerpt = evidence_excerpt
        self.fail_stage_5 = fail_stage_5
        self.confidence = confidence
        self._model_name = model_name
        self.custom_stage_5_handler = custom_stage_5_handler
        self.stage_5_calls = 0

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate_structured(self, prompt: str, response_schema):
        # 1. Understanding stage
        if response_schema == StructuredPaperExtractionOutput:
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

        # 2. Gap validation stage
        if response_schema == GapValidationResult:
            import re
            cited_ids = re.findall(r"\[ID:\s*([0-9a-fA-F-]{36})", prompt)
            return GapValidationResult(
                verdict=ValidationVerdict.VALID,
                reasoning="Gap confirmed across literature evidence.",
                supporting_evidence_statement_ids=cited_ids,
                confidence=0.90,
            )

        # 3. Opportunity generation
        if response_schema == LLMCandidateOpportunitySchema:
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

        # 4. Prior-art assessment
        if response_schema == LLMPriorArtValidationSchema:
            return LLMPriorArtValidationSchema(
                assessment="LOW_OVERLAP",
                reasoning="Existing approaches do not filter multi-passage distractor graphs.",
                differentiation="Introduces graph topological constraints.",
                overlapping_statement_indices=[],
                confidence=0.91,
            )

        # 5. Stage 5 relationship classification
        if response_schema == RelationshipClassification:
            self.stage_5_calls += 1
            if self.custom_stage_5_handler is not None:
                return self.custom_stage_5_handler(prompt, self.stage_5_calls)
            if self.fail_stage_5:
                raise RuntimeError("Simulated Stage 5 classifier crash")

            return RelationshipClassification(
                relationship=self.relationship,
                confidence=self.confidence,
                rationale="Passage provides grounded evidence regarding research opportunity.",
                evidence_excerpt=self.evidence_excerpt,
                prompt_version="v1",
            )

        raise ValueError(f"Unexpected response schema in test provider: {response_schema}")


@pytest.fixture
def temporal_pdf_corpus(tmp_path: Path) -> Path:
    """Creates a synthetic PDF corpus with papers published at different dates."""
    corpus_dir = tmp_path / "temporal_corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    text_content = (
        "Retrieval-augmented generation enables large language models to ground their reasoning in external knowledge. "
        "However, distractors present in multi-document contexts cause severe factual degradation and reasoning drift. "
        "RAG hallucination rates degrade sharply when distractors exceed 3 documents. "
        "Context pruning fails to filter distractor passages in complex scientific QA. "
        "Further research is required to guarantee reliable factual synthesis under adversarial distraction."
    )

    p1 = corpus_dir / "paper_2022_baseline_a.pdf"
    p2 = corpus_dir / "paper_2022_baseline_b.pdf"
    p3 = corpus_dir / "paper_2023_batch1.pdf"
    p4 = corpus_dir / "paper_2024_batch2.pdf"
    p5 = corpus_dir / "paper_undated.pdf"

    _create_synthetic_pdf(p1, f"Paper 2022 Baseline A\n\n{text_content}")
    _create_synthetic_pdf(p2, f"Paper 2022 Baseline B\n\n{text_content}")
    _create_synthetic_pdf(p3, f"Paper 2023 Batch 1\n\n{text_content}")
    _create_synthetic_pdf(p4, f"Paper 2024 Batch 2\n\n{text_content}")
    _create_synthetic_pdf(p5, f"Paper Undated\n\n{text_content}")

    manifest = [
        {"filename": "paper_2022_baseline_a.pdf", "title": "Paper 2022 Baseline A", "publication_date": "2022-03-01"},
        {"filename": "paper_2022_baseline_b.pdf", "title": "Paper 2022 Baseline B", "publication_date": "2022-09-01"},
        {"filename": "paper_2023_batch1.pdf", "title": "Paper 2023 Batch 1", "publication_date": "2023-06-01"},
        {"filename": "paper_2024_batch2.pdf", "title": "Paper 2024 Batch 2", "publication_date": "2024-06-01"},
        {"filename": "paper_undated.pdf", "title": "Paper Undated", "publication_date": None},
    ]

    with open(corpus_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return corpus_dir


def test_stage_5_disabled_by_default(temporal_pdf_corpus: Path, tmp_path: Path):
    """Verifies that Stage 5 is disabled by default and Stage 1-4 behavior remains unchanged."""
    artifacts_dir = tmp_path / "artifacts_default"
    provider = Stage5MockLLMProvider()
    embedder = DeterministicExperimentEmbeddingProvider(64)

    runner = ControlledExperimentRunner(
        llm_provider=provider,
        embedding_provider=embedder,
        artifacts_dir=artifacts_dir,
        temporal_cutoff="2022-12-31",
    )
    assert runner.enable_stage_5 is False

    res = runner.run(temporal_pdf_corpus)
    assert res.success is True
    assert provider.stage_5_calls == 0

    s5_metrics = res.metrics.get("stage_5_evolution", {})
    assert s5_metrics.get("enabled") is False
    assert s5_metrics.get("status") == "DISABLED"

    # Stage 1-4 metrics are present
    assert res.metrics["ingestion"]["ingested_count"] == 5
    assert res.metrics["understanding"]["understood_count"] == 5
    assert res.metrics["discovery"]["validated_gaps_count"] >= 1
    assert res.metrics["opportunities"]["opportunities_count"] >= 1


def test_stage_5_chronological_batches_and_transitions(temporal_pdf_corpus: Path, tmp_path: Path):
    """Verifies Stage 5 evaluates opportunities across chronological batches and records transitions."""
    artifacts_dir = tmp_path / "artifacts_stage5"
    provider = Stage5MockLLMProvider(
        relationship=OpportunityRelationship.SUPPORTS,
        evidence_excerpt="Retrieval-augmented generation enables large language models to ground their reasoning in external knowledge",
    )
    embedder = DeterministicExperimentEmbeddingProvider(64)

    runner = ControlledExperimentRunner(
        llm_provider=provider,
        embedding_provider=embedder,
        artifacts_dir=artifacts_dir,
        temporal_cutoff="2022-12-31",
        enable_stage_5=True,
        stage_5_cutoffs=["2023-12-31", "2024-12-31"],
    )
    assert runner.enable_stage_5 is True

    res = runner.run(temporal_pdf_corpus)
    assert res.success is True
    assert provider.stage_5_calls > 0

    s5 = res.metrics["stage_5_evolution"]
    assert s5["enabled"] is True
    assert s5["status"] == "COMPLETED"
    assert s5["batches_evaluated_count"] == 2
    assert s5["total_events_count"] >= 1

    per_batch = s5["per_batch_metrics"]
    assert len(per_batch) == 2

    # Batch 1 (cutoff 2023-12-31): paper_2023_batch1 is eligible
    b1 = per_batch[0]
    assert b1["cutoff_date"] == "2023-12-31"
    assert b1["eligible_paper_count"] == 1
    assert b1["processed_papers"] == b1["opportunities_evaluated"] * b1["eligible_paper_count"]
    assert b1["relationship_counts"]["SUPPORTS"] >= 1
    assert b1["status_transitions_count"] >= 1

    # Batch 2 (cutoff 2024-12-31): paper_2024_batch2 is eligible
    b2 = per_batch[1]
    assert b2["cutoff_date"] == "2024-12-31"
    assert b2["eligible_paper_count"] == 1
    assert b2["processed_papers"] == b2["opportunities_evaluated"] * b2["eligible_paper_count"]


def test_future_dated_papers_strictly_isolated_from_earlier_cutoff(temporal_pdf_corpus: Path, tmp_path: Path):
    """Verifies that future papers cannot influence earlier cutoff evaluation."""
    artifacts_dir = tmp_path / "artifacts_future_isolation"
    provider = Stage5MockLLMProvider(
        relationship=OpportunityRelationship.SUPPORTS,
        evidence_excerpt="Retrieval-augmented generation enables large language models to ground their reasoning in external knowledge",
    )
    embedder = DeterministicExperimentEmbeddingProvider(64)

    runner = ControlledExperimentRunner(
        llm_provider=provider,
        embedding_provider=embedder,
        artifacts_dir=artifacts_dir,
        temporal_cutoff="2022-12-31",
        enable_stage_5=True,
        stage_5_cutoffs=["2023-12-31", "2024-12-31"],
    )

    res = runner.run(temporal_pdf_corpus)
    assert res.success is True

    s5 = res.metrics["stage_5_evolution"]
    b1 = s5["per_batch_metrics"][0]

    # In Batch 1 (cutoff 2023-12-31): paper_2024_batch2 is in the future
    assert b1["future_papers_excluded_count"] >= 1
    # Check that no transition in batch 1 cites a 2024 paper
    for tr in b1["status_transitions"]:
        assert tr["publication_date"] <= "2023-12-31"


def test_undated_papers_excluded_under_active_cutoff(temporal_pdf_corpus: Path, tmp_path: Path):
    """Verifies that undated papers are strictly excluded from Stage 5 evaluation under active cutoffs."""
    artifacts_dir = tmp_path / "artifacts_undated"
    provider = Stage5MockLLMProvider()
    embedder = DeterministicExperimentEmbeddingProvider(64)

    runner = ControlledExperimentRunner(
        llm_provider=provider,
        embedding_provider=embedder,
        artifacts_dir=artifacts_dir,
        temporal_cutoff="2022-12-31",
        enable_stage_5=True,
        stage_5_cutoffs=["2023-12-31"],
    )

    res = runner.run(temporal_pdf_corpus)
    assert res.success is True

    b1 = res.metrics["stage_5_evolution"]["per_batch_metrics"][0]
    assert b1["undated_papers_excluded_count"] == 1
    # Ensure paper_undated was not processed
    for tr in b1["status_transitions"]:
        assert tr["publication_date"] is not None


def test_opportunities_generated_after_cutoff_unavailable_to_historical_step(tmp_path: Path):
    """Verifies that opportunities created after a cutoff date are unavailable to that historical step."""
    opp_future = Opportunity(
        id=uuid.uuid4(),
        title="Future Opportunity",
        problem_statement="Problem Statement",
        description="Description",
        status=OpportunityStatus.OPEN.value,
        confidence=0.8,
        created_at=datetime.datetime(2025, 6, 1, 0, 0, 0),
    )
    opp_past = Opportunity(
        id=uuid.uuid4(),
        title="Past Opportunity",
        problem_statement="Problem Statement",
        description="Description",
        status=OpportunityStatus.OPEN.value,
        confidence=0.8,
        created_at=datetime.datetime(2023, 1, 1, 0, 0, 0),
    )

    cutoff = datetime.date(2023, 12, 31)
    assert is_opportunity_available_at_cutoff(opp_past, cutoff) is True
    assert is_opportunity_available_at_cutoff(opp_future, cutoff) is False


def test_repeated_batches_do_not_duplicate_evidence_or_history(temporal_pdf_corpus: Path, tmp_path: Path):
    """Verifies that re-running or replaying the same batch skips already-evaluated pairs and does not duplicate evidence."""
    db_file = tmp_path / "persistent_replay.db"
    artifacts_dir = tmp_path / "artifacts_replay"
    provider = Stage5MockLLMProvider(
        relationship=OpportunityRelationship.SUPPORTS,
        evidence_excerpt="Retrieval-augmented generation enables large language models to ground their reasoning in external knowledge",
    )
    embedder = DeterministicExperimentEmbeddingProvider(64)

    runner = ControlledExperimentRunner(
        llm_provider=provider,
        embedding_provider=embedder,
        db_path=db_file,
        artifacts_dir=artifacts_dir,
        temporal_cutoff="2022-12-31",
        enable_stage_5=True,
        stage_5_cutoffs=["2023-12-31"],
    )

    res1 = runner.run(temporal_pdf_corpus)
    assert res1.success is True
    initial_processed = res1.metrics["stage_5_evolution"]["per_batch_metrics"][0]["processed_papers"]
    assert initial_processed >= 1

    # Connect to db to verify evidence count
    engine = create_engine(f"sqlite:///{db_file}")
    with Session(engine) as session:
        ev_count1 = session.query(OpportunityEvidence).count()
        hist_count1 = session.query(OpportunityHistory).count()
        assert ev_count1 >= 1

    # Re-run with identical configuration and database
    runner2 = ControlledExperimentRunner(
        llm_provider=provider,
        embedding_provider=embedder,
        db_path=db_file,
        artifacts_dir=artifacts_dir,
        temporal_cutoff="2022-12-31",
        enable_stage_5=True,
        stage_5_cutoffs=["2023-12-31"],
    )
    res2 = runner2.run(temporal_pdf_corpus)
    assert res2.success is True

    # Re-running the batch skips already recorded (opportunity, paper) pairs
    b1_re = res2.metrics["stage_5_evolution"]["per_batch_metrics"][0]
    assert b1_re["processed_papers"] == 0
    assert b1_re["skipped_papers"] >= 1

    with Session(engine) as session:
        ev_count2 = session.query(OpportunityEvidence).count()
        hist_count2 = session.query(OpportunityHistory).count()
        assert ev_count2 == ev_count1
        assert hist_count2 == hist_count1


def test_resume_rejects_stage_5_mismatched_configuration(temporal_pdf_corpus: Path, tmp_path: Path):
    """Verifies that changing Stage 5 configuration rejects resumption against an existing database."""
    db_file = tmp_path / "config_mismatch.db"
    artifacts_dir = tmp_path / "artifacts_config"
    provider = Stage5MockLLMProvider()
    embedder = DeterministicExperimentEmbeddingProvider(64)

    # Initial run: Stage 5 disabled
    runner1 = ControlledExperimentRunner(
        llm_provider=provider,
        embedding_provider=embedder,
        db_path=db_file,
        artifacts_dir=artifacts_dir,
        enable_stage_5=False,
    )
    res1 = runner1.run(temporal_pdf_corpus)
    assert res1.success is True

    # Attempt resume: Stage 5 enabled
    runner2 = ControlledExperimentRunner(
        llm_provider=provider,
        embedding_provider=embedder,
        db_path=db_file,
        artifacts_dir=artifacts_dir,
        enable_stage_5=True,
    )
    res2 = runner2.run(temporal_pdf_corpus)
    assert res2.success is False
    assert "Configuration mismatch detected" in res2.error
    assert "Stage 5 enabled mismatch" in res2.error


def test_failed_stage_5_marked_failed_rather_than_successful(temporal_pdf_corpus: Path, tmp_path: Path):
    """Verifies that unhandled Stage 5 errors explicitly mark the run as failed in result and manifest."""
    artifacts_dir = tmp_path / "artifacts_fail"
    provider = Stage5MockLLMProvider(fail_stage_5=True)
    embedder = DeterministicExperimentEmbeddingProvider(64)

    runner = ControlledExperimentRunner(
        llm_provider=provider,
        embedding_provider=embedder,
        artifacts_dir=artifacts_dir,
        temporal_cutoff="2022-12-31",
        enable_stage_5=True,
        stage_5_cutoffs=["2023-12-31"],
    )

    res = runner.run(temporal_pdf_corpus)
    assert res.success is False
    assert "Stage 5 evolution failed" in res.error
    assert res.metrics["stage_5_evolution"]["status"] == "FAILED"
    assert "Simulated Stage 5 classifier crash" in res.metrics["stage_5_evolution"]["error"]

    # Verify metrics JSON also records the failure
    metrics_file = artifacts_dir / "experiment_metrics.json"
    assert metrics_file.exists()
    with open(metrics_file, "r", encoding="utf-8") as f:
        saved_metrics = json.load(f)
    assert saved_metrics["stage_5_evolution"]["status"] == "FAILED"


def test_stage_5_metrics_distinguish_counts_from_quality_claims(temporal_pdf_corpus: Path, tmp_path: Path):
    """Verifies that Stage 5 metrics strictly report descriptive counts and do not fabricate quality or accuracy claims."""
    artifacts_dir = tmp_path / "artifacts_quality_distinction"
    provider = Stage5MockLLMProvider(
        relationship=OpportunityRelationship.SUPPORTS,
        evidence_excerpt="Retrieval-augmented generation enables large language models to ground their reasoning in external knowledge",
    )
    embedder = DeterministicExperimentEmbeddingProvider(64)

    runner = ControlledExperimentRunner(
        llm_provider=provider,
        embedding_provider=embedder,
        artifacts_dir=artifacts_dir,
        temporal_cutoff="2022-12-31",
        enable_stage_5=True,
        stage_5_cutoffs=["2023-12-31"],
    )

    res = runner.run(temporal_pdf_corpus)
    assert res.success is True

    s5 = res.metrics["stage_5_evolution"]
    # Check that interpretation note explicitly disclaims empirical accuracy
    note = s5.get("metric_interpretation_note", "")
    assert "descriptive" in note.lower()
    assert "gold standard" in note.lower()

    # Check prohibited quality claim keys are not present
    prohibited_keys = {"accuracy", "precision", "recall", "f1_score", "effectiveness", "validity_rate"}
    assert prohibited_keys.isdisjoint(s5.keys())

    b1 = s5["per_batch_metrics"][0]
    assert prohibited_keys.isdisjoint(b1.keys())

    # Check that legitimate descriptive count keys are present
    assert "status_transitions_count" in b1
    assert "relationship_counts" in b1
    assert "eligible_paper_count" in b1
    assert "excluded_paper_count" in b1
    assert "grounding_rejections_count" in b1


def test_build_temporal_batches_helper_logic():
    """Verifies deterministic batch creation from cutoffs, window days, and distinct dates."""
    p1 = Paper(id=uuid.uuid4(), title="P1", publication_date=datetime.date(2023, 1, 15))
    p2 = Paper(id=uuid.uuid4(), title="P2", publication_date=datetime.date(2023, 6, 20))
    p3 = Paper(id=uuid.uuid4(), title="P3", publication_date=datetime.date(2024, 3, 10))
    p_undated = Paper(id=uuid.uuid4(), title="PU", publication_date=None)

    papers = [p3, p1, p_undated, p2]

    # 1. Explicit cutoffs
    batches_cutoffs = build_temporal_batches(
        papers,
        baseline_cutoff=datetime.date(2022, 12, 31),
        explicit_cutoffs=["2023-12-31", "2024-12-31"],
    )
    assert len(batches_cutoffs) == 2
    assert batches_cutoffs[0].start_date == datetime.date(2023, 1, 1)
    assert batches_cutoffs[0].end_date == datetime.date(2023, 12, 31)
    assert batches_cutoffs[1].start_date == datetime.date(2024, 1, 1)
    assert batches_cutoffs[1].end_date == datetime.date(2024, 12, 31)

    # 2. Window days
    batches_window = build_temporal_batches(
        papers,
        baseline_cutoff=datetime.date(2022, 12, 31),
        batch_window_days=180,
    )
    assert len(batches_window) >= 2
    assert batches_window[0].start_date == datetime.date(2023, 1, 1)

    # 3. Distinct dates default
    batches_dates = build_temporal_batches(
        papers,
        baseline_cutoff=datetime.date(2022, 12, 31),
    )
    assert len(batches_dates) == 3
    assert batches_dates[0].end_date == datetime.date(2023, 1, 15)
    assert batches_dates[1].end_date == datetime.date(2023, 6, 20)
    assert batches_dates[2].end_date == datetime.date(2024, 3, 10)


def test_stage5_batch_durability_persists_evidence(temporal_pdf_corpus: Path, tmp_path: Path):
    """Verifies that successfully processed batches persist evidence durably in the database."""
    artifacts_dir = tmp_path / "artifacts_durability"
    db_path = tmp_path / "durability_state.db"
    provider = Stage5MockLLMProvider(
        relationship=OpportunityRelationship.SUPPORTS,
        evidence_excerpt="Retrieval-augmented generation enables large language models to ground their reasoning in external knowledge",
    )
    embedder = DeterministicExperimentEmbeddingProvider(64)

    runner = ControlledExperimentRunner(
        llm_provider=provider,
        embedding_provider=embedder,
        artifacts_dir=artifacts_dir,
        db_path=str(db_path),
        temporal_cutoff="2022-12-31",
        enable_stage_5=True,
        stage_5_cutoffs=["2023-12-31", "2024-12-31"],
    )

    res = runner.run(temporal_pdf_corpus)
    assert res.success is True
    assert db_path.exists()

    # Open fresh independent connection to database file to verify durable persistence
    verify_engine = create_engine(f"sqlite:///{db_path}")
    with Session(verify_engine) as verify_session:
        evidence_rows = verify_session.query(OpportunityEvidence).all()
        assert len(evidence_rows) >= 2, f"Expected at least 2 persisted evidence rows, got {len(evidence_rows)}"
        opps = verify_session.query(Opportunity).all()
        assert len(opps) >= 1
    verify_engine.dispose()


def test_stage5_failed_pair_leaves_no_partial_writes(temporal_pdf_corpus: Path, tmp_path: Path):
    """Verifies that a failed pair rolls back its savepoint and leaves zero partial writes."""
    artifacts_dir = tmp_path / "artifacts_failed_pair"
    db_path = tmp_path / "failed_pair_state.db"

    # Make the second stage 5 call raise an unhandled error
    def handler(prompt: str, call_count: int):
        if call_count >= 2:
            raise RuntimeError("Fatal unhandled pair crash")
        return RelationshipClassification(
            relationship=OpportunityRelationship.SUPPORTS,
            confidence=0.88,
            rationale="Passage provides grounded evidence regarding research opportunity.",
            evidence_excerpt="Retrieval-augmented generation enables large language models to ground their reasoning in external knowledge",
            prompt_version="v1",
        )

    provider = Stage5MockLLMProvider(custom_stage_5_handler=handler)
    embedder = DeterministicExperimentEmbeddingProvider(64)

    runner = ControlledExperimentRunner(
        llm_provider=provider,
        embedding_provider=embedder,
        artifacts_dir=artifacts_dir,
        db_path=str(db_path),
        temporal_cutoff="2022-12-31",
        enable_stage_5=True,
        stage_5_cutoffs=["2023-12-31", "2024-12-31"],
    )

    res = runner.run(temporal_pdf_corpus)
    # Batch 1 succeeded, Batch 2 failed on pair 2
    assert res.success is False
    assert "Stage 5 evolution failed" in res.error

    # Verify that Batch 1 committed cleanly, and Batch 2's failed pair left no partial records
    verify_engine = create_engine(f"sqlite:///{db_path}")
    with Session(verify_engine) as verify_session:
        evidence_rows = verify_session.query(OpportunityEvidence).all()
        # Exactly 1 evidence row from Batch 1; zero partial rows from Batch 2
        assert len(evidence_rows) == 1
    verify_engine.dispose()


def test_stage5_later_batch_proceeds_after_recoverable_pair_failure(temporal_pdf_corpus: Path, tmp_path: Path):
    """Verifies that a recoverable pair failure (ungrounded evidence) discards the transition,
    persists valid work, and permits subsequent batches to proceed."""
    artifacts_dir = tmp_path / "artifacts_recoverable"
    db_path = tmp_path / "recoverable_state.db"

    # Calls 1 & 2 (Batch 1): ungrounded excerpt (recoverable failure for both opportunities)
    # Calls 3 & 4 (Batch 2): grounded excerpt
    def handler(prompt: str, call_count: int):
        if call_count <= 2:
            return RelationshipClassification(
                relationship=OpportunityRelationship.SUPPORTS,
                confidence=0.90,
                rationale="Fabricated hallucination",
                evidence_excerpt="Completely fabricated citation not present anywhere in the corpus passages",
                prompt_version="v1",
            )
        return RelationshipClassification(
            relationship=OpportunityRelationship.SUPPORTS,
            confidence=0.88,
            rationale="Passage provides grounded evidence regarding research opportunity.",
            evidence_excerpt="Retrieval-augmented generation enables large language models to ground their reasoning in external knowledge",
            prompt_version="v1",
        )

    provider = Stage5MockLLMProvider(custom_stage_5_handler=handler)
    embedder = DeterministicExperimentEmbeddingProvider(64)

    runner = ControlledExperimentRunner(
        llm_provider=provider,
        embedding_provider=embedder,
        artifacts_dir=artifacts_dir,
        db_path=str(db_path),
        temporal_cutoff="2022-12-31",
        enable_stage_5=True,
        stage_5_cutoffs=["2023-12-31", "2024-12-31"],
    )

    res = runner.run(temporal_pdf_corpus)
    assert res.success is True

    s5 = res.metrics["stage_5_evolution"]
    assert len(s5["per_batch_metrics"]) == 2

    # Batch 1 rejected ungrounded excerpt for both opportunities
    b1 = s5["per_batch_metrics"][0]
    assert b1["grounding_rejections_count"] == 2
    assert b1["processed_papers"] == 0

    # Batch 2 proceeded and succeeded for both opportunities
    b2 = s5["per_batch_metrics"][1]
    assert b2["grounding_rejections_count"] == 0
    assert b2["processed_papers"] == 2

    # Database contains exactly the 2 grounded evidence rows from Batch 2
    verify_engine = create_engine(f"sqlite:///{db_path}")
    with Session(verify_engine) as verify_session:
        evidence_rows = verify_session.query(OpportunityEvidence).all()
        assert len(evidence_rows) == 2
        for ev in evidence_rows:
            assert "Retrieval-augmented generation" in ev.supporting_text
    verify_engine.dispose()


def test_stage5_resuming_skips_persisted_pairs(temporal_pdf_corpus: Path, tmp_path: Path):
    """Verifies that resuming against an existing database skips already-persisted pairs
    without re-invoking the LLM classifier."""
    artifacts_dir = tmp_path / "artifacts_resume"
    db_path = tmp_path / "resume_state.db"
    provider_run1 = Stage5MockLLMProvider(
        relationship=OpportunityRelationship.SUPPORTS,
        evidence_excerpt="Retrieval-augmented generation enables large language models to ground their reasoning in external knowledge",
    )
    embedder = DeterministicExperimentEmbeddingProvider(64)

    runner_run1 = ControlledExperimentRunner(
        llm_provider=provider_run1,
        embedding_provider=embedder,
        artifacts_dir=artifacts_dir,
        db_path=str(db_path),
        temporal_cutoff="2022-12-31",
        enable_stage_5=True,
        stage_5_cutoffs=["2023-12-31"],
    )

    res1 = runner_run1.run(temporal_pdf_corpus)
    assert res1.success is True
    # 2 baseline opportunities evaluated against 1 eligible paper = 2 calls
    assert provider_run1.stage_5_calls == 2

    # Second run against the same persistent database with a tracking provider
    provider_run2 = Stage5MockLLMProvider(
        relationship=OpportunityRelationship.SUPPORTS,
        evidence_excerpt="Retrieval-augmented generation enables large language models to ground their reasoning in external knowledge",
    )
    runner_run2 = ControlledExperimentRunner(
        llm_provider=provider_run2,
        embedding_provider=embedder,
        artifacts_dir=artifacts_dir,
        db_path=str(db_path),
        temporal_cutoff="2022-12-31",
        enable_stage_5=True,
        stage_5_cutoffs=["2023-12-31"],
    )

    res2 = runner_run2.run(temporal_pdf_corpus)
    assert res2.success is True

    # Classifier was NOT called on resume because both pairs were already committed in the DB!
    assert provider_run2.stage_5_calls == 0

    s5_run2 = res2.metrics["stage_5_evolution"]
    b1_run2 = s5_run2["per_batch_metrics"][0]
    assert b1_run2["skipped_papers"] == 2
    assert b1_run2["processed_papers"] == 0
