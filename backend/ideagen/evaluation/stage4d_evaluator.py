"""Stage 4D: Independent Relationship Classifier Evaluation Engine.

Provides an independent evaluation runner that:
1. Takes the Stage 4B benchmark dataset (v2.0, N=25).
2. Never passes expected_relationship or gold labels to the classifier or prompts.
3. Invokes RelationshipClassifier.classify() with strictly isolated paper/opportunity context.
4. Categorizes predictions into SUCCESS, PROVIDER_FAILURE, and MALFORMED_OUTPUT.
5. Calculates confusion matrices, per-class metrics, and accuracy solely from genuine model predictions.
6. Isolates DEV vs HELD_OUT splits and Real Literature vs Pilot vs Synthetic source categories.
7. Refuses to emit fabricated accuracy scores when no provider is available.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import time
from typing import Any, Dict, List, Optional, Sequence
import logging

from ideagen.opportunity.models import OpportunityRelationship
from ideagen.opportunity.classifier import (
    RelationshipClassifier,
    RelationshipClassificationInput,
    RELATIONSHIP_CLASSIFIER_PROMPT_VERSION,
)
from ideagen.llm.exceptions import (
    ProviderFailureError,
    MalformedResponseError,
)
from ideagen.evaluation.stage4b_benchmark import (
    Stage4bBenchmarkCase,
    BenchmarkSplit,
    CaseSourceCategory,
    get_stage4b_benchmark_dataset,
)
from ideagen.evaluation.metrics import precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)


class EvaluationPredictionStatus(str, Enum):
    SUCCESS = "SUCCESS"
    PROVIDER_FAILURE = "PROVIDER_FAILURE"
    MALFORMED_OUTPUT = "MALFORMED_OUTPUT"
    UNHANDLED_ERROR = "UNHANDLED_ERROR"
    NO_PROVIDER = "NO_PROVIDER"


@dataclass
class CaseEvaluationRecord:
    case_id: str
    split: BenchmarkSplit
    source_category: CaseSourceCategory
    domain: str
    gold_relationship: OpportunityRelationship
    is_borderline: bool
    borderline_alternative_class: Optional[OpportunityRelationship]
    predicted_relationship: Optional[OpportunityRelationship] = None
    confidence: Optional[float] = None
    rationale: Optional[str] = None
    prediction_status: EvaluationPredictionStatus = EvaluationPredictionStatus.NO_PROVIDER
    error_message: Optional[str] = None
    prompt_version: str = RELATIONSHIP_CLASSIFIER_PROMPT_VERSION
    latency_seconds: Optional[float] = None


@dataclass
class IndependentEvaluationReport:
    timestamp: str
    benchmark_version: str
    llm_provider: Optional[str]
    model_name: Optional[str]
    prompt_version: str
    is_empirical_inference: bool
    total_cases: int
    successful_predictions: int
    failed_predictions: int
    failure_breakdown: Dict[str, int]
    empirical_accuracy: Optional[float]
    metrics_by_split: Dict[str, Dict[str, Any]]
    metrics_by_category: Dict[str, Dict[str, Any]]
    records: List[CaseEvaluationRecord]
    notes: str
    duration_seconds: Optional[float] = None



def build_isolated_classifier_input(case: Stage4bBenchmarkCase) -> RelationshipClassificationInput:
    """Builds the input for RelationshipClassifier strictly without access to expected_relationship.

    Guarantees:
    - Supplies only opportunity context and paper context.
    - Zero gold-label leakage.
    """
    abstract_text = "\n\n".join(p.strip() for p in case.passages if p and p.strip())
    return RelationshipClassificationInput(
        opportunity_title=case.opportunity_title,
        opportunity_problem_statement=case.opportunity_problem,
        opportunity_description=case.opportunity_direction,
        paper_title=case.paper_title,
        paper_abstract=abstract_text,
        paper_excerpt=case.claimed_excerpt,
    )


def compute_contingency_and_metrics(
    records: Sequence[CaseEvaluationRecord],
) -> Dict[str, Any]:
    """Computes a multi-class confusion matrix, per-class Precision/Recall/F1, and macro averages.

    Only operates on records with prediction_status == SUCCESS.
    """
    classes = [
        OpportunityRelationship.SUPPORTS,
        OpportunityRelationship.ADDRESSES,
        OpportunityRelationship.CONTRADICTS,
        OpportunityRelationship.EXTENDS,
        OpportunityRelationship.UNRELATED,
    ]
    class_names = [c.value for c in classes]

    successful_records = [r for r in records if r.prediction_status == EvaluationPredictionStatus.SUCCESS and r.predicted_relationship is not None]
    total_eval = len(successful_records)

    matrix: Dict[str, Dict[str, int]] = {
        true_cls: {pred_cls: 0 for pred_cls in class_names}
        for true_cls in class_names
    }

    for r in successful_records:
        assert r.predicted_relationship is not None
        matrix[r.gold_relationship.value][r.predicted_relationship.value] += 1

    per_class: Dict[str, Dict[str, Any]] = {}
    total_tp = 0

    for c_name in class_names:
        tp = matrix[c_name][c_name]
        fp = sum(matrix[other][c_name] for other in class_names if other != c_name)
        fn = sum(matrix[c_name][other] for other in class_names if other != c_name)
        tn = total_eval - (tp + fp + fn)

        p = precision_score(tp, fp)
        rec = recall_score(tp, fn)
        f = f1_score(p, rec)

        per_class[c_name] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "support": sum(matrix[c_name].values()),
            "precision": round(p, 4),
            "recall": round(rec, 4),
            "f1": round(f, 4),
        }
        total_tp += tp

    macro_p = sum(m["precision"] for m in per_class.values()) / len(classes)
    macro_r = sum(m["recall"] for m in per_class.values()) / len(classes)
    macro_f = sum(m["f1"] for m in per_class.values()) / len(classes)
    acc = total_tp / total_eval if total_eval > 0 else 0.0

    return {
        "evaluated_count": total_eval,
        "overall_accuracy": round(acc, 4) if total_eval > 0 else None,
        "macro_precision": round(macro_p, 4) if total_eval > 0 else None,
        "macro_recall": round(macro_r, 4) if total_eval > 0 else None,
        "macro_f1": round(macro_f, 4) if total_eval > 0 else None,
        "confusion_matrix": matrix,
        "per_class": per_class,
    }


class IndependentClassifierEvaluator:
    """Evaluates RelationshipClassifier predictions independently against benchmark gold labels."""

    def __init__(
        self,
        classifier: Optional[RelationshipClassifier] = None,
        benchmark_cases: Optional[Sequence[Stage4bBenchmarkCase]] = None,
    ):
        self.classifier = classifier
        self.cases = benchmark_cases or get_stage4b_benchmark_dataset()

    def evaluate(self) -> IndependentEvaluationReport:
        now_iso = datetime.now(timezone.utc).isoformat()
        start_time = time.perf_counter()
        records: List[CaseEvaluationRecord] = []
        failure_counts = {
            EvaluationPredictionStatus.PROVIDER_FAILURE.value: 0,
            EvaluationPredictionStatus.MALFORMED_OUTPUT.value: 0,
            EvaluationPredictionStatus.UNHANDLED_ERROR.value: 0,
            EvaluationPredictionStatus.NO_PROVIDER.value: 0,
        }

        # Handle environment where no classifier or provider is available
        if self.classifier is None:
            logger.warning("No classifier provider configured for Stage 4D independent evaluation.")
            for c in self.cases:
                rec = CaseEvaluationRecord(
                    case_id=c.case_id,
                    split=c.split,
                    source_category=c.source_category,
                    domain=c.domain,
                    gold_relationship=c.expected_relationship,
                    is_borderline=c.is_borderline,
                    borderline_alternative_class=c.borderline_alternative_class,
                    prediction_status=EvaluationPredictionStatus.NO_PROVIDER,
                    error_message="No LLM provider available in offline execution environment.",
                    latency_seconds=0.0,
                )
                records.append(rec)
                failure_counts[EvaluationPredictionStatus.NO_PROVIDER.value] += 1

            return IndependentEvaluationReport(
                timestamp=now_iso,
                benchmark_version="v2.0-robustness",
                llm_provider=None,
                model_name=None,
                prompt_version=RELATIONSHIP_CLASSIFIER_PROMPT_VERSION,
                is_empirical_inference=False,
                total_cases=len(self.cases),
                successful_predictions=0,
                failed_predictions=len(self.cases),
                failure_breakdown=failure_counts,
                empirical_accuracy=None,
                metrics_by_split={},
                metrics_by_category={},
                records=records,
                notes="Offline environment check: No live LLM provider available. Zero predictions fabricated.",
                duration_seconds=0.0,
            )

        provider_name = type(self.classifier.provider).__name__
        model_name = getattr(self.classifier.provider, "model_name", "unknown")

        for idx, c in enumerate(self.cases, 1):
            input_data = build_isolated_classifier_input(c)
            rec = CaseEvaluationRecord(
                case_id=c.case_id,
                split=c.split,
                source_category=c.source_category,
                domain=c.domain,
                gold_relationship=c.expected_relationship,
                is_borderline=c.is_borderline,
                borderline_alternative_class=c.borderline_alternative_class,
            )

            case_t0 = time.perf_counter()
            try:
                result = self.classifier.classify(input_data)
                rec.predicted_relationship = result.relationship
                rec.confidence = result.confidence
                rec.rationale = result.rationale
                rec.prediction_status = EvaluationPredictionStatus.SUCCESS
            except ProviderFailureError as e:
                rec.prediction_status = EvaluationPredictionStatus.PROVIDER_FAILURE
                rec.error_message = str(e)
                failure_counts[EvaluationPredictionStatus.PROVIDER_FAILURE.value] += 1
            except MalformedResponseError as e:
                rec.prediction_status = EvaluationPredictionStatus.MALFORMED_OUTPUT
                rec.error_message = str(e)
                failure_counts[EvaluationPredictionStatus.MALFORMED_OUTPUT.value] += 1
            except Exception as e:
                rec.prediction_status = EvaluationPredictionStatus.UNHANDLED_ERROR
                rec.error_message = f"{type(e).__name__}: {e}"
                failure_counts[EvaluationPredictionStatus.UNHANDLED_ERROR.value] += 1
            finally:
                rec.latency_seconds = round(time.perf_counter() - case_t0, 3)

            pred_val = rec.predicted_relationship.value if rec.predicted_relationship else rec.prediction_status.value
            logger.info(
                "[%d/%d] Evaluated %s (%s, %s): %s in %.2fs",
                idx,
                len(self.cases),
                c.case_id,
                c.split.value,
                c.source_category.value,
                pred_val,
                rec.latency_seconds,
            )
            records.append(rec)


        successful_count = sum(1 for r in records if r.prediction_status == EvaluationPredictionStatus.SUCCESS)
        failed_count = len(records) - successful_count
        duration = round(time.perf_counter() - start_time, 2)

        # Compute stratified metrics
        splits = [BenchmarkSplit.DEV, BenchmarkSplit.HELD_OUT]
        metrics_by_split = {}
        for sp in splits:
            sp_records = [r for r in records if r.split == sp]
            metrics_by_split[sp.value] = compute_contingency_and_metrics(sp_records)

        categories = [
            CaseSourceCategory.REAL_LITERATURE,
            CaseSourceCategory.HUMAN_CURATED_PILOT,
            CaseSourceCategory.SYNTHETIC_STRESS_TEST,
        ]
        metrics_by_category = {}
        for cat in categories:
            cat_records = [r for r in records if r.source_category == cat]
            if cat_records:
                metrics_by_category[cat.value] = compute_contingency_and_metrics(cat_records)

        overall_metrics = compute_contingency_and_metrics(records)

        return IndependentEvaluationReport(
            timestamp=now_iso,
            benchmark_version="v2.0-robustness",
            llm_provider=provider_name,
            model_name=model_name,
            prompt_version=RELATIONSHIP_CLASSIFIER_PROMPT_VERSION,
            is_empirical_inference=True,
            total_cases=len(self.cases),
            successful_predictions=successful_count,
            failed_predictions=failed_count,
            failure_breakdown=failure_counts,
            empirical_accuracy=overall_metrics.get("overall_accuracy"),
            metrics_by_split=metrics_by_split,
            metrics_by_category=metrics_by_category,
            records=records,
            notes=f"Evaluated with {provider_name} ({model_name}). Successful: {successful_count}/{len(self.cases)}.",
            duration_seconds=duration,
        )
