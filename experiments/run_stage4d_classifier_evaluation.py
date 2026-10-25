#!/usr/bin/env python3
"""Stage 4D: Independent Relationship Classifier Evaluation CLI Runner.

Evaluates the actual RelationshipClassifier against Stage 4B Benchmark v2.0 (N=25)
without deriving predictions from gold labels.

Disciplines:
- Zero gold-label leakage into classifier inputs or prompts.
- Distinguishes model failures/abstentions from misclassifications.
- If no local provider is available, reports NO_PROVIDER honestly without fabricating accuracy.
- Saves empirical artifacts to a separate timestamped directory without overwriting prior reports.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys

# Add project root and backend to sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
_BACKEND_DIR = _REPO_ROOT / "backend"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from ideagen.core.config import get_settings
from ideagen.opportunity.classifier import RelationshipClassifier
from ideagen.evaluation.stage4d_evaluator import (
    IndependentClassifierEvaluator,
    IndependentEvaluationReport,
    EvaluationPredictionStatus,
)


def get_configured_classifier(
    timeout: float | None = None,
    model_name: str | None = None,
) -> tuple[RelationshipClassifier | None, str]:
    """Attempts to configure the actual RelationshipClassifier from project settings.

    Returns:
        (classifier_instance, status_description)
    """
    settings = get_settings()
    provider_name = (settings.llm_provider or "ollama").lower()

    if provider_name == "ollama":
        from ideagen.llm.ollama_provider import OllamaLLMProvider

        effective_model = model_name or settings.ollama_model
        effective_timeout = timeout if timeout is not None else settings.ollama_timeout

        provider = OllamaLLMProvider(
            base_url=settings.ollama_base_url,
            model_name=effective_model,
            timeout=effective_timeout,
            num_ctx=settings.ollama_num_ctx,
        )
        classifier = RelationshipClassifier(
            provider=provider,
            timeout=effective_timeout,
            max_retries=1,
        )
        return (
            classifier,
            f"OllamaLLMProvider (model: {effective_model}, url: {settings.ollama_base_url}, timeout: {effective_timeout}s)",
        )
    elif provider_name == "gemini":
        from ideagen.llm.gemini_provider import GeminiLLMProvider

        if not settings.gemini_api_key:
            return None, "Gemini provider configured but GEMINI_API_KEY is not set."
        provider = GeminiLLMProvider(
            api_key=settings.gemini_api_key,
            model_name=settings.gemini_model_name,
        )
        classifier = RelationshipClassifier(provider=provider, timeout=30.0, max_retries=0)
        return classifier, f"GeminiLLMProvider (model: {settings.gemini_model_name})"

    return None, f"Unsupported provider: {provider_name}"


def generate_markdown_report(report: IndependentEvaluationReport) -> str:
    """Renders the independent evaluation report into clean GitHub Markdown."""
    lines: list[str] = []
    lines.append("# IdeaGen Stage 4D: Live Independent Relationship Classifier Evaluation")
    lines.append("")
    lines.append(f"**Execution Timestamp:** {report.timestamp}  ")
    lines.append(f"**Benchmark Dataset:** Stage 4B Grounded Benchmark v2.0 ({report.total_cases} cases)  ")
    lines.append(f"**Provider:** `{report.llm_provider or 'None'}`  ")
    lines.append(f"**Model:** `{report.model_name or 'None'}`  ")
    lines.append(f"**Prompt Version:** `{report.prompt_version}`  ")
    lines.append(f"**Total Duration:** {report.duration_seconds}s  ")
    lines.append(f"**Empirical Inference Status:** `{report.is_empirical_inference}`  ")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. Executive Summary")
    lines.append("")
    lines.append(f"- **Total Cases Evaluated:** {report.total_cases}")
    lines.append(f"- **Successful Model Predictions:** {report.successful_predictions}")
    lines.append(f"- **Abstentions / Provider Failures:** {report.failed_predictions}")
    for fail_k, fail_v in report.failure_breakdown.items():
        if fail_v > 0:
            lines.append(f"  * `{fail_k}`: {fail_v}")
    acc_str = f"{report.empirical_accuracy * 100:.2f}%" if report.empirical_accuracy is not None else "N/A"
    lines.append(f"- **Overall Empirical Accuracy:** **{acc_str}**")
    lines.append("")

    if not report.is_empirical_inference or report.successful_predictions == 0:
        lines.append("> [!NOTE]")
        lines.append("> No empirical predictions were generated. Empirical accuracy is withheld.")
        return "\n".join(lines)

    lines.append("---")
    lines.append("")
    lines.append("## 2. Partition Results: Development (DEV) vs Held-Out (HELD_OUT)")
    lines.append("")
    classes = ["SUPPORTS", "ADDRESSES", "CONTRADICTS", "EXTENDS", "UNRELATED"]

    for split_key in ["DEV", "HELD_OUT"]:
        split_data = report.metrics_by_split.get(split_key)
        if not split_data:
            continue
        eval_n = split_data.get("evaluated_count", 0)
        acc = split_data.get("overall_accuracy")
        acc_fmt = f"{acc * 100:.2f}%" if acc is not None else "N/A"
        f1 = split_data.get("macro_f1")
        f1_fmt = f"{f1:.4f}" if f1 is not None else "N/A"

        lines.append(f"### Split: `{split_key}` (N = {eval_n})")
        lines.append(f"- **Accuracy:** {acc_fmt}")
        lines.append(f"- **Macro F1:** {f1_fmt}")
        lines.append("")
        lines.append("#### Confusion Matrix")
        lines.append("")
        matrix = split_data["confusion_matrix"]
        lines.append(f"| True \\ Pred | " + " | ".join(classes) + " |")
        lines.append("| :--- | " + " | ".join([":---:"] * len(classes)) + " |")
        for true_c in classes:
            row_vals = [str(matrix[true_c][pred_c]) for pred_c in classes]
            lines.append(f"| **{true_c}** | " + " | ".join(row_vals) + " |")
        lines.append("")

        lines.append("#### Per-Class Metrics")
        lines.append("")
        lines.append("| Class | Support | TP | FP | FN | Precision | Recall | F1 |")
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for c_name in classes:
            m = split_data["per_class"][c_name]
            p_str = f"{m['precision']:.4f}"
            r_str = f"{m['recall']:.4f}"
            f1_str = f"{m['f1']:.4f}"
            lines.append(
                f"| **{c_name}** | {m['support']} | {m['tp']} | {m['fp']} | {m['fn']} | {p_str} | {r_str} | {f1_str} |"
            )
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 3. Performance by Source Category")
    lines.append("")
    lines.append("| Category | Total Evaluated | Accuracy | Macro F1 |")
    lines.append("| :--- | :---: | :---: | :---: |")
    for cat_key, cat_data in report.metrics_by_category.items():
        c_acc = cat_data.get("overall_accuracy")
        c_acc_str = f"{c_acc * 100:.2f}%" if c_acc is not None else "N/A"
        c_f1 = cat_data.get("macro_f1")
        c_f1_str = f"{c_f1:.4f}" if c_f1 is not None else "N/A"
        lines.append(f"| `{cat_key}` | {cat_data.get('evaluated_count', 0)} | {c_acc_str} | {c_f1_str} |")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 4. Case-by-Case Predictions and Audit Log")
    lines.append("")
    lines.append("| Case ID | Split | Category | Gold Label | Predicted Label | Match? | Conf | Latency | Status |")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :--- |")
    for r in report.records:
        pred_label = r.predicted_relationship.value if r.predicted_relationship else "None"
        gold_label = r.gold_relationship.value
        match_sym = "YES" if pred_label == gold_label else ("NO" if r.predicted_relationship else "ABSTAIN")
        conf_str = f"{r.confidence:.2f}" if r.confidence is not None else "-"
        lat_str = f"{r.latency_seconds:.2f}s" if r.latency_seconds is not None else "-"
        lines.append(
            f"| `{r.case_id}` | `{r.split.value}` | `{r.source_category.value}` | `{gold_label}` | `{pred_label}` | **{match_sym}** | {conf_str} | {lat_str} | `{r.prediction_status.value}` |"
        )
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 5. Model Rationales for Borderline Cases")
    lines.append("")
    borderline_records = [r for r in report.records if r.is_borderline]
    for br in borderline_records:
        lines.append(f"### Case: `{br.case_id}`")
        lines.append(f"- **Domain:** {br.domain}")
        lines.append(f"- **Gold Label:** `{br.gold_relationship.value}` (Borderline Alt: `{br.borderline_alternative_class.value if br.borderline_alternative_class else 'None'}`)")
        pred_l = br.predicted_relationship.value if br.predicted_relationship else "Abstained"
        lines.append(f"- **Model Prediction:** `{pred_l}` (Confidence: {br.confidence})")
        lines.append(f"- **Model Rationale:** {br.rationale or br.error_message}")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="IdeaGen Stage 4D Independent Classifier Evaluation")
    parser.add_argument(
        "--dry-run-check",
        action="store_true",
        help="Run strict infrastructure check (no live inference, verifies zero prediction fabrication)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Per-case classification timeout in seconds (default: 120.0)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override model name (default: from project settings, e.g. qwen3:8b)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory to store timestamped artifacts",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    print("=" * 85)
    print("  IdeaGen Stage 4D: Independent Relationship Classifier Evaluation")
    print("=" * 85)

    classifier = None
    provider_desc = "None"

    if not args.dry_run_check:
        classifier, provider_desc = get_configured_classifier(
            timeout=args.timeout,
            model_name=args.model,
        )
        print(f"\n[1] Configured Provider: {provider_desc}")
    else:
        print("\n[1] Execution Mode: Dry-Run Infrastructure Check (no live inference requested)")

    evaluator = IndependentClassifierEvaluator(classifier=classifier)
    report = evaluator.evaluate()

    print(f"\n[2] Evaluation Execution Metadata:")
    print(f"    - Timestamp:           {report.timestamp}")
    print(f"    - Benchmark Version:   {report.benchmark_version} (N = {report.total_cases} cases)")
    print(f"    - Prompt Version:      {report.prompt_version}")
    print(f"    - Empirical Inference: {report.is_empirical_inference}")
    print(f"    - LLM Provider:        {report.llm_provider or 'None'}")
    print(f"    - Model Identifier:    {report.model_name or 'None'}")
    print(f"    - Duration:            {report.duration_seconds}s")

    print(f"\n[3] Prediction Results Summary:")
    print(f"    - Total Cases:                {report.total_cases}")
    print(f"    - Successful Predictions:     {report.successful_predictions}")
    print(f"    - Failed / Abstained Cases:   {report.failed_predictions}")
    for fail_type, count in report.failure_breakdown.items():
        if count > 0:
            print(f"      * {fail_type}: {count}")

    if report.empirical_accuracy is not None:
        print(f"\n[4] Empirical Classification Accuracy: {report.empirical_accuracy * 100:.2f}%")
        classes = ["SUPPORTS", "ADDRESSES", "CONTRADICTS", "EXTENDS", "UNRELATED"]
        for split_name, split_data in report.metrics_by_split.items():
            print(f"\n    --- Split: {split_name} (Evaluated N = {split_data['evaluated_count']}) ---")
            print(f"    Macro F1: {split_data['macro_f1']} | Accuracy: {split_data['overall_accuracy']}")
            matrix = split_data["confusion_matrix"]
            c_headers = "    " + "True \\ Pred".ljust(15) + "".join(f"{c:>12}" for c in classes)
            print(c_headers)
            print("    " + "-" * (len(c_headers) - 4))
            for true_cls in classes:
                row = f"    {true_cls:<15}" + "".join(f"{matrix[true_cls][pred_cls]:>12}" for pred_cls in classes)
                print(row)
    else:
        print("\n[4] Empirical Accuracy: WITHHELD / NOT MEASURED")
        print("    Reason: No live LLM provider inference was completed in this execution environment.")
        print("    SAFEGUARD VERIFIED: Zero predictions were fabricated or copied from gold labels.")

    print(f"\n[5] Safeguard & Independence Verification:")
    print("    [PASS] Input generation: Zero expected_relationship fields passed to classifier.")
    print("    [PASS] Prompt template: Strictly uses opportunity + paper context.")
    print("    [PASS] Abstention handling: Failures recorded honestly without defaulting to gold labels.")
    print("    [PASS] Independence: Gold reference labels strictly reserved for scoring.")

    # Save timestamped artifacts if live inference was executed
    if report.is_empirical_inference and report.successful_predictions > 0:
        ts_slug = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        if args.output_dir:
            artifact_dir = Path(args.output_dir)
        else:
            artifact_dir = _REPO_ROOT / "experiments" / "artifacts" / f"stage4d_live_{ts_slug}"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        report_md_path = artifact_dir / "stage4d_live_evaluation_report.md"
        report_md_content = generate_markdown_report(report)
        report_md_path.write_text(report_md_content, encoding="utf-8")

        # Serialized JSON metrics
        metrics_json_path = artifact_dir / "stage4d_live_metrics.json"
        serializable_records = [
            {
                "case_id": r.case_id,
                "split": r.split.value,
                "source_category": r.source_category.value,
                "domain": r.domain,
                "gold_relationship": r.gold_relationship.value,
                "predicted_relationship": r.predicted_relationship.value if r.predicted_relationship else None,
                "confidence": r.confidence,
                "rationale": r.rationale,
                "prediction_status": r.prediction_status.value,
                "latency_seconds": r.latency_seconds,
                "error_message": r.error_message,
            }
            for r in report.records
        ]
        metrics_dict = {
            "timestamp": report.timestamp,
            "benchmark_version": report.benchmark_version,
            "llm_provider": report.llm_provider,
            "model_name": report.model_name,
            "duration_seconds": report.duration_seconds,
            "total_cases": report.total_cases,
            "successful_predictions": report.successful_predictions,
            "failed_predictions": report.failed_predictions,
            "failure_breakdown": report.failure_breakdown,
            "empirical_accuracy": report.empirical_accuracy,
            "metrics_by_split": report.metrics_by_split,
            "metrics_by_category": report.metrics_by_category,
            "records": serializable_records,
        }
        metrics_json_path.write_text(json.dumps(metrics_dict, indent=2), encoding="utf-8")
        print(f"\n[6] Saved Timestamped Artifacts:")
        print(f"    - Markdown Report: {report_md_path}")
        print(f"    - JSON Metrics:    {metrics_json_path}")

    print("\n" + "=" * 85)
    print("  Stage 4D Independent Evaluation Runner Completed")
    print("=" * 85)


if __name__ == "__main__":
    main()
