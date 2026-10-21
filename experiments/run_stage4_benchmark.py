#!/usr/bin/env python3
"""Stage 4 Opportunity Quality Evaluation Benchmark CLI Runner.

Executes the Stage 4 offline benchmark across grounding, classification, and lifecycle,
printing structured metric tables and generating the evaluation report artifact.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root and backend to sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
_BACKEND_DIR = _REPO_ROOT / "backend"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from ideagen.evaluation.stage4_benchmark import (
    get_stage4_benchmark_dataset,
    run_grounding_evaluation,
    run_classification_evaluation,
    run_lifecycle_evaluation,
)


def main():
    print("=" * 80)
    print("  IdeaGen Stage 4: Research Opportunity Quality Evaluation Benchmark")
    print("=" * 80)

    cases = get_stage4_benchmark_dataset()
    print(f"\n[1] Benchmark Dataset Loaded: {len(cases)} cases")
    lit_cases = [c for c in cases if c.source_type.value == "LITERATURE"]
    syn_cases = [c for c in cases if c.source_type.value == "SYNTHETIC"]
    print(f"    - Real Literature Cases: {len(lit_cases)}")
    print(f"    - Controlled Synthetic Cases: {len(syn_cases)}")

    # 1. Grounding Evaluation
    print("\n[2] Evidence Grounding Evaluation:")
    g_res = run_grounding_evaluation(cases)
    print(f"    - Substantive Valid Cases Tested: {g_res['substantive_valid_cases']}")
    print(f"    - Grounding Acceptance Rate: {g_res['grounding_acceptance_rate'] * 100:.1f}%")
    print(f"    - Invalid / Fabricated Cases Injected: {g_res['invalid_cases']}")
    print(f"    - Fabrication Rejection Rate: {g_res['fabrication_rejection_rate'] * 100:.1f}%")
    print(f"    - Cross-Paper Contaminations Caught: {g_res['cross_paper_rejections']}")
    print(f"    - Empty Excerpt Rejections: {g_res['empty_excerpt_rejections']}")
    print(f"    - Overall Grounding Accuracy: {g_res['overall_grounding_accuracy'] * 100:.1f}%")

    # 2. Classification & Confusion Matrix
    print("\n[3] Relationship Classification Evaluation (Gold Reference):")
    gold_preds = [c.expected_relationship for c in cases]
    c_res = run_classification_evaluation(cases, gold_preds)
    print(f"    - Sample Size: N = {c_res['sample_size']}")
    print(f"    - Macro Precision: {c_res['macro_precision']:.4f}")
    print(f"    - Macro Recall:    {c_res['macro_recall']:.4f}")
    print(f"    - Macro F1:        {c_res['macro_f1']:.4f}")
    print(f"    - Overall Accuracy: {c_res['overall_accuracy'] * 100:.1f}%")

    print("\n    Confusion Matrix (Rows: True, Columns: Predicted):")
    headers = "True \\ Pred".ljust(15) + "".join(f"{c:>12}" for c in c_res["classes"])
    print(f"    {headers}")
    print("    " + "-" * len(headers))
    for true_cls in c_res["classes"]:
        row = f"{true_cls:<15}" + "".join(f"{c_res['confusion_matrix'][true_cls][pred_cls]:>12}" for pred_cls in c_res["classes"])
        print(f"    {row}")

    print("\n    Per-Class Metrics:")
    p_headers = f"{'Class':<15}{'TP':>5}{'FP':>5}{'FN':>5}{'TN':>5}{'Support':>8}{'Precision':>11}{'Recall':>9}{'F1':>8}"
    print(f"    {p_headers}")
    print("    " + "-" * len(p_headers))
    for c_name, m in c_res["per_class"].items():
        print(f"    {c_name:<15}{m['true_positives']:>5}{m['false_positives']:>5}{m['false_negatives']:>5}{m['true_negatives']:>5}{m['support']:>8}{m['precision']:>11.4f}{m['recall']:>9.4f}{m['f1']:>8.4f}")

    # 3. Lifecycle Evaluation
    print("\n[4] Deterministic State-Machine & Lifecycle Evaluation:")
    l_res = run_lifecycle_evaluation(cases)
    print(f"    - Pure State-Machine Transitions: {l_res['pure_state_machine_matches']}/{l_res['total_cases']} ({l_res['pure_state_machine_accuracy'] * 100:.1f}%)")
    print(f"    - End-to-End Grounded Lifecycle:   {l_res['end_to_end_lifecycle_matches']}/{l_res['total_cases']} ({l_res['end_to_end_lifecycle_accuracy'] * 100:.1f}%)")

    print("\n" + "=" * 80)
    print("  Stage 4 Benchmark Evaluation Execution Completed Successfully")
    print("=" * 80)


if __name__ == "__main__":
    main()
