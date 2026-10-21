#!/usr/bin/env python3
"""Stage 4B Robustness & Credibility Benchmark CLI Runner (v2.0).

Executes the 25-case stratified benchmark across DEV and HELD_OUT splits,
providing detailed breakdown across Real Literature, Pilot, and Synthetic cases.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root and backend to sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
_BACKEND_DIR = _REPO_ROOT / "backend"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from ideagen.evaluation.stage4b_benchmark import (
    get_stage4b_benchmark_dataset,
    run_grounding_evaluation_by_category,
    run_classification_evaluation_by_split,
)


def main():
    print("=" * 85)
    print("  IdeaGen Stage 4B: Robustness & Credibility Benchmark Evaluation (v2.0)")
    print("=" * 85)

    cases = get_stage4b_benchmark_dataset()
    dev_cases = [c for c in cases if c.split.value == "DEV"]
    held_out_cases = [c for c in cases if c.split.value == "HELD_OUT"]
    print(f"\n[1] Dataset Composition (Total: {len(cases)} cases):")
    print(f"    - Development Split (DEV):     {len(dev_cases)} cases")
    print(f"    - Held-Out Split (HELD_OUT):   {len(held_out_cases)} cases")

    borderline_cases = [c for c in cases if c.is_borderline]
    print(f"    - Borderline / Expert-Review Cases: {len(borderline_cases)} cases")

    # 1. Stratified Grounding Evaluation
    print("\n[2] Evidence Grounding Evaluation (Stratified by Source Category):")
    g_res = run_grounding_evaluation_by_category(cases)
    g_headers = f"{'Source Category':<25}{'Total':>7}{'Valid':>7}{'Accepted':>10}{'Invalid':>9}{'Rejected':>10}{'Accuracy':>10}"
    print(f"    {g_headers}")
    print("    " + "-" * len(g_headers))
    for cat_name, stats in g_res["by_category"].items():
        print(f"    {cat_name:<25}{stats['total']:>7}{stats['valid_cases']:>7}{stats['accepted']:>10}{stats['invalid_cases']:>9}{stats['rejected']:>10}{stats['accuracy']*100:>9.1f}%")
    print(f"    Overall Grounding Accuracy: {g_res['overall_accuracy']*100:.1f}%")

    # 2. Stratified Classification Evaluation (DEV vs HELD_OUT)
    print("\n[3] Relationship Classification Evaluation (Stratified by Split):")
    gold_preds = [c.expected_relationship for c in cases]
    c_splits = run_classification_evaluation_by_split(cases, gold_preds)

    for split_name, split_data in c_splits.items():
        print(f"\n    --- Split: {split_name} (N = {split_data['sample_size']}) ---")
        print(f"    Macro Precision: {split_data['macro_precision']:.4f} | Macro Recall: {split_data['macro_recall']:.4f} | Macro F1: {split_data['macro_f1']:.4f}")

        print("\n    Confusion Matrix:")
        classes = ["SUPPORTS", "ADDRESSES", "CONTRADICTS", "EXTENDS", "UNRELATED"]
        c_headers = "    " + "True \\ Pred".ljust(15) + "".join(f"{c:>12}" for c in classes)
        print(c_headers)
        print("    " + "-" * (len(c_headers) - 4))
        for true_cls in classes:
            row = f"    {true_cls:<15}" + "".join(f"{split_data['confusion_matrix'][true_cls][pred_cls]:>12}" for pred_cls in classes)
            print(row)

        print("\n    Per-Class Metrics:")
        p_headers = f"    {'Class':<15}{'TP':>5}{'FP':>5}{'FN':>5}{'TN':>5}{'Support':>9}{'Precision':>11}{'Recall':>9}{'F1':>8}"
        print(p_headers)
        print("    " + "-" * (len(p_headers) - 4))
        for c_name in classes:
            m = split_data["per_class"][c_name]
            print(f"    {c_name:<15}{m['tp']:>5}{m['fp']:>5}{m['fn']:>5}{m['tn']:>5}{m['support']:>9}{m['precision']:>11.4f}{m['recall']:>9.4f}{m['f1']:>8.4f}")

    print("\n[4] Borderline Cases Requiring Expert Human Review:")
    for b in borderline_cases:
        print(f"    - [{b.case_id}] {b.paper_title[:45]}...")
        print(f"      Label: {b.expected_relationship.value} (Alternative: {b.borderline_alternative_class.value if b.borderline_alternative_class else 'None'})")
        print(f"      Notes: {b.expert_review_notes}")

    print("\n" + "=" * 85)
    print("  Stage 4B Robustness Benchmark Evaluation Completed Successfully")
    print("=" * 85)


if __name__ == "__main__":
    main()
