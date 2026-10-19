"""Unit tests for pairwise clustering evaluation metrics.

Verifies:
1. Perfectly matching cluster partitions.
2. Different cluster IDs representing the same partition.
3. One predicted cluster containing items from multiple reference clusters.
4. Over-splitting a reference cluster into multiple predicted clusters.
5. All singleton clusters.
6. All items in one cluster.
7. Empty and single-item input (degenerate cases).
8. Mismatched input lengths, missing labels, and overlapping clusters.
9. Pairwise precision, recall, and F1 values verified against hand-calculated examples.
10. Rand Index verified across diverse clustering topologies.
"""

import pytest

from ideagen.evaluation.metrics import (
    precision_score,
    recall_score,
    f1_score,
    rand_index_score,
    pairwise_cluster_metrics,
)


def test_perfectly_matching_cluster_partitions():
    """Requirement 1: Perfectly matching partitions yield 1.0 across all metrics."""
    # Partition representation
    pred = [{"a", "b"}, {"c", "d"}]
    ref = [{"a", "b"}, {"c", "d"}]

    res = pairwise_cluster_metrics(pred, ref)
    assert res["true_positives"] == 2
    assert res["false_positives"] == 0
    assert res["false_negatives"] == 0
    assert res["true_negatives"] == 4
    assert res["pairwise_precision"] == 1.0
    assert res["pairwise_recall"] == 1.0
    assert res["pairwise_f1"] == 1.0
    assert res["f1"] == 1.0
    assert res["rand_index"] == 1.0


def test_different_cluster_ids_same_partition():
    """Requirement 2: Different cluster IDs/ordering representing the same partition."""
    # Partition representation with different order
    pred = [{"c", "d"}, {"a", "b"}]
    ref = [{"a", "b"}, {"c", "d"}]

    res = pairwise_cluster_metrics(pred, ref)
    assert res["pairwise_precision"] == 1.0
    assert res["pairwise_recall"] == 1.0
    assert res["pairwise_f1"] == 1.0
    assert res["rand_index"] == 1.0

    # Flat label representation with completely different IDs
    pred_labels = ["topic_X", "topic_X", "topic_Y", "topic_Y"]
    ref_labels = [100, 100, 200, 200]

    res_flat = pairwise_cluster_metrics(pred_labels, ref_labels)
    assert res_flat["pairwise_precision"] == 1.0
    assert res_flat["pairwise_recall"] == 1.0
    assert res_flat["pairwise_f1"] == 1.0
    assert res_flat["rand_index"] == 1.0


def test_one_predicted_cluster_multiple_reference_clusters():
    """Requirement 3: Under-clustering / merging multiple reference clusters."""
    # 4 items: {1, 2, 3, 4}. Total pairs = 6.
    # Ref: [{1, 2}, {3, 4}] -> same-cluster pairs: (1, 2), (3, 4) [2 pairs]
    # Pred: [{1, 2, 3}, {4}] -> same-cluster pairs: (1, 2), (1, 3), (2, 3) [3 pairs]
    # TP: (1, 2) [1]
    # FP: (1, 3), (2, 3) [2]
    # FN: (3, 4) [1]
    # TN: (1, 4), (2, 4) [2]
    pred = [{1, 2, 3}, {4}]
    ref = [{1, 2}, {3, 4}]

    res = pairwise_cluster_metrics(pred, ref)
    assert res["true_positives"] == 1
    assert res["false_positives"] == 2
    assert res["false_negatives"] == 1
    assert res["true_negatives"] == 2

    assert pytest.approx(res["pairwise_precision"], abs=1e-6) == 1.0 / 3.0
    assert pytest.approx(res["pairwise_recall"], abs=1e-6) == 0.5
    assert pytest.approx(res["pairwise_f1"], abs=1e-6) == 0.4
    assert pytest.approx(res["f1"], abs=1e-6) == 0.4
    assert pytest.approx(res["rand_index"], abs=1e-6) == 0.5


def test_oversplitting_reference_cluster():
    """Requirement 4: Over-splitting a reference cluster into multiple predicted clusters."""
    # 4 items: {1, 2, 3, 4}. Total pairs = 6.
    # Ref: [{1, 2}, {3, 4}]
    # Pred: [{1}, {2}, {3, 4}]
    # TP: (3, 4) [1]
    # FP: None [0]
    # FN: (1, 2) [1]
    # TN: (1, 3), (1, 4), (2, 3), (2, 4) [4]
    pred = [{1}, {2}, {3, 4}]
    ref = [{1, 2}, {3, 4}]

    res = pairwise_cluster_metrics(pred, ref)
    assert res["true_positives"] == 1
    assert res["false_positives"] == 0
    assert res["false_negatives"] == 1
    assert res["true_negatives"] == 4

    assert res["pairwise_precision"] == 1.0
    assert pytest.approx(res["pairwise_recall"], abs=1e-6) == 0.5
    assert pytest.approx(res["pairwise_f1"], abs=1e-6) == 2.0 / 3.0
    assert pytest.approx(res["rand_index"], abs=1e-6) == 5.0 / 6.0


def test_all_singleton_clusters():
    """Requirement 5: Degenerate singleton topologies."""
    # Case 5a: Both are all singletons
    pred_all_single = [{"x"}, {"y"}, {"z"}]
    ref_all_single = [{"x"}, {"y"}, {"z"}]

    res_same = pairwise_cluster_metrics(pred_all_single, ref_all_single)
    assert res_same["true_positives"] == 0
    assert res_same["false_positives"] == 0
    assert res_same["false_negatives"] == 0
    assert res_same["true_negatives"] == 3
    assert res_same["pairwise_precision"] == 0.0
    assert res_same["pairwise_recall"] == 0.0
    assert res_same["pairwise_f1"] == 0.0
    assert res_same["rand_index"] == 1.0

    # Case 5b: Predicted is singletons, reference has a cluster
    ref_grouped = [{"x", "y"}, {"z"}]
    res_diff = pairwise_cluster_metrics(pred_all_single, ref_grouped)
    assert res_diff["true_positives"] == 0
    assert res_diff["false_positives"] == 0
    assert res_diff["false_negatives"] == 1
    assert res_diff["true_negatives"] == 2
    assert res_diff["pairwise_precision"] == 0.0
    assert res_diff["pairwise_recall"] == 0.0
    assert res_diff["pairwise_f1"] == 0.0
    assert pytest.approx(res_diff["rand_index"], abs=1e-6) == 2.0 / 3.0


def test_all_items_in_one_cluster():
    """Requirement 6: All items lumped into a single cluster."""
    # Case 6a: Both lump everything into one cluster
    pred_lumped = [{"x", "y", "z"}]
    ref_lumped = [{"x", "y", "z"}]

    res_same = pairwise_cluster_metrics(pred_lumped, ref_lumped)
    assert res_same["true_positives"] == 3
    assert res_same["false_positives"] == 0
    assert res_same["false_negatives"] == 0
    assert res_same["true_negatives"] == 0
    assert res_same["pairwise_precision"] == 1.0
    assert res_same["pairwise_recall"] == 1.0
    assert res_same["pairwise_f1"] == 1.0
    assert res_same["rand_index"] == 1.0

    # Case 6b: Predicted lumps everything, reference has two clusters
    ref_split = [{"x", "y"}, {"z"}]
    res_diff = pairwise_cluster_metrics(pred_lumped, ref_split)
    assert res_diff["true_positives"] == 1
    assert res_diff["false_positives"] == 2
    assert res_diff["false_negatives"] == 0
    assert res_diff["true_negatives"] == 0
    assert pytest.approx(res_diff["pairwise_precision"], abs=1e-6) == 1.0 / 3.0
    assert res_diff["pairwise_recall"] == 1.0
    assert pytest.approx(res_diff["pairwise_f1"], abs=1e-6) == 0.5
    assert pytest.approx(res_diff["rand_index"], abs=1e-6) == 1.0 / 3.0


def test_empty_and_single_item_inputs():
    """Requirement 7: Empty and single-item inputs handle degenerate cases safely."""
    # Empty lists
    res_empty = pairwise_cluster_metrics([], [])
    assert res_empty["true_positives"] == 0
    assert res_empty["false_positives"] == 0
    assert res_empty["false_negatives"] == 0
    assert res_empty["true_negatives"] == 0
    assert res_empty["pairwise_precision"] == 0.0
    assert res_empty["pairwise_recall"] == 0.0
    assert res_empty["pairwise_f1"] == 0.0
    assert res_empty["rand_index"] == 1.0

    # Lists of empty sets
    res_empty_sets = pairwise_cluster_metrics([set()], [set()])
    assert res_empty_sets["rand_index"] == 1.0
    assert res_empty_sets["pairwise_precision"] == 0.0

    # Single-item cluster (N = 1, 0 pairs)
    res_single = pairwise_cluster_metrics([{"item1"}], [{"item1"}])
    assert res_single["true_positives"] == 0
    assert res_single["rand_index"] == 1.0
    assert res_single["pairwise_precision"] == 0.0


def test_mismatched_inputs_and_invalid_data():
    """Requirement 8: Validation rejects mismatched lengths, missing labels, and non-disjoint clusters."""
    # None inputs
    with pytest.raises(ValueError, match="cannot be None"):
        pairwise_cluster_metrics(None, [{"a"}])

    # Flat labels of unequal lengths
    with pytest.raises(ValueError, match="Mismatched input lengths"):
        pairwise_cluster_metrics([0, 1], [0, 1, 2])

    # Missing label (None) in flat labels
    with pytest.raises(ValueError, match="Missing label encountered"):
        pairwise_cluster_metrics([0, None, 1], [0, 1, 1])

    # Cluster partitions with mismatched items (item missing from predicted)
    with pytest.raises(ValueError, match="Mismatched items"):
        pairwise_cluster_metrics([{"a", "b"}], [{"a", "b", "c"}])

    # Cluster partitions with different item universe
    with pytest.raises(ValueError, match="Mismatched items"):
        pairwise_cluster_metrics([{"a", "b"}], [{"a", "c"}])

    # Overlapping clusters in partition (non-disjoint)
    with pytest.raises(ValueError, match="partitions must be disjoint"):
        pairwise_cluster_metrics([{"a", "b"}, {"b", "c"}], [{"a", "b", "c"}])

    # Incompatible input representations
    with pytest.raises(ValueError, match="Incompatible input types"):
        pairwise_cluster_metrics([{"a", "b"}], [0, 1])


def test_hand_calculated_six_item_partition():
    """Requirement 9: Detailed multi-cluster hand-calculated partition validation.

    Universe: {1, 2, 3, 4, 5, 6}. N = 6, Total pairs = 15.
    Ground Truth:
      G1 = {1, 2, 3}  (3 pairs: (1,2), (1,3), (2,3))
      G2 = {4, 5}     (1 pair: (4,5))
      G3 = {6}        (0 pairs)
      Total Ref Positive Pairs = 4

    Predicted:
      C1 = {1, 2}     (1 pair: (1,2))
      C2 = {3, 4, 5}  (3 pairs: (3,4), (3,5), (4,5))
      C3 = {6}        (0 pairs)
      Total Pred Positive Pairs = 4

    Intersections:
      C1 cap G1 = {1, 2} -> 1 TP ((1, 2))
      C2 cap G2 = {4, 5} -> 1 TP ((4, 5))
      All other intersections size <= 1.

    Counts:
      TP = 2
      FP = 4 - 2 = 2   ((3, 4), (3, 5))
      FN = 4 - 2 = 2   ((1, 3), (2, 3))
      TN = 15 - 2 - 2 - 2 = 9
      Total = 15

    Metrics:
      Precision = 2 / 4 = 0.5
      Recall = 2 / 4 = 0.5
      F1 = 0.5
      Rand Index = (2 + 9) / 15 = 11 / 15 approx 0.733333
    """
    pred = [{1, 2}, {3, 4, 5}, {6}]
    ref = [{1, 2, 3}, {4, 5}, {6}]

    res = pairwise_cluster_metrics(pred, ref)
    assert res["true_positives"] == 2
    assert res["false_positives"] == 2
    assert res["false_negatives"] == 2
    assert res["true_negatives"] == 9

    assert res["pairwise_precision"] == 0.5
    assert res["pairwise_recall"] == 0.5
    assert res["pairwise_f1"] == 0.5
    assert res["f1"] == 0.5
    assert pytest.approx(res["rand_index"], abs=1e-6) == 11.0 / 15.0


def test_rand_index_score_helper():
    """Requirement 10: Direct verification of rand_index_score helper."""
    # Degenerate case (0 pairs)
    assert rand_index_score(0, 0, 0, 0) == 1.0

    # Perfect agreement
    assert rand_index_score(true_positives=5, true_negatives=5, false_positives=0, false_negatives=0) == 1.0

    # Complete disagreement
    assert rand_index_score(true_positives=0, true_negatives=0, false_positives=5, false_negatives=5) == 0.0

    # Mixed
    assert rand_index_score(true_positives=2, true_negatives=9, false_positives=2, false_negatives=2) == 11.0 / 15.0


def test_scalar_score_helpers():
    """Verifies edge cases for precision_score, recall_score, and f1_score."""
    # Precision 0/0
    assert precision_score(0, 0) == 0.0
    assert precision_score(3, 1) == 0.75

    # Recall 0/0
    assert recall_score(0, 0) == 0.0
    assert recall_score(3, 1) == 0.75

    # F1 0/0
    assert f1_score(0.0, 0.0) == 0.0
    assert f1_score(1.0, 1.0) == 1.0
    assert f1_score(0.5, 0.5) == 0.5
