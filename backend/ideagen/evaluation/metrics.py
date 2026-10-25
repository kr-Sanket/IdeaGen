from typing import Any, Collection, Dict, List, Sequence, Set, Union


def precision_score(true_positives: int, false_positives: int) -> float:
    total = true_positives + false_positives
    return true_positives / total if total > 0 else 0.0


def recall_score(true_positives: int, false_negatives: int) -> float:
    total = true_positives + false_negatives
    return true_positives / total if total > 0 else 0.0


def f1_score(precision: float, recall: float) -> float:
    total = precision + recall
    return (2 * precision * recall) / total if total > 0 else 0.0


def rand_index_score(
    true_positives: int,
    true_negatives: int,
    false_positives: int,
    false_negatives: int,
) -> float:
    """Computes the Rand Index given pairwise contingency counts.

    When there are no pairs (total == 0), defaults to 1.0 by convention
    (vacuous agreement, identical to scikit-learn rand_score behavior).
    """
    total = true_positives + true_negatives + false_positives + false_negatives
    return (true_positives + true_negatives) / total if total > 0 else 1.0


def _comb2(n: int) -> int:
    """Computes n choose 2 = n * (n - 1) // 2 for integer n >= 0."""
    return (n * (n - 1)) // 2 if n >= 2 else 0


def _is_cluster_collection(clusters: Sequence[Any]) -> bool:
    """Detects whether input is a collection of clusters (e.g. List[Set]) or flat labels."""
    if len(clusters) == 0:
        return True
    first = clusters[0]
    return isinstance(first, (set, frozenset, list, tuple))


def pairwise_cluster_metrics(
    predicted_clusters: Union[Sequence[Collection[Any]], Sequence[Any]],
    ground_truth_clusters: Union[Sequence[Collection[Any]], Sequence[Any]],
) -> Dict[str, Any]:
    """Computes pairwise precision, recall, F1, and Rand Index for clustering evaluation.

    Supports two input representations:
    1. Cluster Partitions: Sequence of clusters where each cluster is a collection
       of item identifiers (e.g. ``List[Set[str]]`` or ``List[List[str]]``).
    2. Flat Cluster Labels: Sequence of cluster labels aligned by item index
       (e.g. ``List[int]`` or ``List[str]``).

    Pairwise definitions:
    - True Positive (TP): Both items belong to the same predicted cluster
      AND the same reference cluster.
    - False Positive (FP): Both items belong to the same predicted cluster
      but DIFFERENT reference clusters.
    - False Negative (FN): Both items belong to the same reference cluster
      but DIFFERENT predicted clusters.
    - True Negative (TN): Both items are separated in both clusterings.

    Conventions for edge and degenerate cases:
    - When there are no predicted-positive pairs (TP + FP == 0), pairwise precision is 0.0.
    - When there are no reference-positive pairs (TP + FN == 0), pairwise recall is 0.0.
    - When precision + recall == 0, pairwise F1 is 0.0.
    - When total pairs is 0 (N = 0 or N = 1 items), pairwise precision, recall,
      and F1 are 0.0, and Rand Index is 1.0 (vacuous agreement, matching scikit-learn).
    - If cluster IDs differ between predicted and ground truth, only pairwise
      co-membership relationships are compared. Cluster IDs themselves are not assumed to match.

    Validation rules:
    - For flat labels: lengths of ``predicted_clusters`` and ``ground_truth_clusters``
      must match, and labels cannot be None.
    - For cluster partitions: item universes must match exactly (no missing or extra items),
      and clusters within a partition must be disjoint.

    Returns:
        Dict containing:
        - ``pairwise_precision``: float
        - ``pairwise_recall``: float
        - ``pairwise_f1``: float
        - ``f1``: float (alias for pairwise_f1)
        - ``rand_index``: float
        - ``true_positives``: int
        - ``false_positives``: int
        - ``false_negatives``: int
        - ``true_negatives``: int
    """
    if predicted_clusters is None or ground_truth_clusters is None:
        raise ValueError("predicted_clusters and ground_truth_clusters cannot be None.")

    is_pred_coll = _is_cluster_collection(predicted_clusters)
    is_gt_coll = _is_cluster_collection(ground_truth_clusters)

    # Convert flat labels to cluster partitions if both are flat label sequences
    if not is_pred_coll and not is_gt_coll:
        if len(predicted_clusters) != len(ground_truth_clusters):
            raise ValueError(
                f"Mismatched input lengths: predicted has {len(predicted_clusters)} labels, "
                f"ground truth has {len(ground_truth_clusters)} labels."
            )
        # Check for missing labels (None)
        for idx, lbl in enumerate(predicted_clusters):
            if lbl is None:
                raise ValueError(f"Missing label encountered (None) at index {idx} in predicted labels.")
        for idx, lbl in enumerate(ground_truth_clusters):
            if lbl is None:
                raise ValueError(f"Missing label encountered (None) at index {idx} in ground truth labels.")

        # Map indices to clusters
        pred_map: Dict[Any, Set[int]] = {}
        gt_map: Dict[Any, Set[int]] = {}
        for idx, (p_lbl, g_lbl) in enumerate(zip(predicted_clusters, ground_truth_clusters)):
            pred_map.setdefault(p_lbl, set()).add(idx)
            gt_map.setdefault(g_lbl, set()).add(idx)

        pred_partitions = list(pred_map.values())
        gt_partitions = list(gt_map.values())
    elif is_pred_coll and is_gt_coll:
        pred_partitions = [set(c) for c in predicted_clusters if len(c) > 0]
        gt_partitions = [set(c) for c in ground_truth_clusters if len(c) > 0]

        # Validate disjointness within partitions
        pred_seen: Set[Any] = set()
        for c in pred_partitions:
            overlap = pred_seen.intersection(c)
            if overlap:
                raise ValueError(
                    f"Item(s) {sorted(list(overlap))} appear in multiple predicted clusters; "
                    f"partitions must be disjoint."
                )
            pred_seen.update(c)

        gt_seen: Set[Any] = set()
        for c in gt_partitions:
            overlap = gt_seen.intersection(c)
            if overlap:
                raise ValueError(
                    f"Item(s) {sorted(list(overlap))} appear in multiple ground truth clusters; "
                    f"partitions must be disjoint."
                )
            gt_seen.update(c)

        # Validate identical universe of items
        if pred_seen != gt_seen:
            missing_in_pred = sorted(list(gt_seen - pred_seen), key=str)
            missing_in_gt = sorted(list(pred_seen - gt_seen), key=str)
            raise ValueError(
                f"Mismatched items between predicted and ground truth clusters. "
                f"Missing in predicted: {missing_in_pred}; "
                f"Missing in ground truth: {missing_in_gt}."
            )
    else:
        raise ValueError(
            "Incompatible input types: one argument is a collection of clusters, "
            "while the other is a sequence of flat labels."
        )

    # Total items
    items_count = sum(len(c) for c in pred_partitions)
    total_pairs = _comb2(items_count)

    # Empty or single-item degenerate cases
    if items_count < 2 or total_pairs == 0:
        return {
            "pairwise_precision": 0.0,
            "pairwise_recall": 0.0,
            "pairwise_f1": 0.0,
            "f1": 0.0,
            "rand_index": 1.0,
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "true_negatives": 0,
        }

    # Sum of n_choose_2 for predicted clusters (TP + FP)
    sum_comb_pred = sum(_comb2(len(c)) for c in pred_partitions)

    # Sum of n_choose_2 for ground truth clusters (TP + FN)
    sum_comb_gt = sum(_comb2(len(c)) for c in gt_partitions)

    # True positives: sum over all (i, j) of n_choose_2(|C_i cap G_j|)
    tp = 0
    for p_cluster in pred_partitions:
        for g_cluster in gt_partitions:
            intersection_size = len(p_cluster.intersection(g_cluster))
            if intersection_size >= 2:
                tp += _comb2(intersection_size)

    fp = sum_comb_pred - tp
    fn = sum_comb_gt - tp
    tn = total_pairs - tp - fp - fn

    precision = precision_score(tp, fp)
    recall = recall_score(tp, fn)
    f1 = f1_score(precision, recall)
    rand_index = rand_index_score(tp, tn, fp, fn)

    return {
        "pairwise_precision": precision,
        "pairwise_recall": recall,
        "pairwise_f1": f1,
        "f1": f1,
        "rand_index": rand_index,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
    }

