from typing import List, Set

def precision_score(true_positives: int, false_positives: int) -> float:
    total = true_positives + false_positives
    return true_positives / total if total > 0 else 0.0

def recall_score(true_positives: int, false_negatives: int) -> float:
    total = true_positives + false_negatives
    return true_positives / total if total > 0 else 0.0

def f1_score(precision: float, recall: float) -> float:
    total = precision + recall
    return (2 * precision * recall) / total if total > 0 else 0.0

def pairwise_cluster_metrics(predicted_clusters: List[Set[str]], ground_truth_clusters: List[Set[str]]):
    """
    Computes pairwise precision, recall, and F1 for clustering evaluation.
    This is standard for evaluating discovery quality where we don't have labeled cluster IDs.
    """
    # Implementation placeholder for evaluation
    pass
