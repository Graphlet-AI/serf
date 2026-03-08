"""Evaluation metrics for entity resolution."""

from serf.logs import get_logger

logger = get_logger(__name__)


def _normalize_pairs(pairs: set[tuple[int, int]]) -> set[tuple[int, int]]:
    """Normalize pairs so (a, b) with a < b for consistent ordering.

    Parameters
    ----------
    pairs : set of tuple of (int, int)
        Pairs of entity IDs, possibly in arbitrary order.

    Returns
    -------
    set of tuple of (int, int)
        Pairs normalized with smaller ID first.
    """
    return {(min(a, b), max(a, b)) for a, b in pairs}


def precision(predicted_pairs: set[tuple[int, int]], true_pairs: set[tuple[int, int]]) -> float:
    """Fraction of predicted matches that are true matches.

    Parameters
    ----------
    predicted_pairs : set of tuple of (int, int)
        Pairs predicted as matches.
    true_pairs : set of tuple of (int, int)
        Ground truth matching pairs.

    Returns
    -------
    float
        Precision score in [0, 1]. Returns 0.0 if predicted_pairs is empty.
    """
    pred = _normalize_pairs(predicted_pairs)
    true = _normalize_pairs(true_pairs)
    if not pred:
        return 0.0
    return len(pred & true) / len(pred)


def recall(predicted_pairs: set[tuple[int, int]], true_pairs: set[tuple[int, int]]) -> float:
    """Fraction of true matches that were found.

    Parameters
    ----------
    predicted_pairs : set of tuple of (int, int)
        Pairs predicted as matches.
    true_pairs : set of tuple of (int, int)
        Ground truth matching pairs.

    Returns
    -------
    float
        Recall score in [0, 1]. Returns 0.0 if true_pairs is empty.
    """
    pred = _normalize_pairs(predicted_pairs)
    true = _normalize_pairs(true_pairs)
    if not true:
        return 0.0
    return len(pred & true) / len(true)


def f1_score(predicted_pairs: set[tuple[int, int]], true_pairs: set[tuple[int, int]]) -> float:
    """Harmonic mean of precision and recall.

    Parameters
    ----------
    predicted_pairs : set of tuple of (int, int)
        Pairs predicted as matches.
    true_pairs : set of tuple of (int, int)
        Ground truth matching pairs.

    Returns
    -------
    float
        F1 score in [0, 1]. Returns 0.0 when precision+recall is 0.
    """
    p = precision(predicted_pairs, true_pairs)
    r = recall(predicted_pairs, true_pairs)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def pair_completeness(
    blocked_pairs: set[tuple[int, int]], true_pairs: set[tuple[int, int]]
) -> float:
    """Fraction of true match pairs retained after blocking.

    Parameters
    ----------
    blocked_pairs : set of tuple of (int, int)
        Pairs retained after blocking.
    true_pairs : set of tuple of (int, int)
        Ground truth matching pairs.

    Returns
    -------
    float
        Pair completeness in [0, 1]. Returns 0.0 if true_pairs is empty.
    """
    blocked = _normalize_pairs(blocked_pairs)
    true = _normalize_pairs(true_pairs)
    if not true:
        return 0.0
    return len(blocked & true) / len(true)


def reduction_ratio(num_blocked_pairs: int, total_possible_pairs: int) -> float:
    """1 - (pairs after blocking / total possible pairs).

    Parameters
    ----------
    num_blocked_pairs : int
        Number of pairs retained after blocking.
    total_possible_pairs : int
        Total number of possible pairs before blocking.

    Returns
    -------
    float
        Reduction ratio in [0, 1]. Returns 0.0 if total_possible_pairs is 0.
    """
    if total_possible_pairs == 0:
        return 0.0
    return 1.0 - (num_blocked_pairs / total_possible_pairs)


def _clusters_to_pairs(clusters: dict[int, set[int]]) -> set[tuple[int, int]]:
    """Extract all pairwise links from clusters."""
    pairs: set[tuple[int, int]] = set()
    for entities in clusters.values():
        entities_list = list(entities)
        for i in range(len(entities_list)):
            for j in range(i + 1, len(entities_list)):
                a, b = entities_list[i], entities_list[j]
                pairs.add((min(a, b), max(a, b)))
    return pairs


def cluster_f1(
    predicted_clusters: dict[int, set[int]], true_clusters: dict[int, set[int]]
) -> float:
    """F1 computed at the cluster level using pairwise comparisons within clusters.

    Parameters
    ----------
    predicted_clusters : dict of int to set of int
        Predicted clusters: cluster_id -> set of entity IDs.
    true_clusters : dict of int to set of int
        Ground truth clusters: cluster_id -> set of entity IDs.

    Returns
    -------
    float
        Cluster-level F1 score in [0, 1].
    """
    pred_pairs = _clusters_to_pairs(predicted_clusters)
    true_pairs = _clusters_to_pairs(true_clusters)
    return f1_score(pred_pairs, true_pairs)


def evaluate_resolution(
    predicted_pairs: set[tuple[int, int]], true_pairs: set[tuple[int, int]]
) -> dict[str, float]:
    """Compute all metrics and return as a dict.

    Parameters
    ----------
    predicted_pairs : set of tuple of (int, int)
        Pairs predicted as matches.
    true_pairs : set of tuple of (int, int)
        Ground truth matching pairs.

    Returns
    -------
    dict of str to float
        Dict with keys: precision, recall, f1_score.
    """
    return {
        "precision": precision(predicted_pairs, true_pairs),
        "recall": recall(predicted_pairs, true_pairs),
        "f1_score": f1_score(predicted_pairs, true_pairs),
    }
