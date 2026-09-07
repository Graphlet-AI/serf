"""Baseline matchers for entity resolution -- no LLM calls, for comparison.

Per docs/RESEARCH_LOOP.md Stage 1: get the evaluation harness producing
numbers you trust before anything else matters, and report these baselines
"all of them, forever" alongside every SERF result. Implementations are
vectorized (TF-IDF via sparse cosine similarity, random via sampling
indices rather than iterating pairs) so they scale to the largest benchmark
(DBLP-Scholar: 2,616 x 64,263 = ~168M candidate pairs).
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from serf.dspy.types import Entity
from serf.eval.metrics import evaluate_resolution


def random_baseline(
    left: list[Entity],
    right: list[Entity],
    true_pairs: set[tuple[int, int]],
    seed: int = 0,
) -> dict[str, float | int]:
    """Predict matches uniformly at random, at the observed positive rate.

    Establishes the floor every other baseline, and SERF itself, must clear.

    Parameters
    ----------
    left : list[Entity]
        Left-table entities
    right : list[Entity]
        Right-table entities
    true_pairs : set[tuple[int, int]]
        Ground truth matching (left_id, right_id) pairs
    seed : int
        Random seed

    Returns
    -------
    dict[str, float | int]
        precision, recall, f1_score, true_positives, false_positives
    """
    rng = np.random.default_rng(seed)
    total_possible = len(left) * len(right)
    positive_rate = len(true_pairs) / total_possible if total_possible else 0.0
    k = (
        min(int(rng.binomial(total_possible, positive_rate)), total_possible)
        if total_possible
        else 0
    )
    flat_indices = (
        rng.choice(total_possible, size=k, replace=False) if k else np.array([], dtype=np.int64)
    )

    n_right = len(right)
    predicted = {
        (left[int(idx) // n_right].id, right[int(idx) % n_right].id) for idx in flat_indices
    }
    return evaluate_resolution(predicted, true_pairs)


def exact_name_match_baseline(
    left: list[Entity],
    right: list[Entity],
    true_pairs: set[tuple[int, int]],
) -> dict[str, float | int]:
    """Predict a match wherever the normalized name field is identical.

    Shockingly strong on DBLP-ACM; publishing it keeps everyone honest.

    Parameters
    ----------
    left : list[Entity]
        Left-table entities
    right : list[Entity]
        Right-table entities
    true_pairs : set[tuple[int, int]]
        Ground truth matching (left_id, right_id) pairs

    Returns
    -------
    dict[str, float | int]
        precision, recall, f1_score, true_positives, false_positives
    """
    right_by_name: dict[str, list[int]] = {}
    for r in right:
        right_by_name.setdefault(r.name.strip().lower(), []).append(r.id)

    predicted: set[tuple[int, int]] = set()
    for left_entity in left:
        for rid in right_by_name.get(left_entity.name.strip().lower(), []):
            predicted.add((left_entity.id, rid))
    return evaluate_resolution(predicted, true_pairs)


def tfidf_cosine_baseline(
    left: list[Entity],
    right: list[Entity],
    true_pairs: set[tuple[int, int]],
    threshold: float = 0.5,
) -> dict[str, float | int]:
    """Predict a match wherever TF-IDF cosine similarity of names exceeds threshold.

    The pre-neural baseline every reviewer asks about. Computed as one sparse
    matrix multiplication rather than a pairwise Python loop.

    Parameters
    ----------
    left : list[Entity]
        Left-table entities
    right : list[Entity]
        Right-table entities
    true_pairs : set[tuple[int, int]]
        Ground truth matching (left_id, right_id) pairs
    threshold : float
        Minimum cosine similarity to count as a match

    Returns
    -------
    dict[str, float | int]
        precision, recall, f1_score, true_positives, false_positives
    """
    corpus = [e.name for e in left] + [e.name for e in right]
    vectorizer = TfidfVectorizer(analyzer="word", token_pattern=r"(?u)\b\w+\b")
    tfidf = vectorizer.fit_transform(corpus)
    left_vecs, right_vecs = tfidf[: len(left)], tfidf[len(left) :]
    sims = cosine_similarity(left_vecs, right_vecs)

    left_idx, right_idx = np.where(sims >= threshold)
    predicted = {(left[li].id, right[ri].id) for li, ri in zip(left_idx, right_idx, strict=True)}
    return evaluate_resolution(predicted, true_pairs)
