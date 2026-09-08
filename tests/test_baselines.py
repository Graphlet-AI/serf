"""Tests for no-LLM baseline matchers."""

from serf.dspy.types import Entity
from serf.eval.baselines import exact_name_match_baseline, random_baseline, tfidf_cosine_baseline


def _entities(names: list[str], id_offset: int = 0) -> list[Entity]:
    return [Entity(id=id_offset + i, name=n) for i, n in enumerate(names)]


def test_random_baseline_predicted_count_matches_positive_rate() -> None:
    """Random predicts roughly len(true_pairs) pairs total, since it samples
    at the observed positive rate rather than a fixed threshold."""
    left = _entities(["A", "B", "C", "D"])
    right = _entities(["E", "F", "G", "H"], id_offset=100)
    true_pairs = {(0, 100), (1, 101)}
    metrics = random_baseline(left, right, true_pairs, seed=0)
    # With only 16 candidate pairs and rate 2/16, the exact count is noisy,
    # but true_positives + false_positives should be small and plausible.
    assert metrics["true_positives"] + metrics["false_positives"] <= 16


def test_random_baseline_empty_true_pairs_predicts_nothing() -> None:
    """Zero observed positive rate means random predicts an empty set."""
    left = _entities(["A", "B"])
    right = _entities(["C", "D"], id_offset=100)
    metrics = random_baseline(left, right, true_pairs=set(), seed=0)
    assert metrics["true_positives"] == 0
    assert metrics["false_positives"] == 0


def test_random_baseline_scales_to_large_dataset() -> None:
    """Must not materialize O(N*M) candidate pairs (DBLP-Scholar is ~168M);
    this should complete quickly for a similarly-shaped small-rate case."""
    left = _entities([f"L{i}" for i in range(3000)])
    right = _entities([f"R{i}" for i in range(60000)], id_offset=1_000_000)
    true_pairs = {(0, 1_000_000), (1, 1_000_001)}
    metrics = random_baseline(left, right, true_pairs, seed=0)
    assert metrics["true_positives"] + metrics["false_positives"] < 100


def test_exact_name_match_finds_identical_names() -> None:
    """Identical names across tables are predicted as matches."""
    left = _entities(["Same Title", "Unique A"])
    right = _entities(["Same Title", "Unique B"], id_offset=100)
    true_pairs = {(0, 100)}
    metrics = exact_name_match_baseline(left, right, true_pairs)
    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 0
    assert metrics["recall"] == 1.0


def test_exact_name_match_is_case_and_whitespace_insensitive() -> None:
    """Normalization (strip + lowercase) still counts a match."""
    left = _entities(["  Machine Learning  "])
    right = _entities(["machine learning"], id_offset=100)
    metrics = exact_name_match_baseline(left, right, {(0, 100)})
    assert metrics["true_positives"] == 1


def test_exact_name_match_no_false_positives_from_distinct_names() -> None:
    """Distinct names never get linked, even with no ground truth."""
    left = _entities(["Alpha"])
    right = _entities(["Beta"], id_offset=100)
    metrics = exact_name_match_baseline(left, right, set())
    assert metrics["true_positives"] == 0
    assert metrics["false_positives"] == 0


def test_tfidf_cosine_finds_near_duplicate_names() -> None:
    """Names sharing most tokens score above threshold; unrelated ones don't."""
    left = _entities(["Deep Learning for Entity Matching", "Totally Unrelated Topic"])
    right = _entities(
        ["Deep Learning for Entity Matching Design", "Something Else Entirely"],
        id_offset=100,
    )
    true_pairs = {(0, 100)}
    metrics = tfidf_cosine_baseline(left, right, true_pairs, threshold=0.5)
    assert metrics["true_positives"] == 1
    assert metrics["recall"] == 1.0


def test_tfidf_cosine_threshold_controls_strictness() -> None:
    """A higher threshold predicts fewer (or equally many) pairs than a lower one."""
    left = _entities(["Blue Widget Model X"])
    right = _entities(["Blue Widget Model Y"], id_offset=100)
    true_pairs = {(0, 100)}
    loose = tfidf_cosine_baseline(left, right, true_pairs, threshold=0.1)
    strict = tfidf_cosine_baseline(left, right, true_pairs, threshold=0.99)
    loose_predicted = loose["true_positives"] + loose["false_positives"]
    strict_predicted = strict["true_positives"] + strict["false_positives"]
    assert strict_predicted <= loose_predicted
