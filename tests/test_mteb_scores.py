"""Tests for scoring embedding candidates against MTEB categories."""

from unittest.mock import patch

from serf.eval.mteb_scores import (
    category_tasks,
    configured_categories,
    correlate_categories,
    measured_recall,
    score_models,
    spearman,
)

PUBLISHED = {
    "org/good": {
        "SprintDuplicateQuestions": 0.96,
        "TwitterSemEval2015": 0.80,
        "TwitterURLCorpus": 0.88,
    },
    "org/poor": {
        "SprintDuplicateQuestions": 0.90,
        "TwitterSemEval2015": 0.70,
        "TwitterURLCorpus": 0.80,
    },
    "org/partial": {"SprintDuplicateQuestions": 0.99},
}


def _sweep_row(model: str, dataset: str, recall: float, **extra: object) -> dict[str, object]:
    """Build one row of a blocking sweep result."""
    row: dict[str, object] = {
        "model": model,
        "dataset": dataset,
        "blocking_recall": recall,
        "strategy": "name",
        "round_number": 1,
        "prompt": "",
    }
    row.update(extra)
    return row


def test_pair_classification_is_the_matching_category() -> None:
    """The configured matching tasks are the three MTEB pair tasks."""
    tasks = category_tasks("PairClassification")

    assert tasks == ["SprintDuplicateQuestions", "TwitterSemEval2015", "TwitterURLCorpus"]


def test_clustering_is_configured_for_comparison() -> None:
    """Clustering stays configured, since it is the category being displaced."""
    assert len(category_tasks("Clustering")) == 8
    assert "PairClassification" in configured_categories()
    assert "Clustering" in configured_categories()


def test_unknown_category_has_no_tasks() -> None:
    """An unconfigured category yields nothing rather than raising."""
    assert category_tasks("NotACategory") == []


def test_score_models_averages_the_category_on_a_100_scale() -> None:
    """A model's category score is the mean of its task scores."""
    with patch("serf.eval.mteb_scores.load_task_scores", return_value=PUBLISHED):
        scored = score_models(["org/good", "org/poor"], "PairClassification")

    assert [entry.model for entry in scored] == ["org/good", "org/poor"]
    assert scored[0].score == 88.0
    assert scored[1].score == 80.0
    assert scored[0].complete is True


def test_score_models_flags_partial_coverage() -> None:
    """A model missing tasks is still scored, but says so."""
    with patch("serf.eval.mteb_scores.load_task_scores", return_value=PUBLISHED):
        scored = score_models(["org/partial"], "PairClassification")

    assert scored[0].tasks_found == 1
    assert scored[0].tasks_expected == 3
    assert scored[0].complete is False


def test_score_models_skips_models_with_no_published_score() -> None:
    """An unpublished model drops out instead of scoring zero."""
    with patch("serf.eval.mteb_scores.load_task_scores", return_value=PUBLISHED):
        scored = score_models(["org/good", "org/unpublished"], "PairClassification")

    assert [entry.model for entry in scored] == ["org/good"]


def test_score_models_returns_nothing_for_an_unconfigured_category() -> None:
    """No configured tasks means no scores, and no download attempt."""
    with patch("serf.eval.mteb_scores.load_task_scores") as mock_load:
        assert score_models(["org/good"], "NotACategory") == []

    mock_load.assert_not_called()


def test_spearman_is_one_when_the_orders_agree() -> None:
    """Identical orderings correlate perfectly, whatever the scales."""
    assert spearman([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]) == 1.0


def test_spearman_is_minus_one_when_the_orders_invert() -> None:
    """A ranking that inverts the truth is worse than no ranking."""
    assert spearman([1.0, 2.0, 3.0], [30.0, 20.0, 10.0]) == -1.0


def test_spearman_handles_ties_by_sharing_a_rank() -> None:
    """Tied values share a rank rather than taking an arbitrary order."""
    assert spearman([1.0, 1.0, 2.0], [5.0, 5.0, 9.0]) == 1.0


def test_spearman_is_zero_without_enough_points_or_spread() -> None:
    """A constant or too-short series cannot correlate."""
    assert spearman([1.0], [2.0]) == 0.0
    assert spearman([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) == 0.0
    assert spearman([1.0, 2.0], [1.0, 2.0, 3.0]) == 0.0


def test_measured_recall_averages_over_datasets() -> None:
    """A model's measured recall is its mean across the datasets swept."""
    sweep = [
        _sweep_row("org/good", "dblp-acm", 0.9),
        _sweep_row("org/good", "abt-buy", 0.7),
    ]

    assert measured_recall(sweep) == {"org/good": 0.8}


def test_measured_recall_takes_the_best_prompt_per_dataset() -> None:
    """MTEB publishes one number per model, so the best prefix represents it."""
    sweep = [
        _sweep_row("org/good", "dblp-acm", 0.6, prompt=""),
        _sweep_row("org/good", "dblp-acm", 0.9, prompt="query: "),
    ]

    assert measured_recall(sweep) == {"org/good": 0.9}


def test_measured_recall_reads_one_strategy_at_a_time() -> None:
    """A sweep covering several strategies stays comparable."""
    sweep = [
        _sweep_row("org/good", "dblp-acm", 0.9),
        _sweep_row("org/good", "dblp-acm", 0.5, strategy="json"),
    ]

    assert measured_recall(sweep, strategy="name") == {"org/good": 0.9}
    assert measured_recall(sweep, strategy="json") == {"org/good": 0.5}


def test_measured_recall_ignores_later_rounds() -> None:
    """Only the first blocking pass is comparable across models."""
    sweep = [
        _sweep_row("org/good", "dblp-acm", 0.9),
        _sweep_row("org/good", "dblp-acm", 0.4, round_number=2),
    ]

    assert measured_recall(sweep) == {"org/good": 0.9}


def test_correlate_categories_ranks_the_predictive_category_first() -> None:
    """The category whose order matches measured recall comes out on top."""
    recalls = {"org/good": 0.88, "org/poor": 0.80}
    clustering = {"org/good": {"A": 0.4}, "org/poor": {"A": 0.6}}

    def fake_load(tasks: list[str]) -> dict[str, dict[str, float]]:
        return clustering if tasks == category_tasks("Clustering") else PUBLISHED

    with patch("serf.eval.mteb_scores.load_task_scores", side_effect=fake_load):
        ranked = correlate_categories(recalls, ["PairClassification", "Clustering"])

    assert ranked[0] == ("PairClassification", 1.0, 2)
    assert ranked[1] == ("Clustering", -1.0, 2)


def test_correlate_categories_skips_categories_with_too_few_models() -> None:
    """One shared model cannot produce a correlation."""
    with patch("serf.eval.mteb_scores.load_task_scores", return_value=PUBLISHED):
        assert correlate_categories({"org/good": 0.9}, ["PairClassification"]) == []
