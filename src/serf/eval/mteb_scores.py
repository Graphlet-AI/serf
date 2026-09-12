"""Score embedding candidates on an MTEB task category and rank them.

The large-model candidates were picked by MTEB(eng, v2) *clustering* score, on
the reasoning that blocking clusters records. Measured blocking recall did not
follow that order, which leaves a better question: does any MTEB category order
candidates the way blocking recall does?

``PairClassification`` is the matching category. SprintDuplicateQuestions,
TwitterSemEval2015 and TwitterURLCorpus all ask whether two short texts denote
the same thing, which is the entity matching decision itself, scored by average
precision over cosine similarity. That is much closer to what blocking needs
than partitioning a corpus into topics.

Scores come from the ``mteb/results`` dataset on the Hub, which publishes one
main score per model, revision, task, split and subset. The leaderboard Space
renders client-side, so the dataset is the only readable source.
"""

from dataclasses import dataclass
from typing import Any

import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files

from serf.config import config
from serf.logs import get_logger

logger = get_logger(__name__)

RESULTS_REPO = "mteb/results"
SCORE_COLUMNS = ["model_name", "task_name", "split", "subset", "score"]

# Every task in the categories we read is reported on the test split under the
# default subset. Restricting to those avoids averaging a validation score or a
# per-language subset into a headline number.
SCORE_SPLIT = "test"
SCORE_SUBSET = "default"


@dataclass(frozen=True)
class MtebCategoryScore:
    """One model's mean score over the tasks in one MTEB category.

    Parameters
    ----------
    model : str
        Hugging Face model name
    category : str
        MTEB task category, such as ``"PairClassification"``
    score : float
        Mean main score over the category's tasks, on a 0-100 scale
    tasks_found : int
        Tasks in the category with a published score for this model
    tasks_expected : int
        Tasks configured for the category
    """

    model: str
    category: str
    score: float
    tasks_found: int
    tasks_expected: int

    @property
    def complete(self) -> bool:
        """Return whether every configured task was found."""
        return self.tasks_found == self.tasks_expected


def category_tasks(category: str) -> list[str]:
    """Return the configured MTEB task names for one category.

    Parameters
    ----------
    category : str
        Category key under ``benchmarks.mteb_tasks``

    Returns
    -------
    list[str]
        Task names, empty when the category is not configured
    """
    tasks = config.get(f"benchmarks.mteb_tasks.{category}", [])
    return [str(task) for task in tasks]


def configured_categories() -> list[str]:
    """Return every configured MTEB category name.

    Returns
    -------
    list[str]
        Category keys under ``benchmarks.mteb_tasks``
    """
    categories = config.get("benchmarks.mteb_tasks", {})
    return sorted(str(key) for key in categories)


def load_task_scores(tasks: list[str]) -> dict[str, dict[str, float]]:
    """Read published MTEB scores for a set of tasks.

    A model can appear under several revisions. The best published score per
    task is kept, which is the number the leaderboard shows.

    Parameters
    ----------
    tasks : list[str]
        MTEB task names to keep

    Returns
    -------
    dict[str, dict[str, float]]
        Model name to task name to score, on the source 0-1 scale
    """
    wanted = sorted(set(tasks))
    parts = sorted(
        name for name in list_repo_files(RESULTS_REPO, repo_type="dataset") if ".parquet" in name
    )
    logger.info(f"Reading MTEB scores for {len(wanted)} tasks from {len(parts)} parquet parts")

    best: dict[str, dict[str, float]] = {}
    for part in parts:
        path = hf_hub_download(RESULTS_REPO, part, repo_type="dataset")
        frame = pd.read_parquet(path, columns=SCORE_COLUMNS)
        rows = frame[
            frame["task_name"].isin(wanted)
            & (frame["split"] == SCORE_SPLIT)
            & (frame["subset"] == SCORE_SUBSET)
        ]
        for model, task, score in zip(
            rows["model_name"].tolist(),
            rows["task_name"].tolist(),
            rows["score"].tolist(),
            strict=True,
        ):
            by_task = best.setdefault(str(model), {})
            value = float(score)
            if value > by_task.get(str(task), float("-inf")):
                by_task[str(task)] = value

    return best


def score_models(models: list[str], category: str) -> list[MtebCategoryScore]:
    """Score models on one MTEB category, best first.

    Parameters
    ----------
    models : list[str]
        Hugging Face model names
    category : str
        Category key under ``benchmarks.mteb_tasks``

    Returns
    -------
    list[MtebCategoryScore]
        One entry per model that has at least one published task score,
        ordered by score descending
    """
    tasks = category_tasks(category)
    if not tasks:
        logger.warning(f"No tasks configured for MTEB category {category}")
        return []

    published = load_task_scores(tasks)
    ranked: list[MtebCategoryScore] = []
    for model in models:
        by_task = published.get(model, {})
        if not by_task:
            continue
        ranked.append(
            MtebCategoryScore(
                model=model,
                category=category,
                score=round(100 * sum(by_task.values()) / len(by_task), 2),
                tasks_found=len(by_task),
                tasks_expected=len(tasks),
            )
        )

    missing = set(models) - {entry.model for entry in ranked}
    if missing:
        logger.warning(f"No published {category} score for: {', '.join(sorted(missing))}")

    return sorted(ranked, key=lambda entry: entry.score, reverse=True)


def _ranks(values: list[float]) -> list[float]:
    """Return average ranks, so ties share a rank.

    Parameters
    ----------
    values : list[float]
        Values to rank, largest first

    Returns
    -------
    list[float]
        Rank per value in the input order
    """
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
            end += 1
        shared = (position + end) / 2 + 1
        for index in order[position : end + 1]:
            ranks[index] = shared
        position = end + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float:
    """Return the Spearman rank correlation between two series.

    Rank correlation is the right measure here because the question is whether
    an MTEB category *orders* candidates the way blocking recall does, not
    whether the two scores share a scale.

    Parameters
    ----------
    xs : list[float]
        First series
    ys : list[float]
        Second series, same length and order

    Returns
    -------
    float
        Correlation in ``[-1, 1]``, or 0.0 when either series is constant or
        there are fewer than two points
    """
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0

    rank_x = _ranks(xs)
    rank_y = _ranks(ys)
    mean_x = sum(rank_x) / len(rank_x)
    mean_y = sum(rank_y) / len(rank_y)

    covariance = float(
        sum((a - mean_x) * (b - mean_y) for a, b in zip(rank_x, rank_y, strict=True))
    )
    spread_x = float(sum((a - mean_x) ** 2 for a in rank_x)) ** 0.5
    spread_y = float(sum((b - mean_y) ** 2 for b in rank_y)) ** 0.5
    if spread_x == 0.0 or spread_y == 0.0:
        return 0.0
    return float(round(covariance / (spread_x * spread_y), 4))


def measured_recall(sweep: list[dict[str, Any]], strategy: str = "name") -> dict[str, float]:
    """Average blocking recall per model from a ``serf blocking-sweep`` result.

    A sweep can score the same model under more than one instruction prefix.
    MTEB publishes one number per model, so the best prefix is taken per
    dataset before averaging, which compares the model rather than the prompt.

    Parameters
    ----------
    sweep : list[dict[str, Any]]
        Parsed JSON written by ``serf blocking-sweep --output``
    strategy : str
        Blocking strategy to read, so a sweep covering several stays comparable

    Returns
    -------
    dict[str, float]
        Model name to mean blocking recall over the datasets it was scored on
    """
    best: dict[tuple[str, str], float] = {}
    for row in sweep:
        if str(row.get("strategy", "name")) != strategy:
            continue
        if int(row.get("round_number", 1)) != 1:
            continue
        key = (str(row["model"]), str(row["dataset"]))
        recall = float(row["blocking_recall"])
        if recall > best.get(key, float("-inf")):
            best[key] = recall

    totals: dict[str, list[float]] = {}
    for (model, _dataset), recall in best.items():
        totals.setdefault(model, []).append(recall)
    return {model: sum(values) / len(values) for model, values in totals.items()}


def correlate_categories(
    recalls: dict[str, float], categories: list[str] | None = None
) -> list[tuple[str, float, int]]:
    """Rank MTEB categories by how well they predict measured blocking recall.

    Parameters
    ----------
    recalls : dict[str, float]
        Model name to measured mean blocking recall
    categories : list[str] | None
        Categories to test. Defaults to every configured category.

    Returns
    -------
    list[tuple[str, float, int]]
        ``(category, spearman, models_compared)`` ordered by correlation
        descending
    """
    models = sorted(recalls)
    out: list[tuple[str, float, int]] = []
    for category in categories or configured_categories():
        scored = {entry.model: entry.score for entry in score_models(models, category)}
        shared = [model for model in models if model in scored]
        if len(shared) < 2:
            logger.warning(f"Too few models with a {category} score to correlate")
            continue
        out.append(
            (
                category,
                spearman([scored[model] for model in shared], [recalls[model] for model in shared]),
                len(shared),
            )
        )
    return sorted(out, key=lambda entry: entry[1], reverse=True)
