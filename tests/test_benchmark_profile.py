"""Tests for the Spark SQL benchmark profiler."""

from typing import Any

import pandas as pd
import pytest

from serf.analyze.benchmarks import (
    _column_stats_sql,
    _markdown_table,
    _render_example,
    format_profile,
    profile_benchmark,
)
from serf.eval.benchmarks import RIGHT_ID_OFFSET, BenchmarkDataset


@pytest.fixture(scope="module")
def synthetic_profile(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Profile a tiny hand-built dataset with known answers.

    The four pairs are chosen so that every branch of the report has something
    to say: one match that string equality finds, one that it misses, one that
    only a model code links, and one non-match whose titles are identical.
    """
    table_a = pd.DataFrame(
        [
            {
                "id": "a0",
                "title": "Fast Query Processing Engine",
                "venue": "SIGMOD",
                "year": "1999",
            },
            {
                "id": "a1",
                "title": "Deep Learning for Entity Matching",
                "venue": "VLDB",
                "year": "2018",
            },
            {"id": "a2", "title": "Editors Notes Column Spring", "venue": "SIGMOD", "year": "2001"},
            {"id": "a3", "title": "Widget X100 Pro Adapter", "venue": "SIGMOD", "year": "2005"},
            {"id": "a4", "title": "Fast Query Rewriting Rules", "venue": "VLDB", "year": "2009"},
        ]
    )
    table_b = pd.DataFrame(
        [
            {
                "id": "b0",
                "title": "fast query processing engine",
                "venue": "Management of Data",
                "year": "1999",
            },
            {
                "id": "b1",
                "title": "Deep Learning for Matching Entities in Databases",
                "venue": "Very Large Data Bases",
                "year": "2018",
            },
            {
                "id": "b2",
                "title": "Editors Notes Column Spring",
                "venue": "Management of Data",
                "year": "1995",
            },
            {
                "id": "b3",
                "title": "Widget X-100 Pro Adapter",
                "venue": "Management of Data",
                "year": "2005",
            },
            {
                "id": "b4",
                "title": "Slow Query Processing Survey",
                "venue": "Very Large Data Bases",
                "year": "2012",
            },
        ]
    )
    ground_truth = {(i, i + RIGHT_ID_OFFSET) for i in (0, 1, 3)}
    dataset = BenchmarkDataset(
        name="synthetic",
        table_a=table_a,
        table_b=table_b,
        ground_truth=ground_truth,
        metadata={"domain": "test", "difficulty": "easy"},
    )

    def fake_download(name: str, output_dir: str | None = None) -> BenchmarkDataset:
        return dataset

    original = BenchmarkDataset.download
    BenchmarkDataset.download = staticmethod(fake_download)  # type: ignore[method-assign]
    try:
        return profile_benchmark("synthetic", data_dir=str(tmp_path_factory.mktemp("data")))
    finally:
        BenchmarkDataset.download = original  # type: ignore[method-assign]


def test_column_stats_sql_covers_every_column() -> None:
    """The generated query measures each column once."""
    sql = _column_stats_sql("a", ["title", "year"])
    assert sql.count("UNION ALL") == 1
    assert "'title' AS column_name" in sql
    assert "'year' AS column_name" in sql
    assert "AS discriminativeness" in sql


def test_markdown_table_escapes_pipes() -> None:
    """A value containing a pipe cannot break out of its cell."""
    lines = _markdown_table([{"v": "a|b"}], ["v"])
    assert lines[0] == "| v |"
    assert lines[2] == "| a\\|b |"


def test_render_example_falls_back_to_the_name_column() -> None:
    """A pair with no shared attributes still renders its names."""
    lines = _render_example({"jaccard": 0.5, "left_key": "Widget", "right_key": "Widget Pro"}, [])
    assert lines[0] == "- Jaccard 0.500"
    assert "name=`Widget`" in lines[1]
    assert "name=`Widget Pro`" in lines[2]


def test_profile_reports_shape_and_gold_pairs(synthetic_profile: dict[str, Any]) -> None:
    """Shapes, shared columns and gold pair counts come through unchanged."""
    assert synthetic_profile["rows_a"] == 5
    assert synthetic_profile["rows_b"] == 5
    assert synthetic_profile["gold_pairs"] == 3
    assert synthetic_profile["shared_columns"] == ["title", "venue", "year"]
    assert synthetic_profile["name_column_a"] == "title"


def test_profile_scores_discriminativeness(synthetic_profile: dict[str, Any]) -> None:
    """Title is unique and complete, venue is neither, so title scores higher."""
    scores = synthetic_profile["discriminativeness"]
    assert scores["title"] > scores["venue"]
    # unique 1.0 plus present 1.0 on both sides, so 2.0 * 2.0.
    assert scores["title"] == pytest.approx(4.0)


def test_profile_measures_cardinality(synthetic_profile: dict[str, Any]) -> None:
    """Every gold pair here is strictly one to one."""
    card = synthetic_profile["cardinality"]
    assert card["matched_left"] == 3
    assert card["max_matches_per_left"] == 1
    assert card["one_to_one"] == pytest.approx(1.0)


def test_profile_separates_agreement_on_matches_from_near_misses(
    synthetic_profile: dict[str, Any],
) -> None:
    """Year agrees on every match and on none of the near misses."""
    assert synthetic_profile["agreement_on_matches"]["pairs"] == 3
    assert synthetic_profile["agreement_on_matches"]["agree_year"] == pytest.approx(1.0)
    assert synthetic_profile["agreement_on_near_misses"]["pairs"] > 0
    assert synthetic_profile["agreement_on_near_misses"]["agree_year"] == pytest.approx(0.0)


def test_profile_builds_a_venue_crosswalk(synthetic_profile: dict[str, Any]) -> None:
    """Venue is a small controlled vocabulary, so it gets a crosswalk."""
    crosswalk = synthetic_profile["crosswalks"]["venue"]
    pairs = {(row["a_value"], row["b_value"]) for row in crosswalk}
    assert ("SIGMOD", "Management of Data") in pairs
    assert ("VLDB", "Very Large Data Bases") in pairs


def test_profile_finds_a_code_that_punctuation_hides(
    synthetic_profile: dict[str, Any],
) -> None:
    """`X100` is only found inside `X-100` once separators are stripped."""
    codes = synthetic_profile["code_overlap"]
    assert (
        codes["matches"]["one_code_contained_in_the_other"]
        > (codes["matches"]["share_a_code_exactly"])
    )


def test_profile_surfaces_the_identical_title_non_match(
    synthetic_profile: dict[str, Any],
) -> None:
    """The two `Editors Notes` rows are a non-match string similarity accepts."""
    top = synthetic_profile["hard_non_matches"][0]
    assert top["jaccard"] == pytest.approx(1.0)
    assert top["a_title"] == "Editors Notes Column Spring"
    assert top["b_title"] == "Editors Notes Column Spring"
    assert top["a_year"] != top["b_year"]


def test_profile_treats_a_complete_mapping_as_exhaustive(
    synthetic_profile: dict[str, Any],
) -> None:
    """With no label files, every non-gold pair is a confirmed non-match."""
    assert synthetic_profile["ground_truth_kind"] == "complete mapping"
    assert synthetic_profile["near_misses_confirmed_negative"] == pytest.approx(1.0)


def test_format_profile_renders_every_section(synthetic_profile: dict[str, Any]) -> None:
    """The Markdown report carries each analysis the profile computed."""
    report = format_profile(synthetic_profile)
    for heading in (
        "## synthetic",
        "### Columns",
        "### Common values",
        "### Gold pairs",
        "### Attribute agreement",
        "### Model codes",
        "### Controlled vocabulary crosswalk",
        "### Matches string similarity misses",
        "### Non-matches string similarity accepts",
    ):
        assert heading in report
