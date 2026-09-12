"""Exploratory profiling of the ER benchmark datasets, in Spark SQL.

The questions this answers are the ones you have to answer before you can write
a matching prompt: which attributes carry signal, which are decoration, what the
common values look like, how often the gold pairs agree on each attribute, and
which pairs the obvious string-similarity baseline gets wrong in both directions.

Every number is produced by a SQL query over the two source tables and the gold
mapping, so the same query can be shown to a reader, pasted into another engine,
or handed to an LLM as evidence about the data it is being asked to resolve.
"""

import zipfile
from pathlib import Path
from typing import Any

from pyspark.sql import DataFrame, SparkSession

from serf.config import config
from serf.eval.benchmarks import (
    DATASET_REGISTRY,
    RIGHT_ID_OFFSET,
    BenchmarkDataset,
    _detect_name_column,
    _find_zip_member,
    _load_csv_from_zip,
    _row_id_map,
)
from serf.logs import get_logger

logger = get_logger(__name__)

# Lowercase, drop punctuation, collapse runs of whitespace. Everything that
# compares two values in this module compares them after this normalisation,
# because the raw tables disagree on case and punctuation constantly.
NORMALIZED = "regexp_replace(lower(trim({col})), '[^a-z0-9]+', ' ')"

# Distinct word tokens of at least two characters. Single characters are dropped
# because they are almost all packaging noise ("2 x 4", "vol . 3").
TOKENS = "array_sort(array_distinct(filter(split(trim({expr}), ' +'), t -> length(t) > 1)))"

# Jaccard over those token sets, the string-similarity baseline this module
# measures matches and non-matches against.
JACCARD = "size(array_intersect({a}, {b})) / greatest(size(array_union({a}, {b})), 1)"

# A token held by more than this fraction of a table is a stopword for that
# table ("the", "of", "inc", "cable") and is useless for finding near misses.
MAX_TOKEN_DOC_FRACTION = 0.01

# How many near-miss candidates to keep per left record before ranking.
NEAR_MISS_CANDIDATES = 20

# A shared column is worth a crosswalk only if both sides draw on a small
# controlled vocabulary; above this it is free text and the crosswalk is noise.
MAX_CROSSWALK_CARDINALITY = 25

# A shared column is treated as numeric if at least this fraction of its
# non-empty values on both sides parse as a number.
MIN_NUMERIC_FRACTION = 0.5

# Relative price or quantity gaps that count as "close" and "same ballpark".
NUMERIC_TOLERANCES = (0.05, 0.25)

# Everything except letters and digits removed, so that a model number survives
# whatever spacing and punctuation each retailer chose for it.
SQUASHED = "regexp_replace(lower({col}), '[^a-z0-9]+', '')"

# Tokens long enough and digit-bearing enough to be a model number, SKU, ISBN or
# part code rather than a word.
CODES = (
    "array_distinct(filter(split({expr}, ' +'),"
    " t -> length(t) >= 4 AND t rlike '[0-9]' AND t rlike '[a-z0-9]'))"
)


def _spark_session() -> SparkSession:
    """Build or reuse the local SparkSession used for profiling.

    Returns
    -------
    SparkSession
        Session configured with a small shuffle width, since the benchmark
        tables are tens of thousands of rows and the default 200 partitions
        costs more in scheduling than the work itself.
    """
    return (
        SparkSession.builder.appName("serf-benchmark-profile")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", str(config.get("analyze.spark_driver_memory", "4g")))
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )


def _register(spark: SparkSession, df: Any, view: str) -> DataFrame:
    """Register a pandas table as a Spark view with a stable row index.

    The gold mapping identifies records by row position, not by the source
    ``id`` column, so the row index has to survive into Spark.

    Parameters
    ----------
    spark : SparkSession
        Active session
    df : Any
        Source pandas DataFrame
    view : str
        Name to register the temporary view under

    Returns
    -------
    DataFrame
        The registered Spark DataFrame
    """
    pdf = df.copy()
    pdf.insert(0, "row_id", range(len(pdf)))
    for column in pdf.columns:
        if column != "row_id":
            pdf[column] = pdf[column].astype("string")
    sdf = spark.createDataFrame(pdf)
    sdf.createOrReplaceTempView(view)
    return sdf


def _column_stats_sql(view: str, columns: list[str]) -> str:
    """Build the per-column completeness and discriminativeness query.

    Implements Magellan's attribute discriminativeness score from section 4.2 of
    the technical report: ``unique(x, A) = distinct / non-empty``,
    ``missing(x, A) = empty / rows``, ``s(x, A) = unique + 1 - missing``. A score
    near 2.0 means the attribute is present everywhere and almost a key; a score
    near 0.0 means it is mostly absent or mostly one repeated value.

    Parameters
    ----------
    view : str
        Registered view name
    columns : list[str]
        Columns to measure

    Returns
    -------
    str
        A SQL query returning one row per column
    """
    parts = []
    for column in columns:
        present = f"nullif(trim(`{column}`), '')"
        parts.append(
            f"""
            SELECT
                '{column}' AS column_name,
                count(*) AS rows,
                count({present}) AS non_empty,
                count(DISTINCT {present}) AS distinct_values,
                round(avg(length({present})), 1) AS avg_chars,
                round(1 - count({present}) / count(*), 4) AS missing,
                round(
                    count(DISTINCT {present}) / greatest(count({present}), 1)
                    + count({present}) / count(*),
                    4
                ) AS discriminativeness
            FROM {view}
            """
        )
    return " UNION ALL ".join(parts) + " ORDER BY discriminativeness DESC"


def _top_values(spark: SparkSession, view: str, column: str, limit: int) -> list[dict[str, Any]]:
    """Return the most frequent non-empty values of one column.

    Parameters
    ----------
    spark : SparkSession
        Active session
    view : str
        Registered view name
    column : str
        Column to count
    limit : int
        Number of values to return

    Returns
    -------
    list[dict[str, Any]]
        Value and count, most frequent first
    """
    rows = spark.sql(
        f"""
        SELECT trim(`{column}`) AS value, count(*) AS n
        FROM {view}
        WHERE nullif(trim(`{column}`), '') IS NOT NULL
        GROUP BY trim(`{column}`)
        ORDER BY n DESC, value
        LIMIT {limit}
        """
    ).collect()
    return [{"value": r["value"], "count": r["n"]} for r in rows]


def _agreement_sql(shared: list[str]) -> str:
    """Build the per-attribute agreement projection for a pair view.

    Run against ``gold_pairs`` it says how often a true match agrees on each
    attribute; run against ``near_miss_pairs`` it says how often a convincing
    non-match agrees. The gap between the two is the attribute's worth as
    evidence: one that agrees just as often on non-matches buys the matcher
    nothing no matter how complete it is.

    Parameters
    ----------
    shared : list[str]
        Columns present in both tables

    Returns
    -------
    str
        A SQL fragment selecting one agreement rate per shared column
    """
    jaccard = JACCARD.format(a="left_tokens", b="right_tokens")
    selects = [
        "count(*) AS pairs",
        f"round(avg(CASE WHEN {jaccard} = 1.0 THEN 1 ELSE 0 END), 4) AS exact_name",
        f"round(avg({jaccard}), 4) AS mean_jaccard",
    ]
    for column in shared:
        left = NORMALIZED.format(col=f"a_{column}")
        right = NORMALIZED.format(col=f"b_{column}")
        selects.append(
            f"round(avg(CASE WHEN nullif(trim(a_{column}), '') IS NULL"
            f" OR nullif(trim(b_{column}), '') IS NULL THEN NULL"
            f" WHEN trim({left}) = trim({right}) THEN 1 ELSE 0 END), 4) AS agree_{column}"
        )
        selects.append(
            f"round(avg(CASE WHEN nullif(trim(a_{column}), '') IS NULL"
            f" OR nullif(trim(b_{column}), '') IS NULL THEN 1 ELSE 0 END), 4)"
            f" AS unusable_{column}"
        )
    return ", ".join(selects)


def _pair_view_sql(shared: list[str], name_a: str, name_b: str, source: str) -> str:
    """Project a pair source into a flat view carrying both records.

    Parameters
    ----------
    shared : list[str]
        Columns present in both tables
    name_a : str
        Name-bearing column of the left table
    name_b : str
        Name-bearing column of the right table
    source : str
        A query or view yielding ``a_row`` and ``b_row``

    Returns
    -------
    str
        A SQL query with one row per pair and an ``a_``/``b_`` column per
        attribute. The name-bearing column is aliased ``left_key`` rather than
        ``a_name`` because several of these datasets call that column ``name``,
        which would collide with its own ``a_``-prefixed projection.
    """
    projected = [
        "p.a_row",
        "p.b_row",
        f"a.`{name_a}` AS left_key",
        f"b.`{name_b}` AS right_key",
        f"{TOKENS.format(expr=NORMALIZED.format(col=f'a.`{name_a}`'))} AS left_tokens",
        f"{TOKENS.format(expr=NORMALIZED.format(col=f'b.`{name_b}`'))} AS right_tokens",
    ]
    for column in shared:
        projected.append(f"a.`{column}` AS a_{column}")
        projected.append(f"b.`{column}` AS b_{column}")
    return f"""
        SELECT {", ".join(projected)}
        FROM ({source}) p
        JOIN a ON a.row_id = p.a_row
        JOIN b ON b.row_id = p.b_row
    """


def _labeled_pairs(name: str, root: str, table_a: Any, table_b: Any) -> set[tuple[int, int]] | None:
    """Load every pair a DeepMatcher-format dataset actually labelled.

    This matters for reading the rest of the report honestly. The Leipzig
    datasets ship a complete mapping, so any pair outside it is a true
    non-match. The DeepMatcher datasets ship only the pairs that survived
    Magellan's blocker and were then labelled, a fraction of a percent of the
    cross product, so a pair outside the gold set is usually just a pair nobody
    ever looked at.

    Parameters
    ----------
    name : str
        Dataset name
    root : str
        Directory holding the downloaded datasets
    table_a : Any
        Left table, for the source id to row index map
    table_b : Any
        Right table, for the source id to row index map

    Returns
    -------
    set[tuple[int, int]] | None
        Every labelled pair as row indices, or None for a complete mapping
    """
    archive = Path(root) / name / "data.zip"
    if "mapping_name" in DATASET_REGISTRY.get(name, {}) or not archive.exists():
        return None
    a_ids = _row_id_map(table_a)
    b_ids = _row_id_map(table_b)
    pairs: set[tuple[int, int]] = set()
    with zipfile.ZipFile(archive) as zf:
        for fname in ("train.csv", "valid.csv", "test.csv"):
            member = _find_zip_member(zf, fname)
            if member is None:
                continue
            df = _load_csv_from_zip(zf, member)
            for _, row in df.iterrows():
                left = a_ids.get(str(row["ltable_id"]))
                right = b_ids.get(str(row["rtable_id"]))
                if left is not None and right is not None:
                    pairs.add((left, right))
    return pairs


def _numeric_columns(spark: SparkSession, shared: list[str]) -> list[str]:
    """Find shared columns that hold numbers on both sides.

    Parameters
    ----------
    spark : SparkSession
        Active session
    shared : list[str]
        Candidate columns

    Returns
    -------
    list[str]
        Columns numeric enough on both sides to compare by magnitude
    """
    numeric: list[str] = []
    for column in shared:
        fractions = []
        for view in ("a", "b"):
            row = spark.sql(
                f"""
                SELECT avg(
                    CASE WHEN try_cast(regexp_replace(`{column}`, '[$, ]', '') AS double)
                        IS NOT NULL THEN 1 ELSE 0 END
                ) AS f
                FROM {view}
                WHERE nullif(trim(`{column}`), '') IS NOT NULL
                """
            ).collect()[0]["f"]
            fractions.append(row or 0.0)
        if min(fractions) >= MIN_NUMERIC_FRACTION:
            numeric.append(column)
    return numeric


def profile_benchmark(
    name: str,
    data_dir: str | None = None,
    top_values: int = 6,
    examples: int = 6,
) -> dict[str, Any]:
    """Profile one benchmark dataset end to end.

    Parameters
    ----------
    name : str
        Dataset name from the benchmark registry
    data_dir : str | None
        Directory holding the downloaded datasets. Default from config.
    top_values : int
        How many frequent values to report per column
    examples : int
        How many match and mismatch examples to report

    Returns
    -------
    dict[str, Any]
        Structured profile: shapes, per-column statistics, common values, gold
        pair cardinality, attribute agreement on matches and on near misses,
        controlled-vocabulary crosswalks, numeric agreement, and worked
        examples of both a match and a non-match that string similarity gets
        wrong.
    """
    root = data_dir or str(config.get("benchmarks.output_dir", "data/benchmarks"))
    dataset = BenchmarkDataset.download(name, root)

    spark = _spark_session()
    _register(spark, dataset.table_a, "a")
    _register(spark, dataset.table_b, "b")

    gold = [(left, right - RIGHT_ID_OFFSET) for left, right in sorted(dataset.ground_truth)]
    spark.createDataFrame(gold, "a_row int, b_row int").createOrReplaceTempView("gold")

    cols_a = [str(c) for c in dataset.table_a.columns if str(c) != "id"]
    cols_b = [str(c) for c in dataset.table_b.columns if str(c) != "id"]
    shared = [c for c in cols_a if c in cols_b]
    name_a = _detect_name_column(dataset.table_a) or cols_a[0]
    name_b = _detect_name_column(dataset.table_b) or cols_b[0]

    spark.sql(
        _pair_view_sql(shared, name_a, name_b, "SELECT a_row, b_row FROM gold")
    ).cache().createOrReplaceTempView("gold_pairs")

    # Near misses: the non-matching pairs a token-overlap blocker would rank
    # highest. Frequent tokens are dropped first, which is both what makes the
    # join tractable and what makes the survivors informative.
    rows_b = len(dataset.table_b)
    near_miss_source = f"""
        WITH a_tok AS (
            SELECT row_id, explode({TOKENS.format(expr=NORMALIZED.format(col=f"`{name_a}`"))}) AS t
            FROM a
        ),
        b_tok AS (
            SELECT row_id, explode({TOKENS.format(expr=NORMALIZED.format(col=f"`{name_b}`"))}) AS t
            FROM b
        ),
        stopwords AS (
            SELECT t FROM b_tok GROUP BY t
            HAVING count(DISTINCT row_id) > {int(rows_b * MAX_TOKEN_DOC_FRACTION) + 1}
        ),
        overlap AS (
            SELECT at.row_id AS a_row, bt.row_id AS b_row, count(*) AS shared_tokens
            FROM a_tok at
            JOIN b_tok bt ON at.t = bt.t
            WHERE at.t NOT IN (SELECT t FROM stopwords)
            GROUP BY at.row_id, bt.row_id
        )
        SELECT a_row, b_row FROM (
            SELECT a_row, b_row, row_number() OVER (
                PARTITION BY a_row ORDER BY shared_tokens DESC, b_row
            ) AS rn
            FROM overlap o
            WHERE NOT EXISTS (
                SELECT 1 FROM gold g WHERE g.a_row = o.a_row AND g.b_row = o.b_row
            )
        )
        WHERE rn <= {NEAR_MISS_CANDIDATES}
    """
    # Cached because every downstream question is asked of this view again, and
    # recomputing the token self-join each time is what exhausts the heap on
    # the largest dataset.
    spark.sql(
        _pair_view_sql(shared, name_a, name_b, near_miss_source)
    ).cache().createOrReplaceTempView("near_miss_pairs")

    labeled = _labeled_pairs(name, root, dataset.table_a, dataset.table_b)
    if labeled is None:
        spark.sql("SELECT a_row, b_row FROM gold").createOrReplaceTempView("labeled")
    else:
        spark.createDataFrame(sorted(labeled), "a_row int, b_row int").createOrReplaceTempView(
            "labeled"
        )

    jaccard = JACCARD.format(a="left_tokens", b="right_tokens")
    agreement = _agreement_sql(shared)

    profile: dict[str, Any] = {
        "name": name,
        "domain": dataset.metadata.get("domain", ""),
        "difficulty": dataset.metadata.get("difficulty", ""),
        "rows_a": len(dataset.table_a),
        "rows_b": len(dataset.table_b),
        "columns_a": cols_a,
        "columns_b": cols_b,
        "shared_columns": shared,
        "name_column_a": name_a,
        "name_column_b": name_b,
        "gold_pairs": len(gold),
        "candidate_pairs": len(dataset.table_a) * len(dataset.table_b),
        "ground_truth_kind": "complete mapping" if labeled is None else "labelled candidate set",
        "labeled_pairs": len(gold) if labeled is None else len(labeled),
    }
    profile["match_density"] = profile["gold_pairs"] / profile["candidate_pairs"]
    profile["labeled_fraction"] = profile["labeled_pairs"] / profile["candidate_pairs"]

    profile["column_stats"] = {
        "a": [r.asDict() for r in spark.sql(_column_stats_sql("a", cols_a)).collect()],
        "b": [r.asDict() for r in spark.sql(_column_stats_sql("b", cols_b)).collect()],
    }
    scores_a = {r["column_name"]: r["discriminativeness"] for r in profile["column_stats"]["a"]}
    scores_b = {r["column_name"]: r["discriminativeness"] for r in profile["column_stats"]["b"]}
    profile["discriminativeness"] = {
        column: round(scores_a[column] * scores_b[column], 4) for column in shared
    }

    profile["common_values"] = {
        "a": {c: _top_values(spark, "a", c, top_values) for c in cols_a},
        "b": {c: _top_values(spark, "b", c, top_values) for c in cols_b},
    }

    profile["cardinality"] = (
        spark.sql(
            """
            SELECT
                count(DISTINCT a_row) AS matched_left,
                count(DISTINCT b_row) AS matched_right,
                max(per_left) AS max_matches_per_left,
                max(per_right) AS max_matches_per_right,
                round(avg(CASE WHEN per_left = 1 AND per_right = 1 THEN 1 ELSE 0 END), 4)
                    AS one_to_one
            FROM (
                SELECT a_row, b_row,
                       count(*) OVER (PARTITION BY a_row) AS per_left,
                       count(*) OVER (PARTITION BY b_row) AS per_right
                FROM gold
            )
            """
        )
        .collect()[0]
        .asDict()
    )

    profile["agreement_on_matches"] = (
        spark.sql(f"SELECT {agreement} FROM gold_pairs").collect()[0].asDict()
    )
    profile["agreement_on_near_misses"] = (
        spark.sql(f"SELECT {agreement} FROM near_miss_pairs").collect()[0].asDict()
    )

    profile["jaccard_quantiles"] = {
        "matches": spark.sql(
            f"SELECT percentile_approx({jaccard}, array(0.1, 0.25, 0.5, 0.75, 0.9), 1000) AS q"
            " FROM gold_pairs"
        ).collect()[0]["q"],
        "near_misses": spark.sql(
            f"SELECT percentile_approx({jaccard}, array(0.1, 0.25, 0.5, 0.75, 0.9), 1000) AS q"
            " FROM near_miss_pairs"
        ).collect()[0]["q"],
    }
    # Under a complete mapping everything outside the gold set is a non-match
    # by construction. Under a labelled candidate set it is usually a pair
    # nobody ever looked at, and calling it a false positive is unfair to the
    # matcher.
    profile["near_misses_confirmed_negative"] = (
        1.0
        if labeled is None
        else spark.sql(
            """
            SELECT round(avg(CASE WHEN l.a_row IS NULL THEN 0 ELSE 1 END), 4) AS f
            FROM near_miss_pairs n
            LEFT JOIN labeled l ON l.a_row = n.a_row AND l.b_row = n.b_row
            """
        ).collect()[0]["f"]
    )

    # Model numbers survive across sources but their punctuation does not, so
    # the useful test is whether one side's code appears anywhere inside the
    # other side's name once every separator is stripped out.
    code_columns = [name_a] + [c for c in ("modelno", "model", "sku") if c in shared]
    a_codes = CODES.format(
        expr=NORMALIZED.format(col=" || ' ' || ".join(f"coalesce(a_{c}, '')" for c in code_columns))
        if len(code_columns) > 1
        else NORMALIZED.format(col="left_key")
    )
    b_codes = CODES.format(
        expr=NORMALIZED.format(col=" || ' ' || ".join(f"coalesce(b_{c}, '')" for c in code_columns))
        if len(code_columns) > 1
        else NORMALIZED.format(col="right_key")
    )
    code_query = f"""
        WITH coded AS (
            SELECT
                {a_codes} AS a_codes, {b_codes} AS b_codes,
                {SQUASHED.format(col="left_key")} AS a_squash,
                {SQUASHED.format(col="right_key")} AS b_squash
            FROM {{view}}
        )
        SELECT
            round(avg(CASE WHEN size(a_codes) > 0 AND size(b_codes) > 0 THEN 1 ELSE 0 END), 4)
                AS both_sides_have_a_code,
            round(avg(CASE WHEN size(array_intersect(a_codes, b_codes)) > 0 THEN 1 ELSE 0 END), 4)
                AS share_a_code_exactly,
            round(avg(CASE WHEN exists(a_codes, c -> contains(b_squash, c))
                             OR exists(b_codes, c -> contains(a_squash, c))
                        THEN 1 ELSE 0 END), 4) AS one_code_contained_in_the_other
        FROM coded
    """
    profile["code_overlap"] = {
        "columns": code_columns,
        "matches": spark.sql(code_query.format(view="gold_pairs")).collect()[0].asDict(),
        "near_misses": spark.sql(code_query.format(view="near_miss_pairs")).collect()[0].asDict(),
    }

    # Controlled vocabularies rarely agree across sources, so the crosswalk
    # between them is what a matcher has to know and cannot derive from string
    # equality.
    profile["crosswalks"] = {}
    for column in shared:
        if (
            max(
                next(
                    r["distinct_values"]
                    for r in profile["column_stats"]["a"]
                    if r["column_name"] == column
                ),
                next(
                    r["distinct_values"]
                    for r in profile["column_stats"]["b"]
                    if r["column_name"] == column
                ),
            )
            > MAX_CROSSWALK_CARDINALITY
        ):
            continue
        profile["crosswalks"][column] = [
            r.asDict()
            for r in spark.sql(
                f"""
                SELECT trim(a_{column}) AS a_value, trim(b_{column}) AS b_value, count(*) AS n
                FROM gold_pairs
                WHERE nullif(trim(a_{column}), '') IS NOT NULL
                  AND nullif(trim(b_{column}), '') IS NOT NULL
                GROUP BY trim(a_{column}), trim(b_{column})
                ORDER BY n DESC
                LIMIT {MAX_CROSSWALK_CARDINALITY}
                """
            ).collect()
        ]

    # Numeric attributes never agree exactly across two retailers, so exact
    # agreement understates them badly. Measure how close they get instead.
    close, ballpark = NUMERIC_TOLERANCES
    profile["numeric_agreement"] = {}
    for column in _numeric_columns(spark, shared):
        left = f"try_cast(regexp_replace(a_{column}, '[$, ]', '') AS double)"
        right = f"try_cast(regexp_replace(b_{column}, '[$, ]', '') AS double)"
        gap = f"abs({left} - {right}) / greatest({left}, {right})"
        query = f"""
            SELECT
                count(*) AS comparable,
                round(percentile_approx({gap}, 0.5, 1000), 4) AS median_relative_gap,
                round(avg(CASE WHEN {gap} <= {close} THEN 1 ELSE 0 END), 4) AS within_{int(close * 100)}pct,
                round(avg(CASE WHEN {gap} <= {ballpark} THEN 1 ELSE 0 END), 4)
                    AS within_{int(ballpark * 100)}pct
            FROM {{view}}
            WHERE {left} > 0 AND {right} > 0
        """
        profile["numeric_agreement"][column] = {
            "matches": spark.sql(query.format(view="gold_pairs")).collect()[0].asDict(),
            "near_misses": spark.sql(query.format(view="near_miss_pairs")).collect()[0].asDict(),
        }

    example_columns = ", ".join(
        ["a_row", "b_row", "left_key", "right_key"]
        + [f"a_{c}" for c in shared]
        + [f"b_{c}" for c in shared]
    )
    profile["hard_matches"] = [
        r.asDict()
        for r in spark.sql(
            f"""
            SELECT * FROM (
                SELECT {example_columns}, round({jaccard}, 3) AS jaccard,
                       row_number() OVER (PARTITION BY a_row ORDER BY {jaccard}) AS rn
                FROM gold_pairs
                WHERE size(left_tokens) > 2 AND size(right_tokens) > 2
            )
            WHERE rn = 1
            ORDER BY jaccard, a_row
            LIMIT {examples}
            """
        ).collect()
    ]
    profile["hard_non_matches"] = [
        r.asDict()
        for r in spark.sql(
            f"""
            SELECT * FROM (
                SELECT {example_columns}, round({jaccard}, 3) AS jaccard,
                       row_number() OVER (PARTITION BY a_row ORDER BY {jaccard} DESC) AS rn
                FROM near_miss_pairs
                WHERE size(left_tokens) > 2 AND size(right_tokens) > 2
            )
            WHERE rn = 1
            ORDER BY jaccard DESC, a_row
            LIMIT {examples}
            """
        ).collect()
    ]

    logger.info(
        "Profiled %s: %d x %d rows, %d gold pairs, %d shared columns",
        name,
        profile["rows_a"],
        profile["rows_b"],
        profile["gold_pairs"],
        len(shared),
    )
    return profile


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    """Render a list of row dicts as a Markdown table.

    Parameters
    ----------
    rows : list[dict[str, Any]]
        Rows to render
    columns : list[str]
        Column order

    Returns
    -------
    list[str]
        Markdown lines
    """
    out = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        cells = []
        for column in columns:
            text = str(row.get(column, "")).replace("|", "\\|").replace("\n", " ")
            cells.append(text[:110])
        out.append("| " + " | ".join(cells) + " |")
    return out


def _render_example(example: dict[str, Any], shared: list[str]) -> list[str]:
    """Render one worked pair as a pair of attribute lines.

    Parameters
    ----------
    example : dict[str, Any]
        Row from ``hard_matches`` or ``hard_non_matches``
    shared : list[str]
        Columns present in both tables

    Returns
    -------
    list[str]
        Markdown lines
    """
    lines = [f"- Jaccard {example['jaccard']:.3f}"]
    for side, key in (("a", "left_key"), ("b", "right_key")):
        parts = []
        for column in shared:
            value = example.get(f"{side}_{column}")
            if value is None or str(value).strip() == "":
                continue
            parts.append(f"{column}=`{str(value).strip()[:120]}`")
        if not parts:
            parts.append(f"name=`{str(example[key]).strip()[:120]}`")
        lines.append(f"  - {side.upper()}: " + ", ".join(parts))
    return lines


def format_profile(profile: dict[str, Any]) -> str:
    """Render a profile as Markdown.

    Parameters
    ----------
    profile : dict[str, Any]
        Output of :func:`profile_benchmark`

    Returns
    -------
    str
        Markdown report
    """
    shared = profile["shared_columns"]
    stat_columns = [
        "column_name",
        "non_empty",
        "distinct_values",
        "avg_chars",
        "missing",
        "discriminativeness",
    ]
    lines: list[str] = [
        f"## {profile['name']}",
        "",
        f"Domain {profile['domain']}, difficulty {profile['difficulty']}. "
        f"{profile['rows_a']:,} x {profile['rows_b']:,} rows, "
        f"{profile['gold_pairs']:,} gold pairs out of "
        f"{profile['candidate_pairs']:,} candidate pairs "
        f"({profile['match_density']:.6%} of the cross product). "
        f"Ground truth is a {profile['ground_truth_kind']} covering "
        f"{profile['labeled_pairs']:,} pairs, {profile['labeled_fraction']:.6%} "
        "of the cross product.",
        "",
        "### Columns",
        "",
    ]
    for side in ("a", "b"):
        lines += [f"Table {side.upper()}:", ""]
        lines += _markdown_table(profile["column_stats"][side], stat_columns)
        lines.append("")

    lines += ["Joint discriminativeness `s(x) = s(x,A) * s(x,B)`:", ""]
    lines += _markdown_table(
        [
            {"column": k, "score": v}
            for k, v in sorted(profile["discriminativeness"].items(), key=lambda kv: -kv[1])
        ],
        ["column", "score"],
    )

    lines += ["", "### Common values", ""]
    for side in ("a", "b"):
        for column, values in profile["common_values"][side].items():
            if not values or values[0]["count"] < 2:
                continue
            rendered = ", ".join(f"`{v['value'][:40]}` ({v['count']})" for v in values)
            lines.append(f"- {side.upper()}.{column}: {rendered}")
    lines.append("")

    card = profile["cardinality"]
    lines += [
        "### Gold pairs",
        "",
        f"{card['matched_left']:,} left and {card['matched_right']:,} right records participate. "
        f"Up to {card['max_matches_per_left']} matches per left record and "
        f"{card['max_matches_per_right']} per right record; "
        f"{card['one_to_one']:.1%} of pairs are strictly one to one.",
        "",
        "### Attribute agreement",
        "",
        "Fraction of pairs whose normalised values are equal, on true matches "
        "against the hardest non-matches a token blocker produces. `unusable_` "
        "is the fraction where at least one side is empty.",
        "",
    ]
    lines += _markdown_table(
        [
            {
                "measure": key,
                "matches": profile["agreement_on_matches"][key],
                "near_misses": profile["agreement_on_near_misses"][key],
            }
            for key in profile["agreement_on_matches"]
        ],
        ["measure", "matches", "near_misses"],
    )

    quantiles = profile["jaccard_quantiles"]
    lines += [
        "",
        "Name-token Jaccard at p10/p25/p50/p75/p90: "
        f"matches {[round(q, 3) for q in quantiles['matches']]}, "
        f"near misses {[round(q, 3) for q in quantiles['near_misses']]}. "
        f"{profile['near_misses_confirmed_negative']:.1%} of the near misses are "
        "confirmed non-matches rather than pairs nobody labelled.",
        "",
        "### Model codes",
        "",
        f"Digit-bearing tokens of four characters or more, taken from "
        f"{', '.join(f'`{c}`' for c in profile['code_overlap']['columns'])}. "
        "`contained` strips every separator from both names first, so "
        "`MDREX55WH` still finds `MDR EX55/WHI`.",
        "",
    ]
    lines += _markdown_table(
        [
            {
                "measure": key,
                "matches": profile["code_overlap"]["matches"][key],
                "near_misses": profile["code_overlap"]["near_misses"][key],
            }
            for key in profile["code_overlap"]["matches"]
        ],
        ["measure", "matches", "near_misses"],
    )
    lines.append("")

    if profile["crosswalks"]:
        lines += ["### Controlled vocabulary crosswalk", ""]
        for column, rows in profile["crosswalks"].items():
            lines.append(f"`{column}`, as paired by the gold mapping:")
            lines.append("")
            lines += _markdown_table(rows, ["a_value", "b_value", "n"])
            lines.append("")

    if profile["numeric_agreement"]:
        lines += ["### Numeric agreement", ""]
        for column, sides in profile["numeric_agreement"].items():
            keys = list(sides["matches"].keys())
            lines.append(f"`{column}`:")
            lines.append("")
            lines += _markdown_table(
                [
                    {
                        "measure": key,
                        "matches": sides["matches"][key],
                        "near_misses": sides["near_misses"][key],
                    }
                    for key in keys
                ],
                ["measure", "matches", "near_misses"],
            )
            lines.append("")

    lines += ["### Matches string similarity misses", ""]
    for example in profile["hard_matches"]:
        lines += _render_example(example, shared)
    lines += ["", "### Non-matches string similarity accepts", ""]
    for example in profile["hard_non_matches"]:
        lines += _render_example(example, shared)
    lines.append("")
    return "\n".join(lines)
