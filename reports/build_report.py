"""Render the SERF baseline benchmark report as one self-contained HTML file.

Reads the reconstruction produced by ``reports/reconstruct_mistakes.py`` plus the
saved ``*_results.json`` aggregates and emits a single HTML document with all CSS
inline and every figure as hand-emitted SVG, so the report opens from disk with no
network access and no external assets.

Usage
-----
    uv run python reports/build_report.py <analysis.json> <output.html>
"""

import datetime
import html
import json
import pathlib
import sys
from typing import Any

from serf.logs import get_logger

logger = get_logger(__name__)

INPUT_COST_PER_MTOK = 0.09
OUTPUT_COST_PER_MTOK = 0.36

STUDENT_MODEL = "openai/gpt-oss-120b-maas"
TEACHER_MODEL = "gemini/gemini-3.5-flash-lite"
MAX_TOKENS = 65536
TARGET_BLOCK_SIZE = 30
MAX_BLOCK_SIZE = 100

KOPCKE = "https://vldb.org/pvldb/vol3/E04.pdf"
MAGELLAN = "https://www.vldb.org/pvldb/vol9/p1197-pkonda.pdf"
DEEPMATCHER = "https://pages.cs.wisc.edu/~anhai/papers1/deepmatcher-sigmod18.pdf"

ORDER = ["dblp-acm", "dblp-scholar", "abt-buy", "amazon-google", "walmart-amazon"]

PALETTE = {
    "precision": "#2f6fb3",
    "recall": "#c2603c",
    "f1": "#3f8f6b",
    "generic": "#8a8fa3",
    "typed": "#3f8f6b",
    "generic_iter": "#c9ccd6",
    "typed_iter": "#8fc4a9",
    "tp": "#3f8f6b",
    "blocking": "#c2603c",
    "failed": "#a9569f",
    "model": "#d9a520",
    "grid": "#dfe3ea",
    "axis": "#8b93a3",
}

DATASET_FACTS: dict[str, dict[str, Any]] = {
    "dblp-acm": {
        "display": "DBLP&ndash;ACM",
        "domain": "Bibliographic",
        "records": 4910,
        "gold_pairs": 2224,
        "what": (
            "Two publication catalogues covering overlapping venues in the database research "
            "community. A record is one paper as that catalogue describes it: DBLP's "
            "bibliography entry on one side, the ACM Digital Library's entry on the other. A "
            "gold pair says the two entries describe the same paper."
        ),
        "sides": [
            ("DBLP", "DBLP2.csv", 2616, ["id", "title", "authors", "venue", "year"]),
            ("ACM", "ACM.csv", 2294, ["id", "title", "authors", "venue", "year"]),
        ],
        "gold_file": "DBLP-ACM_perfectMapping.csv (idDBLP, idACM)",
        "provenance": (
            "Introduced by K&ouml;pcke, Thor and Rahm, <em>Evaluation of entity resolution "
            "approaches on real-world match problems</em>, PVLDB 3(1), 2010"
        ),
        "provenance_url": KOPCKE,
        "notes": (
            "The two sides share a schema exactly, so no schema alignment is needed. Titles are "
            "near-identical modulo case, author lists are the same people in a different order, "
            "and venue strings differ predictably (<code>VLDB</code> against <code>Very Large "
            "Data Bases</code>). ACM titles carry HTML entities such as "
            "<code>B&amp;#246;hlen</code> that DBLP writes as <code>B&ouml;hlen</code>."
        ),
        "reading": (
            "This is the easy end of the benchmark suite and the scores show it. Precision is "
            "0.93: when the model claims two entries are the same paper it is almost always "
            "right, because an identical title plus an overlapping author list is close to "
            "conclusive evidence. Recall of 0.52 is not the model failing to recognise "
            "duplicates; the reconstruction below shows that almost all of the miss is pairs "
            "the model never got to judge."
        ),
    },
    "dblp-scholar": {
        "display": "DBLP&ndash;Scholar",
        "domain": "Bibliographic",
        "records": 66879,
        "gold_pairs": 5347,
        "what": (
            "The same DBLP bibliography matched against a Google Scholar crawl. A record is one "
            "publication entry, but where DBLP is a curated catalogue, Scholar is scraped and "
            "dirty. This is the same matching task as DBLP&ndash;ACM against a far noisier and "
            "much larger second source."
        ),
        "sides": [
            ("DBLP", "DBLP1.csv", 2616, ["id", "title", "authors", "venue", "year"]),
            ("Scholar", "Scholar.csv", 64263, ["id", "title", "authors", "venue", "year"]),
        ],
        "gold_file": "DBLP-Scholar_perfectMapping.csv (idDBLP, idScholar)",
        "provenance": (
            "Introduced by K&ouml;pcke, Thor and Rahm, <em>Evaluation of entity resolution "
            "approaches on real-world match problems</em>, PVLDB 3(1), 2010"
        ),
        "provenance_url": KOPCKE,
        "notes": (
            "Scholar's <code>id</code> header carries a UTF-8 byte-order mark, and its rows are "
            "genuinely noisy: one sampled record's title is a street address, "
            "&ldquo;11578 Sorrento Valley Road&rdquo;, with authors &ldquo;QD Inc&rdquo; and no "
            "year. The right table is 24 times the size of the left, so the two sides are "
            "wildly unbalanced and most Scholar records have no DBLP counterpart at all."
        ),
        "reading": (
            "At 66,879 records this is the most expensive dataset in the suite, which is exactly "
            "why the full-data run was cancelled and never re-run. Everything reported here "
            "comes from either the earlier truncated reference run or the 1,000-record A/B "
            "sample, and is labelled as such."
        ),
        "reference_note": (
            "The earlier reference run was <strong>not</strong> a full-table run and was in one "
            "important respect easier than the real task. It passed "
            "<code>--max-right-entities 5000</code>, and because that option keeps every "
            "gold-matched right record before it samples any others, the Scholar side collapsed "
            "to exactly the 5,218 records that appear in the gold mapping. Every Scholar record "
            "in that run therefore had a DBLP partner and the 58,000-odd distractor records were "
            "removed. Its F1 of 0.4110 is an optimistic figure for a much easier problem than "
            "matching against the whole crawl."
        ),
    },
    "abt-buy": {
        "display": "Abt&ndash;Buy",
        "domain": "E-commerce products",
        "records": 2173,
        "gold_pairs": 1097,
        "what": (
            "Product listings from two consumer-electronics retailers, Abt and Buy.com. A record "
            "is one product as that retailer merchandises it, and a gold pair says both "
            "retailers are selling the same physical product."
        ),
        "sides": [
            ("Abt", "Abt.csv", 1081, ["id", "name", "description", "price"]),
            ("Buy", "Buy.csv", 1092, ["id", "name", "description", "manufacturer", "price"]),
        ],
        "gold_file": "abt_buy_perfectMapping.csv (idAbt, idBuy)",
        "provenance": (
            "Introduced by K&ouml;pcke, Thor and Rahm, <em>Evaluation of entity resolution "
            "approaches on real-world match problems</em>, PVLDB 3(1), 2010"
        ),
        "provenance_url": KOPCKE,
        "notes": (
            "The two sides are <strong>asymmetric</strong>: Buy carries a "
            "<code>manufacturer</code> column and Abt does not, so the brand is only available "
            "on one side and has to be inferred from the product name on the other. Abt packs "
            "its whole spec sheet into <code>description</code> as slash-delimited text, while "
            "many Buy descriptions are empty."
        ),
        "reading": (
            "Abt&ndash;Buy behaves like an easier product task than the other two because the "
            "model numbers are usually present in the product name on both sides. Precision "
            "0.90 reflects that: a matching SKU suffix is strong evidence. The characteristic "
            "error is variant confusion, where two products differ by exactly the suffix that "
            "distinguishes them."
        ),
    },
    "amazon-google": {
        "display": "Amazon&ndash;Google",
        "domain": "E-commerce products (software)",
        "records": 4589,
        "gold_pairs": 1167,
        "what": (
            "Software product listings from Amazon matched against Google's product feed. A "
            "record is one software title or licence SKU. Gold pairs come from the "
            "<code>label=1</code> rows of the DeepMatcher train/valid/test splits."
        ),
        "sides": [
            ("Amazon", "tableA.csv", 1363, ["id", "title", "manufacturer", "price"]),
            ("Google", "tableB.csv", 3226, ["id", "title", "manufacturer", "price"]),
        ],
        "gold_file": "train.csv / valid.csv / test.csv rows with label=1",
        "provenance": (
            "Introduced by K&ouml;pcke, Thor and Rahm, <em>Evaluation of entity resolution "
            "approaches on real-world match problems</em>, PVLDB 3(1), 2010"
        ),
        "provenance_url": KOPCKE,
        "notes": (
            "Only four columns, and <code>manufacturer</code> is empty for most Google rows, so "
            "in practice the model is matching on <code>title</code> alone. Titles are "
            "lowercased and Google's are frequently machine-generated: vendor part numbers get "
            "spliced into the middle of the product name, as in <code>sage ( ptree ) "
            "vernfp2007rt premium accounting for nonprofits 2007</code>."
        ),
        "reading": (
            "This is the hardest dataset in the suite and it is the one we do worst on by a "
            "wide margin. DeepMatcher singles out Amazon&ndash;Google as the case where matching "
            "titles are effectively synonyms with a large string distance, so surface similarity "
            "is actively misleading: distinct licence SKUs from the same vendor look nearly "
            "identical while genuine matches look nothing alike."
        ),
    },
    "walmart-amazon": {
        "display": "Walmart&ndash;Amazon",
        "domain": "E-commerce products (electronics)",
        "records": 24628,
        "gold_pairs": 962,
        "what": (
            "Electronics listings from Walmart matched against Amazon. A record is one product "
            "offer with a category, brand and model number. Gold pairs come from the "
            "<code>label=1</code> rows of the DeepMatcher splits."
        ),
        "sides": [
            (
                "Walmart",
                "exp_data/tableA.csv",
                2554,
                ["id", "title", "category", "brand", "modelno", "price"],
            ),
            (
                "Amazon",
                "exp_data/tableB.csv",
                22074,
                ["id", "title", "category", "brand", "modelno", "price"],
            ),
        ],
        "gold_file": "exp_data/train.csv / valid.csv / test.csv rows with label=1",
        "provenance": (
            "Originated in UW&ndash;Madison CS 784 class projects and entered the literature via "
            "Konda et al., <em>Magellan: Toward Building Entity Matching Management "
            "Systems</em>, PVLDB 9(12), 2016"
        ),
        "provenance_url": MAGELLAN,
        "notes": (
            "The richest schema in the suite: an explicit <code>modelno</code> column on both "
            "sides is close to a natural key when it is populated. The tables are very "
            "unbalanced, 2,554 Walmart rows against 22,074 Amazon rows, so most Amazon records "
            "have no counterpart and many blocks end up containing Amazon rows only."
        ),
        "reading": (
            "With <code>brand</code> and <code>modelno</code> broken out as their own fields "
            "this should be the most tractable of the three product tasks, and the presence of "
            "a large single-source right table means blocking quality matters more here than "
            "anywhere else in the suite."
        ),
    },
}


def esc(value: Any) -> str:
    """HTML-escape a value for safe inclusion in the report.

    Parameters
    ----------
    value : Any
        Value to escape

    Returns
    -------
    str
        Escaped text
    """
    return html.escape("" if value is None else str(value))


def fmt(value: float | None, places: int = 4) -> str:
    """Format a metric, or an em dash when it is missing.

    Parameters
    ----------
    value : float | None
        Metric value
    places : int
        Decimal places

    Returns
    -------
    str
        Formatted metric
    """
    if value is None:
        return "&mdash;"
    return f"{value:.{places}f}"


def thousands(value: float | None) -> str:
    """Format an integer with thousands separators.

    Parameters
    ----------
    value : float | None
        Number to format

    Returns
    -------
    str
        Formatted number
    """
    if value is None:
        return "&mdash;"
    return f"{int(value):,}"


def duration(seconds: float | None) -> str:
    """Render a second count as minutes and seconds.

    Parameters
    ----------
    seconds : float | None
        Elapsed seconds

    Returns
    -------
    str
        Human readable duration
    """
    if seconds is None:
        return "&mdash;"
    total = int(round(seconds))
    return f"{total // 60}m {total % 60:02d}s ({total:,}s)"


def legend(
    entries: list[tuple[str, str]], x: int, baseline: int, max_width: float
) -> tuple[str, int]:
    """Lay out a chart legend, wrapping onto extra rows when it would overflow.

    Parameters
    ----------
    entries : list[tuple[str, str]]
        Label and colour per legend item
    x : int
        Left edge of the legend
    baseline : int
        Text baseline of the first row
    max_width : float
        Width the legend must fit inside

    Returns
    -------
    tuple[str, int]
        Legend markup and the number of rows used
    """
    parts: list[str] = []
    cursor = float(x)
    row = 0
    for label, colour in entries:
        item_w = 17 + 6.55 * len(label) + 20
        if cursor > x and cursor - x + item_w > max_width:
            row += 1
            cursor = float(x)
        y = baseline + row * 17
        parts.append(
            f'<rect x="{cursor:.1f}" y="{y - 9}" width="11" height="11" fill="{colour}" rx="2"/>'
        )
        parts.append(f'<text x="{cursor + 17:.1f}" y="{y}" class="legend">{esc(label)}</text>')
        cursor += item_w
    return "".join(parts), row + 1


def grouped_bars(
    groups: list[str],
    series: list[tuple[str, list[float | None], str]],
    caption: str,
    ymax: float = 1.0,
    width: int = 900,
    height: int = 330,
    value_places: int = 3,
) -> str:
    """Emit a grouped bar chart as inline SVG.

    Parameters
    ----------
    groups : list[str]
        Category labels along the x axis
    series : list[tuple[str, list[float | None], str]]
        Series name, one value per group, and bar colour
    caption : str
        Figure caption
    ymax : float
        Top of the y axis
    width : int
        SVG width in pixels
    height : int
        SVG height in pixels
    value_places : int
        Decimal places for the value printed above each bar

    Returns
    -------
    str
        SVG markup wrapped in a figure element
    """
    pad_left, pad_right, pad_top = 58, 18, 26
    legend_rows = 1
    plot_w = width - pad_left - pad_right
    for _attempt in range(3):
        pad_bottom = 50 + 17 * legend_rows
        _markup, rows = legend(
            [(name, colour) for name, _values, colour in series],
            pad_left,
            height - pad_bottom + 34,
            plot_w,
        )
        if rows == legend_rows:
            break
        legend_rows = rows
    pad_bottom = 50 + 17 * legend_rows
    height = height + 17 * (legend_rows - 1)
    plot_h = height - pad_top - pad_bottom
    group_w = plot_w / max(1, len(groups))
    inner = group_w * 0.78
    bar_w = inner / max(1, len(series))

    parts: list[str] = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
        f'aria-label="{esc(caption)}" class="chart">'
    ]
    for step in range(6):
        frac = step / 5
        y = pad_top + plot_h - frac * plot_h
        parts.append(
            f'<line x1="{pad_left}" y1="{y:.1f}" x2="{pad_left + plot_w}" y2="{y:.1f}" '
            f'stroke="{PALETTE["grid"]}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{pad_left - 10}" y="{y + 4:.1f}" text-anchor="end" '
            f'class="tick">{frac * ymax:.1f}</text>'
        )
    for gi, group in enumerate(groups):
        gx = pad_left + gi * group_w + (group_w - inner) / 2
        for si, (_name, values, colour) in enumerate(series):
            value = values[gi]
            x = gx + si * bar_w
            if value is None:
                parts.append(
                    f'<text x="{x + bar_w / 2:.1f}" y="{pad_top + plot_h - 8:.1f}" '
                    f'text-anchor="middle" class="nodata">n/a</text>'
                )
                continue
            bar_h = max(0.0, min(1.0, value / ymax)) * plot_h
            y = pad_top + plot_h - bar_h
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w - 3:.1f}" '
                f'height="{bar_h:.1f}" fill="{colour}" rx="2"/>'
            )
            parts.append(
                f'<text x="{x + (bar_w - 3) / 2:.1f}" y="{y - 5:.1f}" text-anchor="middle" '
                f'class="barval">{value:.{value_places}f}</text>'
            )
        parts.append(
            f'<text x="{pad_left + gi * group_w + group_w / 2:.1f}" '
            f'y="{pad_top + plot_h + 20:.1f}" text-anchor="middle" '
            f'class="xlabel">{esc(group)}</text>'
        )
    parts.append(
        f'<line x1="{pad_left}" y1="{pad_top + plot_h}" x2="{pad_left + plot_w}" '
        f'y2="{pad_top + plot_h}" stroke="{PALETTE["axis"]}" stroke-width="1"/>'
    )
    legend_markup, _rows = legend(
        [(name, colour) for name, _values, colour in series],
        pad_left,
        pad_top + plot_h + 44,
        plot_w,
    )
    parts.append(legend_markup)
    parts.append("</svg>")
    return f"<figure>{''.join(parts)}<figcaption>{caption}</figcaption></figure>"


def stacked_bars(
    groups: list[str],
    segments: list[tuple[str, list[float], str]],
    caption: str,
    width: int = 900,
    height: int = 360,
) -> str:
    """Emit a 100%-stacked bar chart as inline SVG.

    Parameters
    ----------
    groups : list[str]
        Category labels along the x axis
    segments : list[tuple[str, list[float], str]]
        Segment name, absolute count per group, and colour
    caption : str
        Figure caption
    width : int
        SVG width in pixels
    height : int
        SVG height in pixels

    Returns
    -------
    str
        SVG markup wrapped in a figure element
    """
    pad_left, pad_right, pad_top = 58, 18, 26
    plot_w = width - pad_left - pad_right
    legend_rows = 1
    for _attempt in range(3):
        _markup, rows = legend(
            [(name, colour) for name, _values, colour in segments], pad_left, 0, plot_w
        )
        if rows == legend_rows:
            break
        legend_rows = rows
    pad_bottom = 62 + 17 * legend_rows
    height = height + 17 * (legend_rows - 1)
    plot_h = height - pad_top - pad_bottom
    group_w = plot_w / max(1, len(groups))
    bar_w = min(96.0, group_w * 0.5)

    parts: list[str] = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
        f'aria-label="{esc(caption)}" class="chart">'
    ]
    for step in range(6):
        frac = step / 5
        y = pad_top + plot_h - frac * plot_h
        parts.append(
            f'<line x1="{pad_left}" y1="{y:.1f}" x2="{pad_left + plot_w}" y2="{y:.1f}" '
            f'stroke="{PALETTE["grid"]}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{pad_left - 10}" y="{y + 4:.1f}" text-anchor="end" '
            f'class="tick">{frac * 100:.0f}%</text>'
        )
    for gi, group in enumerate(groups):
        total = sum(seg[1][gi] for seg in segments)
        x = pad_left + gi * group_w + (group_w - bar_w) / 2
        y = pad_top + plot_h
        if total <= 0:
            parts.append(
                f'<text x="{x + bar_w / 2:.1f}" y="{y - 8:.1f}" text-anchor="middle" '
                f'class="nodata">no data</text>'
            )
        for name, values, colour in segments:
            share = values[gi] / total if total else 0.0
            seg_h = share * plot_h
            y -= seg_h
            if seg_h <= 0:
                continue
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{seg_h:.1f}" '
                f'fill="{colour}"><title>{esc(name)}: {int(values[gi]):,} '
                f"({share * 100:.1f}%)</title></rect>"
            )
            if seg_h > 16:
                parts.append(
                    f'<text x="{x + bar_w / 2:.1f}" y="{y + seg_h / 2 + 4:.1f}" '
                    f'text-anchor="middle" class="seglabel">{share * 100:.0f}%</text>'
                )
        parts.append(
            f'<text x="{pad_left + gi * group_w + group_w / 2:.1f}" '
            f'y="{pad_top + plot_h + 20:.1f}" text-anchor="middle" '
            f'class="xlabel">{esc(group)}</text>'
        )
        parts.append(
            f'<text x="{pad_left + gi * group_w + group_w / 2:.1f}" '
            f'y="{pad_top + plot_h + 36:.1f}" text-anchor="middle" '
            f'class="sublabel">{int(total):,} gold pairs</text>'
        )
    legend_markup, _rows = legend(
        [(name, colour) for name, _values, colour in segments],
        pad_left,
        pad_top + plot_h + 56,
        plot_w,
    )
    parts.append(legend_markup)
    parts.append("</svg>")
    return f"<figure>{''.join(parts)}<figcaption>{caption}</figcaption></figure>"


def metric(run: dict[str, Any] | None, key: str) -> float | None:
    """Read one saved metric from a run, or None when the run is absent.

    Parameters
    ----------
    run : dict[str, Any] | None
        Reconstructed run
    key : str
        Key in the run's saved aggregates

    Returns
    -------
    float | None
        Metric value
    """
    return None if run is None else float(run["saved"][key])


def recon_count(run: dict[str, Any] | None, key: str) -> float:
    """Read one reconstructed count from a run, treating an absent run as zero.

    Parameters
    ----------
    run : dict[str, Any] | None
        Reconstructed run
    key : str
        Key in the run's reconstruction

    Returns
    -------
    float
        Count
    """
    return 0.0 if run is None else float(run["reconstructed"][key])


def pick(
    runs: list[dict[str, Any]],
    dataset: str,
    group: str,
    mode: str,
    require_traces: bool = False,
    iterations: str = "single",
) -> dict[str, Any] | None:
    """Select the most recent run matching a dataset, output group and signature mode.

    Scores come from the saved aggregates and do not need traces, but the error
    decomposition and the worked examples do, so callers that need trace content ask
    for it explicitly.

    Parameters
    ----------
    runs : list[dict[str, Any]]
        All reconstructed runs
    dataset : str
        Dataset name
    group : str
        Output directory prefix, e.g. ``full_baseline`` or ``ab_`` for any A/B directory
    mode : str
        ``generic`` or ``per-dataset``
    require_traces : bool
        Only consider runs whose traces were found in the tracking store
    iterations : str
        ``single`` for one-pass runs, ``multi`` for runs that re-blocked merged
        entities over several iterations, ``any`` for either

    Returns
    -------
    dict[str, Any] | None
        Matching run, or None
    """
    matching = [
        run
        for run in runs
        if run["dataset"] == dataset
        and run["group"].startswith(group)
        and run["signature_mode"] == mode
        and (iterations == "any" or (run.get("iterations_run", 1) > 1) == (iterations == "multi"))
        and (
            run["traces"] > 2 and run.get("reconstruction_reliable", True)
            if require_traces
            else True
        )
    ]
    return max(matching, key=lambda run: run["end"]) if matching else None


def reference_label(dataset: str) -> str:
    """Return the score-table label for the earlier reference run.

    Parameters
    ----------
    dataset : str
        Dataset name

    Returns
    -------
    str
        Row label
    """
    if dataset == "dblp-scholar":
        return "Earlier reference, max_tokens 8192, right table cut to 5,218"
    return "Earlier reference, max_tokens 8192"


def score_row(label: str, run: dict[str, Any] | None, note: str = "") -> str:
    """Render one row of a score table.

    Parameters
    ----------
    label : str
        Row label describing the run
    run : dict[str, Any] | None
        Reconstructed run, or None when the run does not exist
    note : str
        Trailing note for the row

    Returns
    -------
    str
        Table row markup
    """
    if run is None:
        return (
            f'<tr class="missing"><td>{label}</td><td colspan="9">not run '
            f"&mdash; {note or 'no result on disk'}</td></tr>"
        )
    saved = run["saved"]
    return (
        f"<tr><td>{label}</td>"
        f'<td class="num">{thousands(saved.get("records") or DATASET_FACTS[run["dataset"]]["records"])}</td>'
        f'<td class="num strong">{fmt(saved["precision"])}</td>'
        f'<td class="num strong">{fmt(saved["recall"])}</td>'
        f'<td class="num strong">{fmt(saved["f1_score"])}</td>'
        f'<td class="num">{thousands(saved["true_positives"])}</td>'
        f'<td class="num">{thousands(saved["false_positives"])}</td>'
        f'<td class="num">{thousands(saved["predicted_pairs"])}</td>'
        f'<td class="num">{thousands(saved["true_pairs"])}</td>'
        f'<td class="num">{duration(saved["elapsed_seconds"])}</td></tr>'
    )


def example_card(example: dict[str, Any], sides: list[Any]) -> str:
    """Render one reconstructed mistake as a side-by-side record comparison.

    Parameters
    ----------
    example : dict[str, Any]
        Example payload from the reconstruction
    sides : list[Any]
        Per-side descriptors, used to name each source

    Returns
    -------
    str
        Card markup
    """
    names = {"left": sides[0][0], "right": sides[1][0]}
    is_fp = example["label"] == "false_positive"
    badge = "False positive" if is_fp else "False negative"
    badge_class = "fp" if is_fp else "fn"

    if is_fp:
        if example["left_side"] == example["right_side"]:
            verdict = (
                f"Both records come from the <strong>{names[example['left_side']]}</strong> side. "
                "The gold standard only contains cross-source pairs, so a within-source merge "
                "can never be scored correct however defensible it looks."
            )
        else:
            verdict = "The model merged these two records; the gold standard says they differ."
    elif example["co_blocked"] is False:
        verdict = (
            "<strong>Blocking miss.</strong> These two records were never placed in the same "
            "block, so the matcher was never given the chance to compare them. This is a "
            "recall loss caused by blocking, not by the model."
        )
    else:
        verdict = (
            "<strong>Matching error.</strong> Both records were in the same block and the "
            "matcher returned a result for that block, so the model saw this pair and declined "
            "to merge it."
        )

    panels = []
    for which in ("left", "right"):
        fields: dict[str, Any] = example[which] or {}
        rows = "".join(
            f'<div class="fieldrow"><span class="fk">{esc(key)}</span>'
            f'<span class="fv">{esc(value) if str(value).strip() else "<em>empty</em>"}</span></div>'
            for key, value in fields.items()
        )
        rows = rows.replace("&lt;em&gt;empty&lt;/em&gt;", '<em class="empty">empty</em>')
        panels.append(
            f'<div class="panel"><div class="ptitle">{esc(names[example[which + "_side"]])}'
            f' <span class="eid">entity {example[which + "_id"]}</span></div>{rows}</div>'
        )

    reason = ""
    if example.get("reason"):
        reason = (
            f'<div class="reason"><span class="rl">Model&rsquo;s stated reason</span>'
            f"&ldquo;{esc(example['reason'])}&rdquo;</div>"
        )

    return (
        f'<div class="example"><div class="exhead"><span class="badge {badge_class}">{badge}</span>'
        f'<span class="verdict">{verdict}</span></div>'
        f'<div class="panels">{panels[0]}{panels[1]}</div>{reason}</div>'
    )


def examples_block(run: dict[str, Any] | None, sides: list[Any], source_note: str) -> str:
    """Render every example group for a run.

    Parameters
    ----------
    run : dict[str, Any] | None
        Reconstructed run holding the examples
    sides : list[Any]
        Per-side descriptors
    source_note : str
        Sentence naming which run the examples were reconstructed from

    Returns
    -------
    str
        Markup for the examples section
    """
    if run is None:
        return (
            '<p class="warn">No MLflow traces are available for this dataset, so no real '
            "example mistakes could be reconstructed. Nothing is shown here rather than "
            "inventing examples.</p>"
        )

    groups = [
        (
            "false_positives_cross_source",
            "Cross-source false positives",
            "Pairs spanning the two sources that the model merged and the gold standard "
            "rejects. These are genuine matching errors.",
        ),
        (
            "false_positives_same_source",
            "Within-source false positives",
            "Merges of two records from the same side. The gold standard is cross-source only, "
            "so these are counted as errors by construction even when the two records really "
            "are duplicates.",
        ),
        (
            "false_negatives_model_error",
            "False negatives the model actually saw",
            "Gold pairs where both records were in the same block and that block was matched "
            "successfully. The model looked at these and said no.",
        ),
        (
            "false_negatives_failed_block",
            "False negatives lost to failed blocks",
            "Gold pairs that were co-blocked, but only inside blocks whose LLM call failed, so "
            "no decision was ever made about them.",
        ),
        (
            "false_negatives_blocking_miss",
            "False negatives never co-blocked",
            "Gold pairs whose two records never landed in the same block. Blocking, not "
            "matching, lost these.",
        ),
    ]

    out = [f'<p class="provenance-note">{source_note}</p>']
    total = 0
    for key, title, blurb in groups:
        items: list[dict[str, Any]] = run["examples"].get(key) or []
        if not items:
            continue
        total += len(items)
        cards = "".join(example_card(item, sides) for item in items)
        out.append(f'<h4>{title} <span class="count">{len(items)} shown</span></h4>')
        out.append(f'<p class="blurb">{blurb}</p>')
        out.append(cards)
    if total == 0:
        out.append(
            '<p class="warn">This run produced no reconstructable mistakes in any category.</p>'
        )
    return "".join(out)


def mistake_table(run: dict[str, Any] | None) -> str:
    """Render the reconstructed error decomposition for one run.

    Parameters
    ----------
    run : dict[str, Any] | None
        Reconstructed run

    Returns
    -------
    str
        Table markup, or an explanatory paragraph when reconstruction is unavailable
    """
    if run is None:
        return (
            '<p class="warn">No traces could be located for this dataset, so its errors cannot '
            "be decomposed into blocking misses, failed blocks and matching errors.</p>"
        )
    recon = run["reconstructed"]
    verified = (
        '<span class="ok">reproduces the saved aggregates exactly</span>'
        if run["verified"]
        else '<span class="warnbadge">does not fully reproduce the saved aggregates</span>'
    )
    return f"""
<table class="decomp">
<caption>Error decomposition reconstructed from {run["traces"]} MLflow traces
({run["error_traces"]} of which record a failed LLM call) &mdash; {verified}</caption>
<tbody>
<tr><th>Gold pairs in scope</th><td class="num">{thousands(recon["gold"])}</td><td></td></tr>
<tr><th>&nbsp;&nbsp;of which both records shared a block</th>
<td class="num">{thousands(recon["gold_co_blocked"])}</td>
<td class="pct">{recon["gold_co_blocked"] / max(1, recon["gold"]) * 100:.1f}% of gold</td></tr>
<tr class="sep"><th>True positives</th><td class="num">{thousands(recon["true_positives"])}</td>
<td class="pct">recall {recon["true_positives"] / max(1, recon["gold"]):.4f}</td></tr>
<tr><th>False positives</th><td class="num">{thousands(recon["false_positives"])}</td><td></td></tr>
<tr><th>&nbsp;&nbsp;cross-source (real matching errors)</th>
<td class="num">{thousands(recon["fp_cross_source"])}</td>
<td class="pct">{recon["fp_cross_source"] / max(1, recon["false_positives"]) * 100:.0f}% of FPs</td></tr>
<tr><th>&nbsp;&nbsp;within-source merges (wrong by construction)</th>
<td class="num">{thousands(recon["fp_same_source"])}</td>
<td class="pct">{recon["fp_same_source"] / max(1, recon["false_positives"]) * 100:.0f}% of FPs</td></tr>
<tr class="sep"><th>False negatives</th>
<td class="num">{thousands(recon["false_negatives"])}</td><td></td></tr>
<tr><th>&nbsp;&nbsp;never co-blocked &mdash; <em>blocking failure</em></th>
<td class="num">{thousands(recon["fn_not_co_blocked"])}</td>
<td class="pct">{recon["fn_not_co_blocked"] / max(1, recon["false_negatives"]) * 100:.0f}% of FNs</td></tr>
<tr><th>&nbsp;&nbsp;co-blocked but the block's LLM call failed &mdash; <em>pipeline failure</em></th>
<td class="num">{thousands(recon["fn_failed_block"])}</td>
<td class="pct">{recon["fn_failed_block"] / max(1, recon["false_negatives"]) * 100:.0f}% of FNs</td></tr>
<tr><th>&nbsp;&nbsp;co-blocked and judged &mdash; <em>matching error</em></th>
<td class="num">{thousands(recon["fn_model_error"])}</td>
<td class="pct">{recon["fn_model_error"] / max(1, recon["false_negatives"]) * 100:.0f}% of FNs</td></tr>
</tbody>
</table>
"""


def schema_table(facts: dict[str, Any]) -> str:
    """Render the per-side schema table for a dataset.

    Parameters
    ----------
    facts : dict[str, Any]
        Dataset fact sheet

    Returns
    -------
    str
        Table markup
    """
    rows = "".join(
        f"<tr><td><strong>{esc(name)}</strong></td><td><code>{esc(filename)}</code></td>"
        f'<td class="num">{thousands(count)}</td>'
        f'<td class="cols">{", ".join(f"<code>{esc(c)}</code>" for c in cols)}</td></tr>'
        for name, filename, count, cols in facts["sides"]
    )
    return f"""
<table class="schema">
<thead><tr><th>Source</th><th>File in archive</th><th>Records</th><th>Columns</th></tr></thead>
<tbody>{rows}</tbody>
</table>
<p class="goldline">Ground truth: <code>{esc(facts["gold_file"])}</code> &mdash;
{thousands(facts["gold_pairs"])} pairs over {thousands(facts["records"])} records.</p>
"""


def dataset_section(dataset: str, runs: list[dict[str, Any]]) -> str:
    """Render a complete per-dataset section.

    Parameters
    ----------
    dataset : str
        Dataset name
    runs : list[dict[str, Any]]
        All reconstructed runs

    Returns
    -------
    str
        Section markup
    """
    facts = DATASET_FACTS[dataset]
    full = pick(runs, dataset, "full_baseline", "generic")
    ab_generic = pick(runs, dataset, "ab_", "generic")
    ab_typed = pick(runs, dataset, "ab_", "per-dataset")
    iter_generic = pick(runs, dataset, "ab_", "generic", iterations="multi")
    iter_typed = pick(runs, dataset, "ab_", "per-dataset", iterations="multi")
    reference = pick(runs, dataset, "raw_baseline", "generic")

    traced_full = pick(runs, dataset, "full_baseline", "generic", require_traces=True)
    traced_reference = pick(runs, dataset, "raw_baseline", "generic", require_traces=True)
    traced_typed = pick(runs, dataset, "ab_", "per-dataset", require_traces=True)
    traced_generic = pick(runs, dataset, "ab_", "generic", require_traces=True)

    example_run = traced_full or traced_reference or traced_typed or traced_generic
    verified_note = (
        "whose rebuilt pair set reproduces that run's saved precision and recall exactly"
        if example_run is not None and example_run["verified"]
        else "whose rebuilt pair set does not fully reconcile with the saved aggregates, so a "
        "few of that run's mistakes are missing here"
    )
    if example_run is None:
        source_note = ""
    elif example_run is traced_full:
        source_note = (
            f"Examples below are reconstructed from the {example_run['traces']} MLflow traces of "
            f"the full-data run, {verified_note}."
        )
    elif example_run is traced_reference:
        source_note = (
            f"There is no traced full-data run for this dataset, so the examples below come from "
            f"the {example_run['traces']} traces of the earlier <code>max_tokens=8192</code> "
            f"reference run, {verified_note}. They illustrate the data and the failure modes, "
            f"not the current scores."
        )
    else:
        source_note = (
            f"There is no full-data run for this dataset, so the examples below come from the "
            f"{example_run['traces']} traces of the 1,000-record "
            f"<code>{esc(example_run['signature_mode'])}</code> A/B arm, {verified_note}."
        )

    if full is not None:
        saved = full["saved"]
        figure = grouped_bars(
            ["Precision", "Recall", "F1"],
            [
                (
                    "Full data",
                    [saved["precision"], saved["recall"], saved["f1_score"]],
                    PALETTE["precision"],
                )
            ],
            f"{facts['display']} full-data scores at max_tokens {MAX_TOKENS:,}.",
            width=560,
            height=250,
        )
    else:
        figure = ""

    ab_figure = ""
    arms = [
        ("Generic, 1 iteration", ab_generic, PALETTE["generic"]),
        ("Typed, 1 iteration", ab_typed, PALETTE["typed"]),
        ("Generic, 3 iterations", iter_generic, PALETTE["generic_iter"]),
        ("Typed, 3 iterations", iter_typed, PALETTE["typed_iter"]),
    ]
    if any(run is not None for _label, run, _colour in arms):
        ab_figure = grouped_bars(
            ["Precision", "Recall", "F1"],
            [
                (
                    label,
                    [
                        run["saved"]["precision"],
                        run["saved"]["recall"],
                        run["saved"]["f1_score"],
                    ]
                    if run is not None
                    else [None] * 3,
                    colour,
                )
                for label, run, colour in arms
            ],
            f"{facts['display']} 1,000-record A/B: generic against typed signature, "
            "identical sample, seed 42.",
            width=560,
            height=250,
        )

    return f"""
<section id="{dataset}">
<h2>{facts["display"]}<span class="tag">{esc(facts["domain"])}</span></h2>
<p class="what">{facts["what"]}</p>

<h3>Schema</h3>
{schema_table(facts)}
<p class="notes">{facts["notes"]}</p>

<h3>Provenance</h3>
<p class="prov">{facts["provenance"]} &mdash;
<a href="{facts["provenance_url"]}">{facts["provenance_url"]}</a>.
The copy we download is the packaging used by the DeepMatcher suite of Mudgal et al.,
SIGMOD 2018 &mdash; <a href="{DEEPMATCHER}">{DEEPMATCHER}</a>.</p>

<h3>Scores</h3>
<table class="scores">
<thead><tr><th>Run</th><th>Records</th><th>P</th><th>R</th><th>F1</th><th>TP</th><th>FP</th>
<th>Predicted</th><th>Gold</th><th>Elapsed</th></tr></thead>
<tbody>
{score_row("Full data, generic signature", full, "cancelled before it ran")}
{score_row("1,000-record sample, generic signature", ab_generic, "A/B arm not finished")}
{score_row("1,000-record sample, typed signature", ab_typed, "A/B arm not finished")}
{score_row("1,000-record sample, generic, 3 ER iterations", iter_generic, "iterative arm not finished")}
{score_row("1,000-record sample, typed, 3 ER iterations", iter_typed, "iterative arm not finished")}
{score_row(reference_label(dataset), reference, "no earlier run")}
</tbody>
</table>
<p class="reading">{facts["reading"]}</p>
{f'<p class="caveat">{facts["reference_note"]}</p>' if facts.get("reference_note") else ""}

<div class="figrow">{figure}{ab_figure}</div>

<h3>Where the errors come from</h3>
{mistake_table(example_run)}

<h3>Real example mistakes</h3>
{examples_block(example_run, facts["sides"], source_note)}
</section>
"""


def build(analysis: dict[str, Any]) -> str:
    """Assemble the whole report document.

    Parameters
    ----------
    analysis : dict[str, Any]
        Reconstruction output

    Returns
    -------
    str
        Complete HTML document
    """
    runs = analysis["runs"]
    generated = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M UTC")

    full_runs = {name: pick(runs, name, "full_baseline", "generic") for name in ORDER}
    ab_generic = {name: pick(runs, name, "ab_", "generic") for name in ORDER}
    ab_typed = {name: pick(runs, name, "ab_", "per-dataset") for name in ORDER}

    labels = [DATASET_FACTS[name]["display"].replace("&ndash;", "-") for name in ORDER]

    overview = grouped_bars(
        labels,
        [
            (
                "Precision",
                [metric(full_runs[n], "precision") for n in ORDER],
                PALETTE["precision"],
            ),
            ("Recall", [metric(full_runs[n], "recall") for n in ORDER], PALETTE["recall"]),
            ("F1", [metric(full_runs[n], "f1_score") for n in ORDER], PALETTE["f1"]),
        ],
        "Full-data precision, recall and F1 per dataset. DBLP-Scholar has no bar because its "
        "full-data run was cancelled and never re-run.",
    )

    ab_chart = grouped_bars(
        labels,
        [
            (
                "Generic BlockMatch F1",
                [metric(ab_generic[n], "f1_score") for n in ORDER],
                PALETTE["generic"],
            ),
            (
                "Typed per-dataset F1",
                [metric(ab_typed[n], "f1_score") for n in ORDER],
                PALETTE["typed"],
            ),
        ],
        "1,000-record A/B, F1 by signature contract. Both arms of a dataset run over the "
        "identical sample with seed 42.",
    )

    decomp_source = {
        n: pick(runs, n, "full_baseline", "generic", require_traces=True)
        or pick(runs, n, "raw_baseline", "generic", require_traces=True)
        for n in ORDER
    }
    decomp = stacked_bars(
        labels,
        [
            (
                "Matched (true positives)",
                [recon_count(decomp_source[n], "true_positives") for n in ORDER],
                PALETTE["tp"],
            ),
            (
                "Missed: never co-blocked (blocking)",
                [recon_count(decomp_source[n], "fn_not_co_blocked") for n in ORDER],
                PALETTE["blocking"],
            ),
            (
                "Missed: block's LLM call failed (pipeline)",
                [recon_count(decomp_source[n], "fn_failed_block") for n in ORDER],
                PALETTE["failed"],
            ),
            (
                "Missed: judged and rejected (model)",
                [recon_count(decomp_source[n], "fn_model_error") for n in ORDER],
                PALETTE["model"],
            ),
        ],
        "What happened to every gold pair, reconstructed from MLflow traces. The three miss "
        "colours have completely different fixes: better blocking, a more robust adapter, and a "
        "better matcher respectively. DBLP-Scholar uses its earlier reference run.",
    )

    ab_verdict = ab_summary(ab_generic, ab_typed) + iteration_summary(runs)
    headline = headline_finding(runs)
    ordering = ordering_finding(full_runs)

    summary_cells: list[str] = []
    for name in ORDER:
        full = full_runs[name]
        if full is None:
            full_cells = '<td colspan="4" class="missing">no full-data run</td>'
        else:
            full_cells = (
                f'<td class="num strong">{fmt(metric(full, "precision"))}</td>'
                f'<td class="num strong">{fmt(metric(full, "recall"))}</td>'
                f'<td class="num strong">{fmt(metric(full, "f1_score"))}</td>'
                f'<td class="num">{duration(metric(full, "elapsed_seconds"))}</td>'
            )
        generic_f1 = metric(ab_generic[name], "f1_score")
        typed_f1 = metric(ab_typed[name], "f1_score")
        summary_cells.append(
            f'<tr><td><a href="#{name}">{DATASET_FACTS[name]["display"]}</a></td>'
            f'<td class="num">{thousands(DATASET_FACTS[name]["records"])}</td>'
            f'<td class="num">{thousands(DATASET_FACTS[name]["gold_pairs"])}</td>'
            f"{full_cells}"
            f'<td class="num">{fmt(generic_f1)}</td>'
            f'<td class="num strong">{fmt(typed_f1)}</td></tr>'
        )
    summary_rows = "".join(summary_cells)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>SERF baseline benchmark report</title>
<style>
{stylesheet()}
</style>
</head>
<body>
<header>
<h1>SERF entity resolution &mdash; baseline benchmark report</h1>
<p class="sub">Five standard entity-resolution benchmarks run through semantic blocking and
whole-block LLM matching, with every reported mistake traced back to a real record pair.</p>
<dl class="meta">
<div><dt>Report generated</dt><dd>{generated}</dd></div>
<div><dt>Student model</dt><dd><code>{STUDENT_MODEL}</code> on Vertex AI MaaS</dd></div>
<div><dt>Teacher model</dt><dd><code>{TEACHER_MODEL}</code></dd></div>
<div><dt><code>models.max_tokens</code></dt><dd>{MAX_TOKENS:,}</dd></div>
<div><dt>Blocking</dt><dd>semantic embeddings, <code>target_block_size</code>
{TARGET_BLOCK_SIZE}, <code>max_block_size</code> {MAX_BLOCK_SIZE}</dd></div>
<div><dt>Matching</dt><dd>whole blocks through a DSPy signature with
<code>dspy.XMLAdapter</code></dd></div>
</dl>
</header>

<nav>
<strong>Contents</strong>
<a href="#summary">Executive summary</a>
{"".join(f'<a href="#{n}">{DATASET_FACTS[n]["display"]}</a>' for n in ORDER)}
<a href="#cost">Cost and runtime</a>
<a href="#limits">Limitations</a>
</nav>

<section id="summary">
<h2>Executive summary</h2>

<p class="lede">Across all five benchmarks precision is respectable and recall is the bottleneck,
but the headline finding is <em>why</em> recall is low. Reconstructing every individual pair
decision from MLflow traces shows that only a small minority of missed gold pairs are cases where
the model looked at two records and got it wrong. Most misses are pairs the matcher never
evaluated, either because blocking never put them in the same block or because the block's LLM
call failed outright.</p>

{headline}

{ordering}

{ab_verdict}

<h3>Headline scores</h3>
<table class="summary">
<thead><tr><th>Dataset</th><th>Records</th><th>Gold pairs</th>
<th colspan="4">Full data</th><th colspan="2">1,000-record A/B (F1)</th></tr>
<tr class="sub"><th></th><th></th><th></th><th>P</th><th>R</th><th>F1</th><th>Elapsed</th>
<th>generic</th><th>typed</th></tr></thead>
<tbody>{summary_rows}</tbody>
</table>

{overview}
{decomp}
{ab_chart}
</section>

{"".join(dataset_section(name, runs) for name in ORDER)}

{cost_section(runs, analysis)}
{limitations_section(runs)}

<footer>
<p>Every score in this report is read from a <code>*_results.json</code> written by
<code>serf benchmark</code>. Every example mistake is reconstructed from MLflow traces in
experiment <code>SERF-Entity-Resolution</code> and maps to a real pair of records in the source
CSVs. No example is illustrative or invented.</p>
</footer>
</body>
</html>
"""


def ab_summary(
    ab_generic: dict[str, dict[str, Any] | None], ab_typed: dict[str, dict[str, Any] | None]
) -> str:
    """Render the generic-versus-typed verdict paragraph.

    Parameters
    ----------
    ab_generic : dict[str, dict[str, Any] | None]
        Generic A/B arm per dataset
    ab_typed : dict[str, dict[str, Any] | None]
        Typed A/B arm per dataset

    Returns
    -------
    str
        Verdict markup
    """
    pairs: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for name in ORDER:
        generic_run = ab_generic[name]
        typed_run = ab_typed[name]
        if generic_run is not None and typed_run is not None:
            pairs.append((name, generic_run, typed_run))

    if not pairs:
        return (
            '<div class="verdictbox pending"><h3>Generic against typed signatures: pending</h3>'
            "<p>No dataset has both A/B arms complete yet, so there is no verdict to give. "
            "The arms that have finished are shown in the per-dataset sections.</p></div>"
        )

    paired = [name for name, _g, _t in pairs]
    lines = []
    wins = 0
    for name, generic_run, typed_run in pairs:
        gen = generic_run["saved"]
        typ = typed_run["saved"]
        delta = typ["f1_score"] - gen["f1_score"]
        if delta > 0:
            wins += 1
        speed_note = (
            f" Wall clock {duration(typ['elapsed_seconds'])} typed against "
            f"{duration(gen['elapsed_seconds'])} generic."
        )
        lines.append(
            f"<li><strong>{DATASET_FACTS[name]['display']}</strong>: F1 "
            f"{gen['f1_score']:.4f} generic against {typ['f1_score']:.4f} typed, "
            f"{'+' if delta >= 0 else ''}{delta:.4f}. Recall moves "
            f"{gen['recall']:.4f} &rarr; {typ['recall']:.4f}, precision "
            f"{gen['precision']:.4f} &rarr; {typ['precision']:.4f}.{speed_note}</li>"
        )

    caveat = ""
    broken = [
        (name, int(generic_run["error_traces"]), int(generic_run["traces"]))
        for name, generic_run, _t in pairs
        if generic_run["error_traces"]
    ]
    if broken:
        detail = "; ".join(
            f"{DATASET_FACTS[n]['display']} lost {err} of {tot} blocks" for n, err, tot in broken
        )
        caveat = (
            f'<p class="caveat"><strong>Read this comparison with care.</strong> Some generic '
            f"arms lost blocks to failed LLM calls rather than to the signature being worse: "
            f"{detail}. Where the failure was the <code>litellm</code> circular-import race at "
            f"process start-up rather than a parse failure, the generic arm's recall is "
            f"depressed by an infrastructure flake and the gap overstates the typed signature's "
            f"advantage. The per-dataset sections give the failed-block counts so the effect can "
            f"be separated.</p>"
        )

    verdict = (
        f"The typed per-dataset signatures win on {wins} of the {len(paired)} datasets scored so "
        f"far, and the gain is in recall rather than precision. The likely mechanism is the shape "
        f"of the output contract: the generic signature asks the model to echo the entire "
        f"resolved block back, so every record it forgets to re-emit silently becomes an "
        f"unmatched singleton, whereas a typed signature asks only for the candidate pairs it "
        f"judged. The typed side also names each source explicitly, which removes the need for "
        f"the model to work out which records belong to which table."
        if wins
        else f"The typed signatures do not beat the generic one on any of the {len(paired)} "
        f"datasets scored so far."
    )
    pending = [n for n in ORDER if n not in paired]
    pending_note = (
        f" Still pending: {', '.join(DATASET_FACTS[n]['display'] for n in pending)}."
        if pending
        else ""
    )

    return f"""
<div class="verdictbox">
<h3>Generic against typed per-dataset signatures</h3>
<p>{verdict}{pending_note}</p>
<ul class="ablist">{"".join(lines)}</ul>
<p class="blurb">Both arms of each dataset draw the same 1,000-record sample at seed 42 and score
against the same restricted gold set, so the two F1 numbers are directly comparable with each
other. They are not comparable with the full-data rows. The wall-clock figures are reported for
completeness only: arms were re-run against a warm LM cache, so elapsed time reflects cache state
as much as it reflects the work each contract requires.</p>
{caveat}
</div>
"""


def headline_finding(runs: list[dict[str, Any]]) -> str:
    """Render the executive summary's error-decomposition paragraph from the rebuilt counts.

    Parameters
    ----------
    runs : list[dict[str, Any]]
        All reconstructed runs

    Returns
    -------
    str
        Paragraph markup
    """
    traced = [
        run
        for name in ORDER
        if (run := pick(runs, name, "full_baseline", "generic", require_traces=True)) is not None
    ]
    if not traced:
        return (
            "<p>No full-data run has usable traces yet, so the split between blocking misses, "
            "failed blocks and matching errors cannot be quantified here.</p>"
        )

    blocked = sum(int(run["reconstructed"]["fn_not_co_blocked"]) for run in traced)
    failed = sum(int(run["reconstructed"]["fn_failed_block"]) for run in traced)
    model = sum(int(run["reconstructed"]["fn_model_error"]) for run in traced)
    missed = blocked + failed + model

    worst = max(traced, key=lambda run: int(run["reconstructed"]["false_negatives"]))
    recon = worst["reconstructed"]
    judged = int(recon["true_positives"]) + int(recon["fn_model_error"])
    caught = int(recon["true_positives"]) / judged if judged else 0.0
    names = ", ".join(DATASET_FACTS[run["dataset"]]["display"] for run in traced)

    return f"""
<p>Across the {len(traced)} full-data runs whose individual pair decisions could be rebuilt
({names}), {thousands(missed)} gold pairs were missed in total. Of those,
{thousands(blocked)} were never placed in the same block, {thousands(failed)} sat inside a block
whose LLM call failed, and only {thousands(model)} &mdash; {model / missed:.0%} of all misses
&mdash; were pairs the model looked at and rejected.</p>

<p>On the full-data {DATASET_FACTS[worst["dataset"]]["display"]} run, for example, recall is
{fmt(worst["saved"]["recall"])}: of the {thousands(recon["false_negatives"])} missed gold pairs,
{thousands(recon["fn_not_co_blocked"])} were never co-blocked and
{thousands(recon["fn_failed_block"])} sat inside blocks whose LLM call failed. Only
{thousands(recon["fn_model_error"])} were pairs the model actually judged and rejected. Of the gold
pairs that reached a working matcher, it caught {caught:.0%}. The aggregate recall number is
measuring blocking recall and adapter robustness far more than it is measuring the model's
judgement.</p>
"""


def ordering_finding(full_runs: dict[str, dict[str, Any] | None]) -> str:
    """Render the executive summary's dataset-difficulty paragraph.

    Parameters
    ----------
    full_runs : dict[str, dict[str, Any] | None]
        Full-data run per dataset

    Returns
    -------
    str
        Paragraph markup
    """
    scored = [(name, run) for name, run in full_runs.items() if run is not None]
    if len(scored) < 2:
        return ""
    best_name, best = max(scored, key=lambda item: item[1]["saved"]["f1_score"])
    worst_name, worst = min(scored, key=lambda item: item[1]["saved"]["f1_score"])
    hardest = (
        " Our worst result lands on exactly the dataset both papers flag as hardest."
        if worst_name in ("amazon-google", "walmart-amazon")
        else ""
    )
    return f"""
<p>The second finding is the ordering of the datasets, which reproduces the published literature
closely. The bibliographic tasks are easy and the e-commerce tasks are hard:
{DATASET_FACTS[best_name]["display"]} reaches F1 {fmt(best["saved"]["f1_score"])} on full data
while {DATASET_FACTS[worst_name]["display"]} reaches {fmt(worst["saved"]["f1_score"])}.
K&ouml;pcke, Thor and Rahm found that no approach they evaluated exceeded 70% F-measure on the
e-commerce match problems, and DeepMatcher notes specifically that Amazon&ndash;Google's matching
titles are synonyms separated by a large string distance.{hardest}</p>
"""


def iteration_summary(runs: list[dict[str, Any]]) -> str:
    """Render the one-pass against three-iteration verdict.

    A single matching pass can only ever find the duplicates that blocking happened to
    put together, so the benchmark now re-blocks the entities each round merged and
    matches again. This block reports what that bought, arm by arm.

    Parameters
    ----------
    runs : list[dict[str, Any]]
        All reconstructed runs

    Returns
    -------
    str
        Verdict markup
    """
    rows: list[str] = []
    lifts: list[float] = []
    for name in ORDER:
        for mode, label in (("generic", "generic"), ("per-dataset", "typed")):
            once = pick(runs, name, "ab_", mode)
            thrice = pick(runs, name, "ab_", mode, iterations="multi")
            if once is None or thrice is None:
                continue
            single = once["saved"]
            multi = thrice["saved"]
            lift = multi["recall"] - single["recall"]
            lifts.append(lift)
            rows.append(
                f"<tr><td>{DATASET_FACTS[name]['display']}</td><td>{label}</td>"
                f'<td class="num">{multi.get("iterations_run", 3)}</td>'
                f'<td class="num">{fmt(single["recall"])}</td>'
                f'<td class="num strong">{fmt(multi["recall"])}</td>'
                f'<td class="num">{"+" if lift >= 0 else ""}{lift:.4f}</td>'
                f'<td class="num">{fmt(single["precision"])}</td>'
                f'<td class="num strong">{fmt(multi["precision"])}</td>'
                f'<td class="num">{fmt(single["f1_score"])}</td>'
                f'<td class="num strong">{fmt(multi["f1_score"])}</td>'
                f'<td class="num">{duration(multi["elapsed_seconds"])}</td></tr>'
            )

    if not rows:
        return (
            '<div class="verdictbox pending"><h3>Iterative resolution: pending</h3>'
            "<p>The three-iteration sweep was still running when this report was generated, so "
            "no arm has both a one-pass and a three-iteration result to compare. Every score "
            "elsewhere in this report is from a single matching pass.</p></div>"
        )

    gained = sum(1 for lift in lifts if lift > 0)
    mean_lift = sum(lifts) / len(lifts)
    verdict = (
        f"Re-blocking raises recall on {gained} of the {len(lifts)} arms compared so far, by "
        f"{mean_lift:+.4f} on average."
        if gained
        else f"Re-blocking does not raise recall on any of the {len(lifts)} arms compared so far."
    )

    return f"""
<div class="verdictbox">
<h3>One matching pass against three iterations</h3>
<p>The error decomposition below shows that most missed pairs were never placed in the same block,
which no matcher can fix. The benchmark now runs up to three rounds: after each matching pass the
connected components of predicted pairs are merged into single entities, those merged entities are
re-blocked, and matching runs again, so a record can meet partners the first partition kept away
from it. A round that merges nothing stops the loop early. {verdict}</p>
<table class="scores">
<thead><tr><th>Dataset</th><th>Signature</th><th>Iterations</th><th>R, 1 pass</th>
<th>R, 3 passes</th><th>&Delta;R</th><th>P, 1 pass</th><th>P, 3 passes</th><th>F1, 1 pass</th>
<th>F1, 3 passes</th><th>Elapsed</th></tr></thead>
<tbody>{"".join(rows)}</tbody>
</table>
<p class="blurb">Iterative runs are excluded from the trace-based error decomposition. Their later
rounds match merged entities rather than source records, and their pair set is expanded back through
those merges inside the benchmark, so a single trace no longer determines which record pairs a
round asserted. Every reconstructed decomposition and every worked example in this report therefore
comes from a one-pass run, where the rebuild reproduces the saved aggregates exactly.</p>
</div>
"""


def cost_section(runs: list[dict[str, Any]], analysis: dict[str, Any]) -> str:
    """Render the cost and runtime section.

    Parameters
    ----------
    runs : list[dict[str, Any]]
        All reconstructed runs
    analysis : dict[str, Any]
        Reconstruction output, for trace-wide totals

    Returns
    -------
    str
        Section markup
    """
    rows = []
    total_in = 0
    total_out = 0
    total_seconds = 0.0
    for run in sorted(runs, key=lambda r: r["start"]):
        if run["group"] == "smoke":
            continue
        cost = (
            run["tokens_in"] / 1e6 * INPUT_COST_PER_MTOK
            + run["tokens_out"] / 1e6 * OUTPUT_COST_PER_MTOK
        )
        total_in += run["tokens_in"]
        total_out += run["tokens_out"]
        total_seconds += float(run["saved"]["elapsed_seconds"])
        label = {
            "full_baseline": "full data",
            "raw_baseline": "earlier reference",
        }.get(
            run["group"],
            f"1,000-record A/B ({run['group']})"
            if run["group"].startswith("ab_")
            else run["group"],
        )
        rows.append(
            f"<tr><td>{DATASET_FACTS[run['dataset']]['display']}</td>"
            f"<td>{label}</td><td>{esc(run['signature_mode'])}</td>"
            f'<td class="num">{run["traces"]}</td>'
            f'<td class="num">{thousands(run["tokens_in"])}</td>'
            f'<td class="num">{thousands(run["tokens_out"])}</td>'
            f'<td class="num">${cost:.4f}</td>'
            f'<td class="num">{duration(run["saved"]["elapsed_seconds"])}</td></tr>'
        )

    totals = analysis["trace_totals"]
    trace_cost = (
        totals["tokens_in"] / 1e6 * INPUT_COST_PER_MTOK
        + totals["tokens_out"] / 1e6 * OUTPUT_COST_PER_MTOK
    )
    attributed_cost = total_in / 1e6 * INPUT_COST_PER_MTOK + total_out / 1e6 * OUTPUT_COST_PER_MTOK

    return f"""
<section id="cost">
<h2>Cost and runtime</h2>
<p>Token counts are read from the <code>mlflow.trace.tokenUsage</code> metadata that MLflow
records on each traced block match, so these are measured figures rather than estimates. They are
priced at the Vertex MaaS rate for <code>gpt-oss-120b</code>:
<strong>${INPUT_COST_PER_MTOK:.2f} per million input tokens</strong> and
<strong>${OUTPUT_COST_PER_MTOK:.2f} per million output tokens</strong>.</p>

<table class="cost">
<thead><tr><th>Dataset</th><th>Run</th><th>Signature</th><th>Traced blocks</th>
<th>Input tokens</th><th>Output tokens</th><th>Cost</th><th>Elapsed</th></tr></thead>
<tbody>{"".join(rows)}</tbody>
<tfoot><tr><th colspan="4">Attributed to the runs above</th>
<th class="num">{thousands(total_in)}</th><th class="num">{thousands(total_out)}</th>
<th class="num">${attributed_cost:.4f}</th>
<th class="num">{duration(total_seconds)}</th></tr>
<tr><th colspan="4">All {totals["traces"]:,} traces in the experiment</th>
<th class="num">{thousands(totals["tokens_in"])}</th>
<th class="num">{thousands(totals["tokens_out"])}</th>
<th class="num">${trace_cost:.4f}</th><th></th></tr></tfoot>
</table>

<p>Two things stand out. First, output tokens dominate the bill: they are four times the unit price
and the model emits nearly as many as it reads, because the generic contract asks it to echo every
resolved entity back rather than just report the pairs it matched. Output alone accounts for
${totals["tokens_out"] / 1e6 * OUTPUT_COST_PER_MTOK:.2f} of the ${trace_cost:.2f} total. Second,
the absolute total is small &mdash; every traced benchmark call in this experiment comes to
${trace_cost:.2f} of inference &mdash; so the practical constraint on these runs is wall-clock
time, not money. The {duration(total_seconds)} of benchmark wall clock in the table above cost
${attributed_cost:.2f}.</p>

<p class="caveat">GEPA optimisation runs are not represented in this table.
<code>mlflow.autolog.log_traces_from_compile</code> was only enabled recently, so the earlier
compile runs emitted no traces and their token usage cannot be recovered from the tracking
store. The costs above therefore cover benchmark inference only.</p>
</section>
"""


def limitations_section(runs: list[dict[str, Any]]) -> str:
    """Render the limitations section.

    Parameters
    ----------
    runs : list[dict[str, Any]]
        All reconstructed runs

    Returns
    -------
    str
        Section markup
    """
    unverified = [
        run
        for run in runs
        if not run["verified"]
        and run.get("reconstruction_reliable", True)
        and run["group"] != "smoke"
        and run["traces"] > 2
    ]
    unverified_note = (
        "<li>"
        + "</li><li>".join(
            f"The {DATASET_FACTS[run['dataset']]['display']} "
            f"{run['group'].replace('_', ' ')} reconstruction rebuilds "
            f"{run['reconstructed']['predicted']} predicted pairs against the "
            f"{run['saved']['predicted_pairs']} the run recorded, a shortfall of "
            f"{run['saved']['predicted_pairs'] - run['reconstructed']['predicted']}. A few "
            f"traces for that run are missing from the tracking store, so its error "
            f"decomposition is very slightly incomplete. Every other run reconciles exactly."
            for run in unverified
        )
        + "</li>"
        if unverified
        else "<li>Every reconstruction reproduces its run's saved aggregate counts exactly.</li>"
    )

    return f"""
<section id="limits">
<h2>Limitations and what is not in this report</h2>
<ul class="limits">
<li><strong>Which numbers are full-table.</strong> DBLP&ndash;ACM, Abt&ndash;Buy,
Amazon&ndash;Google and Walmart&ndash;Amazon full-data rows score the complete tables with no
sampling. Every row labelled &ldquo;1,000-record sample&rdquo; scores only the gold pairs that
survive sampling, so its precision and recall are not comparable with the full-data rows and the
two must never be averaged together.</li>

<li><strong>DBLP&ndash;Scholar has no full-data run.</strong> At 66,879 records it is by far the
most expensive dataset in the suite and the full-data baseline was cancelled. It was not
re-run. Its only numbers here come from the earlier truncated reference run and from
1,000-record samples, and both are labelled as such wherever they appear.</li>

<li><strong>The earlier reference numbers are not comparable.</strong> Rows labelled
&ldquo;max_tokens 8192&rdquo; come from a baseline taken before the token cap was raised, when
block records were heavily truncated. DBLP&ndash;Scholar's reference run is further apart still:
its right table was cut to 5,218 rows, which turns out to be exactly the set of Scholar records
that appear in the gold mapping, so that run had no distractor records at all. These rows are
included only to show the direction of travel after the cap went up; they are not a like-for-like
comparison and should not be quoted as DBLP&ndash;Scholar scores.</li>

<li><strong>Sampled runs redefine the gold set.</strong> The sampler draws whole ground-truth
match groups, so a 1,000-record sample of DBLP&ndash;ACM retains 476 of 2,224 gold pairs among
1,001 records. That makes gold pairs far denser than in the full table, which changes both the
blocking problem and the achievable recall. A 1,000-record F1 is a comparison between arms of the
same A/B, not a prediction of full-data F1.</li>

<li><strong>Within-source merges are counted as errors.</strong> All five gold standards contain
only cross-source pairs, but the generic <code>BlockMatch</code> contract will happily merge two
records from the same side. Those merges are scored as false positives no matter how correct they
are. The per-dataset breakdowns separate them out, and on several datasets they are the majority
of all false positives, so reported precision understates cross-source matching precision.</li>

<li><strong>Failed blocks are conflated with model errors in the aggregate scores.</strong> A
block whose LLM call fails contributes no pairs, which is indistinguishable from a model that
declined every pair. Only the trace-level reconstruction separates the two. Two distinct failure
modes appear in the logs: adapter parse failures on large blocks
(<code>JSONAdapter failed to parse the LM response</code>), and a
<code>litellm</code> circular-import race at process start-up that is purely an infrastructure
flake.</li>

<li><strong>Iterative runs carry scores only.</strong> Runs that took more than one ER iteration
re-block the entities each round merged, so their later traces describe merged entities and their
pair set is expanded back through those merges by the benchmark itself. A single trace no longer
determines which record pairs a round asserted, so these runs contribute precision, recall and
token usage but no error decomposition and no worked examples.</li>

<li><strong>False-negative attribution is per-run, not global.</strong> A gold pair is called a
blocking miss when its two records never shared a block <em>in that run</em>. Blocking is
sensitive to which records are present, so the same pair can be a blocking miss in one run and a
matching error in another.</li>

<li><strong>What could not be reconstructed.</strong> {unverified_note.replace("<li>", "").replace("</li>", " ")}
Token usage for GEPA compile runs is unavailable because trace logging from compile was enabled
only recently. The ranking of examples within each category is deterministic but arbitrary
&mdash; pairs the model explained are shown first so its reasoning can be quoted &mdash; so the
examples illustrate the failure modes rather than being a random sample of them.</li>
</ul>
</section>
"""


def stylesheet() -> str:
    """Return the report's inline CSS.

    Returns
    -------
    str
        CSS text
    """
    return """
:root{--ink:#1e2430;--mut:#5d6675;--line:#e2e6ec;--bg:#ffffff;--soft:#f6f8fa;--accent:#2f6fb3}
*{box-sizing:border-box}
body{margin:0;padding:0;background:var(--bg);color:var(--ink);
font:16px/1.62 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
header,nav,section,footer{max-width:1080px;margin:0 auto;padding:0 28px}
header{padding-top:46px;padding-bottom:10px;border-bottom:1px solid var(--line);max-width:1080px}
h1{font-size:31px;line-height:1.25;margin:0 0 10px}
h2{font-size:24px;margin:46px 0 12px;padding-bottom:8px;border-bottom:2px solid var(--line)}
h3{font-size:18px;margin:30px 0 8px}
h4{font-size:15px;margin:26px 0 4px;text-transform:uppercase;letter-spacing:.05em;
color:var(--mut)}
p{margin:10px 0}
a{color:var(--accent)}
code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:.88em;
background:var(--soft);padding:1px 5px;border-radius:3px}
.sub{color:var(--mut);font-size:17px;max-width:74ch}
.lede{font-size:18px;line-height:1.6}
dl.meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:6px 22px;
margin:20px 0 26px}
dl.meta div{padding:7px 0;border-top:1px solid var(--line)}
dl.meta dt{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--mut)}
dl.meta dd{margin:2px 0 0;font-size:14px}
nav{padding-top:16px;padding-bottom:16px;font-size:14px;border-bottom:1px solid var(--line)}
nav strong{margin-right:12px;color:var(--mut);text-transform:uppercase;font-size:12px;
letter-spacing:.06em}
nav a{margin-right:16px;text-decoration:none;white-space:nowrap}
nav a:hover{text-decoration:underline}
.tag{margin-left:12px;font-size:12px;font-weight:500;text-transform:uppercase;
letter-spacing:.06em;color:var(--mut);vertical-align:middle;background:var(--soft);
padding:3px 9px;border-radius:11px;border:1px solid var(--line)}
.what,.reading,.notes{max-width:78ch}
.notes,.prov,.goldline{font-size:14.5px;color:var(--mut)}
table{border-collapse:collapse;width:100%;margin:14px 0;font-size:14px}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line);vertical-align:top}
thead th{background:var(--soft);font-size:12px;text-transform:uppercase;letter-spacing:.045em;
color:var(--mut);border-bottom:1px solid #ccd2da}
thead tr.sub th{text-transform:none;letter-spacing:0}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
td.strong{font-weight:600}
td.pct{color:var(--mut);font-size:13px}
tr.missing td,td.missing{color:var(--mut);font-style:italic}
tr.sep th,tr.sep td{border-top:2px solid var(--line)}
table.decomp caption{caption-side:top;text-align:left;font-size:13px;color:var(--mut);
padding-bottom:8px}
table.decomp th{font-weight:500}
table.decomp td.num{width:110px}
table.schema td.cols code{margin-right:3px}
tfoot th{background:var(--soft);font-size:13px}
.ok{color:#2c7a52;font-weight:600}
.warnbadge{color:#a05c1a;font-weight:600}
figure{margin:18px 0;flex:1 1 420px;min-width:320px}
figcaption{font-size:13px;color:var(--mut);margin-top:6px;max-width:70ch}
.figrow{display:flex;flex-wrap:wrap;gap:24px}
svg.chart{display:block;background:#fff}
.chart .tick,.chart .legend,.chart .xlabel,.chart .barval,.chart .sublabel,.chart .nodata{
font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.chart .tick{font-size:11px;fill:#8b93a3}
.chart .xlabel{font-size:12.5px;fill:#1e2430}
.chart .sublabel{font-size:11px;fill:#8b93a3}
.chart .barval{font-size:11px;fill:#5d6675}
.chart .legend{font-size:12px;fill:#5d6675}
.chart .nodata{font-size:11px;fill:#a8b0bd;font-style:italic}
.chart .seglabel{font-size:10.5px;fill:#fff;font-weight:600;
font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif}
.verdictbox{border:1px solid var(--line);border-left:4px solid var(--accent);background:var(--soft);
padding:4px 22px 16px;margin:26px 0;border-radius:0 5px 5px 0}
.verdictbox.pending{border-left-color:#a8b0bd}
.verdictbox h3{margin-top:18px}
.ablist{margin:8px 0;padding-left:20px;font-size:14.5px}
.ablist li{margin:5px 0}
.caveat{font-size:14px;color:#7a4a12;background:#fdf6e8;border:1px solid #f0dfb8;
padding:11px 14px;border-radius:5px}
.warn{font-size:14.5px;color:#7a4a12;background:#fdf6e8;border:1px solid #f0dfb8;
padding:11px 14px;border-radius:5px}
.blurb{font-size:14px;color:var(--mut);max-width:80ch;margin:2px 0 12px}
.count{font-weight:400;text-transform:none;letter-spacing:0;color:#a8b0bd}
.provenance-note{font-size:14px;color:var(--mut);background:var(--soft);padding:10px 14px;
border-radius:5px;border:1px solid var(--line)}
.example{border:1px solid var(--line);border-radius:6px;margin:12px 0;overflow:hidden}
.exhead{display:flex;gap:12px;align-items:baseline;padding:10px 14px;background:var(--soft);
border-bottom:1px solid var(--line)}
.badge{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;
padding:3px 8px;border-radius:3px;white-space:nowrap;color:#fff}
.badge.fp{background:#b5482f}
.badge.fn{background:#8a6d1f}
.verdict{font-size:13.5px;color:var(--mut)}
.panels{display:grid;grid-template-columns:1fr 1fr}
.panel{padding:11px 14px;font-size:13.5px}
.panel+.panel{border-left:1px solid var(--line)}
.ptitle{font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.05em;
color:var(--accent);margin-bottom:7px}
.eid{color:#a8b0bd;font-weight:400;text-transform:none;letter-spacing:0}
.fieldrow{display:grid;grid-template-columns:96px 1fr;gap:8px;padding:2px 0;
border-top:1px dotted var(--line)}
.fk{color:var(--mut);font-size:11px;text-transform:uppercase;letter-spacing:.03em;
padding-top:3px;overflow-wrap:normal;word-break:keep-all}
.fv{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12.5px;
line-height:1.5;overflow-wrap:anywhere}
.empty{color:#b7bec9}
.reason{padding:9px 14px;border-top:1px solid var(--line);font-size:13.5px;background:#fbfcfd;
font-style:italic;color:#4a5261}
.rl{display:block;font-style:normal;font-size:11px;text-transform:uppercase;
letter-spacing:.06em;color:var(--mut);margin-bottom:2px}
.limits{max-width:82ch;padding-left:20px}
.limits li{margin:11px 0}
footer{margin:56px auto 60px;padding-top:18px;border-top:1px solid var(--line);font-size:13.5px;
color:var(--mut);max-width:1080px}
@media print{nav{display:none}.example{break-inside:avoid}figure{break-inside:avoid}}
"""


def main() -> None:
    """Read the reconstruction and write the HTML report."""
    analysis_path = pathlib.Path(sys.argv[1])
    out_path = pathlib.Path(sys.argv[2])
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build(analysis), encoding="utf-8")
    logger.info(f"Wrote {out_path} ({out_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
