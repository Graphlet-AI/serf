"""Benchmark datasets for entity resolution evaluation."""

import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

from serf.config import config
from serf.dspy.types import Entity
from serf.eval.metrics import evaluate_resolution
from serf.logs import get_logger

logger = get_logger(__name__)

# Dataset registry with download URLs
DATASET_REGISTRY: dict[str, dict[str, str]] = {
    "walmart-amazon": {
        "url": (
            "https://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/"
            "Walmart-Amazon/walmart_amazon_exp_data.zip"
        ),
        "domain": "products",
        "difficulty": "hard",
    },
    "abt-buy": {
        "url": (
            "https://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/"
            "Abt-Buy/abt_buy_exp_data.zip"
        ),
        "domain": "products",
        "difficulty": "hard",
    },
    "amazon-google": {
        "url": (
            "https://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/"
            "Amazon-Google/amazon_google_exp_data.zip"
        ),
        "domain": "products",
        "difficulty": "hard",
    },
    "dblp-acm": {
        "url": (
            "https://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/"
            "DBLP-ACM/dblp_acm_exp_data.zip"
        ),
        "domain": "bibliographic",
        "difficulty": "easy",
    },
    "dblp-scholar": {
        "url": (
            "https://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/"
            "DBLP-GoogleScholar/dblp_scholar_exp_data.zip"
        ),
        "domain": "bibliographic",
        "difficulty": "medium",
    },
}

# Candidate columns for entity name (first match wins)
NAME_CANDIDATES = ("title", "name", "product_name", "product_title", "book_title")

# Offset for right table IDs to avoid collisions with left table
RIGHT_ID_OFFSET = 100000


def _load_csv(path: Path) -> pd.DataFrame:
    """Load CSV with encoding fallback (utf-8, then latin-1)."""
    for encoding in ("utf-8", "latin-1"):
        try:
            df: pd.DataFrame = pd.read_csv(path, encoding=encoding)
            return df
        except UnicodeDecodeError:
            continue
    df = pd.read_csv(path, encoding="latin-1")
    return df


def _find_csv_dir(root: Path) -> Path:
    """Find directory containing tableA.csv and tableB.csv."""
    if (root / "tableA.csv").exists() and (root / "tableB.csv").exists():
        return root
    for path in root.rglob("tableA.csv"):
        parent = path.parent
        if (parent / "tableB.csv").exists():
            return parent
    raise FileNotFoundError(f"Could not find tableA.csv/tableB.csv under {root}")


def _build_ground_truth(csv_dir: Path) -> set[tuple[int, int]]:
    """Build ground truth from train/valid/test label files."""
    pairs: set[tuple[int, int]] = set()
    for fname in ("train.csv", "valid.csv", "test.csv"):
        path = csv_dir / fname
        if not path.exists():
            continue
        df = _load_csv(path)
        if (
            "ltable_id" not in df.columns
            or "rtable_id" not in df.columns
            or "label" not in df.columns
        ):
            logger.warning("Skipping %s: missing ltable_id, rtable_id, or label", fname)
            continue
        matches = df[df["label"] == 1]
        for _, row in matches.iterrows():
            pairs.add((int(row["ltable_id"]), int(row["rtable_id"])))
    return pairs


def _row_to_entity(
    row: pd.Series,
    entity_id: int,
    prefix: str,
    name_col: str | None,
    text_cols: list[str],
) -> Entity:
    """Convert a DataFrame row to an Entity."""
    attrs = {f"{prefix}{k}": v for k, v in row.items() if pd.notna(v)}
    for k, v in list(attrs.items()):
        if isinstance(v, float) and v != v:  # NaN
            del attrs[k]
        elif not isinstance(v, str):
            attrs[k] = str(v)

    name = ""
    if name_col and name_col in row and pd.notna(row[name_col]):
        name = str(row[name_col])
    elif "id" in row and pd.notna(row["id"]):
        name = str(row["id"])

    desc_parts = []
    for col in text_cols:
        if col == name_col or col == "id":
            continue
        if col in row and pd.notna(row[col]) and isinstance(row[col], str):
            desc_parts.append(str(row[col]))
    description = " ".join(desc_parts) if desc_parts else ""

    return Entity(
        id=entity_id,
        name=name or "unknown",
        description=description,
        entity_type="entity",
        attributes=attrs,
    )


def _detect_name_column(df: pd.DataFrame) -> str | None:
    """Detect best column for entity name."""
    cols = [c for c in df.columns if c.lower() in (n.lower() for n in NAME_CANDIDATES)]
    return cols[0] if cols else None


def _get_text_columns(df: pd.DataFrame) -> list[str]:
    """Get columns that look like text (object/string type)."""
    return [c for c in df.columns if df[c].dtype == "object" or str(df[c].dtype) == "string"]


class BenchmarkDataset:
    """Standard ER benchmark dataset in DeepMatcher format."""

    def __init__(
        self,
        name: str,
        table_a: pd.DataFrame,
        table_b: pd.DataFrame,
        ground_truth: set[tuple[int, int]],
        metadata: dict[str, str],
    ) -> None:
        """Initialize benchmark dataset.

        Parameters
        ----------
        name : str
            Dataset name (e.g. "walmart-amazon").
        table_a : pd.DataFrame
            Left entity table with id column.
        table_b : pd.DataFrame
            Right entity table with id column.
        ground_truth : set of tuple of (int, int)
            True matching pairs (ltable_id, rtable_id).
        metadata : dict of str to str
            Extra metadata (domain, difficulty, etc.).
        """
        self.name = name
        self.table_a = table_a
        self.table_b = table_b
        self.ground_truth = ground_truth
        self.metadata = metadata

    @classmethod
    def available_datasets(cls) -> list[str]:
        """Return list of available dataset names."""
        return list(DATASET_REGISTRY.keys())

    @classmethod
    def download(cls, name: str, output_dir: str | None = None) -> "BenchmarkDataset":
        """Download and prepare a benchmark dataset.

        Downloads the zip from the DeepMatcher URL, extracts it, loads the CSVs,
        and builds the ground truth from train/valid/test label files.

        Parameters
        ----------
        name : str
            Dataset name from DATASET_REGISTRY.
        output_dir : str, optional
            Directory to download and extract. Default from config.

        Returns
        -------
        BenchmarkDataset
            Loaded dataset instance.
        """
        if name not in DATASET_REGISTRY:
            raise ValueError(f"Unknown dataset: {name}. Available: {cls.available_datasets()}")

        out = Path(output_dir or config.get("benchmarks.output_dir", "data/benchmarks"))
        out = out / name
        out.mkdir(parents=True, exist_ok=True)

        url = DATASET_REGISTRY[name]["url"]
        zip_path = out / "data.zip"
        logger.info("Downloading %s to %s", url, zip_path)
        urlretrieve(url, zip_path)

        extract_root = out / "extracted"
        extract_root.mkdir(exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_root)

        csv_dir = _find_csv_dir(extract_root)
        table_a = _load_csv(csv_dir / "tableA.csv")
        table_b = _load_csv(csv_dir / "tableB.csv")
        ground_truth = _build_ground_truth(csv_dir)
        metadata = {k: v for k, v in DATASET_REGISTRY[name].items() if k != "url"}

        return cls(
            name=name,
            table_a=table_a,
            table_b=table_b,
            ground_truth=ground_truth,
            metadata=metadata,
        )

    @classmethod
    def load(cls, name: str, data_dir: str) -> "BenchmarkDataset":
        """Load a previously downloaded benchmark dataset from disk.

        Parameters
        ----------
        name : str
            Dataset name.
        data_dir : str
            Root directory containing the dataset (with tableA.csv, tableB.csv,
            train.csv, valid.csv, test.csv).

        Returns
        -------
        BenchmarkDataset
            Loaded dataset instance.
        """
        root = Path(data_dir)
        if not root.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        csv_dir = _find_csv_dir(root)
        table_a = _load_csv(csv_dir / "tableA.csv")
        table_b = _load_csv(csv_dir / "tableB.csv")
        ground_truth = _build_ground_truth(csv_dir)
        metadata = DATASET_REGISTRY.get(name, {}).copy()
        metadata.pop("url", None)

        return cls(
            name=name,
            table_a=table_a,
            table_b=table_b,
            ground_truth=ground_truth,
            metadata=metadata,
        )

    def evaluate(self, predicted_pairs: set[tuple[int, int]]) -> dict[str, float]:
        """Evaluate predictions against ground truth.

        Parameters
        ----------
        predicted_pairs : set of tuple of (int, int)
            Predicted matches as (ltable_id, rtable_id).

        Returns
        -------
        dict of str to float
            Metrics: precision, recall, f1_score.
        """
        return evaluate_resolution(predicted_pairs, self.ground_truth)

    def to_entities(self) -> tuple[list[Entity], list[Entity]]:
        """Convert tables to Entity objects for the pipeline.

        Returns (left_entities, right_entities) where each entity has:
        - id: the original table id (right entities offset by 100000)
        - name: first text column or title column
        - description: concatenation of other text columns
        - attributes: all columns as a dict (l_/r_ prefix)

        Returns
        -------
        tuple of (list of Entity, list of Entity)
            Left and right entities.
        """
        name_col_a = _detect_name_column(self.table_a) or "id"
        name_col_b = _detect_name_column(self.table_b) or "id"
        text_cols_a = _get_text_columns(self.table_a)
        text_cols_b = _get_text_columns(self.table_b)

        left_entities: list[Entity] = []
        for _, row in self.table_a.iterrows():
            eid = int(row["id"])
            ent = _row_to_entity(row, eid, "l_", name_col_a, text_cols_a)
            left_entities.append(ent)

        right_entities: list[Entity] = []
        for _, row in self.table_b.iterrows():
            orig_id = int(row["id"])
            eid = orig_id + RIGHT_ID_OFFSET
            ent = _row_to_entity(row, eid, "r_", name_col_b, text_cols_b)
            right_entities.append(ent)

        return (left_entities, right_entities)
