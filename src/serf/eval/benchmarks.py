"""Benchmark datasets for entity resolution evaluation."""

import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from serf.config import config
from serf.dspy.types import Entity
from serf.eval.metrics import evaluate_resolution
from serf.logs import get_logger

logger = get_logger(__name__)

# Dataset registry with download URLs
# Leipzig format: tableA, tableB, perfectMapping CSV in a zip
DATASET_REGISTRY: dict[str, dict[str, str]] = {
    "dblp-acm": {
        "url": "https://dbs.uni-leipzig.de/files/datasets/DBLP-ACM.zip",
        "table_a_name": "DBLP2.csv",
        "table_b_name": "ACM.csv",
        "mapping_name": "DBLP-ACM_perfectMapping.csv",
        "mapping_col_a": "idDBLP",
        "mapping_col_b": "idACM",
        "domain": "bibliographic",
        "difficulty": "easy",
    },
    "dblp-scholar": {
        "url": "https://dbs.uni-leipzig.de/files/datasets/DBLP-Scholar.zip",
        "table_a_name": "DBLP1.csv",
        "table_b_name": "Scholar.csv",
        "mapping_name": "DBLP-Scholar_perfectMapping.csv",
        "mapping_col_a": "idDBLP",
        "mapping_col_b": "idScholar",
        "domain": "bibliographic",
        "difficulty": "medium",
    },
    "abt-buy": {
        "url": "https://dbs.uni-leipzig.de/files/datasets/Abt-Buy.zip",
        "table_a_name": "Abt.csv",
        "table_b_name": "Buy.csv",
        "mapping_name": "abt_buy_perfectMapping.csv",
        "mapping_col_a": "idAbt",
        "mapping_col_b": "idBuy",
        "domain": "products",
        "difficulty": "hard",
    },
}

# Candidate columns for entity name (first match wins)
NAME_CANDIDATES = ("title", "name", "product_name", "product_title", "book_title")

# Offset for right table IDs to avoid collisions with left table
RIGHT_ID_OFFSET = 100000


def _load_csv(path: Path) -> pd.DataFrame:
    """Load CSV with encoding fallback (utf-8, then latin-1).

    Parameters
    ----------
    path : Path
        Path to the CSV file

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame
    """
    for encoding in ("utf-8", "latin-1"):
        try:
            df: pd.DataFrame = pd.read_csv(path, encoding=encoding)
            return df
        except UnicodeDecodeError:
            continue
    df = pd.read_csv(path, encoding="latin-1")
    return df


def _load_csv_from_zip(zf: zipfile.ZipFile, name: str) -> pd.DataFrame:
    """Load a CSV file from inside a zip archive.

    Parameters
    ----------
    zf : zipfile.ZipFile
        Open zip file
    name : str
        Name of the CSV file within the zip

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame
    """
    import io

    with zf.open(name) as f:
        raw = f.read()
    for encoding in ("utf-8", "latin-1"):
        try:
            text = raw.decode(encoding)
            df: pd.DataFrame = pd.read_csv(io.StringIO(text))
            return df
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    text = raw.decode("latin-1", errors="replace")
    df = pd.read_csv(io.StringIO(text))
    return df


def _build_ground_truth_deepmatcher(csv_dir: Path) -> set[tuple[int, int]]:
    """Build ground truth from DeepMatcher train/valid/test label files.

    Parameters
    ----------
    csv_dir : Path
        Directory containing train.csv, valid.csv, test.csv

    Returns
    -------
    set[tuple[int, int]]
        Set of (ltable_id, rtable_id) match pairs
    """
    pairs: set[tuple[int, int]] = set()
    for fname in ("train.csv", "valid.csv", "test.csv"):
        path = csv_dir / fname
        if not path.exists():
            continue
        df = _load_csv(path)
        if "ltable_id" not in df.columns or "rtable_id" not in df.columns:
            continue
        if "label" not in df.columns:
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
    """Convert a DataFrame row to an Entity.

    Parameters
    ----------
    row : pd.Series
        Row from a pandas DataFrame
    entity_id : int
        ID to assign to the entity
    prefix : str
        Prefix for attribute keys (e.g. "l_" or "r_")
    name_col : str | None
        Column to use as entity name
    text_cols : list[str]
        Columns to include in description

    Returns
    -------
    Entity
        Converted entity
    """
    attrs: dict[str, str] = {}
    for k, v in row.items():
        if pd.notna(v):
            attrs[f"{prefix}{k}"] = str(v)

    name = ""
    if name_col and name_col in row.index and pd.notna(row[name_col]):
        name = str(row[name_col])
    elif "id" in row.index and pd.notna(row["id"]):
        name = str(row["id"])

    desc_parts = []
    for col in text_cols:
        if col == name_col or col == "id":
            continue
        if col in row.index and pd.notna(row[col]) and isinstance(row[col], str):
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
    """Detect best column for entity name.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to inspect

    Returns
    -------
    str | None
        Name of the detected column, or None
    """
    for candidate in NAME_CANDIDATES:
        for col in df.columns:
            if str(col).lower() == candidate.lower():
                return str(col)
    return None


def _get_text_columns(df: pd.DataFrame) -> list[str]:
    """Get columns that contain text data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to inspect

    Returns
    -------
    list[str]
        Names of text columns
    """
    return [c for c in df.columns if df[c].dtype == "object" or str(df[c].dtype).startswith("str")]


class BenchmarkDataset:
    """Standard ER benchmark dataset.

    Supports two formats:
    1. Leipzig format: tableA.csv, tableB.csv, perfectMapping.csv
    2. DeepMatcher format: tableA.csv, tableB.csv, train/valid/test.csv with labels

    Parameters
    ----------
    name : str
        Dataset name
    table_a : pd.DataFrame
        Left entity table
    table_b : pd.DataFrame
        Right entity table
    ground_truth : set[tuple[int, int]]
        True matching pairs (left_id, right_id)
    metadata : dict[str, str]
        Extra metadata
    """

    def __init__(
        self,
        name: str,
        table_a: pd.DataFrame,
        table_b: pd.DataFrame,
        ground_truth: set[tuple[int, int]],
        metadata: dict[str, str],
    ) -> None:
        self.name = name
        self.table_a = table_a
        self.table_b = table_b
        self.ground_truth = ground_truth
        self.metadata = metadata

    @classmethod
    def available_datasets(cls) -> list[str]:
        """Return list of available dataset names.

        Returns
        -------
        list[str]
            Available dataset names
        """
        return list(DATASET_REGISTRY.keys())

    @classmethod
    def download(cls, name: str, output_dir: str | None = None) -> "BenchmarkDataset":
        """Download and prepare a benchmark dataset.

        Parameters
        ----------
        name : str
            Dataset name from DATASET_REGISTRY
        output_dir : str | None
            Directory to save data. Default from config.

        Returns
        -------
        BenchmarkDataset
            Loaded dataset instance
        """
        if name not in DATASET_REGISTRY:
            raise ValueError(f"Unknown dataset: {name}. Available: {cls.available_datasets()}")

        info = DATASET_REGISTRY[name]
        out = Path(output_dir or config.get("benchmarks.output_dir", "data/benchmarks"))
        out = out / name
        out.mkdir(parents=True, exist_ok=True)

        url = info["url"]
        zip_path = out / "data.zip"

        if not zip_path.exists():
            logger.info("Downloading %s to %s", url, zip_path)
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with (
                urllib.request.urlopen(req, timeout=60) as response,
                open(zip_path, "wb") as f,
            ):
                f.write(response.read())
        else:
            logger.info("Using cached %s", zip_path)

        # Load from zip (Leipzig format)
        with zipfile.ZipFile(zip_path, "r") as zf:
            table_a = _load_csv_from_zip(zf, info["table_a_name"])
            table_b = _load_csv_from_zip(zf, info["table_b_name"])
            mapping_df = _load_csv_from_zip(zf, info["mapping_name"])

        # Build ground truth from perfect mapping
        col_a = info["mapping_col_a"]
        col_b = info["mapping_col_b"]
        ground_truth: set[tuple[int, int]] = set()

        # Build ID lookup maps (IDs can be strings like "conf/sigmod/...")
        a_id_to_int: dict[str, int] = {
            str(row["id"]): i for i, (_idx, row) in enumerate(table_a.iterrows())
        }
        b_id_to_int: dict[str, int] = {
            str(row["id"]): i for i, (_idx, row) in enumerate(table_b.iterrows())
        }

        for _, row in mapping_df.iterrows():
            a_key = str(row[col_a])
            b_key = str(row[col_b])
            if a_key in a_id_to_int and b_key in b_id_to_int:
                a_int: int = a_id_to_int[a_key]
                b_int: int = b_id_to_int[b_key] + RIGHT_ID_OFFSET
                ground_truth.add((a_int, b_int))

        metadata = {
            k: v
            for k, v in info.items()
            if k
            not in (
                "url",
                "table_a_name",
                "table_b_name",
                "mapping_name",
                "mapping_col_a",
                "mapping_col_b",
            )
        }

        logger.info(
            "Loaded %s: %d left, %d right, %d ground truth pairs",
            name,
            len(table_a),
            len(table_b),
            len(ground_truth),
        )

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

        Supports both DeepMatcher format (tableA/B + train/valid/test) and
        Leipzig format (source tables + perfectMapping).

        Parameters
        ----------
        name : str
            Dataset name
        data_dir : str
            Root directory containing the dataset CSVs

        Returns
        -------
        BenchmarkDataset
            Loaded dataset instance
        """
        root = Path(data_dir)
        if not root.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        # Check for Leipzig format first (zip file)
        zip_path = root / "data.zip"
        if zip_path.exists() and name in DATASET_REGISTRY:
            return cls.download(name, str(root.parent))

        # DeepMatcher format
        table_a_path = root / "tableA.csv"
        table_b_path = root / "tableB.csv"
        if table_a_path.exists() and table_b_path.exists():
            table_a = _load_csv(table_a_path)
            table_b = _load_csv(table_b_path)
            ground_truth = _build_ground_truth_deepmatcher(root)
            metadata = DATASET_REGISTRY.get(name, {}).copy()
            metadata.pop("url", None)
            return cls(
                name=name,
                table_a=table_a,
                table_b=table_b,
                ground_truth=ground_truth,
                metadata=metadata,
            )

        raise FileNotFoundError(
            f"Could not find benchmark data in {data_dir}. "
            "Expected tableA.csv/tableB.csv or data.zip."
        )

    def evaluate(self, predicted_pairs: set[tuple[int, int]]) -> dict[str, float]:
        """Evaluate predictions against ground truth.

        Parameters
        ----------
        predicted_pairs : set[tuple[int, int]]
            Predicted matches as (left_id, right_id) pairs

        Returns
        -------
        dict[str, float]
            Metrics: precision, recall, f1_score
        """
        return evaluate_resolution(predicted_pairs, self.ground_truth)

    def to_entities(self) -> tuple[list[Entity], list[Entity]]:
        """Convert tables to Entity objects for the pipeline.

        Left entities use row index as ID, right entities use
        row index + RIGHT_ID_OFFSET.

        Returns
        -------
        tuple[list[Entity], list[Entity]]
            Left and right entity lists
        """
        name_col_a = _detect_name_column(self.table_a) or "id"
        name_col_b = _detect_name_column(self.table_b) or "id"
        text_cols_a = _get_text_columns(self.table_a)
        text_cols_b = _get_text_columns(self.table_b)

        left_entities: list[Entity] = []
        for i, (_idx, row) in enumerate(self.table_a.iterrows()):
            ent = _row_to_entity(row, i, "l_", name_col_a, text_cols_a)
            left_entities.append(ent)

        right_entities: list[Entity] = []
        for i, (_idx, row) in enumerate(self.table_b.iterrows()):
            ent = _row_to_entity(row, i + RIGHT_ID_OFFSET, "r_", name_col_b, text_cols_b)
            right_entities.append(ent)

        return (left_entities, right_entities)
