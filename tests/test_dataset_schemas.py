"""Tests for the per-dataset typed record schemas."""

import io
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from serf.dspy.schemas import (
    AbtBuyCandidate,
    AbtProduct,
    AcmPublication,
    AmazonElectronicsProduct,
    AmazonGoogleCandidate,
    AmazonSoftwareProduct,
    BuyProduct,
    DblpAcmCandidate,
    DblpPublication,
    DblpScholarCandidate,
    DblpScholarPublication,
    EntitySide,
    GoogleScholarPublication,
    GoogleSoftwareProduct,
    WalmartAmazonCandidate,
    WalmartProduct,
    entity_side,
    strip_side_prefix,
)
from serf.dspy.schemas.base import SIDE_LEFT, SIDE_RIGHT, SIDE_UNKNOWN, field_guide
from serf.dspy.types import Entity

# Rows copied verbatim from the released benchmark CSVs. The archives live under
# data/ which is not in version control, so the real rows are inlined here and
# the archive-backed test below is skipped when the download is absent.
DBLP_ACM_DBLP_ROW = {
    "id": "journals/sigmod/Mackay99",
    "title": (
        "Semantic Integration of Environmental Models for Application to "
        "Global Information Systems and Decision-Making"
    ),
    "authors": "D. Scott Mackay",
    "venue": "SIGMOD Record",
    "year": "1999",
}
DBLP_ACM_ACM_ROW = {
    "id": "304586",
    "title": "The WASA2 object-oriented workflow management system",
    "authors": "Gottfried Vossen, Mathias Weske",
    "venue": "International Conference on Management of Data",
    "year": "1999",
}
DBLP_SCHOLAR_DBLP_ROW = {
    "id": "conf/vldb/RusinkiewiczKTWM95",
    "title": "Towards a Cooperative Transaction Model - The Cooperative Activity Model",
    "authors": "M Rusinkiewicz, W Klas, T Tesch, J Wäsch, P Muth",
    "venue": "VLDB",
    "year": "1995",
}
DBLP_SCHOLAR_SCHOLAR_ROW = {
    "id": "aKcZKwvwbQwJ",
    "title": "11578 Sorrento Valley Road",
    "authors": "QD Inc",
    "venue": "San Diego,",
    "year": "",
}
ABT_ROW = {
    "id": "552",
    "name": "Sony Turntable - PSLX350H",
    "description": (
        "Sony Turntable - PSLX350H/ Belt Drive System/ 33-1/3 and 45 RPM Speeds/ "
        "Servo Speed Control/ Supplied Moving Magnet Phono Cartridge"
    ),
    "price": "",
}
ABT_PRICED_ROW = {
    "id": "580",
    "name": "Bose Acoustimass 5 Series III Speaker System - AM53BK",
    "description": "Bose Acoustimass 5 Series III Speaker System - AM53BK/ 2 Dual Cube Speakers",
    "price": "$399.00",
}
BUY_ROW = {
    "id": "10011646",
    "name": "Linksys EtherFast EZXS88W Ethernet Switch - EZXS88W",
    "description": "Linksys EtherFast 8-Port 10/100 Switch (New/Workgroup)",
    "manufacturer": "LINKSYS",
    "price": "",
}
AMAZON_GOOGLE_AMAZON_ROW = {
    "id": "0",
    "title": "clickart 950 000 premier image pack ( dvd-rom )",
    "manufacturer": "broderbund",
    "price": "",
}
AMAZON_GOOGLE_GOOGLE_ROW = {
    "id": "0",
    "title": "learning quickbooks 2007",
    "manufacturer": "intuit",
    "price": "38.99",
}
WALMART_ROW = {
    "id": "0",
    "title": "draper infrared remote transmitter",
    "category": "electronics - general",
    "brand": "draper",
    "modelno": "121066",
    "price": "58.45",
}
WALMART_AMAZON_AMAZON_ROW = {
    "id": "0",
    "title": "koss eq50 3-band stereo equalizer",
    "category": "headphone accessories",
    "brand": "koss",
    "modelno": "152132",
    "price": "12.65",
}

# Scholar.csv opens with a UTF-8 byte order mark, so its first header cell reads
# as '\ufeff"id"' to a naive CSV reader.
SCHOLAR_HEADER_BYTES = (
    b'\xef\xbb\xbf"id","title","authors","venue","year"\r\n"aKcZKwvwbQwJ","x","y","z",""\r\n'
)

SIDE_MODELS: list[tuple[type[EntitySide], dict[str, str]]] = [
    (DblpPublication, DBLP_ACM_DBLP_ROW),
    (AcmPublication, DBLP_ACM_ACM_ROW),
    (DblpScholarPublication, DBLP_SCHOLAR_DBLP_ROW),
    (GoogleScholarPublication, DBLP_SCHOLAR_SCHOLAR_ROW),
    (AbtProduct, ABT_ROW),
    (BuyProduct, BUY_ROW),
    (AmazonSoftwareProduct, AMAZON_GOOGLE_AMAZON_ROW),
    (GoogleSoftwareProduct, AMAZON_GOOGLE_GOOGLE_ROW),
    (WalmartProduct, WALMART_ROW),
    (AmazonElectronicsProduct, WALMART_AMAZON_AMAZON_ROW),
]

BENCHMARK_ARCHIVES: dict[str, tuple[str, str]] = {
    "dblp-acm": ("DBLP2.csv", "ACM.csv"),
    "dblp-scholar": ("DBLP1.csv", "Scholar.csv"),
    "abt-buy": ("Abt.csv", "Buy.csv"),
    "amazon-google": ("tableA.csv", "tableB.csv"),
    "walmart-amazon": ("exp_data/tableA.csv", "exp_data/tableB.csv"),
}

ARCHIVE_SIDE_MODELS: dict[str, tuple[type[EntitySide], type[EntitySide]]] = {
    "dblp-acm": (DblpPublication, AcmPublication),
    "dblp-scholar": (DblpScholarPublication, GoogleScholarPublication),
    "abt-buy": (AbtProduct, BuyProduct),
    "amazon-google": (AmazonSoftwareProduct, GoogleSoftwareProduct),
    "walmart-amazon": (WalmartProduct, AmazonElectronicsProduct),
}


def _entity(row: dict[str, str], entity_id: int, prefix: str) -> Entity:
    """Build an Entity the way BenchmarkDataset.to_entities does.

    Parameters
    ----------
    row : dict[str, str]
        Source CSV row
    entity_id : int
        Entity id to assign
    prefix : str
        Attribute prefix, ``l_`` or ``r_``

    Returns
    -------
    Entity
        Entity with prefixed attributes
    """
    attributes = {f"{prefix}{key}": value for key, value in row.items() if value != ""}
    name_key = "title" if "title" in row else "name"
    return Entity(
        id=entity_id,
        name=row.get(name_key, ""),
        attributes=dict(attributes),
    )


def _archive_path(dataset: str) -> Path:
    """Return the downloaded archive path for a dataset.

    Parameters
    ----------
    dataset : str
        Benchmark dataset name

    Returns
    -------
    Path
        Path to the cached data.zip
    """
    return Path("data/benchmarks/full_baseline") / dataset / "data.zip"


@pytest.mark.parametrize(("model", "row"), SIDE_MODELS)
def test_side_model_parses_real_source_row(model: type[EntitySide], row: dict[str, str]) -> None:
    """Every side model validates a real row from its own source CSV."""
    record = model.model_validate({"record_id": 7, **row, "source_id": row["id"]})

    assert record.record_id == 7
    assert record.source_id == row["id"]
    assert record.source_name


@pytest.mark.parametrize(("model", "row"), SIDE_MODELS)
def test_side_model_fields_match_source_columns(
    model: type[EntitySide], row: dict[str, str]
) -> None:
    """Side model fields are exactly the source columns other than id."""
    assert model.source_columns() == [key for key in row if key != "id"]


@pytest.mark.parametrize(("model", "row"), SIDE_MODELS)
def test_side_model_from_entity_round_trips_columns(
    model: type[EntitySide], row: dict[str, str]
) -> None:
    """from_entity strips the l_/r_ prefix and recovers the source values."""
    entity = _entity(row, 3, "l_")
    record = model.from_entity(entity)

    assert record.record_id == 3
    assert record.source_id == row["id"]
    for column in model.source_columns():
        value = getattr(record, column)
        if row[column] == "":
            assert value in (None, "")
        elif isinstance(value, str):
            assert value == row[column]
        else:
            assert value == pytest.approx(float(row[column].lstrip("$")))


def test_abt_and_buy_schemas_are_asymmetric() -> None:
    """Buy.com publishes a manufacturer column and Abt.com does not."""
    assert "manufacturer" in BuyProduct.source_columns()
    assert "manufacturer" not in AbtProduct.source_columns()
    assert AbtProduct.source_columns() == ["name", "description", "price"]
    assert BuyProduct.source_columns() == ["name", "description", "manufacturer", "price"]


def test_abt_price_currency_symbol_is_parsed() -> None:
    """An Abt price stored as '$399.00' becomes a float."""
    record = AbtProduct.from_entity(_entity(ABT_PRICED_ROW, 1, "l_"))

    assert record.price == pytest.approx(399.0)


def test_blank_scholar_year_becomes_none() -> None:
    """Scholar rows frequently omit the year, which must not fail validation."""
    record = GoogleScholarPublication.from_entity(_entity(DBLP_SCHOLAR_SCHOLAR_ROW, 2, "r_"))

    assert record.year is None
    assert record.title == "11578 Sorrento Valley Road"


def test_unparseable_numeric_becomes_none() -> None:
    """Dirty numeric values are treated as missing rather than as errors."""
    record = WalmartProduct.model_validate(
        {"record_id": 1, "title": "x", "price": "call for price", "modelno": "abc"}
    )

    assert record.price is None
    assert record.modelno == "abc"


def test_scholar_header_bom_is_stripped_by_the_csv_loader() -> None:
    """The BOM on Scholar.csv's id header must not survive into a column name."""
    raw = SCHOLAR_HEADER_BYTES.decode("utf-8")

    assert raw.startswith("\ufeff")
    frame = pd.read_csv(io.StringIO(raw))
    assert list(frame.columns) == ["id", "title", "authors", "venue", "year"]


def test_scholar_side_model_handles_bom_prefixed_id_column() -> None:
    """A Scholar row still parses when its id column kept the BOM prefix."""
    entity = Entity(
        id=100000,
        name="11578 Sorrento Valley Road",
        attributes={'r_\ufeff"id"': "aKcZKwvwbQwJ", "r_title": "11578 Sorrento Valley Road"},
    )
    record = GoogleScholarPublication.from_entity(entity)

    assert record.record_id == 100000
    assert record.title == "11578 Sorrento Valley Road"


def test_strip_side_prefix_removes_both_prefixes() -> None:
    """strip_side_prefix drops l_ and r_ and leaves other keys alone."""
    assert strip_side_prefix({"l_title": "a", "r_price": "1", "other": "z"}) == {
        "title": "a",
        "price": "1",
        "other": "z",
    }


def test_entity_side_detects_the_source() -> None:
    """entity_side reads the side off the attribute prefix."""
    assert entity_side(_entity(WALMART_ROW, 0, "l_")) == SIDE_LEFT
    assert entity_side(_entity(WALMART_AMAZON_AMAZON_ROW, 1, "r_")) == SIDE_RIGHT
    assert entity_side(Entity(id=1, name="x")) == SIDE_UNKNOWN


CANDIDATE_CASES: list[tuple[type[Any], dict[str, str], dict[str, str], str]] = [
    (DblpAcmCandidate, DBLP_ACM_DBLP_ROW, DBLP_ACM_ACM_ROW, "dblp-acm"),
    (DblpScholarCandidate, DBLP_SCHOLAR_DBLP_ROW, DBLP_SCHOLAR_SCHOLAR_ROW, "dblp-scholar"),
    (AbtBuyCandidate, ABT_ROW, BUY_ROW, "abt-buy"),
    (
        AmazonGoogleCandidate,
        AMAZON_GOOGLE_AMAZON_ROW,
        AMAZON_GOOGLE_GOOGLE_ROW,
        "amazon-google",
    ),
    (WalmartAmazonCandidate, WALMART_ROW, WALMART_AMAZON_AMAZON_ROW, "walmart-amazon"),
]


@pytest.mark.parametrize(("candidate_type", "left_row", "right_row", "dataset"), CANDIDATE_CASES)
def test_candidate_round_trips_through_json(
    candidate_type: type[Any],
    left_row: dict[str, str],
    right_row: dict[str, str],
    dataset: str,
) -> None:
    """Candidates survive a dump/validate round trip with their typed sides."""
    left_type = candidate_type.model_fields["left"].annotation
    right_type = candidate_type.model_fields["right"].annotation
    candidate = candidate_type(
        left=left_type.from_entity(_entity(left_row, 0, "l_")),
        right=right_type.from_entity(_entity(right_row, 1, "r_")),
        is_match=True,
        confidence=0.9,
        justification=f"same entity in {dataset}",
    )

    restored = candidate_type.model_validate(candidate.model_dump(mode="json"))

    assert restored == candidate
    assert isinstance(restored.left, left_type)
    assert isinstance(restored.right, right_type)
    assert restored.left.record_id == 0
    assert restored.right.record_id == 1


def test_candidate_confidence_is_bounded() -> None:
    """Confidence outside [0, 1] is rejected."""
    with pytest.raises(ValueError):
        DblpAcmCandidate(
            left=DblpPublication(record_id=0),
            right=AcmPublication(record_id=1),
            is_match=True,
            confidence=1.5,
        )


def test_field_guide_renders_every_field_description() -> None:
    """field_guide surfaces the descriptions DSPy would otherwise drop."""
    guide = field_guide(WalmartProduct)

    for name in WalmartProduct.model_fields:
        assert f"- {name}:" in guide
    assert "model number" in guide


@pytest.mark.parametrize("dataset", sorted(BENCHMARK_ARCHIVES))
def test_side_models_parse_rows_from_the_downloaded_archive(dataset: str) -> None:
    """Both side models parse real rows straight out of the released archive."""
    archive = _archive_path(dataset)
    if not archive.exists():
        pytest.skip(f"{archive} not downloaded")

    left_name, right_name = BENCHMARK_ARCHIVES[dataset]
    left_type, right_type = ARCHIVE_SIDE_MODELS[dataset]
    with zipfile.ZipFile(archive) as archive_file:
        for member, model in ((left_name, left_type), (right_name, right_type)):
            raw = archive_file.read(member)
            for encoding in ("utf-8", "latin-1"):
                try:
                    text = raw.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
            frame = pd.read_csv(io.StringIO(text))
            assert model.source_columns() == [c for c in frame.columns if c != "id"]
            row = frame.iloc[0]
            values = {c: ("" if pd.isna(row[c]) else str(row[c])) for c in frame.columns}
            record = model.model_validate({"record_id": 0, "source_id": values["id"], **values})
            assert record.source_id == values["id"]
