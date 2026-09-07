"""Integration tests for LLM-powered analyze + benchmark pipeline.

These tests require:
- GEMINI_API_KEY environment variable
- Network access to download benchmark datasets from Leipzig
- Network access to call Gemini API

They verify that `serf analyze` can generate a valid ER config from
real benchmark data and that the pipeline produces reasonable results.

Run with: uv run pytest tests/test_analyze_benchmark.py -v
"""

import os
import tempfile
from typing import Any

import pandas as pd
import pytest
import yaml

from serf.analyze.profiler import DatasetProfiler, generate_er_config
from serf.eval.benchmarks import BenchmarkDataset
from serf.pipeline import ERConfig

# Skip all tests if no API key
pytestmark = pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set",
)


@pytest.fixture(scope="module")
def dblp_acm_dataset() -> BenchmarkDataset:
    """Download and cache the DBLP-ACM benchmark dataset."""
    return BenchmarkDataset.download("dblp-acm")


@pytest.fixture(scope="module")
def dblp_acm_merged_df(dblp_acm_dataset: BenchmarkDataset) -> pd.DataFrame:
    """Create a merged DataFrame from both DBLP-ACM tables."""
    table_a = dblp_acm_dataset.table_a.copy()
    table_b = dblp_acm_dataset.table_b.copy()
    table_a["source_table"] = "dblp"
    table_b["source_table"] = "acm"
    merged = pd.concat([table_a, table_b], ignore_index=True)
    return merged


@pytest.fixture(scope="module")
def dblp_acm_profile(dblp_acm_merged_df: pd.DataFrame) -> dict[str, Any]:
    """Profile the merged DBLP-ACM dataset."""
    records = dblp_acm_merged_df.to_dict("records")
    profiler = DatasetProfiler()
    profile = profiler.profile(records)
    return {
        "profile": profile,
        "records": records,
    }


@pytest.fixture(scope="module")
def generated_config_yaml(dblp_acm_profile: dict[str, Any]) -> str:
    """Generate an ER config via LLM from the DBLP-ACM profile."""
    profile = dblp_acm_profile["profile"]
    records = dblp_acm_profile["records"]
    config_yaml = generate_er_config(profile, records[:10])
    return config_yaml


def test_profiler_produces_valid_profile(dblp_acm_profile: dict[str, Any]) -> None:
    """Test that the profiler produces a valid profile for DBLP-ACM."""
    profile = dblp_acm_profile["profile"]
    assert profile.record_count > 0
    assert len(profile.field_profiles) > 0
    assert len(profile.recommended_blocking_fields) > 0
    # Should detect 'title' as a name field
    name_fields = [fp for fp in profile.field_profiles if fp.inferred_type == "name"]
    assert len(name_fields) > 0, "Should detect at least one name-type field"


def test_llm_generates_valid_yaml(generated_config_yaml: str) -> None:
    """Test that the LLM generates valid YAML."""
    parsed = yaml.safe_load(generated_config_yaml)
    assert isinstance(parsed, dict), f"Expected dict, got {type(parsed)}"


def test_llm_config_has_name_field(generated_config_yaml: str) -> None:
    """Test that the LLM config specifies a name_field."""
    parsed = yaml.safe_load(generated_config_yaml)
    assert "name_field" in parsed, f"Missing name_field. Keys: {list(parsed.keys())}"
    assert isinstance(parsed["name_field"], str)
    assert len(parsed["name_field"]) > 0


def test_llm_config_has_text_fields(generated_config_yaml: str) -> None:
    """Test that the LLM config specifies text_fields."""
    parsed = yaml.safe_load(generated_config_yaml)
    assert "text_fields" in parsed, f"Missing text_fields. Keys: {list(parsed.keys())}"
    assert isinstance(parsed["text_fields"], list)
    assert len(parsed["text_fields"]) > 0


def test_llm_config_has_entity_type(generated_config_yaml: str) -> None:
    """Test that the LLM config specifies an entity_type."""
    parsed = yaml.safe_load(generated_config_yaml)
    assert "entity_type" in parsed, f"Missing entity_type. Keys: {list(parsed.keys())}"
    assert isinstance(parsed["entity_type"], str)


def test_llm_config_has_blocking_section(generated_config_yaml: str) -> None:
    """Test that the LLM config has blocking parameters."""
    parsed = yaml.safe_load(generated_config_yaml)
    # Blocking config can be top-level or nested
    has_blocking = "blocking" in parsed or "target_block_size" in parsed
    assert has_blocking, f"Missing blocking config. Keys: {list(parsed.keys())}"


def test_llm_config_loads_as_er_config(generated_config_yaml: str) -> None:
    """Test that the generated YAML can be loaded as an ERConfig."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        f.write(generated_config_yaml)
        f.flush()
        config_path = f.name

    try:
        er_config = ERConfig.from_yaml(config_path)
        assert er_config.name_field is not None
        assert er_config.max_iterations > 0
        assert er_config.convergence_threshold > 0
    finally:
        os.unlink(config_path)


def test_llm_config_name_field_exists_in_data(
    generated_config_yaml: str,
    dblp_acm_merged_df: pd.DataFrame,
) -> None:
    """Test that the LLM-chosen name_field actually exists in the dataset."""
    parsed = yaml.safe_load(generated_config_yaml)
    name_field = parsed.get("name_field", "")
    columns = [str(c).lower() for c in dblp_acm_merged_df.columns]
    assert name_field.lower() in columns, (
        f"name_field '{name_field}' not in columns: {list(dblp_acm_merged_df.columns)}"
    )


def test_llm_config_text_fields_exist_in_data(
    generated_config_yaml: str,
    dblp_acm_merged_df: pd.DataFrame,
) -> None:
    """Test that the LLM-chosen text_fields actually exist in the dataset."""
    parsed = yaml.safe_load(generated_config_yaml)
    text_fields = parsed.get("text_fields", [])
    columns = [str(c).lower() for c in dblp_acm_merged_df.columns]
    for field in text_fields:
        assert field.lower() in columns, (
            f"text_field '{field}' not in columns: {list(dblp_acm_merged_df.columns)}"
        )
