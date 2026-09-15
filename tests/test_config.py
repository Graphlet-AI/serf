"""Tests for configuration access, ported from Abzu's suite.

SERF's ``Config`` is Abzu's, so these cover the same ground: dotted-path
lookups, ``${...}`` interpolation, path coercion, reload, and the error a
missing key raises. Abzu's own suite asserts that a missing key returns None,
which its implementation has never done; that assertion is corrected here.

The second half covers the keys added for the resolved record format, because
the point of the port is that an arbitrary key can be read by path.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml

from serf.config import Config
from serf.config import config as shipped_config

SAMPLE = """
base_dir: "data"

test:
  value: 42
  nested:
    value: "nested value"
  flag: true
  ratio: 0.75

paths:
  output: "${base_dir}/output"
  nested: "${paths.output}/nested"
  deep: "${paths.nested}/deep"
  list:
    - "${base_dir}/one"
    - "${base_dir}/two"

process:
  input: "${paths.output}/in.parquet"
  output: "${paths.output}/out.parquet"
  template: "${base_dir}/iteration_{iteration}/blocks"
"""


@pytest.fixture
def sample_config_file(tmp_path: Path) -> Iterator[str]:
    """Write a sample config file and yield its path."""
    path = tmp_path / "config.yml"
    path.write_text(SAMPLE, encoding="utf-8")
    yield str(path)


def test_config_loads_a_file(sample_config_file: str) -> None:
    config = Config(sample_config_file)

    assert config.config_file == sample_config_file
    assert isinstance(config._config, dict)


def test_get_reads_top_level_and_nested_values(sample_config_file: str) -> None:
    config = Config(sample_config_file)

    assert config.get("base_dir") == "data"
    assert config.get("test.value") == 42
    assert config.get("test.nested.value") == "nested value"
    assert config.get("test.flag") is True
    assert config.get("test.ratio") == 0.75


def test_get_returns_the_default_for_a_missing_key(sample_config_file: str) -> None:
    config = Config(sample_config_file)

    assert config.get("test.absent", "fallback") == "fallback"
    assert config.get("no.such.path", 7) == 7


def test_a_missing_key_without_a_default_raises(sample_config_file: str) -> None:
    """Silently returning None would let a typo reach the pipeline as a value."""
    config = Config(sample_config_file)

    with pytest.raises(KeyError, match="not found"):
        config.get("nonexistent_key")


def test_the_error_names_the_keys_that_do_exist(sample_config_file: str) -> None:
    config = Config(sample_config_file)

    with pytest.raises(KeyError, match="Available keys"):
        config.get("test.absent")


def test_traversing_through_a_non_mapping_raises(sample_config_file: str) -> None:
    config = Config(sample_config_file)

    with pytest.raises(KeyError, match="is not a dictionary"):
        config.get("test.value.deeper")


def test_interpolation_resolves_one_reference(sample_config_file: str) -> None:
    config = Config(sample_config_file)

    assert config.get("paths.output") == "data/output"


def test_interpolation_resolves_a_chain(sample_config_file: str) -> None:
    config = Config(sample_config_file)

    assert config.get("paths.nested") == "data/output/nested"
    assert config.get("paths.deep") == "data/output/nested/deep"


def test_interpolation_applies_to_every_item_of_a_string_list(sample_config_file: str) -> None:
    config = Config(sample_config_file)

    assert config.get("paths.list") == ["data/one", "data/two"]


def test_python_format_placeholders_are_left_for_the_caller(sample_config_file: str) -> None:
    """`{iteration}` is expanded by callers after get(), not by Config."""
    config = Config(sample_config_file)

    template = config.get("process.template")
    assert template == "data/iteration_{iteration}/blocks"
    assert template.format(iteration=3) == "data/iteration_3/blocks"


def test_expand_variables_resolves_a_string_the_caller_supplies(sample_config_file: str) -> None:
    config = Config(sample_config_file)

    assert config.expand_variables("${base_dir}/extra") == "data/extra"


def test_get_path_returns_a_path(sample_config_file: str) -> None:
    config = Config(sample_config_file)

    assert config.get_path("paths.output") == Path("data/output")


def test_get_path_returns_a_list_of_paths(sample_config_file: str) -> None:
    config = Config(sample_config_file)

    assert config.get_path("paths.list") == [Path("data/one"), Path("data/two")]


def test_get_path_on_a_missing_key_raises_like_get_does(sample_config_file: str) -> None:
    config = Config(sample_config_file)

    with pytest.raises(KeyError, match="not found"):
        config.get_path("paths.absent")


def test_get_path_on_a_key_whose_value_is_null_raises(tmp_path: Path) -> None:
    """The only way to reach get_path's own error: the key exists and says nothing."""
    path = tmp_path / "config.yml"
    path.write_text("paths:\n  output: null\n", encoding="utf-8")
    config = Config(str(path))

    with pytest.raises(ValueError, match="Configuration value not found"):
        config.get_path("paths.output")


def test_reload_picks_up_a_changed_file(sample_config_file: str) -> None:
    config = Config(sample_config_file)
    assert config.get("test.value") == 42

    with open(sample_config_file, "a", encoding="utf-8") as handle:
        handle.write("\nlate_addition: 'present'\n")
    config.reload()

    assert config.get("late_addition") == "present"


def test_a_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        Config(str(tmp_path / "absent.yml"))


def test_invalid_yaml_raises(tmp_path: Path) -> None:
    path = tmp_path / "config.yml"
    path.write_text("key: [unclosed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Error parsing configuration file"):
        Config(str(path))


def test_the_shipped_config_answers_an_arbitrary_dotted_path() -> None:
    """The whole point of the port: any key is readable by its path."""
    assert shipped_config.get("matcher.similarity.min_threshold") == 0.8
    assert shipped_config.get("matcher.output") == "partition"
    assert shipped_config.get("merge.semantics.name.dedupe") == "fuzzy"
    assert shipped_config.get("merge.semantics.text.dedupe") == "none"
    assert shipped_config.get("merge.conservation.coverage_threshold") == 0.9999


def test_the_shipped_config_is_the_repository_config() -> None:
    assert Path(shipped_config.config_file).name == "config.yml"
    assert yaml.safe_load(Path(shipped_config.config_file).read_text(encoding="utf-8"))


def test_every_merge_semantic_names_a_strategy_the_code_implements() -> None:
    """A typo here would silently fall back to exact deduplication at merge time."""
    from serf.merge.semantics import DEDUPE_STRATEGIES

    semantics = shipped_config.get("merge.semantics", {})
    assert semantics
    for field_type, settings in semantics.items():
        assert settings["dedupe"] in DEDUPE_STRATEGIES, field_type
