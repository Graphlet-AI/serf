"""Dataset profiling for entity resolution."""

import json
import os
from typing import Any

import dspy
import yaml

from serf.analyze.field_detection import detect_field_type
from serf.dspy.baml_adapter import BAMLAdapter
from serf.dspy.signatures import GenerateERConfig
from serf.dspy.types import DatasetProfile, FieldProfile
from serf.logs import get_logger

logger = get_logger(__name__)


class DatasetProfiler:
    """Profile a dataset for entity resolution."""

    def profile(self, records: list[dict[str, Any]]) -> DatasetProfile:
        """Profile a list of records (dicts).

        For each field:
        - Compute completeness (fraction of non-null values)
        - Compute uniqueness (fraction of unique values among non-null)
        - Sample up to 5 values
        - Detect field type using field_detection module
        - Determine if it's a blocking candidate (name/identifier type with high completeness)
        - Determine if it's a matching feature (any text/name/identifier type)

        Recommended blocking fields: fields that are blocking candidates
        Recommended matching fields: fields that are matching features
        Estimated duplicate rate: rough heuristic based on name field uniqueness

        Parameters
        ----------
        records : list[dict[str, Any]]
            List of record dictionaries to profile

        Returns
        -------
        DatasetProfile
            Profile with field stats, recommended fields, and duplicate rate estimate
        """
        if not records:
            return DatasetProfile(
                record_count=0,
                field_profiles=[],
                recommended_blocking_fields=[],
                recommended_matching_fields=[],
                estimated_duplicate_rate=0.0,
            )

        # Collect all field names from records
        all_keys: set[str] = set()
        for rec in records:
            all_keys.update(rec.keys())

        field_profiles: list[FieldProfile] = []
        for field_name in sorted(all_keys):
            values = [rec.get(field_name) for rec in records]
            non_null = [v for v in values if v is not None and v != ""]
            total = len(values)
            completeness = len(non_null) / total if total > 0 else 0.0

            unique_vals = list(set(str(v) for v in non_null))
            unique_count = len(unique_vals)
            uniqueness = unique_count / len(non_null) if non_null else 0.0

            sample_values = [str(v) for v in unique_vals[:5]]

            inferred_type = detect_field_type(field_name, non_null)

            blocking_types = {"name", "identifier"}
            matching_types = {"name", "email", "url", "identifier", "text"}
            is_blocking_candidate = inferred_type in blocking_types and completeness >= 0.5
            is_matching_feature = inferred_type in matching_types

            field_profiles.append(
                FieldProfile(
                    name=field_name,
                    inferred_type=inferred_type,
                    completeness=round(completeness, 4),
                    uniqueness=round(uniqueness, 4),
                    sample_values=sample_values,
                    is_blocking_candidate=is_blocking_candidate,
                    is_matching_feature=is_matching_feature,
                )
            )

        recommended_blocking = [fp.name for fp in field_profiles if fp.is_blocking_candidate]
        recommended_matching = [fp.name for fp in field_profiles if fp.is_matching_feature]

        # Estimate duplicate rate from name-like field uniqueness
        name_profiles = [fp for fp in field_profiles if fp.inferred_type == "name"]
        if name_profiles:
            avg_name_uniqueness = sum(fp.uniqueness for fp in name_profiles) / len(name_profiles)
            estimated_duplicate_rate = round(1.0 - avg_name_uniqueness, 4)
            estimated_duplicate_rate = max(0.0, min(1.0, estimated_duplicate_rate))
        else:
            estimated_duplicate_rate = 0.0

        return DatasetProfile(
            record_count=len(records),
            field_profiles=field_profiles,
            recommended_blocking_fields=recommended_blocking,
            recommended_matching_fields=recommended_matching,
            estimated_duplicate_rate=estimated_duplicate_rate,
        )


def generate_er_config(
    profile: DatasetProfile,
    sample_records: list[dict[str, Any]],
    model: str | None = None,
) -> str:
    """Use an LLM to generate an ER config YAML from a dataset profile.

    Parameters
    ----------
    profile : DatasetProfile
        Statistical profile of the dataset
    sample_records : list[dict[str, Any]]
        Sample records from the dataset (5-10 records)
    model : str
        LLM model to use for config generation

    Returns
    -------
    str
        YAML string with the recommended ER configuration
    """
    from serf.config import config as serf_config

    effective_model = model or serf_config.get("models.analyze_llm")
    api_key = os.environ.get("GEMINI_API_KEY", "")
    lm = dspy.LM(effective_model, api_key=api_key)
    logger.info(f"Using LLM model: {effective_model}")

    predictor = dspy.ChainOfThought(GenerateERConfig)

    profile_json = profile.model_dump_json(indent=2)
    samples_json = json.dumps(sample_records[:10], indent=2, default=str)

    logger.info("Generating ER config with LLM...")
    with dspy.context(lm=lm, adapter=BAMLAdapter()):
        result = predictor(
            dataset_profile=profile_json,
            sample_records=samples_json,
        )

    config_yaml: str = result.er_config_yaml
    # Strip markdown code fences if the LLM wrapped it
    if config_yaml.startswith("```"):
        lines = config_yaml.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        config_yaml = "\n".join(lines)

    # Validate and enforce safe bounds on LLM-generated config
    config_yaml = _sanitize_er_config(config_yaml.strip())
    return config_yaml


# Safe upper bounds for LLM-generated config values
_MAX_ITERATIONS = 10
_MAX_BLOCK_SIZE = 500
_MAX_TARGET_BLOCK_SIZE = 200


def _sanitize_er_config(config_yaml: str) -> str:
    """Validate and enforce safe bounds on LLM-generated ER config.

    Prevents indirect prompt injection from producing dangerous configs
    (e.g. extremely high iteration counts or block sizes).

    Parameters
    ----------
    config_yaml : str
        Raw YAML config from LLM

    Returns
    -------
    str
        Sanitized YAML config with safe bounds enforced
    """
    parsed = yaml.safe_load(config_yaml)
    if not isinstance(parsed, dict):
        logger.warning("LLM generated non-dict config, returning empty config")
        return "name_field: name\ntext_fields: []\nentity_type: entity\n"

    # Enforce safe upper bounds
    if parsed.get("max_iterations", 0) > _MAX_ITERATIONS:
        logger.warning(
            f"Clamping max_iterations from {parsed['max_iterations']} to {_MAX_ITERATIONS}"
        )
        parsed["max_iterations"] = _MAX_ITERATIONS

    blocking = parsed.get("blocking", {})
    if isinstance(blocking, dict):
        if blocking.get("max_block_size", 0) > _MAX_BLOCK_SIZE:
            blocking["max_block_size"] = _MAX_BLOCK_SIZE
        if blocking.get("target_block_size", 0) > _MAX_TARGET_BLOCK_SIZE:
            blocking["target_block_size"] = _MAX_TARGET_BLOCK_SIZE
        parsed["blocking"] = blocking

    # Ensure convergence_threshold is reasonable
    ct = parsed.get("convergence_threshold", 0.01)
    if not isinstance(ct, (int, float)) or ct > 0.5 or ct < 0.001:
        parsed["convergence_threshold"] = 0.01

    result: str = yaml.dump(parsed, default_flow_style=False).strip()
    return result
