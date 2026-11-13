"""DSPy signatures for entity resolution operations.

These signatures define the input/output contracts for LLM-powered
ER operations: block matching, entity merging, edge resolution,
and dataset analysis.
"""

from typing import Annotated

import dspy

from serf.config import config as _config
from serf.dspy.schemas.base import (
    SingularText,
    TextList,
    field_guide,
)
from serf.dspy.types import BlockResolution, DatasetProfile, Entity

_MATCHING_MODEL = _config.get("models.llm")
_ENTITY_FIELD_GUIDE = field_guide(Entity)


class BlockMatch(dspy.Signature):
    __doc__ = f"""Examine all entities in a block, identify duplicates, and merge them.

    For each group of matching entities:
    1. Select the entity with the LOWEST input id as the master record
    2. Collect ALL OTHER input ids into source_ids
    3. Merge all source_ids from inputs into a single output source_ids list
    4. Rely on the LLM to make field-level comparisons at the time of match and merge according to each field's merge semantics:
{chr(10).join(_ENTITY_FIELD_GUIDE.splitlines()[:7])}
    5. Return ALL entities (merged + non-matched)
    """

    block_records: Annotated[
        str,
        dspy.InputField(desc="JSON array of entity records in this block"),
    ]
    schema_info: Annotated[
        str,
        dspy.InputField(
            desc="Field-level schema and merge semantics annotations for singular and list fields"
        ),
    ]
    few_shot_examples: Annotated[
        str,
        dspy.InputField(desc="Examples of correct merge behavior with ID tracking"),
    ]
    resolution: Annotated[
        BlockResolution,
        dspy.OutputField(
            desc="Block resolution containing match decisions and merged entities following field-level merge annotations"
        ),
    ]


class EntityMerge(dspy.Signature):
    __doc__ = f"""Merge two matched entities into a single canonical record.

    Rely on the LLM to compare field values across the two entities and merge each
    field according to its field-level annotation:
{chr(10).join(_ENTITY_FIELD_GUIDE.splitlines()[:7])}
    """

    entity_a: Annotated[str, dspy.InputField(desc="First entity as JSON")]
    entity_b: Annotated[str, dspy.InputField(desc="Second entity as JSON")]
    merged: Annotated[
        Entity,
        dspy.OutputField(
            desc="Canonical merged entity resolved according to the field-level merge annotations"
        ),
    ]


class SelectBestValue(dspy.Signature):
    """Select the single best canonical value for a singular entity field.

    Given candidate values collected from matched records that denote the same
    real-world entity, compare the candidate values and choose the most
    complete, accurate, and well-formed value according to the field's semantic
    type and description. Prefer full forms over abbreviations, clean text over
    corrupted text, and specific values over generic placeholders. Return one
    of the candidate values verbatim without inventing new facts.
    """

    field_name: Annotated[
        str,
        dspy.InputField(desc="Name of the singular entity field being resolved"),
    ]
    field_type: Annotated[
        str,
        dspy.InputField(
            desc="Semantic type of the field (name, address, url, email, phone, identifier, date, numeric, text)"
        ),
    ]
    field_description: Annotated[
        str,
        dspy.InputField(
            desc="Field-level merge annotation describing what the field means and how to select its best value"
        ),
    ]
    candidate_values: Annotated[
        list[str],
        dspy.InputField(desc="Distinct candidate values from the matched records"),
    ]
    best_value: SingularText


class DeduplicateListValues(dspy.Signature):
    """Deduplicate candidate values of a list field into a list of individual values.

    Given candidate values collected across matched records that denote the
    same real-world entity, compare the values at the field level according to
    the field's semantic type and merge annotation. Collapse spelling,
    formatting, or abbreviation variants that refer to the same individual
    value (for example, the same physical address or the same person's name)
    into the single most complete form, while keeping genuinely distinct values
    (such as different addresses or different names) as separate items in the
    returned list.
    """

    field_name: Annotated[
        str,
        dspy.InputField(desc="Name of the list entity field being deduplicated"),
    ]
    field_type: Annotated[
        str,
        dspy.InputField(
            desc="Semantic type of the field (name, address, url, email, phone, identifier, date, numeric, text)"
        ),
    ]
    field_description: Annotated[
        str,
        dspy.InputField(
            desc="Field-level merge annotation describing how values of this list field should be compared and deduplicated"
        ),
    ]
    candidate_values: Annotated[
        list[str],
        dspy.InputField(desc="Candidate values collected across the matched records"),
    ]
    deduplicated_values: TextList


class EdgeResolve(dspy.Signature):
    """Resolve duplicate edges between the same entity pairs.

    Given a block of edges that share the same source and destination
    entities, determine which edges are duplicates and merge them.
    Preserve distinct relationships and merge same-deal relationships.
    """

    edge_block: str = dspy.InputField(desc="JSON array of edges between the same entity pair")
    resolved_edges: str = dspy.OutputField(desc="JSON array of deduplicated/merged edges")


class AnalyzeDataset(dspy.Signature):
    """Analyze a dataset and recommend an entity resolution strategy.

    Examine the dataset summary including field statistics, sample values,
    and schema information to recommend blocking fields, matching features,
    and estimate the duplicate rate.
    """

    dataset_summary: str = dspy.InputField(
        desc="Summary statistics and sample values from the dataset"
    )
    profile: DatasetProfile = dspy.OutputField()


class GenerateERConfig(dspy.Signature):
    __doc__ = f"""Generate an entity resolution configuration for a dataset.

    Given a statistical profile of the dataset including field types,
    completeness, uniqueness, sample values, and record count, produce
    a YAML configuration that specifies:
    - name_field: which column contains the entity name/title (this is the
      PRIMARY field used for embedding-based blocking)
    - text_fields: which text columns are useful for matching (the LLM sees
      these during matching, but blocking uses ONLY name_field by default)
    - blocking_fields: optional list of additional fields to embed for blocking
      beyond name_field. Usually empty — name-only blocking gives the tightest
      clusters. Only add fields that directly help distinguish entity identity
      (e.g. manufacturer for products). Do NOT include year, ID, venue, or
      non-semantic fields.
    - entity_type: what kind of entities these are (e.g. "Publication",
      "Product", "Company", "Person")
    - blocking parameters:
      - target_block_size: aim for 30 entities per block
      - max_block_size: hard cap at 100 entities per block
    - matching model: use "{_MATCHING_MODEL}"
    - max_iterations: at most 5 iterations
    - convergence_threshold: a SMALL number like 0.01 to 0.05, representing
      the minimum fraction of entities reduced per round before stopping.
      For example 0.01 means stop when less than 1% of entities are merged
      in a round. Do NOT set this to a high number like 0.99.
    """

    dataset_profile: str = dspy.InputField(
        desc="JSON statistical profile of the dataset including field types, "
        "completeness, uniqueness, sample values, and record count"
    )
    sample_records: str = dspy.InputField(desc="JSON array of 5-10 sample records from the dataset")
    er_config_yaml: str = dspy.OutputField(
        desc="YAML configuration for entity resolution. Required keys: "
        "name_field (str), text_fields (list of str), "
        "blocking_fields (list of str, usually empty), entity_type (str), "
        "blocking: {method: semantic, target_block_size: 30, max_block_size: 100}, "
        f"matching: {{model: {_MATCHING_MODEL}}}, "
        "max_iterations (int, at most 5), "
        "convergence_threshold (float, small number like 0.01-0.05)"
    )
