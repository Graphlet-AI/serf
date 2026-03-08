"""DSPy signatures for entity resolution operations.

These signatures define the input/output contracts for LLM-powered
ER operations: block matching, entity merging, edge resolution,
and dataset analysis.
"""

import dspy

from serf.dspy.types import BlockResolution, DatasetProfile, Entity


class BlockMatch(dspy.Signature):
    """Examine all entities in a block, identify duplicates, and merge them.

    For each group of matching entities:
    1. Select the entity with the LOWEST input id as the master record
    2. Collect ALL OTHER input ids into source_ids
    3. Merge all source_ids from inputs into a single output source_ids list
    4. Choose the most complete field values across all matched records
    5. Return ALL entities (merged + non-matched)
    """

    block_records: str = dspy.InputField(desc="JSON array of entity records in this block")
    schema_info: str = dspy.InputField(desc="Description of entity fields and their roles")
    few_shot_examples: str = dspy.InputField(
        desc="Examples of correct merge behavior with ID tracking"
    )
    resolution: BlockResolution = dspy.OutputField()


class EntityMerge(dspy.Signature):
    """Merge two matched entities into a single canonical record.

    The entity with the LOWEST id becomes the master. All field values
    are chosen by selecting the most complete/informative value.
    The other entity's id goes into source_ids.
    """

    entity_a: str = dspy.InputField(desc="First entity as JSON")
    entity_b: str = dspy.InputField(desc="Second entity as JSON")
    merged: Entity = dspy.OutputField()


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
    """Generate an entity resolution configuration for a dataset.

    Given a statistical profile of the dataset including field types,
    completeness, uniqueness, sample values, and record count, produce
    a YAML configuration that specifies:
    - name_field: which column contains the entity name/title
    - text_fields: which columns should be used for embedding text
    - entity_type: what kind of entities these are
    - blocking parameters (target_block_size, max_block_size)
    - matching model recommendation
    - max_iterations and convergence_threshold

    Choose the name_field as the column most likely to be a name or title.
    Choose text_fields as columns useful for distinguishing entities.
    Set target_block_size based on dataset size (smaller for small datasets).
    """

    dataset_profile: str = dspy.InputField(
        desc="JSON statistical profile of the dataset including field types, "
        "completeness, uniqueness, sample values, and record count"
    )
    sample_records: str = dspy.InputField(desc="JSON array of 5-10 sample records from the dataset")
    er_config_yaml: str = dspy.OutputField(
        desc="YAML configuration for entity resolution with keys: "
        "name_field, text_fields, entity_type, blocking (method, "
        "target_block_size, max_block_size), matching (model), "
        "max_iterations, convergence_threshold"
    )
