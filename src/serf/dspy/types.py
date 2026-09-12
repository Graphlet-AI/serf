"""Pipeline types for SERF entity resolution.

Domain-agnostic Pydantic types used throughout the ER pipeline.
Domain-specific fields live in the Entity `attributes` dict.
"""

import json
from typing import Any

from pydantic import BaseModel, Field, field_validator

# XML has no null literal, so a model filling in an optional tag writes a
# placeholder word or leaves the tag empty. Pydantic sees those as strings.
XML_NULL_PLACEHOLDERS = frozenset({"", "null", "none", "nil", "n/a", "undefined"})


class Entity(BaseModel):
    """Generic entity for entity resolution.

    Domain-specific fields live in `attributes`. ER metadata fields
    (id, uuid, source_ids, etc.) are fixed across all domains.

    Parameters
    ----------
    id : int
        Unique integer identifier for this entity
    uuid : Optional[str]
        UUID string, assigned during pipeline processing
    name : str
        Primary name/title of the entity
    description : str
        Text description of the entity
    entity_type : str
        Type label (e.g. "product", "publication", "company")
    attributes : dict[str, Any]
        Domain-specific fields from the source data
    source_ids : Optional[list[int]]
        IDs of entities that were merged into this one
    source_uuids : Optional[list[str]]
        UUIDs of entities that were merged into this one
    match_skip : Optional[bool]
        Whether this entity was skipped during matching
    match_skip_reason : Optional[str]
        Reason for skipping (singleton_block, error_recovery, missing_in_match_output)
    match_skip_history : Optional[list[int]]
        Iteration numbers where this entity was skipped
    """

    id: int
    uuid: str | None = None
    name: str
    description: str = ""
    entity_type: str = "entity"
    attributes: dict[str, Any] = Field(default_factory=dict)
    source_ids: list[int] | None = None
    source_uuids: list[str] | None = None
    match_skip: bool | None = None
    match_skip_reason: str | None = None
    match_skip_history: list[int] | None = None

    @field_validator(
        "uuid",
        "source_ids",
        "source_uuids",
        "match_skip",
        "match_skip_reason",
        "match_skip_history",
        mode="before",
    )
    @classmethod
    def _drop_xml_nulls(cls, value: Any) -> Any:
        """Turn XML's null placeholders into real ``None``.

        ``<match_skip>null</match_skip>`` and ``<uuid></uuid>`` are how a model
        says "no value" in XML, and Pydantic rejects both for ``bool | None``
        and friends. Every rejection sends the whole block through DSPy's JSON
        fallback, so this is the difference between one LLM call per block and
        two, and between a parsed block and a lost one.

        Parameters
        ----------
        value : Any
            Raw value from the adapter or the source data

        Returns
        -------
        Any
            ``None`` for a placeholder, a filtered list for a list of them,
            otherwise the input unchanged
        """
        if isinstance(value, str):
            return None if value.strip().lower() in XML_NULL_PLACEHOLDERS else value
        if isinstance(value, list):
            kept = [
                item
                for item in value
                if not (isinstance(item, str) and item.strip().lower() in XML_NULL_PLACEHOLDERS)
            ]
            return kept or None
        return value

    @field_validator("attributes", mode="before")
    @classmethod
    def _parse_attributes(cls, value: Any) -> Any:
        """Accept the JSON-object string an XML tag body carries.

        ``dspy.XMLAdapter`` renders a dict output field as a single
        ``<attributes>...</attributes>`` tag with no nested structure, so the
        model writes a JSON object into the tag body and the adapter hands the
        text back as ``str``. Rejecting that makes every block fall through to
        DSPy's JSON fallback, which doubles the LLM calls and loses the blocks
        whose fallback response is malformed.

        Parameters
        ----------
        value : Any
            Raw value from the adapter or the source data

        Returns
        -------
        Any
            A dict when the input was a JSON object or an empty string,
            otherwise the input unchanged so Pydantic reports the real error
        """
        if not isinstance(value, str):
            return value
        text = value.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return value
        return parsed if isinstance(parsed, dict) else value

    def text_for_embedding(self, blocking_fields: list[str] | None = None) -> str:
        """Return text for embedding-based blocking.

        By default returns ONLY the entity name. This produces tighter
        semantic clusters because name/title fields have the highest
        discriminative power for grouping similar entities. Including
        other fields (year, ID, etc.) adds noise to the embedding.

        When blocking_fields are specified (by agentic config), those
        additional attribute values are appended to the name.

        The LLM matcher sees ALL fields during matching — blocking
        only needs to group potentially similar entities together.

        Parameters
        ----------
        blocking_fields : list[str] | None
            Additional attribute fields to include in embedding text.
            If None, only the name is used.

        Returns
        -------
        str
            Text for embedding
        """
        if not blocking_fields:
            return self.name
        parts = [self.name]
        for field in blocking_fields:
            val = self.attributes.get(field)
            if val and isinstance(val, str):
                parts.append(val)
        return " ".join(parts)

    def json_for_embedding(self) -> str:
        """Return every populated field as a JSON object, field names inline.

        The alternative to name-only blocking. Naming each value lets the model
        read "brand" and "price" as different kinds of thing rather than as one
        undifferentiated string, which matters most where the name alone is
        ambiguous, as with product titles that share a manufacturer.

        Two details keep the two sides of a join comparable. The ``l_`` and
        ``r_`` prefixes that Leipzig-style benchmarks put on every column are
        stripped, because every gold pair crosses sources and a field name that
        encodes which source a record came from pushes the sides apart. Record
        identifiers are dropped for the same reason: they are drawn from
        unrelated namespaces, so they are noise at best.

        Returns
        -------
        str
            Compact JSON with keys sorted, so the text is stable across runs
        """
        record: dict[str, str] = {}
        for key, value in self.attributes.items():
            if value is None or value == "":
                continue
            field = key
            for prefix in ("l_", "r_"):
                if field.startswith(prefix):
                    field = field[len(prefix) :]
                    break
            if field == "id":
                continue
            record[field] = str(value)

        if self.name and self.name not in record.values():
            record["name"] = self.name
        if self.description and self.description not in record.values():
            record["description"] = self.description

        return json.dumps(record, ensure_ascii=False, sort_keys=True)


class Publication(Entity):
    """Publication entity for bibliographic record resolution.

    Parameters
    ----------
    title : str
        Publication title (maps to Entity.name)
    authors : str
        Author names
    venue : str
        Publication venue (journal, conference, etc.)
    year : int | None
        Publication year
    """

    entity_type: str = "publication"
    authors: str = ""
    venue: str = ""
    year: int | None = None


class Product(Entity):
    """Product entity for product matching.

    Parameters
    ----------
    manufacturer : str
        Product manufacturer or brand
    price : float | None
        Product price
    category : str
        Product category
    """

    entity_type: str = "product"
    manufacturer: str = ""
    price: float | None = None
    category: str = ""


class EntityBlock(BaseModel):
    """A block of entities for matching.

    Parameters
    ----------
    block_key : str
        Identifier for this block (e.g. FAISS cluster ID)
    block_key_type : str
        How the block was created: "semantic", "name", "custom"
    block_size : int
        Number of entities in the block
    entities : list[Entity]
        The entities in this block
    """

    block_key: str
    block_key_type: str = "semantic"
    block_size: int
    entities: list[Entity]


class MatchDecision(BaseModel):
    """A single match decision between two entities.

    Parameters
    ----------
    entity_a_id : int
        ID of the first entity
    entity_b_id : int
        ID of the second entity
    is_match : bool
        Whether the entities are a match
    confidence : float
        Confidence score between 0 and 1
    reasoning : str
        Explanation for the match decision
    """

    entity_a_id: int
    entity_b_id: int
    is_match: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


class BlockResolution(BaseModel):
    """Result of resolving all matches within a block.

    Parameters
    ----------
    block_key : str
        The block key this resolution belongs to
    matches : list[MatchDecision]
        All pairwise match decisions made
    resolved_entities : list[Entity]
        Entities after merging (merged + non-matched)
    was_resolved : bool
        Whether any merges occurred
    original_count : int
        Number of entities before resolution
    resolved_count : int
        Number of entities after resolution
    """

    block_key: str = ""
    matches: list[MatchDecision] = Field(default_factory=list)
    resolved_entities: list[Entity] = Field(default_factory=list)
    was_resolved: bool = False
    original_count: int = 0
    resolved_count: int = 0


class FieldProfile(BaseModel):
    """Profile of a single field in the dataset.

    Parameters
    ----------
    name : str
        Field name
    inferred_type : str
        Detected type: name, email, url, phone, address, identifier, date, numeric, text
    completeness : float
        Fraction of non-null values (0.0 to 1.0)
    uniqueness : float
        Fraction of unique values (0.0 to 1.0)
    sample_values : list[str]
        Example values from this field
    is_blocking_candidate : bool
        Whether this field is suitable for blocking
    is_matching_feature : bool
        Whether this field is useful for matching
    """

    name: str
    inferred_type: str = "text"
    completeness: float = 0.0
    uniqueness: float = 0.0
    sample_values: list[str] = Field(default_factory=list)
    is_blocking_candidate: bool = False
    is_matching_feature: bool = False


class DatasetProfile(BaseModel):
    """Profile of the entire input dataset.

    Parameters
    ----------
    record_count : int
        Total number of records
    field_profiles : list[FieldProfile]
        Profile for each field
    recommended_blocking_fields : list[str]
        Fields recommended for blocking
    recommended_matching_fields : list[str]
        Fields recommended for matching
    estimated_duplicate_rate : float
        Estimated fraction of duplicate records
    """

    record_count: int = 0
    field_profiles: list[FieldProfile] = Field(default_factory=list)
    recommended_blocking_fields: list[str] = Field(default_factory=list)
    recommended_matching_fields: list[str] = Field(default_factory=list)
    estimated_duplicate_rate: float = 0.0


class IterationMetrics(BaseModel):
    """Metrics for a single ER iteration.

    Parameters
    ----------
    iteration : int
        Iteration number
    input_entities : int
        Number of entities at start of iteration
    output_entities : int
        Number of entities at end of iteration
    reduction_pct : float
        Percentage reduction this iteration
    overall_reduction_pct : float
        Cumulative reduction from original baseline
    blocks_count : int
        Number of blocks created
    singleton_blocks : int
        Number of blocks with only one entity
    largest_block : int
        Size of the largest block
    """

    iteration: int = 0
    input_entities: int = 0
    output_entities: int = 0
    reduction_pct: float = 0.0
    overall_reduction_pct: float = 0.0
    blocks_count: int = 0
    singleton_blocks: int = 0
    largest_block: int = 0


class BlockingMetrics(BaseModel):
    """Metrics for the blocking phase.

    Parameters
    ----------
    total_blocks : int
        Total number of blocks created
    total_entities : int
        Total entities across all blocks
    avg_block_size : float
        Average entities per block
    max_block_size : int
        Largest block size
    singleton_blocks : int
        Blocks with one entity
    pair_completeness : float
        Fraction of true pairs retained
    reduction_ratio : float
        1 - (pairs after blocking / total possible pairs)
    blocked_pairs : int
        Distinct record pairs sharing at least one block, which is the number
        of comparisons the matcher is asked to make. Counted distinctly rather
        than summed over blocks, because a record can sit in more than one
        block and the same pair must not be billed twice.
    """

    total_blocks: int = 0
    total_entities: int = 0
    avg_block_size: float = 0.0
    max_block_size: int = 0
    singleton_blocks: int = 0
    pair_completeness: float = 0.0
    reduction_ratio: float = 0.0
    blocked_pairs: int = 0
