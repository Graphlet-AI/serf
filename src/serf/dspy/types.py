"""Pipeline types for SERF entity resolution.

Domain-agnostic Pydantic types used throughout the ER pipeline.
Domain-specific fields live in the Entity `attributes` dict.
"""

from typing import Any

from pydantic import BaseModel, Field


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

    def text_for_embedding(self) -> str:
        """Return text representation for embedding.

        Returns
        -------
        str
            Concatenation of name and description for embedding
        """
        parts = [self.name]
        if self.description:
            parts.append(self.description)
        for _key, val in self.attributes.items():
            if isinstance(val, str) and val:
                parts.append(val)
        return " ".join(parts)


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

    def text_for_embedding(self) -> str:
        """Return text optimized for bibliographic embedding.

        Returns
        -------
        str
            Title + authors + venue for embedding
        """
        parts = [self.name]
        if self.authors:
            parts.append(self.authors)
        if self.venue:
            parts.append(self.venue)
        return " ".join(parts)


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

    def text_for_embedding(self) -> str:
        """Return text optimized for product embedding.

        Returns
        -------
        str
            Name + manufacturer + category for embedding
        """
        parts = [self.name]
        if self.manufacturer:
            parts.append(self.manufacturer)
        if self.description:
            parts.append(self.description)
        if self.category:
            parts.append(self.category)
        return " ".join(parts)


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
    """

    total_blocks: int = 0
    total_entities: int = 0
    avg_block_size: float = 0.0
    max_block_size: int = 0
    singleton_blocks: int = 0
    pair_completeness: float = 0.0
    reduction_ratio: float = 0.0
