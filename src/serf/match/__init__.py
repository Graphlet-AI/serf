"""Match module for block-level entity resolution."""

from serf.match.few_shot import format_few_shot_examples, get_default_few_shot_examples
from serf.match.matcher import EntityMatcher
from serf.match.uuid_mapper import UUIDMapper

__all__ = [
    "EntityMatcher",
    "UUIDMapper",
    "get_default_few_shot_examples",
    "format_few_shot_examples",
]
