"""Merge module for entity merging and canonicalization."""

from serf.merge.canonical import CanonicalRecord, canonicalize, canonicalize_groups
from serf.merge.merger import EntityMerger
from serf.merge.semantics import (
    CARDINALITIES,
    CARDINALITY_LIST,
    CARDINALITY_SINGLE,
    MergePolicy,
    flatten_values,
    merge_values,
    policy_for,
    select_best_value,
)

__all__ = [
    "CARDINALITIES",
    "CARDINALITY_LIST",
    "CARDINALITY_SINGLE",
    "CanonicalRecord",
    "EntityMerger",
    "MergePolicy",
    "canonicalize",
    "canonicalize_groups",
    "flatten_values",
    "merge_values",
    "policy_for",
    "select_best_value",
]
