"""Turn what the model returned into a partition that loses nothing.

The matcher asks for a partition of the block, and a partition is a strong
claim: every record in exactly one group. A model will sometimes fail that
claim - it forgets a record, names one twice, or invents an id - and each
failure has to be repaired here rather than carried forward, because a record
that falls out of the cover is a record deleted from the dataset.

Abzu recovers dropped lineage in layers, and this is the same idea applied one
level earlier. Abzu detects a missing uuid after the LLM has already returned
its merged records and adds the input back as a skipped singleton. Doing it on
the partition means the repair happens before anything has been merged, so the
recovered record is an ordinary unmatched record rather than a patch.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field

from serf.logs import get_logger

logger = get_logger(__name__)


@dataclass
class PartitionOutcome:
    """A repaired partition and an account of what had to be repaired.

    Parameters
    ----------
    groups : list[list[int]]
        Partition of the block: every known record id in exactly one group,
        each group sorted, groups ordered by their smallest id
    recovered_ids : list[int]
        Ids the model left out of every group, put back as groups of one
    duplicate_ids : list[int]
        Ids the model named in more than one group. The first mention wins.
    unknown_ids : list[int]
        Ids the model returned that were never in the block
    """

    groups: list[list[int]] = field(default_factory=list)
    recovered_ids: list[int] = field(default_factory=list)
    duplicate_ids: list[int] = field(default_factory=list)
    unknown_ids: list[int] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        """Whether the model returned a valid partition with no repair needed.

        Returns
        -------
        bool
            True when nothing was recovered, deduplicated or discarded
        """
        return not (self.recovered_ids or self.duplicate_ids or self.unknown_ids)

    @property
    def matched_groups(self) -> list[list[int]]:
        """Return only the groups that actually merge something.

        Returns
        -------
        list[list[int]]
            Groups holding more than one record
        """
        return [group for group in self.groups if len(group) > 1]

    def covered(self) -> set[int]:
        """Return every record id the partition accounts for.

        Returns
        -------
        set[int]
            Union of the groups
        """
        return {record_id for group in self.groups for record_id in group}

    def pairs(self) -> set[tuple[int, int]]:
        """Return every pair of records the partition claims are the same.

        Grouping a, b and c asserts all three pairs, not the two the model
        happened to mention, so the pairs come from the groups rather than
        from anything the model wrote down.

        Returns
        -------
        set[tuple[int, int]]
            Within-group pairs, smaller id first
        """
        return {
            (min(left, right), max(left, right))
            for group in self.groups
            for index, left in enumerate(group)
            for right in group[index + 1 :]
        }


def build_partition(
    raw_groups: Iterable[Iterable[int]],
    known_ids: Iterable[int],
    block_key: str = "",
) -> PartitionOutcome:
    """Repair whatever the model returned into a partition of the block.

    Three things can be wrong with the model's answer, and all three are
    repaired rather than reported: an id that was never in the block is
    discarded, an id named twice is kept only where it was first named, and an
    id named nowhere is put back as a group of its own.

    Parameters
    ----------
    raw_groups : Iterable[Iterable[int]]
        Groups of record ids as returned by the model
    known_ids : Iterable[int]
        Every record id that was actually in the block
    block_key : str
        Block identifier, for logging

    Returns
    -------
    PartitionOutcome
        A partition covering exactly ``known_ids``, plus what was repaired
    """
    known = {int(value) for value in known_ids}
    assigned: set[int] = set()
    groups: list[list[int]] = []
    duplicate_ids: list[int] = []
    unknown_ids: list[int] = []

    for raw_group in raw_groups:
        members: list[int] = []
        for value in raw_group:
            record_id = int(value)
            if record_id not in known:
                unknown_ids.append(record_id)
                continue
            if record_id in assigned:
                duplicate_ids.append(record_id)
                continue
            assigned.add(record_id)
            members.append(record_id)
        if members:
            groups.append(sorted(members))

    recovered_ids = sorted(known - assigned)
    groups.extend([record_id] for record_id in recovered_ids)
    groups.sort(key=min)

    outcome = PartitionOutcome(
        groups=groups,
        recovered_ids=recovered_ids,
        duplicate_ids=duplicate_ids,
        unknown_ids=unknown_ids,
    )

    if unknown_ids:
        logger.warning(
            f"Block {block_key}: dropped {len(unknown_ids)} record ids the model returned "
            f"that were not in the block: {sorted(set(unknown_ids))[:10]}"
        )
    if duplicate_ids:
        logger.warning(
            f"Block {block_key}: {len(duplicate_ids)} record ids appeared in more than one "
            f"group; kept the first mention of each: {sorted(set(duplicate_ids))[:10]}"
        )
    if recovered_ids:
        logger.warning(
            f"Block {block_key}: the model left {len(recovered_ids)} of {len(known)} records "
            f"out of every group; recovered them as unmatched: {recovered_ids[:10]}"
        )

    if outcome.covered() != known:
        raise AssertionError(
            f"Block {block_key}: partition covers {len(outcome.covered())} of {len(known)} "
            "records after recovery, which should be impossible"
        )
    return outcome


def partition_from_pairs(
    pairs: Iterable[tuple[int, int]],
    known_ids: Iterable[int],
) -> list[list[int]]:
    """Group ids into the partition a set of pairs implies.

    The bridge from the old pairwise contract, and from a gold standard that is
    expressed as pairs, to the grouping the pipeline now works in.

    Parameters
    ----------
    pairs : Iterable[tuple[int, int]]
        Pairs asserting that two records are the same entity
    known_ids : Iterable[int]
        Every record id that should appear in the partition

    Returns
    -------
    list[list[int]]
        Partition covering exactly ``known_ids``, each group sorted, groups
        ordered by their smallest id
    """
    parent: dict[int, int] = {int(value): int(value) for value in known_ids}

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for left, right in pairs:
        if left not in parent or right not in parent:
            continue
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    grouped: dict[int, list[int]] = {}
    for node in parent:
        grouped.setdefault(find(node), []).append(node)
    return sorted((sorted(group) for group in grouped.values()), key=min)
