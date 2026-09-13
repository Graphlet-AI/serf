"""Tests for type-aware field value merging."""

from serf.merge.semantics import (
    DEDUPE_EXACT,
    DEDUPE_FUZZY,
    DEDUPE_NONE,
    MergePolicy,
    completeness,
    merge_values,
    normalize,
    policy_for,
    similarity,
)


def test_name_variants_collapse_to_the_most_complete_spelling() -> None:
    """The spec's own example: three spellings of one name become one value."""
    values = ["Russell Jurney", "Russell H Jurney", "Russ Journey"]

    assert merge_values(values, "name") == ["Russell H Jurney"]


def test_the_collapse_reaches_through_an_intermediate_spelling() -> None:
    """Single linkage: "Russ Journey" only reaches the fullest name via the middle one."""
    assert similarity("Russell H Jurney", "Russ Journey", "name") < 0.8
    assert similarity("Russell Jurney", "Russ Journey", "name") >= 0.8

    assert merge_values(["Russell H Jurney", "Russ Journey"], "name") == [
        "Russell H Jurney",
        "Russ Journey",
    ]


def test_two_genuinely_different_values_are_both_kept() -> None:
    """ "CA" and "WA" are two states, not two spellings of one."""
    assert merge_values(["WA", "CA"], "address") == ["CA", "WA"]


def test_output_is_ordered_most_complete_first() -> None:
    """A consumer that wants one value takes the head and gets the fullest one."""
    merged = merge_values(["Main St", "123 Main Street Suite 400"], "address")

    assert merged[0] == "123 Main Street Suite 400"


def test_ties_break_on_text_so_the_order_is_not_an_artifact_of_record_order() -> None:
    assert merge_values(["WA", "CA"], "address") == merge_values(["CA", "WA"], "address")


def test_freeform_prose_is_never_merged() -> None:
    """Two descriptions are two facts. Rule 4's explicit exception."""
    values = ["A search company.", "A search and advertising company."]

    assert merge_values(values, "text") == values


def test_freeform_still_drops_exact_repeats() -> None:
    """Not merging is not the same as not deduplicating."""
    assert merge_values(["same", "same"], "text") == ["same"]


def test_address_abbreviations_collapse() -> None:
    assert merge_values(["123 Main St", "123 Main Street"], "address") == ["123 Main Street"]


def test_identifiers_collapse_only_on_punctuation_and_case() -> None:
    """Fuzzy matching on identifiers would merge two different part numbers."""
    assert merge_values(["ABC-123", "abc123"], "identifier") == ["ABC-123"]
    assert merge_values(["CCH1B", "CCH1P"], "identifier") == ["CCH1B", "CCH1P"]


def test_urls_ignore_scheme_and_www_and_trailing_slash() -> None:
    merged = merge_values(["https://www.example.com/", "example.com"], "url")

    assert len(merged) == 1


def test_phones_compare_on_digits_alone() -> None:
    merged = merge_values(["(415) 555-0100", "4155550100"], "phone")

    assert len(merged) == 1


def test_blank_and_missing_values_are_dropped() -> None:
    assert merge_values([None, "", "   ", "real"], "name") == ["real"]


def test_merging_nothing_yields_nothing() -> None:
    assert merge_values([], "name") == []
    assert merge_values([None, ""], "name") == []


def test_normalize_is_stable_for_equal_values() -> None:
    assert normalize("Russell H. Jurney", "name") == normalize("russell h jurney", "name")


def test_similarity_of_a_missing_value_is_zero() -> None:
    assert similarity("", "anything", "name") == 0.0


def test_completeness_prefers_more_tokens_then_more_characters() -> None:
    assert completeness("Russell H Jurney", "name") > completeness("Russell Jurney", "name")
    assert completeness("Russelll", "name") > completeness("Russell", "name")


def test_policies_come_from_config() -> None:
    assert policy_for("name").dedupe == DEDUPE_FUZZY
    assert policy_for("identifier").dedupe == DEDUPE_EXACT
    assert policy_for("text").dedupe == DEDUPE_NONE


def test_an_unconfigured_field_type_deduplicates_exactly() -> None:
    """The safe default: collapse only what is literally the same."""
    assert policy_for("no_such_type") == MergePolicy(dedupe=DEDUPE_EXACT, threshold=0.8)


def test_non_string_values_keep_their_type() -> None:
    assert merge_values([2005, 2005, 1999], "numeric") == [1999, 2005]
