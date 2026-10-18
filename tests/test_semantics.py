"""Tests for type-aware field value merging."""

from unittest.mock import MagicMock, patch

from serf.merge.semantics import (
    CARDINALITY_LIST,
    CARDINALITY_SINGLE,
    DEDUPE_EXACT,
    DEDUPE_FUZZY,
    DEDUPE_NONE,
    MergePolicy,
    completeness,
    flatten_values,
    merge_values,
    normalize,
    normalize_cardinality,
    policy_for,
    select_best_value,
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


def test_addresses_with_different_street_numbers_stay_separate_individual_addresses() -> None:
    """Two street numbers on the same street are two distinct addresses."""
    values = [
        "123 Main St, Springfield, IL",
        "123 Main Street, Springfield, IL",
        "456 Main Street, Springfield, IL",
    ]
    merged = merge_values(values, "address")
    assert merged == [
        "123 Main Street, Springfield, IL",
        "456 Main Street, Springfield, IL",
    ]


def test_addresses_with_different_unit_numbers_stay_separate_individual_addresses() -> None:
    """Different apartments or suites at the same street number are distinct addresses."""
    values = [
        "100 Broadway Apt 2A",
        "100 Broadway Apartment 2A",
        "100 Broadway Apt 3B",
    ]
    merged = merge_values(values, "address")
    assert merged == [
        "100 Broadway Apartment 2A",
        "100 Broadway Apt 3B",
    ]


def test_name_list_deduplicates_variants_into_individual_names() -> None:
    """Variants of each person's name collapse while distinct people stay separate."""
    values = [
        ["Russell Jurney", "Bob Dorf"],
        ["Russell H Jurney", "Bob M. Dorf"],
        "Russ Journey",
    ]
    merged = merge_values(values, "name")
    assert merged == ["Russell H Jurney", "Bob M. Dorf"]


def test_flatten_values_unpacks_nested_sequences() -> None:
    raw = ["a", ["b", ("c", None)], "d"]
    assert flatten_values(raw) == ["a", "b", "c", None, "d"]


def test_normalize_cardinality_accepts_single_and_singular_and_list() -> None:
    assert normalize_cardinality("list") == CARDINALITY_LIST
    assert normalize_cardinality("single") == CARDINALITY_SINGLE
    assert normalize_cardinality("singular") == CARDINALITY_SINGLE
    assert normalize_cardinality("SINGULAR") == CARDINALITY_SINGLE
    assert normalize_cardinality(None) == CARDINALITY_LIST


def test_select_best_value_deterministic_picks_most_complete_representative() -> None:
    best = select_best_value(
        ["Russell Jurney", "Russell H Jurney", "Russ Journey"],
        "name",
    )
    assert best == "Russell H Jurney"


def test_select_best_value_returns_none_when_all_values_are_blank() -> None:
    assert select_best_value([None, "", "   "], "name") is None


def test_select_best_value_skips_llm_when_only_one_candidate_remains() -> None:
    """When all non-blank inputs are identical text, no LLM call is needed."""
    with patch("serf.merge.semantics.dspy.Predict") as mock_predict:
        best = select_best_value(
            ["123 Main Street", "  123 Main Street  ", None],
            "address",
            use_llm=True,
        )
        assert best == "123 Main Street"
        mock_predict.assert_not_called()


def test_select_best_value_uses_llm_when_multiple_distinct_candidates_exist() -> None:
    """When multiple distinct values exist for a singular field, the LLM selects the best one."""
    fake_prediction = MagicMock()
    fake_prediction.best_value = "Acme Corporation International"
    fake_predictor = MagicMock(return_value=fake_prediction)

    with patch("serf.merge.semantics.dspy.Predict", return_value=fake_predictor):
        best = select_best_value(
            ["Brand X", "Acme Corporation International"],
            "name",
            field_name="company_name",
            field_description="Official registered company name",
            use_llm=True,
        )

    assert best == "Acme Corporation International"
    fake_predictor.assert_called_once()


def test_select_best_value_falls_back_to_completeness_if_llm_raises() -> None:
    fake_predictor = MagicMock(side_effect=RuntimeError("LLM unavailable"))

    with patch("serf.merge.semantics.dspy.Predict", return_value=fake_predictor):
        best = select_best_value(
            ["Short Bio", "A much more complete biography with details"],
            "text",
            field_name="bio",
            use_llm=True,
        )

    assert best == "A much more complete biography with details"
