"""Tests for entity name normalization."""

from serf.block.normalize import (
    get_acronyms,
    get_basename,
    get_corporate_ending,
    get_multilingual_stop_words,
    normalize_name,
    remove_domain_suffix,
)


def test_normalize_name_basic() -> None:
    """Test basic name normalization."""
    assert normalize_name("Apple Inc.") == "apple inc"


def test_normalize_name_whitespace() -> None:
    """Test whitespace collapsing."""
    assert normalize_name("  Multiple   Spaces  ") == "multiple spaces"


def test_normalize_name_punctuation() -> None:
    """Test punctuation removal."""
    assert normalize_name("O'Brien & Associates, L.L.C.") == "o brien associates l l c"


def test_normalize_name_unicode() -> None:
    """Test unicode normalization."""
    result = normalize_name("Ünited Tëchnologies")
    assert "united" in result.lower()


def test_get_basename_inc() -> None:
    """Test removing Inc. suffix."""
    assert get_basename("Apple Inc.") == "Apple"


def test_get_basename_llc() -> None:
    """Test removing LLC suffix."""
    assert get_basename("Google LLC") == "Google"


def test_get_basename_no_suffix() -> None:
    """Test name without corporate suffix."""
    result = get_basename("Alphabet")
    assert result == "Alphabet"


def test_get_corporate_ending() -> None:
    """Test extracting corporate ending."""
    ending = get_corporate_ending("Microsoft Corporation")
    assert "Corporation" in ending


def test_get_acronyms_multi_word() -> None:
    """Test acronym generation from multi-word name."""
    result = get_acronyms("International Business Machines")
    assert "IBM" in result


def test_get_acronyms_single_word() -> None:
    """Test that single words don't generate acronyms."""
    result = get_acronyms("Apple")
    assert result == []


def test_get_acronyms_filters_stop_words() -> None:
    """Test that stop words are filtered from acronyms."""
    result = get_acronyms("Bank of America Corporation")
    # "of" should be filtered, so acronym is "BA" not "BOA"
    assert len(result) > 0
    acronym = result[0]
    assert "O" not in acronym  # "of" was filtered


def test_get_multilingual_stop_words() -> None:
    """Test stop words include multiple languages."""
    stop_words = get_multilingual_stop_words()
    assert "the" in stop_words
    assert "and" in stop_words
    assert "der" in stop_words  # German
    assert "le" in stop_words  # French
    assert "el" in stop_words  # Spanish


def test_remove_domain_suffix_com() -> None:
    """Test removing .com suffix."""
    assert remove_domain_suffix("amazon.com") == "amazon"


def test_remove_domain_suffix_co_uk() -> None:
    """Test removing .co.uk suffix."""
    assert remove_domain_suffix("amazon.co.uk") == "amazon"


def test_remove_domain_suffix_none() -> None:
    """Test name without domain suffix."""
    assert remove_domain_suffix("Amazon") == "Amazon"


def test_remove_domain_suffix_ai() -> None:
    """Test removing .ai suffix."""
    assert remove_domain_suffix("graphlet.ai") == "graphlet"
