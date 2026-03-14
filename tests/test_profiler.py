"""Tests for DatasetProfiler."""

from serf.analyze.profiler import DatasetProfiler
from serf.dspy.types import DatasetProfile


def test_profile_with_sample_records() -> None:
    """Test profile with sample records (companies with name, url, revenue)."""
    records = [
        {"name": "Apple Inc.", "url": "https://apple.com", "revenue": 394_000_000_000},
        {"name": "Google LLC", "url": "https://google.com", "revenue": 282_000_000_000},
        {"name": "Microsoft Corp", "url": "https://microsoft.com", "revenue": 211_000_000_000},
    ]
    profiler = DatasetProfiler()
    profile = profiler.profile(records)

    assert isinstance(profile, DatasetProfile)
    assert profile.record_count == 3
    assert len(profile.field_profiles) == 3

    names = {fp.name for fp in profile.field_profiles}
    assert names == {"name", "url", "revenue"}


def test_completeness_calculation() -> None:
    """Test completeness calculation."""
    records = [
        {"name": "A", "optional": "x"},
        {"name": "B", "optional": "y"},
        {"name": "C"},
    ]
    profiler = DatasetProfiler()
    profile = profiler.profile(records)

    name_fp = next(fp for fp in profile.field_profiles if fp.name == "name")
    optional_fp = next(fp for fp in profile.field_profiles if fp.name == "optional")

    assert name_fp.completeness == 1.0
    assert optional_fp.completeness == round(2 / 3, 4)


def test_recommended_fields_detection() -> None:
    """Test recommended fields detection."""
    records = [
        {"name": "Company A", "url": "https://a.com"},
        {"name": "Company B", "url": "https://b.com"},
        {"name": "Company C", "url": "https://c.com"},
    ]
    profiler = DatasetProfiler()
    profile = profiler.profile(records)

    assert "name" in profile.recommended_blocking_fields
    assert "name" in profile.recommended_matching_fields
    assert "url" in profile.recommended_matching_fields


def test_empty_records() -> None:
    """Test empty records."""
    profiler = DatasetProfiler()
    profile = profiler.profile([])

    assert profile.record_count == 0
    assert profile.field_profiles == []
    assert profile.recommended_blocking_fields == []
    assert profile.recommended_matching_fields == []
    assert profile.estimated_duplicate_rate == 0.0
