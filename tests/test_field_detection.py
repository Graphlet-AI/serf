"""Tests for field type detection."""

from serf.analyze.field_detection import detect_field_type


def test_detect_field_type_name() -> None:
    """Test detect_field_type with name fields."""
    assert detect_field_type("name", ["Alice", "Bob", "Charlie"]) == "name"
    assert detect_field_type("title", ["Product A", "Product B"]) == "name"
    assert detect_field_type("product_name", ["Widget X"]) == "name"


def test_detect_field_type_email() -> None:
    """Test with email values."""
    assert detect_field_type("email", ["a@b.com", "x@y.org"]) == "email"
    assert detect_field_type("other", ["user@example.com", "admin@test.org"]) == "email"


def test_detect_field_type_url() -> None:
    """Test with URL values."""
    assert detect_field_type("url", ["https://example.com", "http://test.org"]) == "url"
    assert detect_field_type("website", ["https://a.com"]) == "url"
    assert detect_field_type("x", ["https://x.com", "http://y.net"]) == "url"


def test_detect_field_type_numeric() -> None:
    """Test with numeric values."""
    assert detect_field_type("price", ["10.99", "20.50"]) == "numeric"
    assert detect_field_type("revenue", ["1000000", "2000000"]) == "numeric"
    assert detect_field_type("x", ["123", "456.78", "0"]) == "numeric"


def test_detect_field_type_date() -> None:
    """Test with date values."""
    assert detect_field_type("date", ["2024-01-15", "2023-12-01"]) == "date"
    assert detect_field_type("created", ["01/15/2024", "12/01/2023"]) == "date"
    assert detect_field_type("x", ["2024-01-01", "2023-06-15"]) == "date"


def test_detect_field_type_generic_text() -> None:
    """Test with generic text."""
    assert detect_field_type("description", ["A" * 150]) == "text"
    assert detect_field_type("notes", ["Mixed 123 abc !@#"]) == "text"
    assert detect_field_type("content", ["123 and 456"]) == "text"


def test_detect_field_type_field_name_heuristics() -> None:
    """Test field name heuristics."""
    assert detect_field_type("phone", ["555-1234"]) == "phone"
    assert detect_field_type("address", ["123 Main St"]) == "address"
    assert detect_field_type("id", ["abc-123"]) == "identifier"
    assert detect_field_type("sku", ["SKU-001"]) == "identifier"
