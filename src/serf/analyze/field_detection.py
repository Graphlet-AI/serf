"""Field type auto-detection for dataset profiling."""

import re
from typing import Any

from serf.logs import get_logger

logger = get_logger(__name__)

# Field name patterns that suggest specific types
NAME_PATTERNS = (
    r"^(names?|titles?|product_names?|company_names?|entity_names?|display_names?|full_names?)$",
    r"^(first_names?|last_names?|given_names?|surnames?|alias(es)?|aka|alternate_names?)$",
)
EMAIL_PATTERNS = r"^(emails?|e_mails?|mails?|email_address(es)?)$"
URL_PATTERNS = r"^(urls?|websites?|links?|uris?|homepages?)$"
PHONE_PATTERNS = r"^(phones?|tels?|mobiles?|fax(es)?|contact_numbers?|phone_numbers?)$"
ADDRESS_PATTERNS = (
    r"^(address(es)?|streets?|street_address(es)?|mailing_address(es)?|billing_address(es)?"
    r"|shipping_address(es)?|home_address(es)?|work_address(es)?|city|cities|states?"
    r"|zips?|zip_?codes?|postal_?codes?|country|countries|locations?)$"
)
IDENTIFIER_PATTERNS = r"^(ids?|uuids?|skus?|eans?|upcs?|isbns?|asins?|identifiers?)$"
DATE_PATTERNS = r"^(dates?|created|updated|timestamps?|dobs?|birth_dates?)$"
NUMERIC_PATTERNS = r"^(prices?|revenues?|amounts?|counts?|quantity|quantities|nums?|sizes?)$"

# Value patterns (regex)
EMAIL_VALUE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$", re.IGNORECASE)
URL_VALUE = re.compile(r"^https?://|^www\.", re.IGNORECASE)
PHONE_VALUE = re.compile(r"^[\d\s\-\(\)\.\+]+$")
ADDRESS_VALUE = re.compile(
    r"^\d+\s+.*\b(st|street|ave|avenue|rd|road|blvd|boulevard|dr|drive|ln|lane|ct|court|pl|place|sq|square|pkwy|parkway|hwy|highway)\b",
    re.IGNORECASE,
)
DATE_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}")
DATE_US = re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4}")
NUMERIC_VALUE = re.compile(r"^[\d\.,\-]+$")


def _flatten_sample_values(values: list[Any]) -> list[Any]:
    """Flatten scalar and list/tuple/set sample values into one list."""
    flat: list[Any] = []
    for item in values:
        if isinstance(item, (list, tuple, set)):
            flat.extend(_flatten_sample_values(list(item)))
        else:
            flat.append(item)
    return flat


def detect_field_type(field_name: str, values: list[Any]) -> str:
    """Detect the type of a field based on its name and sample values.

    Returns one of: "name", "email", "url", "phone", "address",
    "identifier", "date", "numeric", "text"

    Uses both the field name (heuristics) and value patterns (regex):
    - "name", "title", "product_name" etc. -> "name"
    - Contains @ -> "email"
    - Starts with http/www -> "url"
    - Phone patterns (digits, dashes, parens) -> "phone"
    - Street/city/state patterns -> "address"
    - All numeric -> "numeric"
    - Date patterns (YYYY-MM-DD, MM/DD/YYYY) -> "date"
    - High uniqueness + alphanumeric -> "identifier"
    - Default -> "text"

    Parameters
    ----------
    field_name : str
        The name of the field
    values : list[Any]
        Sample values from the field (non-null)

    Returns
    -------
    str
        Detected type: name, email, url, phone, address, identifier, date, numeric, text
    """
    name_lower = field_name.lower().replace(" ", "_")

    # Check field name heuristics first
    for pat in NAME_PATTERNS:
        if re.match(pat, name_lower):
            return "name"
    if re.match(EMAIL_PATTERNS, name_lower):
        return "email"
    if re.match(URL_PATTERNS, name_lower):
        return "url"
    if re.match(PHONE_PATTERNS, name_lower):
        return "phone"
    if re.match(ADDRESS_PATTERNS, name_lower):
        return "address"
    if re.match(IDENTIFIER_PATTERNS, name_lower):
        return "identifier"
    if re.match(DATE_PATTERNS, name_lower):
        return "date"
    if re.match(NUMERIC_PATTERNS, name_lower):
        return "numeric"

    # Fall back to value-based detection
    flat_values = _flatten_sample_values(values)
    str_values = [str(v).strip() for v in flat_values if v is not None and str(v).strip()]
    if not str_values:
        return "text"

    # Value pattern checks (date before phone: dates use dashes like 2024-01-15)
    email_count = sum(1 for v in str_values if EMAIL_VALUE.match(v))
    if email_count >= len(str_values) * 0.5:
        return "email"

    url_count = sum(1 for v in str_values if URL_VALUE.search(v))
    if url_count >= len(str_values) * 0.5:
        return "url"

    date_count = sum(1 for v in str_values if DATE_ISO.match(v) or DATE_US.match(v))
    if date_count >= len(str_values) * 0.5:
        return "date"

    phone_count = sum(1 for v in str_values if len(v) >= 7 and PHONE_VALUE.match(v))
    if phone_count >= len(str_values) * 0.5:
        return "phone"

    address_count = sum(1 for v in str_values if ADDRESS_VALUE.match(v))
    if address_count >= len(str_values) * 0.5:
        return "address"

    numeric_count = sum(1 for v in str_values if NUMERIC_VALUE.match(v))
    if numeric_count >= len(str_values) * 0.5:
        return "numeric"

    # Name-like: short, predominantly alphabetic (not mostly digits/symbols)
    def looks_name_like(s: str) -> bool:
        if not s or len(s) >= 100:
            return False
        alpha = sum(1 for c in s if c.isalpha())
        return alpha >= len(s) * 0.5

    name_count = sum(1 for v in str_values if looks_name_like(v))
    if name_count >= len(str_values) * 0.5:
        return "name"

    return "text"
