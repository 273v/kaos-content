"""Tests for the shared normalization module.

These are the canonical correctness tests — if a normalization
function changes behavior here, EVERY consumer (dedup, extraction,
search, citations) is affected. Add tests carefully.
"""

from __future__ import annotations

import pytest

from kaos_content.normalize import (
    canonical_hash,
    normalize_date,
    normalize_entity,
    normalize_number,
    normalize_text,
)


class TestNormalizeText:
    def test_whitespace_collapse(self) -> None:
        assert normalize_text("  hello   world  ") == "hello world"
        assert normalize_text("a\tb\nc") == "a b c"

    def test_lowercase(self) -> None:
        assert normalize_text("HELLO") == "hello"
        assert normalize_text("Hello", lowercase=False) == "Hello"

    def test_unicode_nfkc(self) -> None:
        assert normalize_text("caf\u00e9") == normalize_text("cafe\u0301")

    def test_punctuation_stripping(self) -> None:
        assert normalize_text("Hello, world!", strip_punctuation=True) == "hello world"
        assert normalize_text("Hello, world!", strip_punctuation=False) == "hello, world!"


class TestNormalizeEntity:
    @pytest.mark.parametrize(
        "input_text,expected",
        [
            ("Acme Corp", "acme"),
            ("Acme Corp.", "acme"),
            ("Acme Corporation", "acme"),
            ("Acme Inc.", "acme"),
            ("Acme Incorporated", "acme"),
            ("ACME LLC", "acme"),
            ("Acme Holdings LLC", "acme"),
            ("Beta GmbH", "beta"),
            ("Gamma SA de CV", "gamma"),  # multi-word suffix
            ("Delta K.K.", "delta"),  # Japanese suffix
        ],
    )
    def test_suffix_stripping(self, input_text: str, expected: str) -> None:
        assert normalize_entity(input_text) == expected

    def test_no_suffix_passthrough(self) -> None:
        assert normalize_entity("John Smith") == "john smith"

    def test_all_caps(self) -> None:
        assert normalize_entity("ACME CORP") == "acme"

    def test_whitespace_variation(self) -> None:
        assert normalize_entity("  Acme   Corp.  ") == "acme"


class TestNormalizeDate:
    @pytest.mark.parametrize(
        "input_text,expected",
        [
            ("2024-01-15", "2024-01-15"),
            ("01/15/2024", "2024-01-15"),
            ("15 January 2024", "2024-01-15"),
            ("15 Jan 2024", "2024-01-15"),
            ("January 15, 2024", "2024-01-15"),
        ],
    )
    def test_date_formats(self, input_text: str, expected: str) -> None:
        assert normalize_date(input_text) == expected

    def test_unparseable_falls_back_to_text(self) -> None:
        result = normalize_date("the sixteenth day of September")
        assert result == "the sixteenth day of september"

    def test_never_returns_none(self) -> None:
        result = normalize_date("garbage")
        assert result is not None
        assert isinstance(result, str)


class TestNormalizeNumber:
    def test_integer(self) -> None:
        assert normalize_number("1234") == "1234"

    def test_decimal_normalized(self) -> None:
        # Decimal.normalize() strips trailing zeros for consistent hashing
        assert normalize_number("13.50") == "13.5"
        assert normalize_number("13.5") == normalize_number("13.50")

    def test_currency_stripped(self) -> None:
        assert normalize_number("$1,000,000") == "1000000"
        assert normalize_number("€250.00") == "250"

    def test_unparseable_falls_back_to_text(self) -> None:
        result = normalize_number("ten million dollars")
        assert result == "ten million dollars"


class TestCanonicalHash:
    def test_deterministic(self) -> None:
        assert canonical_hash("hello") == canonical_hash("hello")

    def test_different_inputs_different_hashes(self) -> None:
        assert canonical_hash("hello") != canonical_hash("world")

    def test_sha256_length(self) -> None:
        assert len(canonical_hash("test")) == 64


class TestCrossConsumerConsistency:
    """Verify that the same value produces the same hash across all
    normalization paths. This is the canonical invariant the shared
    module exists to enforce."""

    def test_entity_across_variants(self) -> None:
        variants = [
            "Acme Corp",
            "Acme Corp.",
            "ACME CORP",
            "Acme Corporation",
            "  Acme   Corp.  ",
        ]
        hashes = {canonical_hash(normalize_entity(v)) for v in variants}
        assert len(hashes) == 1, f"Entity variants produced different hashes: {hashes}"

    def test_date_across_formats(self) -> None:
        variants = ["2024-01-15", "01/15/2024", "15 January 2024", "January 15, 2024"]
        hashes = {canonical_hash(normalize_date(v)) for v in variants}
        assert len(hashes) == 1, f"Date variants produced different hashes: {hashes}"

    def test_number_across_formats(self) -> None:
        variants = ["$1,000,000", "1000000", "$1000000"]
        hashes = {canonical_hash(normalize_number(v)) for v in variants}
        assert len(hashes) == 1, f"Number variants produced different hashes: {hashes}"
