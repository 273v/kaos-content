"""Tests for tabular artifact support: JSON round-trip and summary/schema."""

from __future__ import annotations

import datetime
from decimal import Decimal

import pytest

from kaos_content.artifacts import (
    _tabular_from_json,
    _tabular_to_json,
    tabular_schema,
    tabular_summary,
)
from kaos_content.model.metadata import DocumentMetadata
from kaos_content.model.tabular import (
    Column,
    ColumnType,
    Table,
    TabularDocument,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_doc() -> TabularDocument:
    return TabularDocument(
        metadata=DocumentMetadata(title="Test Report"),
        tables=(
            Table(
                name="sales",
                columns=(
                    Column("date", ColumnType.DATE),
                    Column("amount", ColumnType.DECIMAL),
                    Column("region", ColumnType.TEXT),
                ),
                rows=(
                    (datetime.date(2024, 1, 1), Decimal("1234.56"), "US"),
                    (datetime.date(2024, 1, 2), Decimal("789.01"), "EU"),
                ),
                metadata={"source": "database"},
            ),
            Table(
                name="regions",
                columns=(
                    Column("code", ColumnType.TEXT),
                    Column("name", ColumnType.TEXT),
                ),
                rows=(("US", "United States"), ("EU", "European Union")),
            ),
        ),
    )


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------


class TestTabularJsonRoundTrip:
    def test_serialize_deserialize(self, sample_doc: TabularDocument) -> None:
        json_str = _tabular_to_json(sample_doc)
        restored = _tabular_from_json(json_str)

        assert restored.metadata.title == "Test Report"
        assert len(restored.tables) == 2
        assert restored.tables[0].name == "sales"
        assert restored.tables[1].name == "regions"

    def test_column_types_preserved(self, sample_doc: TabularDocument) -> None:
        json_str = _tabular_to_json(sample_doc)
        restored = _tabular_from_json(json_str)

        cols = restored.tables[0].columns
        assert cols[0].column_type is ColumnType.DATE
        assert cols[1].column_type is ColumnType.DECIMAL
        assert cols[2].column_type is ColumnType.TEXT

    def test_row_count_preserved(self, sample_doc: TabularDocument) -> None:
        json_str = _tabular_to_json(sample_doc)
        restored = _tabular_from_json(json_str)
        assert restored.tables[0].row_count == 2

    def test_metadata_preserved(self, sample_doc: TabularDocument) -> None:
        json_str = _tabular_to_json(sample_doc)
        restored = _tabular_from_json(json_str)
        assert restored.tables[0].metadata["source"] == "database"

    def test_empty_document(self) -> None:
        doc = TabularDocument()
        json_str = _tabular_to_json(doc)
        restored = _tabular_from_json(json_str)
        assert restored.tables == ()
        assert restored.metadata.title is None

    def test_all_column_types(self) -> None:
        """Ensure all 13 ColumnTypes survive JSON round-trip."""
        cols = tuple(Column(ct.value, ct) for ct in ColumnType)
        t = Table(name="all_types", columns=cols)
        doc = TabularDocument(tables=(t,))

        json_str = _tabular_to_json(doc)
        restored = _tabular_from_json(json_str)

        for orig, rest in zip(doc.tables[0].columns, restored.tables[0].columns, strict=True):
            assert orig.column_type == rest.column_type, (
                f"Lost type: {orig.column_type} -> {rest.column_type}"
            )

    def test_nullable_preserved(self) -> None:
        t = Table(
            name="t",
            columns=(Column("id", ColumnType.INTEGER, nullable=False),),
        )
        doc = TabularDocument(tables=(t,))
        json_str = _tabular_to_json(doc)
        restored = _tabular_from_json(json_str)
        assert restored.tables[0].columns[0].nullable is False

    def test_valid_json(self, sample_doc: TabularDocument) -> None:
        import json

        json_str = _tabular_to_json(sample_doc)
        data = json.loads(json_str)
        assert "tables" in data
        assert "metadata" in data


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


class TestTabularSummary:
    def test_basic(self, sample_doc: TabularDocument) -> None:
        result = tabular_summary(sample_doc)
        assert result["title"] == "Test Report"
        assert result["table_count"] == 2
        assert result["total_rows"] == 4
        assert len(result["tables"]) == 2

    def test_table_info(self, sample_doc: TabularDocument) -> None:
        result = tabular_summary(sample_doc)
        sales = result["tables"][0]
        assert sales["name"] == "sales"
        assert sales["row_count"] == 2
        assert sales["column_count"] == 3
        assert len(sales["columns"]) == 3
        assert sales["columns"][0]["name"] == "date"
        assert sales["columns"][0]["type"] == "date"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class TestTabularSchema:
    def test_basic(self, sample_doc: TabularDocument) -> None:
        result = tabular_schema(sample_doc.tables[0])
        assert result["name"] == "sales"
        assert result["row_count"] == 2
        assert len(result["columns"]) == 3

    def test_column_details(self, sample_doc: TabularDocument) -> None:
        result = tabular_schema(sample_doc.tables[0])
        cols = result["columns"]
        assert cols[0]["name"] == "date"
        assert cols[0]["type"] == "date"
        assert cols[0]["nullable"] is True

    def test_metadata_included(self, sample_doc: TabularDocument) -> None:
        result = tabular_schema(sample_doc.tables[0])
        assert result["metadata"]["source"] == "database"

    def test_no_metadata(self) -> None:
        t = Table(name="t", columns=(Column("x"),))
        result = tabular_schema(t)
        assert result["metadata"] is None
