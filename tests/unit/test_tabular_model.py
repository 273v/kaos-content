"""Tests for TabularDocument model: ColumnType, Column, Table, TabularDocument."""

from __future__ import annotations

import datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from kaos_content.model.metadata import DocumentMetadata
from kaos_content.model.tabular import (
    Column,
    ColumnType,
    Table,
    TabularDocument,
    column_type_from_python,
    infer_column_type,
)

# ---------------------------------------------------------------------------
# ColumnType
# ---------------------------------------------------------------------------


class TestColumnType:
    """Test ColumnType enum values and membership."""

    def test_all_types_exist(self) -> None:
        # Tier 1 (8) + Tier 2 (3) + Tier 3 (2) + Tier 4 extraction (4) = 17
        assert len(ColumnType) == 17

    def test_tier4_extraction_types(self) -> None:
        tier4 = {
            ColumnType.VERBATIM_QUOTE,
            ColumnType.MONEY,
            ColumnType.SCORE,
            ColumnType.ENTITY_ROLE,
        }
        assert len(tier4) == 4

    def test_tier1_types(self) -> None:
        tier1 = {
            ColumnType.TEXT,
            ColumnType.INTEGER,
            ColumnType.FLOAT,
            ColumnType.BOOLEAN,
            ColumnType.DATE,
            ColumnType.TIME,
            ColumnType.DATETIME,
            ColumnType.NULL,
        }
        assert len(tier1) == 8

    def test_tier2_types(self) -> None:
        tier2 = {ColumnType.DECIMAL, ColumnType.BINARY, ColumnType.DURATION}
        assert len(tier2) == 3

    def test_tier3_types(self) -> None:
        tier3 = {ColumnType.LIST, ColumnType.STRUCT}
        assert len(tier3) == 2

    def test_values_are_lowercase_strings(self) -> None:
        for ct in ColumnType:
            assert ct.value == ct.value.lower()
            assert ct.value == ct.name.lower()

    def test_strenum_usage(self) -> None:
        assert str(ColumnType.TEXT) == "text"
        assert ColumnType("integer") == ColumnType.INTEGER


# ---------------------------------------------------------------------------
# column_type_from_python
# ---------------------------------------------------------------------------


class TestColumnTypeFromPython:
    """Test Python value → ColumnType inference."""

    def test_none(self) -> None:
        assert column_type_from_python(None) is ColumnType.NULL

    def test_str(self) -> None:
        assert column_type_from_python("hello") is ColumnType.TEXT

    def test_int(self) -> None:
        assert column_type_from_python(42) is ColumnType.INTEGER

    def test_float(self) -> None:
        assert column_type_from_python(3.14) is ColumnType.FLOAT

    def test_bool_not_int(self) -> None:
        # bool is subclass of int — must resolve to BOOLEAN
        assert column_type_from_python(True) is ColumnType.BOOLEAN
        assert column_type_from_python(False) is ColumnType.BOOLEAN

    def test_date(self) -> None:
        assert column_type_from_python(datetime.date(2024, 1, 1)) is ColumnType.DATE

    def test_time(self) -> None:
        assert column_type_from_python(datetime.time(12, 30)) is ColumnType.TIME

    def test_datetime_not_date(self) -> None:
        # datetime is subclass of date — must resolve to DATETIME
        assert column_type_from_python(datetime.datetime(2024, 1, 1, 12)) is ColumnType.DATETIME

    def test_decimal(self) -> None:
        assert column_type_from_python(Decimal("1234.56")) is ColumnType.DECIMAL

    def test_bytes(self) -> None:
        assert column_type_from_python(b"\x00\x01") is ColumnType.BINARY

    def test_bytearray(self) -> None:
        assert column_type_from_python(bytearray(b"\x00")) is ColumnType.BINARY

    def test_timedelta(self) -> None:
        assert column_type_from_python(datetime.timedelta(hours=2)) is ColumnType.DURATION

    def test_list(self) -> None:
        assert column_type_from_python([1, 2, 3]) is ColumnType.LIST

    def test_dict(self) -> None:
        assert column_type_from_python({"a": 1}) is ColumnType.STRUCT

    def test_unknown_falls_back_to_text(self) -> None:
        assert column_type_from_python(object()) is ColumnType.TEXT


# ---------------------------------------------------------------------------
# infer_column_type
# ---------------------------------------------------------------------------


class TestInferColumnType:
    """Test column type inference from value sequences."""

    def test_all_none(self) -> None:
        assert infer_column_type((None, None, None)) is ColumnType.NULL

    def test_uniform_int(self) -> None:
        assert infer_column_type((1, 2, 3)) is ColumnType.INTEGER

    def test_uniform_str(self) -> None:
        assert infer_column_type(("a", "b")) is ColumnType.TEXT

    def test_int_with_none(self) -> None:
        assert infer_column_type((1, None, 3)) is ColumnType.INTEGER

    def test_int_float_widens_to_float(self) -> None:
        assert infer_column_type((1, 2.0, 3)) is ColumnType.FLOAT

    def test_int_decimal_widens_to_decimal(self) -> None:
        assert infer_column_type((1, Decimal("2.00"))) is ColumnType.DECIMAL

    def test_float_decimal_widens_to_float(self) -> None:
        assert infer_column_type((1.0, Decimal("2.00"))) is ColumnType.FLOAT

    def test_int_float_decimal_widens_to_float(self) -> None:
        assert infer_column_type((1, 2.0, Decimal("3"))) is ColumnType.FLOAT

    def test_date_datetime_widens_to_datetime(self) -> None:
        assert (
            infer_column_type((datetime.date(2024, 1, 1), datetime.datetime(2024, 1, 2, 12)))
            is ColumnType.DATETIME
        )

    def test_mixed_types_widen_to_text(self) -> None:
        assert infer_column_type((1, "hello", True)) is ColumnType.TEXT

    def test_empty(self) -> None:
        assert infer_column_type(()) is ColumnType.NULL


# ---------------------------------------------------------------------------
# Column
# ---------------------------------------------------------------------------


class TestColumn:
    """Test Column dataclass."""

    def test_defaults(self) -> None:
        col = Column("name")
        assert col.name == "name"
        assert col.column_type is ColumnType.TEXT
        assert col.nullable is True
        assert col.metadata == {}

    def test_explicit_type(self) -> None:
        col = Column("amount", ColumnType.DECIMAL, nullable=False)
        assert col.column_type is ColumnType.DECIMAL
        assert col.nullable is False

    def test_with_metadata(self) -> None:
        col = Column("price", ColumnType.FLOAT, metadata={"format_str": "$#,##0.00"})
        assert col.metadata["format_str"] == "$#,##0.00"

    def test_frozen(self) -> None:
        col = Column("x")
        with pytest.raises(AttributeError):
            col.name = "y"  # ty: ignore[invalid-assignment]


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------


class TestTable:
    """Test Table dataclass."""

    def test_empty_table(self) -> None:
        t = Table(name="empty")
        assert t.name == "empty"
        assert t.columns == ()
        assert t.rows == ()
        assert t.row_count == 0

    def test_row_count_auto_set(self) -> None:
        t = Table(
            name="data",
            columns=(Column("a"),),
            rows=(("x",), ("y",), ("z",)),
        )
        assert t.row_count == 3

    def test_row_count_explicit(self) -> None:
        # Truncated: 2 rows loaded, but source has 1000
        t = Table(
            name="data",
            columns=(Column("a"),),
            rows=(("x",), ("y",)),
            row_count=1000,
        )
        assert t.row_count == 1000
        assert len(t.rows) == 2

    def test_column_names(self) -> None:
        t = Table(
            name="t",
            columns=(Column("a"), Column("b"), Column("c")),
        )
        assert t.column_names() == ("a", "b", "c")

    def test_column_values(self) -> None:
        t = Table(
            name="t",
            columns=(Column("x", ColumnType.INTEGER), Column("y", ColumnType.TEXT)),
            rows=((1, "a"), (2, "b"), (3, "c")),
        )
        assert t.column_values("x") == (1, 2, 3)
        assert t.column_values("y") == ("a", "b", "c")

    def test_column_values_not_found(self) -> None:
        t = Table(name="t", columns=(Column("a"),))
        with pytest.raises(KeyError, match="Column not found"):
            t.column_values("z")

    def test_with_metadata(self) -> None:
        t = Table(
            name="sheet1",
            columns=(Column("a"),),
            rows=(("x",),),
            metadata={"formulas": {"A2": "=SUM(A1)"}},
        )
        assert t.metadata["formulas"]["A2"] == "=SUM(A1)"

    def test_frozen(self) -> None:
        t = Table(name="t")
        with pytest.raises(AttributeError):
            t.name = "new"  # ty: ignore[invalid-assignment]


# ---------------------------------------------------------------------------
# TabularDocument
# ---------------------------------------------------------------------------


class TestTabularDocument:
    """Test TabularDocument Pydantic model."""

    def test_empty_document(self) -> None:
        doc = TabularDocument()
        assert doc.tables == ()
        assert doc.provenance is None
        assert doc.metadata.title is None

    def test_with_metadata(self) -> None:
        doc = TabularDocument(
            metadata=DocumentMetadata(title="Sales Report", document_type="xlsx"),
        )
        assert doc.metadata.title == "Sales Report"

    def test_with_tables(self) -> None:
        t1 = Table(name="sheet1", columns=(Column("a"),), rows=(("x",),))
        t2 = Table(name="sheet2", columns=(Column("b"),), rows=(("y",),))
        doc = TabularDocument(tables=(t1, t2))
        assert len(doc.tables) == 2
        assert doc.table_names() == ("sheet1", "sheet2")

    def test_get_table(self) -> None:
        t = Table(name="data", columns=(Column("x"),))
        doc = TabularDocument(tables=(t,))
        assert doc.get_table("data") is t

    def test_get_table_not_found(self) -> None:
        doc = TabularDocument()
        with pytest.raises(KeyError, match="Table not found"):
            doc.get_table("missing")

    def test_frozen(self) -> None:
        doc = TabularDocument()
        with pytest.raises(ValidationError):
            doc.tables = ()

    def test_realistic_csv_document(self) -> None:
        """Simulate a CSV file loaded as TabularDocument."""
        doc = TabularDocument(
            metadata=DocumentMetadata(title="states.csv"),
            tables=(
                Table(
                    name="states",
                    columns=(
                        Column("state", ColumnType.TEXT),
                        Column("capital", ColumnType.TEXT),
                        Column("population", ColumnType.INTEGER),
                    ),
                    rows=(
                        ("California", "Sacramento", 39538223),
                        ("Texas", "Austin", 29145505),
                        ("Florida", "Tallahassee", 21538187),
                    ),
                ),
            ),
        )
        assert doc.tables[0].row_count == 3
        assert doc.tables[0].column_values("state") == ("California", "Texas", "Florida")

    def test_realistic_xlsx_document(self) -> None:
        """Simulate an XLSX workbook with formulas in metadata."""
        doc = TabularDocument(
            metadata=DocumentMetadata(title="Budget.xlsx"),
            tables=(
                Table(
                    name="Q1",
                    columns=(
                        Column("category", ColumnType.TEXT),
                        Column("amount", ColumnType.DECIMAL),
                    ),
                    rows=(
                        ("Revenue", Decimal("1000000.00")),
                        ("Expenses", Decimal("750000.00")),
                        ("Profit", Decimal("250000.00")),
                    ),
                    metadata={
                        "formulas": {"B4": "=B2-B3"},
                        "frozen_rows": 1,
                    },
                ),
            ),
        )
        assert doc.tables[0].metadata["formulas"]["B4"] == "=B2-B3"

    def test_realistic_sql_document(self) -> None:
        """Simulate a database table loaded as TabularDocument."""
        doc = TabularDocument(
            metadata=DocumentMetadata(title="users table"),
            tables=(
                Table(
                    name="users",
                    columns=(
                        Column("id", ColumnType.INTEGER, nullable=False),
                        Column("email", ColumnType.TEXT),
                        Column("created_at", ColumnType.DATETIME),
                        Column("active", ColumnType.BOOLEAN),
                    ),
                    rows=(
                        (1, "alice@example.com", datetime.datetime(2024, 1, 15, 9, 30), True),
                        (2, "bob@example.com", datetime.datetime(2024, 2, 20, 14, 0), False),
                    ),
                    metadata={
                        "schema": "public",
                        "primary_key": ["id"],
                    },
                ),
            ),
        )
        assert doc.tables[0].columns[0].nullable is False
        assert doc.tables[0].metadata["primary_key"] == ["id"]

    def test_multi_type_document(self) -> None:
        """Verify all ColumnTypes can coexist in one Table."""
        cols = tuple(Column(ct.value, ct) for ct in ColumnType)
        t = Table(name="all_types", columns=cols)
        assert len(t.columns) == len(list(ColumnType))
        for col, ct in zip(t.columns, ColumnType, strict=True):
            assert col.column_type is ct
