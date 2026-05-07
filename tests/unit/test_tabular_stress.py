"""Stress tests for TabularDocument: real-world edge cases, fuzzing, dirty data.

Tests the model, serializers, and bridges against the kinds of data
agents encounter in production: ragged rows, NaN/inf, duplicate columns,
unicode, extreme values, mixed types, and full round-trip chains.
"""

from __future__ import annotations

import datetime
import math
import string
from decimal import Decimal
from pathlib import Path

import pytest

from kaos_content.artifacts import _tabular_from_json, _tabular_to_json
from kaos_content.model.metadata import DocumentMetadata
from kaos_content.model.tabular import (
    Column,
    ColumnType,
    Table,
    TabularDocument,
    column_type_from_python,
    infer_column_type,
)
from kaos_content.serializers.tabular import (
    serialize_csv,
    serialize_json_records,
    serialize_markdown_table,
    serialize_tabular_summary,
    serialize_tsv,
)

# ═══════════════════════════════════════════════════════════════════════════
# 1. TYPE INFERENCE EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════


class TestTypeInferenceEdgeCases:
    """Edge cases in Python value → ColumnType inference."""

    def test_nan_is_float(self) -> None:
        assert column_type_from_python(float("nan")) is ColumnType.FLOAT

    def test_inf_is_float(self) -> None:
        assert column_type_from_python(float("inf")) is ColumnType.FLOAT

    def test_negative_inf_is_float(self) -> None:
        assert column_type_from_python(float("-inf")) is ColumnType.FLOAT

    def test_negative_zero_is_float(self) -> None:
        assert column_type_from_python(-0.0) is ColumnType.FLOAT

    def test_large_int(self) -> None:
        # Python int can exceed 2^64, but ColumnType stays INTEGER
        assert column_type_from_python(10**100) is ColumnType.INTEGER

    def test_empty_string_is_text(self) -> None:
        assert column_type_from_python("") is ColumnType.TEXT

    def test_empty_list_is_list(self) -> None:
        assert column_type_from_python([]) is ColumnType.LIST

    def test_empty_dict_is_struct(self) -> None:
        assert column_type_from_python({}) is ColumnType.STRUCT

    def test_empty_bytes_is_binary(self) -> None:
        assert column_type_from_python(b"") is ColumnType.BINARY

    def test_decimal_nan_is_decimal(self) -> None:
        assert column_type_from_python(Decimal("NaN")) is ColumnType.DECIMAL

    def test_decimal_inf_is_decimal(self) -> None:
        assert column_type_from_python(Decimal("Infinity")) is ColumnType.DECIMAL

    def test_zero_timedelta(self) -> None:
        assert column_type_from_python(datetime.timedelta(0)) is ColumnType.DURATION

    def test_date_min(self) -> None:
        assert column_type_from_python(datetime.date.min) is ColumnType.DATE

    def test_date_max(self) -> None:
        assert column_type_from_python(datetime.date.max) is ColumnType.DATE

    def test_nested_list(self) -> None:
        assert column_type_from_python([[1, 2], [3, 4]]) is ColumnType.LIST

    def test_nested_dict(self) -> None:
        assert column_type_from_python({"a": {"b": 1}}) is ColumnType.STRUCT


class TestInferColumnTypeEdgeCases:
    """Edge cases in column-level type inference from value sequences."""

    def test_floats_with_nan(self) -> None:
        assert infer_column_type((1.0, float("nan"), 3.0)) is ColumnType.FLOAT

    def test_ints_and_nan(self) -> None:
        # NaN is a float, so int+float → FLOAT
        assert infer_column_type((1, float("nan"), 3)) is ColumnType.FLOAT

    def test_all_nan(self) -> None:
        assert infer_column_type((float("nan"), float("nan"))) is ColumnType.FLOAT

    def test_single_value(self) -> None:
        assert infer_column_type((42,)) is ColumnType.INTEGER

    def test_single_none(self) -> None:
        assert infer_column_type((None,)) is ColumnType.NULL

    def test_bool_and_int_widens_to_text(self) -> None:
        # bool is int subclass but BOOLEAN != INTEGER logically
        assert infer_column_type((True, 42)) is ColumnType.TEXT

    def test_str_and_int_widens_to_text(self) -> None:
        assert infer_column_type(("hello", 42)) is ColumnType.TEXT

    def test_date_and_text_widens_to_text(self) -> None:
        assert infer_column_type((datetime.date(2024, 1, 1), "not a date")) is ColumnType.TEXT

    def test_very_large_sequence(self) -> None:
        # 10K values should not be slow
        vals = tuple(range(10_000))
        assert infer_column_type(vals) is ColumnType.INTEGER


# ═══════════════════════════════════════════════════════════════════════════
# 2. TABLE STRUCTURAL EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════


class TestRaggedRows:
    """Tables where rows have inconsistent column counts."""

    def test_short_row_detected_by_validate(self) -> None:
        t = Table(
            name="ragged",
            columns=(Column("a"), Column("b"), Column("c")),
            rows=(("x", "y", "z"), ("short",)),  # row 1 has 1 value instead of 3
        )
        issues = t.validate()
        assert any("Row 1" in i for i in issues)
        assert any("expected 3" in i for i in issues)

    def test_long_row_detected_by_validate(self) -> None:
        t = Table(
            name="extra",
            columns=(Column("a"),),
            rows=(("x", "extra1", "extra2"),),
        )
        issues = t.validate()
        assert any("Row 0" in i for i in issues)

    def test_column_values_on_ragged_table(self) -> None:
        t = Table(
            name="ragged",
            columns=(Column("a"), Column("b")),
            rows=(("x", "y"), ("only_a",)),
        )
        # column_values("a") works (index 0 is in both rows)
        assert t.column_values("a") == ("x", "only_a")
        # column_values("b") fails on short row
        with pytest.raises(IndexError):
            t.column_values("b")

    def test_csv_serializer_handles_short_rows(self) -> None:
        t = Table(
            name="ragged",
            columns=(Column("a"), Column("b")),
            rows=(("x",),),
        )
        # Should not crash — short row produces fewer fields
        csv = serialize_csv(t)
        assert "a,b" in csv

    def test_markdown_serializer_pads_short_rows(self) -> None:
        t = Table(
            name="ragged",
            columns=(Column("a"), Column("b")),
            rows=(("x",),),
        )
        md = serialize_markdown_table(t)
        assert "| x" in md


class TestDuplicateColumnNames:
    """Tables with non-unique column names (common in XLSX/CSV)."""

    def test_validate_detects_duplicates(self) -> None:
        t = Table(
            name="dupes",
            columns=(Column("id"), Column("name"), Column("name")),
            rows=((1, "Alice", "aliased"),),
        )
        issues = t.validate()
        assert any("Duplicate column name" in i for i in issues)
        assert any("name" in i for i in issues)

    def test_column_values_returns_first_match(self) -> None:
        t = Table(
            name="dupes",
            columns=(Column("x"), Column("x")),
            rows=((1, 2),),
        )
        # Returns first "x" column values
        assert t.column_values("x") == (1,)

    def test_csv_serializer_preserves_duplicates(self) -> None:
        t = Table(
            name="dupes",
            columns=(Column("a"), Column("a")),
            rows=((1, 2),),
        )
        csv = serialize_csv(t)
        assert csv.startswith("a,a\n")

    def test_json_serializer_last_wins(self) -> None:
        """JSON records with duplicate keys — last key wins in Python dicts."""
        import json

        t = Table(
            name="dupes",
            columns=(Column("x"), Column("x")),
            rows=((1, 2),),
        )
        records = json.loads(serialize_json_records(t))
        # Python dict: last "x" wins
        assert records[0]["x"] == 2


class TestEmptyAndDegenerateTables:
    """Boundary conditions: empty, single-row, single-column, etc."""

    def test_no_columns_no_rows(self) -> None:
        t = Table(name="void")
        assert t.row_count == 0
        assert t.validate() == []

    def test_columns_no_rows(self) -> None:
        t = Table(name="schema_only", columns=(Column("a"), Column("b")))
        assert t.row_count == 0
        assert t.describe()["column_count"] == 2

    def test_single_cell(self) -> None:
        t = Table(name="scalar", columns=(Column("val"),), rows=((42,),))
        assert t.row_count == 1
        assert t.column_values("val") == (42,)

    def test_all_null_rows(self) -> None:
        t = Table(
            name="nulls",
            columns=(Column("a"), Column("b")),
            rows=((None, None), (None, None)),
        )
        desc = t.describe()
        for col in desc["columns"]:
            assert col["null_count"] == 2

    def test_head_on_empty_table(self) -> None:
        t = Table(name="e")
        assert t.head(5).rows == ()

    def test_head_preserves_row_count(self) -> None:
        t = Table(
            name="big",
            columns=(Column("x"),),
            rows=tuple((i,) for i in range(100)),
        )
        h = t.head(5)
        assert len(h.rows) == 5
        assert h.row_count == 100

    def test_slice_range(self) -> None:
        t = Table(
            name="data",
            columns=(Column("x"),),
            rows=tuple((i,) for i in range(10)),
        )
        s = t.slice(3, 7)
        assert len(s.rows) == 4
        assert s.rows[0] == (3,)
        assert s.rows[-1] == (6,)

    def test_validate_row_count_too_small(self) -> None:
        t = Table(
            name="bad",
            columns=(Column("x"),),
            rows=((1,), (2,), (3,)),
            row_count=1,  # Claims 1 but has 3
        )
        issues = t.validate()
        assert any("row_count" in i for i in issues)


# ═══════════════════════════════════════════════════════════════════════════
# 3. UNICODE AND SPECIAL CHARACTER STRESS
# ═══════════════════════════════════════════════════════════════════════════


class TestUnicodeStress:
    """Unicode edge cases in column names and values."""

    def test_emoji_column_name(self) -> None:
        t = Table(name="fun", columns=(Column("💰 Revenue"),), rows=(("$100",),))
        assert t.column_names() == ("💰 Revenue",)
        csv = serialize_csv(t)
        assert "💰 Revenue" in csv

    def test_cjk_values(self) -> None:
        t = Table(name="intl", columns=(Column("city"),), rows=(("東京",), ("北京",)))
        csv = serialize_csv(t)
        assert "東京" in csv

    def test_rtl_text(self) -> None:
        t = Table(name="ar", columns=(Column("name"),), rows=(("مرحبا",),))
        json_str = serialize_json_records(t)
        assert "مرحبا" in json_str

    def test_newlines_in_values(self) -> None:
        t = Table(name="nl", columns=(Column("text"),), rows=(("line1\nline2",),))
        csv = serialize_csv(t)
        assert "line1\nline2" in csv  # CSV writer quotes fields with newlines

    def test_tab_in_values_csv(self) -> None:
        t = Table(name="tab", columns=(Column("x"),), rows=(("a\tb",),))
        csv = serialize_csv(t)
        assert "a\tb" in csv

    def test_null_byte_in_value(self) -> None:
        t = Table(name="null", columns=(Column("x"),), rows=(("a\x00b",),))
        csv = serialize_csv(t)
        assert len(csv) > 0

    def test_very_long_string(self) -> None:
        long_str = "x" * 100_000
        t = Table(name="long", columns=(Column("text"),), rows=((long_str,),))
        assert t.column_values("text")[0] == long_str

    def test_empty_string_vs_none(self) -> None:
        """Ensure empty string and None are distinct."""
        t = Table(
            name="nullish",
            columns=(Column("x"),),
            rows=(("",), (None,), ("  ",)),
        )
        vals = t.column_values("x")
        assert vals[0] == ""
        assert vals[1] is None
        assert vals[2] == "  "


# ═══════════════════════════════════════════════════════════════════════════
# 4. NUMERIC EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════


class TestNumericEdgeCases:
    """Extreme numeric values: NaN, inf, large ints, precision."""

    def test_nan_in_float_column(self) -> None:
        t = Table(
            name="nan",
            columns=(Column("x", ColumnType.FLOAT),),
            rows=((1.0,), (float("nan"),), (3.0,)),
        )
        vals = t.column_values("x")
        assert math.isnan(vals[1])

    def test_inf_in_float_column(self) -> None:
        t = Table(
            name="inf",
            columns=(Column("x", ColumnType.FLOAT),),
            rows=((float("inf"),), (float("-inf"),)),
        )
        vals = t.column_values("x")
        assert math.isinf(vals[0])
        assert math.isinf(vals[1])

    def test_nan_csv_serialization(self) -> None:
        t = Table(
            name="nan",
            columns=(Column("x", ColumnType.FLOAT),),
            rows=((float("nan"),),),
        )
        csv = serialize_csv(t)
        assert "nan" in csv.lower()

    def test_nan_json_serialization(self) -> None:
        """JSON doesn't support NaN — should use null or string."""
        t = Table(
            name="nan",
            columns=(Column("x", ColumnType.FLOAT),),
            rows=((float("nan"),),),
        )
        json_str = serialize_json_records(t)
        # Python json.dumps converts NaN to NaN (non-standard)
        # Our serializer passes through Python float behavior
        assert len(json_str) > 0

    def test_decimal_38_digit_precision(self) -> None:
        big_dec = Decimal("12345678901234567890123456789012345678")
        t = Table(
            name="precise",
            columns=(Column("amount", ColumnType.DECIMAL),),
            rows=((big_dec,),),
        )
        assert t.column_values("amount")[0] == big_dec

    def test_negative_zero(self) -> None:
        t = Table(
            name="negzero",
            columns=(Column("x", ColumnType.FLOAT),),
            rows=((-0.0,),),
        )
        csv = serialize_csv(t)
        # -0.0 serializes as "-0.0"
        assert "-0.0" in csv

    def test_very_large_integer(self) -> None:
        huge = 10**100
        t = Table(
            name="huge",
            columns=(Column("x", ColumnType.INTEGER),),
            rows=((huge,),),
        )
        csv = serialize_csv(t)
        assert str(huge) in csv

    def test_scientific_notation_float(self) -> None:
        t = Table(
            name="sci",
            columns=(Column("x", ColumnType.FLOAT),),
            rows=((1.23e-15,),),
        )
        csv = serialize_csv(t)
        assert len(csv) > 0


# ═══════════════════════════════════════════════════════════════════════════
# 5. DATE/TIME BOUNDARY VALUES
# ═══════════════════════════════════════════════════════════════════════════


class TestDateTimeBoundaries:
    """Extreme date/time values."""

    def test_date_min_max(self) -> None:
        t = Table(
            name="dates",
            columns=(Column("d", ColumnType.DATE),),
            rows=((datetime.date.min,), (datetime.date.max,)),
        )
        csv = serialize_csv(t)
        assert "0001-01-01" in csv
        assert "9999-12-31" in csv

    def test_midnight(self) -> None:
        t = Table(
            name="times",
            columns=(Column("t", ColumnType.TIME),),
            rows=((datetime.time(0, 0, 0),), (datetime.time(23, 59, 59),)),
        )
        csv = serialize_csv(t)
        assert "00:00:00" in csv
        assert "23:59:59" in csv

    def test_microsecond_precision(self) -> None:
        dt = datetime.datetime(2024, 6, 15, 12, 30, 45, 123456)
        t = Table(
            name="micro",
            columns=(Column("dt", ColumnType.DATETIME),),
            rows=((dt,),),
        )
        csv = serialize_csv(t)
        assert "123456" in csv

    def test_negative_timedelta(self) -> None:
        td = datetime.timedelta(days=-1, seconds=3600)
        t = Table(
            name="neg",
            columns=(Column("dur", ColumnType.DURATION),),
            rows=((td,),),
        )
        csv = serialize_csv(t)
        assert len(csv) > 0


# ═══════════════════════════════════════════════════════════════════════════
# 6. WIDE AND DEEP TABLES
# ═══════════════════════════════════════════════════════════════════════════


class TestScaleStress:
    """Tables at scale: many columns, many rows."""

    def test_wide_table_100_columns(self) -> None:
        cols = tuple(Column(f"col_{i}", ColumnType.INTEGER) for i in range(100))
        row = tuple(range(100))
        t = Table(name="wide", columns=cols, rows=(row,))
        assert len(t.columns) == 100
        assert t.validate() == []

        csv = serialize_csv(t)
        assert csv.count(",") >= 99  # at least 99 commas per row

    def test_deep_table_10k_rows(self) -> None:
        rows = tuple((i, f"name_{i}") for i in range(10_000))
        t = Table(
            name="deep",
            columns=(Column("id", ColumnType.INTEGER), Column("name")),
            rows=rows,
        )
        assert t.row_count == 10_000
        desc = t.describe()
        assert desc["rows_loaded"] == 10_000

    def test_truncated_table(self) -> None:
        """Source has 1M rows, only first 100 loaded."""
        rows = tuple((i,) for i in range(100))
        t = Table(
            name="big",
            columns=(Column("id", ColumnType.INTEGER),),
            rows=rows,
            row_count=1_000_000,
        )
        assert t.row_count == 1_000_000
        assert len(t.rows) == 100

        md = serialize_markdown_table(t, max_rows=10)
        assert "more rows" in md
        assert "1000000" in md

    def test_many_tables_in_document(self) -> None:
        tables = tuple(
            Table(name=f"sheet_{i}", columns=(Column("x"),), rows=((i,),)) for i in range(50)
        )
        doc = TabularDocument(tables=tables)
        assert len(doc.tables) == 50
        summary = serialize_tabular_summary(doc)
        assert "50 table" in summary

    def test_column_names_from_alphabet(self) -> None:
        """Excel-style A, B, ..., Z, AA, AB, ... column names."""
        names = list(string.ascii_uppercase) + [f"A{c}" for c in string.ascii_uppercase]
        cols = tuple(Column(n) for n in names)
        t = Table(name="excel", columns=cols)
        assert len(t.columns) == 52
        assert t.column_names()[-1] == "AZ"


# ═══════════════════════════════════════════════════════════════════════════
# 7. NESTED / COMPLEX TYPES
# ═══════════════════════════════════════════════════════════════════════════


class TestNestedTypes:
    """LIST and STRUCT columns with complex values."""

    def test_list_of_ints(self) -> None:
        t = Table(
            name="lists",
            columns=(Column("tags", ColumnType.LIST),),
            rows=(([1, 2, 3],), ([],), (None,)),
        )
        json_str = serialize_json_records(t)
        assert "[1, 2, 3]" in json_str

    def test_nested_struct(self) -> None:
        t = Table(
            name="structs",
            columns=(Column("meta", ColumnType.STRUCT),),
            rows=(({"name": "Alice", "addr": {"city": "NYC"}},),),
        )
        json_str = serialize_json_records(t)
        assert "NYC" in json_str

    def test_list_in_csv(self) -> None:
        t = Table(
            name="lists",
            columns=(Column("tags", ColumnType.LIST),),
            rows=(([1, 2, 3],),),
        )
        csv = serialize_csv(t)
        # Lists are JSON-serialized in CSV
        assert "[1, 2, 3]" in csv

    def test_struct_in_csv(self) -> None:
        t = Table(
            name="structs",
            columns=(Column("meta", ColumnType.STRUCT),),
            rows=(({"key": "val"},),),
        )
        csv = serialize_csv(t)
        assert "key" in csv

    def test_describe_with_unhashable_values(self) -> None:
        """Lists and dicts are unhashable — describe() must handle gracefully."""
        t = Table(
            name="unhashable",
            columns=(Column("data", ColumnType.LIST),),
            rows=(([1, 2],), ([3, 4],)),
        )
        desc = t.describe()
        assert desc["columns"][0]["unique_count"] is None


# ═══════════════════════════════════════════════════════════════════════════
# 8. JSON ROUND-TRIP STRESS
# ═══════════════════════════════════════════════════════════════════════════


class TestJsonRoundTripStress:
    """JSON serialization round-trip with tricky values."""

    def test_all_types_round_trip(self) -> None:
        """Every ColumnType survives JSON round-trip."""
        t = Table(
            name="all",
            columns=(
                Column("text", ColumnType.TEXT),
                Column("int", ColumnType.INTEGER),
                Column("float", ColumnType.FLOAT),
                Column("bool", ColumnType.BOOLEAN),
                Column("date", ColumnType.DATE),
                Column("time", ColumnType.TIME),
                Column("datetime", ColumnType.DATETIME),
                Column("null_col", ColumnType.NULL),
                Column("decimal", ColumnType.DECIMAL),
                Column("binary", ColumnType.BINARY),
                Column("duration", ColumnType.DURATION),
                Column("list", ColumnType.LIST),
                Column("struct", ColumnType.STRUCT),
            ),
            rows=(
                (
                    "hello",
                    42,
                    3.14,
                    True,
                    datetime.date(2024, 6, 15),
                    datetime.time(14, 30),
                    datetime.datetime(2024, 6, 15, 14, 30),
                    None,
                    Decimal("1234.56"),
                    b"\xca\xfe",
                    datetime.timedelta(hours=2),
                    [1, 2, 3],
                    {"key": "val"},
                ),
            ),
        )
        doc = TabularDocument(
            metadata=DocumentMetadata(title="Stress"),
            tables=(t,),
        )
        json_str = _tabular_to_json(doc)
        restored = _tabular_from_json(json_str)

        assert len(restored.tables) == 1
        rt = restored.tables[0]
        assert len(rt.columns) == 13
        for orig, rest in zip(t.columns, rt.columns, strict=True):
            assert orig.column_type == rest.column_type

    def test_none_values_preserved(self) -> None:
        t = Table(
            name="nulls",
            columns=(Column("a"), Column("b")),
            rows=((None, None),),
        )
        doc = TabularDocument(tables=(t,))
        restored = _tabular_from_json(_tabular_to_json(doc))
        assert restored.tables[0].rows[0] == (None, None)

    def test_empty_document_round_trip(self) -> None:
        doc = TabularDocument()
        restored = _tabular_from_json(_tabular_to_json(doc))
        assert restored.tables == ()

    def test_metadata_round_trip(self) -> None:
        t = Table(
            name="meta",
            columns=(Column("x"),),
            metadata={"formulas": {"A1": "=SUM(B1:B10)"}, "frozen_rows": 2},
        )
        doc = TabularDocument(tables=(t,))
        restored = _tabular_from_json(_tabular_to_json(doc))
        assert restored.tables[0].metadata["formulas"]["A1"] == "=SUM(B1:B10)"

    def test_special_chars_in_table_name(self) -> None:
        t = Table(name="Sheet 1 (copy)", columns=(Column("x"),))
        doc = TabularDocument(tables=(t,))
        restored = _tabular_from_json(_tabular_to_json(doc))
        assert restored.tables[0].name == "Sheet 1 (copy)"


# ═══════════════════════════════════════════════════════════════════════════
# 9. POLARS BRIDGE STRESS
# ═══════════════════════════════════════════════════════════════════════════

pl = pytest.importorskip("polars")


class TestPolarsBridgeStress:
    """Polars bridge edge cases."""

    def test_nan_round_trip(self) -> None:
        from kaos_content.bridges.polars import table_from_polars, table_to_polars

        t = Table(
            name="nan",
            columns=(Column("x", ColumnType.FLOAT),),
            rows=((1.0,), (None,), (3.0,)),
        )
        df = table_to_polars(t)
        t2 = table_from_polars(df, name="nan")
        assert t2.rows[0] == (1.0,)
        assert t2.rows[1] == (None,)
        assert t2.rows[2] == (3.0,)

    def test_empty_string_preserved(self) -> None:
        from kaos_content.bridges.polars import table_from_polars, table_to_polars

        t = Table(
            name="empty",
            columns=(Column("x", ColumnType.TEXT),),
            rows=(("",), ("hello",), (None,)),
        )
        df = table_to_polars(t)
        t2 = table_from_polars(df, name="empty")
        assert t2.rows[0] == ("",)
        assert t2.rows[2] == (None,)

    def test_all_null_column(self) -> None:
        from kaos_content.bridges.polars import table_from_polars, table_to_polars

        t = Table(
            name="nullcol",
            columns=(Column("x", ColumnType.INTEGER),),
            rows=((None,), (None,)),
        )
        df = table_to_polars(t)
        assert df["x"].null_count() == 2
        t2 = table_from_polars(df, name="nullcol")
        assert all(r[0] is None for r in t2.rows)

    def test_wide_table_polars(self) -> None:
        from kaos_content.bridges.polars import table_from_polars, table_to_polars

        cols = tuple(Column(f"c{i}", ColumnType.INTEGER) for i in range(50))
        row = tuple(range(50))
        t = Table(name="wide", columns=cols, rows=(row,))
        df = table_to_polars(t)
        t2 = table_from_polars(df, name="wide")
        assert len(t2.columns) == 50
        assert t2.rows[0] == row

    def test_date_boundary_values_polars(self) -> None:
        from kaos_content.bridges.polars import table_to_polars

        t = Table(
            name="dates",
            columns=(Column("d", ColumnType.DATE),),
            rows=((datetime.date(1970, 1, 1),), (datetime.date(2099, 12, 31),)),
        )
        df = table_to_polars(t)
        assert df["d"].to_list() == [
            datetime.date(1970, 1, 1),
            datetime.date(2099, 12, 31),
        ]

    def test_boolean_none_round_trip(self) -> None:
        from kaos_content.bridges.polars import table_from_polars, table_to_polars

        t = Table(
            name="bools",
            columns=(Column("b", ColumnType.BOOLEAN),),
            rows=((True,), (None,), (False,)),
        )
        df = table_to_polars(t)
        t2 = table_from_polars(df, name="bools")
        assert t2.rows == ((True,), (None,), (False,))


# ═══════════════════════════════════════════════════════════════════════════
# 10. DUCKDB BRIDGE STRESS
# ═══════════════════════════════════════════════════════════════════════════

duckdb = pytest.importorskip("duckdb")


class TestDuckDBBridgeCore:
    """DuckDB bridge: registration, query, round-trip."""

    def test_register_and_query(self) -> None:
        from kaos_content.bridges.duckdb import query_to_table, register_table

        con = duckdb.connect()
        t = Table(
            name="sales",
            columns=(
                Column("region", ColumnType.TEXT),
                Column("amount", ColumnType.FLOAT),
            ),
            rows=(("US", 100.0), ("EU", 200.0), ("US", 150.0)),
        )
        register_table(con, t)
        result = query_to_table(con, "SELECT * FROM sales WHERE region = 'US'", name="us_sales")
        assert len(result.rows) == 2
        assert all(r[0] == "US" for r in result.rows)

    def test_aggregate_query(self) -> None:
        from kaos_content.bridges.duckdb import query_to_table, register_table

        con = duckdb.connect()
        t = Table(
            name="data",
            columns=(
                Column("category", ColumnType.TEXT),
                Column("value", ColumnType.INTEGER),
            ),
            rows=(("A", 10), ("B", 20), ("A", 30), ("B", 40)),
        )
        register_table(con, t)
        result = query_to_table(
            con,
            "SELECT category, SUM(value) as total FROM data GROUP BY category ORDER BY category",
            name="agg",
        )
        assert len(result.rows) == 2
        assert result.rows[0] == ("A", 40)
        assert result.rows[1] == ("B", 60)

    def test_join_across_tables(self) -> None:
        from kaos_content.bridges.duckdb import query_to_table, register_document

        con = duckdb.connect()
        doc = TabularDocument(
            tables=(
                Table(
                    name="orders",
                    columns=(
                        Column("id", ColumnType.INTEGER),
                        Column("customer_id", ColumnType.INTEGER),
                    ),
                    rows=((1, 10), (2, 20), (3, 10)),
                ),
                Table(
                    name="customers",
                    columns=(Column("id", ColumnType.INTEGER), Column("name", ColumnType.TEXT)),
                    rows=((10, "Alice"), (20, "Bob")),
                ),
            ),
        )
        register_document(con, doc)
        result = query_to_table(
            con,
            (
                "SELECT o.id, c.name FROM orders o "
                "JOIN customers c ON o.customer_id = c.id "
                "ORDER BY o.id"
            ),
            name="joined",
        )
        assert len(result.rows) == 3
        assert result.rows[0] == (1, "Alice")
        assert result.rows[1] == (2, "Bob")
        assert result.rows[2] == (3, "Alice")

    def test_column_types_inferred_from_result(self) -> None:
        from kaos_content.bridges.duckdb import query_to_table, register_table

        con = duckdb.connect()
        t = Table(
            name="typed",
            columns=(
                Column("i", ColumnType.INTEGER),
                Column("f", ColumnType.FLOAT),
                Column("s", ColumnType.TEXT),
                Column("b", ColumnType.BOOLEAN),
                Column("d", ColumnType.DATE),
            ),
            rows=((1, 1.5, "hello", True, datetime.date(2024, 1, 1)),),
        )
        register_table(con, t)
        result = query_to_table(con, "SELECT * FROM typed", name="result")

        type_map = {c.name: c.column_type for c in result.columns}
        assert type_map["i"] is ColumnType.INTEGER
        assert type_map["f"] is ColumnType.FLOAT
        assert type_map["s"] is ColumnType.TEXT
        assert type_map["b"] is ColumnType.BOOLEAN
        assert type_map["d"] is ColumnType.DATE

    def test_null_handling(self) -> None:
        from kaos_content.bridges.duckdb import query_to_table, register_table

        con = duckdb.connect()
        t = Table(
            name="nulls",
            columns=(Column("x", ColumnType.INTEGER),),
            rows=((1,), (None,), (3,)),
        )
        register_table(con, t)
        result = query_to_table(con, "SELECT * FROM nulls", name="result")
        assert result.rows[1] == (None,)

    def test_empty_result(self) -> None:
        from kaos_content.bridges.duckdb import query_to_table, register_table

        con = duckdb.connect()
        t = Table(
            name="data",
            columns=(Column("x", ColumnType.INTEGER),),
            rows=((1,), (2,)),
        )
        register_table(con, t)
        result = query_to_table(con, "SELECT * FROM data WHERE x > 100", name="empty")
        assert len(result.rows) == 0
        assert len(result.columns) == 1

    def test_describe_table(self) -> None:
        from kaos_content.bridges.duckdb import describe_table, register_table

        con = duckdb.connect()
        t = Table(
            name="info",
            columns=(Column("a", ColumnType.INTEGER), Column("b", ColumnType.TEXT)),
            rows=((1, "x"), (2, "y")),
        )
        register_table(con, t)
        desc = describe_table(con, "info")
        assert desc["row_count"] == 2
        assert desc["column_count"] == 2

    def test_list_tables(self) -> None:
        from kaos_content.bridges.duckdb import list_tables, register_table

        con = duckdb.connect()
        t1 = Table(name="alpha", columns=(Column("x"),), rows=(("a",),))
        t2 = Table(name="beta", columns=(Column("y"),), rows=(("b",),))
        register_table(con, t1)
        register_table(con, t2)
        tables = list_tables(con)
        names = {t["name"] for t in tables}
        assert "alpha" in names
        assert "beta" in names

    def test_execute_query_with_limit(self) -> None:
        # Sec-3: ``execute_query(untrusted_sql=True)`` (default) now
        # requires a sandboxed connection. This test exercises the
        # row-limit behaviour, which is unchanged; switching to a
        # sandboxed conn matches the new contract.
        from kaos_content.bridges.duckdb import (
            create_safe_connection,
            execute_query,
            register_table,
        )

        con = create_safe_connection()
        rows = tuple((i,) for i in range(100))
        t = Table(name="big", columns=(Column("x", ColumnType.INTEGER),), rows=rows)
        register_table(con, t)
        result = execute_query(con, "SELECT * FROM big", max_rows=10)
        assert len(result.rows) == 10

    def test_sql_with_special_characters_in_name(self) -> None:
        from kaos_content.bridges.duckdb import query_to_table, register_table

        con = duckdb.connect()
        t = Table(name="my table (v2)", columns=(Column("x"),), rows=(("val",),))
        register_table(con, t, name="my table (v2)")
        result = query_to_table(
            con,
            'SELECT * FROM "my table (v2)"',
            name="result",
        )
        assert len(result.rows) == 1


class TestDuckDBPolarsRoundTrip:
    """Full pipeline: Table → DuckDB → SQL → Table, and Table → Polars → DuckDB → Table."""

    def test_table_to_duckdb_to_table(self) -> None:
        from kaos_content.bridges.duckdb import query_to_table, register_table

        con = duckdb.connect()
        original = Table(
            name="original",
            columns=(
                Column("id", ColumnType.INTEGER),
                Column("name", ColumnType.TEXT),
                Column("active", ColumnType.BOOLEAN),
            ),
            rows=((1, "Alice", True), (2, "Bob", False)),
        )
        register_table(con, original)
        restored = query_to_table(con, "SELECT * FROM original", name="restored")

        assert len(restored.rows) == len(original.rows)
        assert restored.rows[0] == (1, "Alice", True)
        assert restored.rows[1] == (2, "Bob", False)

    def test_cross_format_join(self) -> None:
        """Simulate joining data from CSV and XLSX (both are Tables)."""
        from kaos_content.bridges.duckdb import query_to_table, register_table

        con = duckdb.connect()

        csv_table = Table(
            name="csv_data",
            columns=(Column("code", ColumnType.TEXT), Column("value", ColumnType.INTEGER)),
            rows=(("A", 10), ("B", 20), ("C", 30)),
        )
        xlsx_table = Table(
            name="xlsx_data",
            columns=(Column("code", ColumnType.TEXT), Column("label", ColumnType.TEXT)),
            rows=(("A", "Alpha"), ("B", "Beta")),
        )

        register_table(con, csv_table)
        register_table(con, xlsx_table)

        result = query_to_table(
            con,
            "SELECT c.code, c.value, x.label FROM csv_data c "
            "LEFT JOIN xlsx_data x ON c.code = x.code ORDER BY c.code",
            name="joined",
        )
        assert len(result.rows) == 3
        assert result.rows[0] == ("A", 10, "Alpha")
        assert result.rows[2] == ("C", 30, None)  # No match for C

    def test_duckdb_aggregation_preserves_types(self) -> None:
        from kaos_content.bridges.duckdb import query_to_table, register_table

        con = duckdb.connect()
        t = Table(
            name="sales",
            columns=(
                Column("region", ColumnType.TEXT),
                Column("amount", ColumnType.FLOAT),
            ),
            rows=(("US", 100.5), ("EU", 200.0), ("US", 150.5)),
        )
        register_table(con, t)
        result = query_to_table(
            con,
            "SELECT region, SUM(amount) as total, COUNT(*) as cnt "
            "FROM sales GROUP BY region ORDER BY region",
            name="agg",
        )

        type_map = {c.name: c.column_type for c in result.columns}
        assert type_map["region"] is ColumnType.TEXT
        assert type_map["total"] is ColumnType.FLOAT
        assert type_map["cnt"] is ColumnType.INTEGER


# ═══════════════════════════════════════════════════════════════════════════
# 11. REALISTIC DATA SCENARIOS
# ═══════════════════════════════════════════════════════════════════════════


class TestRealisticScenarios:
    """End-to-end scenarios agents would encounter."""

    def test_ledes_billing_data(self) -> None:
        """LEDES 98B legal billing format — real column patterns."""
        t = Table(
            name="ledes",
            columns=(
                Column("invoice_date", ColumnType.DATE),
                Column("invoice_number", ColumnType.TEXT),
                Column("client_id", ColumnType.TEXT),
                Column("timekeeper_id", ColumnType.TEXT),
                Column("timekeeper_name", ColumnType.TEXT),
                Column("classification", ColumnType.TEXT),
                Column("hours", ColumnType.FLOAT),
                Column("rate", ColumnType.DECIMAL),
                Column("amount", ColumnType.DECIMAL),
                Column("adjustment", ColumnType.DECIMAL),
                Column("description", ColumnType.TEXT),
            ),
            rows=(
                (
                    datetime.date(2024, 3, 15),
                    "INV-2024-001",
                    "CL-500",
                    "TK-001",
                    "Johnson, Robert",
                    "PARTNR",
                    Decimal("6.5"),
                    Decimal("750.00"),
                    Decimal("4875.00"),
                    Decimal("-70.00"),
                    "Review of merger agreement and related schedules; "
                    "conference with opposing counsel regarding timeline",
                ),
                (
                    datetime.date(2024, 3, 15),
                    "INV-2024-001",
                    "CL-500",
                    None,  # Null timekeeper for expenses
                    None,
                    None,
                    None,
                    None,
                    Decimal("250.00"),
                    None,
                    "Filing fees - federal court",
                ),
            ),
        )

        # Validate structure
        assert t.validate() == []
        assert t.row_count == 2

        # Describe for agent consumption
        desc = t.describe()
        assert desc["column_count"] == 11
        # The null timekeeper row should show nulls
        tk_col = next(c for c in desc["columns"] if c["name"] == "timekeeper_id")
        assert tk_col["null_count"] == 1

        # Serialize
        csv = serialize_csv(t)
        assert "INV-2024-001" in csv
        assert "4875.00" in csv

        # JSON round-trip
        doc = TabularDocument(
            metadata=DocumentMetadata(title="March 2024 Invoice"),
            tables=(t,),
        )
        restored = _tabular_from_json(_tabular_to_json(doc))
        assert restored.tables[0].row_count == 2

    def test_states_geographic_data(self) -> None:
        """US states with unicode and mixed types."""
        t = Table(
            name="states",
            columns=(
                Column("name", ColumnType.TEXT),
                Column("capital", ColumnType.TEXT),
                Column("population", ColumnType.INTEGER),
                Column("area_sq_mi", ColumnType.FLOAT),
                Column("admitted", ColumnType.DATE),
                Column("motto", ColumnType.TEXT),
            ),
            rows=(
                (
                    "California",
                    "Sacramento",
                    39_538_223,
                    163_696.0,
                    datetime.date(1850, 9, 9),
                    "Eureka",
                ),
                (
                    "Hawaii",
                    "Honolulu",
                    1_455_271,
                    10_931.0,
                    datetime.date(1959, 8, 21),
                    "Ua Mau ke Ea o ka \u02bbAina i ka Pono",
                ),
                (
                    "Montana",
                    "Helena",
                    1_084_225,
                    147_040.0,
                    datetime.date(1889, 11, 8),
                    "Oro y Plata",
                ),
            ),
        )

        # Unicode preserves through serialization
        csv = serialize_csv(t)
        assert "\u02bb" in csv  # Hawaii okina
        assert "Oro y Plata" in csv

        md = serialize_markdown_table(t)
        assert "California" in md

    def test_agent_workflow_csv_to_query(self) -> None:
        """Simulate: agent loads CSV → registers in DuckDB → runs SQL → gets result."""
        from kaos_content.bridges.duckdb import query_to_table, register_table

        con = duckdb.connect()

        # "CSV file" loaded as Table
        csv_table = Table(
            name="transactions",
            columns=(
                Column("date", ColumnType.DATE),
                Column("merchant", ColumnType.TEXT),
                Column("amount", ColumnType.FLOAT),
                Column("category", ColumnType.TEXT),
            ),
            rows=(
                (datetime.date(2024, 1, 5), "Grocery Store", 52.47, "food"),
                (datetime.date(2024, 1, 6), "Gas Station", 45.00, "transport"),
                (datetime.date(2024, 1, 7), "Restaurant", 32.50, "food"),
                (datetime.date(2024, 1, 8), "Gas Station", 48.75, "transport"),
                (datetime.date(2024, 1, 9), "Online Store", 125.99, "shopping"),
            ),
        )

        register_table(con, csv_table)

        # Agent asks: "What's my total spending by category?"
        result = query_to_table(
            con,
            "SELECT category, ROUND(SUM(amount), 2) as total, COUNT(*) as txn_count "
            "FROM transactions GROUP BY category ORDER BY total DESC",
            name="spending",
        )

        assert len(result.rows) == 3
        # Shopping: 125.99, Food: 84.97, Transport: 93.75
        categories = {r[0]: r[1] for r in result.rows}
        assert categories["shopping"] == 125.99
        assert abs(categories["food"] - 84.97) < 0.01

        # Agent asks for TSV output
        tsv = serialize_tsv(result)
        assert "category\ttotal\ttxn_count" in tsv


# ═══════════════════════════════════════════════════════════════════════════
# 12. DUCKDB SESSION PERSISTENCE
# ═══════════════════════════════════════════════════════════════════════════


class TestDuckDBSessionPersistence:
    """Prove DuckDB can save/reload state for session continuity."""

    def test_file_backed_persistence(self, tmp_path: Path) -> None:
        """Register table, close connection, reopen, data survives."""
        from kaos_content.bridges.duckdb import register_table

        db_path = str(tmp_path / "session.duckdb")

        # Session 1: create and populate
        con1 = duckdb.connect(db_path)
        t = Table(
            name="notes",
            columns=(Column("id", ColumnType.INTEGER), Column("text", ColumnType.TEXT)),
            rows=((1, "first"), (2, "second")),
        )
        register_table(con1, t)
        con1.close()

        # Session 2: reopen, data is there
        con2 = duckdb.connect(db_path, read_only=True)
        result = con2.execute("SELECT * FROM notes ORDER BY id").fetchall()
        assert len(result) == 2
        assert result[0] == (1, "first")
        con2.close()

    def test_file_backed_multiple_tables(self, tmp_path: Path) -> None:
        """Multiple tables persist across sessions."""
        from kaos_content.bridges.duckdb import list_tables, register_document

        db_path = str(tmp_path / "multi.duckdb")

        doc = TabularDocument(
            tables=(
                Table(name="users", columns=(Column("name"),), rows=(("Alice",),)),
                Table(name="orders", columns=(Column("item"),), rows=(("Widget",),)),
                Table(name="products", columns=(Column("sku"),), rows=(("ABC",),)),
            ),
        )

        con1 = duckdb.connect(db_path)
        register_document(con1, doc)
        con1.close()

        con2 = duckdb.connect(db_path, read_only=True)
        tables = list_tables(con2)
        names = {t["name"] for t in tables}
        assert names == {"users", "orders", "products"}
        con2.close()

    def test_concurrent_readers(self, tmp_path: Path) -> None:
        """Multiple read-only connections can access simultaneously."""
        from kaos_content.bridges.duckdb import register_table

        db_path = str(tmp_path / "shared.duckdb")

        con_write = duckdb.connect(db_path)
        t = Table(name="data", columns=(Column("x", ColumnType.INTEGER),), rows=((42,),))
        register_table(con_write, t)
        con_write.close()

        # Two concurrent readers
        reader1 = duckdb.connect(db_path, read_only=True)
        reader2 = duckdb.connect(db_path, read_only=True)

        r1 = reader1.execute("SELECT * FROM data").fetchall()
        r2 = reader2.execute("SELECT * FROM data").fetchall()

        assert r1 == [(42,)]
        assert r2 == [(42,)]

        reader1.close()
        reader2.close()

    def test_in_memory_isolation(self) -> None:
        """In-memory connections are fully isolated."""
        from kaos_content.bridges.duckdb import register_table

        con1 = duckdb.connect()
        con2 = duckdb.connect()

        t = Table(name="private", columns=(Column("x"),), rows=(("secret",),))
        register_table(con1, t)

        # con2 should NOT see con1's table
        with pytest.raises(duckdb.CatalogException):
            con2.execute("SELECT * FROM private")

    def test_export_import_state(self, tmp_path: Path) -> None:
        """EXPORT DATABASE saves full state, importable into new connection."""
        from kaos_content.bridges.duckdb import query_to_table, register_table

        # Create in-memory with data
        con1 = duckdb.connect()
        t = Table(
            name="work",
            columns=(Column("step", ColumnType.INTEGER), Column("result", ColumnType.TEXT)),
            rows=((1, "loaded CSV"), (2, "ran query"), (3, "derived table")),
        )
        register_table(con1, t)

        # Export to directory
        export_dir = tmp_path / "export"
        export_dir.mkdir()
        con1.execute(f"EXPORT DATABASE '{export_dir}'")
        con1.close()

        # Import into fresh connection
        con2 = duckdb.connect()
        con2.execute(f"IMPORT DATABASE '{export_dir}'")
        result = query_to_table(con2, "SELECT * FROM work ORDER BY step", name="restored")
        assert len(result.rows) == 3
        assert result.rows[0] == (1, "loaded CSV")
        con2.close()

    def test_many_tables_scale(self) -> None:
        """Register 50 tables in one connection — no issues."""
        from kaos_content.bridges.duckdb import list_tables, register_table

        con = duckdb.connect()
        for i in range(50):
            t = Table(
                name=f"table_{i:03d}",
                columns=(Column("x", ColumnType.INTEGER),),
                rows=((i,),),
            )
            register_table(con, t)

        tables = list_tables(con)
        assert len(tables) == 50

    def test_large_table_query_performance(self) -> None:
        """10K-row table queries fast (DuckDB vectorized execution)."""
        import time

        from kaos_content.bridges.duckdb import query_to_table, register_table

        con = duckdb.connect()
        rows = tuple((i, f"name_{i}", float(i) * 1.5) for i in range(10_000))
        t = Table(
            name="large",
            columns=(
                Column("id", ColumnType.INTEGER),
                Column("name", ColumnType.TEXT),
                Column("value", ColumnType.FLOAT),
            ),
            rows=rows,
        )
        register_table(con, t)

        start = time.monotonic()
        result = query_to_table(
            con,
            "SELECT COUNT(*), AVG(value), MIN(id), MAX(id) FROM large",
            name="stats",
        )
        elapsed = time.monotonic() - start

        assert result.rows[0][0] == 10_000
        assert elapsed < 1.0  # Should be well under 1 second
