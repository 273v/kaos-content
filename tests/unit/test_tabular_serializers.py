"""Tests for tabular serializers: CSV, TSV, markdown, JSON, summary."""

from __future__ import annotations

import datetime
import json
from decimal import Decimal

import pytest

from kaos_content.model.metadata import DocumentMetadata
from kaos_content.model.tabular import (
    Column,
    ColumnType,
    Table,
    TabularDocument,
)
from kaos_content.serializers.tabular import (
    serialize_csv,
    serialize_json_records,
    serialize_markdown_table,
    serialize_tabular_summary,
    serialize_tsv,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_table() -> Table:
    return Table(
        name="data",
        columns=(
            Column("name", ColumnType.TEXT),
            Column("age", ColumnType.INTEGER),
            Column("score", ColumnType.FLOAT),
        ),
        rows=(
            ("Alice", 30, 95.5),
            ("Bob", 25, 87.0),
            ("Charlie", 35, 92.3),
        ),
    )


@pytest.fixture()
def typed_table() -> Table:
    """Table with diverse types for formatting tests."""
    return Table(
        name="typed",
        columns=(
            Column("date", ColumnType.DATE),
            Column("time", ColumnType.TIME),
            Column("dt", ColumnType.DATETIME),
            Column("amount", ColumnType.DECIMAL),
            Column("active", ColumnType.BOOLEAN),
            Column("data", ColumnType.BINARY),
            Column("dur", ColumnType.DURATION),
            Column("tags", ColumnType.LIST),
            Column("meta", ColumnType.STRUCT),
            Column("empty", ColumnType.NULL),
        ),
        rows=(
            (
                datetime.date(2024, 6, 15),
                datetime.time(14, 30, 0),
                datetime.datetime(2024, 6, 15, 14, 30),
                Decimal("1234.56"),
                True,
                b"\xde\xad",
                datetime.timedelta(hours=2, minutes=30),
                ["a", "b"],
                {"key": "val"},
                None,
            ),
        ),
    )


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------


class TestSerializeCSV:
    def test_basic(self, simple_table: Table) -> None:
        result = serialize_csv(simple_table)
        lines = result.strip().split("\n")
        assert lines[0] == "name,age,score"
        assert lines[1] == "Alice,30,95.5"
        assert len(lines) == 4  # header + 3 rows

    def test_custom_delimiter(self, simple_table: Table) -> None:
        result = serialize_csv(simple_table, delimiter="|")
        assert "Alice|30|95.5" in result

    def test_empty_table(self) -> None:
        t = Table(name="e", columns=(Column("a"),))
        result = serialize_csv(t)
        assert result.strip() == "a"

    def test_none_values(self) -> None:
        t = Table(
            name="n",
            columns=(Column("x"),),
            rows=((None,), ("val",)),
        )
        result = serialize_csv(t)
        lines = result.strip().split("\n")
        # csv.writer may quote the empty string
        assert lines[1] in ("", '""')  # None formatted as empty
        assert lines[2] == "val"

    def test_values_with_commas_quoted(self) -> None:
        t = Table(
            name="q",
            columns=(Column("x"),),
            rows=(("hello, world",),),
        )
        result = serialize_csv(t)
        assert '"hello, world"' in result


# ---------------------------------------------------------------------------
# TSV
# ---------------------------------------------------------------------------


class TestSerializeTSV:
    def test_basic(self, simple_table: Table) -> None:
        result = serialize_tsv(simple_table)
        lines = result.strip().split("\n")
        assert "name\tage\tscore" in lines[0]
        assert "Alice\t30\t95.5" in lines[1]

    def test_tab_delimiter(self, simple_table: Table) -> None:
        result = serialize_tsv(simple_table)
        assert "\t" in result
        assert "," not in result.split("\n")[1]  # no commas in data rows


# ---------------------------------------------------------------------------
# Markdown table
# ---------------------------------------------------------------------------


class TestSerializeMarkdownTable:
    def test_basic_structure(self, simple_table: Table) -> None:
        result = serialize_markdown_table(simple_table)
        lines = result.strip().split("\n")
        assert lines[0].startswith("| name")
        assert lines[1].startswith("| ---")
        assert "Alice" in lines[2]
        assert len(lines) == 5  # header + sep + 3 rows

    def test_max_rows_truncation(self) -> None:
        rows = tuple((f"row{i}",) for i in range(100))
        t = Table(name="big", columns=(Column("x"),), rows=rows)
        result = serialize_markdown_table(t, max_rows=5)
        assert "95 more rows" in result
        assert "total: 100" in result

    def test_empty_columns(self) -> None:
        t = Table(name="e")
        result = serialize_markdown_table(t)
        assert "Empty table" in result

    def test_alignment(self, simple_table: Table) -> None:
        result = serialize_markdown_table(simple_table)
        # Check pipe-delimited format
        for line in result.strip().split("\n"):
            if line.startswith("|"):
                assert line.endswith("|")

    def test_max_rows_zero_no_limit(self) -> None:
        rows = tuple((f"row{i}",) for i in range(200))
        t = Table(name="big", columns=(Column("x"),), rows=rows)
        result = serialize_markdown_table(t, max_rows=0)
        assert "more rows" not in result
        lines = [line for line in result.strip().split("\n") if line.startswith("|")]
        assert len(lines) == 202  # header + sep + 200 rows


# ---------------------------------------------------------------------------
# JSON records
# ---------------------------------------------------------------------------


class TestSerializeJsonRecords:
    def test_basic(self, simple_table: Table) -> None:
        result = serialize_json_records(simple_table)
        records = json.loads(result)
        assert len(records) == 3
        assert records[0]["name"] == "Alice"
        assert records[0]["age"] == 30
        assert records[0]["score"] == 95.5

    def test_none_preserved(self) -> None:
        t = Table(
            name="n",
            columns=(Column("x"), Column("y")),
            rows=((1, None),),
        )
        records = json.loads(serialize_json_records(t))
        assert records[0]["y"] is None

    def test_typed_values(self, typed_table: Table) -> None:
        result = serialize_json_records(typed_table)
        records = json.loads(result)
        r = records[0]
        assert r["date"] == "2024-06-15"
        assert r["time"] == "14:30:00"
        assert r["dt"] == "2024-06-15T14:30:00"
        assert r["amount"] == "1234.56"
        assert r["active"] is True
        assert r["data"] == "dead"
        assert r["tags"] == ["a", "b"]
        assert r["meta"] == {"key": "val"}
        assert r["empty"] is None

    def test_indent(self, simple_table: Table) -> None:
        result = serialize_json_records(simple_table, indent=2)
        assert "\n" in result
        assert "  " in result

    def test_empty_table(self) -> None:
        t = Table(name="e", columns=(Column("a"),))
        result = serialize_json_records(t)
        assert json.loads(result) == []


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


class TestSerializeTabularSummary:
    def test_single_table(self, simple_table: Table) -> None:
        doc = TabularDocument(
            metadata=DocumentMetadata(title="Test"),
            tables=(simple_table,),
        )
        result = serialize_tabular_summary(doc)
        assert "Test" in result
        assert "1 table" in result
        assert "3 total rows" in result
        assert "name (text)" in result
        assert "age (integer)" in result

    def test_multi_table(self) -> None:
        doc = TabularDocument(
            metadata=DocumentMetadata(title="Multi"),
            tables=(
                Table(name="t1", columns=(Column("a"),), rows=(("x",),)),
                Table(name="t2", columns=(Column("b"),), rows=(("y",), ("z",))),
            ),
        )
        result = serialize_tabular_summary(doc)
        assert "2 table" in result
        assert "3 total rows" in result
        assert "t1:" in result
        assert "t2:" in result

    def test_untitled(self) -> None:
        doc = TabularDocument(tables=(Table(name="t"),))
        result = serialize_tabular_summary(doc)
        assert "Untitled" in result


# ---------------------------------------------------------------------------
# Type formatting edge cases
# ---------------------------------------------------------------------------


class TestValueFormatting:
    def test_boolean_lowercase(self) -> None:
        t = Table(
            name="b",
            columns=(Column("x", ColumnType.BOOLEAN),),
            rows=((True,), (False,)),
        )
        csv = serialize_csv(t)
        assert "true" in csv
        assert "false" in csv

    def test_timedelta_formatting(self) -> None:
        t = Table(
            name="d",
            columns=(Column("dur", ColumnType.DURATION),),
            rows=((datetime.timedelta(hours=1, minutes=30, seconds=45),),),
        )
        csv = serialize_csv(t)
        assert "1:30:45" in csv

    def test_negative_timedelta(self) -> None:
        t = Table(
            name="d",
            columns=(Column("dur", ColumnType.DURATION),),
            rows=((datetime.timedelta(seconds=-3661),),),
        )
        csv = serialize_csv(t)
        assert "-1:01:01" in csv

    def test_bytes_hex(self) -> None:
        t = Table(
            name="b",
            columns=(Column("data", ColumnType.BINARY),),
            rows=((b"\xca\xfe",),),
        )
        csv = serialize_csv(t)
        assert "cafe" in csv
