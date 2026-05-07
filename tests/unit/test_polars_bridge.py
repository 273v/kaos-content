"""Tests for the Polars bridge: Table ↔ DataFrame round-trip.

Requires polars to be installed. Tests are skipped if polars is not available.
"""

from __future__ import annotations

import datetime

import pytest

from kaos_content.model.metadata import DocumentMetadata
from kaos_content.model.tabular import (
    Column,
    ColumnType,
    Table,
    TabularDocument,
)

pl = pytest.importorskip("polars")

from kaos_content.bridges.polars import (  # noqa: E402
    document_to_polars,
    table_from_polars,
    table_to_polars,
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
            Column("active", ColumnType.BOOLEAN),
        ),
        rows=(
            ("Alice", 30, 95.5, True),
            ("Bob", 25, 87.0, False),
            ("Charlie", 35, 92.3, True),
        ),
    )


# ---------------------------------------------------------------------------
# table_to_polars
# ---------------------------------------------------------------------------


class TestTableToPolars:
    def test_basic_conversion(self, simple_table: Table) -> None:
        df = table_to_polars(simple_table)
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 3
        assert df.columns == ["name", "age", "score", "active"]

    def test_column_dtypes(self, simple_table: Table) -> None:
        df = table_to_polars(simple_table)
        assert df["name"].dtype == pl.String
        assert df["age"].dtype == pl.Int64
        assert df["score"].dtype == pl.Float64
        assert df["active"].dtype == pl.Boolean

    def test_values_preserved(self, simple_table: Table) -> None:
        df = table_to_polars(simple_table)
        assert df["name"].to_list() == ["Alice", "Bob", "Charlie"]
        assert df["age"].to_list() == [30, 25, 35]

    def test_date_column(self) -> None:
        t = Table(
            name="dates",
            columns=(Column("d", ColumnType.DATE),),
            rows=(
                (datetime.date(2024, 1, 1),),
                (datetime.date(2024, 6, 15),),
            ),
        )
        df = table_to_polars(t)
        assert df["d"].dtype == pl.Date
        assert df["d"].to_list() == [
            datetime.date(2024, 1, 1),
            datetime.date(2024, 6, 15),
        ]

    def test_datetime_column(self) -> None:
        t = Table(
            name="dts",
            columns=(Column("dt", ColumnType.DATETIME),),
            rows=((datetime.datetime(2024, 1, 1, 12, 30),),),
        )
        df = table_to_polars(t)
        assert df["dt"].dtype.base_type() == pl.Datetime

    def test_none_values(self) -> None:
        t = Table(
            name="nulls",
            columns=(Column("x", ColumnType.INTEGER),),
            rows=((1,), (None,), (3,)),
        )
        df = table_to_polars(t)
        assert df["x"].null_count() == 1
        assert df["x"].to_list() == [1, None, 3]

    def test_empty_table(self) -> None:
        t = Table(name="e", columns=(Column("x", ColumnType.TEXT),))
        df = table_to_polars(t)
        assert len(df) == 0
        assert df.columns == ["x"]

    def test_no_columns(self) -> None:
        t = Table(name="e")
        df = table_to_polars(t)
        assert len(df) == 0

    def test_binary_column(self) -> None:
        t = Table(
            name="b",
            columns=(Column("data", ColumnType.BINARY),),
            rows=((b"\xca\xfe",), (b"\xde\xad",)),
        )
        df = table_to_polars(t)
        assert df["data"].dtype == pl.Binary


# ---------------------------------------------------------------------------
# table_from_polars
# ---------------------------------------------------------------------------


class TestTableFromPolars:
    def test_basic_conversion(self) -> None:
        df = pl.DataFrame(
            {
                "name": ["Alice", "Bob"],
                "age": [30, 25],
                "score": [95.5, 87.0],
            }
        )
        t = table_from_polars(df, name="people")
        assert t.name == "people"
        assert len(t.columns) == 3
        assert len(t.rows) == 2

    def test_column_types_inferred(self) -> None:
        df = pl.DataFrame(
            {
                "s": ["a", "b"],
                "i": [1, 2],
                "f": [1.0, 2.0],
                "b": [True, False],
            }
        )
        t = table_from_polars(df)
        types = {c.name: c.column_type for c in t.columns}
        assert types["s"] is ColumnType.TEXT
        assert types["i"] is ColumnType.INTEGER
        assert types["f"] is ColumnType.FLOAT
        assert types["b"] is ColumnType.BOOLEAN

    def test_date_type_preserved(self) -> None:
        df = pl.DataFrame(
            {
                "d": [datetime.date(2024, 1, 1), datetime.date(2024, 6, 15)],
            }
        )
        t = table_from_polars(df)
        assert t.columns[0].column_type is ColumnType.DATE

    def test_values_preserved(self) -> None:
        df = pl.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
        t = table_from_polars(df)
        assert t.rows == ((1, "a"), (2, "b"), (3, "c"))


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_table_round_trip(self, simple_table: Table) -> None:
        df = table_to_polars(simple_table)
        t2 = table_from_polars(df, name=simple_table.name)

        assert t2.name == simple_table.name
        assert len(t2.columns) == len(simple_table.columns)
        assert len(t2.rows) == len(simple_table.rows)

        for orig, restored in zip(simple_table.columns, t2.columns, strict=True):
            assert orig.name == restored.name
            assert orig.column_type == restored.column_type

    def test_int_values_round_trip(self) -> None:
        t = Table(
            name="ints",
            columns=(Column("x", ColumnType.INTEGER),),
            rows=((0,), (42,), (-100,)),
        )
        df = table_to_polars(t)
        t2 = table_from_polars(df, name="ints")
        assert t2.rows == ((0,), (42,), (-100,))

    def test_bool_values_round_trip(self) -> None:
        t = Table(
            name="bools",
            columns=(Column("x", ColumnType.BOOLEAN),),
            rows=((True,), (False,), (None,)),
        )
        df = table_to_polars(t)
        t2 = table_from_polars(df, name="bools")
        assert t2.rows[0] == (True,)
        assert t2.rows[1] == (False,)
        assert t2.rows[2] == (None,)


# ---------------------------------------------------------------------------
# document_to_polars
# ---------------------------------------------------------------------------


class TestDocumentToPolars:
    def test_multi_table(self) -> None:
        doc = TabularDocument(
            metadata=DocumentMetadata(title="Test"),
            tables=(
                Table(
                    name="t1",
                    columns=(Column("a", ColumnType.INTEGER),),
                    rows=((1,), (2,)),
                ),
                Table(
                    name="t2",
                    columns=(Column("b", ColumnType.TEXT),),
                    rows=(("x",), ("y",)),
                ),
            ),
        )
        result = document_to_polars(doc)
        assert set(result.keys()) == {"t1", "t2"}
        assert len(result["t1"]) == 2
        assert len(result["t2"]) == 2

    def test_empty_document(self) -> None:
        doc = TabularDocument()
        result = document_to_polars(doc)
        assert result == {}
