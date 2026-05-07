"""Tests for TabularView — hierarchical navigation over TabularDocument."""

from __future__ import annotations

import datetime

import pytest

from kaos_content.model.metadata import DocumentMetadata
from kaos_content.model.tabular import Column, ColumnType, Table, TabularDocument
from kaos_content.views.tabular_view import TabularView


@pytest.fixture()
def multi_table_doc() -> TabularDocument:
    return TabularDocument(
        metadata=DocumentMetadata(title="Test Workbook"),
        tables=(
            Table(
                name="sales",
                columns=(
                    Column("date", ColumnType.DATE),
                    Column("region", ColumnType.TEXT),
                    Column("amount", ColumnType.FLOAT),
                ),
                rows=(
                    (datetime.date(2024, 1, 1), "US", 100.0),
                    (datetime.date(2024, 1, 2), "EU", 200.0),
                    (datetime.date(2024, 1, 3), "US", 150.0),
                    (datetime.date(2024, 1, 4), "EU", None),
                    (datetime.date(2024, 1, 5), "US", 300.0),
                ),
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


class TestTabularViewBasic:
    def test_table_count(self, multi_table_doc: TabularDocument) -> None:
        view = TabularView(multi_table_doc)
        assert view.table_count == 2

    def test_table_names(self, multi_table_doc: TabularDocument) -> None:
        view = TabularView(multi_table_doc)
        assert view.table_names == ("sales", "regions")

    def test_total_rows(self, multi_table_doc: TabularDocument) -> None:
        view = TabularView(multi_table_doc)
        assert view.total_rows == 7  # 5 + 2

    def test_document_property(self, multi_table_doc: TabularDocument) -> None:
        view = TabularView(multi_table_doc)
        assert view.document is multi_table_doc


class TestTabularViewTableAccess:
    def test_get_table(self, multi_table_doc: TabularDocument) -> None:
        view = TabularView(multi_table_doc)
        t = view.get_table("sales")
        assert t.name == "sales"
        assert t.row_count == 5

    def test_get_table_not_found(self, multi_table_doc: TabularDocument) -> None:
        view = TabularView(multi_table_doc)
        with pytest.raises(KeyError, match="Table not found"):
            view.get_table("nonexistent")

    def test_table_head(self, multi_table_doc: TabularDocument) -> None:
        view = TabularView(multi_table_doc)
        t = view.table_head("sales", n=2)
        assert len(t.rows) == 2
        assert t.row_count == 5  # Preserves original count

    def test_table_slice(self, multi_table_doc: TabularDocument) -> None:
        view = TabularView(multi_table_doc)
        t = view.table_slice("sales", 1, 3)
        assert len(t.rows) == 2
        assert t.rows[0][1] == "EU"  # Row index 1


class TestTabularViewSchema:
    def test_table_schema(self, multi_table_doc: TabularDocument) -> None:
        view = TabularView(multi_table_doc)
        schema = view.table_schema("sales")
        assert schema["name"] == "sales"
        assert schema["row_count"] == 5
        assert schema["column_count"] == 3
        assert len(schema["columns"]) == 3
        assert schema["columns"][0]["name"] == "date"
        assert schema["columns"][0]["type"] == "date"

    def test_table_infos(self, multi_table_doc: TabularDocument) -> None:
        view = TabularView(multi_table_doc)
        infos = view.table_infos
        assert len(infos) == 2
        assert infos[0].name == "sales"
        assert infos[0].row_count == 5
        assert infos[0].column_count == 3
        assert infos[1].name == "regions"


class TestTabularViewColumnStats:
    def test_text_column(self, multi_table_doc: TabularDocument) -> None:
        view = TabularView(multi_table_doc)
        stats = view.column_stats("sales", "region")
        assert stats.name == "region"
        assert stats.column_type == "text"
        assert stats.null_count == 0
        assert stats.non_null_count == 5
        assert stats.unique_count == 2  # US, EU
        assert stats.min_value == "EU"
        assert stats.max_value == "US"

    def test_float_column_with_null(self, multi_table_doc: TabularDocument) -> None:
        view = TabularView(multi_table_doc)
        stats = view.column_stats("sales", "amount")
        assert stats.null_count == 1
        assert stats.non_null_count == 4
        assert stats.min_value == 100.0
        assert stats.max_value == 300.0

    def test_date_column(self, multi_table_doc: TabularDocument) -> None:
        view = TabularView(multi_table_doc)
        stats = view.column_stats("sales", "date")
        assert stats.min_value == datetime.date(2024, 1, 1)
        assert stats.max_value == datetime.date(2024, 1, 5)

    def test_all_column_stats(self, multi_table_doc: TabularDocument) -> None:
        view = TabularView(multi_table_doc)
        all_stats = view.all_column_stats("sales")
        assert len(all_stats) == 3
        names = [s.name for s in all_stats]
        assert names == ["date", "region", "amount"]

    def test_column_not_found(self, multi_table_doc: TabularDocument) -> None:
        view = TabularView(multi_table_doc)
        with pytest.raises(KeyError):
            view.column_stats("sales", "nonexistent")


class TestTabularViewSummary:
    def test_summary_dict(self, multi_table_doc: TabularDocument) -> None:
        view = TabularView(multi_table_doc)
        summary = view.summary_dict()
        assert summary["title"] == "Test Workbook"
        assert summary["table_count"] == 2
        assert summary["total_rows"] == 7
        assert len(summary["tables"]) == 2
        assert summary["tables"][0]["name"] == "sales"
        assert summary["tables"][0]["columns"][0]["name"] == "date"
        assert summary["tables"][0]["columns"][0]["type"] == "date"


class TestTabularViewEdgeCases:
    def test_empty_document(self) -> None:
        view = TabularView(TabularDocument())
        assert view.table_count == 0
        assert view.table_names == ()
        assert view.total_rows == 0
        assert view.table_infos == ()

    def test_empty_table(self) -> None:
        doc = TabularDocument(tables=(Table(name="empty"),))
        view = TabularView(doc)
        assert view.table_count == 1
        info = view.table_infos[0]
        assert info.row_count == 0
        assert info.column_count == 0

    def test_all_null_column(self) -> None:
        doc = TabularDocument(
            tables=(
                Table(
                    name="nulls",
                    columns=(Column("x", ColumnType.INTEGER),),
                    rows=((None,), (None,), (None,)),
                ),
            ),
        )
        view = TabularView(doc)
        stats = view.column_stats("nulls", "x")
        assert stats.null_count == 3
        assert stats.non_null_count == 0
        assert stats.min_value is None
        assert stats.max_value is None

    def test_lazy_caching(self, multi_table_doc: TabularDocument) -> None:
        view = TabularView(multi_table_doc)
        # First access computes
        infos1 = view.table_infos
        # Second access returns cached
        infos2 = view.table_infos
        assert infos1 is infos2
