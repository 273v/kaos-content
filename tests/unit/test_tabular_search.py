"""Tests for search_tabular — column-aware text search on TabularDocument."""

from __future__ import annotations

import pytest

from kaos_content.model.metadata import DocumentMetadata
from kaos_content.model.tabular import (
    Column,
    ColumnType,
    Table,
    TabularDocument,
)
from kaos_content.search import search_tabular


@pytest.fixture()
def search_doc() -> TabularDocument:
    return TabularDocument(
        metadata=DocumentMetadata(title="Test"),
        tables=(
            Table(
                name="employees",
                columns=(
                    Column("name", ColumnType.TEXT),
                    Column("department", ColumnType.TEXT),
                    Column("salary", ColumnType.INTEGER),
                ),
                rows=(
                    ("Alice Johnson", "Engineering", 120000),
                    ("Bob Smith", "Marketing", 95000),
                    ("Charlie Brown", "Engineering", 110000),
                    ("Diana Prince", "Sales", 105000),
                ),
            ),
            Table(
                name="departments",
                columns=(
                    Column("name", ColumnType.TEXT),
                    Column("budget", ColumnType.INTEGER),
                ),
                rows=(
                    ("Engineering", 500000),
                    ("Marketing", 200000),
                    ("Sales", 300000),
                ),
            ),
        ),
    )


class TestSearchTabular:
    def test_basic_search(self, search_doc: TabularDocument) -> None:
        results = search_tabular(search_doc, "Alice")
        assert results.total_matches >= 1
        assert any("Alice" in r.text for r in results.results)

    def test_case_insensitive(self, search_doc: TabularDocument) -> None:
        results = search_tabular(search_doc, "engineering")
        assert results.total_matches >= 2  # appears in both tables

    def test_scope_to_table(self, search_doc: TabularDocument) -> None:
        results = search_tabular(search_doc, "Engineering", table_name="employees")
        for r in results.results:
            assert "employees" in r.block_ref

    def test_scope_to_column(self, search_doc: TabularDocument) -> None:
        results = search_tabular(search_doc, "Engineering", column="department")
        for r in results.results:
            assert r.section_title is not None and "department" in r.section_title

    def test_exact_match_scores_higher(self, search_doc: TabularDocument) -> None:
        results = search_tabular(search_doc, "Engineering")
        # Exact match (just "Engineering") should score 2.0
        exact = [r for r in results.results if r.text == "Engineering"]
        partial = [r for r in results.results if r.text != "Engineering"]
        if exact and partial:
            assert exact[0].score > partial[0].score

    def test_top_k(self, search_doc: TabularDocument) -> None:
        results = search_tabular(search_doc, "Engineering", top_k=1)
        assert len(results.results) <= 1
        assert results.has_more is True

    def test_no_matches(self, search_doc: TabularDocument) -> None:
        results = search_tabular(search_doc, "zzz_nonexistent")
        assert results.total_matches == 0
        assert results.results == []
        assert results.has_more is False

    def test_empty_query_raises(self, search_doc: TabularDocument) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            search_tabular(search_doc, "")

    def test_whitespace_query_raises(self, search_doc: TabularDocument) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            search_tabular(search_doc, "   ")

    def test_numeric_values_searched_as_text(self, search_doc: TabularDocument) -> None:
        results = search_tabular(search_doc, "120000")
        assert results.total_matches >= 1

    def test_block_ref_format(self, search_doc: TabularDocument) -> None:
        results = search_tabular(search_doc, "Alice")
        assert results.results[0].block_ref.startswith("#/tables/")

    def test_section_ref_is_table_name(self, search_doc: TabularDocument) -> None:
        results = search_tabular(search_doc, "Alice")
        assert results.results[0].section_ref == "employees"

    def test_nonexistent_table_returns_empty(self, search_doc: TabularDocument) -> None:
        results = search_tabular(search_doc, "Alice", table_name="nonexistent")
        assert results.total_matches == 0

    def test_nonexistent_column_returns_empty(self, search_doc: TabularDocument) -> None:
        results = search_tabular(search_doc, "Alice", column="nonexistent")
        assert results.total_matches == 0

    def test_path_matches_section_title(self, search_doc: TabularDocument) -> None:
        # ``SearchResult.path`` is the canonical citation breadcrumb. For
        # tabular hits it must equal ``(section_title,)`` since there is
        # no heading hierarchy above a cell. Regression for the
        # 0.1.0a11 follow-up where ``search_tabular`` was setting
        # ``section_title`` but leaving ``path=()`` — that empty tuple
        # is the contract for "no structural identifier", so an
        # agent citing the hit would refuse to mention the column.
        results = search_tabular(search_doc, "Alice")
        assert results.results, "fixture should have at least one Alice hit"
        for r in results.results:
            assert r.section_title is not None
            assert r.path == (r.section_title,)
