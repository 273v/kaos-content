"""Tests for ContentDocument Table → TabularDocument bridge."""

from __future__ import annotations

from kaos_content.bridges.content_to_tabular import (
    content_table_to_tabular,
    extract_tables_as_tabular,
)
from kaos_content.model.blocks import Heading, Paragraph
from kaos_content.model.blocks import Table as ContentTable
from kaos_content.model.document import ContentDocument
from kaos_content.model.inlines import Link, Strong, Text
from kaos_content.model.metadata import DocumentMetadata
from kaos_content.model.table import Cell, Row, TableSection


def _text_cell(*texts: str) -> Cell:
    """Create a Cell with simple text paragraphs."""
    content = tuple(
        Paragraph.model_construct(children=(Text.model_construct(value=t),)) for t in texts
    )
    return Cell.model_construct(content=content)


def _row(*texts: str) -> Row:
    return Row.model_construct(cells=tuple(_text_cell(t) for t in texts))


class TestContentTableToTabular:
    def test_simple_table(self) -> None:
        table = ContentTable.model_construct(
            head=TableSection.model_construct(rows=(_row("Name", "Age"),)),
            bodies=(TableSection.model_construct(rows=(_row("Alice", "30"), _row("Bob", "25"))),),
        )
        result = content_table_to_tabular(table, name="people")
        assert result.name == "people"
        assert result.column_names() == ("Name", "Age")
        assert len(result.rows) == 2
        assert result.rows[0] == ("Alice", 30)
        assert result.rows[1] == ("Bob", 25)

    def test_no_header(self) -> None:
        table = ContentTable.model_construct(
            bodies=(TableSection.model_construct(rows=(_row("x", "1"), _row("y", "2"))),),
        )
        result = content_table_to_tabular(table, name="t", use_header=False)
        assert result.column_names() == ("column_0", "column_1")
        assert len(result.rows) == 2

    def test_numeric_coercion(self) -> None:
        table = ContentTable.model_construct(
            head=TableSection.model_construct(rows=(_row("val"),)),
            bodies=(
                TableSection.model_construct(
                    rows=(_row("42"), _row("3.14"), _row("$1,234"), _row("hello"))
                ),
            ),
        )
        result = content_table_to_tabular(table, name="nums")
        assert result.rows[0] == (42,)
        assert result.rows[1] == (3.14,)
        assert result.rows[2] == (1234.0,)  # Currency stripped, float
        assert result.rows[3] == ("hello",)

    def test_empty_cells(self) -> None:
        table = ContentTable.model_construct(
            head=TableSection.model_construct(rows=(_row("a", "b"),)),
            bodies=(
                TableSection.model_construct(
                    rows=(
                        Row.model_construct(
                            cells=(_text_cell("x"), Cell.model_construct(content=()))
                        ),
                    )
                ),
            ),
        )
        result = content_table_to_tabular(table, name="sparse")
        assert result.rows[0] == ("x", None)

    def test_rich_content_flattened(self) -> None:
        """Cells with bold, links, emphasis → plain text."""
        rich_cell = Cell.model_construct(
            content=(
                Paragraph.model_construct(
                    children=(
                        Strong.model_construct(children=(Text.model_construct(value="Bold"),)),
                        Text.model_construct(value=" and "),
                        Link.model_construct(
                            target="http://example.com",
                            children=(Text.model_construct(value="link"),),
                        ),
                    )
                ),
            )
        )
        table = ContentTable.model_construct(
            head=TableSection.model_construct(rows=(_row("content"),)),
            bodies=(TableSection.model_construct(rows=(Row.model_construct(cells=(rich_cell,)),)),),
        )
        result = content_table_to_tabular(table, name="rich")
        assert "Bold" in result.rows[0][0]
        assert "link" in result.rows[0][0]

    def test_empty_table(self) -> None:
        table = ContentTable.model_construct(
            bodies=(),
        )
        result = content_table_to_tabular(table, name="empty")
        assert result.rows == ()
        assert result.columns == ()

    def test_ragged_rows(self) -> None:
        """Rows with different cell counts get padded."""
        table = ContentTable.model_construct(
            head=TableSection.model_construct(rows=(_row("a", "b", "c"),)),
            bodies=(
                TableSection.model_construct(rows=(_row("1"),)),  # Short row
            ),
        )
        result = content_table_to_tabular(table, name="ragged")
        assert len(result.rows[0]) == 3  # Padded to 3 columns


class TestExtractTablesAsTabular:
    def test_document_with_tables(self) -> None:
        doc = ContentDocument(
            metadata=DocumentMetadata(title="Test"),
            body=(
                Heading.model_construct(depth=1, children=(Text.model_construct(value="Title"),)),
                ContentTable.model_construct(
                    head=TableSection.model_construct(rows=(_row("x", "y"),)),
                    bodies=(TableSection.model_construct(rows=(_row("1", "2"),)),),
                ),
                Paragraph.model_construct(children=(Text.model_construct(value="text"),)),
                ContentTable.model_construct(
                    bodies=(TableSection.model_construct(rows=(_row("a", "b"),)),),
                ),
            ),
        )
        result = extract_tables_as_tabular(doc)
        assert len(result.tables) == 2
        assert result.tables[0].column_names() == ("x", "y")
        assert result.tables[0].rows[0] == (1, 2)

    def test_document_without_tables(self) -> None:
        doc = ContentDocument(
            body=(Paragraph.model_construct(children=(Text.model_construct(value="no tables"),)),),
        )
        result = extract_tables_as_tabular(doc)
        assert len(result.tables) == 0

    def test_metadata_preserved(self) -> None:
        doc = ContentDocument(
            metadata=DocumentMetadata(title="Report"),
            body=(
                ContentTable.model_construct(
                    bodies=(TableSection.model_construct(rows=(_row("data"),)),),
                ),
            ),
        )
        result = extract_tables_as_tabular(doc)
        assert result.metadata.title == "Report"


class TestPercentageCoercion:
    def test_percentage(self) -> None:
        table = ContentTable.model_construct(
            head=TableSection.model_construct(rows=(_row("pct"),)),
            bodies=(TableSection.model_construct(rows=(_row("50%"),)),),
        )
        result = content_table_to_tabular(table, name="pct")
        assert result.rows[0][0] == 0.5
