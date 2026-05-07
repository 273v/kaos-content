"""Bridge ContentDocument Table blocks → TabularDocument Tables.

Converts the rich AST table model (cells contain Block children with
formatting, links, images) into flat tabular data (cells are Python
native values) suitable for SQL querying, CSV export, etc.

Usage::

    from kaos_content.bridges.content_to_tabular import (
        content_table_to_tabular,
        extract_tables_as_tabular,
    )

    # Single table block → TabularDocument Table
    tabular_table = content_table_to_tabular(table_block, name="data")

    # All tables from a ContentDocument → TabularDocument
    doc = extract_tables_as_tabular(content_doc)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from kaos_content.model.tabular import (
    Column,
    TabularDocument,
    infer_column_type,
)
from kaos_content.model.tabular import (
    Table as TabularTable,
)
from kaos_content.traversal.visitor import extract_text

if TYPE_CHECKING:
    from kaos_content.model.blocks import Table as ContentTable
    from kaos_content.model.document import ContentDocument
    from kaos_content.model.table import Cell, Row


def content_table_to_tabular(
    table: ContentTable,
    *,
    name: str = "table",
    use_header: bool = True,
) -> TabularTable:
    """Convert a ContentDocument Table block to a TabularDocument Table.

    Flattens cell content using ``extract_text()`` — rich formatting
    (bold, links, images) becomes plain text. Column types are inferred
    from the data.

    Args:
        table: ContentDocument Table block.
        name: Name for the resulting TabularDocument Table.
        use_header: If True and the table has a ``<thead>``, use the
            first header row as column names.

    Returns:
        TabularDocument Table with flat string/typed values.
    """
    # Collect all rows from all sections
    header_rows: list[list[str]] = []
    data_rows: list[list[str]] = []

    if table.head:
        for row in table.head.rows:
            header_rows.append(_flatten_row(row))

    for body in table.bodies:
        for row in body.rows:
            data_rows.append(_flatten_row(row))

    if table.foot:
        for row in table.foot.rows:
            data_rows.append(_flatten_row(row))

    # Determine column count
    all_rows = header_rows + data_rows
    if not all_rows:
        return TabularTable(name=name)

    n_cols = max(len(r) for r in all_rows)

    # Determine column names
    if use_header and header_rows:
        raw_headers = header_rows[0]
        headers = [h.strip() if h.strip() else f"column_{i}" for i, h in enumerate(raw_headers)]
        # Pad if needed
        while len(headers) < n_cols:
            headers.append(f"column_{len(headers)}")
    else:
        headers = [f"column_{i}" for i in range(n_cols)]
        # If no separate header, all rows are data (including what was in thead)
        data_rows = header_rows + data_rows

    # Normalize rows to n_cols width
    normalized: list[tuple[Any, ...]] = []
    for row in data_rows:
        padded = row + [""] * (n_cols - len(row))
        # Try to coerce values to native Python types
        normalized.append(tuple(_coerce_value(v) for v in padded[:n_cols]))

    # Infer column types
    columns: list[Column] = []
    for i, hdr in enumerate(headers):
        col_vals = [row[i] for row in normalized if i < len(row)]
        ct = infer_column_type(col_vals)
        nullable = any(v is None or v == "" for v in col_vals)
        columns.append(Column(name=hdr, column_type=ct, nullable=nullable))

    return TabularTable(
        name=name,
        columns=tuple(columns),
        rows=tuple(normalized),
        row_count=len(normalized),
    )


def extract_tables_as_tabular(
    document: ContentDocument,
    *,
    use_header: bool = True,
) -> TabularDocument:
    """Extract all Table blocks from a ContentDocument as a TabularDocument.

    Args:
        document: The ContentDocument to extract tables from.
        use_header: Use thead rows as column names.

    Returns:
        TabularDocument with one Table per HTML table found.
    """
    from kaos_content.model.blocks import Table as ContentTableType

    tables: list[TabularTable] = []
    for i, block in enumerate(document.body):
        if isinstance(block, ContentTableType):
            tname = f"table_{i + 1}"
            # Use caption text as name if available
            if block.caption and block.caption.short:
                caption_text = " ".join(
                    extract_text(inline) for inline in block.caption.short
                ).strip()
                if caption_text:
                    tname = caption_text
            tables.append(content_table_to_tabular(block, name=tname, use_header=use_header))

    from kaos_content.model.metadata import DocumentMetadata

    return TabularDocument(
        metadata=DocumentMetadata(
            title=document.metadata.title,
            source=document.metadata.source,
        ),
        tables=tuple(tables),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _flatten_row(row: Row) -> list[str]:
    """Flatten a Row's cells to plain text strings."""
    return [_flatten_cell(cell) for cell in row.cells]


def _flatten_cell(cell: Cell) -> str:
    """Flatten a Cell's block content to plain text."""
    if not cell.content:
        return ""
    parts = [extract_text(block).strip() for block in cell.content]
    return " ".join(p for p in parts if p)


def _coerce_value(text: str) -> Any:
    """Try to coerce a string to a Python native type.

    Returns the most specific type that fits:
    int > float > original string. Empty string → None.
    """
    if not text or not text.strip():
        return None

    s = text.strip()

    # Remove common formatting: commas in numbers, currency symbols
    cleaned = s.replace(",", "").replace("$", "").replace("€", "").replace("£", "")

    # Try int
    try:
        return int(cleaned)
    except ValueError:
        pass

    # Try float
    try:
        return float(cleaned)
    except ValueError:
        pass

    # Try percentage
    if s.endswith("%"):
        try:
            return float(s[:-1].strip()) / 100.0
        except ValueError:
            pass

    return s
