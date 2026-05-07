"""Serializers for TabularDocument / Table.

Provides CSV, TSV, markdown table, and JSON records output formats.
TSV is the default MCP output format (30-40% fewer tokens than JSON).
Markdown tables are used for CLI and small inline results.

Usage::

    from kaos_content.serializers.tabular import (
        serialize_csv, serialize_tsv, serialize_markdown_table,
        serialize_json_records, serialize_tabular_summary,
    )

    table = Table(name="data", columns=(...), rows=(...))
    print(serialize_tsv(table))
    print(serialize_markdown_table(table))
"""

from __future__ import annotations

import csv
import io
import json
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kaos_content.model.tabular import Table, TabularDocument


def _format_value(value: Any) -> str:
    """Format a cell value as a string for text-based serialization.

    Handles None, dates, times, datetimes, Decimal, bytes, booleans,
    lists, and dicts. Everything else uses str().
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, time):
        return value.isoformat()
    if isinstance(value, timedelta):
        seconds = int(value.total_seconds())
        h, rem = divmod(abs(seconds), 3600)
        m, s = divmod(rem, 60)
        sign = "-" if seconds < 0 else ""
        return f"{sign}{h}:{m:02d}:{s:02d}"
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (bytes, bytearray)):
        return value.hex()
    if isinstance(value, (list, dict)):
        return json.dumps(value, default=str, ensure_ascii=False)
    return str(value)


def _format_value_json(value: Any) -> Any:
    """Format a cell value for JSON serialization.

    Returns JSON-native types where possible. Converts dates,
    Decimal, bytes, and timedelta to strings.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, time):
        return value.isoformat()
    if isinstance(value, timedelta):
        return _format_value(value)
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (bytes, bytearray)):
        return value.hex()
    if isinstance(value, (list, dict)):
        return value
    return str(value)


# ---------------------------------------------------------------------------
# CSV / TSV
# ---------------------------------------------------------------------------


def serialize_csv(table: Table, *, delimiter: str = ",") -> str:
    """Serialize a Table as CSV (or TSV with ``delimiter='\\t'``).

    Args:
        table: The table to serialize.
        delimiter: Field delimiter. Default is comma.

    Returns:
        CSV-formatted string with header row.
    """
    buf = io.StringIO()
    writer = csv.writer(buf, delimiter=delimiter, lineterminator="\n")
    writer.writerow(c.name for c in table.columns)
    for row in table.rows:
        writer.writerow(_format_value(v) for v in row)
    return buf.getvalue()


def serialize_tsv(table: Table) -> str:
    """Serialize a Table as TSV (tab-separated values).

    TSV is the recommended MCP output format: 30-40% fewer tokens
    than JSON, and better LLM accuracy than CSV for data with commas.
    """
    return serialize_csv(table, delimiter="\t")


# ---------------------------------------------------------------------------
# Markdown table
# ---------------------------------------------------------------------------


def serialize_markdown_table(table: Table, *, max_rows: int = 50) -> str:
    """Serialize a Table as a GFM markdown table.

    Args:
        table: The table to serialize.
        max_rows: Maximum number of data rows to include. 0 = no limit.

    Returns:
        Markdown-formatted table string.
    """
    if not table.columns:
        return f"*Empty table: {table.name}*\n"

    names = [c.name for c in table.columns]
    col_widths = [len(n) for n in names]

    # Determine display rows
    rows = table.rows
    truncated = False
    if max_rows > 0 and len(rows) > max_rows:
        rows = rows[:max_rows]
        truncated = True

    # Pre-format all cell values
    formatted_rows: list[list[str]] = []
    for row in rows:
        cells = [_format_value(v) for v in row]
        # Pad if row is shorter than columns
        while len(cells) < len(names):
            cells.append("")
        formatted_rows.append(cells)
        for i, cell in enumerate(cells):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(cell))

    # Ensure minimum separator width of 3
    col_widths = [max(w, 3) for w in col_widths]

    # Build header
    header = "| " + " | ".join(n.ljust(w) for n, w in zip(names, col_widths, strict=True)) + " |"
    separator = "| " + " | ".join("-" * w for w in col_widths) + " |"

    lines = [header, separator]

    # Build data rows
    for cells in formatted_rows:
        line = (
            "| "
            + " | ".join(
                cells[i].ljust(col_widths[i]) if i < len(cells) else " " * col_widths[i]
                for i in range(len(names))
            )
            + " |"
        )
        lines.append(line)

    if truncated:
        remaining = table.row_count - max_rows
        lines.append(f"\n*... {remaining} more rows (total: {table.row_count})*")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# JSON records
# ---------------------------------------------------------------------------


def serialize_json_records(table: Table, *, indent: int | None = None) -> str:
    """Serialize a Table as a JSON array of records.

    Each row becomes ``{"col1": val1, "col2": val2, ...}``.

    Args:
        table: The table to serialize.
        indent: JSON indentation level. None for compact output.

    Returns:
        JSON string.
    """
    names = [c.name for c in table.columns]
    records: list[dict[str, Any]] = []
    for row in table.rows:
        record = {}
        for i, name in enumerate(names):
            val = row[i] if i < len(row) else None
            record[name] = _format_value_json(val)
        records.append(record)
    return json.dumps(records, indent=indent, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Document summary
# ---------------------------------------------------------------------------


def serialize_tabular_summary(doc: TabularDocument) -> str:
    """Generate a concise text summary of a TabularDocument.

    Suitable for inline MCP results and CLI output. Lists table
    names, dimensions, and column types.

    Args:
        doc: The tabular document to summarize.

    Returns:
        Human-readable summary string.
    """
    title = doc.metadata.title or "Untitled"
    n_tables = len(doc.tables)
    total_rows = sum(t.row_count for t in doc.tables)

    lines = [f"{title} — {n_tables} table(s), {total_rows} total rows"]

    for table in doc.tables:
        cols_desc = ", ".join(f"{c.name} ({c.column_type.value})" for c in table.columns)
        lines.append(f"  {table.name}: {table.row_count} rows x {len(table.columns)} cols")
        if cols_desc:
            lines.append(f"    Columns: {cols_desc}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Full document markdown
# ---------------------------------------------------------------------------


def serialize_tabular_markdown(
    doc: TabularDocument,
    *,
    max_rows: int = 50,
    include_metadata: bool = True,
) -> str:
    """Serialize a full TabularDocument as markdown.

    Produces a complete markdown document with title, metadata, and
    one section per table (with heading + markdown table). This is the
    tabular equivalent of ``serialize_markdown()`` for ContentDocument.

    Works for any TabularDocument regardless of source — an XLSX
    workbook with 3 sheets and a database with 3 tables produce the
    same structure.

    Args:
        doc: The TabularDocument to serialize.
        max_rows: Maximum data rows per table. 0 = no limit.
        include_metadata: Include a metadata section with source info.

    Returns:
        Complete markdown string.
    """
    parts: list[str] = []

    # Title
    title = doc.metadata.title or "Untitled"
    parts.append(f"# {title}\n")

    # Metadata block
    if include_metadata:
        meta_lines: list[str] = []
        if doc.metadata.source:
            meta_lines.append(f"- **Source**: {doc.metadata.source.uri}")
        if doc.metadata.document_type:
            meta_lines.append(f"- **Format**: {doc.metadata.document_type}")
        n_tables = len(doc.tables)
        total_rows = sum(t.row_count for t in doc.tables)
        meta_lines.append(f"- **Tables**: {n_tables}")
        meta_lines.append(f"- **Total rows**: {total_rows}")
        if doc.provenance and doc.provenance.extractor:
            meta_lines.append(f"- **Extractor**: {doc.provenance.extractor}")
        if meta_lines:
            parts.append("\n".join(meta_lines))
            parts.append("")

    # Each table as a section
    for table in doc.tables:
        # Section heading
        col_count = len(table.columns)
        parts.append(f"## {table.name}")
        parts.append(f"*{table.row_count} rows, {col_count} columns*\n")

        # Table content
        if table.columns:
            parts.append(serialize_markdown_table(table, max_rows=max_rows))
        else:
            parts.append("*Empty table*\n")

        # Table metadata (formulas, primary keys, etc.)
        if table.metadata:
            interesting = {
                k: v
                for k, v in table.metadata.items()
                if k in ("formulas", "primary_key", "schema", "merged_ranges")
            }
            if interesting:
                meta_items = []
                for k, v in interesting.items():
                    if isinstance(v, dict) and len(v) <= 5:
                        meta_items.append(f"- **{k}**: {v}")
                    elif isinstance(v, list) and len(v) <= 5:
                        meta_items.append(f"- **{k}**: {', '.join(str(x) for x in v)}")
                    elif isinstance(v, (dict, list)):
                        meta_items.append(f"- **{k}**: ({len(v)} entries)")
                    else:
                        meta_items.append(f"- **{k}**: {v}")
                if meta_items:
                    parts.append("\n".join(meta_items))
                    parts.append("")

    return "\n".join(parts)
