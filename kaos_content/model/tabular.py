"""Universal tabular document model — peer to ContentDocument.

TabularDocument is the universal AST for all tabular data: CSV files,
XLSX sheets, database tables, Parquet files, JSON arrays, markdown tables.
Column-level typing (like Arrow/Polars/DuckDB/SQL), not cell-level.
Format-specific extras (formulas, foreign keys, merged cells) go in
metadata dicts, not core model fields.

This is the LOGICAL type system (like Pandoc's AST), not a storage system.
int8 vs int64 is storage; INTEGER is logical. float32 vs float64 is
storage; FLOAT is logical.

Usage::

    from kaos_content.model.tabular import (
        ColumnType, Column, Table, TabularDocument,
    )

    table = Table(
        name="sales",
        columns=(
            Column("date", ColumnType.DATE),
            Column("amount", ColumnType.DECIMAL),
            Column("region", ColumnType.TEXT),
        ),
        rows=(
            (date(2024, 1, 1), Decimal("1234.56"), "US"),
            (date(2024, 1, 2), Decimal("789.01"), "EU"),
        ),
        row_count=2,
    )
    doc = TabularDocument(tables=(table,))
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict

from kaos_content.model.attr import Provenance
from kaos_content.model.metadata import DocumentMetadata


class ColumnType(StrEnum):
    """Universal column type — maps cleanly to Arrow, Polars, DuckDB, SQL.

    Four tiers:
    - Tier 1: Present in all tabular sources (TEXT through NULL)
    - Tier 2: Present in most analytical sources (DECIMAL, BINARY, DURATION)
    - Tier 3: Present in structured sources like JSON/Parquet (LIST, STRUCT)
    - Tier 4: Domain-specific extraction types (KAOS structured-extraction).
      These widen to TEXT, DECIMAL, INTEGER, or STRUCT under bridges.
    """

    # Tier 1: Universal
    TEXT = "text"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATE = "date"
    TIME = "time"
    DATETIME = "datetime"
    NULL = "null"

    # Tier 2: Analytical
    DECIMAL = "decimal"
    BINARY = "binary"
    DURATION = "duration"

    # Tier 3: Structured
    LIST = "list"
    STRUCT = "struct"

    # Tier 4: Extraction-specific (KAOS)
    VERBATIM_QUOTE = "verbatim_quote"
    """Exact span from a source document. Logically TEXT but flagged so
    serializers know not to normalize whitespace or punctuation."""

    MONEY = "money"
    """Currency-aware monetary value. Stored as
    ``{"amount": Decimal, "currency": str}`` (ISO 4217 code).
    Widens to STRUCT under analytical bridges."""

    SCORE = "score"
    """Ordinal score on a small fixed scale (e.g., Relativity's 0-4).
    Logically INTEGER but flagged so UIs render as a chip/badge."""

    ENTITY_ROLE = "entity_role"
    """Tagged entity with a role label, e.g.,
    ``{"name": str, "role": str, "entity_type": str | None}``.
    Common in legal extraction (parties, signatories). Widens to STRUCT."""


# ---------------------------------------------------------------------------
# Python type → ColumnType mapping
# ---------------------------------------------------------------------------

_PYTHON_TYPE_MAP: dict[type, ColumnType] = {
    str: ColumnType.TEXT,
    int: ColumnType.INTEGER,
    float: ColumnType.FLOAT,
    bool: ColumnType.BOOLEAN,
    datetime.date: ColumnType.DATE,
    datetime.time: ColumnType.TIME,
    datetime.datetime: ColumnType.DATETIME,
    Decimal: ColumnType.DECIMAL,
    bytes: ColumnType.BINARY,
    bytearray: ColumnType.BINARY,
    datetime.timedelta: ColumnType.DURATION,
    list: ColumnType.LIST,
    dict: ColumnType.STRUCT,
    type(None): ColumnType.NULL,
}


def column_type_from_python(value: Any) -> ColumnType:
    """Infer ColumnType from a Python value.

    Checks exact type first (bool before int, datetime before date),
    then falls back to isinstance checks. Returns TEXT as the
    universal fallback.

    Special cases:
    - ``float('nan')`` and ``float('inf')`` → FLOAT (not NULL)
    - ``True`` / ``False`` → BOOLEAN (not INTEGER, even though bool <: int)
    - ``datetime.datetime`` → DATETIME (not DATE, even though datetime <: date)
    """
    if value is None:
        return ColumnType.NULL

    vtype = type(value)

    # Exact match — handles bool before int (bool is subclass of int)
    ct = _PYTHON_TYPE_MAP.get(vtype)
    if ct is not None:
        return ct

    # isinstance fallback for subclasses
    # Order matters: datetime before date, bool before int
    if isinstance(value, datetime.datetime):
        return ColumnType.DATETIME
    if isinstance(value, datetime.date):
        return ColumnType.DATE
    if isinstance(value, datetime.time):
        return ColumnType.TIME
    if isinstance(value, datetime.timedelta):
        return ColumnType.DURATION
    if isinstance(value, bool):
        return ColumnType.BOOLEAN
    if isinstance(value, int):
        return ColumnType.INTEGER
    if isinstance(value, float):
        return ColumnType.FLOAT
    if isinstance(value, Decimal):
        return ColumnType.DECIMAL
    if isinstance(value, (bytes, bytearray)):
        return ColumnType.BINARY
    if isinstance(value, list):
        return ColumnType.LIST
    if isinstance(value, dict):
        return ColumnType.STRUCT

    return ColumnType.TEXT


def infer_column_type(values: tuple[Any, ...] | list[Any]) -> ColumnType:
    """Infer the best ColumnType for a sequence of values.

    Skips None values. If all non-null values share the same type,
    returns that type. If mixed types are present, widens to TEXT.
    Returns NULL if all values are None.
    """
    seen: set[ColumnType] = set()
    for v in values:
        ct = column_type_from_python(v)
        if ct != ColumnType.NULL:
            seen.add(ct)

    if not seen:
        return ColumnType.NULL
    if len(seen) == 1:
        return next(iter(seen))

    # INTEGER + FLOAT → FLOAT (numeric widening)
    if seen == {ColumnType.INTEGER, ColumnType.FLOAT}:
        return ColumnType.FLOAT
    # INTEGER + DECIMAL → DECIMAL
    if seen == {ColumnType.INTEGER, ColumnType.DECIMAL}:
        return ColumnType.DECIMAL
    # FLOAT + DECIMAL → FLOAT (Decimal precision lost but pragmatic)
    if seen == {ColumnType.FLOAT, ColumnType.DECIMAL}:
        return ColumnType.FLOAT
    # INTEGER + FLOAT + DECIMAL → FLOAT
    if seen == {ColumnType.INTEGER, ColumnType.FLOAT, ColumnType.DECIMAL}:
        return ColumnType.FLOAT
    # DATE + DATETIME → DATETIME
    if seen == {ColumnType.DATE, ColumnType.DATETIME}:
        return ColumnType.DATETIME

    # Default: widen to TEXT
    return ColumnType.TEXT


@dataclass(frozen=True, slots=True)
class Column:
    """Column metadata — universal across all tabular sources.

    The ``metadata`` dict carries format-specific extras:
    - XLSX: ``{"format_str": "$#,##0.00", "width": 120}``
    - SQL:  ``{"pg_type": "numeric(10,2)", "default": "0"}``
    - Parquet: ``{"compression": "snappy"}``
    """

    name: str
    column_type: ColumnType = ColumnType.TEXT
    nullable: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Table:
    """A single named table — universal representation.

    Could be: an XLSX sheet, a CSV file, a DB table, a Parquet file,
    a JSON array, or a markdown table. The AST doesn't care.

    ``rows`` is row-major: each row is a tuple of Python native values
    (int, float, str, bool, None, datetime.date, Decimal, etc.).
    Column metadata carries type info; individual cells don't need their
    own type tag.

    ``row_count`` may differ from ``len(rows)`` when the table was
    truncated (e.g., only first 100 rows loaded from a 10M-row source).

    The ``metadata`` dict carries format-specific extras:
    - XLSX: ``{"frozen_rows": 1, "merged_ranges": ["A1:C1"],
              "formulas": {"B5": "=SUM(B2:B4)"}}``
    - SQL:  ``{"schema": "public", "primary_key": ["id"]}``
    - CSV:  ``{"delimiter": ",", "encoding": "utf-8"}``
    """

    name: str
    columns: tuple[Column, ...] = ()
    rows: tuple[tuple[Any, ...], ...] = ()
    row_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Set row_count to len(rows) if not explicitly provided."""
        if self.row_count == 0 and self.rows:
            object.__setattr__(self, "row_count", len(self.rows))

    def column_names(self) -> tuple[str, ...]:
        """Return column names as a tuple."""
        return tuple(c.name for c in self.columns)

    def column_values(self, name: str) -> tuple[Any, ...]:
        """Extract all values for a column by name.

        Raises:
            KeyError: If column name not found.
        """
        idx = self._column_index(name)
        return tuple(row[idx] for row in self.rows)

    def _column_index(self, name: str) -> int:
        """Find column index by name."""
        for i, col in enumerate(self.columns):
            if col.name == name:
                return i
        msg = f"Column not found: {name!r}. Available: {self.column_names()}"
        raise KeyError(msg)

    def head(self, n: int = 5) -> Table:
        """Return a new Table with at most the first ``n`` rows.

        Preserves the original ``row_count`` so consumers know the
        table was truncated.
        """
        return Table(
            name=self.name,
            columns=self.columns,
            rows=self.rows[:n],
            row_count=self.row_count,
            metadata=self.metadata,
        )

    def slice(self, start: int, end: int | None = None) -> Table:
        """Return a new Table with rows ``[start:end]``.

        Preserves the original ``row_count``.
        """
        return Table(
            name=self.name,
            columns=self.columns,
            rows=self.rows[start:end],
            row_count=self.row_count,
            metadata=self.metadata,
        )

    def validate(self) -> list[str]:
        """Check structural integrity. Returns a list of issues (empty = valid).

        Checks:
        - Rows have the expected number of columns (ragged detection)
        - No duplicate column names
        - row_count is consistent with rows
        """
        issues: list[str] = []
        n_cols = len(self.columns)

        # Duplicate column names
        seen_names: dict[str, int] = {}
        for col in self.columns:
            seen_names[col.name] = seen_names.get(col.name, 0) + 1
        for cname, count in seen_names.items():
            if count > 1:
                issues.append(f"Duplicate column name: {cname!r} (appears {count} times)")

        # Ragged rows
        for i, row in enumerate(self.rows):
            if len(row) != n_cols:
                issues.append(f"Row {i}: expected {n_cols} values, got {len(row)}")

        # row_count consistency
        if self.row_count < len(self.rows):
            issues.append(f"row_count ({self.row_count}) < len(rows) ({len(self.rows)})")

        return issues

    def describe(self) -> dict[str, Any]:
        """Compute column-level statistics for agent consumption.

        Returns a dict with table dimensions and per-column stats:
        null count, unique count (for small tables), and sample values.
        """
        result: dict[str, Any] = {
            "name": self.name,
            "row_count": self.row_count,
            "rows_loaded": len(self.rows),
            "column_count": len(self.columns),
            "columns": [],
        }

        for i, col in enumerate(self.columns):
            vals = [row[i] for row in self.rows if i < len(row)]
            null_count = sum(1 for v in vals if v is None)
            non_null = [v for v in vals if v is not None]

            col_info: dict[str, Any] = {
                "name": col.name,
                "type": col.column_type.value,
                "nullable": col.nullable,
                "null_count": null_count,
                "non_null_count": len(non_null),
            }

            # Unique count (cap at 1000 to avoid memory issues)
            if len(non_null) <= 1000:
                try:
                    col_info["unique_count"] = len(set(non_null))
                except TypeError:
                    # Unhashable values (lists, dicts)
                    col_info["unique_count"] = None

            # Sample values (first 3 non-null)
            samples = non_null[:3]
            col_info["sample_values"] = [str(v) for v in samples]

            result["columns"].append(col_info)

        return result


class TabularDocument(BaseModel):
    """Universal tabular document — peer to ContentDocument.

    Represents any source of tabular data: XLSX workbooks, CSV files,
    database schemas, Parquet datasets, JSON arrays, markdown tables.
    Multiple tables when the source has them (XLSX sheets, DB schemas).
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    metadata: DocumentMetadata = DocumentMetadata()
    tables: tuple[Table, ...] = ()
    provenance: Provenance | None = None

    def table_names(self) -> tuple[str, ...]:
        """Return the names of all tables."""
        return tuple(t.name for t in self.tables)

    def get_table(self, name: str) -> Table:
        """Get a table by name.

        Raises:
            KeyError: If table name not found.
        """
        for t in self.tables:
            if t.name == name:
                return t
        msg = f"Table not found: {name!r}. Available: {self.table_names()}"
        raise KeyError(msg)

    @classmethod
    def from_cells(
        cls,
        cells: Any,  # Sequence[ExtractionCell] — typed in body to avoid circular import
        *,
        column_specs: tuple[tuple[str, ColumnType], ...],
        table_name: str = "extraction",
        doc_id_column: str = "doc_id",
    ) -> TabularDocument:
        """Pivot a sequence of ``ExtractionCell`` into a TabularDocument.

        Each unique ``cell.doc_id`` becomes one row. Columns are built from
        ``column_specs`` (an ordered tuple of ``(column_id, ColumnType)``).
        A leading ``doc_id`` column is prepended automatically.

        Cell values are projected from ``cell.reviewed_value`` when set,
        else ``cell.ai_value``. Refusals (``status != "extracted"`` and no
        reviewed value) become ``None``.

        Args:
            cells: Sequence of ``ExtractionCell`` instances (any order).
            column_specs: Ordered tuple of ``(column_id, ColumnType)``
                pairs defining the table's column layout. Order is
                preserved in the output.
            table_name: Name for the produced ``Table``.
            doc_id_column: Name for the auto-prepended doc-identity column.

        Returns:
            A new ``TabularDocument`` with one ``Table`` containing the
            pivoted cells.

        Raises:
            ValueError: If ``column_specs`` is empty or contains duplicates.
        """
        from kaos_content.model.extraction import ExtractionCell

        if not column_specs:
            msg = "column_specs must contain at least one (column_id, ColumnType) pair"
            raise ValueError(msg)
        seen_ids: set[str] = set()
        for col_id, _ in column_specs:
            if col_id in seen_ids:
                msg = f"Duplicate column_id in column_specs: {col_id!r}"
                raise ValueError(msg)
            seen_ids.add(col_id)

        # Group cells by doc_id, then column_id. Preserve doc_id insertion order.
        rows_by_doc: dict[str, dict[str, Any]] = {}
        for cell in cells:
            if not isinstance(cell, ExtractionCell):
                msg = f"from_cells expected ExtractionCell instances, got {type(cell).__name__}"
                raise TypeError(msg)
            doc_bucket = rows_by_doc.setdefault(cell.doc_id, {})
            value = cell.reviewed_value if cell.reviewed_value is not None else cell.ai_value
            doc_bucket[cell.column_id] = value

        # Build table columns.
        columns: list[Column] = [Column(name=doc_id_column, column_type=ColumnType.TEXT)]
        for col_id, col_type in column_specs:
            columns.append(Column(name=col_id, column_type=col_type))

        # Build rows in doc_id insertion order.
        rows: list[tuple[Any, ...]] = []
        for doc_id, doc_bucket in rows_by_doc.items():
            row: list[Any] = [doc_id]
            for col_id, _ in column_specs:
                row.append(doc_bucket.get(col_id))
            rows.append(tuple(row))

        table = Table(
            name=table_name,
            columns=tuple(columns),
            rows=tuple(rows),
            row_count=len(rows),
        )
        return cls(tables=(table,))
