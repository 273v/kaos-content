"""TabularView — hierarchical navigation over TabularDocument.

Peer to DocumentView for ContentDocument. Provides lazy-cached views
for navigating tabular data by table, column, and row range.

Usage::

    from kaos_content.views.tabular_view import TabularView

    view = TabularView(doc)
    view.table_names          # → ('Sheet1', 'Sheet2')
    view.table_schema('Sheet1')  # → column names, types, nullability
    view.table_head('Sheet1', n=5)  # → first 5 rows as Table
    view.table_slice('Sheet1', 10, 20)  # → rows 10-19 as Table
    view.column_stats('Sheet1', 'amount')  # → min, max, null count, uniques
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from kaos_content.model.tabular import (
    ColumnType,
    Table,
    TabularDocument,
)


@dataclass(frozen=True, slots=True)
class ColumnStats:
    """Statistics for a single column."""

    name: str
    column_type: str
    nullable: bool
    null_count: int
    non_null_count: int
    unique_count: int | None  # None if unhashable values
    min_value: Any | None
    max_value: Any | None
    sample_values: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TableInfo:
    """Summary info for a table — cheap to compute."""

    name: str
    row_count: int
    rows_loaded: int
    column_count: int
    column_names: tuple[str, ...]
    column_types: tuple[str, ...]


class TabularView:
    """Computed views over a TabularDocument.

    All views are lazily computed on first access and cached.
    The document itself is never modified.
    """

    __slots__ = ("_document", "_infos", "_table_map")

    def __init__(self, document: TabularDocument) -> None:
        self._document = document
        self._table_map: dict[str, Table] | None = None
        self._infos: tuple[TableInfo, ...] | None = None

    # -- Properties --------------------------------------------------------

    @property
    def document(self) -> TabularDocument:
        """Underlying TabularDocument the view wraps."""
        return self._document

    @property
    def table_count(self) -> int:
        """Number of tables in the wrapped document."""
        return len(self._document.tables)

    @property
    def table_names(self) -> tuple[str, ...]:
        """Names of all tables, in document order."""
        return self._document.table_names()

    @property
    def total_rows(self) -> int:
        """Sum of row counts across every table."""
        return sum(t.row_count for t in self._document.tables)

    @property
    def table_infos(self) -> tuple[TableInfo, ...]:
        """Summary info for all tables."""
        if self._infos is None:
            self._infos = tuple(self._compute_info(t) for t in self._document.tables)
        return self._infos

    # -- Table access ------------------------------------------------------

    def get_table(self, name: str) -> Table:
        """Get a table by name."""
        if self._table_map is None:
            self._table_map = {t.name: t for t in self._document.tables}
        table = self._table_map.get(name)
        if table is None:
            available = ", ".join(self.table_names)
            msg = f"Table not found: {name!r}. Available: {available}"
            raise KeyError(msg)
        return table

    def table_head(self, name: str, n: int = 5) -> Table:
        """First n rows of a table."""
        return self.get_table(name).head(n)

    def table_slice(self, name: str, start: int, end: int | None = None) -> Table:
        """Row range [start:end] of a table."""
        return self.get_table(name).slice(start, end)

    # -- Schema ------------------------------------------------------------

    def table_schema(self, name: str) -> dict[str, Any]:
        """Detailed schema for a table."""
        table = self.get_table(name)
        return {
            "name": table.name,
            "row_count": table.row_count,
            "rows_loaded": len(table.rows),
            "column_count": len(table.columns),
            "columns": [
                {
                    "name": c.name,
                    "type": c.column_type.value,
                    "nullable": c.nullable,
                    "metadata": c.metadata if c.metadata else None,
                }
                for c in table.columns
            ],
            "metadata": table.metadata if table.metadata else None,
        }

    # -- Column stats ------------------------------------------------------

    def column_stats(self, table_name: str, column_name: str) -> ColumnStats:
        """Compute statistics for a single column."""
        table = self.get_table(table_name)
        col_idx = table._column_index(column_name)
        col = table.columns[col_idx]

        values = [row[col_idx] for row in table.rows if col_idx < len(row)]
        null_count = sum(1 for v in values if v is None)
        non_null = [v for v in values if v is not None]

        # Unique count (skip for unhashable types)
        import contextlib

        unique: int | None = None
        with contextlib.suppress(TypeError):
            unique = len(set(non_null))

        # Min/max (skip for non-comparable types)
        min_val: Any = None
        max_val: Any = None
        if non_null and col.column_type in _COMPARABLE_TYPES:
            try:
                comparable = [v for v in non_null if not _is_nan(v)]
                if comparable:
                    min_val = min(comparable)
                    max_val = max(comparable)
            except TypeError:
                pass

        samples = tuple(str(v) for v in non_null[:5])

        return ColumnStats(
            name=col.name,
            column_type=col.column_type.value,
            nullable=col.nullable,
            null_count=null_count,
            non_null_count=len(non_null),
            unique_count=unique,
            min_value=min_val,
            max_value=max_val,
            sample_values=samples,
        )

    def all_column_stats(self, table_name: str) -> list[ColumnStats]:
        """Compute stats for all columns in a table."""
        table = self.get_table(table_name)
        return [self.column_stats(table_name, c.name) for c in table.columns]

    # -- Serialization helpers (for MCP resources) -------------------------

    def summary_dict(self) -> dict[str, Any]:
        """Full document summary for MCP resource."""
        return {
            "title": self._document.metadata.title,
            "table_count": self.table_count,
            "total_rows": self.total_rows,
            "tables": [
                {
                    "name": info.name,
                    "row_count": info.row_count,
                    "rows_loaded": info.rows_loaded,
                    "column_count": info.column_count,
                    "columns": [
                        {"name": n, "type": t}
                        for n, t in zip(info.column_names, info.column_types, strict=True)
                    ],
                }
                for info in self.table_infos
            ],
        }

    # -- Internal ----------------------------------------------------------

    @staticmethod
    def _compute_info(table: Table) -> TableInfo:
        return TableInfo(
            name=table.name,
            row_count=table.row_count,
            rows_loaded=len(table.rows),
            column_count=len(table.columns),
            column_names=table.column_names(),
            column_types=tuple(c.column_type.value for c in table.columns),
        )


_COMPARABLE_TYPES = frozenset(
    {
        ColumnType.TEXT,
        ColumnType.INTEGER,
        ColumnType.FLOAT,
        ColumnType.DATE,
        ColumnType.TIME,
        ColumnType.DATETIME,
        ColumnType.DECIMAL,
    }
)


def _is_nan(v: Any) -> bool:
    return isinstance(v, float) and math.isnan(v)
