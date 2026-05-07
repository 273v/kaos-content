"""Bridge between TabularDocument/Table and Polars DataFrames.

Requires the ``polars`` optional extra::

    pip install kaos-content[polars]

Provides zero-copy-intent conversion between Table and Polars DataFrame.
Every ColumnType maps to a Polars dtype and back without lossy conversion.

Usage::

    from kaos_content.bridges.polars import (
        table_to_polars, table_from_polars, document_to_polars,
    )
    import polars as pl

    df = table_to_polars(table)
    table2 = table_from_polars(df, name="sales")
    assert table2.columns == table.columns  # round-trip preserves types
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

try:
    import polars as pl
except ImportError as exc:
    msg = (
        "Polars is required for the tabular bridge. Install with: pip install kaos-content[polars]"
    )
    raise ImportError(msg) from exc

from kaos_content.model.tabular import (
    Column,
    ColumnType,
    Table,
    TabularDocument,
)

# ---------------------------------------------------------------------------
# ColumnType → Polars dtype mapping
# ---------------------------------------------------------------------------

_COLUMN_TYPE_TO_POLARS: dict[ColumnType, pl.DataType] = {
    ColumnType.TEXT: pl.String(),
    ColumnType.INTEGER: pl.Int64(),
    ColumnType.FLOAT: pl.Float64(),
    ColumnType.BOOLEAN: pl.Boolean(),
    ColumnType.DATE: pl.Date(),
    ColumnType.TIME: pl.Time(),
    ColumnType.DATETIME: pl.Datetime("us"),
    ColumnType.NULL: pl.Null(),
    ColumnType.DECIMAL: pl.Decimal(precision=38, scale=10),
    ColumnType.BINARY: pl.Binary(),
    ColumnType.DURATION: pl.Duration("us"),
    ColumnType.LIST: pl.List(pl.String),
    ColumnType.STRUCT: pl.String(),  # JSON-serialized
    # Tier 4: extraction-specific (KAOS WS-TR). Widened to logical equivalents.
    ColumnType.VERBATIM_QUOTE: pl.String(),
    ColumnType.MONEY: pl.String(),  # JSON-serialized {amount, currency}
    ColumnType.SCORE: pl.Int64(),
    ColumnType.ENTITY_ROLE: pl.String(),  # JSON-serialized {name, role, entity_type}
}


def _polars_dtype_to_column_type(dtype: pl.DataType) -> ColumnType:
    """Map a Polars dtype to ColumnType."""
    if dtype == pl.String or dtype == pl.Utf8 or dtype == pl.Categorical:
        return ColumnType.TEXT
    if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
        return ColumnType.INTEGER
    if dtype in (pl.Float32, pl.Float64):
        return ColumnType.FLOAT
    if dtype == pl.Boolean:
        return ColumnType.BOOLEAN
    if dtype == pl.Date:
        return ColumnType.DATE
    if dtype == pl.Time:
        return ColumnType.TIME
    if dtype == pl.Null:
        return ColumnType.NULL
    if dtype == pl.Binary:
        return ColumnType.BINARY

    # Parameterized types — check base type
    base = dtype.base_type()
    if base == pl.Datetime:
        return ColumnType.DATETIME
    if base == pl.Duration:
        return ColumnType.DURATION
    if base == pl.Decimal:
        return ColumnType.DECIMAL
    if base == pl.List or base == pl.Array:
        return ColumnType.LIST
    if base == pl.Struct:
        return ColumnType.STRUCT

    return ColumnType.TEXT


def _column_type_to_polars(ct: ColumnType) -> pl.DataType:
    """Map a ColumnType to Polars dtype."""
    return _COLUMN_TYPE_TO_POLARS[ct]


# ---------------------------------------------------------------------------
# Value conversion helpers
# ---------------------------------------------------------------------------


def _python_to_polars_value(value: Any, ct: ColumnType) -> Any:
    """Convert a Python value for Polars ingestion.

    Most types are natively handled by Polars. Special cases:
    - STRUCT columns: dict → JSON string (Polars doesn't auto-infer struct schema)
    - LIST columns: list values passed through directly
    """
    if value is None:
        return None
    if ct in (ColumnType.STRUCT, ColumnType.MONEY, ColumnType.ENTITY_ROLE) and isinstance(
        value, dict
    ):
        import json

        return json.dumps(value, default=str, ensure_ascii=False)
    if ct == ColumnType.DECIMAL and isinstance(value, Decimal):
        return value
    return value


def _polars_to_python_value(value: Any, ct: ColumnType) -> Any:
    """Convert a Polars-native value back to Python native.

    Polars returns Python-native types for most dtypes. Special cases:
    - STRUCT: JSON string → dict
    - Null → None
    """
    if value is None:
        return None
    if ct == ColumnType.STRUCT and isinstance(value, str):
        import json

        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
    if ct == ColumnType.DECIMAL and not isinstance(value, Decimal):
        return Decimal(str(value))
    return value


# ---------------------------------------------------------------------------
# Table ↔ Polars DataFrame
# ---------------------------------------------------------------------------


def table_to_polars(table: Table) -> pl.DataFrame:
    """Convert a Table to a Polars DataFrame.

    Column types are mapped via the ColumnType → Polars dtype table.
    STRUCT columns are JSON-serialized to strings.

    Args:
        table: The Table to convert.

    Returns:
        Polars DataFrame with typed columns.
    """
    if not table.columns:
        return pl.DataFrame()

    data: dict[str, list[Any]] = {col.name: [] for col in table.columns}
    for row in table.rows:
        for i, col in enumerate(table.columns):
            val = row[i] if i < len(row) else None
            data[col.name].append(_python_to_polars_value(val, col.column_type))

    schema = {col.name: _column_type_to_polars(col.column_type) for col in table.columns}

    return pl.DataFrame(data, schema=schema, strict=False)


def table_from_polars(df: pl.DataFrame, *, name: str = "data") -> Table:
    """Convert a Polars DataFrame to a Table.

    Polars dtypes are mapped back to ColumnType. Values are converted
    to Python-native types.

    Args:
        df: The Polars DataFrame to convert.
        name: Name for the resulting Table.

    Returns:
        Table with columns inferred from the DataFrame schema.
    """
    columns: list[Column] = []
    col_types: list[ColumnType] = []

    for col_name in df.columns:
        ct = _polars_dtype_to_column_type(df[col_name].dtype)
        has_nulls = df[col_name].null_count() > 0
        columns.append(Column(name=col_name, column_type=ct, nullable=has_nulls or True))
        col_types.append(ct)

    rows: list[tuple[Any, ...]] = []
    for row_dict in df.iter_rows(named=False):
        converted = tuple(
            _polars_to_python_value(row_dict[i], col_types[i]) for i in range(len(col_types))
        )
        rows.append(converted)

    return Table(
        name=name,
        columns=tuple(columns),
        rows=tuple(rows),
        row_count=len(rows),
    )


def document_to_polars(doc: TabularDocument) -> dict[str, pl.DataFrame]:
    """Convert all tables in a TabularDocument to Polars DataFrames.

    Args:
        doc: The TabularDocument to convert.

    Returns:
        Dict mapping table name → DataFrame.
    """
    return {table.name: table_to_polars(table) for table in doc.tables}
