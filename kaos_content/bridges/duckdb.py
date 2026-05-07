"""Bridge between TabularDocument/Table and DuckDB.

Requires the ``duckdb`` optional extra::

    pip install kaos-content[duckdb]

DuckDB is the SQL engine for kaos-tabular. This bridge handles:
- Registering a Table as a queryable DuckDB view
- Converting DuckDB query results back to Table
- ColumnType ↔ DuckDB type mapping (full round-trip)

The pipeline is: Table → Polars DataFrame → DuckDB (via Arrow).
DuckDB and Polars share Apache Arrow as their interchange format,
so conversion is zero-copy when possible.

Usage::

    import duckdb
    from kaos_content.bridges.duckdb import (
        register_table, query_to_table, execute_query,
    )

    con = duckdb.connect()
    register_table(con, table, name="sales")
    result = query_to_table(con, "SELECT * FROM sales WHERE amount > 100", name="filtered")
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any
from weakref import WeakSet

logger = logging.getLogger(__name__)

try:
    import duckdb
except ImportError as exc:
    msg = "DuckDB is required for the DuckDB bridge. Install with: pip install kaos-content[duckdb]"
    raise ImportError(msg) from exc

from kaos_content.model.tabular import (  # noqa: E402  (after duckdb import-guard)
    Column,
    ColumnType,
    Table,
    TabularDocument,
)

# ---------------------------------------------------------------------------
# SQL safety
# ---------------------------------------------------------------------------
#
# DuckDB exposes file-system reads (read_csv, read_parquet, read_ndjson,
# read_json_objects, read_xlsx, read_duckdb, parquet_scan, iceberg_scan,
# glob, ...), database attachment (ATTACH), and extension management
# (INSTALL, LOAD) directly through SQL. A connection with external
# access enabled can read any file the OS user can read.
#
# **Sec-3 (security finding #2) reorganised the defence**:
#
#   1. **Engine-level sandbox** (PRIMARY control) —
#      :func:`create_safe_connection` issues
#      ``SET enable_external_access = false`` +
#      ``SET allow_unsigned_extensions = false`` +
#      ``SET lock_configuration = true``. The third statement is the
#      lynchpin: it makes the first two unmodifiable for the lifetime
#      of the connection, so user SQL cannot ``SET enable_external_access
#      = true`` to undo the sandbox. With the sandbox locked, every
#      file-reader / network / extension function DuckDB ships now or
#      in the future is blocked at the engine level — no regex update
#      required when DuckDB adds a new ``read_*``.
#
#   2. **Connection identity check** (STRUCTURAL control) —
#      :func:`execute_query` with ``untrusted_sql=True`` REQUIRES that
#      ``con`` was returned by :func:`create_safe_connection`. Connections
#      are tracked in a :class:`weakref.WeakSet`; passing a raw
#      ``duckdb.connect()`` connection raises ``ValueError`` with
#      guidance pointing at ``create_safe_connection()``.
#
#   3. **Telemetry deny-list** (DEFENCE-IN-DEPTH only) — the historical
#      regex matchers below remain available as :func:`_assert_sql_safe`
#      for callers that want an early reject, but :func:`execute_query`
#      no longer relies on them. The structural+sandbox combo is the
#      security boundary; the regex is for logging/visibility.

_DANGEROUS_SQL_FUNCS: tuple[str, ...] = (
    "read_csv",
    "read_csv_auto",
    "read_parquet",
    "read_json",
    "read_json_auto",
    "read_ndjson",
    "read_json_objects",
    "read_blob",
    "read_text",
    "read_xml",
    "read_xlsx",
    "read_duckdb",
    "parquet_scan",
    "iceberg_scan",
    "delta_scan",
    "glob",
    "scan_jsonl",
)
"""Functions that read from external files. Used by :func:`_assert_sql_safe`
(historical raise-on-match enforcement, kept for back-compat) and by
:func:`_warn_if_suspicious_sql` (telemetry-only logging from inside
:func:`execute_query`). The structural defence in
:func:`create_safe_connection` is what actually prevents these from
executing — this list is for early-warning visibility, not enforcement."""

_DANGEROUS_SQL_KEYWORDS: tuple[str, ...] = (
    "attach",
    "detach",
    "copy",
    "install",
    "load",
    "pragma",
    "export",
    "import",
)
"""Statements / keywords that escalate filesystem or extension access.
Same status as :data:`_DANGEROUS_SQL_FUNCS` — sandbox blocks them at
engine level; this list is for telemetry."""

_SQL_LINE_COMMENT_RE = re.compile(r"--[^\n]*")
_SQL_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)

# Tracks connections returned from :func:`create_safe_connection`. Used
# by :func:`execute_query` to enforce the structural rule that
# ``untrusted_sql=True`` requires a sandboxed connection. ``WeakSet`` so
# garbage-collected connections drop out automatically.
_SAFE_CONNECTIONS: WeakSet[duckdb.DuckDBPyConnection] = WeakSet()


def _strip_sql_comments(sql: str) -> str:
    """Remove SQL line and block comments so they can't hide a payload."""
    return _SQL_BLOCK_COMMENT_RE.sub("", _SQL_LINE_COMMENT_RE.sub("", sql))


def _assert_sql_safe(sql: str) -> None:
    """Raise ``ValueError`` if ``sql`` matches a deny-listed pattern.

    Historical enforcement entry point — kept for back-compat with
    callers that want an early reject before reaching the engine.
    :func:`execute_query` no longer calls this directly; the actual
    security boundary is the locked-sandbox connection from
    :func:`create_safe_connection`. See module docstring for the full
    layered model.

    Strips SQL comments first so comment-based bypasses (e.g.
    ``read_/*x*/csv('file')``) cannot evade the regex.
    """
    sanitised = _strip_sql_comments(sql).lower()
    for fn in _DANGEROUS_SQL_FUNCS:
        if re.search(rf"\b{re.escape(fn)}\s*\(", sanitised):
            msg = (
                f"SQL safety: function '{fn}()' is not allowed in untrusted "
                "queries. Pass untrusted_sql=False if the query is trusted."
            )
            raise ValueError(msg)
    for kw in _DANGEROUS_SQL_KEYWORDS:
        if re.search(rf"\b{re.escape(kw)}\b", sanitised):
            msg = (
                f"SQL safety: keyword '{kw}' is not allowed in untrusted "
                "queries. Pass untrusted_sql=False if the query is trusted."
            )
            raise ValueError(msg)


def _warn_if_suspicious_sql(sql: str) -> None:
    """Log a warning if SQL contains deny-listed patterns. Does NOT raise.

    Used by :func:`execute_query` for telemetry — the structural defence
    (sandboxed connection) is the actual security boundary. We log so
    operators see when an LLM/agent attempts a file-reader, even if the
    sandbox is what stops it from succeeding.
    """
    sanitised = _strip_sql_comments(sql).lower()
    for fn in _DANGEROUS_SQL_FUNCS:
        if re.search(rf"\b{re.escape(fn)}\s*\(", sanitised):
            logger.warning(
                "SQL contains suspicious function '%s()' — sandbox should block at "
                "engine level, but caller may want to refuse upstream. SQL: %r",
                fn,
                sql[:200],
            )
            return
    for kw in _DANGEROUS_SQL_KEYWORDS:
        if re.search(rf"\b{re.escape(kw)}\b", sanitised):
            logger.warning(
                "SQL contains suspicious keyword '%s' — sandbox should block at "
                "engine level, but caller may want to refuse upstream. SQL: %r",
                kw,
                sql[:200],
            )
            return


def create_safe_connection(*, allow_unsigned_extensions: bool = False) -> duckdb.DuckDBPyConnection:
    """Create an in-memory DuckDB connection sandboxed against file access.

    Issues three engine-level pragmas, in this order:

    1. ``SET enable_external_access = false`` — blocks read_csv,
       read_parquet, read_ndjson, read_json_objects, read_xlsx,
       read_duckdb, parquet_scan, iceberg_scan, glob, ATTACH, COPY,
       INSTALL, LOAD, PRAGMA, EXPORT/IMPORT DATABASE, ``SET
       file_search_path``, and every other DuckDB function or
       statement that touches the local filesystem or network.
    2. ``SET allow_unsigned_extensions = false`` (when
       ``allow_unsigned_extensions=False``, the default) — extensions
       can re-enable filesystem access; block unsigned ones.
    3. ``SET lock_configuration = true`` — **the lynchpin**. Without
       this, untrusted SQL could ``SET enable_external_access = true``
       to undo the sandbox. With it, *no* SET statement succeeds for
       the rest of the connection's life. CREATE TABLE / INSERT /
       SELECT / .register() / .from_df() / etc. all still work.

    The connection is then registered in :data:`_SAFE_CONNECTIONS` so
    :func:`execute_query` can verify, via identity, that callers passing
    ``untrusted_sql=True`` actually got their connection from this
    factory rather than a raw ``duckdb.connect()``.

    Use this connection when running SQL authored by an LLM, agent, or
    other untrusted source.
    """
    con = duckdb.connect(":memory:")
    con.execute("SET enable_external_access = false")
    if not allow_unsigned_extensions:
        con.execute("SET allow_unsigned_extensions = false")
    # Lock LAST. Once set, no further SET statement succeeds (DuckDB
    # raises). This prevents any SQL — trusted or otherwise — from
    # re-enabling external access, loading extensions, or otherwise
    # tampering with the sandbox.
    con.execute("SET lock_configuration = true")
    _SAFE_CONNECTIONS.add(con)
    return con


def _is_safe_connection(con: duckdb.DuckDBPyConnection) -> bool:
    """True iff ``con`` was returned by :func:`create_safe_connection`.

    Membership is tracked via :data:`_SAFE_CONNECTIONS` (a ``WeakSet``).
    Used by :func:`execute_query` to enforce the structural rule that
    ``untrusted_sql=True`` requires a sandboxed connection.
    """
    return con in _SAFE_CONNECTIONS


# ---------------------------------------------------------------------------
# ColumnType → DuckDB SQL type mapping
# ---------------------------------------------------------------------------

_COLUMN_TYPE_TO_DUCKDB: dict[ColumnType, str] = {
    ColumnType.TEXT: "VARCHAR",
    ColumnType.INTEGER: "BIGINT",
    ColumnType.FLOAT: "DOUBLE",
    ColumnType.BOOLEAN: "BOOLEAN",
    ColumnType.DATE: "DATE",
    ColumnType.TIME: "TIME",
    ColumnType.DATETIME: "TIMESTAMP",
    ColumnType.NULL: "VARCHAR",  # DuckDB doesn't have a NULL type; VARCHAR is safest
    ColumnType.DECIMAL: "DECIMAL(38,10)",
    ColumnType.BINARY: "BLOB",
    ColumnType.DURATION: "INTERVAL",
    ColumnType.LIST: "VARCHAR",  # JSON-serialized
    ColumnType.STRUCT: "VARCHAR",  # JSON-serialized
    # Tier 4: extraction-specific (KAOS WS-TR). Widened to logical equivalents.
    ColumnType.VERBATIM_QUOTE: "VARCHAR",
    ColumnType.MONEY: "VARCHAR",  # JSON-serialized {amount, currency}
    ColumnType.SCORE: "BIGINT",
    ColumnType.ENTITY_ROLE: "VARCHAR",  # JSON-serialized {name, role, entity_type}
}


def _duckdb_type_to_column_type(type_str: str) -> ColumnType:
    """Map a DuckDB type string to ColumnType.

    DuckDB type strings come from ``description`` on cursor results
    or from ``DESCRIBE table``. They look like: ``VARCHAR``, ``BIGINT``,
    ``DECIMAL(18,2)``, ``TIMESTAMP``, etc.
    """
    t = type_str.upper().strip()

    # Exact matches
    if t in ("VARCHAR", "TEXT", "STRING", "CHAR", "BPCHAR"):
        return ColumnType.TEXT
    if t in ("BIGINT", "INT8", "LONG", "INT64", "SIGNED"):
        return ColumnType.INTEGER
    if t in ("INTEGER", "INT4", "INT", "INT32"):
        return ColumnType.INTEGER
    if t in ("SMALLINT", "INT2", "SHORT", "INT16"):
        return ColumnType.INTEGER
    if t in ("TINYINT", "INT1"):
        return ColumnType.INTEGER
    if t in ("UBIGINT", "UINTEGER", "USMALLINT", "UTINYINT"):
        return ColumnType.INTEGER
    if t in ("HUGEINT", "UHUGEINT"):
        return ColumnType.INTEGER
    if t in ("DOUBLE", "FLOAT8", "NUMERIC", "FLOAT"):
        return ColumnType.FLOAT
    if t in ("REAL", "FLOAT4"):
        return ColumnType.FLOAT
    if t in ("BOOLEAN", "BOOL", "LOGICAL"):
        return ColumnType.BOOLEAN
    if t == "DATE":
        return ColumnType.DATE
    if t == "TIME":
        return ColumnType.TIME
    if t in ("TIMESTAMP", "DATETIME", "TIMESTAMP_S", "TIMESTAMP_MS", "TIMESTAMP_NS"):
        return ColumnType.DATETIME
    if t in ("TIMESTAMP WITH TIME ZONE", "TIMESTAMPTZ", "TIMESTAMP_TZ"):
        return ColumnType.DATETIME
    if t == "BLOB" or t == "BYTEA":
        return ColumnType.BINARY
    if t == "INTERVAL":
        return ColumnType.DURATION
    if t == "NULL":
        return ColumnType.NULL

    # Parameterized types
    if t.startswith("DECIMAL") or t.startswith("NUMERIC"):
        return ColumnType.DECIMAL
    if t.startswith("VARCHAR") or t.startswith("CHAR"):
        return ColumnType.TEXT
    if t.startswith("TIMESTAMP"):
        return ColumnType.DATETIME
    if t.endswith("[]") or t.startswith("LIST"):
        return ColumnType.LIST
    if t.startswith("STRUCT") or t.startswith("MAP"):
        return ColumnType.STRUCT

    return ColumnType.TEXT


# ---------------------------------------------------------------------------
# Value conversion
# ---------------------------------------------------------------------------


def _python_to_duckdb_value(value: Any, ct: ColumnType) -> Any:
    """Convert a Python value for DuckDB ingestion."""
    if value is None:
        return None
    if ct in (ColumnType.LIST, ColumnType.STRUCT):
        import json

        return json.dumps(value, default=str, ensure_ascii=False)
    if (
        ct == ColumnType.FLOAT
        and isinstance(value, float)
        and (math.isnan(value) or math.isinf(value))
    ):
        return None  # DuckDB doesn't handle NaN/inf in all contexts
    return value


def _duckdb_to_python_value(value: Any, ct: ColumnType) -> Any:
    """Convert a DuckDB result value back to Python native."""
    if value is None:
        return None
    if ct in (ColumnType.LIST, ColumnType.STRUCT) and isinstance(value, str):
        import json

        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
    return value


# ---------------------------------------------------------------------------
# Registration: Table → DuckDB
# ---------------------------------------------------------------------------


def register_table(
    con: duckdb.DuckDBPyConnection,
    table: Table,
    *,
    name: str | None = None,
) -> str:
    """Register a Table as a queryable DuckDB view.

    Uses the Polars bridge internally (Table → Polars DataFrame → DuckDB).
    This is the most robust path because Polars ↔ DuckDB share Arrow.

    Args:
        con: DuckDB connection.
        table: The Table to register.
        name: View name in DuckDB. Defaults to ``table.name``.

    Returns:
        The registered view name.
    """
    view_name = name or table.name

    try:
        from kaos_content.bridges.polars import table_to_polars

        df = table_to_polars(table)
        con.register(view_name, df)
    except ImportError:
        # Fallback: build CREATE TABLE + INSERT from Python values
        _register_table_fallback(con, table, view_name)

    return view_name


def _register_table_fallback(
    con: duckdb.DuckDBPyConnection,
    table: Table,
    view_name: str,
) -> None:
    """Fallback registration without Polars — uses raw SQL."""
    # Use _quote_ident() so a column name containing `"` is properly
    # escaped (`"` -> `""`). The previous implementation was
    # `f'"{c.name}"'` which would break out of the identifier on
    # any embedded quote — a SQL injection path through user-supplied
    # column names.
    col_defs = ", ".join(
        f"{_quote_ident(c.name)} {_COLUMN_TYPE_TO_DUCKDB[c.column_type]}" for c in table.columns
    )
    con.execute(f"CREATE OR REPLACE TABLE {_quote_ident(view_name)} ({col_defs})")

    if table.rows:
        placeholders = ", ".join("?" for _ in table.columns)
        # ``_quote_ident`` is the only string interpolation here; row
        # values flow through the ``?`` placeholders below. nosec B608.
        insert_sql = f"INSERT INTO {_quote_ident(view_name)} VALUES ({placeholders})"  # nosec B608
        for row in table.rows:
            converted = [
                _python_to_duckdb_value(
                    row[i] if i < len(row) else None, table.columns[i].column_type
                )
                for i in range(len(table.columns))
            ]
            con.execute(insert_sql, converted)


def register_document(
    con: duckdb.DuckDBPyConnection,
    doc: TabularDocument,
    *,
    prefix: str = "",
) -> list[str]:
    """Register all tables from a TabularDocument.

    Args:
        con: DuckDB connection.
        doc: The document whose tables to register.
        prefix: Optional prefix for table names (e.g., "sheet_").

    Returns:
        List of registered view names.
    """
    names = []
    for table in doc.tables:
        view_name = f"{prefix}{table.name}" if prefix else table.name
        register_table(con, table, name=view_name)
        names.append(view_name)
    return names


# ---------------------------------------------------------------------------
# Query: DuckDB → Table
# ---------------------------------------------------------------------------


def query_to_table(
    con: duckdb.DuckDBPyConnection,
    sql: str,
    *,
    name: str = "result",
) -> Table:
    """Execute a SQL query and return the result as a Table.

    Args:
        con: DuckDB connection.
        sql: SQL query to execute.
        name: Name for the resulting Table.

    Returns:
        Table with typed columns inferred from the DuckDB result.
    """
    result = con.execute(sql)
    description = result.description

    if description is None:
        return Table(name=name)

    # Build columns from result description
    # Note: description[1] is a DuckDBPyType object, not a string
    columns: list[Column] = []
    col_types: list[ColumnType] = []
    for col_desc in description:
        col_name = col_desc[0]
        duckdb_type = str(col_desc[1])  # DuckDBPyType → string
        ct = _duckdb_type_to_column_type(duckdb_type)
        columns.append(Column(name=col_name, column_type=ct))
        col_types.append(ct)

    # Fetch rows
    raw_rows = result.fetchall()
    rows: list[tuple[Any, ...]] = []
    for raw_row in raw_rows:
        converted = tuple(
            _duckdb_to_python_value(raw_row[i], col_types[i]) for i in range(len(col_types))
        )
        rows.append(converted)

    return Table(
        name=name,
        columns=tuple(columns),
        rows=tuple(rows),
        row_count=len(rows),
    )


def execute_query(
    con: duckdb.DuckDBPyConnection,
    sql: str,
    *,
    max_rows: int = 1000,
    untrusted_sql: bool = True,
) -> Table:
    """Execute SQL with a row limit. Wraps the query in a LIMIT if needed.

    The previous implementation accepted ANY SQL, including
    ``SELECT * FROM read_csv('/etc/passwd')`` — DuckDB can read local
    files when external access is enabled. This is the application-
    level half of the layered defence. The engine-level half is
    :func:`create_safe_connection`, which disables external access on
    the connection itself.

    Args:
        con: DuckDB connection. For untrusted SQL, pass a connection
            from ``create_safe_connection()``.
        sql: SQL query.
        max_rows: Maximum rows to return. Default 1000, hard cap 10000.
        untrusted_sql: If True (default), the SQL is regex-rejected if
            it contains any of the deny-listed file-read or extension
            functions/keywords (see ``_DANGEROUS_SQL_FUNCS`` and
            ``_DANGEROUS_SQL_KEYWORDS``). Set to False ONLY when the
            SQL is authored by trusted code; never set to False for
            SQL coming from an LLM, agent, or external user.

    Raises:
        ValueError: If ``untrusted_sql=True`` and the SQL matches a
            deny-listed pattern.

    Returns:
        Table with results. ``row_count`` reflects the total if a
        ``COUNT(*)`` can be cheaply determined, otherwise equals
        ``len(rows)``.
    """
    if untrusted_sql:
        # Sec-3 structural enforcement: untrusted SQL requires a connection
        # from create_safe_connection() — that's the lock_configuration-
        # protected sandbox where every file-reader / network / extension
        # function is refused at the engine level. A raw duckdb.connect()
        # passed here would (pre-Sec-3) have relied on the regex deny-
        # list, which is perpetually behind every new DuckDB release's
        # table functions (read_ndjson, parquet_scan, iceberg_scan, ...).
        if not _is_safe_connection(con):
            msg = (
                "execute_query(untrusted_sql=True) requires a sandboxed "
                "connection. Pass a connection from "
                "kaos_content.bridges.duckdb.create_safe_connection() — "
                "raw duckdb.connect() connections do not have the engine-"
                "level filesystem/network/extension sandbox enabled and "
                "would let untrusted SQL read any file the OS user can "
                "read. If the SQL is authored by trusted code (NEVER an "
                "LLM, agent, or external user), pass untrusted_sql=False."
            )
            raise ValueError(msg)
        # Defence-in-depth: log suspicious patterns. This is for
        # operator visibility — the sandbox above is what actually
        # blocks the query at the engine level.
        _warn_if_suspicious_sql(sql)
    capped = min(max_rows, 10_000)
    # ``con`` has been verified sandboxed (when untrusted_sql=True) so
    # any file-reader inside ``sql`` will be rejected by DuckDB's locked
    # ``enable_external_access=false`` configuration before producing
    # results. ``capped`` is an int we just bounded.
    # nosec B608.
    limited_sql = f"SELECT * FROM ({sql}) AS _q LIMIT {capped}"  # nosec B608
    return query_to_table(con, limited_sql, name="result")


def describe_table(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
) -> dict[str, Any]:
    """Describe a registered table: columns, types, row count, samples.

    Returns a dict suitable for MCP tool responses.
    """
    # Column info via DESCRIBE
    # table_name is routed through _quote_ident. nosec B608.
    desc_result = con.execute(
        f"DESCRIBE {_quote_ident(table_name)}"  # nosec B608
    ).fetchall()
    columns = []
    for row in desc_result:
        col_name = row[0]
        col_type_str = row[1]
        nullable = row[2] == "YES" if len(row) > 2 else True
        ct = _duckdb_type_to_column_type(col_type_str)
        columns.append(
            {
                "name": col_name,
                "type": ct.value,
                "duckdb_type": col_type_str,
                "nullable": nullable,
            }
        )

    # Row count — table_name is routed through _quote_ident. nosec B608.
    count_result = con.execute(
        f"SELECT COUNT(*) FROM {_quote_ident(table_name)}"  # nosec B608
    ).fetchone()
    row_count = count_result[0] if count_result else 0

    # Sample (first 5 rows) — table_name is routed through _quote_ident. nosec B608.
    sample = query_to_table(
        con,
        f"SELECT * FROM {_quote_ident(table_name)} LIMIT 5",  # nosec B608
        name=table_name,
    )

    return {
        "name": table_name,
        "row_count": row_count,
        "column_count": len(columns),
        "columns": columns,
        "sample_rows": len(sample.rows),
    }


def list_tables(con: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    """List all tables/views in the DuckDB connection.

    Returns a list of dicts with name, type (TABLE/VIEW), and column count.
    """
    result = con.execute(
        "SELECT table_name, table_type FROM information_schema.tables "
        "WHERE table_schema = 'main' ORDER BY table_name"
    ).fetchall()

    tables = []
    for row in result:
        table_name = row[0]
        table_type = row[1]
        # Parameterised query — table_name comes from
        # information_schema and is therefore implicitly trusted, but
        # we still parameterise to (a) avoid the f-string SQL anti-
        # pattern in tracked source and (b) survive future refactors
        # where the source of table_name may change.
        cols = con.execute(
            "SELECT COUNT(*) FROM information_schema.columns "
            "WHERE table_name = ? AND table_schema = 'main'",
            [table_name],
        ).fetchone()
        col_count = cols[0] if cols else 0
        tables.append(
            {
                "name": table_name,
                "type": table_type,
                "column_count": col_count,
            }
        )
    return tables


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _quote_ident(name: str) -> str:
    """Quote a DuckDB identifier to handle special characters."""
    escaped = name.replace('"', '""')
    return f'"{escaped}"'
