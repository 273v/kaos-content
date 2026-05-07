"""DuckDB SQL-safety regression tests.

Pins the safe-by-default contract for the DuckDB bridge introduced
in 0.1.0a1:

- ``execute_query(..., untrusted_sql=True)`` (the default) rejects SQL
  containing file-read functions (``read_csv``, ``read_parquet``, ...)
  or filesystem/extension keywords (``attach``, ``copy``, ``install``,
  ``load``, ``pragma``).
- ``create_safe_connection()`` returns an in-memory DuckDB connection
  with ``enable_external_access = false`` and unsigned-extension loads
  disabled — the engine-level half of the layered defence.
- ``_register_table_fallback`` quotes column identifiers via
  ``_quote_ident`` so embedded ``"`` characters are correctly escaped.
- ``list_tables`` uses parameterised queries instead of f-string
  interpolation.

Audit findings addressed: C5 (execute_query takes arbitrary SQL),
C6 (unescaped table_name in list_tables), C7 (column identifier not
properly quoted in CREATE TABLE).
"""

from __future__ import annotations

import pytest

duckdb = pytest.importorskip("duckdb")

from kaos_content.bridges.duckdb import (  # noqa: E402
    _DANGEROUS_SQL_FUNCS,
    _DANGEROUS_SQL_KEYWORDS,
    _assert_sql_safe,
    _register_table_fallback,
    create_safe_connection,
    execute_query,
    list_tables,
    register_table,
)
from kaos_content.model.tabular import Column, ColumnType, Table  # noqa: E402

# ────────────────────────────────────────────────────────────────────
# _assert_sql_safe — application-level deny-list
# ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("fn", _DANGEROUS_SQL_FUNCS)
def test_assert_sql_safe_rejects_each_dangerous_function(fn: str) -> None:
    """Every entry in _DANGEROUS_SQL_FUNCS is rejected when used as a call."""
    with pytest.raises(ValueError, match="not allowed"):
        _assert_sql_safe(f"SELECT * FROM {fn}('/etc/passwd')")


@pytest.mark.parametrize("kw", _DANGEROUS_SQL_KEYWORDS)
def test_assert_sql_safe_rejects_each_dangerous_keyword(kw: str) -> None:
    """Every entry in _DANGEROUS_SQL_KEYWORDS is rejected as a token."""
    with pytest.raises(ValueError, match="not allowed"):
        _assert_sql_safe(f"{kw} foo")


def test_assert_sql_safe_passes_legitimate_query() -> None:
    """A plain SELECT must pass."""
    _assert_sql_safe("SELECT id, name FROM users WHERE active = true")
    _assert_sql_safe("SELECT COUNT(*) FROM orders WHERE created_at > '2024-01-01'")
    _assert_sql_safe("SELECT * FROM (SELECT 1 AS x) AS sub")


def test_assert_sql_safe_strips_line_comments() -> None:
    """SQL line comments cannot hide a payload from the filter."""
    with pytest.raises(ValueError, match="read_csv"):
        # The deny-listed call is in plain SQL, not a comment.
        # A naive implementation that reads only "SELECT 1" would miss it.
        _assert_sql_safe("SELECT 1 -- harmless\nUNION SELECT * FROM read_csv('/etc/passwd')")


def test_assert_sql_safe_strips_block_comments() -> None:
    """Block comments cannot evade the deny-list either."""
    with pytest.raises(ValueError, match="attach"):
        _assert_sql_safe("SELECT 1; /* hidden */ ATTACH 'sneak.db'")


def test_assert_sql_safe_does_not_match_substrings() -> None:
    """A column or table name CONTAINING a deny-listed substring must pass."""
    # 'load' is a deny-listed keyword; 'unloaded' is a legitimate name.
    _assert_sql_safe("SELECT unloaded FROM products")
    _assert_sql_safe("SELECT * FROM copy_log")  # 'copy_log' is not 'copy'
    # 'read_csv' is only rejected when followed by '('.
    _assert_sql_safe("SELECT read_csv_count FROM stats")


# ────────────────────────────────────────────────────────────────────
# execute_query — wired to the deny-list
# ────────────────────────────────────────────────────────────────────


def _make_test_table() -> Table:
    """Helper: a tiny Table for register/query tests."""
    return Table(
        name="t",
        columns=(
            Column(name="id", column_type=ColumnType.INTEGER),
            Column(name="x", column_type=ColumnType.TEXT),
        ),
        rows=((1, "a"), (2, "b"), (3, "c")),
        row_count=3,
    )


def test_execute_query_rejects_unsafe_by_default() -> None:
    # Sec-3 (security finding #2) made ``execute_query(untrusted_sql=True)``
    # require a sandboxed connection. Passing a raw ``duckdb.connect()``
    # connection now raises before the regex deny-list ever runs —
    # the structural defence is the security boundary.
    con = duckdb.connect(":memory:")
    register_table(con, _make_test_table(), name="t")
    with pytest.raises(ValueError, match="requires a sandboxed connection"):
        execute_query(con, "SELECT * FROM read_csv('/etc/passwd')")


def test_execute_query_allows_unsafe_when_explicitly_trusted() -> None:
    """untrusted_sql=False is the documented escape hatch for trusted code."""
    con = duckdb.connect(":memory:")
    register_table(con, _make_test_table(), name="t")
    # Trusted SQL — the structural check is bypassed by untrusted_sql=False.
    # The query is benign here (just exercises the escape hatch); we only
    # need it to not raise.
    result = execute_query(con, "SELECT id FROM t", untrusted_sql=False)
    assert result.row_count == 3


def test_execute_query_allows_legitimate_select() -> None:
    # Two valid configurations after Sec-3:
    #
    #   (a) sandboxed connection + ``untrusted_sql=True`` (default,
    #       recommended for any caller running LLM/agent-authored SQL).
    #   (b) raw connection + ``untrusted_sql=False`` (the documented
    #       escape hatch for trusted code).
    #
    # The combo "raw connection + untrusted_sql=True" no longer passes;
    # the structural check rejects it with a clear error.
    safe_con = create_safe_connection()
    register_table(safe_con, _make_test_table(), name="t")
    result = execute_query(safe_con, "SELECT x FROM t WHERE id > 1")
    assert result.row_count == 2

    raw_con = duckdb.connect(":memory:")
    register_table(raw_con, _make_test_table(), name="t")
    result = execute_query(raw_con, "SELECT x FROM t WHERE id > 1", untrusted_sql=False)
    assert result.row_count == 2


# ────────────────────────────────────────────────────────────────────
# create_safe_connection — engine-level sandbox
# ────────────────────────────────────────────────────────────────────


def test_safe_connection_disables_external_access() -> None:
    """Even if execute_query is bypassed (untrusted_sql=False),
    a sandboxed connection refuses file reads at the engine level."""
    con = create_safe_connection()
    # Trying to read a file directly through DuckDB MUST fail at the
    # engine level — `enable_external_access = false` is set.
    with pytest.raises((duckdb.PermissionException, duckdb.Error)):
        con.execute("SELECT * FROM read_csv('/etc/passwd')")


def test_safe_connection_allows_in_memory_queries() -> None:
    """The sandbox doesn't break legitimate in-memory work."""
    con = create_safe_connection()
    result = con.execute("SELECT 42 AS x").fetchone()
    assert result == (42,)


def test_safe_connection_layered_defense() -> None:
    """The defences compose: structural conn check + locked engine sandbox.

    Sec-3 (security finding #2) restructured the defence:

    - **Structural**: ``execute_query(untrusted_sql=True)`` requires a
      sandboxed connection (raises if not).
    - **Engine**: ``create_safe_connection()`` issues
      ``SET enable_external_access=false`` +
      ``SET lock_configuration=true``. The lock is the lynchpin —
      the SQL itself cannot ``SET enable_external_access=true`` to
      undo the sandbox.
    - **Telemetry**: the regex deny-list logs (does NOT enforce) from
      inside execute_query.

    A sandboxed connection PLUS the default ``execute_query`` is the
    recommended posture for any code that runs SQL authored outside
    the trust boundary.
    """
    con = create_safe_connection()
    register_table(con, _make_test_table(), name="t")
    # The structural check passes (con is sandboxed). The engine then
    # refuses ``read_csv`` because external access is locked off.
    with pytest.raises((duckdb.PermissionException, duckdb.Error, duckdb.IOException)):
        execute_query(con, "SELECT * FROM read_csv('/x')")


# ────────────────────────────────────────────────────────────────────
# _register_table_fallback — column identifier quoting (audit C7)
# ────────────────────────────────────────────────────────────────────


def test_register_table_fallback_handles_embedded_quotes_in_column_name() -> None:
    """A column name containing `"` was previously a SQL injection path:
    the f-string `f'"{c.name}"'` would break out of the identifier on
    the embedded quote. The fix routes through _quote_ident which
    correctly escapes `"` -> `""`."""
    con = duckdb.connect(":memory:")
    weird = Table(
        name="weird",
        columns=(
            Column(name='evil"; DROP TABLE x; --', column_type=ColumnType.TEXT),
            Column(name="ok", column_type=ColumnType.INTEGER),
        ),
        rows=(("a", 1), ("b", 2)),
        row_count=2,
    )
    # The fallback path is exercised when polars is not available;
    # we invoke it directly to test the quoting in isolation.
    _register_table_fallback(con, weird, view_name="weird_view")
    # The table was created with a single column whose name has the
    # embedded quote — readable back via DESCRIBE.
    described = con.execute("DESCRIBE weird_view").fetchall()
    column_names = {row[0] for row in described}
    assert 'evil"; DROP TABLE x; --' in column_names
    assert "ok" in column_names


# ────────────────────────────────────────────────────────────────────
# list_tables — parameterised query (audit C6)
# ────────────────────────────────────────────────────────────────────


def test_list_tables_returns_registered_tables() -> None:
    """Smoke test the parameterised query path."""
    con = duckdb.connect(":memory:")
    register_table(con, _make_test_table(), name="alpha")
    register_table(con, _make_test_table(), name="beta")
    listed = list_tables(con)
    names = {row["name"] for row in listed}
    assert "alpha" in names
    assert "beta" in names
    # Each entry has a column count
    for entry in listed:
        if entry["name"] in {"alpha", "beta"}:
            assert entry["column_count"] == 2
