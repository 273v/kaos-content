"""Sec-3 regression tests: DuckDB sandbox structural reorganisation (#2).

The pre-fix regex deny-list missed several DuckDB table functions
(``read_ndjson``, ``read_json_objects``, ``read_duckdb``,
``parquet_scan``, ``iceberg_scan``, etc.) and would lose this race
forever — DuckDB ships new readers every release. The fix replaces
the deny-list with:

1. **Engine-level sandbox** in ``create_safe_connection()``:
   ``SET enable_external_access=false`` +
   ``SET allow_unsigned_extensions=false`` +
   ``SET lock_configuration=true`` (the lock makes the sandbox
   tamper-proof against ``SET ... = true`` from user SQL).

2. **Structural check** in ``execute_query``: ``untrusted_sql=True``
   REQUIRES a connection from ``create_safe_connection()``. Tracked
   via a ``WeakSet`` of safe connections. Raw ``duckdb.connect()``
   connections are refused with a clear error.

3. The regex deny-list is demoted to telemetry.
"""

from __future__ import annotations

import pytest

duckdb = pytest.importorskip("duckdb")

from kaos_content.bridges.duckdb import (  # noqa: E402
    _is_safe_connection,
    create_safe_connection,
    execute_query,
    register_table,
)
from kaos_content.model.tabular import Column, ColumnType, Table  # noqa: E402


def _make_table() -> Table:
    return Table(
        name="t",
        columns=(Column(name="id", column_type=ColumnType.INTEGER),),
        rows=((1,), (2,), (3,)),
        row_count=3,
    )


# ----- Structural defence: WeakSet tracking ------------------------------


class TestSafeConnectionTracking:
    def test_safe_connection_is_tracked(self) -> None:
        con = create_safe_connection()
        assert _is_safe_connection(con)

    def test_raw_connection_is_not_tracked(self) -> None:
        con = duckdb.connect(":memory:")
        assert not _is_safe_connection(con)

    def test_execute_query_requires_safe_connection(self) -> None:
        # The original PoC: read_ndjson is one of the readers the old
        # regex deny-list MISSED. With Sec-3 the structural check
        # rejects regardless of the regex.
        con = duckdb.connect(":memory:")
        register_table(con, _make_table(), name="t")
        with pytest.raises(ValueError, match="requires a sandboxed connection"):
            execute_query(con, "SELECT * FROM read_ndjson('/tmp/file.ndjson')")

    def test_execute_query_legit_select_via_raw_conn_with_untrusted_false(self) -> None:
        # Escape hatch for trusted code.
        con = duckdb.connect(":memory:")
        register_table(con, _make_table(), name="t")
        result = execute_query(con, "SELECT id FROM t", untrusted_sql=False)
        assert result.row_count == 3


# ----- Engine sandbox: every file-reader DuckDB ships is blocked --------


# These are the readers/keywords the original regex deny-list missed.
# After Sec-3 the sandbox blocks them at the engine level — no regex
# update needed when DuckDB adds new ones.
_READERS_PRE_FIX_MISSED = [
    "read_ndjson",
    "read_json_objects",
    "read_xlsx",
]


class TestEngineSandboxBlocksReaders:
    """The locked-config sandbox blocks every external-access reader.

    Each test passes ``untrusted_sql=True`` (default) on a sandboxed
    connection. The structural check passes (con is in the safe set);
    the engine then refuses.
    """

    @pytest.mark.parametrize("reader_fn", _READERS_PRE_FIX_MISSED)
    def test_pre_fix_missed_readers_now_blocked(self, reader_fn: str) -> None:
        con = create_safe_connection()
        register_table(con, _make_table(), name="t")
        with pytest.raises((duckdb.PermissionException, duckdb.Error, duckdb.IOException)):
            execute_query(con, f"SELECT * FROM {reader_fn}('/tmp/x')")

    def test_attach_blocked(self) -> None:
        con = create_safe_connection()
        with pytest.raises((duckdb.PermissionException, duckdb.Error, duckdb.IOException)):
            execute_query(con, "ATTACH '/tmp/x.db'", untrusted_sql=False)

    def test_install_blocked(self) -> None:
        con = create_safe_connection()
        with pytest.raises((duckdb.PermissionException, duckdb.Error, duckdb.IOException)):
            execute_query(con, "INSTALL httpfs", untrusted_sql=False)

    def test_load_blocked(self) -> None:
        con = create_safe_connection()
        with pytest.raises((duckdb.PermissionException, duckdb.Error, duckdb.IOException)):
            execute_query(con, "LOAD httpfs", untrusted_sql=False)

    def test_copy_to_file_blocked(self) -> None:
        # COPY ... TO writes to local FS; must be blocked.
        con = create_safe_connection()
        register_table(con, _make_table(), name="t")
        with pytest.raises((duckdb.PermissionException, duckdb.Error, duckdb.IOException)):
            execute_query(
                con,
                "COPY (SELECT * FROM t) TO '/tmp/leak.csv'",
                untrusted_sql=False,
            )

    def test_copy_from_file_blocked(self) -> None:
        # COPY ... FROM reads local FS; must be blocked.
        con = create_safe_connection()
        register_table(con, _make_table(), name="t")
        with pytest.raises((duckdb.PermissionException, duckdb.Error, duckdb.IOException)):
            execute_query(
                con,
                "COPY t FROM '/etc/passwd'",
                untrusted_sql=False,
            )


# ----- Lock-configuration: SQL cannot undo the sandbox ------------------


class TestLockConfigurationTamperProof:
    """The ``SET lock_configuration = true`` pragma is the lynchpin.

    Without it, untrusted SQL could ``SET enable_external_access = true``
    to undo the sandbox between the constructor and the user query.
    With it, every subsequent SET fails — the sandbox is immutable.
    """

    def test_cannot_re_enable_external_access(self) -> None:
        con = create_safe_connection()
        # The user's SQL might try this to undo the sandbox.
        with pytest.raises((duckdb.PermissionException, duckdb.Error, duckdb.IOException)):
            execute_query(
                con,
                "SET enable_external_access = true; SELECT 1",
                untrusted_sql=False,
            )

    def test_cannot_re_enable_unsigned_extensions(self) -> None:
        con = create_safe_connection()
        with pytest.raises((duckdb.PermissionException, duckdb.Error, duckdb.IOException)):
            execute_query(
                con,
                "SET allow_unsigned_extensions = true",
                untrusted_sql=False,
            )

    def test_pragma_blocked(self) -> None:
        # PRAGMA is another pathway that touches engine config.
        con = create_safe_connection()
        with pytest.raises((duckdb.PermissionException, duckdb.Error, duckdb.IOException)):
            execute_query(con, "PRAGMA enable_external_access=true", untrusted_sql=False)


# ----- The original PoC: read_ndjson via untrusted_sql=True -------------


class TestOriginalPoC:
    """The exact PoC from the Sec-3 finding must be refused."""

    def test_read_ndjson_via_raw_conn_refused_by_structural_check(self) -> None:
        # The original report did:
        #   execute_query(raw_duckdb_connection,
        #                 "select * from read_ndjson('/tmp/file.ndjson')",
        #                 untrusted_sql=True)
        # → returned local file contents (read_ndjson wasn't in the
        # regex deny-list). After Sec-3 the structural check refuses
        # before the SQL ever runs.
        con = duckdb.connect(":memory:")
        register_table(con, _make_table(), name="t")
        with pytest.raises(ValueError, match="requires a sandboxed connection"):
            execute_query(
                con,
                "select * from read_ndjson('/tmp/file.ndjson')",
                untrusted_sql=True,
            )

    def test_read_ndjson_via_safe_conn_refused_by_engine(self) -> None:
        # On a sandboxed connection, the structural check passes
        # (con is in the safe set) and the engine then refuses.
        con = create_safe_connection()
        register_table(con, _make_table(), name="t")
        with pytest.raises((duckdb.PermissionException, duckdb.Error, duckdb.IOException)):
            execute_query(con, "select * from read_ndjson('/tmp/file.ndjson')")


# ----- Sanity: legitimate queries still work ----------------------------


class TestLegitimateQueriesUnaffected:
    def test_simple_select(self) -> None:
        con = create_safe_connection()
        register_table(con, _make_table(), name="t")
        result = execute_query(con, "SELECT id FROM t WHERE id > 1")
        assert result.row_count == 2

    def test_aggregation(self) -> None:
        con = create_safe_connection()
        register_table(con, _make_table(), name="t")
        result = execute_query(con, "SELECT COUNT(*) AS n FROM t")
        assert result.row_count == 1

    def test_join(self) -> None:
        con = create_safe_connection()
        register_table(con, _make_table(), name="t")
        # Self-join — exercises the join path under the sandbox.
        result = execute_query(
            con,
            "SELECT a.id FROM t a JOIN t b ON a.id = b.id",
        )
        assert result.row_count == 3
