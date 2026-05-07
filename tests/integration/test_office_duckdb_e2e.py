"""Cross-module end-to-end tests for Office -> Content -> DuckDB flows."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

# ``from kaos_office.xlsx.reader import parse_xlsx`` transitively imports
# ``kaos_office.pptx.writer`` via ``kaos_office/__init__.py``, which requires
# ``python-pptx``. That optional dep isn't declared on kaos-content's env, so
# we skip cleanly rather than fail at collection when it's absent.
pytest.importorskip("pptx")

from kaos_office.xlsx.reader import parse_xlsx

from kaos_content.bridges.duckdb import query_to_table, register_document

FIXTURES = Path(__file__).resolve().parents[3] / "kaos-office" / "tests" / "fixtures" / "xlsx"
CBS = FIXTURES / "CBS-BNSF-2015-Q4.xlsx"


@pytest.mark.integration
def test_xlsx_document_registers_in_duckdb_end_to_end() -> None:
    """Parse a real XLSX fixture and query it through the DuckDB bridge."""
    doc = parse_xlsx(CBS)
    con = duckdb.connect()

    try:
        names = register_document(con, doc)
        assert len(names) == 2

        result = query_to_table(con, 'SELECT COUNT(*) FROM "4Q15 CBS"', name="count")
        assert result.rows[0][0] > 0
    finally:
        con.close()
