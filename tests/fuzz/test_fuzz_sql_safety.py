"""Hypothesis fuzz tests for ``bridges.duckdb._assert_sql_safe``.

Properties:

- Any SQL containing a deny-listed function call (``read_csv(...)``)
  is rejected, regardless of:
    - case mixing (``ReAd_CsV``)
    - leading/trailing whitespace
    - line or block comments interleaved
    - whitespace between the function name and its open-paren
- Any SQL containing a deny-listed keyword as a token is rejected,
  regardless of comment / case tricks.
- The function NEVER raises anything but ``ValueError`` — no IndexError,
  no TypeError on adversarial input — and never loops or hangs.
- Strings containing the deny-listed *substring* (e.g. ``unloaded``,
  ``copy_log``, ``read_csv_count``) are NOT spuriously rejected.
"""

from __future__ import annotations

import contextlib

import pytest

duckdb = pytest.importorskip("duckdb")

from hypothesis import HealthCheck, example, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from kaos_content.bridges.duckdb import (  # noqa: E402
    _DANGEROUS_SQL_FUNCS,
    _DANGEROUS_SQL_KEYWORDS,
    _assert_sql_safe,
)

_dangerous_func = st.sampled_from(_DANGEROUS_SQL_FUNCS)
_dangerous_kw = st.sampled_from(_DANGEROUS_SQL_KEYWORDS)


def _random_case(s: str, draw) -> str:  # type: ignore[no-untyped-def]
    return "".join(c.upper() if draw(st.booleans()) else c.lower() for c in s)


@st.composite
def _random_case_strategy(draw, s: str):  # type: ignore[no-untyped-def]
    return _random_case(s, draw)


# ────────────────────────────────────────────────────────────────────
# Deny-listed FUNCTION calls are always rejected
# ────────────────────────────────────────────────────────────────────


@given(fn=_dangerous_func, payload=st.text(max_size=32))
@example(fn="read_csv", payload="/etc/passwd")
@example(fn="read_parquet", payload="s3://evil/x.parquet")
def test_dangerous_func_rejected(fn: str, payload: str) -> None:
    with pytest.raises(ValueError, match="not allowed"):
        _assert_sql_safe(f"SELECT * FROM {fn}('{payload}')")


@given(fn=_dangerous_func)
def test_dangerous_func_random_case_rejected(fn: str) -> None:
    """Case mixing must not bypass the filter."""
    perm = "".join(c.upper() if i % 2 == 0 else c.lower() for i, c in enumerate(fn))
    with pytest.raises(ValueError, match="not allowed"):
        _assert_sql_safe(f"SELECT * FROM {perm}('x')")


@given(fn=_dangerous_func, gap=st.sampled_from(["", " ", "  ", "\t", "\n"]))
def test_dangerous_func_with_whitespace_before_paren_rejected(fn: str, gap: str) -> None:
    """``read_csv (...)`` with whitespace must still be rejected."""
    with pytest.raises(ValueError, match="not allowed"):
        _assert_sql_safe(f"SELECT * FROM {fn}{gap}('x')")


@given(fn=_dangerous_func, comment=st.text(max_size=20))
def test_dangerous_func_after_line_comment_rejected(fn: str, comment: str) -> None:
    """Hiding the call after a line comment doesn't help — the comment
    is stripped before the deny-list runs."""
    # Sanitise the comment string so it doesn't accidentally contain a
    # newline that closes the comment early; line comments end at \n.
    safe_comment = comment.replace("\n", " ").replace("\r", " ")
    sql = f"SELECT 1 -- {safe_comment}\nUNION SELECT * FROM {fn}('x')"
    with pytest.raises(ValueError, match="not allowed"):
        _assert_sql_safe(sql)


@given(fn=_dangerous_func)
def test_dangerous_func_after_block_comment_rejected(fn: str) -> None:
    """Block comments don't help either."""
    sql = f"SELECT 1; /* {'a' * 32} */ {fn}('x')"
    with pytest.raises(ValueError, match="not allowed"):
        _assert_sql_safe(sql)


# ────────────────────────────────────────────────────────────────────
# Deny-listed KEYWORDS are always rejected
# ────────────────────────────────────────────────────────────────────


@given(kw=_dangerous_kw, target=st.text(max_size=24))
def test_dangerous_keyword_rejected(kw: str, target: str) -> None:
    sql = f"{kw} 'sneak.db' AS {target}"
    with pytest.raises(ValueError, match="not allowed"):
        _assert_sql_safe(sql)


@given(kw=_dangerous_kw)
def test_dangerous_keyword_random_case_rejected(kw: str) -> None:
    perm = "".join(c.upper() if i % 2 == 0 else c.lower() for i, c in enumerate(kw))
    with pytest.raises(ValueError, match="not allowed"):
        _assert_sql_safe(f"{perm} 'x'")


@given(kw=_dangerous_kw)
def test_dangerous_keyword_after_block_comment_rejected(kw: str) -> None:
    sql = f"SELECT 1; /* hidden */ {kw} 'x'"
    with pytest.raises(ValueError, match="not allowed"):
        _assert_sql_safe(sql)


# ────────────────────────────────────────────────────────────────────
# False-positive guards — substring NOT a token must pass
# ────────────────────────────────────────────────────────────────────


@given(fn=_dangerous_func, suffix=st.from_regex(r"[a-z_]{1,10}", fullmatch=True))
def test_substring_function_name_not_rejected(fn: str, suffix: str) -> None:
    """``read_csv_count`` (no opening paren immediately after the name)
    is a legitimate column / table identifier and must pass."""
    name = f"{fn}{suffix}"
    # Ensure ``name`` is not itself accidentally a deny-listed token.
    if name in _DANGEROUS_SQL_FUNCS:
        return
    _assert_sql_safe(f"SELECT {name} FROM stats")


@given(kw=_dangerous_kw, prefix=st.from_regex(r"[a-z_]{1,10}", fullmatch=True))
def test_substring_keyword_not_rejected(kw: str, prefix: str) -> None:
    """``unloaded`` (a longer word that *contains* ``load``) must pass —
    deny-list matches are word-bounded."""
    name = f"{prefix}{kw}"
    if name in _DANGEROUS_SQL_KEYWORDS:
        return
    _assert_sql_safe(f"SELECT {name} FROM products")


# ────────────────────────────────────────────────────────────────────
# Legitimate queries pass
# ────────────────────────────────────────────────────────────────────


@given(
    cols=st.lists(
        st.from_regex(r"[a-z_][a-z0-9_]{0,15}", fullmatch=True),
        min_size=1,
        max_size=4,
    ),
    table=st.from_regex(r"[a-z_][a-z0-9_]{0,15}", fullmatch=True),
)
@settings(suppress_health_check=[HealthCheck.filter_too_much])
def test_plain_select_passes(cols: list[str], table: str) -> None:
    """Vanilla SELECT against simple identifiers always passes."""
    # Avoid identifiers that collide with deny-listed tokens.
    bad = set(_DANGEROUS_SQL_FUNCS) | set(_DANGEROUS_SQL_KEYWORDS)
    if table in bad or any(c in bad for c in cols):
        return
    sql = f"SELECT {', '.join(cols)} FROM {table}"
    _assert_sql_safe(sql)


# ────────────────────────────────────────────────────────────────────
# Robustness — never raise anything but ValueError, never hang
# ────────────────────────────────────────────────────────────────────


@given(s=st.text(max_size=512))
def test_assert_sql_safe_only_raises_value_error(s: str) -> None:
    """Adversarial input either passes or raises ValueError. No
    IndexError, TypeError, or unexpected exception class."""
    # Deny-listed patterns raise; everything else passes silently.
    with contextlib.suppress(ValueError):
        _assert_sql_safe(s)
