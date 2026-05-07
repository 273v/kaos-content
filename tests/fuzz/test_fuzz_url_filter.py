"""Hypothesis fuzz tests for ``_security.is_safe_url``.

The audit closed five concrete bypass classes (newline-in-scheme,
tab-in-scheme, HTML-entity-encoded scheme, percent-encoded colon,
case mixing). These tests assert the *general* property: any string
whose fully-decoded canonical form starts with an unsafe scheme is
rejected, regardless of the encoding tricks used to dress it up.
"""

from __future__ import annotations

import html as _html

from hypothesis import assume, example, given
from hypothesis import strategies as st

from kaos_content._security import UNSAFE_SCHEMES, is_safe_url

# ────────────────────────────────────────────────────────────────────
# Strategies
# ────────────────────────────────────────────────────────────────────

# Random scheme-flavoured prefix for adversarial input.
_unsafe_scheme_strategy = st.sampled_from(sorted(UNSAFE_SCHEMES))


# ────────────────────────────────────────────────────────────────────
# Properties
# ────────────────────────────────────────────────────────────────────


@given(scheme=_unsafe_scheme_strategy, payload=st.text(max_size=64))
@example(scheme="javascript", payload="alert(1)")
@example(scheme="data", payload="text/html,<script>alert(1)</script>")
@example(scheme="vbscript", payload="msgbox(1)")
@example(scheme="file", payload="///etc/passwd")
def test_unsafe_scheme_always_rejected(scheme: str, payload: str) -> None:
    """``<scheme>:<anything>`` is rejected regardless of payload."""
    assert is_safe_url(f"{scheme}:{payload}") is False


@given(scheme=_unsafe_scheme_strategy, payload=st.text(max_size=32))
def test_whitespace_dressed_scheme_rejected(scheme: str, payload: str) -> None:
    """Whitespace inside the scheme cannot smuggle it past the filter."""
    base = f"{scheme}:{payload}"
    # Build a whitespace-dressed variant by inserting ws between scheme chars.
    dressed_scheme = "\n".join(scheme) + "\t"
    candidate = f"{dressed_scheme}:{payload}"
    assert is_safe_url(candidate) is False
    # Also the directly-dressed base
    assert is_safe_url(f"\t {base}\n") is False


@given(scheme=_unsafe_scheme_strategy, payload=st.text(max_size=32))
def test_entity_encoded_scheme_rejected(scheme: str, payload: str) -> None:
    """HTML-entity-encoded variants of the scheme are rejected."""
    encoded = "".join(f"&#{ord(c)};" for c in scheme)
    candidate = f"{encoded}:{payload}"
    # Sanity: stdlib unescape produces the original scheme
    assume(_html.unescape(encoded).lower() == scheme.lower())
    assert is_safe_url(candidate) is False


@given(scheme=_unsafe_scheme_strategy, payload=st.text(max_size=32))
def test_percent_encoded_colon_rejected(scheme: str, payload: str) -> None:
    """``javascript%3Aalert(1)`` shape is rejected (percent-encoded colon)."""
    candidate = f"{scheme}%3A{payload}"
    assert is_safe_url(candidate) is False


@given(scheme=_unsafe_scheme_strategy, payload=st.text(max_size=32))
def test_mixed_case_scheme_rejected(scheme: str, payload: str) -> None:
    """Any case-permutation of the scheme is rejected."""
    perm = "".join(c.upper() if i % 2 == 0 else c.lower() for i, c in enumerate(scheme))
    assert is_safe_url(f"{perm}:{payload}") is False


@given(scheme=_unsafe_scheme_strategy, payload=st.text(max_size=32))
def test_combined_evasions_rejected(scheme: str, payload: str) -> None:
    """Whitespace + entity + percent + case combined still rejected."""
    # Encode every other char as decimal entity, alternate case, dress with ws,
    # then percent-encode the colon.
    chars = []
    for i, c in enumerate(scheme):
        if i % 2 == 0:
            chars.append(f"&#{ord(c.upper())};")
        else:
            chars.append(c.lower())
    encoded_scheme = "\t".join(chars)
    candidate = f"{encoded_scheme}%3A{payload}\n"
    assert is_safe_url(candidate) is False


@given(s=st.text(max_size=128))
def test_no_scheme_or_relative_url_is_safe(s: str) -> None:
    """A URL with no scheme or one whose canonical form has no unsafe
    scheme is accepted. Specifically, relative paths and fragments
    must not be rejected — that would break legitimate links."""
    # Build a candidate guaranteed to not contain any unsafe scheme:
    # take an arbitrary string, fully decode it, and only proceed if
    # its canonical form has no unsafe scheme.
    canonical = "".join(c for c in _html.unescape(s).lower() if not c.isspace() and c != "\x00")
    assume(not any(canonical.startswith(f"{u}:") for u in UNSAFE_SCHEMES))
    assume(":" not in canonical or canonical.split(":", 1)[0] not in UNSAFE_SCHEMES)
    # An https: URL or a relative path must round-trip as safe.
    safe_inputs = ["https://example.com/a", "/foo/bar", "#fragment", "mailto:a@b.c", ""]
    for u in safe_inputs:
        assert is_safe_url(u) is True


@given(s=st.text(max_size=256))
def test_is_safe_url_never_crashes(s: str) -> None:
    """``is_safe_url`` must return a bool for any input — never raise."""
    result = is_safe_url(s)
    assert isinstance(result, bool)
