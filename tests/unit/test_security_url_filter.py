"""URL-scheme filter regression tests for the HTML parser.

The previous ``_is_safe_url`` implementation used
``url.strip().lower().startswith(...)`` which let through every variant
where the scheme was mutated through embedded whitespace, HTML entities,
or URL percent-encoding. This file pins the new contract:

- HTML entities and URL percent-encoding are decoded before scheme
  matching.
- ALL whitespace (including embedded \\t, \\n, \\r, \\f, \\v and Unicode
  whitespace) plus the NUL byte are removed.
- Lowercase + ``urllib.parse.urlparse`` is the primary check, with a
  defence-in-depth ``startswith`` on the same canonical form.
- The unsafe-scheme set is ``{javascript, data, vbscript, file}``;
  ``file://`` was added in 0.1.0a1 to block local-file disclosure.

External reporters: please add new bypass attempts as parametrised
cases here (regression-style) when filing security advisories.
"""

from __future__ import annotations

import pytest

from kaos_content.parsers.html import _is_safe_url


@pytest.mark.parametrize(
    "url",
    [
        # Plain unsafe schemes
        "javascript:alert(1)",
        "JaVaScRiPt:alert(1)",
        "  javascript:alert(1)",
        "javascript :alert(1)",
        # Embedded-whitespace bypasses (the audit's headline class)
        "jav\nascript:alert(1)",
        "jav\tascript:alert(1)",
        "jav\rascript:alert(1)",
        "jav\fascript:alert(1)",
        "jav\vascript:alert(1)",
        # NUL byte injection
        "jav\x00ascript:alert(1)",
        # HTML entity bypasses
        "&#x6A;avascript:alert(1)",  # hex
        "&#106;avascript:alert(1)",  # decimal
        "&#x6A;avascript&#x3A;alert(1)",  # entity-encoded colon too
        # URL percent-encoding bypasses
        "javascript%3Aalert(1)",  # %3A == ':'
        "%6Aavascript:alert(1)",  # %6A == 'j'
        # Other unsafe schemes
        "vbscript:msgbox",
        "data:text/html,<script>alert(1)</script>",
        "file:///etc/passwd",
        "FILE:///etc/passwd",
        # Combinations
        "  jav\nasc%72ipt:alert(1)",  # whitespace + percent-encoded 'r'
    ],
)
def test_unsafe_url_is_rejected(url: str) -> None:
    """Every variant of an unsafe-scheme URL must be rejected."""
    assert _is_safe_url(url) is False, (
        f"Expected _is_safe_url({url!r}) == False; was True. "
        "An unsafe URL slipped through the filter."
    )


@pytest.mark.parametrize(
    "url",
    [
        # Standard safe URLs
        "https://example.com/foo",
        "http://example.com/foo",
        "https://example.com/foo?bar=baz#qux",
        # Relative paths
        "/about",
        "/page/foo/bar",
        "../other",
        "./same-dir",
        # Fragment + query only
        "#anchor",
        "?query=string",
        # Mailto and tel are allowed
        "mailto:foo@example.com",
        "tel:+1-555-0100",
        # Empty / whitespace-only edge cases
        "",
        "   ",
        "\t\n",
    ],
)
def test_safe_url_is_accepted(url: str) -> None:
    """Legitimate URLs must NOT be falsely rejected."""
    assert _is_safe_url(url) is True, (
        f"Expected _is_safe_url({url!r}) == True; was False. "
        "A legitimate URL was incorrectly rejected."
    )


def test_filter_handles_idempotent_double_encoding() -> None:
    """Double percent-encoding doesn't allow scheme smuggling.

    ``unquote`` only decodes one layer; a maliciously double-encoded
    ``%2525javascript:`` survives as ``%25javascript:`` after one
    unquote pass — not a valid scheme, so safe.
    """
    assert _is_safe_url("%2525javascript:alert(1)") is True


def test_filter_does_not_mutate_input() -> None:
    """The filter is a pure predicate; it must not modify its argument."""
    url = "https://example.com/path?q=javascript:alert(1)"
    _ = _is_safe_url(url)
    assert url == "https://example.com/path?q=javascript:alert(1)"
