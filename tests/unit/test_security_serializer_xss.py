"""XSS regression tests for the HTML and Markdown serializers.

Pins the safe-by-default contract introduced in 0.1.0a1:

- Raw HTML blocks (``RawBlock``/``RawInline`` with ``format="html"``)
  are dropped from output unless the caller passes
  ``allow_raw_html=True``.
- ``<a href>`` and ``<img src>`` (HTML) and ``[](url)`` / ``![](url)``
  (Markdown) URLs whose canonical scheme is in
  :data:`kaos_content._security.UNSAFE_SCHEMES` are replaced with ``#``
  in safe mode. The HTML serializer additionally captures the
  original (HTML-escaped) URL in ``data-unsafe-url`` for forensics.

Audit-cited bypasses (kaos-content audit, May 2026):

- ``<script>alert(1)</script>`` smuggled through ``RawBlock(format="html")``
- ``<a href="javascript:alert(1)">`` constructed by HTML serializer
- ``[](javascript:alert(1))`` constructed by Markdown serializer
- All the URL-canonicalisation bypasses tested in
  ``test_security_url_filter.py`` (\\n, \\t, &#x6A;, %3A, etc.)
"""

from __future__ import annotations

import pytest

from kaos_content.model.blocks import Paragraph, RawBlock
from kaos_content.model.document import ContentDocument
from kaos_content.model.inlines import Image, Link, Text
from kaos_content.serializers.html import serialize_html
from kaos_content.serializers.markdown import serialize_markdown

# ────────────────────────────────────────────────────────────────────
# Raw HTML stripping (HTML serializer)
# ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "payload",
    [
        "<script>alert(1)</script>",
        "<img src=x onerror=alert(1)>",
        '<svg onload="alert(1)">',
        "<iframe src=javascript:alert(1)></iframe>",
        '<a href="javascript:alert(1)">click</a>',
        '<style>body{background:url("javascript:alert(1)")}</style>',
    ],
)
def test_html_serializer_strips_raw_block_by_default(payload: str) -> None:
    doc = ContentDocument(body=(RawBlock(format="html", value=payload),))
    out = serialize_html(doc)
    assert payload not in out
    assert "raw HTML stripped" in out


def test_html_serializer_passthrough_when_explicitly_allowed() -> None:
    """allow_raw_html=True is the explicit opt-in for trusted ASTs."""
    payload = "<custom-element>x</custom-element>"
    doc = ContentDocument(body=(RawBlock(format="html", value=payload),))
    out = serialize_html(doc, allow_raw_html=True)
    assert payload in out


# ────────────────────────────────────────────────────────────────────
# javascript: / data: / vbscript: / file: in <a href> and <img src>
# (HTML serializer)
# ────────────────────────────────────────────────────────────────────


def _doc_with_link(url: str) -> ContentDocument:
    return ContentDocument(
        body=(Paragraph(children=(Link(url=url, children=(Text(value="click"),)),)),)
    )


def _doc_with_image(src: str) -> ContentDocument:
    return ContentDocument(body=(Paragraph(children=(Image(src=src, alt="x"),)),))


@pytest.mark.parametrize(
    "url",
    [
        "javascript:alert(1)",
        "JaVaScRiPt:alert(1)",
        "jav\nascript:alert(1)",  # newline bypass
        "&#x6A;avascript:alert(1)",  # entity bypass
        "javascript%3Aalert(1)",  # percent bypass
        "vbscript:msgbox",
        "data:text/html,<script>alert(1)</script>",
        "file:///etc/passwd",
    ],
)
def test_html_link_with_unsafe_scheme_is_neutered(url: str) -> None:
    """Link href is replaced with '#' and original captured in data-unsafe-url."""
    out = serialize_html(_doc_with_link(url))
    # The href must NOT contain the original URL
    assert f'href="{url}"' not in out
    # The href is replaced with #
    assert 'href="#"' in out
    # Forensic attribute is present
    assert "data-unsafe-url=" in out


@pytest.mark.parametrize(
    "src",
    [
        "javascript:alert(1)",
        "data:text/html,<script>alert(1)</script>",
        "jav\tascript:alert(1)",
    ],
)
def test_html_image_with_unsafe_scheme_is_neutered(src: str) -> None:
    out = serialize_html(_doc_with_image(src))
    assert f'src="{src}"' not in out
    assert 'src="#"' in out
    assert "data-unsafe-url=" in out


def test_html_link_with_safe_scheme_is_emitted_unchanged() -> None:
    out = serialize_html(_doc_with_link("https://example.com/foo"))
    assert 'href="https://example.com/foo"' in out
    assert "data-unsafe-url=" not in out


def test_html_link_with_unsafe_scheme_passes_through_when_explicitly_allowed() -> None:
    """allow_raw_html=True disables the URL filter too — caller takes responsibility."""
    out = serialize_html(_doc_with_link("javascript:alert(1)"), allow_raw_html=True)
    assert 'href="javascript:alert(1)"' in out


# ────────────────────────────────────────────────────────────────────
# Raw block / unsafe URL stripping (Markdown serializer)
# ────────────────────────────────────────────────────────────────────


def test_markdown_serializer_strips_raw_block_html_by_default() -> None:
    payload = "<script>alert(1)</script>"
    doc = ContentDocument(body=(RawBlock(format="html", value=payload),))
    out = serialize_markdown(doc)
    assert payload not in out
    assert "raw html stripped" in out


def test_markdown_serializer_strips_raw_block_markdown_by_default() -> None:
    """`format="markdown"` is also stripped — RawBlock-markdown can carry
    inline HTML or javascript: links the same way RawBlock-html can."""
    payload = "[click](javascript:alert(1))"
    doc = ContentDocument(body=(RawBlock(format="markdown", value=payload),))
    out = serialize_markdown(doc)
    assert payload not in out
    assert "raw markdown stripped" in out


@pytest.mark.parametrize(
    "url",
    [
        "javascript:alert(1)",
        "jav\nascript:alert(1)",
        "&#x6A;avascript:alert(1)",
        "javascript%3Aalert(1)",
        "data:text/html,<script>alert(1)</script>",
        "file:///etc/passwd",
    ],
)
def test_markdown_link_with_unsafe_scheme_is_neutered(url: str) -> None:
    """Markdown rendered to HTML by a downstream consumer can execute
    the same XSS payloads as direct HTML — we strip unsafe URLs at
    markdown-serialization time."""
    out = serialize_markdown(_doc_with_link(url))
    assert url not in out
    assert "[click](#)" in out


def test_markdown_image_with_unsafe_scheme_is_neutered() -> None:
    out = serialize_markdown(_doc_with_image("javascript:alert(1)"))
    assert "javascript:alert(1)" not in out
    assert "![x](#)" in out


def test_markdown_link_with_safe_scheme_is_emitted_unchanged() -> None:
    out = serialize_markdown(_doc_with_link("https://example.com/foo"))
    assert "(https://example.com/foo)" in out


def test_markdown_link_passes_through_when_explicitly_allowed() -> None:
    # When ``allow_raw_html=True`` the unsafe scheme survives. Sec-2
    # (security finding #1) added backslash-escaping of ``(`` ``)``
    # ``<`` ``>`` in inline link destinations; the resulting markdown
    # has CommonMark-escaped parens but the AST-level URL is
    # unchanged. The right assertion is "the scheme + meaning survive",
    # not "the literal text bytes survive".
    out = serialize_markdown(_doc_with_link("javascript:alert(1)"), allow_raw_html=True)
    assert "javascript:alert" in out
    assert "\\(1\\)" in out


# ────────────────────────────────────────────────────────────────────
# End-to-end: round-trip through markdown -> HTML preserves stripping
# ────────────────────────────────────────────────────────────────────


def test_markdown_then_html_double_strip() -> None:
    """A document with both raw_block and unsafe-link must be safe
    after markdown serialization AND after HTML serialization.

    Note: the HTML serializer captures the original (HTML-escaped)
    unsafe URL in a ``data-unsafe-url`` attribute for forensics, so
    the substring ``javascript:`` will appear there. The XSS contract
    is that the URL never reaches an *executable* attribute slot
    (``href`` / ``src``), not that the substring is purged everywhere.
    """
    doc = ContentDocument(
        body=(
            RawBlock(format="html", value="<script>alert('a')</script>"),
            Paragraph(children=(Link(url="javascript:alert('b')", children=(Text(value="c"),)),)),
        )
    )
    md = serialize_markdown(doc)
    html = serialize_html(doc)
    # The script payload must not appear anywhere in either output.
    for output in (md, html):
        assert "alert('a')" not in output
        assert "<script>" not in output
    # Markdown: the unsafe URL is replaced with '#'; nothing executable.
    assert "alert('b')" not in md
    assert "javascript:" not in md
    # HTML: href is '#' (executable slot is safe), but data-unsafe-url
    # captures the original for forensics — that's the documented contract.
    assert 'href="javascript:' not in html
    assert 'href="#"' in html
    assert 'data-unsafe-url="javascript:alert(&#x27;b&#x27;)"' in html
