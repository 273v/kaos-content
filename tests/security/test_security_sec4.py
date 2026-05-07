"""Sec-4 regression tests: HTML link-stripping drops visible text (#4).

Pre-fix the parser's dangerous-URL handler returned ``None`` instead
of preserving the inner text. ``<p>before <a href="javascript:...">click
me</a> after</p>`` collapsed to ``before  after`` — the user-visible
"click me" was silently swallowed.

Fix: return a ``Span`` wrapping the original children. The link is
removed (no longer reachable), but every glyph the human reader saw
in the source survives in the AST.
"""

from __future__ import annotations

import pytest

from kaos_content.model.inlines import Link, Span, Text
from kaos_content.parsers.html import parse_html
from kaos_content.serializers.text import serialize_text


def _flatten_text(doc) -> str:
    """Concatenate all Text content in the document, preserving order."""
    return serialize_text(doc).strip()


@pytest.mark.parametrize(
    "scheme",
    [
        "javascript:void(0)",
        "javascript:alert(1)",
        "vbscript:msgbox(1)",
        "data:text/html,<script>alert(1)</script>",
        "file:///etc/passwd",
    ],
)
def test_dangerous_link_preserves_inner_text(scheme: str) -> None:
    # The PoC from the security report — and parametrised over each
    # unsafe scheme.
    html = f'<p>before <a href="{scheme}">click me</a> after</p>'
    doc = parse_html(html)
    text = _flatten_text(doc)
    assert "before" in text
    assert "click me" in text, (
        f"Inner link text was dropped (scheme={scheme!r}). Pre-fix this branch "
        f"returned None, swallowing the visible text. Got: {text!r}"
    )
    assert "after" in text


def test_dangerous_link_does_not_emit_link_node() -> None:
    """The link itself is removed — no Link node remains in the AST."""
    html = '<p>before <a href="javascript:void(0)">click me</a> after</p>'
    doc = parse_html(html)
    # Walk the AST and assert no Link with a javascript: URL.
    for block in doc.body:
        for inline in getattr(block, "children", ()):
            if isinstance(inline, Link):
                assert not inline.url.startswith("javascript:"), (
                    f"Dangerous Link survived in AST: {inline!r}"
                )


def test_dangerous_link_emits_span_wrapper() -> None:
    """The stripped link's children survive as a Span container."""
    html = '<p>before <a href="javascript:void(0)">click me</a> after</p>'
    doc = parse_html(html)
    # First (and only) block is a Paragraph; find a Span child.
    found_span_with_click_me = False
    for block in doc.body:
        for inline in getattr(block, "children", ()):
            if isinstance(inline, Span):
                # The span should carry the original link children — at
                # least one Text node containing "click me".
                for child in inline.children:
                    if isinstance(child, Text) and "click me" in child.value:
                        found_span_with_click_me = True
                        break
    assert found_span_with_click_me, (
        f"Expected a Span containing 'click me' from the stripped link. Doc body: {doc.body!r}"
    )


def test_safe_link_unchanged() -> None:
    """Sanity: links with safe URLs still produce a Link node."""
    html = '<p>before <a href="https://example.com">click me</a> after</p>'
    doc = parse_html(html)
    found_link = False
    for block in doc.body:
        for inline in getattr(block, "children", ()):
            if isinstance(inline, Link):
                assert inline.url == "https://example.com"
                found_link = True
    assert found_link, "Safe link was incorrectly stripped"


def test_dangerous_link_with_no_inner_text_returns_nothing() -> None:
    """Edge case: <a href="javascript:..."></a> has no text to preserve."""
    html = '<p>before <a href="javascript:void(0)"></a> after</p>'
    doc = parse_html(html)
    text = _flatten_text(doc)
    # No "click me" because the link was empty; "before" and "after"
    # still survive.
    assert "before" in text
    assert "after" in text
    # No Span dropped in.
    for block in doc.body:
        for inline in getattr(block, "children", ()):
            assert not isinstance(inline, Span), (
                f"Empty stripped link should not produce a Span: {inline!r}"
            )


def test_nested_inlines_inside_dangerous_link_preserved() -> None:
    """Strong / em / etc. inside a dangerous link survive too."""
    html = '<p>before <a href="javascript:void(0)">click <strong>me</strong> now</a> after</p>'
    doc = parse_html(html)
    text = _flatten_text(doc)
    assert "click" in text
    assert "me" in text
    assert "now" in text
