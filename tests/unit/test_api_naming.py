"""Top-level ``parse_*`` API naming for kaos-content.

The verb-vocabulary table in docs/python/naming.md (and PA3 in the
cross-module guide docs/guides/python-api-naming.md) pins ``parse_<format>``
as the canonical "external bytes/string → ContentDocument" entry point.

``parse_plain_text`` is always available. ``parse_markdown`` and
``parse_html`` are guarded by optional extras and are re-exported only
when the supporting libraries are importable.
"""

from __future__ import annotations

import pytest


def test_parse_plain_text_importable_from_top_level() -> None:
    from kaos_content import parse_plain_text

    assert callable(parse_plain_text)


def test_parse_plain_text_in_all() -> None:
    import kaos_content

    assert "parse_plain_text" in kaos_content.__all__


def test_parse_markdown_importable_when_extra_installed() -> None:
    pytest.importorskip("markdown_it")
    from kaos_content import parse_markdown

    assert callable(parse_markdown)


def test_parse_markdown_in_all_when_available() -> None:
    pytest.importorskip("markdown_it")
    import kaos_content

    assert "parse_markdown" in kaos_content.__all__


def test_parse_html_importable_when_extra_installed() -> None:
    """PA3: parse_html is re-exported at the top level when [html] is installed."""
    pytest.importorskip("lxml")
    from kaos_content import parse_html

    assert callable(parse_html)


def test_parse_html_in_all_when_available() -> None:
    pytest.importorskip("lxml")
    import kaos_content

    assert "parse_html" in kaos_content.__all__


def test_parse_html_returns_content_document() -> None:
    pytest.importorskip("lxml")
    from kaos_content import ContentDocument, parse_html

    doc = parse_html("<html><body><p>Hello world</p></body></html>")
    assert isinstance(doc, ContentDocument)
