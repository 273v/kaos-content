"""Tests for ``parse_html(..., pre_content_mode=...)``.

The mode parameter lets callers declare how ``<pre>`` should be
interpreted — as source code (default, emits ``CodeBlock``) or as
paragraph-separated prose (emits ``Paragraph`` blocks). The point of
the parameter is to avoid hard-coded source-sniffing in callers: a
consumer whose source abuses ``<pre>`` as a plain-text container
(e.g. Federal Register ``raw_text_url``) sets the mode once; the
parser stays general.
"""

from __future__ import annotations

from kaos_content.model.blocks import CodeBlock, Paragraph
from kaos_content.parsers.html import parse_html


class TestPreContentModeCode:
    """Default ``pre_content_mode="code"`` emits ``CodeBlock``."""

    def test_default_is_code_mode(self) -> None:
        html = "<html><body><pre>line 1\nline 2</pre></body></html>"
        doc = parse_html(html)
        assert len(doc.body) == 1
        assert isinstance(doc.body[0], CodeBlock)
        assert "line 1" in doc.body[0].value

    def test_pre_with_code_child_still_code(self) -> None:
        html = (
            "<html><body><pre><code class='language-python'>"
            "def f(): pass"
            "</code></pre></body></html>"
        )
        doc = parse_html(html)
        assert len(doc.body) == 1
        cb = doc.body[0]
        assert isinstance(cb, CodeBlock)
        assert cb.language == "python"


class TestPreContentModeProse:
    """``pre_content_mode="prose"`` emits blank-line-separated ``Paragraph``s."""

    def test_blank_line_separators(self) -> None:
        html = (
            "<html><body><pre>"
            "First paragraph.\n\n"
            "Second paragraph.\n\n"
            "Third paragraph."
            "</pre></body></html>"
        )
        doc = parse_html(html, pre_content_mode="prose")
        paragraphs = [b for b in doc.body if isinstance(b, Paragraph)]
        assert len(paragraphs) == 3
        texts = [getattr(p.children[0], "value", "") for p in paragraphs]
        assert texts == ["First paragraph.", "Second paragraph.", "Third paragraph."]

    def test_single_block_without_blank_lines_is_one_paragraph(self) -> None:
        html = "<html><body><pre>just one paragraph here</pre></body></html>"
        doc = parse_html(html, pre_content_mode="prose")
        paragraphs = [b for b in doc.body if isinstance(b, Paragraph)]
        assert len(paragraphs) == 1
        assert getattr(paragraphs[0].children[0], "value", "") == "just one paragraph here"

    def test_runs_of_blank_lines_collapse(self) -> None:
        """Any number of consecutive blank lines collapses to one separator."""
        html = "<html><body><pre>A\n\n\n\nB\n\n\nC</pre></body></html>"
        doc = parse_html(html, pre_content_mode="prose")
        paragraphs = [b for b in doc.body if isinstance(b, Paragraph)]
        assert len(paragraphs) == 3

    def test_empty_pre_produces_no_blocks(self) -> None:
        html = "<html><body><pre></pre></body></html>"
        doc = parse_html(html, pre_content_mode="prose")
        assert not [b for b in doc.body if isinstance(b, (Paragraph, CodeBlock))]

    def test_whitespace_only_chunks_are_dropped(self) -> None:
        html = "<html><body><pre>   \n\nreal text\n\n   </pre></body></html>"
        doc = parse_html(html, pre_content_mode="prose")
        paragraphs = [b for b in doc.body if isinstance(b, Paragraph)]
        assert len(paragraphs) == 1
        assert getattr(paragraphs[0].children[0], "value", "") == "real text"


class TestPreContentModeScope:
    """The mode propagates through nested containers."""

    def test_nested_in_div(self) -> None:
        html = "<html><body><div><section><pre>One.\n\nTwo.</pre></section></div></body></html>"
        doc = parse_html(html, pre_content_mode="prose")
        # Unwrap to find Paragraphs regardless of nesting
        paras: list[Paragraph] = []

        def walk(node: object) -> None:
            if isinstance(node, Paragraph):
                paras.append(node)
                return
            for c in getattr(node, "children", ()) or ():
                walk(c)
            for c in getattr(node, "body", ()) or ():
                walk(c)

        walk(doc)
        assert len(paras) == 2

    def test_scope_does_not_leak_between_calls(self) -> None:
        """Calling with prose mode must not change the subsequent default."""
        html = "<html><body><pre>A\n\nB</pre></body></html>"
        parse_html(html, pre_content_mode="prose")
        # Second call reverts to default (code)
        doc = parse_html(html)
        assert isinstance(doc.body[0], CodeBlock)


class TestPreContentModeUnknown:
    """Unknown mode is tolerated — defaults to code behavior silently.

    (The mode is a string parameter rather than an enum for API ergonomics;
    the parser treats anything other than the known values as the
    default ``"code"`` to avoid surprising the caller with a raise from
    deep inside an HTML walker. Callers are trusted.)
    """

    def test_unknown_mode_behaves_as_code(self) -> None:
        html = "<html><body><pre>A\n\nB</pre></body></html>"
        doc = parse_html(html, pre_content_mode="totally-made-up")
        assert isinstance(doc.body[0], CodeBlock)
