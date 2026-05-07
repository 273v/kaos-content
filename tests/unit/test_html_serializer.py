"""Tests for the HTML serializer (Phase 5).

Tests cover:
- All block types → correct HTML tags
- All inline types → correct HTML tags
- Provenance as data-* attributes
- Redaction → [REDACTED]
- Table cell spans (rowspan/colspan)
- Footnotes as <section class="footnotes">
- include_provenance=False option
"""

from __future__ import annotations

from kaos_content import (
    Annotation,
    AnnotationTarget,
    AnnotationType,
    ContentDocument,
    DocumentBuilder,
    Provenance,
    SourceRef,
    serialize_html,
)

# ── Helpers ──


def _html(md_builder: DocumentBuilder) -> str:
    """Build and serialize to HTML."""
    return serialize_html(md_builder.build())


# ── Block types ──


class TestHtmlBlocks:
    def test_paragraph(self):
        result = _html(DocumentBuilder().paragraph("Hello"))
        assert "<p>" in result
        assert "Hello" in result
        assert "</p>" in result

    def test_heading(self):
        for depth in range(1, 7):
            result = _html(DocumentBuilder().heading(depth, "Title"))
            assert f"<h{depth}>" in result
            assert f"</h{depth}>" in result

    def test_blockquote(self):
        b = DocumentBuilder()
        b.begin_blockquote()
        b.paragraph("Quoted")
        b.end()
        result = _html(b)
        assert "<blockquote>" in result
        assert "Quoted" in result

    def test_bullet_list(self):
        b = DocumentBuilder()
        b.begin_list(ordered=False)
        b.begin_list_item()
        b.paragraph("Item 1")
        b.end()
        b.end()
        result = _html(b)
        assert "<ul>" in result
        assert "<li>" in result
        assert "Item 1" in result

    def test_ordered_list(self):
        b = DocumentBuilder()
        b.begin_list(ordered=True, start=3)
        b.begin_list_item()
        b.paragraph("Item")
        b.end()
        b.end()
        result = _html(b)
        assert '<ol start="3">' in result
        assert "<li>" in result

    def test_task_list(self):
        b = DocumentBuilder()
        b.begin_list(ordered=False)
        b.begin_list_item(checked=True)
        b.paragraph("Done")
        b.end()
        b.begin_list_item(checked=False)
        b.paragraph("Todo")
        b.end()
        b.end()
        result = _html(b)
        assert "task-list-item" in result
        assert "checked" in result

    def test_code_block(self):
        result = _html(DocumentBuilder().code_block("x = 1", "python"))
        assert "<pre>" in result
        assert "<code" in result
        assert "language-python" in result
        assert "x = 1" in result

    def test_code_block_no_language(self):
        result = _html(DocumentBuilder().code_block("code"))
        assert "<pre>" in result
        assert "language-" not in result

    def test_thematic_break(self):
        result = _html(DocumentBuilder().thematic_break())
        assert "<hr" in result

    def test_math_block(self):
        result = _html(DocumentBuilder().math_block("E = mc^2"))
        assert "math-display" in result
        assert "E = mc^2" in result

    def test_table(self):
        result = _html(
            DocumentBuilder().table(
                ["A", "B"],
                [["1", "2"], ["3", "4"]],
            )
        )
        assert "<table>" in result
        assert "<thead>" in result
        assert "<tbody>" in result
        assert "<th>" in result
        assert "<td>" in result

    def test_admonition(self):
        from kaos_content import Paragraph, Text

        b = DocumentBuilder()
        b.admonition("warning", Paragraph(children=(Text(value="Be careful!"),)))
        result = _html(b)
        assert "admonition-warning" in result

    def test_raw_block_html_stripped_by_default(self):
        # Safe-by-default: raw HTML blocks are dropped to prevent XSS.
        # Callers must opt in with allow_raw_html=True to pass through.
        b = DocumentBuilder()
        b.raw_block("<div>raw</div>", "html")
        result = _html(b)
        assert "<div>raw</div>" not in result
        assert "raw HTML stripped" in result

    def test_raw_block_html_passthrough_when_explicitly_allowed(self):
        b = DocumentBuilder()
        b.raw_block("<div>raw</div>", "html")
        # Caller asserts the AST is trusted.
        result = serialize_html(b.build(), allow_raw_html=True)
        assert "<div>raw</div>" in result

    def test_div(self):
        b = DocumentBuilder()
        b.begin_div()
        b.paragraph("Inside div")
        b.end()
        result = _html(b)
        assert "<div>" in result


# ── Inline types ──


class TestHtmlInlines:
    def test_emphasis(self):
        result = _html(DocumentBuilder().paragraph(DocumentBuilder.italic("italic")))
        assert "<em>italic</em>" in result

    def test_strong(self):
        result = _html(DocumentBuilder().paragraph(DocumentBuilder.bold("bold")))
        assert "<strong>bold</strong>" in result

    def test_strikethrough(self):
        from kaos_content import Strikethrough, Text

        doc = DocumentBuilder()
        doc.paragraph(Strikethrough(children=(Text(value="struck"),)))
        result = _html(doc)
        assert "<del>struck</del>" in result

    def test_code(self):
        result = _html(DocumentBuilder().paragraph(DocumentBuilder.code("x")))
        assert "<code>x</code>" in result

    def test_link(self):
        result = _html(
            DocumentBuilder().paragraph(DocumentBuilder.link("text", "http://example.com", "Title"))
        )
        assert '<a href="http://example.com"' in result
        assert 'title="Title"' in result
        assert "text</a>" in result

    def test_image(self):
        b = DocumentBuilder()
        b.image("image.png", "alt text", "Title")
        result = _html(b)
        assert '<img src="image.png"' in result
        assert 'alt="alt text"' in result

    def test_footnote_ref(self):
        result = _html(DocumentBuilder().paragraph(DocumentBuilder.footnote_ref("fn1")))
        assert "#fn-fn1" in result
        assert "<sup>" in result

    def test_math_inline(self):
        result = _html(DocumentBuilder().paragraph(DocumentBuilder.math("x=y")))
        assert "math-inline" in result
        assert "$x=y$" in result

    def test_superscript(self):
        from kaos_content import Superscript, Text

        doc = DocumentBuilder()
        doc.paragraph(Superscript(children=(Text(value="sup"),)))
        result = _html(doc)
        assert "<sup>sup</sup>" in result

    def test_subscript(self):
        from kaos_content import Subscript, Text

        doc = DocumentBuilder()
        doc.paragraph(Subscript(children=(Text(value="sub"),)))
        result = _html(doc)
        assert "<sub>sub</sub>" in result

    def test_underline(self):
        from kaos_content import Text, Underline

        doc = DocumentBuilder()
        doc.paragraph(Underline(children=(Text(value="under"),)))
        result = _html(doc)
        assert "<u>under</u>" in result

    def test_line_break(self):
        from kaos_content import LineBreak

        doc = DocumentBuilder()
        doc.paragraph(DocumentBuilder.text("a"), LineBreak(), DocumentBuilder.text("b"))
        result = _html(doc)
        assert "<br />" in result

    def test_html_escaping(self):
        result = _html(DocumentBuilder().paragraph("<script>alert('xss')</script>"))
        assert "<script>" not in result
        assert "&lt;script&gt;" in result


# ── Provenance ──


class TestHtmlProvenance:
    def test_data_page(self):
        from kaos_content import Heading, Text

        prov = Provenance(
            source=SourceRef(uri="test.pdf", mime_type="application/pdf"),
            page=7,
            confidence=0.95,
        )
        doc = ContentDocument(
            body=(Heading(depth=1, children=(Text(value="Title"),), provenance=prov),)
        )
        result = serialize_html(doc)
        assert 'data-page="7"' in result
        assert 'data-confidence="0.95"' in result

    def test_no_provenance_option(self):
        from kaos_content import Heading, Text

        prov = Provenance(
            source=SourceRef(uri="test.pdf", mime_type="application/pdf"),
            page=3,
        )
        doc = ContentDocument(
            body=(Heading(depth=1, children=(Text(value="Title"),), provenance=prov),)
        )
        result = serialize_html(doc, include_provenance=False)
        assert "data-page" not in result


# ── Redaction ──


class TestHtmlRedaction:
    def test_redacted_block(self):
        from kaos_content import Paragraph, Text

        doc = ContentDocument(
            body=(Paragraph(children=(Text(value="Secret"),)),),
            annotations=(
                Annotation(
                    id="r1",
                    type=AnnotationType.REDACTION,
                    targets=(AnnotationTarget(node_ref="#/body/0"),),
                ),
            ),
        )
        result = serialize_html(doc)
        assert "[REDACTED]" in result
        assert "Secret" not in result


# ── Footnotes ──


class TestHtmlFootnotes:
    def test_footnote_section(self):
        b = DocumentBuilder()
        b.paragraph(DocumentBuilder.footnote_ref("fn1"))
        b.add_footnote("fn1", "Footnote body")
        result = _html(b)
        assert 'class="footnotes"' in result
        assert 'id="fn-fn1"' in result
        assert "Footnote body" in result
