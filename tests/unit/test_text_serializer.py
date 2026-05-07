"""Tests for the plain text serializer (Phase 5).

Tests cover:
- All block types → plain text
- All inline types → plain text (formatting stripped)
- Configurable separators
- Table formats: plain and CSV
- Footnotes
- Task lists
"""

from __future__ import annotations

from kaos_content import (
    ContentDocument,
    DocumentBuilder,
    serialize_text,
)

# ── Helpers ──


def _text(builder: DocumentBuilder, **kwargs) -> str:
    """Build and serialize to text."""
    return serialize_text(builder.build(), **kwargs)


# ── Block types ──


class TestTextBlocks:
    def test_paragraph(self):
        result = _text(DocumentBuilder().paragraph("Hello world"))
        assert "Hello world" in result

    def test_multiple_paragraphs(self):
        result = _text(DocumentBuilder().paragraph("First").paragraph("Second"))
        assert "First" in result
        assert "Second" in result

    def test_heading(self):
        result = _text(DocumentBuilder().heading(1, "Title"))
        assert "Title" in result

    def test_blockquote(self):
        b = DocumentBuilder()
        b.begin_blockquote()
        b.paragraph("Quoted text")
        b.end()
        result = _text(b)
        assert "Quoted text" in result

    def test_bullet_list(self):
        b = DocumentBuilder()
        b.begin_list(ordered=False)
        b.begin_list_item()
        b.paragraph("Item 1")
        b.end()
        b.begin_list_item()
        b.paragraph("Item 2")
        b.end()
        b.end()
        result = _text(b)
        assert "- Item 1" in result
        assert "- Item 2" in result

    def test_ordered_list(self):
        b = DocumentBuilder()
        b.begin_list(ordered=True, start=3)
        b.begin_list_item()
        b.paragraph("Third")
        b.end()
        b.begin_list_item()
        b.paragraph("Fourth")
        b.end()
        b.end()
        result = _text(b)
        assert "3. Third" in result
        assert "4. Fourth" in result

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
        result = _text(b)
        assert "[x]" in result
        assert "[ ]" in result

    def test_code_block(self):
        result = _text(DocumentBuilder().code_block("print('hi')", "python"))
        assert "print('hi')" in result

    def test_thematic_break(self):
        result = _text(DocumentBuilder().paragraph("Before").thematic_break().paragraph("After"))
        assert "---" in result

    def test_math_block(self):
        result = _text(DocumentBuilder().math_block("E = mc^2"))
        assert "E = mc^2" in result

    def test_admonition(self):
        from kaos_content import Paragraph, Text

        b = DocumentBuilder()
        b.admonition("warning", Paragraph(children=(Text(value="Be careful!"),)))
        result = _text(b)
        assert "[WARNING]" in result
        assert "Be careful!" in result


# ── Inline formatting stripped ──


class TestTextInlines:
    def test_emphasis_stripped(self):
        result = _text(DocumentBuilder().paragraph(DocumentBuilder.italic("italic")))
        assert "italic" in result
        assert "*" not in result

    def test_strong_stripped(self):
        result = _text(DocumentBuilder().paragraph(DocumentBuilder.bold("bold")))
        assert "bold" in result
        assert "**" not in result

    def test_link_text_only(self):
        result = _text(
            DocumentBuilder().paragraph(DocumentBuilder.link("text", "http://example.com"))
        )
        assert "text" in result
        # URL is not included in plain text
        assert "http://example.com" not in result

    def test_image_alt_text(self):
        b = DocumentBuilder()
        b.image("image.png", "alt text")
        result = _text(b)
        assert "alt text" in result

    def test_footnote_ref(self):
        result = _text(DocumentBuilder().paragraph(DocumentBuilder.footnote_ref("fn1")))
        assert "[fn1]" in result

    def test_math_inline(self):
        result = _text(DocumentBuilder().paragraph(DocumentBuilder.math("x=y")))
        assert "x=y" in result

    def test_code_inline(self):
        result = _text(DocumentBuilder().paragraph(DocumentBuilder.code("code")))
        assert "code" in result


# ── Tables ──


class TestTextTable:
    def test_plain_format(self):
        result = _text(
            DocumentBuilder().table(["A", "B"], [["1", "2"]]),
            table_format="plain",
        )
        assert "A" in result
        assert "|" in result

    def test_csv_format(self):
        result = _text(
            DocumentBuilder().table(["A", "B"], [["1", "2"]]),
            table_format="csv",
        )
        assert "A" in result
        assert "," in result


# ── Separators ──


class TestTextSeparators:
    def test_custom_block_separator(self):
        result = serialize_text(
            DocumentBuilder().paragraph("First").paragraph("Second").build(),
            block_separator="\n---\n",
        )
        assert "\n---\n" in result

    def test_custom_heading_separator(self):
        result = serialize_text(
            DocumentBuilder().heading(1, "Title").paragraph("Body").build(),
            heading_separator="\n\n",
        )
        assert "Title\n\n" in result


# ── Footnotes ──


class TestTextFootnotes:
    def test_footnote_at_end(self):
        b = DocumentBuilder()
        b.paragraph(DocumentBuilder.footnote_ref("fn1"))
        b.add_footnote("fn1", "Footnote body")
        result = _text(b)
        assert "[fn1]" in result
        assert "Footnote body" in result


# ── Integration ──


class TestTextIntegration:
    def test_parsed_document(self):
        from kaos_content import parse_markdown

        md = """# Title

Some **bold** and *italic* text.

- Item 1
- Item 2
"""
        doc = parse_markdown(md)
        result = serialize_text(doc)
        assert "Title" in result
        assert "bold" in result
        assert "italic" in result
        assert "Item 1" in result

    def test_empty_document(self):
        result = serialize_text(ContentDocument())
        assert result.strip() == ""
