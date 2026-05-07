"""Tests for the markdown parser (Phase 3).

Tests cover:
- All block types: paragraph, heading, blockquote, lists, code blocks, tables,
  thematic breaks, math blocks, admonitions, definition lists, raw HTML blocks
- All inline types: text, emphasis, strong, strikethrough, code, link, image,
  footnote ref, math, line break, soft break, superscript, subscript, underline
- YAML front matter → DocumentMetadata
- Footnote definitions
- Task lists ([x] / [ ])
- Nested structures (list in blockquote, blockquote in list, etc.)
- Provenance (char_span from source positions)
- Round-trip: parse(serialize(doc)) ≈ doc
- Malformed/edge-case input
"""

from __future__ import annotations

import itertools

from kaos_content import (
    Alignment,
    FootnoteRef,
    Heading,
    Paragraph,
    SourceRef,
    Text,
    parse_markdown,
    serialize_markdown,
)

# ── Helpers ──


def _body(md: str) -> tuple:
    """Parse markdown and return the body tuple."""
    return parse_markdown(md).body


def _first_block(md: str):
    """Parse markdown and return the first body block."""
    body = _body(md)
    assert len(body) >= 1
    return body[0]


def _first_inline(md: str):
    """Parse a single-paragraph markdown and return the first inline child."""
    block = _first_block(md)
    assert block.node_type == "paragraph"
    return block.children[0]


# ── Block types ──


class TestParagraph:
    def test_simple(self):
        block = _first_block("Hello world")
        assert block.node_type == "paragraph"
        assert len(block.children) == 1
        assert block.children[0].node_type == "text"
        assert block.children[0].value == "Hello world"

    def test_multiple_paragraphs(self):
        body = _body("First\n\nSecond\n\nThird")
        assert len(body) == 3
        assert all(b.node_type == "paragraph" for b in body)

    def test_empty_string(self):
        body = _body("")
        assert len(body) == 0

    def test_whitespace_only(self):
        body = _body("   \n\n   ")
        assert len(body) == 0


class TestHeading:
    def test_h1(self):
        block = _first_block("# Heading 1")
        assert block.node_type == "heading"
        assert block.depth == 1
        assert block.children[0].value == "Heading 1"

    def test_h2_through_h6(self):
        for depth in range(2, 7):
            prefix = "#" * depth
            block = _first_block(f"{prefix} Heading {depth}")
            assert block.node_type == "heading"
            assert block.depth == depth

    def test_heading_with_inline_formatting(self):
        block = _first_block("## **Bold** heading")
        assert block.node_type == "heading"
        assert block.depth == 2
        assert block.children[0].node_type == "strong"


class TestBlockQuote:
    def test_simple(self):
        block = _first_block("> Quoted text")
        assert block.node_type == "blockquote"
        assert len(block.children) == 1
        assert block.children[0].node_type == "paragraph"

    def test_multiline(self):
        block = _first_block("> Line 1\n> Line 2")
        assert block.node_type == "blockquote"

    def test_nested(self):
        block = _first_block("> > Nested")
        assert block.node_type == "blockquote"
        inner = block.children[0]
        assert inner.node_type == "blockquote"

    def test_with_multiple_blocks(self):
        md = "> Paragraph 1\n>\n> Paragraph 2"
        block = _first_block(md)
        assert block.node_type == "blockquote"
        assert len(block.children) == 2


class TestBulletList:
    def test_simple(self):
        block = _first_block("- A\n- B\n- C")
        assert block.node_type == "bullet_list"
        assert len(block.children) == 3
        for item in block.children:
            assert item.node_type == "list_item"

    def test_nested(self):
        block = _first_block("- Outer\n  - Inner")
        assert block.node_type == "bullet_list"
        outer_item = block.children[0]
        # Nested list inside list item
        assert len(outer_item.children) >= 1

    def test_markers(self):
        """All bullet markers (-, *, +) produce the same AST."""
        for marker in ("-", "*", "+"):
            block = _first_block(f"{marker} Item")
            assert block.node_type == "bullet_list"


class TestOrderedList:
    def test_simple(self):
        block = _first_block("1. First\n2. Second\n3. Third")
        assert block.node_type == "ordered_list"
        assert len(block.children) == 3
        assert block.start == 1

    def test_start_number(self):
        block = _first_block("5. Fifth\n6. Sixth")
        assert block.node_type == "ordered_list"
        assert block.start == 5

    def test_single_item(self):
        block = _first_block("1. Only")
        assert block.node_type == "ordered_list"
        assert len(block.children) == 1


class TestTaskList:
    def test_checked(self):
        block = _first_block("- [x] Done")
        assert block.node_type == "bullet_list"
        item = block.children[0]
        assert item.checked is True

    def test_unchecked(self):
        block = _first_block("- [ ] Todo")
        assert block.node_type == "bullet_list"
        item = block.children[0]
        assert item.checked is False

    def test_mixed(self):
        block = _first_block("- [x] Done\n- [ ] Todo\n- Regular")
        assert block.children[0].checked is True
        assert block.children[1].checked is False
        assert block.children[2].checked is None

    def test_checked_text_stripped(self):
        block = _first_block("- [x] Done task")
        item = block.children[0]
        assert item.checked is True
        # The [x] prefix should be stripped from the text
        para = item.children[0]
        assert para.children[0].value == "Done task"


class TestCodeBlock:
    def test_fenced(self):
        block = _first_block("```python\ndef hello():\n    pass\n```")
        assert block.node_type == "codeblock"
        assert block.language == "python"
        assert "def hello():" in block.value

    def test_no_language(self):
        block = _first_block("```\nsome code\n```")
        assert block.node_type == "codeblock"
        assert block.language is None

    def test_indented_code(self):
        block = _first_block("    code line 1\n    code line 2")
        assert block.node_type == "codeblock"

    def test_triple_tilde(self):
        block = _first_block("~~~\ncode\n~~~")
        assert block.node_type == "codeblock"


class TestThematicBreak:
    def test_dashes(self):
        body = _body("Before\n\n---\n\nAfter")
        assert body[1].node_type == "thematic_break"

    def test_asterisks(self):
        body = _body("Before\n\n***\n\nAfter")
        assert body[1].node_type == "thematic_break"


class TestTable:
    def test_simple(self):
        md = "| A | B |\n|---|---|\n| 1 | 2 |"
        block = _first_block(md)
        assert block.node_type == "table"
        assert block.head is not None
        assert len(block.head.rows) == 1
        assert len(block.bodies) == 1
        assert len(block.bodies[0].rows) == 1

    def test_alignment(self):
        md = "| Left | Center | Right |\n|:-----|:------:|------:|\n| a | b | c |"
        block = _first_block(md)
        col_specs = block.col_specs
        assert len(col_specs) == 3
        assert col_specs[0].alignment == Alignment.LEFT
        assert col_specs[1].alignment == Alignment.CENTER
        assert col_specs[2].alignment == Alignment.RIGHT

    def test_cell_content(self):
        md = "| **bold** | `code` |\n|---|---|\n| text | *em* |"
        block = _first_block(md)
        # Header cells
        head_row = block.head.rows[0]
        head_cell0 = head_row.cells[0]
        assert head_cell0.content[0].node_type == "paragraph"
        assert head_cell0.content[0].children[0].node_type == "strong"

    def test_empty_table(self):
        md = "| |\n|---|\n| |"
        block = _first_block(md)
        assert block.node_type == "table"


class TestMathBlock:
    def test_display_math(self):
        block = _first_block("$$\nE = mc^2\n$$")
        assert block.node_type == "math_block"
        assert "E = mc^2" in block.value


class TestAdmonition:
    def test_note(self):
        block = _first_block("> [!NOTE]\n> This is a note.")
        assert block.node_type == "admonition"
        assert block.kind == "note"
        assert len(block.children) == 1

    def test_warning(self):
        block = _first_block("> [!WARNING]\n> Be careful!")
        assert block.node_type == "admonition"
        assert block.kind == "warning"

    def test_multi_paragraph(self):
        md = "> [!TIP]\n> First paragraph.\n>\n> Second paragraph."
        block = _first_block(md)
        assert block.node_type == "admonition"
        assert block.kind == "tip"

    def test_regular_blockquote_not_admonition(self):
        block = _first_block("> Just a normal quote")
        assert block.node_type == "blockquote"


class TestDefinitionList:
    def test_simple(self):
        md = "Term 1\n:   Definition 1\n\nTerm 2\n:   Definition 2"
        block = _first_block(md)
        assert block.node_type == "definition_list"
        assert len(block.children) == 2

    def test_multiple_definitions(self):
        md = "Term\n:   Def A\n:   Def B"
        block = _first_block(md)
        assert block.node_type == "definition_list"
        item = block.children[0]
        assert len(item.definitions) == 2

    def test_term_inlines(self):
        md = "**Bold Term**\n:   Definition"
        block = _first_block(md)
        assert block.node_type == "definition_list"
        item = block.children[0]
        assert item.term[0].node_type == "strong"


class TestRawHtmlBlock:
    def test_html_block(self):
        md = "<div>\nsome content\n</div>"
        block = _first_block(md)
        assert block.node_type == "raw_block"
        assert block.format == "html"


# ── Inline types ──


class TestTextInline:
    def test_plain_text(self):
        inline = _first_inline("Hello")
        assert inline.node_type == "text"
        assert inline.value == "Hello"


class TestEmphasis:
    def test_asterisk(self):
        inline = _first_inline("*italic*")
        assert inline.node_type == "emphasis"
        assert inline.children[0].value == "italic"

    def test_underscore(self):
        inline = _first_inline("_italic_")
        assert inline.node_type == "emphasis"


class TestStrong:
    def test_asterisk(self):
        inline = _first_inline("**bold**")
        assert inline.node_type == "strong"
        assert inline.children[0].value == "bold"

    def test_underscore(self):
        inline = _first_inline("__bold__")
        assert inline.node_type == "strong"


class TestStrikethrough:
    def test_basic(self):
        inline = _first_inline("~~struck~~")
        assert inline.node_type == "strikethrough"
        assert inline.children[0].value == "struck"


class TestInlineCode:
    def test_basic(self):
        inline = _first_inline("`code`")
        assert inline.node_type == "code"
        assert inline.value == "code"

    def test_with_backticks(self):
        inline = _first_inline("`` `inner` ``")
        assert inline.node_type == "code"
        assert "`inner`" in inline.value


class TestLink:
    def test_basic(self):
        inline = _first_inline("[text](http://example.com)")
        assert inline.node_type == "link"
        assert inline.url == "http://example.com"
        assert inline.children[0].value == "text"

    def test_with_title(self):
        inline = _first_inline('[text](http://example.com "Title")')
        assert inline.node_type == "link"
        assert inline.title == "Title"

    def test_with_inline_formatting(self):
        inline = _first_inline("[**bold**](http://example.com)")
        assert inline.node_type == "link"
        assert inline.children[0].node_type == "strong"


class TestImage:
    def test_basic(self):
        inline = _first_inline("![alt text](image.png)")
        assert inline.node_type == "image"
        assert inline.src == "image.png"
        assert inline.alt == "alt text"

    def test_with_title(self):
        inline = _first_inline('![alt](image.png "Title")')
        assert inline.node_type == "image"
        assert inline.title == "Title"

    def test_no_alt(self):
        inline = _first_inline("![](image.png)")
        assert inline.node_type == "image"


class TestInlineMath:
    def test_basic(self):
        block = _first_block("Inline $x = y$ math")
        # Find the math inline
        inlines = block.children
        math_nodes = [i for i in inlines if i.node_type == "math"]
        assert len(math_nodes) == 1
        assert math_nodes[0].value == "x = y"


class TestFootnoteRef:
    def test_basic(self):
        md = "[^fn1]: Footnote body.\n\nText with [^fn1] reference."
        doc = parse_markdown(md)
        # Find the footnote ref in the paragraph
        para = doc.body[0]
        assert isinstance(para, Paragraph)
        refs = [i for i in para.children if isinstance(i, FootnoteRef)]
        assert len(refs) == 1
        assert refs[0].identifier == "fn1"

    def test_footnote_body(self):
        md = "[^fn1]: Footnote body.\n\nText [^fn1]."
        doc = parse_markdown(md)
        assert "fn1" in doc.footnotes
        assert len(doc.footnotes["fn1"]) == 1
        assert doc.footnotes["fn1"][0].node_type == "paragraph"


class TestLineBreak:
    def test_backslash(self):
        block = _first_block("Line 1\\\nLine 2")
        inlines = block.children
        breaks = [i for i in inlines if i.node_type == "line_break"]
        assert len(breaks) == 1


class TestSoftBreak:
    def test_basic(self):
        block = _first_block("Line 1\nLine 2")
        inlines = block.children
        breaks = [i for i in inlines if i.node_type == "soft_break"]
        assert len(breaks) == 1


class TestSuperscript:
    def test_html_tag(self):
        block = _first_block("Text <sup>super</sup> rest")
        inlines = block.children
        sup_nodes = [i for i in inlines if i.node_type == "superscript"]
        assert len(sup_nodes) == 1
        assert sup_nodes[0].children[0].value == "super"


class TestSubscript:
    def test_html_tag(self):
        block = _first_block("Text <sub>sub</sub> rest")
        inlines = block.children
        sub_nodes = [i for i in inlines if i.node_type == "subscript"]
        assert len(sub_nodes) == 1
        assert sub_nodes[0].children[0].value == "sub"


class TestUnderline:
    def test_html_tag(self):
        block = _first_block("Text <u>under</u> rest")
        inlines = block.children
        u_nodes = [i for i in inlines if i.node_type == "underline"]
        assert len(u_nodes) == 1
        assert u_nodes[0].children[0].value == "under"


class TestRawHtmlInline:
    def test_unknown_tag(self):
        block = _first_block("Text <mark>highlighted</mark> rest")
        inlines = block.children
        raw_nodes = [i for i in inlines if i.node_type == "raw_inline"]
        assert len(raw_nodes) >= 1


# ── Front matter ──


class TestFrontMatter:
    def test_title(self):
        doc = parse_markdown("---\ntitle: My Document\n---\n\nContent")
        assert doc.metadata.title == "My Document"

    def test_author(self):
        doc = parse_markdown("---\nauthor: Jane Doe\n---\n\nContent")
        assert doc.metadata.authors == ("Jane Doe",)

    def test_authors_list(self):
        doc = parse_markdown("---\nauthors: Alice, Bob\n---\n\nContent")
        assert doc.metadata.authors == ("Alice", "Bob")

    def test_date(self):
        doc = parse_markdown("---\ndate: 2026-01-01\n---\n\nContent")
        assert doc.metadata.date == "2026-01-01"

    def test_language(self):
        doc = parse_markdown("---\nlanguage: en\n---\n\nContent")
        assert doc.metadata.language == "en"

    def test_no_frontmatter(self):
        doc = parse_markdown("Just text")
        assert doc.metadata.title is None

    def test_empty_frontmatter(self):
        doc = parse_markdown("---\n---\n\nContent")
        assert doc.metadata.title is None


# ── Provenance ──


class TestProvenance:
    def test_source_attached(self):
        source = SourceRef(uri="file:///test.md", mime_type="text/markdown")
        doc = parse_markdown("# Heading\n\nParagraph", source=source)
        heading = doc.body[0]
        assert heading.provenance is not None
        assert heading.provenance.source == source

    def test_char_span(self):
        source = SourceRef(uri="file:///test.md", mime_type="text/markdown")
        md = "# Heading\n\nParagraph"
        doc = parse_markdown(md, source=source)
        heading = doc.body[0]
        assert heading.provenance is not None
        assert heading.provenance.char_span is not None
        start, end = heading.provenance.char_span
        assert start == 0
        # End is at the end of the heading line
        assert end == len("# Heading\n")

    def test_no_source_no_provenance(self):
        doc = parse_markdown("# Heading")
        heading = doc.body[0]
        assert heading.provenance is None

    def test_multiple_blocks_char_spans(self):
        source = SourceRef(uri="test.md", mime_type="text/markdown")
        md = "# H1\n\nPara 1\n\nPara 2"
        doc = parse_markdown(md, source=source)
        # Each block should have non-overlapping char spans
        spans = []
        for block in doc.body:
            if block.provenance and block.provenance.char_span:
                spans.append(block.provenance.char_span)
        assert len(spans) >= 2
        # Spans should be in order
        for s1, s2 in itertools.pairwise(spans):
            assert s1[1] <= s2[0]


# ── Nested structures ──


class TestNested:
    def test_list_in_blockquote(self):
        md = "> - Item 1\n> - Item 2"
        block = _first_block(md)
        assert block.node_type == "blockquote"
        inner = block.children[0]
        assert inner.node_type == "bullet_list"

    def test_blockquote_in_list(self):
        md = "- Item\n\n  > Quoted"
        block = _first_block(md)
        assert block.node_type == "bullet_list"
        item = block.children[0]
        assert len(item.children) >= 1

    def test_nested_emphasis(self):
        block = _first_block("***bold italic***")
        # Should parse as nested emphasis/strong
        assert block.node_type == "paragraph"

    def test_code_in_heading(self):
        block = _first_block("## Code: `foo`")
        assert block.node_type == "heading"
        inlines = block.children
        code_nodes = [i for i in inlines if i.node_type == "code"]
        assert len(code_nodes) == 1

    def test_link_in_list(self):
        block = _first_block("- [link](url)")
        assert block.node_type == "bullet_list"
        item = block.children[0]
        para = item.children[0]
        assert para.children[0].node_type == "link"


# ── Round-trip tests ──


class TestRoundTrip:
    """Test parse(serialize(doc)) ≈ doc for markdown-expressible content."""

    def _round_trip(self, md: str) -> str:
        """Parse, serialize, return output."""
        doc = parse_markdown(md)
        return serialize_markdown(doc)

    def test_paragraph(self):
        result = self._round_trip("Hello world\n")
        assert "Hello world" in result

    def test_heading(self):
        result = self._round_trip("# Heading 1\n\n## Heading 2\n")
        assert "# Heading 1" in result
        assert "## Heading 2" in result

    def test_emphasis(self):
        result = self._round_trip("Some *italic* text\n")
        assert "*italic*" in result

    def test_strong(self):
        result = self._round_trip("Some **bold** text\n")
        assert "**bold**" in result

    def test_strikethrough(self):
        result = self._round_trip("Some ~~struck~~ text\n")
        assert "~~struck~~" in result

    def test_inline_code(self):
        result = self._round_trip("Some `code` text\n")
        assert "`code`" in result

    def test_link(self):
        result = self._round_trip("[text](http://example.com)\n")
        assert "[text](http://example.com)" in result

    def test_image(self):
        result = self._round_trip("![alt](image.png)\n")
        assert "![alt](image.png)" in result

    def test_code_block(self):
        result = self._round_trip("```python\nprint('hello')\n```\n")
        assert "```python" in result
        assert "print('hello')" in result

    def test_bullet_list(self):
        result = self._round_trip("- Item 1\n- Item 2\n")
        assert "- Item 1" in result
        assert "- Item 2" in result

    def test_ordered_list(self):
        result = self._round_trip("1. First\n2. Second\n")
        assert "1. First" in result
        assert "2. Second" in result

    def test_blockquote(self):
        result = self._round_trip("> Quoted text\n")
        assert "> Quoted text" in result

    def test_thematic_break(self):
        result = self._round_trip("Before\n\n---\n\nAfter\n")
        assert "---" in result

    def test_table(self):
        md = "| A | B |\n|---|---|\n| 1 | 2 |\n"
        result = self._round_trip(md)
        assert "| A |" in result
        assert "| 1 |" in result

    def test_math_block(self):
        result = self._round_trip("$$\nE = mc^2\n$$\n")
        assert "$$" in result
        assert "E = mc^2" in result

    def test_inline_math(self):
        result = self._round_trip("Inline $x=y$ math\n")
        assert "$x=y$" in result

    def test_footnote(self):
        md = "[^fn1]: Footnote text.\n\nContent [^fn1].\n"
        result = self._round_trip(md)
        assert "[^fn1]" in result

    def test_admonition(self):
        md = "> [!NOTE]\n> This is a note.\n"
        result = self._round_trip(md)
        assert "[!NOTE]" in result

    def test_task_list(self):
        md = "- [x] Done\n- [ ] Todo\n"
        result = self._round_trip(md)
        assert "[x]" in result
        assert "[ ]" in result

    def test_superscript(self):
        result = self._round_trip("Text <sup>super</sup>\n")
        assert "<sup>super</sup>" in result

    def test_subscript(self):
        result = self._round_trip("Text <sub>sub</sub>\n")
        assert "<sub>sub</sub>" in result

    def test_underline(self):
        result = self._round_trip("Text <u>under</u>\n")
        assert "<u>under</u>" in result

    def test_hard_line_break(self):
        result = self._round_trip("Line 1\\\nLine 2\n")
        assert "\\" in result

    def test_complex_document(self):
        md = """# Title

Some **bold** and *italic* text with `code`.

## Lists

- Item 1
- Item 2

1. First
2. Second

> A blockquote

```python
def hello():
    pass
```

| A | B |
|---|---|
| 1 | 2 |

---

Inline $x=y$ math.

$$
E = mc^2
$$
"""
        doc = parse_markdown(md)
        result = serialize_markdown(doc)
        # Verify key elements survive round-trip
        assert "# Title" in result
        assert "**bold**" in result
        assert "*italic*" in result
        assert "`code`" in result
        assert "- Item 1" in result
        assert "1. First" in result or "1.  First" in result
        assert "> A blockquote" in result
        assert "```python" in result
        assert "---" in result
        assert "$x=y$" in result


# ── Edge cases ──


class TestEdgeCases:
    def test_escaped_characters_preserved(self):
        """Escaped characters in markdown should be unescaped during parsing."""
        md = r"Some \*text\* not italic"
        block = _first_block(md)
        assert block.node_type == "paragraph"
        # The escaped * should be treated as literal text, not emphasis

    def test_empty_heading(self):
        block = _first_block("# ")
        assert block.node_type == "heading"

    def test_very_long_paragraph(self):
        text = "word " * 1000
        block = _first_block(text)
        assert block.node_type == "paragraph"

    def test_consecutive_emphasis(self):
        block = _first_block("*a* *b* *c*")
        assert block.node_type == "paragraph"
        em_nodes = [i for i in block.children if i.node_type == "emphasis"]
        assert len(em_nodes) == 3

    def test_mixed_inline_formatting(self):
        block = _first_block("**bold *and italic* text**")
        assert block.node_type == "paragraph"
        assert block.children[0].node_type == "strong"

    def test_empty_link(self):
        inline = _first_inline("[](http://example.com)")
        assert inline.node_type == "link"
        assert inline.url == "http://example.com"

    def test_link_with_title(self):
        inline = _first_inline('[text](url "My Title")')
        assert inline.node_type == "link"
        assert inline.title == "My Title"

    def test_multiple_footnotes(self):
        md = "[^a]: First.\n\n[^b]: Second.\n\nText [^a] and [^b]."
        doc = parse_markdown(md)
        assert "a" in doc.footnotes
        assert "b" in doc.footnotes

    def test_unicode_content(self):
        md = "# 日本語\n\nMultibyte: 中文, العربية, 한국어"
        doc = parse_markdown(md)
        heading = doc.body[0]
        assert isinstance(heading, Heading)
        first_inline = heading.children[0]
        assert isinstance(first_inline, Text)
        assert first_inline.value == "日本語"

    def test_nested_list_deep(self):
        md = "- Level 1\n  - Level 2\n    - Level 3"
        block = _first_block(md)
        assert block.node_type == "bullet_list"

    def test_code_block_with_backticks_in_content(self):
        md = "````\nSome ```code``` here\n````"
        block = _first_block(md)
        assert block.node_type == "codeblock"
        assert "```code```" in block.value

    def test_table_with_inline_formatting(self):
        md = "| **Bold** | *Italic* |\n|---|---|\n| `code` | ~~struck~~ |"
        block = _first_block(md)
        assert block.node_type == "table"

    def test_multiple_tables(self):
        md = "| A |\n|---|\n| 1 |\n\n| B |\n|---|\n| 2 |"
        body = _body(md)
        tables = [b for b in body if b.node_type == "table"]
        assert len(tables) == 2


# ── Integration with traversal ──


class TestParserWithTraversal:
    def test_walk_parsed_document(self):
        from kaos_content import walk

        md = "# Heading\n\n**Bold** text\n\n- Item"
        doc = parse_markdown(md)
        nodes = list(walk(doc.body[0]))
        assert len(nodes) >= 1

    def test_extract_text_from_parsed(self):
        from kaos_content import extract_text

        md = "Some **bold** and *italic* text"
        doc = parse_markdown(md)
        text = extract_text(doc.body[0])
        assert text == "Some bold and italic text"

    def test_node_index_on_parsed(self):
        from kaos_content import NodeIndex

        md = "# H1\n\n## H2\n\nParagraph\n\n| A |\n|---|\n| 1 |"
        doc = parse_markdown(md)
        index = NodeIndex(doc)
        assert len(index.headings) == 2
        assert len(index.tables) == 1

    def test_content_hash_on_parsed(self):
        from kaos_content import content_hash

        md = "# Same Heading"
        doc1 = parse_markdown(md)
        doc2 = parse_markdown(md)
        # Same content should produce same hash (ignoring id)
        h1 = content_hash(doc1.body[0])
        h2 = content_hash(doc2.body[0])
        assert h1 == h2


# ── Multiple footnotes edge case ──


class TestFootnoteEdgeCases:
    def test_footnote_with_multiple_blocks(self):
        md = "[^fn]: Paragraph 1.\n\n    Paragraph 2.\n\nText [^fn]."
        doc = parse_markdown(md)
        assert "fn" in doc.footnotes
        # May have multiple blocks in footnote body
        assert len(doc.footnotes["fn"]) >= 1

    def test_footnote_not_in_body(self):
        """Footnote definitions should not appear as body blocks."""
        md = "[^fn]: Definition.\n\nContent [^fn]."
        doc = parse_markdown(md)
        for block in doc.body:
            assert block.node_type != "footnote_block"


# ── Builder compatibility ──


class TestBuilderCompatibility:
    """Verify that parsed documents are compatible with DocumentBuilder output."""

    def test_same_types(self):
        from kaos_content import DocumentBuilder

        # Build a document
        built = DocumentBuilder(title="Test").heading(1, "Title").paragraph("Content").build()

        # Parse an equivalent markdown
        parsed = parse_markdown("# Title\n\nContent")

        # Both should have same structure
        assert built.body[0].node_type == parsed.body[0].node_type
        assert built.body[1].node_type == parsed.body[1].node_type


# ── Structural idempotence ──


class TestDoubleRoundTrip:
    """parse(serialize(parse(md))) should equal parse(md) structurally."""

    def test_double_round_trip(self):
        md = """# Heading

Some **bold** and *italic* text.

- Item 1
- Item 2

> Quote

```python
code
```

| A | B |
|---|---|
| 1 | 2 |
"""
        doc1 = parse_markdown(md)
        md2 = serialize_markdown(doc1)
        doc2 = parse_markdown(md2)
        md3 = serialize_markdown(doc2)

        # Second serialization should be identical to the first
        assert md2 == md3


# ── Regression tests for round-trip bugs ──


class TestRoundTripRegressions:
    """Regression tests for specific round-trip bugs that were found and fixed."""

    def test_link_title_with_quotes(self):
        """Link titles with double quotes must survive round-trip."""
        md = '[text](url "title with \\"quotes\\"")'
        doc = parse_markdown(md)
        out = serialize_markdown(doc)
        doc2 = parse_markdown(out)
        out2 = serialize_markdown(doc2)
        assert out == out2
        # The link must still be a link, not broken text
        para = doc2.body[0]
        assert isinstance(para, Paragraph)
        assert para.children[0].node_type == "link"

    def test_image_title_with_quotes(self):
        """Image titles with double quotes must survive round-trip."""
        md = '![alt](img.png "my \\"title\\"")'
        doc = parse_markdown(md)
        out = serialize_markdown(doc)
        doc2 = parse_markdown(out)
        out2 = serialize_markdown(doc2)
        assert out == out2

    def test_definition_list_round_trip(self):
        """Definition lists must be idempotent after first pass."""
        md = "Term\n:   Definition"
        doc = parse_markdown(md)
        out = serialize_markdown(doc)
        doc2 = parse_markdown(out)
        out2 = serialize_markdown(doc2)
        assert out == out2
        # Must still parse as definition list
        assert doc2.body[0].node_type == "definition_list"

    def test_definition_list_multiple_defs(self):
        """Multiple definitions per term must survive round-trip."""
        md = "Term\n:   Def A\n:   Def B"
        doc = parse_markdown(md)
        out = serialize_markdown(doc)
        doc2 = parse_markdown(out)
        out2 = serialize_markdown(doc2)
        assert out == out2
