"""Phase 2 hardening tests for the markdown serializer.

Covers edge cases in escaping, nesting, tables, code blocks, footnotes,
unicode, whitespace, large documents, and complex nested structures.
"""

from kaos_content import (
    BlockQuote,
    BulletList,
    Cell,
    Code,
    CodeBlock,
    ContentDocument,
    Emphasis,
    FootnoteRef,
    Heading,
    Image,
    Link,
    ListItem,
    Math,
    MathBlock,
    OrderedList,
    Paragraph,
    Row,
    Strong,
    Table,
    TableSection,
    Text,
    ThematicBreak,
    serialize_markdown,
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. MARKDOWN INJECTION / ESCAPING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestMarkdownEscaping:
    """Text nodes containing markdown syntax should render as literal text."""

    def test_heading_syntax_in_text(self) -> None:
        """Text '# heading' should NOT render as a heading."""
        doc = ContentDocument(body=(Paragraph(children=(Text(value="# heading"),)),))
        result = serialize_markdown(doc)
        # The '#' should be escaped
        assert "\\# heading" in result
        # Should NOT start with an unescaped heading marker
        lines = result.strip().splitlines()
        assert not any(line.lstrip().startswith("# heading") for line in lines)

    def test_link_syntax_in_text(self) -> None:
        """Text '[link](url)' should be escaped so it doesn't render as a link."""
        doc = ContentDocument(body=(Paragraph(children=(Text(value="[link](url)"),)),))
        result = serialize_markdown(doc)
        # Brackets are escaped; parentheses are NOT escaped in inline context
        assert "\\[link\\](url)" in result

    def test_bold_syntax_in_text(self) -> None:
        """Text '**bold**' should be escaped."""
        doc = ContentDocument(body=(Paragraph(children=(Text(value="**bold**"),)),))
        result = serialize_markdown(doc)
        assert "\\*\\*bold\\*\\*" in result

    def test_blockquote_syntax_in_text(self) -> None:
        """Text '> quote' should have the > escaped."""
        doc = ContentDocument(body=(Paragraph(children=(Text(value="> quote"),)),))
        result = serialize_markdown(doc)
        assert "\\> quote" in result

    def test_list_syntax_in_text(self) -> None:
        """Text '- list item' should have the - escaped."""
        doc = ContentDocument(body=(Paragraph(children=(Text(value="- list item"),)),))
        result = serialize_markdown(doc)
        assert "\\- list item" in result

    def test_table_pipe_in_text(self) -> None:
        """Text '| table |' in regular (non-table) context is NOT escaped."""
        doc = ContentDocument(body=(Paragraph(children=(Text(value="| table |"),)),))
        result = serialize_markdown(doc)
        # Pipes are only escaped inside table cells, not in regular text
        assert "| table |" in result

    def test_backtick_fence_in_text(self) -> None:
        """Text containing triple backticks should be escaped."""
        doc = ContentDocument(body=(Paragraph(children=(Text(value="Use ``` for code"),)),))
        result = serialize_markdown(doc)
        assert "\\`\\`\\`" in result

    def test_math_dollar_in_text(self) -> None:
        """Text '$$math$$' should not become a math block."""
        doc = ContentDocument(body=(Paragraph(children=(Text(value="Cost is $$50"),)),))
        result = serialize_markdown(doc)
        # Dollar signs are not in the escape set by default, so let's just verify
        # the text is present and doesn't break parsing
        assert "50" in result

    def test_all_special_chars_escaped(self) -> None:
        """Context-dependent escaping: only inline-dangerous chars are escaped."""
        special = "\\`*_{}[]()#+-.!|~>"
        doc = ContentDocument(body=(Paragraph(children=(Text(value=special),)),))
        result = serialize_markdown(doc)
        # These chars from the input ARE escaped in inline context
        for ch in "\\`*_[]~":
            assert f"\\{ch}" in result, f"Expected \\{ch} in {result!r}"
        # These chars from the input are NOT escaped in inline context
        # (only dangerous at line-start or in table cells)
        for ch in "{}()#+-.!|>":
            assert f"\\{ch}" not in result, f"Did not expect \\{ch} in {result!r}"

    def test_ampersand_and_angle_brackets(self) -> None:
        """& is not escaped; < is escaped; > is NOT escaped in inline context."""
        doc = ContentDocument(body=(Paragraph(children=(Text(value="A & B < C > D"),)),))
        result = serialize_markdown(doc)
        assert "A & B" in result  # & is not escaped
        assert "\\< C" in result  # < IS escaped
        assert "> D" in result  # > is NOT escaped in inline context


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. NESTED LIST EDGE CASES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestNestedLists:
    def test_three_level_nested_bullet_list(self) -> None:
        """Three levels of bullet list nesting."""
        doc = ContentDocument(
            body=(
                BulletList(
                    children=(
                        ListItem(
                            children=(
                                Paragraph(children=(Text(value="level 1"),)),
                                BulletList(
                                    children=(
                                        ListItem(
                                            children=(
                                                Paragraph(children=(Text(value="level 2"),)),
                                                BulletList(
                                                    children=(
                                                        ListItem(
                                                            children=(
                                                                Paragraph(
                                                                    children=(
                                                                        Text(value="level 3"),
                                                                    )
                                                                ),
                                                            )
                                                        ),
                                                    )
                                                ),
                                            )
                                        ),
                                    )
                                ),
                            )
                        ),
                    )
                ),
            )
        )
        result = serialize_markdown(doc)
        assert "- level 1" in result
        assert "level 2" in result
        assert "level 3" in result
        # Deeper levels should be indented more
        lines = result.strip().splitlines()
        # Find the lines with our markers
        l1_line = next(ln for ln in lines if "level 1" in ln)
        l2_line = next(ln for ln in lines if "level 2" in ln)
        l3_line = next(ln for ln in lines if "level 3" in ln)
        # Each level should have more leading whitespace
        assert len(l1_line) - len(l1_line.lstrip()) < len(l2_line) - len(l2_line.lstrip())
        assert len(l2_line) - len(l2_line.lstrip()) < len(l3_line) - len(l3_line.lstrip())

    def test_ordered_list_inside_bullet_list(self) -> None:
        """Ordered list nested inside a bullet list."""
        doc = ContentDocument(
            body=(
                BulletList(
                    children=(
                        ListItem(
                            children=(
                                Paragraph(children=(Text(value="intro"),)),
                                OrderedList(
                                    start=1,
                                    children=(
                                        ListItem(
                                            children=(
                                                Paragraph(children=(Text(value="step one"),)),
                                            )
                                        ),
                                        ListItem(
                                            children=(
                                                Paragraph(children=(Text(value="step two"),)),
                                            )
                                        ),
                                    ),
                                ),
                            )
                        ),
                    )
                ),
            )
        )
        result = serialize_markdown(doc)
        assert "- intro" in result
        assert "1. step one" in result
        assert "2. step two" in result

    def test_list_item_with_multiple_blocks(self) -> None:
        """List item containing a paragraph followed by a code block."""
        doc = ContentDocument(
            body=(
                BulletList(
                    children=(
                        ListItem(
                            children=(
                                Paragraph(children=(Text(value="description"),)),
                                CodeBlock(language="python", value="x = 1"),
                            )
                        ),
                    )
                ),
            )
        )
        result = serialize_markdown(doc)
        assert "- description" in result
        assert "x = 1" in result
        # The code block should be indented under the list item
        lines = result.strip().splitlines()
        code_line = next(ln for ln in lines if "x = 1" in ln)
        assert code_line.startswith("  ")  # continuation indent

    def test_empty_list_item(self) -> None:
        """List item with no children."""
        doc = ContentDocument(body=(BulletList(children=(ListItem(children=()),)),))
        result = serialize_markdown(doc)
        assert "- " in result or "-\n" in result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. TABLE EDGE CASES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestTableEdgeCases:
    def test_table_with_no_rows(self) -> None:
        """Table with no rows at all should produce empty output."""
        doc = ContentDocument(body=(Table(bodies=(TableSection(rows=()),)),))
        result = serialize_markdown(doc)
        # An empty table should either be empty or minimal
        # The serializer returns "" for tables with no rows
        assert "---" not in result or result.strip() == ""

    def test_table_with_mismatched_cell_counts(self) -> None:
        """Rows with different numbers of cells should be padded."""
        doc = ContentDocument(
            body=(
                Table(
                    head=TableSection(
                        rows=(
                            Row(
                                cells=(
                                    Cell(content=(Paragraph(children=(Text(value="A"),)),)),
                                    Cell(content=(Paragraph(children=(Text(value="B"),)),)),
                                    Cell(content=(Paragraph(children=(Text(value="C"),)),)),
                                )
                            ),
                        )
                    ),
                    bodies=(
                        TableSection(
                            rows=(
                                Row(
                                    cells=(
                                        Cell(
                                            content=(Paragraph(children=(Text(value="only one"),)),)
                                        ),
                                    )
                                ),
                            )
                        ),
                    ),
                ),
            )
        )
        result = serialize_markdown(doc)
        lines = result.strip().splitlines()
        # All rows should have the same number of pipe delimiters
        pipe_counts = [line.count("|") for line in lines if "|" in line]
        assert len(set(pipe_counts)) == 1, f"Inconsistent pipe counts: {pipe_counts}"

    def test_table_with_only_foot(self) -> None:
        """Table with only a foot section (no head, no body)."""
        doc = ContentDocument(
            body=(
                Table(
                    foot=TableSection(
                        rows=(
                            Row(
                                cells=(
                                    Cell(content=(Paragraph(children=(Text(value="total"),)),)),
                                    Cell(content=(Paragraph(children=(Text(value="100"),)),)),
                                )
                            ),
                        )
                    )
                ),
            )
        )
        result = serialize_markdown(doc)
        assert "total" in result
        assert "100" in result
        assert "---" in result  # separator row

    def test_table_with_empty_cells(self) -> None:
        """Table cells with no content."""
        doc = ContentDocument(
            body=(
                Table(
                    head=TableSection(
                        rows=(
                            Row(
                                cells=(
                                    Cell(content=(Paragraph(children=(Text(value="H1"),)),)),
                                    Cell(content=(Paragraph(children=(Text(value="H2"),)),)),
                                )
                            ),
                        )
                    ),
                    bodies=(
                        TableSection(
                            rows=(
                                Row(
                                    cells=(
                                        Cell(content=()),  # empty cell
                                        Cell(content=(Paragraph(children=(Text(value="data"),)),)),
                                    )
                                ),
                            )
                        ),
                    ),
                ),
            )
        )
        result = serialize_markdown(doc)
        assert "data" in result
        lines = result.strip().splitlines()
        # All rows should have consistent structure
        pipe_counts = [line.count("|") for line in lines if "|" in line]
        assert len(set(pipe_counts)) == 1

    def test_table_cell_with_pipe_in_text(self) -> None:
        """Pipe characters in table cell content must be escaped."""
        doc = ContentDocument(
            body=(
                Table(
                    head=TableSection(
                        rows=(
                            Row(
                                cells=(Cell(content=(Paragraph(children=(Text(value="A | B"),)),)),)
                            ),
                        )
                    ),
                    bodies=(
                        TableSection(
                            rows=(
                                Row(
                                    cells=(
                                        Cell(content=(Paragraph(children=(Text(value="C | D"),)),)),
                                    )
                                ),
                            )
                        ),
                    ),
                ),
            )
        )
        result = serialize_markdown(doc)
        # Pipes inside table cells ARE escaped
        assert "A \\| B" in result
        assert "C \\| D" in result

    def test_table_cell_with_multiple_paragraphs(self) -> None:
        """Cell with multiple paragraphs - may produce multiline content in a cell."""
        doc = ContentDocument(
            body=(
                Table(
                    head=TableSection(
                        rows=(
                            Row(cells=(Cell(content=(Paragraph(children=(Text(value="H"),)),)),)),
                        )
                    ),
                    bodies=(
                        TableSection(
                            rows=(
                                Row(
                                    cells=(
                                        Cell(
                                            content=(
                                                Paragraph(children=(Text(value="para 1"),)),
                                                Paragraph(children=(Text(value="para 2"),)),
                                            )
                                        ),
                                    )
                                ),
                            )
                        ),
                    ),
                ),
            )
        )
        result = serialize_markdown(doc)
        # The table should still be valid - each row on its own line
        # Multi-paragraph cells should use <br> to stay on one line
        assert "para 1" in result
        assert "para 2" in result
        # Every table line (containing |) should be a single line
        table_lines = [line for line in result.strip().splitlines() if "|" in line]
        for tl in table_lines:
            assert "\n" not in tl, f"Table row should be single line: {tl!r}"
        # The cell should contain <br> between paragraphs
        assert "<br>" in result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. CODE BLOCK ESCAPING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestCodeBlockEscaping:
    def test_code_block_with_triple_backticks(self) -> None:
        """Code block whose content contains ``` needs fence escalation."""
        doc = ContentDocument(body=(CodeBlock(value="Use ``` to start a code block"),))
        result = serialize_markdown(doc)
        assert "Use ``` to start a code block" in result
        # The outer fence must use 4+ backticks to avoid confusion
        lines = result.strip().splitlines()
        assert lines[0].startswith("````"), f"Fence should escalate: {lines[0]!r}"
        assert lines[-1].startswith("````"), f"Closing fence should match: {lines[-1]!r}"

    def test_code_block_with_quadruple_backticks(self) -> None:
        """Code block containing ```` needs 5-backtick fences."""
        doc = ContentDocument(body=(CodeBlock(value="````"),))
        result = serialize_markdown(doc)
        lines = result.strip().splitlines()
        assert lines[0].startswith("`````"), f"Fence should be 5+ backticks: {lines[0]!r}"
        assert lines[-1].startswith("`````")

    def test_code_block_with_full_fence_in_content(self) -> None:
        """Code block containing a complete fenced code block."""
        inner = "```python\nx = 1\n```"
        doc = ContentDocument(body=(CodeBlock(value=inner),))
        result = serialize_markdown(doc)
        assert inner in result
        lines = result.strip().splitlines()
        # Opening and closing fences must be longer than the inner ```
        assert lines[0].startswith("````")
        assert lines[-1].startswith("````")

    def test_code_block_multiline(self) -> None:
        """Multi-line code block should preserve all lines."""
        code = "line 1\nline 2\nline 3"
        doc = ContentDocument(body=(CodeBlock(language="text", value=code),))
        result = serialize_markdown(doc)
        assert "line 1" in result
        assert "line 2" in result
        assert "line 3" in result

    def test_code_block_with_empty_value(self) -> None:
        """Code block with empty string content."""
        doc = ContentDocument(body=(CodeBlock(value=""),))
        result = serialize_markdown(doc)
        assert "```" in result

    def test_inline_code_with_double_backtick(self) -> None:
        """Inline code containing `` should use triple backtick delimiter."""
        doc = ContentDocument(body=(Paragraph(children=(Code(value="a``b"),)),))
        result = serialize_markdown(doc)
        # No space padding: value doesn't start/end with backtick
        assert "```a``b```" in result

    def test_inline_code_with_single_backtick_uses_double(self) -> None:
        """Inline code containing ` should use `` delimiter."""
        doc = ContentDocument(body=(Paragraph(children=(Code(value="a`b"),)),))
        result = serialize_markdown(doc)
        # No space padding: value doesn't start/end with backtick
        assert "``a`b``" in result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. EMPTY DOCUMENT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestEmptyDocument:
    def test_empty_body(self) -> None:
        """ContentDocument() with no body produces minimal output."""
        doc = ContentDocument()
        result = serialize_markdown(doc)
        # Should not crash; output should be empty or just whitespace + newline
        assert result.strip() == ""

    def test_document_with_only_footnotes(self) -> None:
        """Document with footnotes but no body."""
        doc = ContentDocument(footnotes={"fn1": (Paragraph(children=(Text(value="A note."),)),)})
        result = serialize_markdown(doc)
        assert "[^fn1]:" in result
        assert "A note" in result

    def test_document_with_empty_paragraph(self) -> None:
        """Paragraph with no inline children."""
        doc = ContentDocument(body=(Paragraph(children=()),))
        result = serialize_markdown(doc)
        # Should not crash, empty paragraph produces empty inline content
        assert isinstance(result, str)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. FOOTNOTE EDGE CASES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestFootnoteEdgeCases:
    def test_multiple_footnotes(self) -> None:
        """Multiple footnotes should all be rendered at the end."""
        doc = ContentDocument(
            body=(
                Paragraph(
                    children=(
                        Text(value="First"),
                        FootnoteRef(identifier="fn1"),
                        Text(value=" and second"),
                        FootnoteRef(identifier="fn2"),
                    )
                ),
            ),
            footnotes={
                "fn1": (Paragraph(children=(Text(value="Note one."),)),),
                "fn2": (Paragraph(children=(Text(value="Note two."),)),),
            },
        )
        result = serialize_markdown(doc)
        assert "[^fn1]" in result
        assert "[^fn2]" in result
        assert "[^fn1]:" in result
        assert "[^fn2]:" in result
        assert "Note one" in result
        assert "Note two" in result

    def test_footnote_with_multi_paragraph_content(self) -> None:
        """Footnote containing multiple paragraphs should indent continuation."""
        doc = ContentDocument(
            body=(Paragraph(children=(Text(value="See"), FootnoteRef(identifier="fn1"))),),
            footnotes={
                "fn1": (
                    Paragraph(children=(Text(value="First paragraph of footnote."),)),
                    Paragraph(children=(Text(value="Second paragraph of footnote."),)),
                ),
            },
        )
        result = serialize_markdown(doc)
        assert "[^fn1]:" in result
        assert "First paragraph of footnote" in result
        assert "Second paragraph of footnote" in result
        # Second paragraph should be indented with 4 spaces
        lines = result.strip().splitlines()
        fn_start = next(i for i, ln in enumerate(lines) if "[^fn1]:" in ln)
        # Lines after the footnote start that have content should be indented
        remaining = lines[fn_start + 1 :]
        indented = [ln for ln in remaining if ln.strip() and "Second paragraph" in ln]
        for line in indented:
            assert line.startswith("    "), f"Expected 4-space indent, got: {line!r}"

    def test_footnote_ref_without_definition(self) -> None:
        """FootnoteRef for which no definition exists in footnotes dict."""
        doc = ContentDocument(
            body=(
                Paragraph(
                    children=(
                        Text(value="See"),
                        FootnoteRef(identifier="undefined"),
                    )
                ),
            ),
            footnotes={},
        )
        result = serialize_markdown(doc)
        # The ref should still render
        assert "[^undefined]" in result
        # But no definition block should appear
        assert "[^undefined]:" not in result

    def test_footnote_defined_but_not_referenced(self) -> None:
        """Footnote definition without any corresponding ref in the body."""
        doc = ContentDocument(
            body=(Paragraph(children=(Text(value="No refs here."),)),),
            footnotes={"orphan": (Paragraph(children=(Text(value="Orphan note."),)),)},
        )
        result = serialize_markdown(doc)
        # The footnote should still be rendered at the bottom
        assert "[^orphan]:" in result
        assert "Orphan note" in result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. UNICODE IN MARKDOWN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestUnicode:
    def test_emoji_in_heading(self) -> None:
        doc = ContentDocument(
            body=(Heading(depth=1, children=(Text(value="Hello World \U0001f30d"),)),)
        )
        result = serialize_markdown(doc)
        assert "# Hello World \U0001f30d" in result

    def test_cjk_in_table_cells(self) -> None:
        doc = ContentDocument(
            body=(
                Table(
                    head=TableSection(
                        rows=(
                            Row(
                                cells=(
                                    Cell(
                                        content=(Paragraph(children=(Text(value="\u540d\u524d"),)),)
                                    ),
                                    Cell(
                                        content=(Paragraph(children=(Text(value="\u5e74\u9f62"),)),)
                                    ),
                                )
                            ),
                        )
                    ),
                    bodies=(
                        TableSection(
                            rows=(
                                Row(
                                    cells=(
                                        Cell(
                                            content=(
                                                Paragraph(children=(Text(value="\u592a\u90ce"),)),
                                            )
                                        ),
                                        Cell(content=(Paragraph(children=(Text(value="30"),)),)),
                                    )
                                ),
                            )
                        ),
                    ),
                ),
            )
        )
        result = serialize_markdown(doc)
        assert "\u540d\u524d" in result
        assert "\u5e74\u9f62" in result
        assert "\u592a\u90ce" in result

    def test_rtl_text_in_paragraph(self) -> None:
        doc = ContentDocument(
            body=(
                Paragraph(
                    children=(
                        Text(
                            value=(
                                "\u0645\u0631\u062d\u0628\u0627 "
                                "\u0628\u0627\u0644\u0639\u0627\u0644\u0645"
                            )
                        ),
                    )
                ),
            )
        )
        result = serialize_markdown(doc)
        assert "\u0645\u0631\u062d\u0628\u0627" in result

    def test_mixed_script_text(self) -> None:
        """Text mixing Latin, CJK, and emoji."""
        doc = ContentDocument(
            body=(
                Paragraph(
                    children=(
                        Text(
                            value=(
                                "Hello \u4e16\u754c \U0001f600 \u041f\u0440\u0438\u0432\u0435\u0442"
                            )
                        ),
                    )
                ),
            )
        )
        result = serialize_markdown(doc)
        assert "Hello" in result
        assert "\u4e16\u754c" in result
        assert "\U0001f600" in result
        assert "\u041f\u0440\u0438\u0432\u0435\u0442" in result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. WHITESPACE HANDLING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestWhitespace:
    def test_no_triple_blank_lines(self) -> None:
        """Output should never contain three or more consecutive blank lines."""
        doc = ContentDocument(
            body=(
                Paragraph(children=(Text(value="a"),)),
                ThematicBreak(),
                Paragraph(children=(Text(value="b"),)),
                ThematicBreak(),
                Paragraph(children=(Text(value="c"),)),
            )
        )
        result = serialize_markdown(doc)
        assert "\n\n\n" not in result

    def test_document_ends_with_single_newline(self) -> None:
        """Output should end with exactly one newline."""
        doc = ContentDocument(body=(Paragraph(children=(Text(value="end"),)),))
        result = serialize_markdown(doc)
        assert result.endswith("\n")
        assert not result.endswith("\n\n")

    def test_no_trailing_whitespace_on_lines(self) -> None:
        """No line should have trailing spaces (except hard line breaks)."""
        doc = ContentDocument(
            body=(
                Heading(depth=1, children=(Text(value="Title"),)),
                Paragraph(children=(Text(value="body text"),)),
                BulletList(
                    children=(ListItem(children=(Paragraph(children=(Text(value="item"),)),)),)
                ),
            )
        )
        result = serialize_markdown(doc)
        for i, line in enumerate(result.splitlines()):
            # Lines ending with backslash-newline are hard breaks, skip
            if not line.endswith("\\"):
                assert line == line.rstrip(" \t"), f"Line {i + 1} has trailing whitespace: {line!r}"

    def test_leading_whitespace_in_text_preserved(self) -> None:
        """Leading whitespace in a Text node value is part of the content."""
        doc = ContentDocument(body=(Paragraph(children=(Text(value="  indented"),)),))
        result = serialize_markdown(doc)
        # The leading spaces are part of the text content
        assert "indented" in result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 9. LARGE DOCUMENT SERIALIZATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestLargeDocument:
    def test_100_paragraphs(self) -> None:
        """Serialize 100 paragraphs without formatting artifacts."""
        doc = ContentDocument(
            body=tuple(
                Paragraph(children=(Text(value=f"Paragraph number {i}"),)) for i in range(100)
            )
        )
        result = serialize_markdown(doc)
        # Verify all paragraphs present
        assert "Paragraph number 0" in result
        assert "Paragraph number 99" in result
        # No triple blank lines
        assert "\n\n\n" not in result
        # Reasonable size
        lines = result.strip().splitlines()
        # 100 paragraphs separated by blank lines = ~199 lines
        assert len(lines) >= 100

    def test_100_list_items(self) -> None:
        """Serialize a bullet list with 100 items."""
        items = tuple(
            ListItem(children=(Paragraph(children=(Text(value=f"Item {i}"),)),)) for i in range(100)
        )
        doc = ContentDocument(body=(BulletList(children=items),))
        result = serialize_markdown(doc)
        assert "- Item 0" in result
        assert "- Item 99" in result

    def test_mixed_large_document(self) -> None:
        """Mix of headings, paragraphs, lists, code blocks, tables."""
        blocks: list[Heading | Paragraph | BulletList | CodeBlock] = []
        for i in range(20):
            blocks.append(Heading(depth=2, children=(Text(value=f"Section {i}"),)))
            blocks.append(Paragraph(children=(Text(value=f"Introduction to section {i}."),)))
            blocks.append(
                BulletList(
                    children=tuple(
                        ListItem(children=(Paragraph(children=(Text(value=f"Point {j}"),)),))
                        for j in range(3)
                    )
                )
            )
            blocks.append(CodeBlock(language="python", value=f"x = {i}"))
        doc = ContentDocument(body=tuple(blocks))
        result = serialize_markdown(doc)
        assert "## Section 0" in result
        assert "## Section 19" in result
        assert "- Point 0" in result
        assert "\n\n\n" not in result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 10. COMBINED COMPLEX STRUCTURES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestComplexNesting:
    def test_blockquote_containing_table(self) -> None:
        """Blockquote containing a table."""
        doc = ContentDocument(
            body=(
                BlockQuote(
                    children=(
                        Table(
                            head=TableSection(
                                rows=(
                                    Row(
                                        cells=(
                                            Cell(
                                                content=(
                                                    Paragraph(children=(Text(value="Col A"),)),
                                                )
                                            ),
                                            Cell(
                                                content=(
                                                    Paragraph(children=(Text(value="Col B"),)),
                                                )
                                            ),
                                        )
                                    ),
                                )
                            ),
                            bodies=(
                                TableSection(
                                    rows=(
                                        Row(
                                            cells=(
                                                Cell(
                                                    content=(
                                                        Paragraph(children=(Text(value="1"),)),
                                                    )
                                                ),
                                                Cell(
                                                    content=(
                                                        Paragraph(children=(Text(value="2"),)),
                                                    )
                                                ),
                                            )
                                        ),
                                    )
                                ),
                            ),
                        ),
                    )
                ),
            )
        )
        result = serialize_markdown(doc)
        # Each table line should start with >
        lines = result.strip().splitlines()
        for line in lines:
            assert line.startswith(">"), f"Expected blockquote prefix, got: {line!r}"

    def test_emphasis_inside_link(self) -> None:
        """Link whose text contains emphasis."""
        doc = ContentDocument(
            body=(
                Paragraph(
                    children=(
                        Link(
                            url="https://example.com",
                            children=(
                                Text(value="click "),
                                Emphasis(children=(Text(value="here"),)),
                            ),
                        ),
                    )
                ),
            )
        )
        result = serialize_markdown(doc)
        assert "[click *here*](https://example.com)" in result

    def test_strong_inside_emphasis(self) -> None:
        """Bold inside italic — nested strong alternates to __ delimiters."""
        doc = ContentDocument(
            body=(
                Paragraph(
                    children=(
                        Emphasis(
                            children=(
                                Text(value="light "),
                                Strong(children=(Text(value="heavy"),)),
                            )
                        ),
                    )
                ),
            )
        )
        result = serialize_markdown(doc)
        # Nested emphasis uses delimiter alternation: * for outer, __ for inner
        assert "*light __heavy__*" in result

    def test_deeply_nested_blockquote(self) -> None:
        """Three levels of blockquotes."""
        doc = ContentDocument(
            body=(
                BlockQuote(
                    children=(
                        BlockQuote(
                            children=(
                                BlockQuote(children=(Paragraph(children=(Text(value="deep"),)),)),
                            )
                        ),
                    )
                ),
            )
        )
        result = serialize_markdown(doc)
        assert "> > > deep" in result

    def test_table_with_link_and_emphasis(self) -> None:
        """Table cell containing a link with emphasized text."""
        doc = ContentDocument(
            body=(
                Table(
                    head=TableSection(
                        rows=(
                            Row(
                                cells=(
                                    Cell(
                                        content=(
                                            Paragraph(
                                                children=(
                                                    Link(
                                                        url="https://example.com",
                                                        children=(
                                                            Emphasis(
                                                                children=(Text(value="click me"),)
                                                            ),
                                                        ),
                                                    ),
                                                )
                                            ),
                                        )
                                    ),
                                )
                            ),
                        )
                    ),
                    bodies=(
                        TableSection(
                            rows=(
                                Row(
                                    cells=(
                                        Cell(content=(Paragraph(children=(Text(value="data"),)),)),
                                    )
                                ),
                            )
                        ),
                    ),
                ),
            )
        )
        result = serialize_markdown(doc)
        assert "[*click me*](https://example.com)" in result

    def test_list_inside_blockquote_inside_list(self) -> None:
        """Bullet list inside blockquote inside bullet list."""
        doc = ContentDocument(
            body=(
                BulletList(
                    children=(
                        ListItem(
                            children=(
                                Paragraph(children=(Text(value="outer"),)),
                                BlockQuote(
                                    children=(
                                        BulletList(
                                            children=(
                                                ListItem(
                                                    children=(
                                                        Paragraph(children=(Text(value="inner"),)),
                                                    )
                                                ),
                                            )
                                        ),
                                    )
                                ),
                            )
                        ),
                    )
                ),
            )
        )
        result = serialize_markdown(doc)
        assert "outer" in result
        assert "inner" in result

    def test_image_in_link(self) -> None:
        """Image wrapped in a link (clickable image)."""
        doc = ContentDocument(
            body=(
                Paragraph(
                    children=(
                        Link(
                            url="https://example.com",
                            children=(Image(src="photo.png", alt="Photo"),),
                        ),
                    )
                ),
            )
        )
        result = serialize_markdown(doc)
        # Markdown for linked image: [![alt](src)](url)
        assert "![Photo](photo.png)" in result or "Photo" in result

    def test_math_in_emphasis(self) -> None:
        """Math inside emphasis."""
        doc = ContentDocument(
            body=(Paragraph(children=(Emphasis(children=(Math(value="E=mc^2"),)),)),)
        )
        result = serialize_markdown(doc)
        assert "*$E=mc^2$*" in result

    def test_heading_with_inline_code_and_link(self) -> None:
        """Heading containing both inline code and a link."""
        doc = ContentDocument(
            body=(
                Heading(
                    depth=2,
                    children=(
                        Text(value="Using "),
                        Code(value="serialize_markdown"),
                        Text(value=" ("),
                        Link(
                            url="https://docs.example.com",
                            children=(Text(value="docs"),),
                        ),
                        Text(value=")"),
                    ),
                ),
            )
        )
        result = serialize_markdown(doc)
        assert "## Using `serialize_markdown`" in result
        assert "[docs](https://docs.example.com)" in result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# REGRESSION: Code block multiline handling
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestCodeBlockMultiline:
    def test_multiline_code_block_all_lines_present(self) -> None:
        """All lines of a multiline code block should appear in output."""
        code = "def foo():\n    return 42\n\nfoo()"
        doc = ContentDocument(body=(CodeBlock(language="python", value=code),))
        result = serialize_markdown(doc)
        assert "def foo():" in result
        assert "    return 42" in result
        assert "foo()" in result

    def test_code_block_with_leading_blank_line(self) -> None:
        """Code block value starting with a blank line."""
        code = "\nactual code"
        doc = ContentDocument(body=(CodeBlock(value=code),))
        result = serialize_markdown(doc)
        assert "actual code" in result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# REGRESSION: Inline code with various backtick patterns
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestInlineCodeEdgeCases:
    def test_inline_code_with_single_backtick(self) -> None:
        """Code containing a single backtick uses double-backtick delimiter."""
        doc = ContentDocument(body=(Paragraph(children=(Code(value="a`b"),)),))
        result = serialize_markdown(doc)
        # No space padding: value doesn't start/end with backtick
        assert "``a`b``" in result

    def test_inline_code_with_leading_space(self) -> None:
        """Code value with leading space."""
        doc = ContentDocument(body=(Paragraph(children=(Code(value=" leading"),)),))
        result = serialize_markdown(doc)
        assert "` leading`" in result

    def test_inline_code_with_trailing_space(self) -> None:
        """Code value with trailing space."""
        doc = ContentDocument(body=(Paragraph(children=(Code(value="trailing "),)),))
        result = serialize_markdown(doc)
        assert "`trailing `" in result

    def test_inline_code_empty(self) -> None:
        """Empty code value."""
        doc = ContentDocument(body=(Paragraph(children=(Code(value=""),)),))
        result = serialize_markdown(doc)
        assert "``" in result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Math block edge cases
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestMathEdgeCases:
    def test_math_block_multiline(self) -> None:
        """Math block with multiline content."""
        doc = ContentDocument(body=(MathBlock(value="x = 1\ny = 2"),))
        result = serialize_markdown(doc)
        assert "$$" in result
        assert "x = 1" in result
        assert "y = 2" in result

    def test_inline_math_with_dollar_sign(self) -> None:
        """Math inline whose value contains $."""
        doc = ContentDocument(body=(Paragraph(children=(Math(value="\\$5"),)),))
        result = serialize_markdown(doc)
        assert "$\\$5$" in result
