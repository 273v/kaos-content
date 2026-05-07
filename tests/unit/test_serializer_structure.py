"""Structural correctness tests for the markdown serializer.

Focus: complex nested structures produce well-formed markdown.
NOT about escaping (that is covered in test_phase2_hardening.py).
"""

from kaos_content import (
    Admonition,
    BlockQuote,
    BulletList,
    Caption,
    Cell,
    CodeBlock,
    ContentDocument,
    Figure,
    Heading,
    Image,
    ListItem,
    OrderedList,
    Paragraph,
    Row,
    Strong,
    Table,
    TableSection,
    Text,
    serialize_markdown,
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. TABLE CELL BLOCK CONTENT FLATTENING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _simple_table(cell_content: tuple) -> ContentDocument:  # type: ignore[type-arg]
    """Helper: one-column table with a header and one body cell containing *cell_content*."""
    return ContentDocument(
        body=(
            Table(
                head=TableSection(
                    rows=(
                        Row(cells=(Cell(content=(Paragraph(children=(Text(value="Header"),)),)),)),
                    )
                ),
                bodies=(TableSection(rows=(Row(cells=(Cell(content=cell_content),)),)),),
            ),
        )
    )


def _assert_valid_gfm_table(md: str) -> list[str]:
    """Assert every pipe-containing line is a valid single-line GFM row. Returns those lines."""
    table_lines = [line for line in md.strip().splitlines() if "|" in line]
    assert len(table_lines) >= 2, f"Expected at least header + separator, got:\n{md}"
    for tl in table_lines:
        assert "\n" not in tl, f"Table row must be single line: {tl!r}"
        # Each row starts and ends with pipe
        stripped = tl.strip()
        assert stripped.startswith("|"), f"Row must start with pipe: {stripped!r}"
        assert stripped.endswith("|"), f"Row must end with pipe: {stripped!r}"
    return table_lines


class TestTableCellFlattening:
    """Block types inside a Cell must be flattened to valid single-line cell content."""

    def test_cell_heading_becomes_bold(self) -> None:
        """A Heading inside a Cell should render as bold text."""
        doc = _simple_table((Heading(depth=2, children=(Text(value="Title"),)),))
        md = serialize_markdown(doc)
        lines = _assert_valid_gfm_table(md)
        body_row = lines[-1]  # last pipe-row is the body row
        assert "**Title**" in body_row

    def test_cell_codeblock_becomes_inline_code(self) -> None:
        """A CodeBlock inside a Cell should render as inline code."""
        doc = _simple_table((CodeBlock(language="py", value="x = 1"),))
        md = serialize_markdown(doc)
        lines = _assert_valid_gfm_table(md)
        body_row = lines[-1]
        assert "`x = 1`" in body_row

    def test_cell_bullet_list_becomes_comma_separated(self) -> None:
        """A BulletList inside a Cell should render as comma-separated items."""
        doc = _simple_table(
            (
                BulletList(
                    children=(
                        ListItem(children=(Paragraph(children=(Text(value="alpha"),)),)),
                        ListItem(children=(Paragraph(children=(Text(value="beta"),)),)),
                        ListItem(children=(Paragraph(children=(Text(value="gamma"),)),)),
                    )
                ),
            )
        )
        md = serialize_markdown(doc)
        lines = _assert_valid_gfm_table(md)
        body_row = lines[-1]
        assert "alpha" in body_row
        assert "beta" in body_row
        assert "gamma" in body_row
        # Items should be comma-separated
        assert "alpha, beta, gamma" in body_row

    def test_cell_multiple_paragraphs_joined_with_br(self) -> None:
        """Multiple paragraphs inside a Cell should be joined with <br>."""
        doc = _simple_table(
            (
                Paragraph(children=(Text(value="first"),)),
                Paragraph(children=(Text(value="second"),)),
            )
        )
        md = serialize_markdown(doc)
        lines = _assert_valid_gfm_table(md)
        body_row = lines[-1]
        assert "first" in body_row
        assert "second" in body_row
        assert "<br>" in body_row

    def test_cell_blockquote_becomes_plain_text(self) -> None:
        """A BlockQuote inside a Cell should render as plain text."""
        doc = _simple_table((BlockQuote(children=(Paragraph(children=(Text(value="quoted"),)),)),))
        md = serialize_markdown(doc)
        lines = _assert_valid_gfm_table(md)
        body_row = lines[-1]
        assert "quoted" in body_row
        # Must NOT contain blockquote markdown syntax inside the cell
        cell_content = body_row.split("|")[1]  # content between first two pipes
        assert ">" not in cell_content or "\\>" in cell_content

    def test_cell_nested_table_becomes_plain_text(self) -> None:
        """A nested Table inside a Cell should NOT produce pipe-table syntax."""
        inner_table = Table(
            head=TableSection(
                rows=(
                    Row(
                        cells=(Cell(content=(Paragraph(children=(Text(value="inner header"),)),)),)
                    ),
                )
            ),
            bodies=(
                TableSection(
                    rows=(
                        Row(
                            cells=(
                                Cell(content=(Paragraph(children=(Text(value="inner data"),)),)),
                            )
                        ),
                    )
                ),
            ),
        )
        doc = _simple_table((inner_table,))
        md = serialize_markdown(doc)
        lines = _assert_valid_gfm_table(md)
        # The body row should contain the text but not nested pipe-table structure
        body_row = lines[-1]
        assert "inner" in body_row
        # Should be valid single-line: no literal nested pipes creating extra columns
        # Count pipe delimiters — header has 2 (|Header|), body should have same count
        header_pipes = lines[0].count("|")
        body_pipes = body_row.count("|")
        assert body_pipes == header_pipes, (
            f"Nested table created extra pipes: header={header_pipes}, body={body_pipes}"
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. ORDERED LIST MARKER WIDTH CONSISTENCY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestOrderedListMarkerWidth:
    def test_ten_items_consistent_indent(self) -> None:
        """OrderedList with 10 items: all markers should have same width (padded to widest)."""
        items = tuple(
            ListItem(children=(Paragraph(children=(Text(value=f"item {i}"),)),))
            for i in range(1, 11)
        )
        doc = ContentDocument(body=(OrderedList(start=1, children=items),))
        md = serialize_markdown(doc)
        lines = [line for line in md.strip().splitlines() if line.strip()]

        # Verify markers are present
        assert "1. " in md
        assert "10. " in md

        # All continuation lines should align. The widest marker is "10. " = 4 chars.
        # So single-digit markers should be padded: "1.  " (4 chars).
        for line in lines:
            if line[0].isdigit():
                # Extract the marker: everything up to and including ". "
                dot_pos = line.index(".")
                marker_end = dot_pos + 1
                # After the dot there should be spaces; total marker width should be consistent
                rest = line[marker_end:]
                space_count = len(rest) - len(rest.lstrip(" "))
                marker_width = marker_end + space_count
                assert marker_width == 4, (
                    f"Marker width should be 4 (like '10. '), got {marker_width} for: {line!r}"
                )

    def test_start_99_continuation_consistent(self) -> None:
        """OrderedList(start=99, items=[..., ...]) — markers '99. ' and '100. ' should align."""
        items = (
            ListItem(children=(Paragraph(children=(Text(value="a"),)),)),
            ListItem(children=(Paragraph(children=(Text(value="b"),)),)),
        )
        doc = ContentDocument(body=(OrderedList(start=99, children=items),))
        md = serialize_markdown(doc)

        # The widest marker is "100. " = 5 chars
        assert "99." in md
        assert "100." in md

        lines = [line for line in md.strip().splitlines() if line.strip()]
        marker_widths = []
        for line in lines:
            if line[0].isdigit():
                dot_pos = line.index(".")
                marker_end = dot_pos + 1
                rest = line[marker_end:]
                space_count = len(rest) - len(rest.lstrip(" "))
                marker_widths.append(marker_end + space_count)

        assert len(marker_widths) == 2
        # Both should have width 5 (the width of "100. ")
        assert marker_widths[0] == 5, f"99 marker width should be 5, got {marker_widths[0]}"
        assert marker_widths[1] == 5, f"100 marker width should be 5, got {marker_widths[1]}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. DEEPLY NESTED LISTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestDeeplyNestedLists:
    def test_three_level_bullet_list_indentation(self) -> None:
        """3-level bullet list: each level increases indentation by marker width (2 for '- ')."""
        doc = ContentDocument(
            body=(
                BulletList(
                    children=(
                        ListItem(
                            children=(
                                Paragraph(children=(Text(value="L1"),)),
                                BulletList(
                                    children=(
                                        ListItem(
                                            children=(
                                                Paragraph(children=(Text(value="L2"),)),
                                                BulletList(
                                                    children=(
                                                        ListItem(
                                                            children=(
                                                                Paragraph(
                                                                    children=(Text(value="L3"),)
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
        md = serialize_markdown(doc)
        lines = md.strip().splitlines()

        l1 = next(ln for ln in lines if "L1" in ln)
        l2 = next(ln for ln in lines if "L2" in ln)
        l3 = next(ln for ln in lines if "L3" in ln)

        indent1 = len(l1) - len(l1.lstrip())
        indent2 = len(l2) - len(l2.lstrip())
        indent3 = len(l3) - len(l3.lstrip())

        # Level 1: 0 indent, Level 2: 2 indent, Level 3: 4 indent
        assert indent1 == 0, f"L1 indent should be 0, got {indent1}"
        assert indent2 == 2, f"L2 indent should be 2, got {indent2}"
        assert indent3 == 4, f"L3 indent should be 4, got {indent3}"

        # Each level should have "- " marker
        assert l1.lstrip().startswith("- L1")
        assert l2.lstrip().startswith("- L2")
        assert l3.lstrip().startswith("- L3")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. BLOCKQUOTE CONTAINING A TABLE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestBlockquoteContainingTable:
    def test_all_table_rows_prefixed(self) -> None:
        """Table inside a blockquote: every row and the separator get '> ' prefix."""
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
                                                content=(Paragraph(children=(Text(value="Col1"),)),)
                                            ),
                                            Cell(
                                                content=(Paragraph(children=(Text(value="Col2"),)),)
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
                                                        Paragraph(children=(Text(value="A"),)),
                                                    )
                                                ),
                                                Cell(
                                                    content=(
                                                        Paragraph(children=(Text(value="B"),)),
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
        md = serialize_markdown(doc)
        lines = md.strip().splitlines()

        # Every line in a blockquoted table must start with "> "
        for line in lines:
            assert line.startswith("> ") or line.strip() == ">", (
                f"Line not properly blockquoted: {line!r}"
            )

        # The separator row (containing ---) must also be prefixed
        sep_lines = [ln for ln in lines if "---" in ln]
        assert len(sep_lines) >= 1, "Expected a separator row"
        for sl in sep_lines:
            assert sl.startswith("> "), f"Separator not blockquoted: {sl!r}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. BLOCKQUOTE CONTAINING A CODE BLOCK
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestBlockquoteContainingCodeBlock:
    def test_code_fence_and_content_prefixed(self) -> None:
        """Code block inside blockquote: fences and content lines all get '> ' prefix."""
        doc = ContentDocument(
            body=(BlockQuote(children=(CodeBlock(language="python", value="x = 1\ny = 2"),)),)
        )
        md = serialize_markdown(doc)
        lines = md.strip().splitlines()

        # Every line must start with "> "
        for line in lines:
            assert line.startswith("> ") or line.strip() == ">", (
                f"Line not properly blockquoted: {line!r}"
            )

        # Should contain the code fence
        fence_lines = [ln for ln in lines if "```" in ln]
        assert len(fence_lines) >= 2, f"Expected opening and closing fence, got {fence_lines}"

        # The first fence should have the language
        assert any("```python" in ln for ln in fence_lines), "Opening fence should have language"

        # The code content should be there
        assert any("x = 1" in ln for ln in lines)
        assert any("y = 2" in ln for ln in lines)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. EMPTY TABLE (NO ROWS)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestEmptyTable:
    def test_no_rows_produces_empty(self) -> None:
        """Table with no head, empty bodies, no foot → empty string (no malformed header)."""
        doc = ContentDocument(body=(Table(bodies=(TableSection(rows=()),)),))
        md = serialize_markdown(doc)
        # Should not produce a broken separator row like "| --- |" with no header
        assert "---" not in md or md.strip() == ""

    def test_empty_head_empty_body(self) -> None:
        """Table with empty head rows and empty body rows."""
        doc = ContentDocument(
            body=(Table(head=TableSection(rows=()), bodies=(TableSection(rows=()),)),)
        )
        md = serialize_markdown(doc)
        assert "---" not in md or md.strip() == ""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. TABLE WITH FOOT BUT NO HEAD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestTableFootNoHead:
    def test_foot_only_synthesizes_empty_header(self) -> None:
        """Table with foot but no head should synthesize an empty header row."""
        doc = ContentDocument(
            body=(
                Table(
                    foot=TableSection(
                        rows=(
                            Row(
                                cells=(
                                    Cell(content=(Paragraph(children=(Text(value="Total"),)),)),
                                    Cell(content=(Paragraph(children=(Text(value="42"),)),)),
                                )
                            ),
                        )
                    )
                ),
            )
        )
        md = serialize_markdown(doc)
        lines = md.strip().splitlines()

        # Should have: empty header row, separator, foot row
        assert len(lines) >= 3, f"Expected 3+ lines, got:\n{md}"

        # First line is the synthesized empty header
        assert lines[0].strip().startswith("|"), f"First line should be header: {lines[0]!r}"

        # Second line is the separator
        assert "---" in lines[1], f"Second line should be separator: {lines[1]!r}"

        # Data should be present in foot row
        assert "Total" in md
        assert "42" in md


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. ADMONITION WITH MULTIPLE BLOCKS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestAdmonitionMultiBlock:
    def test_all_lines_prefixed(self) -> None:
        """Admonition with multiple blocks: all content lines get '> ', blank lines get '>'."""
        doc = ContentDocument(
            body=(
                Admonition(
                    kind="note",
                    children=(
                        Paragraph(children=(Text(value="First paragraph."),)),
                        Paragraph(children=(Text(value="Second paragraph."),)),
                        CodeBlock(language="python", value="x = 1"),
                    ),
                ),
            )
        )
        md = serialize_markdown(doc)
        lines = md.strip().splitlines()

        # First line is the admonition header
        assert lines[0].strip() == "> [!NOTE]"

        # Every subsequent line must start with "> " (content) or be bare ">" (blank separator)
        for line in lines[1:]:
            assert line.startswith("> ") or line.strip() == ">", (
                f"Admonition line not properly prefixed: {line!r}"
            )

        # Content should all be present
        assert any("First paragraph." in ln for ln in lines)
        assert any("Second paragraph." in ln for ln in lines)
        assert any("x = 1" in ln for ln in lines)

    def test_blank_lines_between_blocks_have_bare_gt(self) -> None:
        """Blank lines between blocks inside an admonition should be '>' (no trailing space)."""
        doc = ContentDocument(
            body=(
                Admonition(
                    kind="tip",
                    children=(
                        Paragraph(children=(Text(value="Para one."),)),
                        Paragraph(children=(Text(value="Para two."),)),
                    ),
                ),
            )
        )
        md = serialize_markdown(doc)
        lines = md.strip().splitlines()

        # Find blank separator lines (between the two paragraphs)
        blank_lines = [ln for ln in lines if ln.strip() == ">"]
        assert len(blank_lines) >= 1, (
            f"Expected at least one bare '>' blank line between paragraphs, got:\n{md}"
        )
        # Bare ">" should NOT have a trailing space
        for bl in blank_lines:
            assert bl.rstrip() == bl.rstrip(" ").rstrip() or bl.strip() == ">", (
                f"Blank line has trailing content: {bl!r}"
            )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 9. CODE BLOCK WITH EMPTY LANGUAGE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestCodeBlockEmptyLanguage:
    def test_empty_string_language(self) -> None:
        """CodeBlock(language='') should produce ``` not ```<empty>."""
        doc = ContentDocument(body=(CodeBlock(language="", value="hello"),))
        md = serialize_markdown(doc)
        lines = md.strip().splitlines()
        # The opening fence should be exactly "```" with nothing after it
        assert lines[0] == "```", f"Opening fence should be bare '```', got: {lines[0]!r}"

    def test_none_language(self) -> None:
        """CodeBlock(language=None) should produce bare ```."""
        doc = ContentDocument(body=(CodeBlock(language=None, value="hello"),))
        md = serialize_markdown(doc)
        lines = md.strip().splitlines()
        assert lines[0] == "```", f"Opening fence should be bare '```', got: {lines[0]!r}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 10. FIGURE WITH IMAGE + CAPTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestFigureWithCaption:
    def test_image_and_caption_rendered(self) -> None:
        """Figure with Image child and caption → image syntax + italic caption below."""
        doc = ContentDocument(
            body=(
                Figure(
                    children=(Paragraph(children=(Image(src="photo.jpg", alt="A photo"),)),),
                    caption=Caption(body=(Paragraph(children=(Text(value="Figure 1: A photo"),)),)),
                ),
            )
        )
        md = serialize_markdown(doc)

        # Image should be rendered
        assert "![A photo](photo.jpg)" in md

        # Caption should be italic
        assert "*Figure 1: A photo*" in md

        # Image should come before caption
        img_pos = md.index("![A photo]")
        cap_pos = md.index("*Figure 1:")
        assert img_pos < cap_pos, "Image should appear before caption"

    def test_figure_without_caption(self) -> None:
        """Figure with only an image child, no caption."""
        doc = ContentDocument(
            body=(
                Figure(
                    children=(Paragraph(children=(Image(src="img.png", alt="Alt"),)),),
                ),
            )
        )
        md = serialize_markdown(doc)
        assert "![Alt](img.png)" in md
        # No italic caption
        assert "*" not in md or md.count("*") == 0

    def test_figure_with_strong_caption(self) -> None:
        """Figure caption containing inline formatting."""
        doc = ContentDocument(
            body=(
                Figure(
                    children=(Paragraph(children=(Image(src="img.png", alt="Alt"),)),),
                    caption=Caption(
                        body=(
                            Paragraph(
                                children=(
                                    Text(value="Figure "),
                                    Strong(children=(Text(value="1"),)),
                                )
                            ),
                        )
                    ),
                ),
            )
        )
        md = serialize_markdown(doc)
        assert "![Alt](img.png)" in md
        # Caption is wrapped in italics
        assert "*Figure **1***" in md or "*Figure" in md
