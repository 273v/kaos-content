"""Tests for kaos_content.shortcuts (inline / block construction helpers)."""

from __future__ import annotations

from kaos_content.model.blocks import (
    BulletList,
    CodeBlock,
    Heading,
    ListItem,
    OrderedList,
    Paragraph,
    Table,
)
from kaos_content.model.inlines import (
    Code,
    Emphasis,
    LineBreak,
    Link,
    SoftBreak,
    Strikethrough,
    Strong,
    Subscript,
    Superscript,
    Text,
    Underline,
)
from kaos_content.shortcuts import (
    bold,
    bullet_list,
    code,
    code_block,
    heading,
    italic,
    linebreak,
    link,
    ordered_list,
    paragraph,
    softbreak,
    strike,
    sub,
    sup,
    table_from_rows,
    text,
)
from kaos_content.shortcuts import (
    underline as underline_,
)

# ---------------------------------------------------------------------------
# Inline constructors
# ---------------------------------------------------------------------------


class TestInlineConstructors:
    def test_text(self) -> None:
        t = text("hello")
        assert isinstance(t, Text)
        assert t.value == "hello"

    def test_bold_from_string(self) -> None:
        s = bold("word")
        assert isinstance(s, Strong)
        assert len(s.children) == 1
        assert isinstance(s.children[0], Text)
        assert s.children[0].value == "word"

    def test_bold_multiple(self) -> None:
        s = bold("hello ", italic("world"))
        assert isinstance(s, Strong)
        assert len(s.children) == 2
        assert isinstance(s.children[0], Text)
        assert isinstance(s.children[1], Emphasis)

    def test_italic(self) -> None:
        e = italic("x")
        assert isinstance(e, Emphasis)
        child = e.children[0]
        assert isinstance(child, Text)
        assert child.value == "x"

    def test_strike(self) -> None:
        s = strike("gone")
        assert isinstance(s, Strikethrough)

    def test_underline(self) -> None:
        u = underline_("text")
        assert isinstance(u, Underline)

    def test_code(self) -> None:
        c = code("x = 1")
        assert isinstance(c, Code)
        assert c.value == "x = 1"

    def test_link_basic(self) -> None:
        lk = link("https://example.com", "click here")
        assert isinstance(lk, Link)
        assert lk.url == "https://example.com"
        assert len(lk.children) == 1
        child = lk.children[0]
        assert isinstance(child, Text)
        assert child.value == "click here"

    def test_link_with_title_and_mixed_content(self) -> None:
        lk = link("https://a.com", "click ", bold("here"), title="hint")
        assert lk.title == "hint"
        assert len(lk.children) == 2
        assert isinstance(lk.children[1], Strong)

    def test_sup_sub(self) -> None:
        assert isinstance(sup("2"), Superscript)
        assert isinstance(sub("2"), Subscript)

    def test_breaks(self) -> None:
        assert isinstance(linebreak(), LineBreak)
        assert isinstance(softbreak(), SoftBreak)

    def test_nested_composition(self) -> None:
        """bold(italic("x")) → Strong(Emphasis(Text))"""
        n = bold(italic("x"))
        assert isinstance(n, Strong)
        assert isinstance(n.children[0], Emphasis)
        child = n.children[0].children[0]
        assert isinstance(child, Text)
        assert child.value == "x"


# ---------------------------------------------------------------------------
# Block constructors
# ---------------------------------------------------------------------------


class TestBlockConstructors:
    def test_paragraph_from_strings(self) -> None:
        p = paragraph("Visit ", bold("site"), ".")
        assert isinstance(p, Paragraph)
        assert len(p.children) == 3
        assert isinstance(p.children[0], Text)
        assert isinstance(p.children[1], Strong)
        assert isinstance(p.children[2], Text)

    def test_heading(self) -> None:
        h = heading(2, "Subtitle")
        assert isinstance(h, Heading)
        assert h.depth == 2
        child = h.children[0]
        assert isinstance(child, Text)
        assert child.value == "Subtitle"

    def test_code_block(self) -> None:
        cb = code_block("print('hi')", language="python")
        assert isinstance(cb, CodeBlock)
        assert cb.language == "python"

    def test_bullet_list_from_strings(self) -> None:
        bl = bullet_list("first", "second", "third")
        assert isinstance(bl, BulletList)
        assert len(bl.children) == 3
        for item in bl.children:
            assert isinstance(item, ListItem)
            # each item wraps a Paragraph
            assert isinstance(item.children[0], Paragraph)

    def test_bullet_list_mixed(self) -> None:
        """BulletList accepts ListItem, string, or inline."""
        bl = bullet_list(
            "plain",
            bold("bolded"),
            ListItem(children=(paragraph("prebuilt"),)),
        )
        assert len(bl.children) == 3

    def test_ordered_list_with_start(self) -> None:
        ol = ordered_list("a", "b", start=5)
        assert isinstance(ol, OrderedList)
        assert ol.start == 5

    def test_table_from_rows(self) -> None:
        tbl = table_from_rows(
            headers=["Name", "Age"],
            rows=[["Alice", "30"], ["Bob", "25"]],
        )
        assert isinstance(tbl, Table)
        assert tbl.head is not None
        assert len(tbl.head.rows) == 1
        assert len(tbl.bodies[0].rows) == 2

    def test_table_without_header(self) -> None:
        tbl = table_from_rows(headers=None, rows=[["a", "b"]])
        assert tbl.head is None
        assert len(tbl.bodies[0].rows) == 1

    def test_table_with_inline_cells(self) -> None:
        tbl = table_from_rows(
            headers=None,
            rows=[[bold("X"), italic("Y")]],
        )
        row = tbl.bodies[0].rows[0]
        cell0 = row.cells[0]
        p = cell0.content[0]
        assert isinstance(p, Paragraph)
        assert isinstance(p.children[0], Strong)


# ---------------------------------------------------------------------------
# Real-world example: how much shorter is it?
# ---------------------------------------------------------------------------


class TestBoilerplateReduction:
    def test_equivalent_output_to_verbose_form(self) -> None:
        """Confirm shortcuts produce AST equivalent to the verbose construction."""
        # Verbose
        verbose = Paragraph(
            children=(
                Text(value="Visit "),
                Strong(children=(Text(value="our site"),)),
                Text(value="."),
            )
        )
        # Shortcuts
        terse = paragraph("Visit ", bold("our site"), ".")

        # IDs differ (UUIDs), but the text-extracted content and structure
        # should match:
        verbose_text = verbose.children[0]
        terse_text = terse.children[0]
        assert isinstance(verbose_text, Text)
        assert isinstance(terse_text, Text)
        assert verbose_text.value == terse_text.value
        verbose_inline = verbose.children[1]
        terse_inline = terse.children[1]
        assert isinstance(verbose_inline, Strong)
        assert isinstance(terse_inline, Strong)
        verbose_nested = verbose_inline.children[0]
        terse_nested = terse_inline.children[0]
        assert isinstance(verbose_nested, Text)
        assert isinstance(terse_nested, Text)
        assert verbose_nested.value == terse_nested.value
        assert type(verbose.children[1]) is type(terse.children[1])
