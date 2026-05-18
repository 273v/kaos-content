"""Tests for ``numbering_label`` emission across all serializers.

Confirms that a DOCX-sourced rendered label (``"11."``, ``"(a)"``,
``"Section 11."``, ``"11(a)(i)"``) survives serialization to text,
markdown, and HTML. Position-based marker fallback continues to apply
when ``numbering_label is None``.
"""

from __future__ import annotations

from kaos_content import (
    BulletList,
    ContentDocument,
    DocumentBuilder,
    Heading,
    ListItem,
    OrderedList,
    Paragraph,
    Text,
    serialize_html,
    serialize_markdown,
    serialize_text,
)


def _doc(*blocks: object) -> ContentDocument:
    builder = DocumentBuilder()
    for block in blocks:
        builder.add_block(block)  # ty: ignore[invalid-argument-type]
    return builder.build()


class TestTextSerializer:
    def test_paragraph_label_prepended(self) -> None:
        para = Paragraph(
            children=(Text(value="GOVERNING LAW. The validity..."),),
            numbering_label="11.",
        )
        text = serialize_text(_doc(para))
        assert text.startswith("11. GOVERNING LAW. The validity...")

    def test_paragraph_without_label_unchanged(self) -> None:
        para = Paragraph(children=(Text(value="A sentence."),))
        text = serialize_text(_doc(para))
        assert text.strip() == "A sentence."

    def test_heading_label_prepended(self) -> None:
        heading = Heading(
            depth=2,
            children=(Text(value="GOVERNING LAW"),),
            numbering_label="Section 11.",
        )
        text = serialize_text(_doc(heading))
        assert "Section 11. GOVERNING LAW" in text

    def test_list_item_label_overrides_position(self) -> None:
        ol = OrderedList(
            start=1,
            children=(
                ListItem(
                    children=(Paragraph(children=(Text(value="Clause one."),)),),
                    numbering_label="(a)",
                ),
                ListItem(
                    children=(Paragraph(children=(Text(value="Clause two."),)),),
                    numbering_label="(b)",
                ),
            ),
        )
        text = serialize_text(_doc(ol))
        assert "(a) Clause one." in text
        assert "(b) Clause two." in text
        # Position-based "1." marker must NOT appear when labels are set.
        assert "1. Clause one." not in text

    def test_list_item_without_label_uses_position(self) -> None:
        ol = OrderedList(
            start=7,
            children=(
                ListItem(children=(Paragraph(children=(Text(value="x"),)),)),
                ListItem(children=(Paragraph(children=(Text(value="y"),)),)),
            ),
        )
        text = serialize_text(_doc(ol))
        assert "7. x" in text
        assert "8. y" in text


class TestMarkdownSerializer:
    def test_paragraph_label_prepended(self) -> None:
        para = Paragraph(
            children=(Text(value="GOVERNING LAW."),),
            numbering_label="11.",
        )
        md = serialize_markdown(_doc(para))
        assert md.startswith("11. GOVERNING LAW.")

    def test_heading_label_after_hash(self) -> None:
        heading = Heading(
            depth=2,
            children=(Text(value="GOVERNING LAW"),),
            numbering_label="Section 11.",
        )
        md = serialize_markdown(_doc(heading))
        # Heading should render as "## Section 11. GOVERNING LAW"
        assert "## Section 11. GOVERNING LAW" in md

    def test_ordered_list_label_replaces_marker(self) -> None:
        ol = OrderedList(
            start=1,
            children=(
                ListItem(
                    children=(Paragraph(children=(Text(value="Clause one."),)),),
                    numbering_label="(a)",
                ),
                ListItem(
                    children=(Paragraph(children=(Text(value="Clause two."),)),),
                    numbering_label="(b)",
                ),
            ),
        )
        md = serialize_markdown(_doc(ol))
        assert "(a) Clause one." in md
        assert "(b) Clause two." in md
        assert "1. Clause one." not in md

    def test_ordered_list_without_label_uses_position(self) -> None:
        ol = OrderedList(
            start=1,
            children=(
                ListItem(children=(Paragraph(children=(Text(value="x"),)),)),
                ListItem(children=(Paragraph(children=(Text(value="y"),)),)),
            ),
        )
        md = serialize_markdown(_doc(ol))
        assert "1. x" in md
        assert "2. y" in md

    def test_bullet_list_label_overrides_dash(self) -> None:
        # Some Word numbering formats land as BulletList in the AST
        # (when numFmt is "bullet" with a Word-defined character in
        # lvlText). The reader still resolves a label; the serializer
        # honors it.
        bl = BulletList(
            children=(
                ListItem(
                    children=(Paragraph(children=(Text(value="x"),)),),
                    numbering_label="◆",
                ),
            )
        )
        md = serialize_markdown(_doc(bl))
        assert "◆ x" in md

    def test_section_eleven_a_i(self) -> None:
        """The textbook attorney citation token survives end-to-end."""
        ol = OrderedList(
            children=(
                ListItem(
                    children=(Paragraph(children=(Text(value="GOVERNING LAW."),)),),
                    numbering_label="11(a)(i)",
                ),
            )
        )
        md = serialize_markdown(_doc(ol))
        assert "11(a)(i) GOVERNING LAW." in md


class TestHtmlSerializer:
    def test_paragraph_label_emits_data_attr_and_inline(self) -> None:
        para = Paragraph(
            children=(Text(value="GOVERNING LAW."),),
            numbering_label="11.",
        )
        html = serialize_html(_doc(para))
        assert 'data-numbering-label="11."' in html
        # Inline text matters: default CSS does not render the
        # data-attribute, so the visible text must include the label.
        assert ">11. GOVERNING LAW." in html

    def test_heading_label_emits_data_attr_and_inline(self) -> None:
        heading = Heading(
            depth=2,
            children=(Text(value="GOVERNING LAW"),),
            numbering_label="Section 11.",
        )
        html = serialize_html(_doc(heading))
        assert 'data-numbering-label="Section 11."' in html
        assert ">Section 11. GOVERNING LAW</h2>" in html

    def test_list_item_label_emits_data_attr_and_inline(self) -> None:
        ol = OrderedList(
            children=(
                ListItem(
                    children=(Paragraph(children=(Text(value="Clause one."),)),),
                    numbering_label="(a)",
                ),
            )
        )
        html = serialize_html(_doc(ol))
        assert 'data-numbering-label="(a)"' in html
        # Inline visible label preserves attorney citation in the
        # default CSS rendering.
        assert "(a) " in html

    def test_label_escapes_html(self) -> None:
        """Numbering labels must not be a XSS injection vector."""
        para = Paragraph(
            children=(Text(value="body"),),
            numbering_label='<script>alert("x")</script>',
        )
        html = serialize_html(_doc(para))
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_no_label_emits_no_data_attr(self) -> None:
        para = Paragraph(children=(Text(value="plain"),))
        html = serialize_html(_doc(para))
        assert "data-numbering-label" not in html
