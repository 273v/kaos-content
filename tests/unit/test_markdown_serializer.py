"""Tests for markdown serializer."""

from kaos_content import (
    Admonition,
    Alignment,
    Annotation,
    AnnotationTarget,
    AnnotationType,
    Attr,
    BlockQuote,
    BulletList,
    Caption,
    Cell,
    Citation,
    Code,
    CodeBlock,
    ColSpec,
    ContentDocument,
    DefinitionItem,
    DefinitionList,
    Div,
    DocumentMetadata,
    Emphasis,
    Figure,
    FootnoteRef,
    Heading,
    Image,
    LineBreak,
    Link,
    ListItem,
    Math,
    MathBlock,
    OrderedList,
    PageBreak,
    Paragraph,
    Provenance,
    RawBlock,
    RawInline,
    Row,
    SoftBreak,
    Span,
    Strikethrough,
    Strong,
    Subscript,
    Superscript,
    Table,
    TableSection,
    Text,
    ThematicBreak,
    Underline,
    serialize_markdown,
)


class TestBlockSerialization:
    def test_paragraph(self) -> None:
        doc = ContentDocument(body=(Paragraph(children=(Text(value="Hello world"),)),))
        result = serialize_markdown(doc)
        assert "Hello world" in result

    def test_heading_depths(self) -> None:
        doc = ContentDocument(
            body=(
                Heading(depth=1, children=(Text(value="H1"),)),
                Heading(depth=2, children=(Text(value="H2"),)),
                Heading(depth=3, children=(Text(value="H3"),)),
            )
        )
        result = serialize_markdown(doc)
        assert "# H1" in result
        assert "## H2" in result
        assert "### H3" in result

    def test_blockquote(self) -> None:
        doc = ContentDocument(
            body=(BlockQuote(children=(Paragraph(children=(Text(value="quoted"),)),)),)
        )
        result = serialize_markdown(doc)
        assert "> quoted" in result

    def test_nested_blockquote(self) -> None:
        doc = ContentDocument(
            body=(
                BlockQuote(
                    children=(BlockQuote(children=(Paragraph(children=(Text(value="deep"),)),)),)
                ),
            )
        )
        result = serialize_markdown(doc)
        assert "> > deep" in result

    def test_bullet_list(self) -> None:
        doc = ContentDocument(
            body=(
                BulletList(
                    children=(
                        ListItem(children=(Paragraph(children=(Text(value="one"),)),)),
                        ListItem(children=(Paragraph(children=(Text(value="two"),)),)),
                    )
                ),
            )
        )
        result = serialize_markdown(doc)
        assert "- one" in result
        assert "- two" in result

    def test_ordered_list(self) -> None:
        doc = ContentDocument(
            body=(
                OrderedList(
                    start=1,
                    children=(
                        ListItem(children=(Paragraph(children=(Text(value="first"),)),)),
                        ListItem(children=(Paragraph(children=(Text(value="second"),)),)),
                    ),
                ),
            )
        )
        result = serialize_markdown(doc)
        assert "1. first" in result
        assert "2. second" in result

    def test_ordered_list_custom_start(self) -> None:
        doc = ContentDocument(
            body=(
                OrderedList(
                    start=5,
                    children=(ListItem(children=(Paragraph(children=(Text(value="fifth"),)),)),),
                ),
            )
        )
        result = serialize_markdown(doc)
        assert "5. fifth" in result

    def test_task_list(self) -> None:
        doc = ContentDocument(
            body=(
                BulletList(
                    children=(
                        ListItem(
                            checked=True,
                            children=(Paragraph(children=(Text(value="done"),)),),
                        ),
                        ListItem(
                            checked=False,
                            children=(Paragraph(children=(Text(value="todo"),)),),
                        ),
                    )
                ),
            )
        )
        result = serialize_markdown(doc)
        assert "[x] done" in result
        assert "[ ] todo" in result

    def test_code_block(self) -> None:
        doc = ContentDocument(body=(CodeBlock(language="python", value="x = 1"),))
        result = serialize_markdown(doc)
        assert "```python" in result
        assert "x = 1" in result
        assert "```" in result

    def test_code_block_no_language(self) -> None:
        doc = ContentDocument(body=(CodeBlock(value="plain code"),))
        result = serialize_markdown(doc)
        assert "```\n" in result

    def test_thematic_break(self) -> None:
        doc = ContentDocument(body=(ThematicBreak(),))
        result = serialize_markdown(doc)
        assert "---" in result

    def test_page_break(self) -> None:
        doc = ContentDocument(body=(PageBreak(),))
        result = serialize_markdown(doc)
        assert "---" in result

    def test_math_block(self) -> None:
        doc = ContentDocument(body=(MathBlock(value="E = mc^2"),))
        result = serialize_markdown(doc)
        assert "$$" in result
        assert "E = mc^2" in result

    def test_admonition(self) -> None:
        doc = ContentDocument(
            body=(
                Admonition(
                    kind="warning",
                    children=(Paragraph(children=(Text(value="Be careful"),)),),
                ),
            )
        )
        result = serialize_markdown(doc)
        assert "> [!WARNING]" in result
        assert "> Be careful" in result

    def test_div_transparent(self) -> None:
        doc = ContentDocument(body=(Div(children=(Paragraph(children=(Text(value="inside"),)),)),))
        result = serialize_markdown(doc)
        assert "inside" in result

    def test_raw_block_html_stripped_by_default(self) -> None:
        # Safe-by-default: raw HTML blocks are dropped to prevent XSS
        # when the markdown is later rendered to HTML by a downstream
        # consumer.
        doc = ContentDocument(body=(RawBlock(format="html", value="<div>raw</div>"),))
        result = serialize_markdown(doc)
        assert "<div>raw</div>" not in result
        assert "raw html stripped" in result

    def test_raw_block_html_passthrough_when_explicitly_allowed(self) -> None:
        doc = ContentDocument(body=(RawBlock(format="html", value="<div>raw</div>"),))
        result = serialize_markdown(doc, allow_raw_html=True)
        assert "<div>raw</div>" in result

    def test_raw_block_other_skipped(self) -> None:
        doc = ContentDocument(body=(RawBlock(format="latex", value="\\section{x}"),))
        result = serialize_markdown(doc)
        assert "\\section" not in result

    def test_definition_list(self) -> None:
        doc = ContentDocument(
            body=(
                DefinitionList(
                    children=(
                        DefinitionItem(
                            term=(Text(value="Term"),),
                            definitions=((Paragraph(children=(Text(value="definition"),)),),),
                        ),
                    )
                ),
            )
        )
        result = serialize_markdown(doc)
        assert "Term\n" in result
        assert ":   definition" in result


class TestInlineSerialization:
    def test_text_escaping(self) -> None:
        doc = ContentDocument(body=(Paragraph(children=(Text(value="hello *world* and [link]"),)),))
        result = serialize_markdown(doc)
        assert "\\*world\\*" in result
        assert "\\[link\\]" in result

    def test_emphasis(self) -> None:
        doc = ContentDocument(
            body=(Paragraph(children=(Emphasis(children=(Text(value="italic"),)),)),)
        )
        result = serialize_markdown(doc)
        assert "*italic*" in result

    def test_strong(self) -> None:
        doc = ContentDocument(body=(Paragraph(children=(Strong(children=(Text(value="bold"),)),)),))
        result = serialize_markdown(doc)
        assert "**bold**" in result

    def test_strikethrough(self) -> None:
        doc = ContentDocument(
            body=(Paragraph(children=(Strikethrough(children=(Text(value="struck"),)),)),)
        )
        result = serialize_markdown(doc)
        assert "~~struck~~" in result

    def test_code_inline(self) -> None:
        doc = ContentDocument(body=(Paragraph(children=(Code(value="x = 1"),)),))
        result = serialize_markdown(doc)
        assert "`x = 1`" in result

    def test_code_with_backtick(self) -> None:
        doc = ContentDocument(body=(Paragraph(children=(Code(value="a`b"),)),))
        result = serialize_markdown(doc)
        # No space padding: value doesn't start/end with backtick
        assert "``a`b``" in result

    def test_link(self) -> None:
        doc = ContentDocument(
            body=(
                Paragraph(
                    children=(
                        Link(
                            url="https://example.com",
                            children=(Text(value="click"),),
                        ),
                    )
                ),
            )
        )
        result = serialize_markdown(doc)
        assert "[click](https://example.com)" in result

    def test_link_with_title(self) -> None:
        doc = ContentDocument(
            body=(
                Paragraph(
                    children=(
                        Link(
                            url="u",
                            title="Title",
                            children=(Text(value="text"),),
                        ),
                    )
                ),
            )
        )
        result = serialize_markdown(doc)
        assert '[text](u "Title")' in result

    def test_image(self) -> None:
        doc = ContentDocument(body=(Paragraph(children=(Image(src="img.png", alt="Photo"),)),))
        result = serialize_markdown(doc)
        assert "![Photo](img.png)" in result

    def test_footnote_ref(self) -> None:
        doc = ContentDocument(
            body=(Paragraph(children=(Text(value="See"), FootnoteRef(identifier="fn1"))),),
            footnotes={"fn1": (Paragraph(children=(Text(value="A note."),)),)},
        )
        result = serialize_markdown(doc)
        assert "[^fn1]" in result
        assert "[^fn1]:" in result

    def test_math_inline(self) -> None:
        doc = ContentDocument(body=(Paragraph(children=(Math(value="x^2"),)),))
        result = serialize_markdown(doc)
        assert "$x^2$" in result

    def test_line_break(self) -> None:
        doc = ContentDocument(
            body=(Paragraph(children=(Text(value="a"), LineBreak(), Text(value="b"))),)
        )
        result = serialize_markdown(doc)
        assert "a\\\nb" in result

    def test_soft_break(self) -> None:
        doc = ContentDocument(
            body=(Paragraph(children=(Text(value="a"), SoftBreak(), Text(value="b"))),)
        )
        result = serialize_markdown(doc)
        assert "a\nb" in result

    def test_span_transparent(self) -> None:
        doc = ContentDocument(body=(Paragraph(children=(Span(children=(Text(value="inner"),)),)),))
        result = serialize_markdown(doc)
        assert "inner" in result

    def test_superscript(self) -> None:
        doc = ContentDocument(
            body=(Paragraph(children=(Superscript(children=(Text(value="2"),)),)),)
        )
        result = serialize_markdown(doc)
        assert "<sup>2</sup>" in result

    def test_subscript(self) -> None:
        doc = ContentDocument(body=(Paragraph(children=(Subscript(children=(Text(value="i"),)),)),))
        result = serialize_markdown(doc)
        assert "<sub>i</sub>" in result

    def test_underline(self) -> None:
        doc = ContentDocument(
            body=(Paragraph(children=(Underline(children=(Text(value="under"),)),)),)
        )
        result = serialize_markdown(doc)
        assert "<u>under</u>" in result

    def test_citation(self) -> None:
        doc = ContentDocument(
            body=(
                Paragraph(
                    children=(
                        Citation(
                            identifiers=("smith2024",),
                            children=(Text(value="Smith (2024)"),),
                        ),
                    )
                ),
            )
        )
        result = serialize_markdown(doc)
        # Parentheses are NOT escaped in inline context
        assert "Smith (2024)" in result

    def test_raw_inline_html_stripped_by_default(self) -> None:
        """RawInline(format="html") is gated on allow_raw_html — the
        same safe-default contract as RawBlock. Without it, an attacker
        could smuggle ``<script>`` past the serializer's URL filter via
        a RawInline node (audit finding)."""
        doc = ContentDocument(
            body=(Paragraph(children=(RawInline(format="html", value="<br/>"),)),)
        )
        result = serialize_markdown(doc)
        assert "<br/>" not in result
        assert "<!-- raw html stripped -->" in result

    def test_raw_inline_html_passthrough_when_allowed(self) -> None:
        """Explicit opt-in still works for trusted callers."""
        doc = ContentDocument(
            body=(Paragraph(children=(RawInline(format="html", value="<br/>"),)),)
        )
        result = serialize_markdown(doc, allow_raw_html=True)
        assert "<br/>" in result

    def test_raw_inline_other_skipped(self) -> None:
        doc = ContentDocument(
            body=(Paragraph(children=(RawInline(format="latex", value="\\textbf{x}"),)),)
        )
        result = serialize_markdown(doc)
        assert "\\textbf" not in result


class TestTableSerialization:
    def test_simple_table(self) -> None:
        doc = ContentDocument(
            body=(
                Table(
                    head=TableSection(
                        rows=(
                            Row(
                                cells=(
                                    Cell(content=(Paragraph(children=(Text(value="Name"),)),)),
                                    Cell(content=(Paragraph(children=(Text(value="Age"),)),)),
                                )
                            ),
                        )
                    ),
                    bodies=(
                        TableSection(
                            rows=(
                                Row(
                                    cells=(
                                        Cell(content=(Paragraph(children=(Text(value="Alice"),)),)),
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
        assert "Name" in result
        assert "Age" in result
        assert "Alice" in result
        assert "---" in result

    def test_table_alignment(self) -> None:
        doc = ContentDocument(
            body=(
                Table(
                    col_specs=(
                        ColSpec(alignment=Alignment.LEFT),
                        ColSpec(alignment=Alignment.CENTER),
                        ColSpec(alignment=Alignment.RIGHT),
                    ),
                    head=TableSection(
                        rows=(
                            Row(
                                cells=(
                                    Cell(content=(Paragraph(children=(Text(value="L"),)),)),
                                    Cell(content=(Paragraph(children=(Text(value="C"),)),)),
                                    Cell(content=(Paragraph(children=(Text(value="R"),)),)),
                                )
                            ),
                        )
                    ),
                    bodies=(TableSection(rows=()),),
                ),
            )
        )
        result = serialize_markdown(doc)
        assert ":---" in result
        assert ":---:" in result
        assert "---:" in result

    def test_table_with_caption(self) -> None:
        doc = ContentDocument(
            body=(
                Table(
                    caption=Caption(body=(Paragraph(children=(Text(value="My Table"),)),)),
                    head=TableSection(
                        rows=(
                            Row(cells=(Cell(content=(Paragraph(children=(Text(value="H"),)),)),)),
                        )
                    ),
                    bodies=(
                        TableSection(
                            rows=(
                                Row(
                                    cells=(Cell(content=(Paragraph(children=(Text(value="D"),)),)),)
                                ),
                            )
                        ),
                    ),
                ),
            )
        )
        result = serialize_markdown(doc)
        assert "*My Table*" in result

    def test_table_no_header(self) -> None:
        doc = ContentDocument(
            body=(
                Table(
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
                    )
                ),
            )
        )
        result = serialize_markdown(doc)
        assert "data" in result
        assert "---" in result


class TestFigureSerialization:
    def test_figure_with_image_and_caption(self) -> None:
        doc = ContentDocument(
            body=(
                Figure(
                    children=(Paragraph(children=(Image(src="photo.png", alt="Photo"),)),),
                    caption=Caption(body=(Paragraph(children=(Text(value="Figure 1"),)),)),
                ),
            )
        )
        result = serialize_markdown(doc)
        assert "![Photo](photo.png)" in result
        assert "*Figure 1*" in result


class TestLossyDegradation:
    def test_provenance_dropped(self) -> None:
        doc = ContentDocument(
            body=(
                Paragraph(
                    children=(Text(value="text"),),
                    provenance=Provenance(page=7, confidence=0.95),
                ),
            )
        )
        result = serialize_markdown(doc)
        assert "text" in result
        assert "page" not in result.lower()
        assert "confidence" not in result.lower()

    def test_attr_dropped(self) -> None:
        doc = ContentDocument(
            body=(
                Paragraph(
                    children=(Text(value="text"),),
                    attr=Attr(id="p1", classes=("legal",)),
                ),
            )
        )
        result = serialize_markdown(doc)
        assert "text" in result
        assert "p1" not in result
        assert "legal" not in result

    def test_annotations_dropped(self) -> None:
        doc = ContentDocument(
            body=(Paragraph(children=(Text(value="text"),)),),
            annotations=(
                Annotation(
                    id="a1",
                    type=AnnotationType.HIGHLIGHT,
                    targets=(AnnotationTarget(node_ref="#/body/0"),),
                ),
            ),
        )
        result = serialize_markdown(doc)
        assert "text" in result
        assert "highlight" not in result.lower()

    def test_redaction_replaces_block(self) -> None:
        """REDACTION annotations replace targeted block content with [REDACTED]."""
        doc = ContentDocument(
            body=(
                Paragraph(children=(Text(value="public info"),)),
                Paragraph(children=(Text(value="secret info"),)),
            ),
            annotations=(
                Annotation(
                    id="r1",
                    type=AnnotationType.REDACTION,
                    targets=(AnnotationTarget(node_ref="#/body/1"),),
                ),
            ),
        )
        result = serialize_markdown(doc)
        assert "public info" in result
        assert "secret info" not in result
        assert "[REDACTED]" in result

    def test_redaction_replaces_inline(self) -> None:
        """REDACTION annotations can target inline nodes."""
        doc = ContentDocument(
            body=(
                Paragraph(
                    children=(
                        Text(value="before "),
                        Text(value="secret"),
                        Text(value=" after"),
                    )
                ),
            ),
            annotations=(
                Annotation(
                    id="r1",
                    type=AnnotationType.REDACTION,
                    targets=(AnnotationTarget(node_ref="#/body/0/children/1"),),
                ),
            ),
        )
        result = serialize_markdown(doc)
        assert "before" in result
        assert "secret" not in result
        assert "after" in result
        assert "[REDACTED]" in result

    def test_redaction_multiple_targets(self) -> None:
        """Multiple redaction targets in the same document."""
        doc = ContentDocument(
            body=(
                Paragraph(children=(Text(value="public"),)),
                Paragraph(children=(Text(value="secret1"),)),
                Paragraph(children=(Text(value="secret2"),)),
            ),
            annotations=(
                Annotation(
                    id="r1",
                    type=AnnotationType.REDACTION,
                    targets=(
                        AnnotationTarget(node_ref="#/body/1"),
                        AnnotationTarget(node_ref="#/body/2"),
                    ),
                ),
            ),
        )
        result = serialize_markdown(doc)
        assert "public" in result
        assert "secret1" not in result
        assert "secret2" not in result
        assert result.count("[REDACTED]") == 2

    def test_redaction_in_footnote(self) -> None:
        """REDACTION annotations targeting footnote content should work."""
        doc = ContentDocument(
            body=(Paragraph(children=(Text(value="See note"), FootnoteRef(identifier="fn1"))),),
            footnotes={
                "fn1": (Paragraph(children=(Text(value="secret footnote"),)),),
            },
            annotations=(
                Annotation(
                    id="r1",
                    type=AnnotationType.REDACTION,
                    targets=(AnnotationTarget(node_ref="#/footnotes/fn1/0"),),
                ),
            ),
        )
        result = serialize_markdown(doc)
        assert "secret footnote" not in result
        assert "[REDACTED]" in result
        assert "[^fn1]" in result  # footnote ref still present


class TestNestedStructures:
    def test_list_in_blockquote(self) -> None:
        doc = ContentDocument(
            body=(
                BlockQuote(
                    children=(
                        BulletList(
                            children=(
                                ListItem(children=(Paragraph(children=(Text(value="item"),)),)),
                            )
                        ),
                    )
                ),
            )
        )
        result = serialize_markdown(doc)
        assert "> - item" in result

    def test_complex_document(self) -> None:
        doc = ContentDocument(
            metadata=DocumentMetadata(title="Test"),
            body=(
                Heading(depth=1, children=(Text(value="Title"),)),
                Paragraph(
                    children=(
                        Text(value="Text with "),
                        Strong(children=(Text(value="bold"),)),
                        Text(value=" and "),
                        Emphasis(children=(Text(value="italic"),)),
                    )
                ),
                CodeBlock(language="python", value="x = 1"),
                BulletList(
                    children=(
                        ListItem(children=(Paragraph(children=(Text(value="a"),)),)),
                        ListItem(children=(Paragraph(children=(Text(value="b"),)),)),
                    )
                ),
            ),
        )
        result = serialize_markdown(doc)
        assert "# Title" in result
        assert "**bold**" in result
        assert "*italic*" in result
        assert "```python" in result
        assert "- a" in result

    def test_empty_document(self) -> None:
        doc = ContentDocument()
        result = serialize_markdown(doc)
        assert result.strip() == ""
