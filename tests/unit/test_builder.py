"""Tests for DocumentBuilder (Phase 4)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from kaos_content import (
    Admonition,
    Alignment,
    AnnotationTarget,
    AnnotationType,
    BlockQuote,
    BoundingBox,
    BulletList,
    CodeBlock,
    ContentDocument,
    DefinitionList,
    Div,
    DocumentBuilder,
    Emphasis,
    Figure,
    Heading,
    ListItem,
    MathBlock,
    NodeIndex,
    OrderedList,
    PageBreak,
    Paragraph,
    RawBlock,
    Strong,
    Table,
    Text,
    ThematicBreak,
    content_hash,
    extract_text,
    serialize_markdown,
)


class TestBasicBlocks:
    """Test adding each block type via the builder."""

    def test_heading(self) -> None:
        doc = DocumentBuilder().heading(1, "Title").build()
        assert len(doc.body) == 1
        h = doc.body[0]
        assert isinstance(h, Heading)
        assert h.depth == 1
        assert extract_text(h) == "Title"

    def test_paragraph_with_string(self) -> None:
        doc = DocumentBuilder().paragraph("Hello world").build()
        assert len(doc.body) == 1
        p = doc.body[0]
        assert isinstance(p, Paragraph)
        assert extract_text(p) == "Hello world"

    def test_paragraph_with_inlines(self) -> None:
        b = DocumentBuilder()
        doc = b.paragraph("Before ", b.bold("bold"), " after").build()
        p = doc.body[0]
        assert isinstance(p, Paragraph)
        assert len(p.children) == 3
        assert isinstance(p.children[1], Strong)

    def test_paragraph_mixed_str_and_inline(self) -> None:
        b = DocumentBuilder()
        doc = b.paragraph("text", b.italic("em"), b.code("x")).build()
        p = doc.body[0]
        assert isinstance(p, Paragraph)
        assert len(p.children) == 3
        assert isinstance(p.children[0], Text)
        assert isinstance(p.children[1], Emphasis)

    def test_code_block(self) -> None:
        doc = DocumentBuilder().code_block("x = 1", language="python").build()
        cb = doc.body[0]
        assert isinstance(cb, CodeBlock)
        assert cb.value == "x = 1"
        assert cb.language == "python"

    def test_code_block_no_language(self) -> None:
        doc = DocumentBuilder().code_block("plain").build()
        cb = doc.body[0]
        assert isinstance(cb, CodeBlock)
        assert cb.language is None

    def test_math_block(self) -> None:
        doc = DocumentBuilder().math_block("E = mc^2").build()
        mb = doc.body[0]
        assert isinstance(mb, MathBlock)
        assert mb.value == "E = mc^2"

    def test_thematic_break(self) -> None:
        doc = DocumentBuilder().thematic_break().build()
        assert isinstance(doc.body[0], ThematicBreak)

    def test_page_break(self) -> None:
        doc = DocumentBuilder().page_break().build()
        assert isinstance(doc.body[0], PageBreak)

    def test_image_as_figure(self) -> None:
        doc = DocumentBuilder().image("photo.png", alt="A photo").build()
        fig = doc.body[0]
        assert isinstance(fig, Figure)

    def test_raw_block(self) -> None:
        doc = DocumentBuilder().raw_block("<div>hello</div>").build()
        rb = doc.body[0]
        assert isinstance(rb, RawBlock)
        assert rb.format == "html"
        assert rb.value == "<div>hello</div>"

    def test_blockquote_flat(self) -> None:
        p = Paragraph(children=(Text(value="quoted"),))
        doc = DocumentBuilder().blockquote(p).build()
        bq = doc.body[0]
        assert isinstance(bq, BlockQuote)
        assert extract_text(bq) == "quoted"

    def test_admonition(self) -> None:
        p = Paragraph(children=(Text(value="Be careful"),))
        doc = DocumentBuilder().admonition("warning", p, title="Warning").build()
        adm = doc.body[0]
        assert isinstance(adm, Admonition)
        assert adm.kind == "warning"
        assert adm.title == "Warning"

    def test_definition_list(self) -> None:
        doc = DocumentBuilder().definition_list(("Term1", "Def1"), ("Term2", "Def2")).build()
        dl = doc.body[0]
        assert isinstance(dl, DefinitionList)
        assert len(dl.children) == 2

    def test_add_block_direct(self) -> None:
        """add_block() accepts a pre-built block node."""
        para = Paragraph(children=(Text(value="direct"),))
        doc = DocumentBuilder().add_block(para).build()
        assert doc.body[0] is para


class TestTable:
    def test_simple_table(self) -> None:
        doc = (
            DocumentBuilder()
            .table(
                headers=["Name", "Age"],
                rows=[["Alice", "30"], ["Bob", "25"]],
            )
            .build()
        )
        t = doc.body[0]
        assert isinstance(t, Table)
        assert t.head is not None
        assert len(t.head.rows) == 1
        assert len(t.head.rows[0].cells) == 2
        assert len(t.bodies) == 1
        assert len(t.bodies[0].rows) == 2

    def test_table_with_alignment(self) -> None:
        doc = (
            DocumentBuilder()
            .table(
                headers=["L", "C", "R"],
                rows=[["a", "b", "c"]],
                alignments=[Alignment.LEFT, Alignment.CENTER, Alignment.RIGHT],
            )
            .build()
        )
        t = doc.body[0]
        assert isinstance(t, Table)
        assert len(t.col_specs) == 3
        assert t.col_specs[0].alignment == Alignment.LEFT
        assert t.col_specs[2].alignment == Alignment.RIGHT

    def test_table_serializes_to_markdown(self) -> None:
        doc = DocumentBuilder().table(headers=["H"], rows=[["D"]]).build()
        md = serialize_markdown(doc)
        assert "H" in md
        assert "D" in md
        assert "---" in md


class TestNesting:
    """Test nested block construction via begin/end stack."""

    def test_blockquote_nested(self) -> None:
        doc = DocumentBuilder().begin_blockquote().paragraph("Inside blockquote").end().build()
        bq = doc.body[0]
        assert isinstance(bq, BlockQuote)
        assert len(bq.children) == 1
        assert extract_text(bq) == "Inside blockquote"

    def test_bullet_list(self) -> None:
        doc = (
            DocumentBuilder()
            .begin_list()
            .begin_list_item()
            .paragraph("Item 1")
            .end()
            .begin_list_item()
            .paragraph("Item 2")
            .end()
            .end()
            .build()
        )
        bl = doc.body[0]
        assert isinstance(bl, BulletList)
        assert len(bl.children) == 2

    def test_ordered_list(self) -> None:
        doc = (
            DocumentBuilder()
            .begin_list(ordered=True, start=5)
            .begin_list_item()
            .paragraph("Fifth")
            .end()
            .end()
            .build()
        )
        ol = doc.body[0]
        assert isinstance(ol, OrderedList)
        assert ol.start == 5

    def test_task_list(self) -> None:
        doc = (
            DocumentBuilder()
            .begin_list()
            .begin_list_item(checked=True)
            .paragraph("Done")
            .end()
            .begin_list_item(checked=False)
            .paragraph("Todo")
            .end()
            .end()
            .build()
        )
        bl = doc.body[0]
        assert isinstance(bl, BulletList)
        items = bl.children
        assert isinstance(items[0], ListItem)
        assert items[0].checked is True
        assert isinstance(items[1], ListItem)
        assert items[1].checked is False

    def test_div_with_attr(self) -> None:
        doc = (
            DocumentBuilder().begin_div(role="exhibit").paragraph("Exhibit A content").end().build()
        )
        div = doc.body[0]
        assert isinstance(div, Div)
        assert div.attr.kv["role"] == "exhibit"

    def test_figure(self) -> None:
        b = DocumentBuilder()
        doc = (
            b.begin_figure(caption_text="Figure 1")
            .paragraph(b.text("Image placeholder"))
            .end()
            .build()
        )
        fig = doc.body[0]
        assert isinstance(fig, Figure)
        assert fig.caption is not None
        assert extract_text(fig.caption.body[0]) == "Figure 1"

    def test_deep_nesting(self) -> None:
        """List inside blockquote inside div."""
        doc = (
            DocumentBuilder()
            .begin_div()
            .begin_blockquote()
            .begin_list()
            .begin_list_item()
            .paragraph("Deep item")
            .end()
            .end()
            .end()
            .end()
            .build()
        )
        div = doc.body[0]
        assert isinstance(div, Div)
        bq = div.children[0]
        assert isinstance(bq, BlockQuote)
        bl = bq.children[0]
        assert isinstance(bl, BulletList)
        assert extract_text(bl) == "Deep item"

    def test_mismatched_end_raises(self) -> None:
        """end() without begin raises ValueError."""
        with pytest.raises(ValueError, match=r"end.*without matching begin"):
            DocumentBuilder().end()

    def test_unclosed_nesting_raises(self) -> None:
        """build() with unclosed nesting raises ValueError."""
        b = DocumentBuilder().begin_blockquote().paragraph("unclosed")
        with pytest.raises(ValueError, match="Unclosed nesting"):
            b.build()

    def test_multiple_unclosed_raises(self) -> None:
        b = DocumentBuilder().begin_div().begin_list()
        with pytest.raises(ValueError, match=r"Unclosed.*div.*bullet_list"):
            b.build()


class TestInlineHelpers:
    """Test static inline factory methods."""

    def test_text(self) -> None:
        t = DocumentBuilder.text("hello")
        assert isinstance(t, Text)
        assert t.value == "hello"

    def test_bold(self) -> None:
        s = DocumentBuilder.bold("strong")
        assert isinstance(s, Strong)
        assert extract_text(s) == "strong"

    def test_italic(self) -> None:
        e = DocumentBuilder.italic("em")
        assert isinstance(e, Emphasis)
        assert extract_text(e) == "em"

    def test_link(self) -> None:
        lk = DocumentBuilder.link("Click", "https://example.com", title="Example")
        from kaos_content import Link

        assert isinstance(lk, Link)
        assert lk.url == "https://example.com"
        assert lk.title == "Example"

    def test_code(self) -> None:
        from kaos_content import Code

        c = DocumentBuilder.code("x = 1")
        assert isinstance(c, Code)
        assert c.value == "x = 1"

    def test_math(self) -> None:
        from kaos_content import Math

        m = DocumentBuilder.math("x^2")
        assert isinstance(m, Math)
        assert m.value == "x^2"

    def test_footnote_ref(self) -> None:
        from kaos_content import FootnoteRef

        fr = DocumentBuilder.footnote_ref("fn1")
        assert isinstance(fr, FootnoteRef)
        assert fr.identifier == "fn1"


class TestMetadata:
    def test_title(self) -> None:
        doc = DocumentBuilder(title="My Doc").build()
        assert doc.metadata.title == "My Doc"

    def test_set_metadata(self) -> None:
        doc = DocumentBuilder().set_metadata(title="Title", document_type="contract").build()
        assert doc.metadata.title == "Title"
        assert doc.metadata.document_type == "contract"

    def test_set_source(self) -> None:
        doc = (
            DocumentBuilder()
            .set_source("file:///doc.pdf", mime_type="application/pdf")
            .paragraph("Content")
            .with_provenance(page=1)
            .build()
        )
        p = doc.body[0]
        assert isinstance(p, Paragraph)
        assert p.provenance is not None
        assert p.provenance.source is not None
        assert p.provenance.source.uri == "file:///doc.pdf"
        assert p.provenance.page == 1


class TestProvenance:
    def test_basic_provenance(self) -> None:
        doc = (
            DocumentBuilder()
            .paragraph("Extracted text")
            .with_provenance(page=7, confidence=0.95)
            .build()
        )
        p = doc.body[0]
        assert isinstance(p, Paragraph)
        assert p.provenance is not None
        assert p.provenance.page == 7
        assert p.provenance.confidence == 0.95

    def test_provenance_with_bbox(self) -> None:
        bbox = BoundingBox(left=10, top=20, right=100, bottom=50)
        doc = DocumentBuilder().heading(1, "Title").with_provenance(page=1, bbox=bbox).build()
        h = doc.body[0]
        assert isinstance(h, Heading)
        assert h.provenance is not None
        assert h.provenance.bbox == bbox

    def test_provenance_with_extractor(self) -> None:
        doc = (
            DocumentBuilder()
            .code_block("code", language="python")
            .with_provenance(extractor="tesseract")
            .build()
        )
        cb = doc.body[0]
        assert isinstance(cb, CodeBlock)
        assert cb.provenance is not None
        assert cb.provenance.extractor == "tesseract"

    def test_provenance_before_block_raises(self) -> None:
        with pytest.raises(ValueError, match="before adding any block"):
            DocumentBuilder().with_provenance(page=1)

    def test_provenance_on_multiple_blocks(self) -> None:
        doc = (
            DocumentBuilder()
            .paragraph("Page 1")
            .with_provenance(page=1)
            .paragraph("Page 2")
            .with_provenance(page=2)
            .build()
        )
        p0 = doc.body[0]
        p1 = doc.body[1]
        assert p0.provenance is not None
        assert p0.provenance.page == 1
        assert p1.provenance is not None
        assert p1.provenance.page == 2


class TestFootnotes:
    def test_add_footnote_string(self) -> None:
        doc = (
            DocumentBuilder()
            .paragraph("See note", DocumentBuilder.footnote_ref("fn1"))
            .add_footnote("fn1", "This is the footnote.")
            .build()
        )
        assert "fn1" in doc.footnotes
        assert len(doc.footnotes["fn1"]) == 1
        assert extract_text(doc.footnotes["fn1"][0]) == "This is the footnote."

    def test_add_footnote_block(self) -> None:
        para = Paragraph(children=(Text(value="Footnote content"),))
        doc = DocumentBuilder().paragraph("Body").add_footnote("fn1", para).build()
        assert doc.footnotes["fn1"][0] is para


class TestDefinitions:
    def test_add_definition(self) -> None:
        doc = DocumentBuilder().add_definition("example", "https://example.com").build()
        assert doc.definitions["example"] == "https://example.com"


class TestAnnotations:
    def test_annotate(self) -> None:
        doc = (
            DocumentBuilder()
            .paragraph("Some text")
            .annotate(
                AnnotationType.HIGHLIGHT,
                [AnnotationTarget(node_ref="#/body/0")],
            )
            .build()
        )
        assert len(doc.annotations) == 1
        ann = doc.annotations[0]
        assert ann.type == AnnotationType.HIGHLIGHT
        assert ann.targets[0].node_ref == "#/body/0"

    def test_annotate_with_body(self) -> None:
        doc = (
            DocumentBuilder()
            .paragraph("Term usage")
            .annotate(
                AnnotationType.DEFINED_TERM,
                [AnnotationTarget(node_ref="#/body/0", start_offset=0, end_offset=4)],
                body={"definition_id": "dt-001"},
            )
            .build()
        )
        assert doc.annotations[0].body["definition_id"] == "dt-001"

    def test_annotate_generates_id(self) -> None:
        doc = (
            DocumentBuilder()
            .paragraph("x")
            .annotate(AnnotationType.HIGHLIGHT, [AnnotationTarget(node_ref="#/body/0")])
            .build()
        )
        assert len(doc.annotations[0].id) == 32  # UUID v7 hex


class TestBuildProducesValidDocument:
    """The built document should be a valid, frozen ContentDocument."""

    def test_document_is_frozen(self) -> None:
        doc = DocumentBuilder().paragraph("x").build()
        assert isinstance(doc, ContentDocument)
        with pytest.raises(ValidationError):
            doc.body = ()

    def test_body_is_tuple(self) -> None:
        doc = DocumentBuilder().paragraph("x").build()
        assert isinstance(doc.body, tuple)

    def test_node_index_works(self) -> None:
        doc = DocumentBuilder().heading(1, "Title").paragraph("Text").build()
        index = NodeIndex(doc)
        assert len(index.headings) == 1
        assert index.headings[0].depth == 1

    def test_serializes_to_markdown(self) -> None:
        doc = (
            DocumentBuilder(title="Test")
            .heading(1, "Chapter 1")
            .paragraph("First paragraph.")
            .code_block("x = 1", language="python")
            .build()
        )
        md = serialize_markdown(doc)
        assert "# Chapter 1" in md
        assert "First paragraph." in md
        assert "```python" in md

    def test_json_roundtrip(self) -> None:
        doc = DocumentBuilder().heading(1, "Title").paragraph("Body text").build()
        j = doc.model_dump_json()
        doc2 = ContentDocument.model_validate_json(j)
        assert len(doc2.body) == 2
        assert doc2.body[0].id == doc.body[0].id

    def test_content_hash_deterministic(self) -> None:
        """Two documents built the same way have the same content hash per block."""

        def make() -> ContentDocument:
            return DocumentBuilder().heading(1, "Title").paragraph("Same content").build()

        d1 = make()
        d2 = make()
        for b1, b2 in zip(d1.body, d2.body, strict=True):
            assert content_hash(b1) == content_hash(b2)


class TestComplexDocument:
    """Build a realistic document exercising many features."""

    def test_legal_contract(self) -> None:
        b = DocumentBuilder(title="Service Agreement")
        b.set_metadata(document_type="contract", jurisdiction="Delaware")
        b.set_source("file:///contracts/sa-001.pdf", mime_type="application/pdf")

        b.heading(1, "ARTICLE I — DEFINITIONS")
        b.with_provenance(page=1)

        b.definition_list(
            ("Service Provider", "The party providing services under this Agreement."),
            ("Client", "The party receiving services under this Agreement."),
        )

        b.heading(1, "ARTICLE II — SCOPE OF SERVICES")
        b.with_provenance(page=2)

        b.paragraph("The Service Provider shall provide the following services:")

        (
            b.begin_list(ordered=True)
            .begin_list_item()
            .paragraph("Software development and maintenance")
            .end()
            .begin_list_item()
            .paragraph("Technical consultation")
            .end()
            .begin_list_item()
            .paragraph("Documentation and training")
            .end()
            .end()
        )

        b.heading(1, "ARTICLE III — COMPENSATION")
        b.with_provenance(page=3)

        b.table(
            headers=["Service", "Rate", "Unit"],
            rows=[
                ["Development", "$200", "hour"],
                ["Consultation", "$250", "hour"],
                ["Training", "$150", "session"],
            ],
        )

        b.thematic_break()
        b.paragraph(b.italic("Confidential — Do not distribute"))

        b.add_footnote("fn1", "All rates subject to annual adjustment.")

        doc = b.build()

        # Verify structure
        assert doc.metadata.title == "Service Agreement"
        assert doc.metadata.document_type == "contract"
        assert len(doc.body) >= 7  # headings + content
        assert "fn1" in doc.footnotes

        # Verify provenance
        assert doc.body[0].provenance is not None
        assert doc.body[0].provenance.page == 1
        assert doc.body[0].provenance.source is not None
        assert doc.body[0].provenance.source.uri == "file:///contracts/sa-001.pdf"

        # Verify serialization
        md = serialize_markdown(doc)
        assert "# ARTICLE I" in md
        assert "Development" in md
        assert "[^fn1]:" in md  # footnote definition rendered at end

        # Verify indexing
        index = NodeIndex(doc)
        assert len(index.headings) == 3
        assert len(index.tables) == 1
