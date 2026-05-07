"""Realistic legal-document traversal and NodeIndex correctness tests.

These tests build document structures that mirror real-world PDF/DOCX extraction
output and verify that traversal, indexing, text extraction, annotation
resolution, and provenance queries are all correct.
"""

from __future__ import annotations

from kaos_content import (
    Annotation,
    AnnotationTarget,
    AnnotationType,
    Attr,
    Block,
    BlockQuote,
    BulletList,
    Caption,
    Cell,
    Code,
    CodeBlock,
    ContentDocument,
    DefinitionItem,
    DefinitionList,
    Div,
    Emphasis,
    FootnoteRef,
    Heading,
    LineBreak,
    Link,
    ListItem,
    Math,
    MathBlock,
    NodeIndex,
    OrderedList,
    Paragraph,
    Provenance,
    RawBlock,
    RawInline,
    Row,
    SoftBreak,
    SourceRef,
    Span,
    Strong,
    Table,
    TableSection,
    Text,
    extract_text,
    find,
    walk_blocks,
    walk_inlines,
)


# ---------------------------------------------------------------------------
# Helper: build a realistic legal contract document
# ---------------------------------------------------------------------------
def _legal_contract() -> ContentDocument:
    """Simulate a multi-section legal contract with definitions, obligations,
    schedules (table inside a Div), annotations, and footnotes.
    """
    source = SourceRef(uri="contract.pdf", mime_type="application/pdf")

    # -- Section 1: Heading "Agreement" --
    h1 = Heading(
        depth=1,
        children=(Text(value="Agreement"),),
        provenance=Provenance(source=source, page=1),
    )

    # Preamble paragraph with a defined term in a Span
    preamble_span = Span(
        children=(Text(value="Licensor"),),
        attr=Attr(classes=("defined-term",)),
    )
    preamble = Paragraph(
        children=(
            Text(value="This Agreement is entered into by "),
            preamble_span,
            Text(value=" and Licensee."),
        ),
        provenance=Provenance(source=source, page=1),
    )

    # -- Section 2: Definitions --
    h2_def = Heading(
        depth=2,
        children=(Text(value="Definitions"),),
        provenance=Provenance(source=source, page=1),
    )
    def_list = DefinitionList(
        children=(
            DefinitionItem(
                term=(Strong(children=(Text(value="Affiliate"),)),),
                definitions=(
                    (
                        Paragraph(
                            children=(
                                Text(
                                    value=(
                                        "Any entity that controls, is controlled by, "
                                        "or is under common control with a party."
                                    )
                                ),
                            )
                        ),
                    ),
                ),
            ),
            DefinitionItem(
                term=(Strong(children=(Text(value="Confidential Information"),)),),
                definitions=(
                    (
                        Paragraph(
                            children=(
                                Text(
                                    value=(
                                        "All non-public information disclosed under this Agreement."
                                    )
                                ),
                            )
                        ),
                    ),
                ),
            ),
        ),
        provenance=Provenance(source=source, page=1),
    )

    # -- Section 3: Obligations (numbered clauses) --
    h2_oblig = Heading(
        depth=2,
        children=(Text(value="Obligations"),),
        provenance=Provenance(source=source, page=2),
    )
    obligations = OrderedList(
        start=1,
        children=(
            ListItem(
                children=(
                    Paragraph(
                        children=(
                            Text(value="Licensee shall not reverse-engineer the Software"),
                            FootnoteRef(identifier="fn1"),
                            Text(value="."),
                        )
                    ),
                )
            ),
            ListItem(
                children=(
                    Paragraph(
                        children=(
                            Text(value="Licensee shall maintain confidentiality per Section 2"),
                            FootnoteRef(identifier="fn2"),
                            Text(value="."),
                        )
                    ),
                )
            ),
            ListItem(
                children=(
                    Paragraph(
                        children=(
                            Text(value="All fees are set forth in Schedule A"),
                            FootnoteRef(identifier="fn3"),
                            Text(value="."),
                        )
                    ),
                )
            ),
        ),
        provenance=Provenance(source=source, page=2),
    )

    # -- Section 4: Schedules (Div with class "schedule" containing a Table) --
    h2_sched = Heading(
        depth=2,
        children=(Text(value="Schedules"),),
        provenance=Provenance(source=source, page=3),
    )
    schedule_table = Table(
        caption=Caption(
            short=(Text(value="Schedule A"),),
            body=(Paragraph(children=(Text(value="Fee Schedule"),)),),
        ),
        head=TableSection(
            rows=(
                Row(
                    cells=(
                        Cell(content=(Paragraph(children=(Text(value="Service"),)),)),
                        Cell(content=(Paragraph(children=(Text(value="Fee"),)),)),
                    )
                ),
            )
        ),
        bodies=(
            TableSection(
                rows=(
                    Row(
                        cells=(
                            Cell(content=(Paragraph(children=(Text(value="Platform License"),)),)),
                            Cell(content=(Paragraph(children=(Text(value="$50,000/yr"),)),)),
                        )
                    ),
                    Row(
                        cells=(
                            Cell(content=(Paragraph(children=(Text(value="Support"),)),)),
                            Cell(content=(Paragraph(children=(Text(value="$10,000/yr"),)),)),
                        )
                    ),
                )
            ),
        ),
        provenance=Provenance(source=source, page=3),
    )
    schedule_div = Div(
        children=(schedule_table,),
        attr=Attr(classes=("schedule",)),
        provenance=Provenance(source=source, page=3),
    )

    # A redacted paragraph
    redacted_para = Paragraph(
        children=(Text(value="[REDACTED MATERIAL]"),),
        provenance=Provenance(source=source, page=3),
    )

    # -- Footnotes --
    footnotes: dict[str, tuple[Block, ...]] = {
        "fn1": (Paragraph(children=(Text(value="See Software License Agreement, Exhibit B."),)),),
        "fn2": (
            Paragraph(
                children=(
                    Text(value="Subject to "),
                    Link(
                        url="#section-2",
                        children=(Text(value="Section 2"),),
                    ),
                    Text(value=" limitations."),
                )
            ),
        ),
        "fn3": (Paragraph(children=(Text(value="Fees are subject to annual CPI adjustment."),)),),
    }

    return ContentDocument(
        body=(
            h1,  # 0
            preamble,  # 1
            h2_def,  # 2
            def_list,  # 3
            h2_oblig,  # 4
            obligations,  # 5
            h2_sched,  # 6
            schedule_div,  # 7 -> contains table at children/0
            redacted_para,  # 8  (index shifted — see below)
        ),
        footnotes=footnotes,
        annotations=(
            # DEFINED_TERM on the Span "Licensor" inside preamble
            Annotation(
                id="ann-dt-1",
                type=AnnotationType.DEFINED_TERM,
                targets=(AnnotationTarget(node_ref="#/body/1/children/1"),),
                body={"term": "Licensor"},
            ),
            # REDACTION on the redacted paragraph
            Annotation(
                id="ann-redact-1",
                type=AnnotationType.REDACTION,
                targets=(AnnotationTarget(node_ref="#/body/8"),),
            ),
        ),
    )


# ---------------------------------------------------------------------------
# 1. Legal contract structure
# ---------------------------------------------------------------------------
class TestLegalContractStructure:
    """Verify NodeIndex over a realistic legal contract document."""

    def test_headings_in_order(self) -> None:
        doc = _legal_contract()
        index = NodeIndex(doc)
        headings = index.headings
        assert len(headings) == 4
        texts = [extract_text(h) for h in headings]
        assert texts == ["Agreement", "Definitions", "Obligations", "Schedules"]
        depths = [h.depth for h in headings]
        assert depths == [1, 2, 2, 2]

    def test_definition_items_found(self) -> None:
        doc = _legal_contract()
        index = NodeIndex(doc)
        items = index.by_type(DefinitionItem)
        assert len(items) == 2
        term_texts = [extract_text(item) for item in items]
        assert "Affiliate" in term_texts[0]
        assert "Confidential Information" in term_texts[1]

    def test_annotation_defined_term_resolves(self) -> None:
        doc = _legal_contract()
        index = NodeIndex(doc)
        # The Span "Licensor" is at #/body/1/children/1
        anns = index.annotations_for("#/body/1/children/1")
        assert len(anns) == 1
        assert anns[0].type == AnnotationType.DEFINED_TERM
        assert anns[0].body["term"] == "Licensor"

    def test_annotation_redaction_resolves(self) -> None:
        doc = _legal_contract()
        index = NodeIndex(doc)
        anns = index.annotations_for("#/body/8")
        assert len(anns) == 1
        assert anns[0].type == AnnotationType.REDACTION

    def test_schedule_div_contains_table(self) -> None:
        doc = _legal_contract()
        index = NodeIndex(doc)
        div_node = index.get("#/body/7")
        assert isinstance(div_node, Div)
        assert "schedule" in div_node.attr.classes
        table_node = index.get("#/body/7/children/0")
        assert isinstance(table_node, Table)

    def test_table_cells_indexed(self) -> None:
        doc = _legal_contract()
        index = NodeIndex(doc)
        # Head row: Service, Fee
        head_ref = "#/body/7/children/0/head/rows/0/cells/0/content/0"
        head_para = index.get(head_ref)
        assert head_para is not None
        assert extract_text(head_para) == "Service"

        # Body row 0 cell 1: "$50,000/yr"
        body_ref = "#/body/7/children/0/bodies/0/rows/0/cells/1/content/0"
        body_para = index.get(body_ref)
        assert body_para is not None
        assert extract_text(body_para) == "$50,000/yr"

    def test_footnote_refs_in_body(self) -> None:
        doc = _legal_contract()
        index = NodeIndex(doc)
        refs = index.by_type(FootnoteRef)
        assert len(refs) == 3
        identifiers = [r.identifier for r in refs]
        assert identifiers == ["fn1", "fn2", "fn3"]

    def test_extract_text_whole_document_body(self) -> None:
        """extract_text on each body block produces all text content."""
        doc = _legal_contract()
        all_text = " ".join(extract_text(block) for block in doc.body)
        # Check key phrases are present
        assert "Agreement" in all_text
        assert "Licensor" in all_text
        assert "Affiliate" in all_text
        assert "Confidential Information" in all_text
        assert "reverse-engineer" in all_text
        assert "Schedule A" in all_text
        assert "$50,000/yr" in all_text
        assert "REDACTED" in all_text

    def test_validate_annotations_all_valid(self) -> None:
        doc = _legal_contract()
        index = NodeIndex(doc)
        invalid = index.validate_annotations()
        assert invalid == []

    def test_ordered_list_items_indexed(self) -> None:
        doc = _legal_contract()
        index = NodeIndex(doc)
        # OrderedList is at #/body/5, items are children
        for i in range(3):
            li_ref = f"#/body/5/children/{i}"
            li = index.get(li_ref)
            assert isinstance(li, ListItem), f"Expected ListItem at {li_ref}, got {type(li)}"

    def test_walk_visits_all_footnote_content(self) -> None:
        doc = _legal_contract()
        inlines = list(walk_inlines(doc))
        # Footnote fn2 contains a Link — make sure walk_inlines finds it
        link_inlines = [i for i in inlines if isinstance(i, Link)]
        assert any(link.url == "#section-2" for link in link_inlines)


# ---------------------------------------------------------------------------
# Helper: PDF extraction output with per-node provenance
# ---------------------------------------------------------------------------
def _pdf_extraction_doc() -> ContentDocument:
    """Simulate messy PDF extraction: every node tagged with page numbers.
    Pages 1-4, with a table spanning pages 2-3.
    """
    src = SourceRef(uri="report.pdf", mime_type="application/pdf")

    page1_blocks = [
        Heading(
            depth=1,
            children=(Text(value="Annual Report"),),
            provenance=Provenance(source=src, page=1),
        ),
        Paragraph(
            children=(Text(value="This report summarizes fiscal year 2025 results."),),
            provenance=Provenance(source=src, page=1),
        ),
        Paragraph(
            children=(Text(value="Revenue grew 15% year-over-year."),),
            provenance=Provenance(source=src, page=1, confidence=0.95),
        ),
    ]

    # Table spanning pages 2-3: header on page 2, some cells on page 2, some on page 3
    spanning_table = Table(
        caption=Caption(
            body=(
                Paragraph(
                    children=(Text(value="Table 1: Revenue by Region"),),
                    provenance=Provenance(source=src, page=2),
                ),
            )
        ),
        head=TableSection(
            rows=(
                Row(
                    cells=(
                        Cell(
                            content=(
                                Paragraph(
                                    children=(Text(value="Region"),),
                                    provenance=Provenance(source=src, page=2),
                                ),
                            )
                        ),
                        Cell(
                            content=(
                                Paragraph(
                                    children=(Text(value="Revenue ($M)"),),
                                    provenance=Provenance(source=src, page=2),
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
                    # Row on page 2
                    Row(
                        cells=(
                            Cell(
                                content=(
                                    Paragraph(
                                        children=(Text(value="North America"),),
                                        provenance=Provenance(source=src, page=2),
                                    ),
                                )
                            ),
                            Cell(
                                content=(
                                    Paragraph(
                                        children=(Text(value="120.5"),),
                                        provenance=Provenance(source=src, page=2),
                                    ),
                                )
                            ),
                        )
                    ),
                    # Row that starts on page 2 but continues to page 3
                    Row(
                        cells=(
                            Cell(
                                content=(
                                    Paragraph(
                                        children=(Text(value="Europe"),),
                                        provenance=Provenance(source=src, page=3),
                                    ),
                                )
                            ),
                            Cell(
                                content=(
                                    Paragraph(
                                        children=(Text(value="85.2"),),
                                        provenance=Provenance(source=src, page=3),
                                    ),
                                )
                            ),
                        )
                    ),
                    # Row fully on page 3
                    Row(
                        cells=(
                            Cell(
                                content=(
                                    Paragraph(
                                        children=(Text(value="Asia-Pacific"),),
                                        provenance=Provenance(source=src, page=3),
                                    ),
                                )
                            ),
                            Cell(
                                content=(
                                    Paragraph(
                                        children=(Text(value="62.1"),),
                                        provenance=Provenance(source=src, page=3),
                                    ),
                                )
                            ),
                        )
                    ),
                )
            ),
        ),
        provenance=Provenance(source=src, page=2),
    )

    page4_blocks = [
        Paragraph(
            children=(Text(value="Looking ahead, we project continued growth."),),
            provenance=Provenance(source=src, page=4),
        ),
        Paragraph(
            children=(
                Text(value="For more details, see "),
                Link(url="https://example.com/report", children=(Text(value="full report"),)),
                Text(value="."),
            ),
            provenance=Provenance(source=src, page=4),
        ),
    ]

    return ContentDocument(body=(*page1_blocks, spanning_table, *page4_blocks))


# ---------------------------------------------------------------------------
# 2. PDF extraction output
# ---------------------------------------------------------------------------
class TestPdfExtractionOutput:
    """Verify provenance-based queries on a PDF-extracted document."""

    def test_by_provenance_page_1(self) -> None:
        doc = _pdf_extraction_doc()
        index = NodeIndex(doc)
        page1_nodes = index.by_provenance_page(1)
        # Heading + 2 Paragraphs
        assert len(page1_nodes) == 3
        texts = [extract_text(n) for n in page1_nodes]
        assert "Annual Report" in texts
        assert "Revenue grew 15% year-over-year." in texts

    def test_by_provenance_page_2(self) -> None:
        doc = _pdf_extraction_doc()
        index = NodeIndex(doc)
        page2_nodes = index.by_provenance_page(2)
        # Table itself (page=2) + caption para + 2 head cell paras + 2 body row0 cell paras = 6
        assert len(page2_nodes) == 6
        page2_texts = [extract_text(n) for n in page2_nodes]
        assert "Region" in page2_texts
        assert "Revenue ($M)" in page2_texts
        assert "North America" in page2_texts
        assert "120.5" in page2_texts

    def test_by_provenance_page_3(self) -> None:
        doc = _pdf_extraction_doc()
        index = NodeIndex(doc)
        page3_nodes = index.by_provenance_page(3)
        # Europe row (2 cells) + Asia-Pacific row (2 cells) = 4 paragraphs
        assert len(page3_nodes) == 4
        page3_texts = [extract_text(n) for n in page3_nodes]
        assert "Europe" in page3_texts
        assert "85.2" in page3_texts
        assert "Asia-Pacific" in page3_texts
        assert "62.1" in page3_texts

    def test_by_provenance_page_4(self) -> None:
        doc = _pdf_extraction_doc()
        index = NodeIndex(doc)
        page4_nodes = index.by_provenance_page(4)
        assert len(page4_nodes) == 2
        texts = [extract_text(n) for n in page4_nodes]
        assert any("continued growth" in t for t in texts)

    def test_no_nodes_on_page_5(self) -> None:
        doc = _pdf_extraction_doc()
        index = NodeIndex(doc)
        assert index.by_provenance_page(5) == []

    def test_table_ref_paths_correct(self) -> None:
        """Verify the JSON pointer paths for the cross-page table cells."""
        doc = _pdf_extraction_doc()
        index = NodeIndex(doc)
        # Table is body[3]
        table_ref = "#/body/3"
        assert isinstance(index[table_ref], Table)

        # Head cell 0
        head_ref = f"{table_ref}/head/rows/0/cells/0/content/0"
        assert extract_text(index[head_ref]) == "Region"

        # Body row 2 (Asia-Pacific) cell 0
        ap_ref = f"{table_ref}/bodies/0/rows/2/cells/0/content/0"
        assert extract_text(index[ap_ref]) == "Asia-Pacific"

    def test_extract_text_includes_all_table_data(self) -> None:
        doc = _pdf_extraction_doc()
        index = NodeIndex(doc)
        table = index["#/body/3"]
        table_text = extract_text(table)
        for expected in [
            "Region",
            "Revenue ($M)",
            "North America",
            "120.5",
            "Europe",
            "85.2",
            "Asia-Pacific",
            "62.1",
            "Table 1: Revenue by Region",
        ]:
            assert expected in table_text, f"Missing '{expected}' in table extract_text"


# ---------------------------------------------------------------------------
# Helper: multi-footnote document
# ---------------------------------------------------------------------------
def _multi_footnote_doc() -> ContentDocument:
    """Document with 3 footnote refs in body and complex footnote content."""
    return ContentDocument(
        body=(
            Paragraph(
                children=(
                    Text(value="Introduction text"),
                    FootnoteRef(identifier="fn1"),
                    Text(value=". More text"),
                    FootnoteRef(identifier="fn2"),
                    Text(value=". Final text"),
                    FootnoteRef(identifier="fn3"),
                    Text(value="."),
                )
            ),
        ),
        footnotes={
            "fn1": (Paragraph(children=(Text(value="Simple footnote."),)),),
            "fn2": (
                Paragraph(children=(Text(value="Footnote with list:"),)),
                BulletList(
                    children=(
                        ListItem(children=(Paragraph(children=(Text(value="Point A"),)),)),
                        ListItem(children=(Paragraph(children=(Text(value="Point B"),)),)),
                    )
                ),
            ),
            "fn3": (
                Paragraph(children=(Text(value="Footnote with quotation:"),)),
                BlockQuote(
                    children=(
                        Paragraph(
                            children=(Emphasis(children=(Text(value="The court held that..."),)),)
                        ),
                    )
                ),
            ),
        },
    )


# ---------------------------------------------------------------------------
# 3. Multi-footnote document
# ---------------------------------------------------------------------------
class TestMultiFootnoteDocument:
    """Verify NodeIndex indexes all footnote content, refs are correct,
    and walk visits footnote content."""

    def test_footnote_block_count(self) -> None:
        doc = _multi_footnote_doc()
        # fn1: 1 Para
        # fn2: 1 Para + 1 BulletList (2 ListItems each with 1 Para)
        # fn3: 1 Para + 1 BlockQuote (1 Para)
        blocks = list(walk_blocks(doc))
        # Body: 1 Para
        # fn1: 1 Para
        # fn2: 1 Para + BulletList + 2 ListItem + 2 Para = 6
        # fn3: 1 Para + BlockQuote + 1 Para = 3
        # Total: 1 + 1 + 6 + 3 = 11
        assert len(blocks) == 11

    def test_footnote_refs_indexed(self) -> None:
        doc = _multi_footnote_doc()
        index = NodeIndex(doc)
        # fn1 simple paragraph
        fn1_para = index.get("#/footnotes/fn1/0")
        assert isinstance(fn1_para, Paragraph)
        assert extract_text(fn1_para) == "Simple footnote."

    def test_fn2_bullet_list_indexed(self) -> None:
        doc = _multi_footnote_doc()
        index = NodeIndex(doc)
        # fn2[0] = Paragraph, fn2[1] = BulletList
        fn2_list = index.get("#/footnotes/fn2/1")
        assert isinstance(fn2_list, BulletList)

        # First list item
        fn2_li0 = index.get("#/footnotes/fn2/1/children/0")
        assert isinstance(fn2_li0, ListItem)

        # Paragraph inside first list item
        fn2_li0_para = index.get("#/footnotes/fn2/1/children/0/children/0")
        assert isinstance(fn2_li0_para, Paragraph)
        assert extract_text(fn2_li0_para) == "Point A"

        # Second list item paragraph
        fn2_li1_para = index.get("#/footnotes/fn2/1/children/1/children/0")
        assert isinstance(fn2_li1_para, Paragraph)
        assert extract_text(fn2_li1_para) == "Point B"

    def test_fn3_blockquote_indexed(self) -> None:
        doc = _multi_footnote_doc()
        index = NodeIndex(doc)
        # fn3[1] = BlockQuote
        fn3_bq = index.get("#/footnotes/fn3/1")
        assert isinstance(fn3_bq, BlockQuote)

        # The paragraph inside the blockquote
        fn3_bq_para = index.get("#/footnotes/fn3/1/children/0")
        assert isinstance(fn3_bq_para, Paragraph)
        assert extract_text(fn3_bq_para) == "The court held that..."

        # The Emphasis node inside
        fn3_em = index.get("#/footnotes/fn3/1/children/0/children/0")
        assert isinstance(fn3_em, Emphasis)

    def test_walk_visits_footnote_content(self) -> None:
        doc = _multi_footnote_doc()
        all_inlines = list(walk_inlines(doc))
        text_values = [getattr(i, "value", None) for i in all_inlines if hasattr(i, "value")]
        assert "Simple footnote." in text_values
        assert "Point A" in text_values
        assert "Point B" in text_values
        assert "The court held that..." in text_values

    def test_find_footnote_refs_in_body(self) -> None:
        doc = _multi_footnote_doc()
        refs = find(doc, lambda n: isinstance(n, FootnoteRef))
        assert len(refs) == 3
        ids = [r.identifier for r in refs if isinstance(r, FootnoteRef)]
        assert ids == ["fn1", "fn2", "fn3"]

    def test_footnote_text_extraction(self) -> None:
        """extract_text on footnote blocks produces correct text."""
        doc = _multi_footnote_doc()
        # fn2: "Footnote with list:" + "Point A" + "Point B"
        fn2_text = "".join(extract_text(block) for block in doc.footnotes["fn2"])
        assert "Footnote with list:" in fn2_text
        assert "Point A" in fn2_text
        assert "Point B" in fn2_text


# ---------------------------------------------------------------------------
# 4. Annotation validation edge cases
# ---------------------------------------------------------------------------
class TestAnnotationValidationEdgeCases:
    """Build a document with valid and invalid annotation targets
    and verify validate_annotations returns exactly the invalid refs."""

    def _build_doc(self) -> ContentDocument:
        return ContentDocument(
            body=(
                Paragraph(
                    children=(
                        Text(value="First paragraph text."),
                        Span(children=(Text(value="defined term"),)),
                    )
                ),
                Paragraph(children=(Text(value="Second paragraph."),)),
                Paragraph(children=(Text(value="Third paragraph."),)),
            ),
            annotations=(
                # Valid: targets existing paragraph
                Annotation(
                    id="valid-1",
                    type=AnnotationType.HIGHLIGHT,
                    targets=(AnnotationTarget(node_ref="#/body/0", start_offset=0, end_offset=5),),
                ),
                # Valid: targets existing Span inside first paragraph
                Annotation(
                    id="valid-2",
                    type=AnnotationType.DEFINED_TERM,
                    targets=(AnnotationTarget(node_ref="#/body/0/children/1"),),
                ),
                # Valid: targets existing node
                Annotation(
                    id="valid-3",
                    type=AnnotationType.COMMENT,
                    targets=(AnnotationTarget(node_ref="#/body/2"),),
                ),
                # INVALID: #/body/99 does not exist
                Annotation(
                    id="invalid-1",
                    type=AnnotationType.REDACTION,
                    targets=(AnnotationTarget(node_ref="#/body/99"),),
                ),
                # INVALID: parent exists but child index is out of range
                Annotation(
                    id="invalid-2",
                    type=AnnotationType.ENTITY,
                    targets=(AnnotationTarget(node_ref="#/body/0/children/99"),),
                ),
            ),
        )

    def test_validate_returns_exactly_invalid_refs(self) -> None:
        doc = self._build_doc()
        index = NodeIndex(doc)
        invalid = index.validate_annotations()
        assert len(invalid) == 2
        assert "#/body/99" in invalid
        assert "#/body/0/children/99" in invalid

    def test_valid_annotations_resolve(self) -> None:
        doc = self._build_doc()
        index = NodeIndex(doc)
        # valid-1 targets #/body/0
        anns_body0 = index.annotations_for("#/body/0")
        assert any(a.id == "valid-1" for a in anns_body0)
        # valid-2 targets #/body/0/children/1
        anns_span = index.annotations_for("#/body/0/children/1")
        assert any(a.id == "valid-2" for a in anns_span)
        # valid-3 targets #/body/2
        anns_body2 = index.annotations_for("#/body/2")
        assert any(a.id == "valid-3" for a in anns_body2)

    def test_invalid_annotations_still_appear_in_annotations_for(self) -> None:
        """annotations_for is a string-keyed lookup; it returns annotations
        targeting a ref even if the ref doesn't point to an existing node.
        Use validate_annotations() to detect invalid refs."""
        doc = self._build_doc()
        index = NodeIndex(doc)
        # The annotation map is keyed by node_ref string, so invalid targets
        # are still retrievable — this is by design.
        anns_99 = index.annotations_for("#/body/99")
        assert len(anns_99) == 1
        assert anns_99[0].id == "invalid-1"

        anns_child_99 = index.annotations_for("#/body/0/children/99")
        assert len(anns_child_99) == 1
        assert anns_child_99[0].id == "invalid-2"

        # But a ref that was never targeted returns empty
        assert index.annotations_for("#/body/0/children/0") == []

    def test_annotation_with_multiple_targets_mixed_validity(self) -> None:
        """An annotation with one valid and one invalid target."""
        doc = ContentDocument(
            body=(Paragraph(children=(Text(value="text"),)),),
            annotations=(
                Annotation(
                    id="mixed",
                    type=AnnotationType.HIGHLIGHT,
                    targets=(
                        AnnotationTarget(node_ref="#/body/0"),  # valid
                        AnnotationTarget(node_ref="#/body/42"),  # invalid
                    ),
                ),
            ),
        )
        index = NodeIndex(doc)
        invalid = index.validate_annotations()
        assert invalid == ["#/body/42"]
        # The valid target still resolves
        anns = index.annotations_for("#/body/0")
        assert len(anns) == 1
        assert anns[0].id == "mixed"


# ---------------------------------------------------------------------------
# 5. extract_text fidelity — every text-producing node type
# ---------------------------------------------------------------------------
class TestExtractTextFidelity:
    """Build a document that exercises every text-producing node type and
    verify extract_text produces the expected concatenation."""

    def test_all_text_producing_types(self) -> None:
        """Verify extract_text with Text, Code, CodeBlock, MathBlock, Math,
        SoftBreak, LineBreak, RawInline, RawBlock all in one document."""
        doc = ContentDocument(
            body=(
                # Paragraph with Text, Code(inline), Math(inline), SoftBreak, LineBreak,
                # RawInline
                Paragraph(
                    children=(
                        Text(value="Hello"),
                        SoftBreak(),
                        Text(value="world"),
                        LineBreak(),
                        Code(value="x = 1"),
                        Text(value=" then "),
                        Math(value="E = mc^2"),
                        Text(value=" and "),
                        RawInline(format="html", value="<b>bold</b>"),
                    )
                ),
                # CodeBlock
                CodeBlock(language="python", value="def foo():\n    pass"),
                # MathBlock
                MathBlock(value="\\int_0^1 f(x) dx"),
                # RawBlock
                RawBlock(format="latex", value="\\newpage"),
            )
        )

        # Extract text from each block individually and check
        para_text = extract_text(doc.body[0])
        # "Hello" + " " (SoftBreak) + "world" + "\n" (LineBreak) + "x = 1"
        # + " then " + "E = mc^2" + " and " + "<b>bold</b>"
        assert para_text == "Hello world\nx = 1 then E = mc^2 and <b>bold</b>"

        cb_text = extract_text(doc.body[1])
        assert cb_text == "def foo():\n    pass"

        mb_text = extract_text(doc.body[2])
        assert mb_text == "\\int_0^1 f(x) dx"

        rb_text = extract_text(doc.body[3])
        assert rb_text == "\\newpage"

    def test_extract_text_empty_nodes(self) -> None:
        """Nodes with no text-producing content return empty string."""
        from kaos_content import PageBreak, ThematicBreak

        assert extract_text(ThematicBreak()) == ""
        assert extract_text(PageBreak()) == ""
        assert extract_text(Paragraph(children=())) == ""

    def test_extract_text_nested_emphasis_strong(self) -> None:
        """Text inside nested Emphasis > Strong > Text is extracted."""
        p = Paragraph(children=(Emphasis(children=(Strong(children=(Text(value="important"),)),)),))
        assert extract_text(p) == "important"

    def test_extract_text_link_with_text(self) -> None:
        """Link text is extracted, URL is not."""
        p = Paragraph(
            children=(
                Text(value="See "),
                Link(url="https://example.com", children=(Text(value="here"),)),
                Text(value="."),
            )
        )
        assert extract_text(p) == "See here."

    def test_extract_text_definition_item(self) -> None:
        """Definition item: term text + definition text are concatenated."""
        di = DefinitionItem(
            term=(Text(value="API"),),
            definitions=(
                (Paragraph(children=(Text(value="Application Programming Interface"),)),),
            ),
        )
        assert extract_text(di) == "APIApplication Programming Interface"

    def test_extract_text_table_all_cells(self) -> None:
        """extract_text on a table includes all cell text."""
        table = Table(
            head=TableSection(
                rows=(
                    Row(
                        cells=(
                            Cell(content=(Paragraph(children=(Text(value="A"),)),)),
                            Cell(content=(Paragraph(children=(Text(value="B"),)),)),
                        )
                    ),
                )
            ),
            bodies=(
                TableSection(
                    rows=(
                        Row(
                            cells=(
                                Cell(content=(Paragraph(children=(Text(value="1"),)),)),
                                Cell(content=(Paragraph(children=(Text(value="2"),)),)),
                            )
                        ),
                    )
                ),
            ),
            foot=TableSection(
                rows=(
                    Row(
                        cells=(
                            Cell(content=(Paragraph(children=(Text(value="X"),)),)),
                            Cell(content=(Paragraph(children=(Text(value="Y"),)),)),
                        )
                    ),
                )
            ),
        )
        text = extract_text(table)
        assert "A" in text
        assert "B" in text
        assert "1" in text
        assert "2" in text
        assert "X" in text
        assert "Y" in text

    def test_extract_text_consecutive_soft_breaks(self) -> None:
        """Multiple SoftBreaks produce multiple spaces."""
        p = Paragraph(
            children=(
                Text(value="a"),
                SoftBreak(),
                SoftBreak(),
                Text(value="b"),
            )
        )
        assert extract_text(p) == "a  b"

    def test_extract_text_code_inline_vs_block(self) -> None:
        """Inline Code and CodeBlock both have their value extracted."""
        code_inline = Code(value="let x = 42;")
        code_block = CodeBlock(value="fn main() {}", language="rust")
        assert extract_text(code_inline) == "let x = 42;"
        assert extract_text(code_block) == "fn main() {}"


# ---------------------------------------------------------------------------
# Integration: JSON round-trip preserves index structure
# ---------------------------------------------------------------------------
class TestRealisticRoundTrip:
    """Verify that the legal contract survives JSON serialization and
    the NodeIndex built from the restored document is identical."""

    def test_legal_contract_roundtrip(self) -> None:
        doc = _legal_contract()
        index_before = NodeIndex(doc)
        refs_before = set(index_before.refs)

        json_str = doc.model_dump_json()
        restored = ContentDocument.model_validate_json(json_str)
        index_after = NodeIndex(restored)
        refs_after = set(index_after.refs)

        assert refs_before == refs_after

        # Verify types match
        for ref in index_before.refs:
            assert type(index_before[ref]).__name__ == type(index_after[ref]).__name__

    def test_pdf_extraction_roundtrip(self) -> None:
        doc = _pdf_extraction_doc()
        index_before = NodeIndex(doc)

        restored = ContentDocument.model_validate_json(doc.model_dump_json())
        index_after = NodeIndex(restored)

        assert set(index_before.refs) == set(index_after.refs)

        # Provenance pages should match
        for page in [1, 2, 3, 4]:
            before_count = len(index_before.by_provenance_page(page))
            after_count = len(index_after.by_provenance_page(page))
            assert before_count == after_count, f"Page {page}: {before_count} != {after_count}"

    def test_multi_footnote_roundtrip(self) -> None:
        doc = _multi_footnote_doc()
        index_before = NodeIndex(doc)

        restored = ContentDocument.model_validate_json(doc.model_dump_json())
        index_after = NodeIndex(restored)

        assert set(index_before.refs) == set(index_after.refs)
        assert len(index_before) == len(index_after)
