"""Tests for ContentDocument and DocumentMetadata."""

import pickle

from kaos_content import (
    Annotation,
    AnnotationTarget,
    AnnotationType,
    Attr,
    BlockQuote,
    BulletList,
    Caption,
    Cell,
    CodeBlock,
    ColSpec,
    ContentDocument,
    DocumentMetadata,
    Emphasis,
    Heading,
    ListItem,
    Paragraph,
    Provenance,
    Row,
    SourceRef,
    Strong,
    Table,
    TableSection,
    Text,
)


class TestDocumentMetadata:
    def test_defaults(self) -> None:
        m = DocumentMetadata()
        assert m.title is None
        assert m.authors == ()
        assert m.extra == {}

    def test_full(self) -> None:
        m = DocumentMetadata(
            title="Contract",
            authors=("Alice", "Bob"),
            date="2026-01-15",
            language="en",
            source=SourceRef(uri="file:///contract.pdf", mime_type="application/pdf"),
            document_type="contract",
            extra={"jurisdiction": "Delaware"},
        )
        assert m.title == "Contract"
        assert m.document_type == "contract"

    def test_json_roundtrip(self) -> None:
        m = DocumentMetadata(title="T", authors=("A",), date="2026-01-01")
        assert DocumentMetadata.model_validate_json(m.model_dump_json()) == m


class TestContentDocument:
    def test_empty(self) -> None:
        doc = ContentDocument()
        assert doc.body == ()
        assert doc.footnotes == {}
        assert doc.definitions == {}
        assert doc.annotations == ()

    def test_simple_document(self) -> None:
        doc = ContentDocument(
            metadata=DocumentMetadata(title="Test Doc"),
            body=(
                Heading(depth=1, children=(Text(value="Introduction"),)),
                Paragraph(children=(Text(value="Hello world."),)),
            ),
        )
        assert doc.metadata.title == "Test Doc"
        assert len(doc.body) == 2

    def test_with_footnotes(self) -> None:
        doc = ContentDocument(
            body=(Paragraph(children=(Text(value="See note."),)),),
            footnotes={
                "fn1": (Paragraph(children=(Text(value="A footnote."),)),),
            },
        )
        assert "fn1" in doc.footnotes

    def test_with_definitions(self) -> None:
        doc = ContentDocument(
            definitions={"example": "https://example.com"},
        )
        assert doc.definitions["example"] == "https://example.com"

    def test_with_annotations(self) -> None:
        doc = ContentDocument(
            body=(Paragraph(children=(Text(value="Redacted text here."),)),),
            annotations=(
                Annotation(
                    id="a1",
                    type=AnnotationType.REDACTION,
                    targets=(AnnotationTarget(node_ref="#/body/0", start_offset=0, end_offset=13),),
                    body={"reason": "PII"},
                ),
            ),
        )
        assert len(doc.annotations) == 1
        assert doc.annotations[0].type == AnnotationType.REDACTION

    def test_complex_document(self) -> None:
        """Build a realistic document with multiple node types."""
        doc = ContentDocument(
            metadata=DocumentMetadata(
                title="Sample Legal Document",
                authors=("Counsel A",),
                document_type="contract",
            ),
            body=(
                Heading(depth=1, children=(Text(value="Preamble"),)),
                Paragraph(
                    children=(
                        Text(value="This "),
                        Strong(children=(Text(value="Agreement"),)),
                        Text(value=" is entered into by the parties."),
                    )
                ),
                Heading(depth=2, children=(Text(value="Definitions"),)),
                BulletList(
                    children=(
                        ListItem(
                            children=(
                                Paragraph(
                                    children=(
                                        Emphasis(children=(Text(value="Force Majeure"),)),
                                        Text(value=" means any event beyond control."),
                                    )
                                ),
                            )
                        ),
                    )
                ),
                Heading(depth=2, children=(Text(value="Obligations"),)),
                BlockQuote(
                    children=(
                        Paragraph(children=(Text(value="The Seller shall deliver the goods."),)),
                    )
                ),
                Table(
                    caption=Caption(body=(Paragraph(children=(Text(value="Payment Schedule"),)),)),
                    head=TableSection(
                        rows=(
                            Row(
                                cells=(
                                    Cell(content=(Paragraph(children=(Text(value="Date"),)),)),
                                    Cell(content=(Paragraph(children=(Text(value="Amount"),)),)),
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
                                                Paragraph(children=(Text(value="2026-06-01"),)),
                                            )
                                        ),
                                        Cell(
                                            content=(Paragraph(children=(Text(value="$10,000"),)),)
                                        ),
                                    )
                                ),
                            )
                        ),
                    ),
                ),
                CodeBlock(language="json", value='{"clause": "indemnity"}'),
            ),
            annotations=(
                Annotation(
                    id="dt-1",
                    type=AnnotationType.DEFINED_TERM,
                    targets=(
                        AnnotationTarget(
                            node_ref="#/body/3/children/0/children/0",
                            start_offset=0,
                            end_offset=14,
                        ),
                    ),
                    body={"term": "Force Majeure"},
                ),
            ),
        )
        assert len(doc.body) == 8
        assert len(doc.annotations) == 1

    def test_json_roundtrip_empty(self) -> None:
        doc = ContentDocument()
        restored = ContentDocument.model_validate_json(doc.model_dump_json())
        assert restored == doc

    def test_json_roundtrip_full(self) -> None:
        doc = ContentDocument(
            metadata=DocumentMetadata(
                title="Round-trip Test",
                authors=("Author",),
                date="2026-03-25",
                language="en",
                source=SourceRef(uri="file:///test.md"),
                document_type="memo",
                extra={"key": "value"},
            ),
            body=(
                Heading(
                    depth=1,
                    children=(Text(value="Title"),),
                    attr=Attr(id="h1"),
                    provenance=Provenance(page=1),
                ),
                Paragraph(
                    children=(
                        Text(value="Text with "),
                        Strong(children=(Text(value="bold"),)),
                    ),
                    provenance=Provenance(page=1, char_span=(0, 20)),
                ),
                Table(
                    col_specs=(ColSpec(alignment=None),),
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
                                            row_span=1,
                                            col_span=1,
                                            content=(Paragraph(children=(Text(value="D"),)),),
                                        ),
                                    )
                                ),
                            )
                        ),
                    ),
                ),
            ),
            footnotes={"fn1": (Paragraph(children=(Text(value="Note."),)),)},
            definitions={"ref1": "https://example.com"},
            annotations=(
                Annotation(
                    id="a1",
                    type=AnnotationType.HIGHLIGHT,
                    targets=(AnnotationTarget(node_ref="#/body/1", start_offset=0, end_offset=10),),
                ),
            ),
        )
        json_str = doc.model_dump_json()
        restored = ContentDocument.model_validate_json(json_str)
        assert restored == doc

    def test_pickle_roundtrip(self) -> None:
        doc = ContentDocument(
            body=(
                Heading(depth=1, children=(Text(value="H1"),)),
                Paragraph(children=(Text(value="Body"),)),
            ),
        )
        pickled = pickle.dumps(doc)
        restored = pickle.loads(pickled)
        assert restored == doc

    def test_discriminated_union_in_body(self) -> None:
        """Verify the discriminated union deserializes different block types."""
        doc = ContentDocument(
            body=(
                Heading(depth=1, children=(Text(value="H"),)),
                Paragraph(children=(Text(value="P"),)),
                CodeBlock(value="code"),
            )
        )
        json_str = doc.model_dump_json()
        restored = ContentDocument.model_validate_json(json_str)
        assert type(restored.body[0]).__name__ == "Heading"
        assert type(restored.body[1]).__name__ == "Paragraph"
        assert type(restored.body[2]).__name__ == "CodeBlock"
