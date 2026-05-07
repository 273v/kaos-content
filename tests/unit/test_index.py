"""Tests for NodeIndex."""

import pytest

from kaos_content import (
    Annotation,
    AnnotationTarget,
    AnnotationType,
    BulletList,
    Caption,
    Cell,
    CodeBlock,
    ContentDocument,
    DefinitionItem,
    DefinitionList,
    Heading,
    Image,
    Link,
    ListItem,
    NodeIndex,
    Paragraph,
    Provenance,
    Row,
    Strong,
    Table,
    TableSection,
    Text,
)


def _sample_document() -> ContentDocument:
    """Multi-level document for index tests."""
    return ContentDocument(
        body=(
            Heading(depth=1, children=(Text(value="Title"),)),
            Paragraph(
                children=(
                    Text(value="Hello "),
                    Strong(children=(Text(value="world"),)),
                    Text(value=". See "),
                    Link(url="https://example.com", children=(Text(value="link"),)),
                    Text(value="."),
                ),
                provenance=Provenance(page=1),
            ),
            Heading(depth=2, children=(Text(value="Section A"),)),
            BulletList(
                children=(
                    ListItem(children=(Paragraph(children=(Text(value="item 1"),)),)),
                    ListItem(children=(Paragraph(children=(Text(value="item 2"),)),)),
                )
            ),
            Table(
                caption=Caption(body=(Paragraph(children=(Text(value="Table 1"),)),)),
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
                                    Cell(
                                        content=(
                                            Paragraph(
                                                children=(Text(value="Alice"),),
                                                provenance=Provenance(page=2),
                                            ),
                                        ),
                                    ),
                                    Cell(content=(Paragraph(children=(Text(value="30"),)),)),
                                )
                            ),
                        )
                    ),
                ),
            ),
            CodeBlock(language="python", value="x = 1"),
            Paragraph(
                children=(Image(src="photo.png", alt="Photo"),),
                provenance=Provenance(page=2),
            ),
        ),
        footnotes={
            "fn1": (Paragraph(children=(Text(value="A footnote."),)),),
        },
        annotations=(
            Annotation(
                id="a1",
                type=AnnotationType.HIGHLIGHT,
                targets=(AnnotationTarget(node_ref="#/body/1", start_offset=0, end_offset=5),),
            ),
            Annotation(
                id="a2",
                type=AnnotationType.DEFINED_TERM,
                targets=(AnnotationTarget(node_ref="#/body/1/children/1"),),
            ),
        ),
    )


class TestNodeIndexConstruction:
    def test_builds_without_error(self) -> None:
        doc = _sample_document()
        index = NodeIndex(doc)
        assert len(index) > 0

    def test_empty_document(self) -> None:
        doc = ContentDocument()
        index = NodeIndex(doc)
        assert len(index) == 0

    def test_node_count(self) -> None:
        doc = _sample_document()
        index = NodeIndex(doc)
        # Verify we indexed a reasonable number of nodes
        assert len(index) > 20


class TestNodeIndexLookup:
    def test_body_root(self) -> None:
        doc = _sample_document()
        index = NodeIndex(doc)
        node = index.get("#/body/0")
        assert node is not None
        assert isinstance(node, Heading)

    def test_body_paragraph(self) -> None:
        doc = _sample_document()
        index = NodeIndex(doc)
        node = index.get("#/body/1")
        assert isinstance(node, Paragraph)

    def test_inline_child(self) -> None:
        doc = _sample_document()
        index = NodeIndex(doc)
        node = index.get("#/body/1/children/0")
        assert isinstance(node, Text)
        assert node.value == "Hello "

    def test_nested_inline(self) -> None:
        doc = _sample_document()
        index = NodeIndex(doc)
        node = index.get("#/body/1/children/1/children/0")
        assert isinstance(node, Text)
        assert node.value == "world"

    def test_list_item(self) -> None:
        doc = _sample_document()
        index = NodeIndex(doc)
        node = index.get("#/body/3/children/0")
        assert isinstance(node, ListItem)

    def test_table_cell_content(self) -> None:
        doc = _sample_document()
        index = NodeIndex(doc)
        # head/rows/0/cells/0/content/0 → Paragraph with "Name"
        node = index.get("#/body/4/head/rows/0/cells/0/content/0")
        assert isinstance(node, Paragraph)

    def test_table_body_cell(self) -> None:
        doc = _sample_document()
        index = NodeIndex(doc)
        node = index.get("#/body/4/bodies/0/rows/0/cells/0/content/0")
        assert isinstance(node, Paragraph)

    def test_table_caption(self) -> None:
        doc = _sample_document()
        index = NodeIndex(doc)
        node = index.get("#/body/4/caption/body/0")
        assert isinstance(node, Paragraph)

    def test_footnote(self) -> None:
        doc = _sample_document()
        index = NodeIndex(doc)
        node = index.get("#/footnotes/fn1/0")
        assert isinstance(node, Paragraph)

    def test_not_found_returns_none(self) -> None:
        doc = _sample_document()
        index = NodeIndex(doc)
        assert index.get("#/body/999") is None

    def test_getitem_raises(self) -> None:
        doc = _sample_document()
        index = NodeIndex(doc)
        with pytest.raises(KeyError):
            index["#/body/999"]

    def test_contains(self) -> None:
        doc = _sample_document()
        index = NodeIndex(doc)
        assert "#/body/0" in index
        assert "#/body/999" not in index


class TestNodeIndexCollections:
    def test_headings(self) -> None:
        doc = _sample_document()
        index = NodeIndex(doc)
        assert len(index.headings) == 2
        assert all(isinstance(h, Heading) for h in index.headings)
        depths = [h.depth for h in index.headings]
        assert depths == [1, 2]

    def test_tables(self) -> None:
        doc = _sample_document()
        index = NodeIndex(doc)
        assert len(index.tables) == 1

    def test_images(self) -> None:
        doc = _sample_document()
        index = NodeIndex(doc)
        assert len(index.images) == 1
        assert index.images[0].src == "photo.png"

    def test_code_blocks(self) -> None:
        doc = _sample_document()
        index = NodeIndex(doc)
        assert len(index.code_blocks) == 1
        assert index.code_blocks[0].language == "python"

    def test_links(self) -> None:
        doc = _sample_document()
        index = NodeIndex(doc)
        assert len(index.links) == 1
        assert index.links[0].url == "https://example.com"

    def test_by_type(self) -> None:
        doc = _sample_document()
        index = NodeIndex(doc)
        paragraphs = index.by_type(Paragraph)
        assert len(paragraphs) > 3

    def test_by_type_empty(self) -> None:
        doc = _sample_document()
        index = NodeIndex(doc)
        from kaos_content import Admonition

        assert index.by_type(Admonition) == []


class TestNodeIndexProvenance:
    def test_by_provenance_page(self) -> None:
        doc = _sample_document()
        index = NodeIndex(doc)
        page1 = index.by_provenance_page(1)
        assert len(page1) == 1  # The paragraph with provenance page=1

    def test_by_provenance_page_2(self) -> None:
        doc = _sample_document()
        index = NodeIndex(doc)
        page2 = index.by_provenance_page(2)
        assert len(page2) == 2  # Alice paragraph in table cell + Image paragraph

    def test_by_provenance_page_missing(self) -> None:
        doc = _sample_document()
        index = NodeIndex(doc)
        assert index.by_provenance_page(99) == []


class TestNodeIndexAnnotations:
    def test_annotations_for_node(self) -> None:
        doc = _sample_document()
        index = NodeIndex(doc)
        anns = index.annotations_for("#/body/1")
        assert len(anns) == 1
        assert anns[0].id == "a1"

    def test_annotations_for_child(self) -> None:
        doc = _sample_document()
        index = NodeIndex(doc)
        anns = index.annotations_for("#/body/1/children/1")
        assert len(anns) == 1
        assert anns[0].id == "a2"

    def test_annotations_for_no_match(self) -> None:
        doc = _sample_document()
        index = NodeIndex(doc)
        assert index.annotations_for("#/body/0") == []

    def test_validate_annotations_valid(self) -> None:
        doc = _sample_document()
        index = NodeIndex(doc)
        assert index.validate_annotations() == []

    def test_validate_annotations_invalid(self) -> None:
        doc = ContentDocument(
            body=(Paragraph(children=(Text(value="x"),)),),
            annotations=(
                Annotation(
                    id="bad",
                    type=AnnotationType.HIGHLIGHT,
                    targets=(AnnotationTarget(node_ref="#/body/99"),),
                ),
            ),
        )
        index = NodeIndex(doc)
        invalid = index.validate_annotations()
        assert "#/body/99" in invalid


class TestNodeIndexJsonRoundtrip:
    def test_refs_stable_through_roundtrip(self) -> None:
        """Refs computed before and after JSON round-trip are identical."""
        doc = _sample_document()
        index_before = NodeIndex(doc)
        refs_before = set(index_before.refs)

        json_str = doc.model_dump_json()
        restored = ContentDocument.model_validate_json(json_str)
        index_after = NodeIndex(restored)
        refs_after = set(index_after.refs)

        assert refs_before == refs_after

    def test_node_types_match_after_roundtrip(self) -> None:
        doc = _sample_document()
        index_before = NodeIndex(doc)

        restored = ContentDocument.model_validate_json(doc.model_dump_json())
        index_after = NodeIndex(restored)

        for ref in index_before.refs:
            node_before = index_before[ref]
            node_after = index_after[ref]
            assert type(node_before).__name__ == type(node_after).__name__


class TestNodeIndexDefinitionList:
    def test_definition_item_refs(self) -> None:
        doc = ContentDocument(
            body=(
                DefinitionList(
                    children=(
                        DefinitionItem(
                            term=(Text(value="Term"),),
                            definitions=(
                                (Paragraph(children=(Text(value="Def 1"),)),),
                                (Paragraph(children=(Text(value="Def 2"),)),),
                            ),
                        ),
                    )
                ),
            )
        )
        index = NodeIndex(doc)
        # DL → #/body/0
        # DI → #/body/0/children/0
        # term Text → #/body/0/children/0/term/0
        # def 1 para → #/body/0/children/0/definitions/0/0
        # def 2 para → #/body/0/children/0/definitions/1/0
        assert index.get("#/body/0/children/0/term/0") is not None
        assert isinstance(index["#/body/0/children/0/term/0"], Text)
        assert index.get("#/body/0/children/0/definitions/0/0") is not None
        assert index.get("#/body/0/children/0/definitions/1/0") is not None


class TestNodeIndexCaptionShort:
    def test_caption_short_indexed(self) -> None:
        doc = ContentDocument(
            body=(
                Table(
                    caption=Caption(
                        short=(Text(value="T1"),),
                        body=(Paragraph(children=(Text(value="Table 1"),)),),
                    ),
                ),
            )
        )
        index = NodeIndex(doc)
        node = index.get("#/body/0/caption/short/0")
        assert isinstance(node, Text)
        assert node.value == "T1"


class TestNodeIndexTableFoot:
    def test_foot_indexed(self) -> None:
        doc = ContentDocument(
            body=(
                Table(
                    foot=TableSection(
                        rows=(
                            Row(
                                cells=(Cell(content=(Paragraph(children=(Text(value="total"),)),)),)
                            ),
                        )
                    ),
                ),
            )
        )
        index = NodeIndex(doc)
        node = index.get("#/body/0/foot/rows/0/cells/0/content/0")
        assert isinstance(node, Paragraph)
