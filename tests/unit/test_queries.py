"""Tests for kaos_content.traversal.queries."""

from __future__ import annotations

from kaos_content.model.annotation import Annotation, AnnotationTarget, AnnotationType
from kaos_content.model.attr import Attr
from kaos_content.model.blocks import Div, Heading, Paragraph, Table
from kaos_content.model.document import ContentDocument, DocumentMetadata
from kaos_content.model.inlines import FootnoteRef, Image, Link, Span, Strong, Text
from kaos_content.model.table import Cell, Row, TableSection
from kaos_content.traversal import (
    find_annotations_of_type,
    find_by_class,
    find_by_kv,
    find_by_type,
    find_footnote_refs,
    find_headings,
    find_images,
    find_links,
    find_tables,
)


def _fixture() -> ContentDocument:
    return ContentDocument(
        metadata=DocumentMetadata(title="Q"),
        body=(
            Heading(depth=1, children=(Text(value="H1"),)),
            Paragraph(
                children=(
                    Text(value="Plain "),
                    Strong(children=(Text(value="bold"),)),
                    Text(value=" and "),
                    Link(url="https://a.example", children=(Text(value="linkA"),)),
                    Text(value=" and "),
                    Link(url="https://b.example", children=(Text(value="linkB"),)),
                )
            ),
            Heading(depth=2, children=(Text(value="H2"),)),
            Paragraph(
                children=(
                    Span(
                        attr=Attr(
                            classes=("rev-ins",),
                            kv={"rev:id": "1", "rev:author": "Alice"},
                        ),
                        children=(Text(value="inserted"),),
                    ),
                )
            ),
        ),
        annotations=(
            Annotation(
                id="a0",
                type=AnnotationType.COMMENT,
                targets=(AnnotationTarget(node_ref="#/body/0"),),
                body={"author": "Alice", "text": "hmm"},
            ),
            Annotation(
                id="a1",
                type=AnnotationType.TRACKED_CHANGE,
                targets=(AnnotationTarget(node_ref="#/body/3"),),
                body={"revision_id": "1", "change_type": "insertion", "author": "Alice"},
            ),
        ),
    )


def _fixture_with_valid_inline_span() -> ContentDocument:
    """Fixture where rev-ins Span is inside a Paragraph (valid AST)."""
    return ContentDocument(
        metadata=DocumentMetadata(title=""),
        body=(
            Heading(depth=1, children=(Text(value="H1"),)),
            Paragraph(
                children=(
                    Span(
                        attr=Attr(classes=("rev-ins",), kv={"rev:id": "1", "rev:author": "Alice"}),
                        children=(Text(value="ins1"),),
                    ),
                    Span(
                        attr=Attr(classes=("rev-del",), kv={"rev:id": "2", "rev:author": "Bob"}),
                        children=(Text(value="del1"),),
                    ),
                )
            ),
            Div(
                attr=Attr(classes=("speaker-notes",)),
                children=(Paragraph(children=(Text(value="note"),)),),
            ),
            Heading(depth=2, children=(Text(value="H2"),)),
            Paragraph(
                children=(
                    Link(url="https://a.example", children=(Text(value="A"),)),
                    Link(url="https://b.example", children=(Text(value="B"),)),
                    Image(src="img.png", alt="An image"),
                    FootnoteRef(identifier="1"),
                )
            ),
        ),
        annotations=(
            Annotation(
                id="ann-c",
                type=AnnotationType.COMMENT,
                targets=(AnnotationTarget(node_ref="#/body/0"),),
                body={"text": "hmm"},
            ),
            Annotation(
                id="ann-r",
                type=AnnotationType.TRACKED_CHANGE,
                targets=(),
                body={"revision_id": "1"},
            ),
        ),
    )


class TestFindByType:
    def test_headings(self) -> None:
        doc = _fixture_with_valid_inline_span()
        headings = list(find_by_type(doc, Heading))
        assert len(headings) == 2
        assert headings[0].depth == 1
        assert headings[1].depth == 2

    def test_spans(self) -> None:
        doc = _fixture_with_valid_inline_span()
        spans = list(find_by_type(doc, Span))
        assert len(spans) == 2

    def test_links(self) -> None:
        doc = _fixture_with_valid_inline_span()
        links = list(find_by_type(doc, Link))
        assert len(links) == 2

    def test_images(self) -> None:
        doc = _fixture_with_valid_inline_span()
        images = list(find_by_type(doc, Image))
        assert len(images) == 1
        assert images[0].src == "img.png"

    def test_empty_document(self) -> None:
        doc = ContentDocument(metadata=DocumentMetadata(title=""), body=())
        assert list(find_by_type(doc, Heading)) == []


class TestFindByClass:
    def test_rev_ins(self) -> None:
        doc = _fixture_with_valid_inline_span()
        matches = list(find_by_class(doc, "rev-ins"))
        assert len(matches) == 1
        assert isinstance(matches[0], Span)

    def test_rev_del(self) -> None:
        doc = _fixture_with_valid_inline_span()
        matches = list(find_by_class(doc, "rev-del"))
        assert len(matches) == 1

    def test_speaker_notes(self) -> None:
        doc = _fixture_with_valid_inline_span()
        matches = list(find_by_class(doc, "speaker-notes"))
        assert len(matches) == 1
        assert isinstance(matches[0], Div)

    def test_unknown_class(self) -> None:
        doc = _fixture_with_valid_inline_span()
        assert list(find_by_class(doc, "nonexistent")) == []


class TestFindByKv:
    def test_exact_match(self) -> None:
        doc = _fixture_with_valid_inline_span()
        matches = list(find_by_kv(doc, "rev:author", "Alice"))
        assert len(matches) == 1
        assert matches[0].attr.kv.get("rev:id") == "1"

    def test_any_value(self) -> None:
        """Passing value=None matches any value with that key."""
        doc = _fixture_with_valid_inline_span()
        matches = list(find_by_kv(doc, "rev:id"))
        assert len(matches) == 2

    def test_no_match(self) -> None:
        doc = _fixture_with_valid_inline_span()
        assert list(find_by_kv(doc, "rev:author", "Nonexistent")) == []


class TestFindAnnotations:
    def test_comment_annotations(self) -> None:
        doc = _fixture_with_valid_inline_span()
        comments = list(find_annotations_of_type(doc, AnnotationType.COMMENT))
        assert len(comments) == 1

    def test_tracked_change_annotations(self) -> None:
        doc = _fixture_with_valid_inline_span()
        tc = list(find_annotations_of_type(doc, AnnotationType.TRACKED_CHANGE))
        assert len(tc) == 1

    def test_missing_type(self) -> None:
        doc = _fixture_with_valid_inline_span()
        assert list(find_annotations_of_type(doc, AnnotationType.REDACTION)) == []


class TestDomainHelpers:
    def test_find_headings_all(self) -> None:
        doc = _fixture_with_valid_inline_span()
        assert len(list(find_headings(doc))) == 2

    def test_find_headings_filtered_by_depth(self) -> None:
        doc = _fixture_with_valid_inline_span()
        h1s = list(find_headings(doc, depth=1))
        assert len(h1s) == 1
        assert h1s[0].depth == 1

    def test_find_links(self) -> None:
        doc = _fixture_with_valid_inline_span()
        links = list(find_links(doc))
        assert len(links) == 2
        assert {link.url for link in links} == {"https://a.example", "https://b.example"}

    def test_find_images(self) -> None:
        doc = _fixture_with_valid_inline_span()
        images = list(find_images(doc))
        assert len(images) == 1
        assert images[0].alt == "An image"

    def test_find_footnote_refs(self) -> None:
        doc = _fixture_with_valid_inline_span()
        refs = list(find_footnote_refs(doc))
        assert len(refs) == 1
        assert refs[0].identifier == "1"


class TestTableTraversal:
    def test_find_tables_in_nested_structure(self) -> None:
        """find_tables must descend into all block subtrees including Divs."""
        tbl = Table(
            head=TableSection(
                rows=(Row(cells=(Cell(content=(Paragraph(children=(Text(value="h1"),)),)),)),)
            ),
        )
        doc = ContentDocument(
            metadata=DocumentMetadata(title=""),
            body=(Div(children=(tbl,)),),
        )
        tables = list(find_tables(doc))
        assert len(tables) == 1
