"""Tests for kaos_content.transforms."""

from __future__ import annotations

from kaos_content.model.blocks import Paragraph
from kaos_content.model.document import ContentDocument, DocumentMetadata
from kaos_content.model.inlines import Text
from kaos_content.revision import accept_all
from kaos_content.transforms import DocumentTransform, apply, compose


def _doc(text: str = "hello") -> ContentDocument:
    return ContentDocument(
        metadata=DocumentMetadata(title=""),
        body=(Paragraph(children=(Text(value=text),)),),
    )


def _append(marker: str) -> DocumentTransform:
    """Build a transform that appends a paragraph with ``marker`` text."""

    def _run(doc: ContentDocument) -> ContentDocument:
        new_body = (*doc.body, Paragraph(children=(Text(value=marker),)))
        return doc.model_copy(update={"body": new_body})

    return _run


def _body_texts(doc: ContentDocument) -> list[str]:
    texts: list[str] = []
    for block in doc.body:
        assert isinstance(block, Paragraph)
        first_child = block.children[0]
        assert isinstance(first_child, Text)
        texts.append(first_child.value)
    return texts


class TestCompose:
    def test_empty_compose_is_identity(self) -> None:
        doc = _doc()
        pipeline = compose()
        assert pipeline(doc) is doc

    def test_single_transform_returned_as_is(self) -> None:
        t = _append("X")
        assert compose(t) is t

    def test_two_transforms_apply_in_order(self) -> None:
        doc = _doc()
        pipeline = compose(_append("A"), _append("B"))
        result = pipeline(doc)
        assert _body_texts(result) == ["hello", "A", "B"]

    def test_three_transforms(self) -> None:
        doc = _doc()
        pipeline = compose(_append("A"), _append("B"), _append("C"))
        result = pipeline(doc)
        assert _body_texts(result) == ["hello", "A", "B", "C"]

    def test_input_is_not_mutated(self) -> None:
        doc = _doc()
        orig_body_len = len(doc.body)
        pipeline = compose(_append("X"))
        _ = pipeline(doc)
        assert len(doc.body) == orig_body_len


class TestApply:
    def test_apply_equivalent_to_compose(self) -> None:
        doc = _doc()
        via_compose = compose(_append("A"), _append("B"))(doc)
        via_apply = apply(doc, _append("A"), _append("B"))
        assert _body_texts(via_compose) == _body_texts(via_apply)

    def test_apply_with_no_transforms_returns_input(self) -> None:
        doc = _doc()
        assert apply(doc) is doc


class TestProtocolCompatibility:
    def test_accept_all_matches_protocol(self) -> None:
        """revision.accept_all conforms to DocumentTransform."""
        # runtime_checkable Protocols support isinstance on callables
        assert isinstance(accept_all, DocumentTransform)

    def test_compose_accepts_accept_all(self) -> None:
        """Smoke: accept_all composes with other transforms."""
        doc = _doc()
        pipeline = compose(accept_all, _append("X"))
        result = pipeline(doc)
        assert _body_texts(result) == ["hello", "X"]
