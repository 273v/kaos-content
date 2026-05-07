"""Tests for tracked-changes view modes in serializers.

Three views are supported on ``serialize_text``, ``serialize_markdown``,
and ``serialize_html``:

- ``"final"`` (default) — accepted version, rev-del and rev-move-from skipped
- ``"original"`` — pre-change version, rev-ins and rev-move-to skipped
- ``"markup"`` — both versions, with visual differentiation
"""

from __future__ import annotations

from typing import Literal

import pytest

from kaos_content.model.attr import Attr
from kaos_content.model.blocks import Div, Paragraph
from kaos_content.model.document import ContentDocument, DocumentMetadata
from kaos_content.model.inlines import Span, Text
from kaos_content.serializers.html import serialize_html
from kaos_content.serializers.markdown import serialize_markdown
from kaos_content.serializers.text import serialize_text

type _ViewMode = Literal["final", "original", "markup"]


def _inline_redline() -> ContentDocument:
    """Build a document with an inline rev-del/rev-ins pair.

    "The deadline is {-Monday-}{+Friday+}."
    """
    return ContentDocument(
        metadata=DocumentMetadata(title="test"),
        body=(
            Paragraph(
                children=(
                    Text(value="The deadline is "),
                    Span(
                        attr=Attr(
                            classes=("rev-del",),
                            kv={"rev:id": "0", "rev:author": "Alice"},
                        ),
                        children=(Text(value="Monday"),),
                    ),
                    Span(
                        attr=Attr(
                            classes=("rev-ins",),
                            kv={"rev:id": "1", "rev:author": "Alice"},
                        ),
                        children=(Text(value="Friday"),),
                    ),
                    Text(value="."),
                ),
            ),
        ),
    )


def _block_redline() -> ContentDocument:
    """A paragraph inserted as a whole block."""
    return ContentDocument(
        metadata=DocumentMetadata(title="test"),
        body=(
            Paragraph(children=(Text(value="Original paragraph."),)),
            Div(
                attr=Attr(classes=("rev-ins",), kv={"rev:id": "2", "rev:author": "Bob"}),
                children=(Paragraph(children=(Text(value="New inserted paragraph."),)),),
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Text serializer
# ---------------------------------------------------------------------------


class TestTextViewModes:
    def test_default_is_final(self) -> None:
        doc = _inline_redline()
        assert serialize_text(doc) == serialize_text(doc, view="final")

    def test_final_inline(self) -> None:
        out = serialize_text(_inline_redline(), view="final")
        assert "Friday" in out
        assert "Monday" not in out

    def test_original_inline(self) -> None:
        out = serialize_text(_inline_redline(), view="original")
        assert "Monday" in out
        assert "Friday" not in out

    def test_markup_inline(self) -> None:
        out = serialize_text(_inline_redline(), view="markup")
        assert "{-Monday-}" in out
        assert "{+Friday+}" in out

    def test_final_block(self) -> None:
        out = serialize_text(_block_redline(), view="final")
        assert "Original paragraph" in out
        assert "New inserted paragraph" in out

    def test_original_block(self) -> None:
        out = serialize_text(_block_redline(), view="original")
        assert "Original paragraph" in out
        assert "New inserted paragraph" not in out


# ---------------------------------------------------------------------------
# Markdown serializer
# ---------------------------------------------------------------------------


class TestMarkdownViewModes:
    def test_default_is_final(self) -> None:
        doc = _inline_redline()
        assert serialize_markdown(doc) == serialize_markdown(doc, view="final")

    def test_final_inline(self) -> None:
        out = serialize_markdown(_inline_redline(), view="final")
        assert "Friday" in out
        assert "Monday" not in out

    def test_original_inline(self) -> None:
        out = serialize_markdown(_inline_redline(), view="original")
        assert "Monday" in out
        assert "Friday" not in out

    def test_markup_inline(self) -> None:
        out = serialize_markdown(_inline_redline(), view="markup")
        assert "<del>Monday</del>" in out
        assert "<ins>Friday</ins>" in out


# ---------------------------------------------------------------------------
# HTML serializer
# ---------------------------------------------------------------------------


class TestHtmlViewModes:
    def test_default_is_final(self) -> None:
        doc = _inline_redline()
        assert serialize_html(doc) == serialize_html(doc, view="final")

    def test_final_inline(self) -> None:
        out = serialize_html(_inline_redline(), view="final")
        assert "Friday" in out
        assert "Monday" not in out

    def test_original_inline(self) -> None:
        out = serialize_html(_inline_redline(), view="original")
        assert "Monday" in out
        assert "Friday" not in out

    def test_markup_inline(self) -> None:
        out = serialize_html(_inline_redline(), view="markup")
        assert '<del class="rev-del">Monday</del>' in out
        assert '<ins class="rev-ins">Friday</ins>' in out

    def test_original_block(self) -> None:
        out = serialize_html(_block_redline(), view="original")
        assert "Original paragraph" in out
        assert "New inserted paragraph" not in out


# ---------------------------------------------------------------------------
# Sanity: view modes don't touch unrelated content
# ---------------------------------------------------------------------------


class TestNoRegression:
    """Documents without rev-* nodes must render identically across views."""

    def _plain_doc(self) -> ContentDocument:
        return ContentDocument(
            metadata=DocumentMetadata(title=""),
            body=(
                Paragraph(children=(Text(value="Just plain text."),)),
                Paragraph(children=(Text(value="Second paragraph."),)),
            ),
        )

    @pytest.mark.parametrize("view", ["final", "original", "markup"])
    def test_plain_text(self, view: _ViewMode) -> None:
        doc = self._plain_doc()
        assert serialize_text(doc, view=view) == serialize_text(doc, view="final")

    @pytest.mark.parametrize("view", ["final", "original", "markup"])
    def test_plain_markdown(self, view: _ViewMode) -> None:
        doc = self._plain_doc()
        assert serialize_markdown(doc, view=view) == serialize_markdown(doc, view="final")

    @pytest.mark.parametrize("view", ["final", "original", "markup"])
    def test_plain_html(self, view: _ViewMode) -> None:
        doc = self._plain_doc()
        assert serialize_html(doc, view=view) == serialize_html(doc, view="final")
