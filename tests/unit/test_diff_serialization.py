"""Integration: redline engine output through the revision-aware serializers.

``compare_documents`` produces ``rev-*`` markup; the markdown / text / html
serializers accept a ``view`` parameter (``final`` / ``original`` /
``markup``). Together they give a serialization-level round-trip:
``serialize(compare(a, b), view="final")`` renders ``b`` and
``view="original"`` renders ``a``, while ``markup`` shows both sides.
"""

from __future__ import annotations

from kaos_content import compare_documents
from kaos_content.model.blocks import Paragraph
from kaos_content.model.document import ContentDocument, DocumentMetadata
from kaos_content.model.inlines import Text
from kaos_content.serializers import serialize_html, serialize_markdown, serialize_text


def _doc(*paras: str) -> ContentDocument:
    return ContentDocument(
        metadata=DocumentMetadata(title=""),
        body=tuple(Paragraph(children=(Text(value=p),)) for p in paras),
    )


_ORIGINAL = _doc("The quick brown fox jumps.")
_REVISED = _doc("The quick red fox leaps.")


class TestTextSerializerViews:
    def test_final_renders_revised(self) -> None:
        redline = compare_documents(_ORIGINAL, _REVISED)
        assert serialize_text(redline, view="final").strip() == serialize_text(_REVISED).strip()

    def test_original_renders_original(self) -> None:
        redline = compare_documents(_ORIGINAL, _REVISED)
        assert serialize_text(redline, view="original").strip() == serialize_text(_ORIGINAL).strip()

    def test_markup_shows_both_sides(self) -> None:
        redline = compare_documents(_ORIGINAL, _REVISED)
        markup = serialize_text(redline, view="markup")
        # CriticMarkup-style del/ins markers carry both old and new text.
        assert "{-brown-}" in markup
        assert "{+red+}" in markup


class TestMarkdownSerializerViews:
    def test_final_and_original(self) -> None:
        redline = compare_documents(_ORIGINAL, _REVISED)
        assert (
            serialize_markdown(redline, view="final").strip()
            == serialize_markdown(_REVISED).strip()
        )
        assert (
            serialize_markdown(redline, view="original").strip()
            == serialize_markdown(_ORIGINAL).strip()
        )

    def test_markup_wraps_ins_and_del(self) -> None:
        redline = compare_documents(_ORIGINAL, _REVISED)
        markup = serialize_markdown(redline, view="markup")
        assert "<del>brown</del>" in markup
        assert "<ins>red</ins>" in markup


class TestHtmlSerializerViews:
    def test_final_and_original_text(self) -> None:
        redline = compare_documents(_ORIGINAL, _REVISED)
        assert "red" in serialize_html(redline, view="final")
        assert "brown" not in serialize_html(redline, view="final")
        assert "brown" in serialize_html(redline, view="original")
        assert "red" not in serialize_html(redline, view="original")

    def test_markup_uses_classed_ins_del(self) -> None:
        redline = compare_documents(_ORIGINAL, _REVISED)
        markup = serialize_html(redline, view="markup")
        assert '<del class="rev-del">brown</del>' in markup
        assert '<ins class="rev-ins">red</ins>' in markup


class TestInsertionDeletionRendering:
    def test_inserted_paragraph_appears_only_in_final(self) -> None:
        original = _doc("Stays.")
        revised = _doc("Stays.", "Newly inserted clause.")
        redline = compare_documents(original, revised)
        assert "Newly inserted clause." in serialize_text(redline, view="final")
        assert "Newly inserted clause." not in serialize_text(redline, view="original")

    def test_deleted_paragraph_appears_only_in_original(self) -> None:
        original = _doc("Stays.", "To be removed clause.")
        revised = _doc("Stays.")
        redline = compare_documents(original, revised)
        assert "To be removed clause." in serialize_text(redline, view="original")
        assert "To be removed clause." not in serialize_text(redline, view="final")
