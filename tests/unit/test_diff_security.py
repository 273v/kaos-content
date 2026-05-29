"""Security: revision content must be escaped when rendered as markup.

The ``markup`` serializer view wraps tracked-change content in ``<ins>`` /
``<del>`` (html) or ``{+...+}`` / ``{-...-}`` (text). Malicious content in
a ``rev-*`` span must not break out of that wrapper — otherwise a redline
rendered in a browser would be an XSS vector.
"""

from __future__ import annotations

from kaos_content.model.blocks import Paragraph
from kaos_content.model.document import ContentDocument, DocumentMetadata
from kaos_content.model.inlines import Inline, Text
from kaos_content.revision import make_inline_deletion, make_inline_insertion
from kaos_content.serializers import serialize_html, serialize_text

_XSS = "<script>alert(1)</script>"


def _doc_with(span: Inline) -> ContentDocument:
    return ContentDocument(
        metadata=DocumentMetadata(title=""),
        body=(Paragraph(children=(Text(value="lead "), span)),),
    )


class TestHtmlMarkupEscaping:
    def test_inserted_script_is_escaped(self) -> None:
        doc = _doc_with(make_inline_insertion(Text(value=_XSS), author="A", revision_id="0"))
        html = serialize_html(doc, view="markup")
        assert _XSS not in html
        assert "&lt;script&gt;" in html
        assert '<ins class="rev-ins">' in html

    def test_deleted_script_is_escaped(self) -> None:
        doc = _doc_with(make_inline_deletion(Text(value=_XSS), author="A", revision_id="0"))
        html = serialize_html(doc, view="markup")
        assert _XSS not in html
        assert "&lt;script&gt;" in html
        assert '<del class="rev-del">' in html

    def test_final_view_also_escapes_inserted_content(self) -> None:
        # The accepted (final) view still renders inserted content — escaped.
        doc = _doc_with(make_inline_insertion(Text(value=_XSS), author="A", revision_id="0"))
        html = serialize_html(doc, view="final")
        assert _XSS not in html
        assert "&lt;script&gt;" in html


class TestTextMarkupMarkers:
    def test_text_markup_wraps_without_executable_payload(self) -> None:
        doc = _doc_with(make_inline_insertion(Text(value=_XSS), author="A", revision_id="0"))
        text = serialize_text(doc, view="markup")
        # Plain text carries the literal content inside CriticMarkup markers;
        # there is no HTML execution surface, but the markers must be intact.
        assert "{+" in text and "+}" in text
        assert _XSS in text  # literal, not interpreted
