"""Format-agnostic differencing: diff two Markdown documents.

The redline engine works on any ``ContentDocument``, not just DOCX. This
parses two Markdown sources, diffs them, and checks the round-trip plus
the markup rendering — exercising "diff/comparison via kaos-content" for
a non-Office format. Requires the ``[markdown]`` extra.
"""

from __future__ import annotations

import pytest

pytest.importorskip("markdown_it", reason="requires the [markdown] extra")

from kaos_content import compare_documents
from kaos_content.model.document import ContentDocument
from kaos_content.parsers import parse_markdown
from kaos_content.revision import Revisions, accept_all, reject_all
from kaos_content.serializers import serialize_markdown, serialize_text
from kaos_content.traversal.visitor import extract_text

_A = "# Title\n\nThe agreement is governed by Delaware law.\n"
_B = "# Title\n\nThe agreement is governed by New York law.\n"


def _body_text(doc: ContentDocument) -> str:
    return "\n".join(extract_text(b) for b in doc.body)


def test_markdown_diff_round_trips() -> None:
    a, b = parse_markdown(_A), parse_markdown(_B)
    redline = compare_documents(a, b, author="MD")
    assert Revisions.from_document(redline)
    assert _body_text(accept_all(redline)) == _body_text(b)
    assert _body_text(reject_all(redline)) == _body_text(a)


def test_markdown_diff_markup_shows_word_change() -> None:
    redline = compare_documents(parse_markdown(_A), parse_markdown(_B), author="MD")
    text_markup = serialize_text(redline, view="markup")
    assert "{-Delaware-}" in text_markup
    assert "{+New York+}" in text_markup
    md_markup = serialize_markdown(redline, view="markup")
    assert "<del>Delaware</del>" in md_markup
    assert "<ins>New York</ins>" in md_markup


def test_markdown_diff_final_renders_revised() -> None:
    a, b = parse_markdown(_A), parse_markdown(_B)
    redline = compare_documents(a, b, author="MD")
    assert "New York" in serialize_markdown(redline, view="final")
    assert "Delaware" not in serialize_markdown(redline, view="final")
