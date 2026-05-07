"""Hypothesis fuzz tests for ``ContentDocument`` JSON deserialization.

Properties:

- Arbitrary JSON bytes either parse into a valid ``ContentDocument``
  or raise ``ValidationError`` / ``ValueError``. The model never
  produces an exception class outside that contract, never loops, and
  never crashes Python.
- A ``ContentDocument`` constructed via the public API and serialized
  to JSON parses back into an equal document.
- ``Annotation.body`` (open ``dict[str, Any]`` shape) tolerates any
  JSON-roundtrippable payload without crashing the parent document.

Pathological inputs covered include: deeply nested JSON, very long
strings, control characters, surrogate pairs, mixed-type unions across
the discriminated AST nodes.
"""

from __future__ import annotations

import contextlib
import json

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from kaos_content.model.annotation import Annotation, AnnotationTarget, AnnotationType
from kaos_content.model.attr import Provenance, SourceRef
from kaos_content.model.blocks import Heading, Paragraph
from kaos_content.model.document import ContentDocument
from kaos_content.model.inlines import Text
from kaos_content.model.metadata import DocumentMetadata

# ────────────────────────────────────────────────────────────────────
# Strategies
# ────────────────────────────────────────────────────────────────────


# JSON-compatible Python values. Limited depth so we don't run away.
_json_scalars = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(2**53), max_value=2**53),
    st.floats(allow_nan=False, allow_infinity=False),
    st.text(max_size=64),
)


def _json_values(max_leaves: int = 50) -> st.SearchStrategy[object]:
    return st.recursive(
        _json_scalars,
        lambda children: st.one_of(
            st.lists(children, max_size=8),
            st.dictionaries(st.text(max_size=16), children, max_size=8),
        ),
        max_leaves=max_leaves,
    )


# Strategy for arbitrary JSON byte strings (the wire format).
_json_bytes = _json_values().map(lambda v: json.dumps(v).encode("utf-8"))


# Strategy for syntactically broken JSON.
_broken_bytes = st.binary(max_size=512)


# ────────────────────────────────────────────────────────────────────
# Robustness — model_validate_json must never crash
# ────────────────────────────────────────────────────────────────────


@given(payload=_broken_bytes)
def test_random_bytes_either_parses_or_raises(payload: bytes) -> None:
    """``model_validate_json`` on arbitrary bytes either succeeds or
    raises ``ValidationError`` / ``ValueError``. No SegFault, no
    UnicodeError leaking through, no infinite loop."""
    with contextlib.suppress(ValidationError, ValueError):
        ContentDocument.model_validate_json(payload)


@given(payload=_json_bytes)
def test_random_json_either_parses_or_raises(payload: bytes) -> None:
    """Same as above but for syntactically-valid-JSON-but-semantically-
    arbitrary payloads."""
    with contextlib.suppress(ValidationError, ValueError):
        ContentDocument.model_validate_json(payload)


# ────────────────────────────────────────────────────────────────────
# Round-trip property
# ────────────────────────────────────────────────────────────────────


@st.composite
def _simple_documents(draw):  # type: ignore[no-untyped-def]
    """A small but realistic ContentDocument generator."""
    n_paragraphs = draw(st.integers(min_value=0, max_value=4))
    blocks = []
    for _ in range(n_paragraphs):
        children = draw(
            st.lists(
                st.text(max_size=64).map(lambda s: Text(value=s)),
                min_size=1,
                max_size=4,
            )
        )
        blocks.append(Paragraph(children=tuple(children)))
    if draw(st.booleans()):
        blocks.insert(
            0,
            Heading(
                depth=draw(st.integers(min_value=1, max_value=6)),
                children=(Text(value=draw(st.text(max_size=64))),),
            ),
        )
    title = draw(st.one_of(st.none(), st.text(max_size=64)))
    return ContentDocument(
        metadata=DocumentMetadata(title=title),
        body=tuple(blocks),
    )


@given(doc=_simple_documents())
def test_document_json_roundtrip(doc: ContentDocument) -> None:
    """Generated documents survive ``model_dump_json`` →
    ``model_validate_json`` with structural equality."""
    payload = doc.model_dump_json()
    restored = ContentDocument.model_validate_json(payload)
    assert restored == doc


# ────────────────────────────────────────────────────────────────────
# Provenance — invariants enforced after JSON deserialization
# ────────────────────────────────────────────────────────────────────


@given(
    page=st.integers(min_value=-100, max_value=100),
    confidence=st.floats(min_value=-2.0, max_value=2.0, allow_nan=False),
)
def test_provenance_constraints_enforced_on_deserialize(page: int, confidence: float) -> None:
    """Even when a ``Provenance`` is reconstructed from JSON, the
    audit-M1 field constraints fire — page>=1, confidence in [0,1]."""
    payload = json.dumps({"page": page, "confidence": confidence})
    if page >= 1 and 0.0 <= confidence <= 1.0:
        # Valid case round-trips.
        p = Provenance.model_validate_json(payload)
        assert p.page == page
        assert p.confidence == pytest.approx(confidence)
    else:
        with pytest.raises(ValidationError):
            Provenance.model_validate_json(payload)


# ────────────────────────────────────────────────────────────────────
# Annotation.body — open shape tolerates arbitrary JSON
# ────────────────────────────────────────────────────────────────────


@given(body=st.dictionaries(st.text(max_size=16), _json_values(), max_size=8))
def test_annotation_body_accepts_arbitrary_dict(body: dict[str, object]) -> None:
    """``Annotation.body`` is ``dict[str, Any]`` — any JSON-roundtrippable
    payload must be accepted and round-trip cleanly."""
    ann = Annotation(
        id="ann-1",
        type=AnnotationType.COMMENT,
        targets=(AnnotationTarget(node_ref="#/body/0"),),
        body=body,
    )
    payload = ann.model_dump_json()
    restored = Annotation.model_validate_json(payload)
    assert restored.body == body


# ────────────────────────────────────────────────────────────────────
# SourceRef — adversarial URI strings shouldn't crash
# ────────────────────────────────────────────────────────────────────


@given(uri=st.text(max_size=256))
def test_source_ref_accepts_any_string(uri: str) -> None:
    """``SourceRef.uri`` is a free string. The model accepts it; the
    URL-safety filter runs at *render* time, not at construction."""
    ref = SourceRef(uri=uri)
    payload = ref.model_dump_json()
    restored = SourceRef.model_validate_json(payload)
    assert restored.uri == uri


# ────────────────────────────────────────────────────────────────────
# Discriminated union — wrong node_type rejected
# ────────────────────────────────────────────────────────────────────


@given(bogus_type=st.text(max_size=24))
def test_unknown_block_node_type_rejected(bogus_type: str) -> None:
    """The ``Block`` discriminated union rejects unknown ``node_type``
    values rather than silently accepting them."""
    if bogus_type in {
        "paragraph",
        "heading",
        "codeblock",
        "blockquote",
        "bullet_list",
        "ordered_list",
        "list_item",
        "table",
        "thematic_break",
        "raw_block",
        "math_block",
        "definition_list",
        "definition_item",
        "page_break",
        "div",
        "figure",
        "admonition",
    }:
        return  # genuine type, would be accepted
    payload = json.dumps(
        {
            "metadata": {"title": "t"},
            "body": [{"node_type": bogus_type}],
        }
    )
    with pytest.raises(ValidationError):
        ContentDocument.model_validate_json(payload)
