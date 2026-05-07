"""Hypothesis fuzz tests for the safe-by-default serializer contract.

Properties under test:

- ``serialize_html(allow_raw_html=False)`` and
  ``serialize_markdown(allow_raw_html=False)`` never emit a live
  ``javascript:``, ``data:``, ``vbscript:``, or ``file:`` URL in any
  ``href`` or ``src`` attribute regardless of how it was smuggled
  into a ``Link.url`` or ``Image.src``.
- Raw HTML/MathML/SVG block content is *stripped* (not passed through)
  by default. The literal ``<script>`` token never appears in the
  serialized output.
- The serializers don't crash on adversarial input. Anything they accept
  through the AST round-trips to a string.
"""

from __future__ import annotations

import re

from hypothesis import given
from hypothesis import strategies as st

from kaos_content._security import UNSAFE_SCHEMES
from kaos_content.model.attr import SourceRef
from kaos_content.model.blocks import Paragraph, RawBlock
from kaos_content.model.document import ContentDocument
from kaos_content.model.inlines import Image, Link, RawInline, Text
from kaos_content.model.metadata import DocumentMetadata
from kaos_content.serializers.html import serialize_html
from kaos_content.serializers.markdown import serialize_markdown

_unsafe_scheme = st.sampled_from(sorted(UNSAFE_SCHEMES))

# Strategy for adversarial URL strings — combines case, whitespace,
# entity encoding, and percent encoding around an unsafe scheme.
_url_payload = st.text(max_size=64)


@st.composite
def _adversarial_url(draw):  # type: ignore[no-untyped-def]
    scheme = draw(_unsafe_scheme)
    payload = draw(_url_payload)
    # Build the scheme one character at a time, optionally entity-encoding
    # each char. ``parts`` contains *strings* (of length 1 or longer for
    # entity-encoded chars), so don't re-call ord() on them.
    parts: list[str] = []
    encode_entities = draw(st.booleans())
    for c in scheme:
        if encode_entities and draw(st.booleans()):
            parts.append(f"&#{ord(c)};")
        else:
            parts.append(c)
    if draw(st.booleans()):
        # Insert whitespace between adjacent scheme parts.
        ws = draw(st.sampled_from(["\t", "\n", "\r", " ", "\x0b"]))
        parts = [p + ws if i < len(parts) - 1 else p for i, p in enumerate(parts)]
    scheme_part = "".join(parts)
    if draw(st.booleans()):
        return f"{scheme_part}%3A{payload}"
    return f"{scheme_part}:{payload}"


def _wrap(inline) -> ContentDocument:
    """Wrap an inline in a minimal document for serialization."""
    return ContentDocument(
        metadata=DocumentMetadata(title="t"),
        body=(Paragraph(children=(inline,)),),
    )


def _wrap_block(block) -> ContentDocument:
    return ContentDocument(metadata=DocumentMetadata(title="t"), body=(block,))


# ────────────────────────────────────────────────────────────────────
# Link.url / Image.src under HTML serializer
# ────────────────────────────────────────────────────────────────────

# Pattern that detects a *live* unsafe URL surviving in an attribute
# value. Forensic ``data-unsafe-url`` is allowed (intentional).
_LIVE_UNSAFE_HREF_RE = re.compile(
    r'\s(?:href|src)\s*=\s*"(?:[^"]*\b)(?:javascript|data|vbscript|file)\s*:',
    re.IGNORECASE,
)


@given(url=_adversarial_url())
def test_html_link_url_neutered(url: str) -> None:
    doc = _wrap(Link(url=url, children=(Text(value="x"),)))
    out = serialize_html(doc, allow_raw_html=False)
    assert _LIVE_UNSAFE_HREF_RE.search(out) is None, f"unsafe URL leaked: {out!r}"


@given(url=_adversarial_url())
def test_html_image_src_neutered(url: str) -> None:
    doc = _wrap(Image(src=url))
    out = serialize_html(doc, allow_raw_html=False)
    assert _LIVE_UNSAFE_HREF_RE.search(out) is None, f"unsafe URL leaked: {out!r}"


@given(url=_adversarial_url())
def test_markdown_link_url_neutered(url: str) -> None:
    doc = _wrap(Link(url=url, children=(Text(value="x"),)))
    out = serialize_markdown(doc, allow_raw_html=False)
    # In markdown, ``[x](url)`` — the URL is between parens. Assert no
    # unsafe scheme appears as the start of a parenthesized URL.
    for scheme in UNSAFE_SCHEMES:
        # Tolerate the scheme appearing as plain text outside any link
        # syntax. Only a parenthesized URL (the link target) is risky.
        live_link_re = re.compile(rf"]\(\s*{re.escape(scheme)}\s*:", re.IGNORECASE)
        assert live_link_re.search(out) is None, (
            f"unsafe URL leaked into markdown link target: {out!r}"
        )


@given(url=_adversarial_url())
def test_markdown_image_src_neutered(url: str) -> None:
    doc = _wrap(Image(src=url))
    out = serialize_markdown(doc, allow_raw_html=False)
    for scheme in UNSAFE_SCHEMES:
        live_img_re = re.compile(rf"!\[[^\]]*]\(\s*{re.escape(scheme)}\s*:", re.IGNORECASE)
        assert live_img_re.search(out) is None, f"unsafe URL leaked into markdown image: {out!r}"


# ────────────────────────────────────────────────────────────────────
# Raw HTML stripping
# ────────────────────────────────────────────────────────────────────

# Pathological raw-HTML payloads — each has produced an XSS in the wild.
_RAW_HTML_PAYLOADS = [
    "<script>alert(1)</script>",
    "<img src=x onerror=alert(1)>",
    "<svg/onload=alert(1)>",
    "<iframe src=javascript:alert(1)></iframe>",
    "<a href=javascript:alert(1)>x</a>",
    "<style>@import 'evil.css';</style>",
    "<meta http-equiv=refresh content='0;url=javascript:alert(1)'>",
    "<object data=javascript:alert(1)>",
    "<details ontoggle=alert(1) open>",
    "<form action=javascript:alert(1)><input type=submit>",
]


@given(payload=st.sampled_from(_RAW_HTML_PAYLOADS))
def test_raw_html_block_stripped_in_html(payload: str) -> None:
    doc = _wrap_block(RawBlock(format="html", value=payload))
    out = serialize_html(doc, allow_raw_html=False)
    # The literal payload must not appear verbatim. We accept the
    # stripped-marker comment.
    assert payload not in out
    assert "<script" not in out.lower()
    # Active onerror/onload handlers must not appear.
    assert "onerror=" not in out.lower()
    assert "onload=" not in out.lower()


@given(payload=st.sampled_from(_RAW_HTML_PAYLOADS))
def test_raw_html_block_stripped_in_markdown(payload: str) -> None:
    doc = _wrap_block(RawBlock(format="html", value=payload))
    out = serialize_markdown(doc, allow_raw_html=False)
    assert payload not in out
    assert "<script" not in out.lower()


@given(payload=st.sampled_from(_RAW_HTML_PAYLOADS))
def test_raw_inline_stripped_in_html(payload: str) -> None:
    doc = _wrap(RawInline(format="html", value=payload))
    out = serialize_html(doc, allow_raw_html=False)
    assert payload not in out
    assert "<script" not in out.lower()


# ────────────────────────────────────────────────────────────────────
# Safe URLs are NOT mangled
# ────────────────────────────────────────────────────────────────────


@given(
    scheme=st.sampled_from(["https", "http", "mailto", "ftp", "tel"]),
    rest=st.text(
        alphabet=st.characters(
            min_codepoint=0x21,
            max_codepoint=0x7E,
            blacklist_characters='"<>',
        ),
        max_size=64,
    ),
)
def test_safe_urls_preserved_in_html(scheme: str, rest: str) -> None:
    """Whitelisted schemes survive the serializer — we must not be
    over-eager and mangle legitimate links."""
    url = f"{scheme}://example.com/{rest}"
    doc = _wrap(Link(url=url, children=(Text(value="ok"),)))
    out = serialize_html(doc, allow_raw_html=False)
    # The href contains the scheme. We don't compare the URL byte-for-byte
    # because the serializer HTML-escapes some chars; we just assert
    # the href landed in the output (not replaced with #).
    assert f'href="{scheme}' in out, f"safe URL was rewritten: {out!r}"


# ────────────────────────────────────────────────────────────────────
# allow_raw_html=True passes content through (escape hatch)
# ────────────────────────────────────────────────────────────────────


@given(payload=st.sampled_from(_RAW_HTML_PAYLOADS))
def test_raw_html_passthrough_when_explicitly_allowed(payload: str) -> None:
    """The opt-in escape hatch is honoured — content authors who pass
    ``allow_raw_html=True`` get the raw content. We assert the contract
    flips, not that they should use it."""
    doc = _wrap_block(RawBlock(format="html", value=payload))
    out = serialize_html(doc, allow_raw_html=True)
    assert payload in out


# ────────────────────────────────────────────────────────────────────
# No-crash property
# ────────────────────────────────────────────────────────────────────


@given(url=st.text(max_size=256))
def test_serialize_never_crashes_on_arbitrary_url(url: str) -> None:
    """Any URL that survives Pydantic validation must not crash the serializer."""
    doc = _wrap(Link(url=url, children=(Text(value="x"),)))
    serialize_html(doc, allow_raw_html=False)
    serialize_markdown(doc, allow_raw_html=False)


@given(text=st.text(max_size=256))
def test_text_value_serializes_safely(text: str) -> None:
    """Plain Text content with arbitrary chars never produces a live
    ``<script>`` tag in the output."""
    doc = _wrap(Text(value=text))
    html_out = serialize_html(doc, allow_raw_html=False)
    md_out = serialize_markdown(doc, allow_raw_html=False)
    # ``<`` and ``>`` in Text must be HTML-escaped
    assert "<script" not in html_out.lower()
    # In markdown the contract is weaker (markdown allows inline HTML
    # by spec) but our safe-by-default serializer should still escape
    # angle brackets in plain Text. Accept either escaping or absence.
    assert "<script>" not in md_out.lower() or text.lower().count("<script>") == 0


# ────────────────────────────────────────────────────────────────────
# Source URI provenance is also URL-filtered
# ────────────────────────────────────────────────────────────────────


@given(url=_adversarial_url())
def test_provenance_source_url_does_not_leak(url: str) -> None:
    """A SourceRef on document metadata can carry an arbitrary URI.
    The serializer should not surface it as a live href."""
    doc = ContentDocument(
        metadata=DocumentMetadata(
            title="t",
            source=SourceRef(uri=url),
        ),
        body=(Paragraph(children=(Text(value="hi"),)),),
    )
    out = serialize_html(doc, allow_raw_html=False)
    assert _LIVE_UNSAFE_HREF_RE.search(out) is None
