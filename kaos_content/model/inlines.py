"""Inline AST node types."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from kaos_content.model.node import BaseInline


class Text(BaseInline):
    """Plain text leaf node."""

    node_type: Literal["text"] = "text"
    value: str


class Emphasis(BaseInline):
    """Emphasized (italic) content."""

    node_type: Literal["emphasis"] = "emphasis"
    children: tuple[Inline, ...]


class Strong(BaseInline):
    """Strong (bold) content."""

    node_type: Literal["strong"] = "strong"
    children: tuple[Inline, ...]


class Strikethrough(BaseInline):
    """Struck-through content."""

    node_type: Literal["strikethrough"] = "strikethrough"
    children: tuple[Inline, ...]


class Code(BaseInline):
    """Inline code."""

    node_type: Literal["code"] = "code"
    value: str


class Link(BaseInline):
    """Hyperlink."""

    node_type: Literal["link"] = "link"
    url: str
    title: str | None = None
    children: tuple[Inline, ...]


class Image(BaseInline):
    """Image reference. ``src`` may be an artifact URI.

    Optional dimensions:
    - ``width`` and ``height`` are in points (1/72 inch) by default. For
      HTML-native content use pixels; the ``Attr.kv["unit"]`` key can
      override ("px" | "pt" | "em"). Readers should set both to preserve
      aspect ratio on round-trip.
    """

    node_type: Literal["image"] = "image"
    src: str
    alt: str | None = None
    title: str | None = None
    # Width/height must be positive when set. Negative or zero
    # dimensions are structurally invalid (no display backend renders
    # them) and were silently accepted before 0.1.0a1.
    width: float | None = Field(default=None, gt=0)
    height: float | None = Field(default=None, gt=0)


class FootnoteRef(BaseInline):
    """Reference to a footnote. Content in document.footnotes[identifier]."""

    node_type: Literal["footnote_ref"] = "footnote_ref"
    identifier: str


class Citation(BaseInline):
    """Citation reference. Payload in Attr or annotation layer."""

    node_type: Literal["citation"] = "citation"
    identifiers: tuple[str, ...]
    children: tuple[Inline, ...]


class Math(BaseInline):
    """Inline math (LaTeX)."""

    node_type: Literal["math"] = "math"
    value: str


class RawInline(BaseInline):
    """Raw inline content in a specific format."""

    node_type: Literal["raw_inline"] = "raw_inline"
    format: str
    value: str


class LineBreak(BaseInline):
    """Hard line break."""

    node_type: Literal["line_break"] = "line_break"


class SoftBreak(BaseInline):
    """Soft line break."""

    node_type: Literal["soft_break"] = "soft_break"


class Span(BaseInline):
    """Generic inline container. Carries Attr for domain-specific semantics."""

    node_type: Literal["span"] = "span"
    children: tuple[Inline, ...]


class Superscript(BaseInline):
    """Superscript content."""

    node_type: Literal["superscript"] = "superscript"
    children: tuple[Inline, ...]


class Subscript(BaseInline):
    """Subscript content."""

    node_type: Literal["subscript"] = "subscript"
    children: tuple[Inline, ...]


class Underline(BaseInline):
    """Underlined content."""

    node_type: Literal["underline"] = "underline"
    children: tuple[Inline, ...]


Inline = Annotated[
    Text
    | Emphasis
    | Strong
    | Strikethrough
    | Code
    | Link
    | Image
    | FootnoteRef
    | Citation
    | Math
    | RawInline
    | LineBreak
    | SoftBreak
    | Span
    | Superscript
    | Subscript
    | Underline,
    Field(discriminator="node_type"),
]
