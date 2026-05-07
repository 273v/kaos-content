"""Parse HTML into a ContentDocument AST.

This module walks an lxml HTML element tree and produces a
``ContentDocument`` composed of Block and Inline AST nodes from
``kaos_content.model``.

Requires the ``html`` optional dependency (``lxml>=5.0.0``).

The public API is :func:`parse_html`, which converts raw HTML into a
``ContentDocument``.  The lower-level walker functions
(``process_children_as_blocks``, ``process_inlines``, etc.) are also
exported so that ``kaos-web`` can compose readability extraction on top.
"""

from __future__ import annotations

import contextlib
import contextvars
import re
import threading
import uuid
from urllib.parse import urljoin

from kaos_core.logging import get_logger
from lxml import html as lxml_html

# lxml ships partial type stubs; ty can't resolve `lxml.etree`. The runtime
# import is fine — see test_html_serializer / fuzz_html. Suppression is local
# to this one symbol so other lxml typos still surface as ty errors.
from lxml.etree import LxmlError  # ty: ignore[unresolved-import]
from lxml.html import HtmlElement

from kaos_content._security import is_safe_url as _is_safe_url
from kaos_content.model.attr import Attr, Caption, Provenance, SourceRef
from kaos_content.model.blocks import (
    Block,
    BlockQuote,
    BulletList,
    CodeBlock,
    DefinitionItem,
    DefinitionList,
    Figure,
    Heading,
    ListItem,
    OrderedList,
    Paragraph,
    Table,
    ThematicBreak,
)
from kaos_content.model.document import ContentDocument
from kaos_content.model.inlines import (
    Code,
    Emphasis,
    Image,
    Inline,
    LineBreak,
    Link,
    Span,
    Strikethrough,
    Strong,
    Subscript,
    Superscript,
    Text,
)
from kaos_content.model.metadata import DocumentMetadata
from kaos_content.model.table import Cell, Row, TableSection

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})

SKIP_TAGS = frozenset(
    {
        "script",
        "style",
        "nav",
        "footer",
        "header",
        "form",
        "iframe",
        "embed",
        "object",
        "button",
        "svg",
        "noscript",
        "template",
    }
)

INLINE_FORMATTING_TAGS = frozenset(
    {
        "strong",
        "b",
        "em",
        "i",
        "s",
        "del",
        "strike",
        "code",
        "a",
        "img",
        "br",
        "sub",
        "sup",
        "u",
        "mark",
        "abbr",
        "time",
        "span",
        "small",
    }
)

# Tags whose children are processed transparently as blocks.
TRANSPARENT_BLOCK_TAGS = frozenset({"div", "section", "article", "main", "aside", "details"})

_WS_RE = re.compile(r"[ \t\n\r]+")

# Block-level tags -- used to decide if <li> has block vs inline content.
BLOCK_LEVEL_TAGS = frozenset(
    {
        "p",
        "div",
        "blockquote",
        "pre",
        "ul",
        "ol",
        "dl",
        "table",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "hr",
        "figure",
        "section",
        "article",
    }
)
_LANG_RE = re.compile(r"\b(?:language|lang|highlight)-(\S+)")

# `_is_safe_url` is imported from kaos_content._security so the same
# canonicalisation logic is used by the HTML parser, the HTML serializer,
# and the Markdown serializer. A single fix to the filter applies
# everywhere.

# CSS classes whose elements should be filtered entirely.
# These are well-known noise patterns from major sites.
SKIP_CLASSES = frozenset(
    {
        "mw-editsection",  # Wikipedia [edit] section links
        "mw-jump-link",  # Wikipedia "jump to" navigation
        "mw-cite-backlink",  # Wikipedia citation back-links
        "sr-only",  # Bootstrap screen-reader only
        "visually-hidden",  # Modern screen-reader only
        "screen-reader-text",  # WordPress screen-reader only
        "noprint",  # Wikipedia print-hide elements
    }
)

# Link href patterns that indicate UI action controls, not content links.
_ACTION_LINK_RE = re.compile(
    r"(?:^|[?&/])"
    r"(?:vote|upvote|downvote|flag|hide|collapse|fav|unfav)"
    r"(?:[?&=]|$)",
    re.IGNORECASE,
)

# --- Inline XBRL preprocessing -----------------------------------------
#
# SEC EDGAR filings use Inline XBRL (iXBRL): standard HTML wrapped in
# ``ix:`` namespace elements.  lxml parses namespace-prefixed tags as
# ``{uri}localname`` which the AST builder doesn't recognize.  The fix:
# strip the XBRL wrapper before AST conversion.
#
# Reference: https://www.xbrl.org/Specification/inlineXBRL-part1/REC-2013-11-18/inlineXBRL-part1-REC-2013-11-18.html

_XBRL_HIDDEN_RE = re.compile(
    r"<ix:header\b[^>]*>.*?</ix:header>",
    re.DOTALL | re.IGNORECASE,
)
_XBRL_TAG_RE = re.compile(r"</?ix:[^>]*>", re.IGNORECASE)
_DISPLAY_NONE_RE = re.compile(
    r'<div\b[^>]*style\s*=\s*"[^"]*display\s*:\s*none[^"]*"[^>]*>.*?</div>',
    re.DOTALL | re.IGNORECASE,
)
_XML_DECL_RE = re.compile(r"<\?xml[^?]*\?>")
_XMLNS_RE = re.compile(r'\s+xmlns:[a-z_-]+="[^"]*"', re.IGNORECASE)


def strip_inline_xbrl(html: str) -> str:
    """Remove Inline XBRL markup from an HTML string.

    Inline XBRL (iXBRL) wraps standard HTML in ``ix:`` namespace
    elements.  This function:

    1. Removes ``<ix:header>`` blocks (XBRL metadata, not visible).
    2. Removes ``<div style="display:none">`` blocks (hidden XBRL data).
    3. Unwraps all remaining ``ix:`` tags, keeping their text content.
       For example ``<ix:nonNumeric ...>42</ix:nonNumeric>`` -> ``42``.
    4. Strips the XML declaration and XBRL namespace attributes so
       lxml can parse the result as plain HTML.

    The returned string is standard HTML suitable for AST conversion.
    """
    # Order matters: remove hidden blocks before unwrapping tags,
    # so we don't accidentally keep hidden metadata text.
    result = _XBRL_HIDDEN_RE.sub("", html)
    result = _DISPLAY_NONE_RE.sub("", result)
    result = _XBRL_TAG_RE.sub("", result)
    result = _XML_DECL_RE.sub("", result)
    result = _XMLNS_RE.sub("", result)
    return result


def looks_like_xbrl(html: str) -> bool:
    """Heuristic: does this HTML contain Inline XBRL markup?"""
    # Check the first 2000 chars for XBRL signatures.
    head = html[:2000]
    return "ix:" in head or "inlineXBRL" in head or "xbrl" in head.lower()


# Shared default Attr instance (frozen, safe to reuse).
_DEFAULT_ATTR = Attr()


def _fast_id() -> str:
    """Generate a fast unique ID (UUID4 hex -- standard, faster than UUID7)."""
    return uuid.uuid4().hex


def should_skip_element(el: HtmlElement) -> bool:
    """Check if an element should be skipped based on its CSS classes."""
    cls = el.get("class", "")
    if not cls:
        return False
    return bool(SKIP_CLASSES.intersection(cls.split()))


def _is_action_link(href: str) -> bool:
    """Check if a link href is a UI action control, not a content link."""
    if not href:
        return False
    return _ACTION_LINK_RE.search(href) is not None


def empty_document() -> ContentDocument:
    """Return an empty ContentDocument."""
    return ContentDocument.model_construct(
        metadata=DocumentMetadata.model_construct(
            title=None,
            authors=(),
            date=None,
            language=None,
            source=None,
            document_type=None,
            extra={},
        ),
        body=(),
        footnotes={},
        definitions={},
        annotations=(),
    )


# ---------------------------------------------------------------------------
# Fast node constructors -- bypass Pydantic validation for trusted code.
# Uses model_construct() which skips schema validation and deepcopy.
# ---------------------------------------------------------------------------


def mk_text(value: str) -> Text:
    return Text.model_construct(
        id=_fast_id(), attr=_DEFAULT_ATTR, provenance=None, node_type="text", value=value
    )


def mk_strong(children: tuple[Inline, ...]) -> Strong:
    return Strong.model_construct(
        id=_fast_id(), attr=_DEFAULT_ATTR, provenance=None, node_type="strong", children=children
    )


def mk_emphasis(children: tuple[Inline, ...]) -> Emphasis:
    return Emphasis.model_construct(
        id=_fast_id(), attr=_DEFAULT_ATTR, provenance=None, node_type="emphasis", children=children
    )


def mk_code(value: str) -> Code:
    return Code.model_construct(
        id=_fast_id(), attr=_DEFAULT_ATTR, provenance=None, node_type="code", value=value
    )


def mk_link(url: str, children: tuple[Inline, ...], title: str | None = None) -> Link:
    return Link.model_construct(
        id=_fast_id(),
        attr=_DEFAULT_ATTR,
        provenance=None,
        node_type="link",
        url=url,
        title=title,
        children=children,
    )


def mk_span(children: tuple[Inline, ...]) -> Span:
    """Construct a generic inline Span container.

    Used by Sec-4 (security finding #4) when stripping a dangerous-URL
    Link to preserve its visible text without an attribute marker.
    Future callers may attach Attr semantics via the Span's attr field.
    """
    return Span.model_construct(
        id=_fast_id(),
        attr=_DEFAULT_ATTR,
        provenance=None,
        node_type="span",
        children=children,
    )


def mk_image(src: str, alt: str | None = None, title: str | None = None) -> Image:
    return Image.model_construct(
        id=_fast_id(),
        attr=_DEFAULT_ATTR,
        provenance=None,
        node_type="image",
        src=src,
        alt=alt,
        title=title,
    )


def mk_linebreak() -> LineBreak:
    return LineBreak.model_construct(
        id=_fast_id(), attr=_DEFAULT_ATTR, provenance=None, node_type="line_break"
    )


def mk_paragraph(children: tuple[Inline, ...], prov: Provenance | None) -> Paragraph:
    return Paragraph.model_construct(
        id=_fast_id(),
        attr=_DEFAULT_ATTR,
        provenance=prov,
        node_type="paragraph",
        children=children,
    )


def mk_heading(depth: int, children: tuple[Inline, ...], prov: Provenance | None) -> Heading:
    return Heading.model_construct(
        id=_fast_id(),
        attr=_DEFAULT_ATTR,
        provenance=prov,
        node_type="heading",
        depth=depth,
        children=children,
    )


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------


def resolve_url(href: str, base_url: str) -> str:
    """Resolve a relative URL against a base URL.

    Fast path for common cases (absolute-path URLs like /about) avoids
    the full urljoin RFC parsing which is ~17x slower.
    """
    if not href or not base_url:
        return href
    # Already absolute
    if href.startswith(("http://", "https://", "//")):
        return href
    # Absolute-path relative (most common case): /about, /page/foo
    if href.startswith("/"):
        # Extract scheme + netloc from base
        idx = base_url.find("/", 8)  # skip past https://
        if idx > 0:
            return base_url[:idx] + href
        return base_url + href
    # Fall back to full RFC urljoin for relative paths, query strings, etc.
    return urljoin(base_url, href)


# `_is_safe_url` lives in kaos_content._security; re-imported above as
# a module-private alias so existing callers keep working.


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def collapse_whitespace(text: str) -> str:
    """Collapse runs of whitespace into single spaces.

    Uses str.split()/join() which is ~4x faster than re.sub for this pattern.
    Note: split() also strips leading/trailing whitespace, so we re-add a
    single space if the original had leading/trailing whitespace.
    """
    if not text:
        return text
    leading = text[0] in " \t\n\r"
    trailing = text[-1] in " \t\n\r"
    collapsed = " ".join(text.split())
    if leading and collapsed:
        collapsed = " " + collapsed
    if trailing and collapsed:
        collapsed = collapsed + " "
    return collapsed


def _strip_or_empty(text: str | None) -> str:
    """Return stripped text or empty string."""
    if text is None:
        return ""
    return text


def trim_inline_whitespace(inlines: list[Inline]) -> list[Inline]:
    """Strip leading/trailing whitespace from a list of inline nodes.

    Trims whitespace from leading/trailing Text nodes and removes empty ones.
    """
    # Trim leading
    while inlines and isinstance(inlines[0], Text):
        stripped = inlines[0].value.lstrip()
        if stripped:
            inlines[0] = mk_text(stripped)
            break
        inlines.pop(0)
    # Trim trailing
    while inlines and isinstance(inlines[-1], Text):
        stripped = inlines[-1].value.rstrip()
        if stripped:
            inlines[-1] = mk_text(stripped)
            break
        inlines.pop()
    return inlines


def merge_adjacent_text(inlines: list[Inline]) -> list[Inline]:
    """Merge adjacent Text nodes and adjacent same-type inline formatting nodes.

    - Adjacent Text nodes: merge and collapse double spaces.
    - Adjacent Strong+Strong, Emphasis+Emphasis, Strikethrough+Strikethrough:
      merge children into a single node.
    """
    if not inlines:
        return inlines
    result: list[Inline] = []
    for node in inlines:
        if not result:
            result.append(node)
            continue
        prev = result[-1]
        # Merge adjacent Text nodes
        if isinstance(node, Text) and isinstance(prev, Text):
            merged = _WS_RE.sub(" ", prev.value + node.value)
            result[-1] = mk_text(merged)
        # Merge adjacent Strong nodes
        elif isinstance(node, Strong) and isinstance(prev, Strong):
            result[-1] = mk_strong(prev.children + node.children)
        # Merge adjacent Emphasis nodes
        elif isinstance(node, Emphasis) and isinstance(prev, Emphasis):
            result[-1] = mk_emphasis(prev.children + node.children)
        # Merge adjacent Strikethrough nodes
        elif isinstance(node, Strikethrough) and isinstance(prev, Strikethrough):
            result[-1] = Strikethrough.model_construct(
                id=_fast_id(),
                attr=_DEFAULT_ATTR,
                provenance=None,
                node_type="strikethrough",
                children=prev.children + node.children,
            )
        else:
            result.append(node)
    return result


def is_whitespace_only_inlines(inlines: tuple[Inline, ...] | list[Inline]) -> bool:
    """Check if inline nodes contain only whitespace text."""
    for c in inlines:
        if isinstance(c, Text):
            if c.value.strip():
                return False
        else:
            return False  # Non-text node means not whitespace-only
    return True


# ---------------------------------------------------------------------------
# Language / image helpers
# ---------------------------------------------------------------------------


def _extract_language(el: HtmlElement) -> str | None:
    """Extract code language from class attribute (e.g. ``language-python``)."""
    cls = el.get("class", "")
    m = _LANG_RE.search(cls)
    if m:
        return m.group(1)
    # Also check bare class names like "python", "json".
    for c in cls.split():
        if c and c not in ("highlight", "code", "sourceCode", "source"):
            return c
    return None


def _get_image_src(el: HtmlElement) -> str | None:
    """Get image source, preferring lazy-load attributes."""
    for attr in ("data-src", "data-lazy-src", "data-original", "src"):
        val = el.get(attr)
        if val and val.strip():
            return val.strip()
    return None


# ---------------------------------------------------------------------------
# Provenance factory
# ---------------------------------------------------------------------------


_PROVENANCE_CACHE: dict[str, Provenance] = {}
_PROVENANCE_LOCK = threading.Lock()

# Per-context parser state. ContextVar (not module-global) so concurrent
# parses — under threading, asyncio, or trio — get isolated values rather
# than racing for the global slot. Reads pick up the nearest active
# scope's value via the standard ``ContextVar.get()`` semantics.
#
# Defaults: ``"kaos-content"`` extractor name (kaos-web overrides via
# :func:`extractor_scope`), and ``"code"`` for ``<pre>`` content
# interpretation (callers whose source abuses ``<pre>`` for wrapped
# prose — e.g. the Federal Register ``raw_text_url`` endpoint — switch
# to ``"prose"`` via :func:`parse_html(..., pre_content_mode="prose")`).
_extractor_name_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "kaos_content.parsers.html._extractor_name", default="kaos-content"
)
_pre_content_mode_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "kaos_content.parsers.html._pre_content_mode", default="code"
)


@contextlib.contextmanager
def extractor_scope(name: str):  # type: ignore[type-arg]
    """Context manager to set the extractor name for provenance within a scope.

    Usage::

        with extractor_scope("kaos-web"):
            doc = parse_html(html, url=url)

    Concurrent-safe: each task/thread that enters its own scope sees its
    own value (backed by :class:`contextvars.ContextVar`).
    """
    token = _extractor_name_var.set(name)
    try:
        yield
    finally:
        _extractor_name_var.reset(token)


@contextlib.contextmanager
def pre_content_scope(mode: str):  # type: ignore[type-arg]
    """Context manager to set the ``<pre>`` content mode within a scope.

    ``mode`` is ``"code"`` (default, produce ``CodeBlock``) or
    ``"prose"`` (produce blank-line-separated ``Paragraph`` blocks).

    Concurrent-safe via :class:`contextvars.ContextVar`.
    """
    token = _pre_content_mode_var.set(mode)
    try:
        yield
    finally:
        _pre_content_mode_var.reset(token)


def make_provenance(url: str, *, extractor: str | None = None) -> Provenance | None:
    """Create a Provenance for block nodes. Cached per URL (frozen, safe to share).

    If *extractor* is not specified, uses the module-level default
    (``"kaos-content"``, or whatever was set via :func:`extractor_scope`).
    """
    if not url:
        return None
    ext = extractor if extractor is not None else _extractor_name_var.get()
    cache_key = f"{url}:{ext}"
    cached = _PROVENANCE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    prov = Provenance.model_construct(
        source=SourceRef.model_construct(uri=url, mime_type="text/html", artifact_id=None),
        page=None,
        bbox=None,
        char_span=None,
        confidence=None,
        extractor=ext,
    )
    with _PROVENANCE_LOCK:
        _PROVENANCE_CACHE[cache_key] = prov
    return prov


# ---------------------------------------------------------------------------
# Inline processing
# ---------------------------------------------------------------------------


def process_inlines(el: HtmlElement, url: str) -> list[Inline]:
    """Process an element's children as inline content.

    Handles the element's own text, child elements, and tail text.
    This returns the inline nodes for the *children* of ``el`` (including
    el.text), but NOT el's own tail text -- that belongs to the parent.
    """
    result: list[Inline] = []

    # Leading text of the element itself.
    text = _strip_or_empty(el.text)
    if text:
        collapsed = collapse_whitespace(text)
        if collapsed:
            result.append(mk_text(collapsed))

    for child in el:
        if not isinstance(child.tag, str):
            # Processing instruction or comment -- skip, but grab tail.
            tail = _strip_or_empty(child.tail)
            if tail:
                collapsed = collapse_whitespace(tail)
                if collapsed:
                    result.append(mk_text(collapsed))
            continue

        tag = child.tag.lower()

        # Skip elements that should not produce inline content.
        if tag in SKIP_TAGS:
            tail = _strip_or_empty(child.tail)
            if tail:
                collapsed = collapse_whitespace(tail)
                if collapsed:
                    result.append(mk_text(collapsed))
            continue

        # Skip elements with well-known noise classes (e.g. mw-editsection).
        if should_skip_element(child):
            tail = _strip_or_empty(child.tail)
            if tail:
                collapsed = collapse_whitespace(tail)
                if collapsed:
                    result.append(mk_text(collapsed))
            continue

        # Transparent inline elements: expand their children directly.
        if tag in ("span", "u", "mark", "abbr", "time", "small", "font"):
            result.extend(process_inlines(child, url))
        else:
            inline = element_to_inline(child, url)
            if inline is not None:
                result.append(inline)

        # Tail text after the child element.
        tail = _strip_or_empty(child.tail)
        if tail:
            collapsed = collapse_whitespace(tail)
            if collapsed:
                result.append(mk_text(collapsed))

    return merge_adjacent_text(result)


def element_to_inline(el: HtmlElement, url: str) -> Inline | None:
    """Convert a single element to an Inline node (or None to skip)."""
    tag = el.tag.lower() if isinstance(el.tag, str) else ""

    if tag in ("strong", "b"):
        children = tuple(process_inlines(el, url))
        if not children or is_whitespace_only_inlines(children):
            return None
        # Collapse redundant nesting: <b><b>text</b></b> -> Strong(text)
        if len(children) == 1 and isinstance(children[0], Strong):
            return children[0]
        return mk_strong(children)

    if tag in ("em", "i"):
        children = tuple(process_inlines(el, url))
        if not children or is_whitespace_only_inlines(children):
            return None
        # Collapse redundant nesting
        if len(children) == 1 and isinstance(children[0], Emphasis):
            return children[0]
        return mk_emphasis(children)

    if tag in ("s", "del", "strike"):
        children = tuple(process_inlines(el, url))
        if not children:
            return None
        return Strikethrough.model_construct(
            id=_fast_id(),
            attr=_DEFAULT_ATTR,
            provenance=None,
            node_type="strikethrough",
            children=children,
        )

    if tag == "code":
        # Inline code: use text_content to flatten children.
        value = el.text_content() or ""
        if not value:
            return None
        return mk_code(value)

    if tag == "a":
        href = el.get("href", "")
        # Skip UI action links (vote, hide, flag, etc.)
        if href and _is_action_link(href):
            return None
        resolved = resolve_url(href, url) if href else ""
        if resolved and not _is_safe_url(resolved):
            # Sec-4 (security finding #4): drop the dangerous link but
            # preserve the user-visible text. Pre-fix this branch
            # returned ``None``, silently swallowing the inner text —
            # ``<p>before <a href="javascript:...">click me</a> after</p>``
            # collapsed to ``before  after`` with the link contents
            # erased entirely.
            #
            # Return a Span wrapping the original children so the text
            # survives. The Span boundary lets downstream consumers
            # (annotations, transforms) attribute the change to a
            # stripped link via Attr if/when needed; for now no Attr
            # marker is attached because there's no in-tree consumer.
            stripped_children = tuple(process_inlines(el, url))
            if not stripped_children:
                # Anchor had no inline children — nothing to preserve.
                return None
            return mk_span(stripped_children)
        children = tuple(process_inlines(el, url))
        if not children and not resolved:
            return None
        if not children:
            # Link with no visible text -- use href as text.
            children = (mk_text(resolved),)
        title = el.get("title")
        return mk_link(resolved, children, title or None)

    if tag == "img":
        src = _get_image_src(el)
        if not src:
            return None
        resolved = resolve_url(src, url)
        if not _is_safe_url(resolved):
            return None
        alt = el.get("alt", "")
        title = el.get("title")
        return mk_image(resolved, alt or None, title or None)

    if tag == "br":
        return mk_linebreak()

    if tag == "sub":
        children = tuple(process_inlines(el, url))
        if not children:
            return None
        return Subscript.model_construct(
            id=_fast_id(),
            attr=_DEFAULT_ATTR,
            provenance=None,
            node_type="subscript",
            children=children,
        )

    if tag == "sup":
        children = tuple(process_inlines(el, url))
        if not children:
            return None
        return Superscript.model_construct(
            id=_fast_id(),
            attr=_DEFAULT_ATTR,
            provenance=None,
            node_type="superscript",
            children=children,
        )

    # Unknown inline-ish tag -- flatten text content.
    text = el.text_content() or ""
    if text.strip():
        return mk_text(collapse_whitespace(text))
    return None


# ---------------------------------------------------------------------------
# Block processing
# ---------------------------------------------------------------------------


def process_element(el: HtmlElement, url: str) -> list[Block]:
    """Convert a single HTML element into Block AST nodes."""
    if not isinstance(el.tag, str):
        return []

    tag = el.tag.lower()
    prov = make_provenance(url)

    # Skip non-content elements.
    if tag in SKIP_TAGS:
        return []

    # Skip elements with well-known noise classes (e.g. mw-editsection).
    if should_skip_element(el):
        return []

    # Headings.
    if tag in HEADING_TAGS:
        depth = int(tag[1])
        children = trim_inline_whitespace(list(process_inlines(el, url)))
        if not children:
            return []
        return [mk_heading(depth, tuple(children), prov)]

    # Paragraph.
    if tag == "p":
        children = tuple(process_inlines(el, url))
        if not children:
            return []
        # Skip whitespace-only paragraphs (only Text children, all whitespace)
        if all(isinstance(c, Text) for c in children):
            text = "".join(c.value for c in children if isinstance(c, Text)).strip()
            if not text:
                return []
        return [mk_paragraph(children, prov)]

    # Blockquote.
    if tag == "blockquote":
        blocks = tuple(process_children_as_blocks(el, url))
        if not blocks:
            # Try as inline content wrapped in a paragraph.
            inlines = tuple(process_inlines(el, url))
            if inlines:
                blocks = (mk_paragraph(inlines, prov),)
        if not blocks:
            return []
        return [
            BlockQuote.model_construct(
                id=_fast_id(),
                attr=_DEFAULT_ATTR,
                provenance=prov,
                node_type="blockquote",
                children=blocks,
            )
        ]

    # Preformatted / code blocks. Mode is threaded from ``parse_html``
    # via a module-global (see ``pre_content_scope``) so every recursion
    # level sees the same choice without churning every call-site
    # signature. Matches the existing ``extractor_scope`` pattern.
    if tag == "pre":
        return _process_pre(el, url, prov, mode=_pre_content_mode_var.get())

    # Lists.
    if tag == "ul":
        items = _process_list_items(el, url)
        if not items:
            return []
        return [
            BulletList.model_construct(
                id=_fast_id(),
                attr=_DEFAULT_ATTR,
                provenance=prov,
                node_type="bullet_list",
                children=tuple(items),
            )
        ]

    if tag == "ol":
        items = _process_list_items(el, url)
        if not items:
            return []
        start = 1
        start_attr = el.get("start")
        if start_attr is not None:
            with contextlib.suppress(ValueError):
                start = int(start_attr)
        return [
            OrderedList.model_construct(
                id=_fast_id(),
                attr=_DEFAULT_ATTR,
                provenance=prov,
                node_type="ordered_list",
                start=start,
                children=tuple(items),
            )
        ]

    # Definition list.
    if tag == "dl":
        return _process_definition_list(el, url, prov)

    # Table.
    if tag == "table":
        return _process_table(el, url, prov)

    # Horizontal rule.
    if tag == "hr":
        return [
            ThematicBreak.model_construct(
                id=_fast_id(), attr=_DEFAULT_ATTR, provenance=prov, node_type="thematic_break"
            )
        ]

    # Figure.
    if tag == "figure":
        return _process_figure(el, url, prov)

    # Transparent block containers.
    if tag in TRANSPARENT_BLOCK_TAGS:
        blocks = process_children_as_blocks(el, url)
        return blocks

    # Inline-level elements encountered at block level -- wrap in Paragraph.
    if tag in INLINE_FORMATTING_TAGS:
        inlines = _element_to_inline_list(el, url)
        if inlines:
            return [mk_paragraph(tuple(inlines), prov)]
        return []

    # Unknown tags -- try to process children as blocks.
    blocks = process_children_as_blocks(el, url)
    if blocks:
        return blocks

    # Last resort: try as inline content.
    inlines = tuple(process_inlines(el, url))
    if inlines:
        return [mk_paragraph(inlines, prov)]

    return []


def _element_to_inline_list(el: HtmlElement, url: str) -> list[Inline]:
    """Convert an inline-level element to a list of Inline nodes.

    Unlike element_to_inline which returns a single node, this returns a list
    so transparent elements can contribute multiple children.
    """
    tag = el.tag.lower() if isinstance(el.tag, str) else ""

    if tag in ("span", "u", "mark", "abbr", "time", "small", "font"):
        return process_inlines(el, url)

    node = element_to_inline(el, url)
    if node is not None:
        return [node]
    return []


def process_children_as_blocks(el: HtmlElement, url: str) -> list[Block]:
    """Process all children of an element as block nodes.

    Also handles stray text between child elements by wrapping it in Paragraphs.
    After the initial pass, runs a merge pass that collapses "orphan inline"
    paragraphs into the preceding paragraph (see :func:`merge_orphan_inlines`).
    """
    result: list[Block] = []
    prov = make_provenance(url)

    # Leading text in the element.
    text = _strip_or_empty(el.text)
    if text:
        collapsed = collapse_whitespace(text).strip()
        if collapsed:
            result.append(mk_paragraph((mk_text(collapsed),), prov))

    for child in el:
        if not isinstance(child.tag, str):
            # Comment/PI -- grab tail text.
            tail = _strip_or_empty(child.tail)
            if tail:
                collapsed = collapse_whitespace(tail).strip()
                if collapsed:
                    result.append(mk_paragraph((mk_text(collapsed),), prov))
            continue

        blocks = process_element(child, url)
        result.extend(blocks)

        # Tail text after this child element.
        tail = _strip_or_empty(child.tail)
        if tail:
            collapsed = collapse_whitespace(tail).strip()
            if collapsed:
                result.append(mk_paragraph((mk_text(collapsed),), prov))

    return merge_orphan_inlines(result)


# Characters that are inline decorations when they appear as the sole
# content of a block-level element.  These are typically trademark,
# copyright, or footnote symbols that styled HTML (EDGAR, PDF-to-HTML)
# places in separate <div> or <span> elements at block level.
_ORPHAN_INLINE_CHARS = frozenset("®™©℠¹²³⁴⁵⁶⁷⁸⁹⁰*†‡§¶")

# Maximum length of a paragraph's text for it to be considered an orphan.
_ORPHAN_MAX_LEN = 4


def _is_orphan_inline(block: Block) -> bool:
    """True if ``block`` is a short paragraph that's really inline content.

    Detects paragraphs that contain only:
    - Trademark/copyright symbols (R), (TM), (C))
    - Superscript numbers
    - Footnote markers (*, dagger, double-dagger)
    - Other single characters that styled HTML placed in their own block

    These arise from EDGAR/XBRL, PDF-to-HTML converters, and other tools
    that use ``<div>`` or ``<span>`` at block level for what should be
    inline content.
    """
    if not isinstance(block, Paragraph):
        return False
    children = block.children
    if not children:
        return False
    # Extract the text content of the paragraph.
    text = "".join(
        c.value for c in children if hasattr(c, "value") and isinstance(c.value, str)
    ).strip()
    if not text or len(text) > _ORPHAN_MAX_LEN:
        return False
    # All characters must be in the orphan set.
    return all(ch in _ORPHAN_INLINE_CHARS for ch in text)


def merge_orphan_inlines(blocks: list[Block]) -> list[Block]:
    """Merge orphan-inline paragraphs into the preceding paragraph.

    After the main block-processing pass, the block list may contain
    sequences like::

        Paragraph("iPhone")
        Paragraph("(R)")           <- orphan inline
        Paragraph(" is the ...")

    This function collapses them into::

        Paragraph("iPhone(R) is the ...")

    Only paragraphs whose entire text content is in
    :data:`_ORPHAN_INLINE_CHARS` and is <= :data:`_ORPHAN_MAX_LEN`
    characters are merged.  All other blocks pass through unchanged.
    """
    if len(blocks) < 2:
        return blocks

    merged: list[Block] = [blocks[0]]
    for block in blocks[1:]:
        prev = merged[-1]
        if (
            _is_orphan_inline(block)
            and merged
            and isinstance(prev, Paragraph)
            and isinstance(block, Paragraph)
        ):
            # Merge: append the orphan's children to the preceding paragraph.
            new_children = prev.children + block.children
            merged[-1] = mk_paragraph(new_children, prev.provenance)
        elif (
            merged
            and isinstance(prev, Paragraph)
            and isinstance(block, Paragraph)
            and _is_orphan_inline(prev)
        ):
            # Edge case: the PREVIOUS block was an orphan that didn't
            # have a predecessor to merge into -- merge forward instead.
            new_children = prev.children + block.children
            merged[-1] = mk_paragraph(new_children, block.provenance)
        else:
            merged.append(block)
    return merged


# ---------------------------------------------------------------------------
# Specialised element processors
# ---------------------------------------------------------------------------


def _process_pre(
    el: HtmlElement,
    url: str,
    prov: Provenance | None,
    *,
    mode: str = "code",
) -> list[Block]:
    """Process ``<pre>`` and ``<pre><code>`` elements.

    Two modes:

    - ``"code"`` (default) — emit a ``CodeBlock`` preserving whitespace
      and language hint. Correct for actual source code and for
      consumers that need whitespace fidelity.
    - ``"prose"`` — treat the inner text as paragraph-separated prose
      (blank lines delimit paragraphs, per the Markdown / RFC / plain-
      text convention). Emit one ``Paragraph`` per blank-line-delimited
      chunk. Correct for sources that abuse ``<pre>`` as a container
      for wrapped plain text (e.g. the Federal Register
      ``raw_text_url`` endpoint) where the caller knows the content is
      prose, not code.

    The caller chooses the mode via the ``pre_content_mode`` parameter
    on :func:`parse_html`; parsers never guess based on content
    heuristics.
    """
    code_el = el.find("code")
    if code_el is not None:
        language = _extract_language(code_el) or _extract_language(el)
        value = code_el.text_content() or ""
    else:
        language = _extract_language(el)
        value = el.text_content() or ""

    if not value:
        return []
    # Strip leading newline per HTML spec (browsers do this for <pre>)
    if value.startswith("\n"):
        value = value[1:]

    if mode == "prose":
        # Blank-line-delimited paragraphs. Empty chunks (from runs of
        # newlines) are dropped; a single continuous block with no
        # blank line becomes one Paragraph.
        blocks: list[Block] = []
        for chunk in re.split(r"\n\s*\n", value):
            text = chunk.strip()
            if not text:
                continue
            blocks.append(
                Paragraph.model_construct(
                    id=_fast_id(),
                    attr=_DEFAULT_ATTR,
                    provenance=prov,
                    node_type="paragraph",
                    children=(mk_text(text),),
                )
            )
        return blocks

    return [
        CodeBlock.model_construct(
            id=_fast_id(),
            attr=_DEFAULT_ATTR,
            provenance=prov,
            node_type="codeblock",
            language=language,
            value=value,
        )
    ]


def _process_list_items(el: HtmlElement, url: str) -> list[ListItem]:
    """Process <li> children of a list element."""
    items: list[ListItem] = []
    prov = make_provenance(url)

    for child in el:
        if not isinstance(child.tag, str):
            continue
        if child.tag.lower() != "li":
            continue

        # Check if <li> has block-level children (p, ul, ol, blockquote, etc.)
        has_block_children = any(
            isinstance(c.tag, str) and c.tag.lower() in BLOCK_LEVEL_TAGS for c in child
        )

        if has_block_children:
            blocks = process_children_as_blocks(child, url)
        else:
            # Inline-only content -- wrap in a single paragraph
            inlines = tuple(process_inlines(child, url))
            if inlines and not is_whitespace_only_inlines(inlines):
                inlines = tuple(trim_inline_whitespace(list(inlines)))
                blocks = [mk_paragraph(inlines, prov)]
            else:
                blocks = []

        if blocks:
            items.append(
                ListItem.model_construct(
                    id=_fast_id(),
                    attr=_DEFAULT_ATTR,
                    provenance=prov,
                    node_type="list_item",
                    checked=None,
                    children=tuple(blocks),
                )
            )

    return items


def _process_definition_list(el: HtmlElement, url: str, prov: Provenance | None) -> list[Block]:
    """Process <dl> into DefinitionList."""
    items: list[DefinitionItem] = []

    current_terms: list[tuple[Inline, ...]] = []
    current_defs: list[tuple[Block, ...]] = []

    for child in el:
        if not isinstance(child.tag, str):
            continue
        tag = child.tag.lower()

        if tag == "dt":
            # If we have accumulated terms+defs, flush.
            if current_terms and current_defs:
                for term in current_terms:
                    items.append(
                        DefinitionItem.model_construct(
                            id=_fast_id(),
                            attr=_DEFAULT_ATTR,
                            provenance=prov,
                            node_type="definition_item",
                            term=term,
                            definitions=tuple(current_defs),
                        )
                    )
                current_terms = []
                current_defs = []
            elif current_terms and not current_defs:
                # Multiple terms before any definition -- keep accumulating.
                pass

            inlines = tuple(process_inlines(child, url))
            if inlines:
                current_terms.append(inlines)

        elif tag == "dd":
            blocks = process_children_as_blocks(child, url)
            if not blocks:
                inlines = tuple(process_inlines(child, url))
                if inlines:
                    blocks = [mk_paragraph(inlines, prov)]
            if blocks:
                current_defs.append(tuple(blocks))

    # Flush remaining.
    if current_terms:
        for term in current_terms:
            items.append(
                DefinitionItem.model_construct(
                    id=_fast_id(),
                    attr=_DEFAULT_ATTR,
                    provenance=prov,
                    node_type="definition_item",
                    term=term,
                    definitions=tuple(current_defs) if current_defs else (),
                )
            )

    if not items:
        return []
    return [
        DefinitionList.model_construct(
            id=_fast_id(),
            attr=_DEFAULT_ATTR,
            provenance=prov,
            node_type="definition_list",
            children=tuple(items),
        )
    ]


def _process_table(el: HtmlElement, url: str, prov: Provenance | None) -> list[Block]:
    """Process <table> into Table AST node."""
    head: TableSection | None = None
    bodies: list[TableSection] = []

    # Process <thead>.
    thead = el.find("thead")
    if thead is not None:
        rows = _process_table_rows(thead, url, is_header=True)
        if rows:
            head = TableSection.model_construct(
                id=_fast_id(),
                attr=_DEFAULT_ATTR,
                provenance=None,
                node_type="table_section",
                rows=tuple(rows),
            )

    # Process <tbody> elements.
    tbodies = el.findall("tbody")
    if tbodies:
        for tbody in tbodies:
            rows = _process_table_rows(tbody, url, is_header=False)
            if rows:
                bodies.append(
                    TableSection.model_construct(
                        id=_fast_id(),
                        attr=_DEFAULT_ATTR,
                        provenance=None,
                        node_type="table_section",
                        rows=tuple(rows),
                    )
                )
    else:
        # No explicit <tbody>: process direct <tr> children.
        rows = _process_table_rows(el, url, is_header=False)
        if rows:
            bodies.append(
                TableSection.model_construct(
                    id=_fast_id(),
                    attr=_DEFAULT_ATTR,
                    provenance=None,
                    node_type="table_section",
                    rows=tuple(rows),
                )
            )

    # Process <tfoot>.
    foot: TableSection | None = None
    tfoot = el.find("tfoot")
    if tfoot is not None:
        rows = _process_table_rows(tfoot, url, is_header=False)
        if rows:
            foot = TableSection.model_construct(
                id=_fast_id(),
                attr=_DEFAULT_ATTR,
                provenance=None,
                node_type="table_section",
                rows=tuple(rows),
            )

    # Extract caption.
    caption: Caption | None = None
    cap_el = el.find("caption")
    if cap_el is not None:
        inlines = tuple(process_inlines(cap_el, url))
        if inlines:
            caption = Caption.model_construct(short=None, body=(mk_paragraph(inlines, prov),))

    if not head and not bodies and not foot:
        return []

    return [
        Table.model_construct(
            id=_fast_id(),
            attr=_DEFAULT_ATTR,
            provenance=prov,
            node_type="table",
            caption=caption,
            col_specs=(),
            head=head,
            bodies=tuple(bodies),
            foot=foot,
        )
    ]


def _process_table_rows(container: HtmlElement, url: str, *, is_header: bool) -> list[Row]:
    """Process <tr> elements within a table section."""
    rows: list[Row] = []

    for tr in container:
        if not isinstance(tr.tag, str) or tr.tag.lower() != "tr":
            continue
        cells: list[Cell] = []

        for td in tr:
            if not isinstance(td.tag, str):
                continue
            tag = td.tag.lower()
            if tag not in ("td", "th"):
                continue

            # Parse content of cell.
            blocks = process_children_as_blocks(td, url)
            if not blocks:
                inlines = tuple(process_inlines(td, url))
                if inlines:
                    blocks = [mk_paragraph(inlines, make_provenance(url))]

            col_span = 1
            row_span = 1
            cs = td.get("colspan")
            if cs:
                with contextlib.suppress(ValueError):
                    col_span = max(1, int(cs))
            rs = td.get("rowspan")
            if rs:
                with contextlib.suppress(ValueError):
                    row_span = max(1, int(rs))

            cells.append(
                Cell.model_construct(
                    id=_fast_id(),
                    attr=_DEFAULT_ATTR,
                    provenance=None,
                    node_type="cell",
                    alignment=None,
                    content=tuple(blocks),
                    col_span=col_span,
                    row_span=row_span,
                )
            )

        if cells:
            rows.append(
                Row.model_construct(
                    id=_fast_id(),
                    attr=_DEFAULT_ATTR,
                    provenance=None,
                    node_type="row",
                    cells=tuple(cells),
                )
            )

    return rows


def _process_figure(el: HtmlElement, url: str, prov: Provenance | None) -> list[Block]:
    """Process <figure> into Figure AST node."""
    caption: Caption | None = None
    children: list[Block] = []

    for child in el:
        if not isinstance(child.tag, str):
            continue
        tag = child.tag.lower()

        if tag == "figcaption":
            inlines = tuple(process_inlines(child, url))
            if inlines:
                caption = Caption.model_construct(short=None, body=(mk_paragraph(inlines, prov),))
        elif tag == "img":
            src = _get_image_src(child)
            if src:
                resolved = resolve_url(src, url)
                if _is_safe_url(resolved):
                    alt = child.get("alt", "")
                    img = mk_image(resolved, alt or None)
                    children.append(mk_paragraph((img,), prov))
        else:
            blocks = process_element(child, url)
            children.extend(blocks)

    if not children and not caption:
        return []

    return [
        Figure.model_construct(
            id=_fast_id(),
            attr=_DEFAULT_ATTR,
            provenance=prov,
            node_type="figure",
            caption=caption,
            children=tuple(children),
        )
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_html(
    html_content: str,
    *,
    url: str = "",
    strip_xbrl: bool | None = None,
    pre_content_mode: str = "code",
) -> ContentDocument:
    """Convert raw HTML to a ContentDocument AST.

    This is the raw HTML->AST converter.  It does NOT apply readability
    extraction (for that, use ``kaos_web.html_to_document``).  The entire
    HTML ``<body>`` is converted to AST blocks.

    Args:
        html_content: Raw HTML string.
        url: Source URL for provenance and relative URL resolution.
        strip_xbrl: If ``True``, preprocess the HTML to strip Inline
            XBRL (iXBRL) markup before parsing.  If ``None``
            (default), auto-detect XBRL and strip if present.  If
            ``False``, skip XBRL stripping even if detected.
        pre_content_mode: How to interpret ``<pre>`` tag content.
            ``"code"`` (default) emits a ``CodeBlock`` preserving
            whitespace. ``"prose"`` treats the inner text as
            blank-line-separated prose and emits ``Paragraph`` blocks.
            Use ``"prose"`` for sources (e.g. Federal Register
            ``raw_text_url``) that abuse ``<pre>`` as a plain-text
            container.

    Returns:
        ContentDocument with Block/Inline AST nodes and provenance.
    """
    if not html_content or not html_content.strip():
        return empty_document()

    # Strip Inline XBRL if requested or auto-detected.
    if strip_xbrl is True or (strip_xbrl is None and looks_like_xbrl(html_content)):
        html_content = strip_inline_xbrl(html_content)

    # Parse full document and use <body>. lxml raises ParserError /
    # XMLSyntaxError (both subclass LxmlError) on malformed markup, and
    # ValueError on empty / whitespace-only input. Anything broader
    # (TypeError, MemoryError) is a programmer bug, not a parse failure,
    # and should propagate.
    try:
        full_doc = lxml_html.document_fromstring(html_content)
    except (LxmlError, ValueError):
        logger.debug("HTML parse error", exc_info=True)
        return empty_document()

    root = full_doc.body if full_doc is not None else None
    if root is None:
        return empty_document()

    # Convert the element tree to AST blocks. ``pre_content_mode`` is
    # threaded via ``pre_content_scope`` so every ``<pre>`` anywhere in
    # the tree (including inside nested divs) sees the same choice.
    with pre_content_scope(pre_content_mode):
        blocks = process_children_as_blocks(root, url)

    # Extract title from the original HTML for metadata. The guards make
    # the body safe against missing / empty <title>; the narrow except
    # catches only lxml's tree-walk errors (rare on a successfully-parsed
    # document but possible for pathological markup).
    title: str | None = None
    try:
        title_el = full_doc.find(".//title")
        if title_el is not None and title_el.text:
            title = title_el.text.strip() or None
    except LxmlError:
        logger.debug("HTML title extraction error", exc_info=True)

    metadata = DocumentMetadata.model_construct(
        title=title,
        authors=(),
        date=None,
        language=None,
        source=SourceRef.model_construct(uri=url, mime_type="text/html", artifact_id=None)
        if url
        else None,
        document_type=None,
        extra={},
    )

    return ContentDocument.model_construct(
        metadata=metadata,
        body=tuple(blocks),
        footnotes={},
        definitions={},
        annotations=(),
    )
