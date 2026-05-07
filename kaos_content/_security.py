"""Internal security primitives shared across parsers and serializers.

This module is intentionally kept package-private (``_security``).
External callers should not depend on it; if you need URL-safety
checking outside of kaos-content, file an issue describing the use case
so we can promote a stable public API.

Currently provides:

- :data:`UNSAFE_SCHEMES`: the URI schemes treated as XSS-dangerous.
- :func:`is_safe_url`: predicate that returns ``False`` for a URL whose
  canonical scheme is in :data:`UNSAFE_SCHEMES`. Canonicalises through
  HTML-entity decoding, URL percent-decoding, removal of all whitespace
  and NUL bytes, and lowercase before checking.

The HTML parser, the HTML serializer, and the Markdown serializer all
delegate to :func:`is_safe_url` so a single fix applies everywhere.
"""

from __future__ import annotations

import html as _stdlib_html
from urllib.parse import unquote, urlparse

UNSAFE_SCHEMES: frozenset[str] = frozenset({"javascript", "data", "vbscript", "file"})
"""URI schemes that are rejected by :func:`is_safe_url`.

- ``javascript``, ``vbscript``: execute as scripts when navigated.
- ``data``: can carry inline ``text/html`` payloads with scripts.
- ``file``: can disclose local filesystem paths in HTML/PDF contexts.
"""


def is_safe_url(url: str) -> bool:
    """Return ``False`` for URLs whose canonical scheme is unsafe.

    The previous in-tree implementation used
    ``url.strip().lower().startswith(...)`` which let through every
    variant where the scheme was mutated through embedded whitespace,
    HTML entities, or URL percent-encoding. Concrete bypasses observed
    in audit:

    - ``"jav\\nascript:alert(1)"``        (newline inside the scheme)
    - ``"jav\\tascript:alert(1)"``        (tab inside the scheme)
    - ``"&#x6A;avascript:alert(1)"``     (HTML hex entity)
    - ``"&#106;avascript:alert(1)"``     (HTML decimal entity)
    - ``"javascript%3Aalert(1)"``        (percent-encoded colon)

    This implementation canonicalises through HTML-entity decoding, URL
    percent-decoding, removal of all whitespace and NUL bytes, and
    lowercase, then checks the scheme via :func:`urllib.parse.urlparse`
    with a defence-in-depth ``startswith`` check on the canonical form.

    Returns ``True`` for safe URLs (including the empty string and
    relative paths) and ``False`` for any URL whose canonical scheme
    is in :data:`UNSAFE_SCHEMES`.
    """
    if not url:
        return True
    decoded = _stdlib_html.unescape(url)
    decoded = unquote(decoded)
    canonical = "".join(c for c in decoded if not c.isspace() and c != "\x00").lower()
    if not canonical:
        return True
    try:
        parsed = urlparse(canonical)
    except (ValueError, TypeError):
        # Malformed URLs (e.g. ``http://[`` — bare IPv6 bracket without a
        # closing ``]``) make :func:`urlparse` raise ``ValueError`` on
        # Python ≥3.6 because the netloc parser refuses to interpret the
        # bracket. Treat any URL we cannot parse as unsafe (defence-in-
        # depth + crash-safety: the fuzz contract is that arbitrary
        # input must never propagate an exception out of this predicate).
        return False
    if parsed.scheme in UNSAFE_SCHEMES:
        return False
    return all(not canonical.startswith(f"{scheme}:") for scheme in UNSAFE_SCHEMES)
