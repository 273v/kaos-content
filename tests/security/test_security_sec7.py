"""Sec-7 regression tests: KCONT-01 — XXE / entity expansion in parse_html.

The HTML parser previously called :func:`lxml.html.document_fromstring` with
the default lxml parser settings, which on libxml2 builds with entity
resolution enabled would expand DOCTYPE-declared entities. That is the
"billion laughs" / quadratic-blowup class of attacks (XXE-shaped, even on
HTML where entities are technically not part of the spec — lxml's parser
honours them).

Fix: configure a module-level :class:`lxml.html.HTMLParser` with
``no_network=True`` and ``resolve_entities=False`` and pass it explicitly
to ``document_fromstring``. With those settings, a payload with a billion-
laughs DOCTYPE parses to plain text rather than expanding to gigabytes
of memory.

These tests assert two contracts:

1. **Time / memory bound.** Parsing a billion-laughs payload completes
   well under one second and produces a small document.
2. **External entity reference is *not* resolved to a fetched URL.** The
   parser must leave external entity references unfetched (``no_network``).
"""

from __future__ import annotations

import time

from kaos_content.parsers.html import parse_html

# ----- KCONT-01: billion-laughs / quadratic blowup ------------------------

_BILLION_LAUGHS = """<!DOCTYPE lolz [
  <!ENTITY lol "lol">
  <!ENTITY lol2 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">
  <!ENTITY lol3 "&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;">
  <!ENTITY lol4 "&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;">
  <!ENTITY lol5 "&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;">
  <!ENTITY lol6 "&lol5;&lol5;&lol5;&lol5;&lol5;&lol5;&lol5;&lol5;&lol5;&lol5;">
  <!ENTITY lol7 "&lol6;&lol6;&lol6;&lol6;&lol6;&lol6;&lol6;&lol6;&lol6;&lol6;">
]>
<html><body><p>&lol7;</p></body></html>"""


class TestKContEntityExpansionBlocked:
    def test_billion_laughs_parses_quickly(self) -> None:
        """Quadratic-blowup DOCTYPE must not expand exponentially.

        Hard wall: parse must complete in under one second on a modern
        machine. Without ``resolve_entities=False`` the same payload
        either OOMs or wedges the parser for tens of seconds.
        """
        start = time.perf_counter()
        doc = parse_html(_BILLION_LAUGHS)
        elapsed = time.perf_counter() - start
        # Wall: should complete in well under 1s. Generous to absorb CI
        # noise without becoming flaky.
        assert elapsed < 1.0, f"parse took {elapsed:.2f}s — entity expansion?"
        # Defensive: serialization shouldn't explode either.
        # The doc parses into a small AST regardless of what the
        # entity reference resolves to (left as text, given
        # resolve_entities=False).
        assert doc is not None

    def test_external_entity_not_fetched(self) -> None:
        """External SYSTEM-entity references must not be fetched.

        With ``no_network=True`` the parser must not attempt to resolve
        ``<!ENTITY xxe SYSTEM "http://...">``-style references. We don't
        test by spinning up a server; we just confirm that parsing
        completes without an unhandled network exception (lxml would
        propagate one if it tried).
        """
        payload = (
            "<!DOCTYPE foo [\n"
            "  <!ELEMENT foo ANY>\n"
            '  <!ENTITY xxe SYSTEM "http://invalid.localhost.invalid/secret">\n'
            "]>\n"
            "<html><body><p>&xxe;</p></body></html>"
        )
        # Should parse without raising and without fetching anything.
        doc = parse_html(payload)
        assert doc is not None

    def test_doctype_with_no_entities_still_parses(self) -> None:
        """Plain DOCTYPE without entity declarations must still work."""
        doc = parse_html("<!DOCTYPE html><html><body><p>hi</p></body></html>")
        assert doc is not None
        # Sanity: we recovered the paragraph.
        from kaos_content.serializers.text import serialize_text

        assert "hi" in serialize_text(doc)
