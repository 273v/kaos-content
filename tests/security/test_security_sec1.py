"""Sec-1 regression tests: urlparse ValueError + parser ContextVar isolation.

Fixes #3 (urlparse ValueError leak) and #6 (parser globals →
ContextVar) — both small foundational fixes bundled because they
both touch the parser layer.
"""

from __future__ import annotations

import asyncio

import pytest

from kaos_content._security import is_safe_url
from kaos_content.model.document import ContentDocument
from kaos_content.parsers.html import (
    extractor_scope,
    parse_html,
    pre_content_scope,
)
from kaos_content.serializers.markdown import serialize_markdown
from kaos_content.shortcuts import link, paragraph

# ----- Sec-1 / finding #3: urlparse ValueError must not leak ---------------


class TestIsSafeUrlMalformed:
    """``is_safe_url`` is the central security predicate — must never raise.

    Per the fuzz contract: any input that doesn't crash lxml/markdown-it
    shouldn't crash kaos-content. ``urlparse`` raises ``ValueError`` on
    inputs like ``http://[`` (bare IPv6 bracket without close); the
    predicate must catch and treat as unsafe.
    """

    @pytest.mark.parametrize(
        "url",
        [
            pytest.param("http://[", id="bare-ipv6-open-bracket"),
            pytest.param("https://[::1", id="incomplete-ipv6"),
            pytest.param("http://[invalid::ipv6:address", id="malformed-ipv6-netloc"),
            # Explicit short ``id`` is critical here: pytest sets the
            # full param value into the ``PYTEST_CURRENT_TEST`` env var
            # for each parametrized test, and Windows caps a single env
            # var at 32 767 characters. Without the id override, the
            # 100 000-char host name plus the path-prefix overflows
            # that cap with ``ValueError: the environment variable is
            # longer than 32767 characters``, and the test fails to
            # set up on the Windows-x64 CI leg. The id lets pytest
            # use a short string for the env-var while the test still
            # receives the full giant URL.
            pytest.param(
                "http://" + "a" * 100000 + ":99999999999999999999",
                id="huge-host-and-port",
            ),
        ],
    )
    def test_does_not_raise_on_malformed_url(self, url: str) -> None:
        # Returning True or False is acceptable; raising is not.
        result = is_safe_url(url)
        assert isinstance(result, bool)

    def test_malformed_returns_false(self) -> None:
        # Defence-in-depth: a URL we can't parse is treated as unsafe.
        assert is_safe_url("http://[") is False


class TestParseHtmlMalformedUrl:
    """The parser must survive malformed hrefs in <a> tags."""

    def test_parse_html_with_malformed_href(self) -> None:
        # Must not raise.
        doc = parse_html('<p>before <a href="http://[">click me</a> after</p>')
        # Doesn't matter what the AST looks like — only that we got one.
        assert doc is not None

    def test_parse_html_with_malformed_href_in_image_src(self) -> None:
        doc = parse_html('<p><img src="http://[" alt="x" /></p>')
        assert doc is not None


class TestSerializeMarkdownMalformedUrl:
    """The serializer must survive malformed URLs in Link nodes."""

    def test_serialize_link_with_malformed_url(self) -> None:
        # Must not raise — the contract is: serializer survives any
        # well-formed AST regardless of URL content.
        doc = ContentDocument(body=(paragraph(link("click", "http://[")),))
        out = serialize_markdown(doc)
        assert isinstance(out, str)


# ----- Sec-1 / finding #6: parser globals → ContextVar ---------------------


class TestExtractorScopeContextVarIsolation:
    """Concurrent ``extractor_scope`` blocks must NOT cross-contaminate.

    Pre-fix: ``_extractor_name`` was a module global mutated under a
    lock-free context manager — concurrent asyncio tasks could see each
    other's values mid-flight, polluting provenance metadata.
    Post-fix: ``ContextVar`` gives each task an isolated value.
    """

    @pytest.mark.asyncio
    async def test_concurrent_extractor_scopes_isolated(self) -> None:
        async def parse_with_extractor(name: str) -> str:
            with extractor_scope(name):
                # Yield to the loop several times so concurrent tasks
                # interleave inside the scope.
                await asyncio.sleep(0)
                doc = parse_html(f"<p>{name}</p>", url=f"https://example.test/{name}")
                await asyncio.sleep(0)
                # Find a block with provenance and read its extractor.
                for block in doc.body:
                    if block.provenance is not None:
                        return block.provenance.extractor or ""
                return ""

        # 8 concurrent tasks, 4 distinct extractor names — assert each
        # task sees only its own value in the resulting provenance.
        names = ["alpha", "beta", "gamma", "delta"] * 2
        results = await asyncio.gather(*(parse_with_extractor(n) for n in names))
        assert results == names

    def test_extractor_scope_is_reentrant(self) -> None:
        # Nested scopes restore the outer value on exit.
        with extractor_scope("outer"):
            with extractor_scope("inner"):
                doc = parse_html("<p>x</p>", url="https://example.test/")
                inner_extractor = next(
                    b.provenance.extractor for b in doc.body if b.provenance is not None
                )
                assert inner_extractor == "inner"
            # After the inner scope exits, outer is restored.
            doc = parse_html("<p>x</p>", url="https://example.test/2")
            outer_extractor = next(
                b.provenance.extractor for b in doc.body if b.provenance is not None
            )
            assert outer_extractor == "outer"


class TestPreContentScopeContextVarIsolation:
    """Concurrent ``pre_content_scope`` blocks must NOT cross-contaminate."""

    @pytest.mark.asyncio
    async def test_concurrent_pre_content_scopes_isolated(self) -> None:
        # Note: parse_html() takes a ``pre_content_mode`` parameter and
        # sets its own ``pre_content_scope`` internally, so this test
        # uses the bare ContextVar via the public scope manager and
        # asserts the value survives an asyncio.sleep(0) yield without
        # being clobbered by sibling tasks.
        from kaos_content.parsers.html import _pre_content_mode_var

        async def observe_after_yield(mode: str) -> str:
            with pre_content_scope(mode):
                # Yield several times so concurrent tasks can interleave
                # and pollute each other's value if the var were a global.
                for _ in range(3):
                    await asyncio.sleep(0)
                return _pre_content_mode_var.get()

        modes = ["code", "prose", "code", "prose", "prose", "code"]
        observed = await asyncio.gather(*(observe_after_yield(m) for m in modes))
        assert observed == modes
