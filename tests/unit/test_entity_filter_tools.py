"""Unit tests for the K3 entity-filter MCP tools."""

from __future__ import annotations

from typing import Any

import pytest
from kaos_core.base.context import KaosContext
from kaos_core.registry.container import KaosRuntime
from kaos_core.types.results import ToolResult

from kaos_content.artifacts import store_document
from kaos_content.model.document import ContentDocument
from kaos_content.shortcuts import heading, paragraph
from kaos_content.tools import (
    SentencesWithDatesTool,
    SentencesWithDurationsTool,
    SentencesWithMoneyTool,
    SentencesWithNumbersTool,
    SentencesWithPercentsTool,
    register_content_tools,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def runtime() -> KaosRuntime:
    rt = KaosRuntime()
    register_content_tools(rt)
    return rt


@pytest.fixture
def context(runtime: KaosRuntime) -> KaosContext:
    return KaosContext.create(session_id="test-session", runtime=runtime)


def _entity_doc() -> ContentDocument:
    return ContentDocument(
        body=(
            heading(1, "Test Agreement"),
            paragraph("Effective January 1, 2026, the cap is $100,000 for 12 months."),
            paragraph("Liability runs at 15% per annum."),
            paragraph("This paragraph has no extractable entities."),
        ),
    )


async def _store(doc: ContentDocument, ctx: KaosContext) -> str:
    """Helper to store a doc and return its artifact_id."""
    manifest = await store_document(doc, ctx.runtime, ctx, name="entity-test-doc")
    return manifest.artifact_id


def _payload(result: ToolResult) -> dict[str, Any]:
    """Narrow ``result.structuredContent`` to a non-None dict for typing.

    Every entity-filter tool always returns structured content on
    success; this should never raise.
    """
    assert result.structuredContent is not None
    return result.structuredContent


def _error_text(result: ToolResult) -> str:
    """Pull the text out of ``result.content[0]``.

    Duck-types the lookup. The TextContent type pulled from mcp.types
    in this test module is not always the same class as the one
    populated by ToolResult.create_error (different import paths can
    yield distinct identity even though they're structurally
    identical). ``hasattr(first, 'text')`` is the working contract.
    """
    first = result.content[0]
    text = getattr(first, "text", None)
    assert text is not None, f"expected text-bearing content, got {type(first).__name__}"
    return str(text)


# ---------------------------------------------------------------------------
# Tool metadata
# ---------------------------------------------------------------------------


class TestToolMetadata:
    def test_dates_tool_name(self) -> None:
        tool = SentencesWithDatesTool()
        assert tool.metadata.name == "kaos-content-sentences-with-dates"

    def test_money_tool_name(self) -> None:
        tool = SentencesWithMoneyTool()
        assert tool.metadata.name == "kaos-content-sentences-with-money"

    def test_percents_tool_name(self) -> None:
        tool = SentencesWithPercentsTool()
        assert tool.metadata.name == "kaos-content-sentences-with-percents"

    def test_durations_tool_name(self) -> None:
        tool = SentencesWithDurationsTool()
        assert tool.metadata.name == "kaos-content-sentences-with-durations"

    def test_numbers_tool_name(self) -> None:
        tool = SentencesWithNumbersTool()
        assert tool.metadata.name == "kaos-content-sentences-with-numbers"

    def test_all_tools_are_read_only(self) -> None:
        for tool_cls in (
            SentencesWithDatesTool,
            SentencesWithMoneyTool,
            SentencesWithPercentsTool,
            SentencesWithDurationsTool,
            SentencesWithNumbersTool,
        ):
            tool = tool_cls()
            ann = tool.metadata.annotations
            assert ann is not None
            assert ann.readOnlyHint is True
            assert ann.destructiveHint is False
            assert ann.idempotentHint is True

    def test_input_schema_has_artifact_id_granularity_max_results(self) -> None:
        tool = SentencesWithDatesTool()
        params = {p.name for p in tool.metadata.input_schema}
        assert params == {"artifact_id", "granularity", "max_results", "select_by"}


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


class TestToolRegistration:
    def test_all_five_registered(self, runtime: KaosRuntime) -> None:
        names = [t.metadata.name for t in runtime.tools.list_tool_objects()]
        for entity_type in ("dates", "money", "percents", "durations", "numbers"):
            assert f"kaos-content-sentences-with-{entity_type}" in names


# ---------------------------------------------------------------------------
# Execute — dates
# ---------------------------------------------------------------------------


class TestDatesTool:
    async def test_finds_one_date_sentence(self, context: KaosContext) -> None:
        artifact_id = await _store(_entity_doc(), context)
        tool = SentencesWithDatesTool()
        result = await tool.execute({"artifact_id": artifact_id}, context)
        assert not result.isError
        out = _payload(result)
        assert out["entity_type"] == "dates"
        assert out["total_matches"] >= 1
        # The matched sentence must contain the date text
        first = out["matches"][0]
        assert "January" in first["text"] or "2026" in first["text"]
        assert any("January" in e["text"] or "2026" in e["text"] for e in first["entities"])

    async def test_typed_value_is_iso_date(self, context: KaosContext) -> None:
        artifact_id = await _store(_entity_doc(), context)
        tool = SentencesWithDatesTool()
        result = await tool.execute({"artifact_id": artifact_id}, context)
        first = _payload(result)["matches"][0]
        # Value is serialised as ISO datetime string
        entity = first["entities"][0]
        assert "2026-01" in entity["value"]


# ---------------------------------------------------------------------------
# Execute — money
# ---------------------------------------------------------------------------


class TestMoneyTool:
    async def test_finds_money_with_typed_value(self, context: KaosContext) -> None:
        artifact_id = await _store(_entity_doc(), context)
        tool = SentencesWithMoneyTool()
        result = await tool.execute({"artifact_id": artifact_id}, context)
        first = _payload(result)["matches"][0]
        entity = first["entities"][0]
        # MoneyMatch is serialised as {"amount": "100000", "currency": "USD"}
        assert isinstance(entity["value"], dict)
        assert "amount" in entity["value"]
        assert entity["value"]["amount"] == "100000"
        assert entity["value"]["currency"] == "USD"


# ---------------------------------------------------------------------------
# Execute — paragraph granularity
# ---------------------------------------------------------------------------


class TestGranularity:
    async def test_paragraph_granularity(self, context: KaosContext) -> None:
        artifact_id = await _store(_entity_doc(), context)
        tool = SentencesWithDatesTool()
        result = await tool.execute(
            {"artifact_id": artifact_id, "granularity": "paragraph"}, context
        )
        assert _payload(result)["granularity"] == "paragraph"
        # Paragraph-level results still have text + entities
        first = _payload(result)["matches"][0]
        assert "January" in first["text"] or "2026" in first["text"]

    async def test_invalid_granularity_rejected(self, context: KaosContext) -> None:
        artifact_id = await _store(_entity_doc(), context)
        tool = SentencesWithDatesTool()
        result = await tool.execute({"artifact_id": artifact_id, "granularity": "page"}, context)
        assert result.isError
        assert "granularity" in _error_text(result).lower()


# ---------------------------------------------------------------------------
# Execute — max_results
# ---------------------------------------------------------------------------


class TestMaxResults:
    async def test_max_results_caps_returned_matches(self, context: KaosContext) -> None:
        # Build a doc with lots of dates so total > 2
        body = tuple(paragraph(f"On January {d}, 2026 the parties met.") for d in (1, 2, 3, 4, 5))
        doc = ContentDocument(body=body)
        artifact_id = await _store(doc, context)
        tool = SentencesWithDatesTool()
        result = await tool.execute({"artifact_id": artifact_id, "max_results": 2}, context)
        out = _payload(result)
        assert len(out["matches"]) == 2
        assert out["total_matches"] == 5
        assert out["has_more"] is True

    async def test_max_results_validation(self, context: KaosContext) -> None:
        artifact_id = await _store(_entity_doc(), context)
        tool = SentencesWithDatesTool()
        result = await tool.execute({"artifact_id": artifact_id, "max_results": 0}, context)
        assert result.isError


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class TestErrors:
    async def test_missing_artifact_id(self, context: KaosContext) -> None:
        tool = SentencesWithDatesTool()
        result = await tool.execute({}, context)
        assert result.isError
        assert "artifact_id" in _error_text(result)

    async def test_missing_context(self) -> None:
        tool = SentencesWithDatesTool()
        result = await tool.execute({"artifact_id": "x"}, None)
        assert result.isError

    async def test_unknown_artifact(self, context: KaosContext) -> None:
        tool = SentencesWithDatesTool()
        result = await tool.execute({"artifact_id": "does-not-exist"}, context)
        assert result.isError
        assert "Failed to load" in _error_text(result)


# ---------------------------------------------------------------------------
# Salience + select_by (PA9)
# ---------------------------------------------------------------------------


def _salience_doc() -> ContentDocument:
    """An NDA-shaped fixture with early boilerplate dates and a mid-body
    term clause containing two dates. Mirrors the real-world failure
    mode: first-N-by-position picks the boilerplate; PA9 salience picks
    the load-bearing sentence."""
    from kaos_content.shortcuts import heading as _h
    from kaos_content.shortcuts import paragraph as _p

    return ContentDocument(
        body=(
            _h(1, "Mutual Non-Disclosure Agreement"),
            _p(
                "WHEREAS the parties have been discussing matters since "
                "January 1, 2025, they wish to formalise the terms below."
            ),
            _p("In the year 2026, the parties continue to confer."),
            _p("This Agreement is effective as of March 15, 2026 and expires on March 15, 2027."),
            _p("Notice was given on April 1, 2026 of certain matters."),
            _h(2, "Signatures"),
            _p("Dated: April 22, 2026."),
        ),
    )


class TestSelectBy:
    async def test_salience_default_promotes_load_bearing_sentence(
        self, context: KaosContext
    ) -> None:
        """Default select_by='salience' surfaces the dense term clause
        ahead of the WHEREAS boilerplate when max_results truncates the
        list. This is the PA9 win condition."""
        artifact_id = await _store(_salience_doc(), context)
        tool = SentencesWithDatesTool()
        result = await tool.execute({"artifact_id": artifact_id, "max_results": 1}, context)
        out = _payload(result)
        assert out["select_by"] == "salience"
        first_text = out["matches"][0]["text"]
        # The dense "effective ... expires" sentence must beat the early
        # WHEREAS boilerplate.
        assert "March 15" in first_text
        assert "January 1, 2025" not in first_text

    async def test_position_reproduces_pre_pa9_ordering(self, context: KaosContext) -> None:
        """select_by='position' returns hits in strict document order —
        the pre-PA9 behaviour. Backward-compat contract."""
        artifact_id = await _store(_salience_doc(), context)
        tool = SentencesWithDatesTool()
        result = await tool.execute(
            {"artifact_id": artifact_id, "select_by": "position"},
            context,
        )
        out = _payload(result)
        assert out["select_by"] == "position"
        # First hit is the WHEREAS boilerplate (earliest date in doc order).
        assert "January 1, 2025" in out["matches"][0]["text"]

    async def test_payload_carries_salience_field(self, context: KaosContext) -> None:
        """Every match in the payload must include the salience score
        so downstream agents can re-rank, threshold, or audit."""
        artifact_id = await _store(_salience_doc(), context)
        tool = SentencesWithDatesTool()
        result = await tool.execute({"artifact_id": artifact_id}, context)
        out = _payload(result)
        for m in out["matches"]:
            assert "salience" in m
            assert 0.0 <= float(m["salience"]) <= 1.0

    async def test_invalid_select_by_rejected(self, context: KaosContext) -> None:
        artifact_id = await _store(_salience_doc(), context)
        tool = SentencesWithDatesTool()
        result = await tool.execute({"artifact_id": artifact_id, "select_by": "magic"}, context)
        assert result.isError
        assert "select_by" in _error_text(result)

    async def test_topk_by_salience_differs_from_topk_by_position(
        self, context: KaosContext
    ) -> None:
        """Top-3 by salience must NOT equal top-3 by position on the
        fixture — the regression test for PA9's value proposition."""
        artifact_id = await _store(_salience_doc(), context)
        tool = SentencesWithDatesTool()
        salience_res = await tool.execute(
            {"artifact_id": artifact_id, "max_results": 3, "select_by": "salience"},
            context,
        )
        position_res = await tool.execute(
            {"artifact_id": artifact_id, "max_results": 3, "select_by": "position"},
            context,
        )
        s_texts = [m["text"] for m in _payload(salience_res)["matches"]]
        p_texts = [m["text"] for m in _payload(position_res)["matches"]]
        assert s_texts != p_texts
