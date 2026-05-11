"""Unit tests for the K4 corpus-level MCP tools."""

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
    CorpusNarrowTool,
    CorpusSummarizeTool,
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


def _nda_doc() -> ContentDocument:
    """A small NDA-flavored doc."""
    return ContentDocument(
        body=(
            heading(1, "Mutual Non-Disclosure Agreement"),
            paragraph(
                "Effective January 1, 2026, Acme Corp and Beta Inc. agree "
                "that Confidential Information includes business plans, "
                "financial projections, customer lists, and trade secrets."
            ),
            paragraph(
                "The Term is twenty-four (24) months from the Effective Date. "
                "Either party may terminate with 30 days written notice."
            ),
            paragraph(
                "Liability is capped at $100,000 per occurrence and includes "
                "indemnification carve-outs for gross negligence."
            ),
        ),
    )


def _msa_doc() -> ContentDocument:
    """A distinctly different doc — Master Services Agreement."""
    return ContentDocument(
        body=(
            heading(1, "Master Services Agreement"),
            paragraph(
                "This MSA governs the provision of professional services "
                "between Vendor Corp and Customer Inc. effective March 15, 2026."
            ),
            paragraph(
                "Fees are billed monthly at $250 per hour. The Statement of "
                "Work attached as Exhibit A defines scope and deliverables."
            ),
            paragraph(
                "Payment terms are net thirty (30) days. Late payment incurs "
                "interest at 1.5% per month."
            ),
        ),
    )


async def _store(doc: ContentDocument, ctx: KaosContext, name: str) -> str:
    manifest = await store_document(doc, ctx.runtime, ctx, name=name)
    return manifest.artifact_id


def _payload(result: ToolResult) -> dict[str, Any]:
    assert result.structuredContent is not None
    return result.structuredContent


def _error_text(result: ToolResult) -> str:
    first = result.content[0]
    text = getattr(first, "text", None)
    assert text is not None
    return str(text)


# ---------------------------------------------------------------------------
# CorpusSummarizeTool
# ---------------------------------------------------------------------------


class TestCorpusSummarizeMetadata:
    def test_name_and_annotations(self) -> None:
        tool = CorpusSummarizeTool()
        assert tool.metadata.name == "kaos-content-corpus-summarize"
        ann = tool.metadata.annotations
        assert ann is not None
        assert ann.readOnlyHint is True


class TestCorpusSummarize:
    async def test_builds_summary_for_each_artifact(self, context: KaosContext) -> None:
        a1 = await _store(_nda_doc(), context, name="nda")
        a2 = await _store(_msa_doc(), context, name="msa")

        tool = CorpusSummarizeTool()
        result = await tool.execute({"artifact_ids": [a1, a2]}, context)
        assert not result.isError
        out = _payload(result)
        assert out["built"] == 2
        assert out["skipped"] == 0
        assert out["failed"] == []
        assert len(out["summaries"]) == 2

        # Each summary preview must have the structural fields.
        for s in out["summaries"]:
            assert "artifact_id" in s
            assert "head_snippet" in s
            assert isinstance(s["top_ngrams"], list)
            assert isinstance(s["bottom_ngrams"], list)
            assert "char_length" in s
            assert "entity_counts" in s

    async def test_nda_summary_has_confidential_signal(self, context: KaosContext) -> None:
        a = await _store(_nda_doc(), context, name="nda")
        tool = CorpusSummarizeTool()
        result = await tool.execute({"artifact_ids": [a]}, context)
        summary = _payload(result)["summaries"][0]
        ngrams = {ng["ngram"] for ng in summary["top_ngrams"]}
        # NDA should surface "confidential" or "information" as top n-grams
        assert any("confidential" in n or "information" in n for n in ngrams), (
            f"NDA summary top_ngrams missing confidential/information: {ngrams}"
        )

    async def test_failed_load_is_recorded_not_raised(self, context: KaosContext) -> None:
        a1 = await _store(_nda_doc(), context, name="nda")
        bad = "definitely-not-a-real-artifact-id"
        tool = CorpusSummarizeTool()
        result = await tool.execute({"artifact_ids": [a1, bad]}, context)
        assert not result.isError  # the tool itself didn't fail
        out = _payload(result)
        assert out["built"] == 1
        assert len(out["failed"]) == 1
        assert out["failed"][0]["artifact_id"] == bad

    async def test_entity_counts_populated_by_default(self, context: KaosContext) -> None:
        a = await _store(_nda_doc(), context, name="nda")
        tool = CorpusSummarizeTool()
        result = await tool.execute({"artifact_ids": [a]}, context)
        summary = _payload(result)["summaries"][0]
        counts = summary["entity_counts"]
        # The NDA has dates, money, durations
        assert counts["dates"] >= 1
        assert counts["money"] >= 1
        assert counts["durations"] >= 1

    async def test_entity_counts_skipped_when_disabled(self, context: KaosContext) -> None:
        a = await _store(_nda_doc(), context, name="nda")
        tool = CorpusSummarizeTool()
        result = await tool.execute({"artifact_ids": [a], "with_entities": False}, context)
        summary = _payload(result)["summaries"][0]
        # entity_counts is empty dict when with_entities=False
        assert summary["entity_counts"] == {}

    async def test_missing_artifact_ids_rejected(self, context: KaosContext) -> None:
        tool = CorpusSummarizeTool()
        result = await tool.execute({}, context)
        assert result.isError
        assert "artifact_ids" in _error_text(result)


# ---------------------------------------------------------------------------
# CorpusNarrowTool
# ---------------------------------------------------------------------------


class TestCorpusNarrowMetadata:
    def test_name_and_annotations(self) -> None:
        tool = CorpusNarrowTool()
        assert tool.metadata.name == "kaos-content-corpus-narrow"
        ann = tool.metadata.annotations
        assert ann is not None
        assert ann.readOnlyHint is True


class TestCorpusNarrow:
    async def test_query_picks_relevant_artifact(self, context: KaosContext) -> None:
        """A query for confidentiality should rank the NDA above the MSA."""
        a_nda = await _store(_nda_doc(), context, name="nda")
        a_msa = await _store(_msa_doc(), context, name="msa")
        tool = CorpusNarrowTool()
        result = await tool.execute(
            {
                "query": "confidential information disclosure",
                "artifact_ids": [a_nda, a_msa],
            },
            context,
        )
        assert not result.isError
        selected = _payload(result)["selected"]
        assert len(selected) >= 1
        assert selected[0]["artifact_id"] == a_nda, (
            f"Expected NDA top-ranked for confidentiality query; "
            f"selected: {[s['artifact_id'] for s in selected]}"
        )

    async def test_query_picks_msa_for_billing_query(self, context: KaosContext) -> None:
        """A query about billing should rank the MSA above the NDA."""
        a_nda = await _store(_nda_doc(), context, name="nda")
        a_msa = await _store(_msa_doc(), context, name="msa")
        tool = CorpusNarrowTool()
        result = await tool.execute(
            {
                "query": "hourly billing rate payment terms",
                "artifact_ids": [a_nda, a_msa],
            },
            context,
        )
        selected = _payload(result)["selected"]
        # MSA mentions "billed monthly", "hour", "payment terms"
        # NDA does not, so MSA should rank first.
        assert selected[0]["artifact_id"] == a_msa

    async def test_top_k_caps_results(self, context: KaosContext) -> None:
        a_nda = await _store(_nda_doc(), context, name="nda")
        a_msa = await _store(_msa_doc(), context, name="msa")
        tool = CorpusNarrowTool()
        result = await tool.execute(
            {"query": "agreement", "artifact_ids": [a_nda, a_msa], "top_k": 1},
            context,
        )
        selected = _payload(result)["selected"]
        assert len(selected) <= 1

    async def test_each_hit_has_distinguishing_ngrams(self, context: KaosContext) -> None:
        a_nda = await _store(_nda_doc(), context, name="nda")
        a_msa = await _store(_msa_doc(), context, name="msa")
        tool = CorpusNarrowTool()
        result = await tool.execute({"query": "agreement", "artifact_ids": [a_nda, a_msa]}, context)
        for hit in _payload(result)["selected"]:
            assert "distinguishing_ngrams" in hit
            assert isinstance(hit["distinguishing_ngrams"], list)
            assert "head_snippet" in hit
            assert "entity_counts" in hit
            assert "score" in hit

    async def test_missing_query_rejected(self, context: KaosContext) -> None:
        a = await _store(_nda_doc(), context, name="nda")
        tool = CorpusNarrowTool()
        result = await tool.execute({"artifact_ids": [a]}, context)
        assert result.isError
        assert "query" in _error_text(result).lower()

    async def test_missing_artifact_ids_rejected(self, context: KaosContext) -> None:
        tool = CorpusNarrowTool()
        result = await tool.execute({"query": "x"}, context)
        assert result.isError
        assert "artifact_ids" in _error_text(result)

    async def test_top_k_zero_rejected(self, context: KaosContext) -> None:
        a = await _store(_nda_doc(), context, name="nda")
        tool = CorpusNarrowTool()
        result = await tool.execute({"query": "x", "artifact_ids": [a], "top_k": 0}, context)
        assert result.isError

    async def test_unloadable_artifacts_skipped(self, context: KaosContext) -> None:
        """Bad artifact IDs are silently skipped; valid ones still get ranked."""
        a = await _store(_nda_doc(), context, name="nda")
        tool = CorpusNarrowTool()
        result = await tool.execute({"query": "agreement", "artifact_ids": [a, "bogus"]}, context)
        # Tool doesn't fail; output reflects what was searchable
        assert not result.isError
        out = _payload(result)
        assert out["total_requested"] == 2
        assert out["total_searched"] == 1


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_both_tools_registered(self, runtime: KaosRuntime) -> None:
        names = [t.metadata.name for t in runtime.tools.list_tool_objects()]
        assert "kaos-content-corpus-summarize" in names
        assert "kaos-content-corpus-narrow" in names
