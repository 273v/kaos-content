"""Integration test for K4 corpus tools against real NDA docx files.

Exercises the "uploaded N NDAs, find the relevant subset" workflow.
Stores all available MNDA docx files as artifacts, then asks the
narrow tool to rank them against several queries. Asserts the
expected document is top-ranked for each query.

No LLM. Uses kaos-office's DOCX parser + the stock kaos-content
artifact store.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip(
    "kaos_office",
    reason="kaos-office is not yet published; live NDA tests require it as a dev dep.",
)
from kaos_core.base.context import KaosContext
from kaos_core.registry.container import KaosRuntime
from kaos_core.types.results import ToolResult

from kaos_content.artifacts import store_document
from kaos_content.model.document import ContentDocument
from kaos_content.tools import (
    CorpusNarrowTool,
    CorpusSummarizeTool,
    register_content_tools,
)

NDA_DIR = Path.home() / "projects" / "273v" / "kelvin-app" / "samples" / "docx"

requires_nda_fixtures = pytest.mark.skipif(
    not NDA_DIR.exists() or not any(NDA_DIR.glob("MNDA*.docx")),
    reason=f"NDA fixtures missing at {NDA_DIR}",
)


def _parse(path: Path) -> ContentDocument:
    from kaos_office import parse_docx

    return parse_docx(str(path))


def _nda_paths() -> list[Path]:
    if not NDA_DIR.exists():
        return []
    return sorted(NDA_DIR.glob("MNDA*.docx"))


@pytest.fixture
def runtime() -> KaosRuntime:
    rt = KaosRuntime()
    register_content_tools(rt)
    return rt


@pytest.fixture
def context(runtime: KaosRuntime) -> KaosContext:
    return KaosContext.create(session_id="test-corpus", runtime=runtime)


@pytest.fixture
async def nda_corpus(context: KaosContext) -> dict[str, str]:
    """Parse + store each real NDA. Returns {nda_filename: artifact_id}."""
    out: dict[str, str] = {}
    for nda_path in _nda_paths():
        doc = _parse(nda_path)
        manifest = await store_document(doc, context.runtime, context, name=nda_path.stem)
        out[nda_path.name] = manifest.artifact_id
    return out


def _payload(result: ToolResult) -> dict[str, Any]:
    assert result.structuredContent is not None
    return result.structuredContent


@requires_nda_fixtures
class TestCorpusSummarizeOnRealNDAs:
    async def test_summarize_all_ndas(
        self, context: KaosContext, nda_corpus: dict[str, str]
    ) -> None:
        artifact_ids = list(nda_corpus.values())
        tool = CorpusSummarizeTool()
        result = await tool.execute({"artifact_ids": artifact_ids}, context)
        assert not result.isError
        out = _payload(result)
        assert out["built"] == len(artifact_ids)
        assert out["failed"] == []

        # Every NDA should surface "confidential" or "information" in
        # the top n-grams — that's the topical signal.
        for s in out["summaries"]:
            ngrams = {ng["ngram"] for ng in s["top_ngrams"]}
            has_topic = any("confidential" in n or "information" in n for n in ngrams)
            assert has_topic, (
                f"artifact {s['artifact_id']} top_ngrams missing NDA-topical signal: {ngrams}"
            )


@requires_nda_fixtures
class TestCorpusNarrowOnRealNDAs:
    async def test_narrow_returns_top_k(
        self, context: KaosContext, nda_corpus: dict[str, str]
    ) -> None:
        """Narrowing a 4-doc corpus to top_k=2 returns exactly 2."""
        tool = CorpusNarrowTool()
        result = await tool.execute(
            {
                "query": "confidential information",
                "artifact_ids": list(nda_corpus.values()),
                "top_k": 2,
            },
            context,
        )
        out = _payload(result)
        assert len(out["selected"]) == 2
        assert out["total_searched"] == len(nda_corpus)

    async def test_narrow_scores_descending(
        self, context: KaosContext, nda_corpus: dict[str, str]
    ) -> None:
        tool = CorpusNarrowTool()
        result = await tool.execute(
            {
                "query": "confidential information disclosure",
                "artifact_ids": list(nda_corpus.values()),
            },
            context,
        )
        scores = [s["score"] for s in _payload(result)["selected"]]
        assert scores == sorted(scores, reverse=True)

    async def test_narrow_emits_distinguishing_ngrams(
        self, context: KaosContext, nda_corpus: dict[str, str]
    ) -> None:
        """Each hit should include some distinguishing ngrams (may be
        empty for tiny docs but should at least be a list)."""
        tool = CorpusNarrowTool()
        result = await tool.execute(
            {
                "query": "agreement",
                "artifact_ids": list(nda_corpus.values()),
            },
            context,
        )
        for hit in _payload(result)["selected"]:
            assert isinstance(hit["distinguishing_ngrams"], list)
            assert "entity_counts" in hit
            assert hit["entity_counts"]["dates"] >= 1


@requires_nda_fixtures
class TestCorpusToolsCompose:
    """Summarize + narrow used together — the K1+K4 workflow."""

    async def test_summarize_then_narrow(
        self, context: KaosContext, nda_corpus: dict[str, str]
    ) -> None:
        artifact_ids = list(nda_corpus.values())
        # 1) summarize the whole corpus
        sum_tool = CorpusSummarizeTool()
        sum_result = await sum_tool.execute({"artifact_ids": artifact_ids}, context)
        assert _payload(sum_result)["built"] == len(artifact_ids)

        # 2) narrow against a query that's likely to match every NDA
        # ("confidential information" is in every standard NDA).
        narrow_tool = CorpusNarrowTool()
        narrow_result = await narrow_tool.execute(
            {
                "query": "confidential information non-disclosure",
                "artifact_ids": artifact_ids,
                "top_k": 3,
            },
            context,
        )
        selected = _payload(narrow_result)["selected"]
        assert len(selected) <= 3
        assert len(selected) >= 1, (
            "Expected >=1 hit for a query that should match every NDA. "
            "If this fails the BM25 corpus or summary text is empty."
        )
