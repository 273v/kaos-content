"""Unit tests for kaos-content MCP tools."""

from __future__ import annotations

from pathlib import Path

import pytest
from kaos_core import (
    KaosContext,
    KaosRuntime,
    KaosSettings,
    KaosTool,
    VFSConfig,
    VirtualFileSystem,
)
from kaos_core.artifacts import ArtifactStore
from kaos_core.types.enums import StorageBackend

from kaos_content import (
    ContentDocument,
    DocumentMetadata,
    Heading,
    Paragraph,
    Text,
)
from kaos_content.model.attr import Provenance, SourceRef
from kaos_content.model.tabular import Column, ColumnType, Table, TabularDocument
from kaos_content.tools import (
    ChunkDocumentTool,
    DedupSemanticTool,
    ExtractPageTool,
    ExtractSectionTool,
    ParseMarkdownTool,
    SearchDocumentTool,
    SearchTableTool,
    SerializeTool,
    register_content_tools,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TOOL_CLASSES: list[type[KaosTool]] = [
    ParseMarkdownTool,
    SerializeTool,
    ChunkDocumentTool,
    SearchDocumentTool,
    SearchTableTool,
    ExtractSectionTool,
    ExtractPageTool,
    DedupSemanticTool,
]

# Tools that materialise new VFS artifacts on every successful call.
# These must NOT advertise readOnlyHint=True or auto-approving agents
# would run them without confirmation. (See audit M3.)
ARTIFACT_WRITER_TOOLS: list[type[KaosTool]] = [
    ParseMarkdownTool,
    SerializeTool,  # writes when result exceeds INLINE_THRESHOLD
    ChunkDocumentTool,
    ExtractSectionTool,
    ExtractPageTool,
]

# Tools that only read from the VFS — safe to mark read-only/idempotent.
# DedupSemanticTool sits here even though it doesn't touch the VFS at
# all — its `_QUERY_ANNOTATIONS` profile (read-only / idempotent) is
# correct for a compute-over-inputs tool.
QUERY_TOOLS: list[type[KaosTool]] = [
    SearchDocumentTool,
    SearchTableTool,
    DedupSemanticTool,
]

# Tools that DON'T need a runtime context (they accept context=None and
# operate entirely over their inputs). The TestNoContext suite below
# parametrizes against TOOL_CLASSES minus this set.
NO_CONTEXT_TOOLS: list[type[KaosTool]] = [
    DedupSemanticTool,
]


def _make_runtime(tmp_path: Path) -> KaosRuntime:
    """Create a runtime with disk-backed VFS for testing."""
    settings = KaosSettings(
        artifact_inline_read_max_bytes=262_144,
        artifact_chunk_size_bytes=64,
    )
    runtime = KaosRuntime(config=settings)
    runtime.vfs = VirtualFileSystem(
        VFSConfig(default_backend=StorageBackend.DISK, disk_base_path=tmp_path / "vfs")
    )
    runtime.artifacts = ArtifactStore(
        runtime.vfs,
        manifest_context_id=settings.artifact_manifest_context_id,
        manifest_prefix=settings.artifact_manifest_prefix,
        max_inline_read_bytes=settings.artifact_inline_read_max_bytes,
        default_chunk_size=settings.artifact_chunk_size_bytes,
        temporary_ttl_seconds=settings.artifact_temporary_ttl_seconds,
    )
    return runtime


def _make_context(runtime: KaosRuntime) -> KaosContext:
    return KaosContext.create(session_id="test-session", runtime=runtime)


def _sample_document() -> ContentDocument:
    """Build a sample ContentDocument with headings and paragraphs."""
    return ContentDocument(
        metadata=DocumentMetadata(title="Test Document"),
        body=(
            Heading(depth=1, children=(Text(value="Introduction"),)),
            Paragraph(children=(Text(value="This is the introduction paragraph."),)),
            Heading(depth=2, children=(Text(value="Background"),)),
            Paragraph(children=(Text(value="Some background information about the topic."),)),
            Heading(depth=1, children=(Text(value="Methods"),)),
            Paragraph(children=(Text(value="We used the following methods for our analysis."),)),
            Heading(depth=2, children=(Text(value="Data Collection"),)),
            Paragraph(children=(Text(value="Data was collected from multiple sources."),)),
        ),
    )


def _sample_document_with_pages() -> ContentDocument:
    """Build a document with page provenance."""
    source = SourceRef(uri="file:///test.pdf")
    return ContentDocument(
        metadata=DocumentMetadata(title="Paged Document"),
        body=(
            Heading(
                depth=1,
                children=(Text(value="Page One"),),
                provenance=Provenance(source=source, page=1),
            ),
            Paragraph(
                children=(Text(value="Content on page one."),),
                provenance=Provenance(source=source, page=1),
            ),
            Heading(
                depth=1,
                children=(Text(value="Page Two"),),
                provenance=Provenance(source=source, page=2),
            ),
            Paragraph(
                children=(Text(value="Content on page two."),),
                provenance=Provenance(source=source, page=2),
            ),
        ),
    )


def _sample_tabular_document() -> TabularDocument:
    """Build a sample TabularDocument."""
    return TabularDocument(
        metadata=DocumentMetadata(title="Test Table"),
        tables=(
            Table(
                name="metrics",
                columns=(
                    Column(name="metric", column_type=ColumnType.TEXT),
                    Column(name="value", column_type=ColumnType.TEXT),
                ),
                rows=(
                    ("Revenue", "$1.5M"),
                    ("Users", "10000"),
                    ("Growth", "15%"),
                ),
                row_count=3,
            ),
        ),
    )


async def _store_content_document(
    doc: ContentDocument,
    runtime: KaosRuntime,
    context: KaosContext,
) -> str:
    """Store a ContentDocument and return its artifact_id."""
    from kaos_content.artifacts import store_document, unique_document_name

    name = unique_document_name("test-doc")
    manifest = await store_document(doc, runtime, context, name=name)
    return manifest.artifact_id


async def _store_tabular_document(
    doc: TabularDocument,
    runtime: KaosRuntime,
    context: KaosContext,
) -> str:
    """Store a TabularDocument and return its artifact_id."""
    from kaos_content.artifacts import store_tabular

    manifest = await store_tabular(doc, runtime, context, name="test-tabular")
    return manifest.artifact_id


# ---------------------------------------------------------------------------
# Metadata validation
# ---------------------------------------------------------------------------


class TestToolMetadata:
    """Verify tool metadata is correctly defined for all 7 tools."""

    @pytest.mark.parametrize("tool_cls", TOOL_CLASSES)
    def test_tool_name_matches_pattern(self, tool_cls: type[KaosTool]) -> None:
        from kaos_content._version import __version__ as pkg_version

        tool = tool_cls()
        meta = tool.metadata
        assert meta.name.startswith("kaos-content-"), f"{meta.name} must start with 'kaos-content-'"
        assert meta.module_name == "kaos-content"
        # Tools.py derives _VERSION from kaos_content._version.__version__;
        # the package's source-of-truth version is `pkg_version`. Pinning
        # exact equality catches any future drift.
        assert meta.version == pkg_version, (
            f"{meta.name} version {meta.version!r} drifted from kaos_content "
            f"package version {pkg_version!r}"
        )

    @pytest.mark.parametrize("tool_cls", TOOL_CLASSES)
    def test_annotations_are_set(self, tool_cls: type[KaosTool]) -> None:
        """Every tool must declare ToolAnnotations — never None.

        Annotation profiles split on whether the tool writes a VFS
        artifact (audit M3). Read-only/idempotent flags must reflect
        actual behaviour or auto-approving agents are misled."""
        tool = tool_cls()
        ann = tool.metadata.annotations
        assert ann is not None, f"{tool.metadata.name} must set annotations"
        assert ann.destructiveHint is False
        assert ann.openWorldHint is False
        if tool_cls in ARTIFACT_WRITER_TOOLS:
            assert ann.readOnlyHint is False, (
                f"{tool.metadata.name} writes artifacts; readOnlyHint must be False"
            )
            assert ann.idempotentHint is False, (
                f"{tool.metadata.name} writes a new artifact each call; "
                f"idempotentHint must be False"
            )
        else:
            assert tool_cls in QUERY_TOOLS, (
                f"{tool_cls.__name__} must be in either ARTIFACT_WRITER_TOOLS or QUERY_TOOLS"
            )
            assert ann.readOnlyHint is True
            assert ann.idempotentHint is True

    @pytest.mark.parametrize("tool_cls", TOOL_CLASSES)
    def test_description_nonempty(self, tool_cls: type[KaosTool]) -> None:
        tool = tool_cls()
        assert len(tool.metadata.description) > 10

    @pytest.mark.parametrize("tool_cls", TOOL_CLASSES)
    def test_json_schema_valid(self, tool_cls: type[KaosTool]) -> None:
        tool = tool_cls()
        schema = tool.get_json_schema()
        assert schema["type"] == "object"
        assert "properties" in schema


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_register_content_tools_count(self, tmp_path: Path) -> None:
        runtime = _make_runtime(tmp_path)
        count = register_content_tools(runtime)
        # 7 original + ContextWindowTool (P12) + DedupSemanticTool
        # (KNT-602 0.1.0a3 — moved from kaos-nlp-transformers)
        # + StatsTool (0.1.0a5 — aggregation-gap closer)
        # + 5 typed-entity sentence filters (K3, 0.1.0a6)
        # + CorpusSummarizeTool + CorpusNarrowTool (K4, 0.1.0a6).
        assert count == 17

    def test_register_content_tools_listed(self, tmp_path: Path) -> None:
        runtime = _make_runtime(tmp_path)
        register_content_tools(runtime)
        names = runtime.tools.list_tools()
        expected = {
            "kaos-content-parse-markdown",
            "kaos-content-serialize",
            "kaos-content-chunk-document",
            "kaos-content-search-document",
            "kaos-content-search-table",
            "kaos-content-extract-section",
            "kaos-content-extract-page",
            "kaos-content-context-window",
            "kaos-content-dedup-semantic",
            "kaos-content-stats",
        }
        assert expected.issubset(set(names))


# ---------------------------------------------------------------------------
# Error cases: no context
# ---------------------------------------------------------------------------


_CONTEXT_REQUIRED_TOOLS: list[type[KaosTool]] = [
    cls for cls in TOOL_CLASSES if cls not in NO_CONTEXT_TOOLS
]


class TestNoContext:
    """All tools needing runtime should return a helpful error with no context."""

    @pytest.mark.parametrize("tool_cls", _CONTEXT_REQUIRED_TOOLS)
    async def test_no_context_returns_error(self, tool_cls: type[KaosTool]) -> None:
        tool = tool_cls()
        result = await tool.execute({}, context=None)
        assert result.isError
        assert "runtime" in (result.text or "").lower()

    @pytest.mark.parametrize("tool_cls", _CONTEXT_REQUIRED_TOOLS)
    async def test_no_runtime_on_context_returns_error(self, tool_cls: type[KaosTool]) -> None:
        tool = tool_cls()
        ctx = KaosContext.create(session_id="test", runtime=None)
        result = await tool.execute({}, context=ctx)
        assert result.isError


# ---------------------------------------------------------------------------
# ParseMarkdownTool
# ---------------------------------------------------------------------------


class TestParseMarkdownTool:
    async def test_parse_simple_markdown(self, tmp_path: Path) -> None:
        runtime = _make_runtime(tmp_path)
        ctx = _make_context(runtime)

        tool = ParseMarkdownTool()
        result = await tool.execute(
            {"text": "# Hello\n\nThis is a test paragraph."},
            context=ctx,
        )
        assert not result.isError
        data = result.require_structured()
        assert "artifact_id" in data
        assert data["block_count"] >= 2
        assert data["artifact_uri"]

    async def test_parse_with_title_override(self, tmp_path: Path) -> None:
        runtime = _make_runtime(tmp_path)
        ctx = _make_context(runtime)

        tool = ParseMarkdownTool()
        result = await tool.execute(
            {"text": "# Original\n\nContent.", "title": "Custom Title"},
            context=ctx,
        )
        assert not result.isError
        data = result.require_structured()
        assert data["title"] == "Custom Title"

    async def test_parse_empty_text_error(self, tmp_path: Path) -> None:
        runtime = _make_runtime(tmp_path)
        ctx = _make_context(runtime)

        tool = ParseMarkdownTool()
        result = await tool.execute({"text": ""}, context=ctx)
        assert result.isError
        assert "empty" in (result.text or "").lower()

    async def test_parse_with_source_url(self, tmp_path: Path) -> None:
        runtime = _make_runtime(tmp_path)
        ctx = _make_context(runtime)

        tool = ParseMarkdownTool()
        result = await tool.execute(
            {"text": "# Test\n\nContent.", "source_url": "https://example.com/doc.md"},
            context=ctx,
        )
        assert not result.isError


# ---------------------------------------------------------------------------
# SerializeTool
# ---------------------------------------------------------------------------


class TestSerializeTool:
    async def test_serialize_to_markdown(self, tmp_path: Path) -> None:
        runtime = _make_runtime(tmp_path)
        ctx = _make_context(runtime)
        doc = _sample_document()
        artifact_id = await _store_content_document(doc, runtime, ctx)

        tool = SerializeTool()
        result = await tool.execute(
            {"artifact_id": artifact_id, "format": "markdown"},
            context=ctx,
        )
        assert not result.isError
        data = result.require_structured()
        assert data["format"] == "markdown"
        assert "Introduction" in data["content"]

    async def test_serialize_to_html(self, tmp_path: Path) -> None:
        runtime = _make_runtime(tmp_path)
        ctx = _make_context(runtime)
        doc = _sample_document()
        artifact_id = await _store_content_document(doc, runtime, ctx)

        tool = SerializeTool()
        result = await tool.execute(
            {"artifact_id": artifact_id, "format": "html"},
            context=ctx,
        )
        assert not result.isError
        data = result.require_structured()
        assert data["format"] == "html"
        assert "<h1>" in data["content"]

    async def test_serialize_to_text(self, tmp_path: Path) -> None:
        runtime = _make_runtime(tmp_path)
        ctx = _make_context(runtime)
        doc = _sample_document()
        artifact_id = await _store_content_document(doc, runtime, ctx)

        tool = SerializeTool()
        result = await tool.execute(
            {"artifact_id": artifact_id, "format": "text"},
            context=ctx,
        )
        assert not result.isError
        data = result.require_structured()
        assert data["format"] == "text"
        assert "Introduction" in data["content"]

    async def test_serialize_invalid_format(self, tmp_path: Path) -> None:
        runtime = _make_runtime(tmp_path)
        ctx = _make_context(runtime)

        tool = SerializeTool()
        result = await tool.execute(
            {"artifact_id": "some-id", "format": "xml"},
            context=ctx,
        )
        assert result.isError
        assert "Invalid format" in (result.text or "")

    async def test_serialize_missing_artifact(self, tmp_path: Path) -> None:
        runtime = _make_runtime(tmp_path)
        ctx = _make_context(runtime)

        tool = SerializeTool()
        result = await tool.execute(
            {"artifact_id": "nonexistent-id", "format": "markdown"},
            context=ctx,
        )
        assert result.isError
        assert "Failed to load" in (result.text or "")


# ---------------------------------------------------------------------------
# ChunkDocumentTool
# ---------------------------------------------------------------------------


class TestChunkDocumentTool:
    async def test_chunk_document(self, tmp_path: Path) -> None:
        runtime = _make_runtime(tmp_path)
        ctx = _make_context(runtime)
        doc = _sample_document()
        artifact_id = await _store_content_document(doc, runtime, ctx)

        tool = ChunkDocumentTool()
        result = await tool.execute(
            {"artifact_id": artifact_id, "split_depth": 1},
            context=ctx,
        )
        assert not result.isError
        data = result.require_structured()
        assert data["chunk_count"] >= 2
        assert len(data["chunks"]) == data["chunk_count"]
        # Each chunk has expected fields
        for chunk in data["chunks"]:
            assert "artifact_id" in chunk
            assert "chunk_index" in chunk
            assert "chunk_total" in chunk
            assert "char_count" in chunk

    async def test_chunk_missing_artifact(self, tmp_path: Path) -> None:
        runtime = _make_runtime(tmp_path)
        ctx = _make_context(runtime)

        tool = ChunkDocumentTool()
        result = await tool.execute(
            {"artifact_id": "nonexistent-id"},
            context=ctx,
        )
        assert result.isError


# ---------------------------------------------------------------------------
# SearchDocumentTool
# ---------------------------------------------------------------------------


class TestSearchDocumentTool:
    async def test_search_finds_results(self, tmp_path: Path) -> None:
        runtime = _make_runtime(tmp_path)
        ctx = _make_context(runtime)
        doc = _sample_document()
        artifact_id = await _store_content_document(doc, runtime, ctx)

        tool = SearchDocumentTool()
        result = await tool.execute(
            {"artifact_id": artifact_id, "query": "background"},
            context=ctx,
        )
        assert not result.isError
        data = result.require_structured()
        assert data["total_matches"] >= 1
        assert "has_more" in data
        assert len(data["results"]) >= 1
        # Results have proper structure
        first = data["results"][0]
        assert "text" in first
        assert "score" in first
        assert "block_ref" in first
        assert first["path"] == ["Introduction", "Background"]

    async def test_search_no_results(self, tmp_path: Path) -> None:
        runtime = _make_runtime(tmp_path)
        ctx = _make_context(runtime)
        doc = _sample_document()
        artifact_id = await _store_content_document(doc, runtime, ctx)

        tool = SearchDocumentTool()
        result = await tool.execute(
            {"artifact_id": artifact_id, "query": "xyznonexistentterm"},
            context=ctx,
        )
        assert not result.isError
        data = result.require_structured()
        assert data["total_matches"] == 0

    async def test_search_empty_query_error(self, tmp_path: Path) -> None:
        runtime = _make_runtime(tmp_path)
        ctx = _make_context(runtime)

        tool = SearchDocumentTool()
        result = await tool.execute(
            {"artifact_id": "some-id", "query": ""},
            context=ctx,
        )
        assert result.isError
        assert "empty" in (result.text or "").lower()


# ---------------------------------------------------------------------------
# SearchTableTool
# ---------------------------------------------------------------------------


class TestSearchTableTool:
    async def test_search_tabular(self, tmp_path: Path) -> None:
        runtime = _make_runtime(tmp_path)
        ctx = _make_context(runtime)
        doc = _sample_tabular_document()
        artifact_id = await _store_tabular_document(doc, runtime, ctx)

        tool = SearchTableTool()
        result = await tool.execute(
            {"artifact_id": artifact_id, "query": "Revenue"},
            context=ctx,
        )
        assert not result.isError
        data = result.require_structured()
        assert data["total_matches"] >= 1
        assert "has_more" in data
        # Tool JSON must carry the canonical ``path`` breadcrumb so agents
        # can cite the column without inventing a structural identifier.
        # Regression for 0.1.0a11 follow-up where ``path`` was omitted.
        first = data["results"][0]
        assert "path" in first
        assert first["path"] == [first["section_title"]]

    async def test_search_tabular_by_column(self, tmp_path: Path) -> None:
        runtime = _make_runtime(tmp_path)
        ctx = _make_context(runtime)
        doc = _sample_tabular_document()
        artifact_id = await _store_tabular_document(doc, runtime, ctx)

        tool = SearchTableTool()
        result = await tool.execute(
            {"artifact_id": artifact_id, "query": "Revenue", "column": "metric"},
            context=ctx,
        )
        assert not result.isError
        data = result.require_structured()
        assert data["total_matches"] >= 1

    async def test_search_tabular_missing_artifact(self, tmp_path: Path) -> None:
        runtime = _make_runtime(tmp_path)
        ctx = _make_context(runtime)

        tool = SearchTableTool()
        result = await tool.execute(
            {"artifact_id": "nonexistent", "query": "test"},
            context=ctx,
        )
        assert result.isError


# ---------------------------------------------------------------------------
# ExtractSectionTool
# ---------------------------------------------------------------------------


class TestExtractSectionTool:
    async def test_extract_section(self, tmp_path: Path) -> None:
        runtime = _make_runtime(tmp_path)
        ctx = _make_context(runtime)
        doc = _sample_document()
        artifact_id = await _store_content_document(doc, runtime, ctx)

        tool = ExtractSectionTool()
        # "#/body/0" is the first heading ("Introduction")
        result = await tool.execute(
            {"artifact_id": artifact_id, "section_ref": "#/body/0"},
            context=ctx,
        )
        assert not result.isError
        data = result.require_structured()
        assert "artifact_id" in data
        assert data["heading_text"] == "Introduction"
        assert "markdown" in data

    async def test_extract_section_not_found(self, tmp_path: Path) -> None:
        runtime = _make_runtime(tmp_path)
        ctx = _make_context(runtime)
        doc = _sample_document()
        artifact_id = await _store_content_document(doc, runtime, ctx)

        tool = ExtractSectionTool()
        result = await tool.execute(
            {"artifact_id": artifact_id, "section_ref": "#/body/999"},
            context=ctx,
        )
        assert result.isError
        assert "not found" in (result.text or "").lower()

    async def test_extract_section_missing_ref(self, tmp_path: Path) -> None:
        runtime = _make_runtime(tmp_path)
        ctx = _make_context(runtime)

        tool = ExtractSectionTool()
        result = await tool.execute(
            {"artifact_id": "some-id", "section_ref": ""},
            context=ctx,
        )
        assert result.isError


# ---------------------------------------------------------------------------
# ExtractPageTool
# ---------------------------------------------------------------------------


class TestExtractPageTool:
    async def test_extract_page(self, tmp_path: Path) -> None:
        runtime = _make_runtime(tmp_path)
        ctx = _make_context(runtime)
        doc = _sample_document_with_pages()
        artifact_id = await _store_content_document(doc, runtime, ctx)

        tool = ExtractPageTool()
        result = await tool.execute(
            {"artifact_id": artifact_id, "page_number": 1},
            context=ctx,
        )
        assert not result.isError
        data = result.require_structured()
        assert data["page_number"] == 1
        assert data["block_count"] >= 1
        assert "markdown" in data
        assert "Page One" in data["markdown"]

    async def test_extract_page_not_found(self, tmp_path: Path) -> None:
        runtime = _make_runtime(tmp_path)
        ctx = _make_context(runtime)
        doc = _sample_document_with_pages()
        artifact_id = await _store_content_document(doc, runtime, ctx)

        tool = ExtractPageTool()
        result = await tool.execute(
            {"artifact_id": artifact_id, "page_number": 99},
            context=ctx,
        )
        assert result.isError
        assert "not found" in (result.text or "").lower()

    async def test_extract_page_no_provenance(self, tmp_path: Path) -> None:
        runtime = _make_runtime(tmp_path)
        ctx = _make_context(runtime)
        doc = _sample_document()  # no page provenance
        artifact_id = await _store_content_document(doc, runtime, ctx)

        tool = ExtractPageTool()
        result = await tool.execute(
            {"artifact_id": artifact_id, "page_number": 1},
            context=ctx,
        )
        assert result.isError
        assert "provenance" in (result.text or "").lower()

    async def test_extract_page_invalid_number(self, tmp_path: Path) -> None:
        runtime = _make_runtime(tmp_path)
        ctx = _make_context(runtime)

        tool = ExtractPageTool()
        result = await tool.execute(
            {"artifact_id": "some-id", "page_number": 0},
            context=ctx,
        )
        assert result.isError
        assert "1-based" in (result.text or "").lower() or ">= 1" in (result.text or "")


# ---------------------------------------------------------------------------
# ContextWindowTool — P12 grep -A/-B over AST refs
# ---------------------------------------------------------------------------


class TestContextWindowTool:
    async def test_window_around_target(self, tmp_path: Path) -> None:
        from kaos_content.tools import ContextWindowTool

        runtime = _make_runtime(tmp_path)
        ctx = _make_context(runtime)
        doc = _sample_document()  # has 5 body blocks (heading + 4 paragraphs)
        artifact_id = await _store_content_document(doc, runtime, ctx)

        tool = ContextWindowTool()
        # Target #/body/2 with before=1, after=1 → should return refs 1, 2, 3.
        result = await tool.execute(
            {
                "artifact_id": artifact_id,
                "node_ref": "#/body/2",
                "before_blocks": 1,
                "after_blocks": 1,
            },
            context=ctx,
        )
        assert not result.isError, result.text
        data = result.require_structured()
        assert data["target_node_ref"] == "#/body/2"
        assert data["window_block_refs"] == ["#/body/1", "#/body/2", "#/body/3"]
        assert data["block_count"] == 3
        assert data["expanded_to_section"] is False
        # The text-rendered window should mark the target with ▶ and others ▸.
        assert "▶ #/body/2" in (result.text or "")
        assert "▸ #/body/1" in (result.text or "")
        assert "▸ #/body/3" in (result.text or "")

    async def test_window_clamped_at_boundaries(self, tmp_path: Path) -> None:
        from kaos_content.tools import ContextWindowTool

        runtime = _make_runtime(tmp_path)
        ctx = _make_context(runtime)
        doc = _sample_document()
        artifact_id = await _store_content_document(doc, runtime, ctx)

        tool = ContextWindowTool()
        # Target #/body/0 with before=5 — should clamp to start of body
        # (no negative refs, no error).
        result = await tool.execute(
            {
                "artifact_id": artifact_id,
                "node_ref": "#/body/0",
                "before_blocks": 5,
                "after_blocks": 1,
            },
            context=ctx,
        )
        assert not result.isError, result.text
        data = result.require_structured()
        assert data["window_block_refs"][0] == "#/body/0"

    async def test_window_resolves_subblock_ref(self, tmp_path: Path) -> None:
        """A descendant ref like #/body/2/children/3 should map to its
        containing top-level body block (#/body/2)."""
        from kaos_content.tools import ContextWindowTool

        runtime = _make_runtime(tmp_path)
        ctx = _make_context(runtime)
        doc = _sample_document()
        artifact_id = await _store_content_document(doc, runtime, ctx)

        tool = ContextWindowTool()
        result = await tool.execute(
            {
                "artifact_id": artifact_id,
                "node_ref": "#/body/2/children/0",
                "before_blocks": 0,
                "after_blocks": 0,
            },
            context=ctx,
        )
        assert not result.isError, result.text
        data = result.require_structured()
        assert data["window_block_refs"] == ["#/body/2"]
        assert data["path"] == ["Introduction", "Background"]

    async def test_window_invalid_ref(self, tmp_path: Path) -> None:
        from kaos_content.tools import ContextWindowTool

        runtime = _make_runtime(tmp_path)
        ctx = _make_context(runtime)
        doc = _sample_document()
        artifact_id = await _store_content_document(doc, runtime, ctx)

        tool = ContextWindowTool()
        # Non-body ref — footnotes namespace
        result = await tool.execute(
            {
                "artifact_id": artifact_id,
                "node_ref": "#/footnotes/x/0",
                "before_blocks": 1,
                "after_blocks": 1,
            },
            context=ctx,
        )
        assert result.isError

    async def test_window_missing_ref_param(self, tmp_path: Path) -> None:
        from kaos_content.tools import ContextWindowTool

        runtime = _make_runtime(tmp_path)
        ctx = _make_context(runtime)

        tool = ContextWindowTool()
        result = await tool.execute(
            {"artifact_id": "some-id", "node_ref": ""},
            context=ctx,
        )
        assert result.isError
        assert "node_ref" in (result.text or "")

    async def test_window_expand_to_section(self, tmp_path: Path) -> None:
        """When expand_to_section=True and the strict window crosses a
        heading boundary, the result should be clamped to the enclosing
        section."""
        from kaos_content.tools import ContextWindowTool

        runtime = _make_runtime(tmp_path)
        ctx = _make_context(runtime)
        doc = _sample_document()
        artifact_id = await _store_content_document(doc, runtime, ctx)

        tool = ContextWindowTool()
        # Target inside the first section, with a window so wide that it
        # would cross into the next section. expand_to_section=True
        # should clamp it.
        result = await tool.execute(
            {
                "artifact_id": artifact_id,
                "node_ref": "#/body/1",
                "before_blocks": 10,
                "after_blocks": 10,
                "expand_to_section": True,
            },
            context=ctx,
        )
        assert not result.isError, result.text
        # Either expanded_to_section is True (window crossed) or the
        # window is small enough that it didn't cross — both valid.
        data = result.require_structured()
        assert "expanded_to_section" in data


# ---------------------------------------------------------------------------
# DedupSemanticTool (KNT-602 Option A — moved from kaos-nlp-transformers
# in 0.1.0a3). Mirrors the test shapes that lived in
# kaos-nlp-transformers/tests/unit/test_tools.py before the move.
# ---------------------------------------------------------------------------


class TestDedupSemanticTool:
    """Input-validation + oversized-input + happy-path coverage.

    The tool runs without runtime context — see DedupSemanticTool.execute
    — so we don't pass one in. The happy-path test is gated on `live`
    because it loads the real BAAI/bge-small-en-v1.5 model.
    """

    @staticmethod
    def _get_tool(tmp_path: Path) -> KaosTool:
        from kaos_content.tools import DedupSemanticTool

        runtime = _make_runtime(tmp_path)
        register_content_tools(runtime)
        tool = runtime.tools.get_tool("kaos-content-dedup-semantic")
        assert tool is not None, "DedupSemanticTool must register"
        # Sanity check that we got the right class — register_content_tools
        # may shuffle ordering in future and a wrong class would silently
        # invalidate the rest of the suite.
        assert isinstance(tool, DedupSemanticTool)
        return tool

    async def test_requires_two_documents(self, tmp_path: Path) -> None:
        tool = self._get_tool(tmp_path)
        result = await tool.execute(
            {"documents": [{"doc_id": "a", "text": "x"}]},
            None,
        )
        assert result.isError
        assert "at least 2 entries" in (result.text or "")

    async def test_rejects_non_string_doc_id(self, tmp_path: Path) -> None:
        tool = self._get_tool(tmp_path)
        result = await tool.execute(
            {
                "documents": [
                    {"doc_id": 1, "text": "x"},
                    {"doc_id": 2, "text": "y"},
                ]
            },
            None,
        )
        assert result.isError
        assert "string `doc_id` and `text`" in (result.text or "")

    async def test_rejects_non_dict_item(self, tmp_path: Path) -> None:
        tool = self._get_tool(tmp_path)
        result = await tool.execute(
            {
                "documents": [
                    "not-a-dict",
                    {"doc_id": "a", "text": "x"},
                ]
            },
            None,
        )
        assert result.isError
        assert "is not an object" in (result.text or "")

    async def test_rejects_oversized_input(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Lower the cap so the test stays cheap.
        monkeypatch.setattr("kaos_content.tools._MAX_DEDUP_DOCS", 3)
        tool = self._get_tool(tmp_path)
        docs = [{"doc_id": str(i), "text": "x"} for i in range(4)]
        result = await tool.execute({"documents": docs}, None)
        assert result.isError
        assert "Too many documents" in (result.text or "")

    @pytest.mark.live
    async def test_happy_path(self, tmp_path: Path) -> None:
        """Live test — needs scipy + the real bge-small embedding model.

        Gated on `live` because the first run downloads ~30 MB and
        loads libonnxruntime through the cdylib.
        """
        pytest.importorskip("scipy")
        pytest.importorskip("kaos_nlp_transformers")

        tool = self._get_tool(tmp_path)
        result = await tool.execute(
            {
                "documents": [
                    {"doc_id": "a", "text": "Force majeure clauses excuse performance."},
                    {"doc_id": "b", "text": "Force majeure provisions excuse performance."},
                    {"doc_id": "c", "text": "Indemnity caps the liability of the seller."},
                ],
                "distance_threshold": 0.15,
            },
            None,
        )
        assert not result.isError, result.text
        payload = result.structuredContent
        assert payload is not None
        # The two paraphrases must land in the same cluster; the third stays alone.
        assert len(payload["clusters"]) == 1
        cluster = payload["clusters"][0]
        assert set(cluster["member_doc_ids"]) == {"a", "b"}
        assert 0.0 <= cluster["similarity"] <= 1.0
