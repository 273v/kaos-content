"""Tests for kaos-content Phase 7: VFS integration and artifact helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
from kaos_core import (
    ArtifactStore,
    KaosContext,
    KaosRuntime,
    KaosSettings,
    VFSConfig,
    VirtualFileSystem,
)
from kaos_core.types.enums import StorageBackend

from kaos_content import (
    ContentDocument,
    DocumentBuilder,
    DocumentMetadata,
    Paragraph,
    Text,
    parse_markdown,
)
from kaos_content.artifacts import (
    document_annotations_by_type,
    document_definitions,
    document_metadata,
    document_node_subtree,
    document_outline,
    document_tables_summary,
    document_to_resource_views,
    document_to_summary,
    load_document,
    store_document,
    unique_document_name,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_runtime(tmp_path: Path) -> KaosRuntime:
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


def _sample_document() -> ContentDocument:
    """Build a sample document with headings, paragraphs, and definitions."""
    builder = DocumentBuilder(title="Test Document")
    builder.set_metadata(authors=("Alice",))
    builder.heading(1, "Introduction")
    builder.paragraph("This is the first paragraph.")
    builder.heading(2, "Background")
    builder.paragraph("Some background information here.")
    builder.heading(1, "Methods")
    builder.paragraph("We used the following methods.")
    builder.add_definition("API", "https://example.com/api")
    return builder.build()


def _sample_markdown() -> str:
    return """\
---
title: Sample Report
---

# Overview

This report covers findings from Q1 2026.

## Key Metrics

Revenue grew by 15%.

| Metric | Value |
|--------|-------|
| Revenue | $1.5M |
| Users | 10,000 |

## Definitions

API
:   Application Programming Interface

MCP
:   Model Context Protocol
"""


# ---------------------------------------------------------------------------
# store / load round-trip
# ---------------------------------------------------------------------------


async def test_store_and_load_json(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    context = KaosContext.create(session_id="test", runtime=runtime)
    doc = _sample_document()

    manifest = await store_document(doc, runtime, context, name="test-doc")
    assert manifest.mime_type == "application/json"
    assert manifest.size > 0
    assert manifest.name == "test-doc"

    loaded = await load_document(manifest.artifact_id, runtime)
    assert loaded.metadata.title == "Test Document"
    assert len(loaded.body) == len(doc.body)


async def test_store_and_load_markdown(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    context = KaosContext.create(session_id="test", runtime=runtime)
    doc = _sample_document()

    manifest = await store_document(doc, runtime, context, name="test-doc-md", format="markdown")
    assert manifest.mime_type == "text/markdown"
    assert manifest.size > 0

    # Markdown format can't be loaded back as JSON
    text = await runtime.artifacts.read_text(manifest.artifact_id)
    assert "Introduction" in text
    assert "Background" in text


async def test_store_with_custom_metadata(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    context = KaosContext.create(session_id="test", runtime=runtime)
    doc = _sample_document()

    manifest = await store_document(
        doc,
        runtime,
        context,
        name="custom",
        description="A custom document",
        metadata={"source": "test"},
    )
    assert manifest.description == "A custom document"
    assert manifest.metadata == {"source": "test"}


async def test_load_by_artifact_ref(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    context = KaosContext.create(session_id="test", runtime=runtime)
    doc = _sample_document()

    manifest = await store_document(doc, runtime, context, name="ref-test")
    ref = manifest.to_ref()

    loaded = await load_document(ref, runtime)
    assert loaded.metadata.title == doc.metadata.title


# ---------------------------------------------------------------------------
# Resource view helpers
# ---------------------------------------------------------------------------


def test_document_outline() -> None:
    doc = parse_markdown(_sample_markdown())
    outline = document_outline(doc)

    assert len(outline) >= 3
    assert outline[0]["depth"] == 1
    assert "Overview" in outline[0]["text"]

    depths = [h["depth"] for h in outline]
    assert 2 in depths


def test_document_tables_summary() -> None:
    doc = parse_markdown(_sample_markdown())
    tables = document_tables_summary(doc)

    assert len(tables) >= 1
    assert tables[0]["rows"] >= 2  # header + data
    assert tables[0]["cols"] >= 2


def test_document_annotations_empty() -> None:
    doc = _sample_document()
    annotations = document_annotations_by_type(doc)
    assert annotations == []


def test_document_annotations_with_filter() -> None:
    doc = _sample_document()
    annotations = document_annotations_by_type(doc, annotation_type="highlight")
    assert annotations == []


def test_document_definitions() -> None:
    doc = _sample_document()
    defs = document_definitions(doc)
    assert defs == {"API": "https://example.com/api"}


def test_document_metadata_helper() -> None:
    doc = _sample_document()
    meta = document_metadata(doc)
    assert meta["title"] == "Test Document"
    assert meta["authors"] == ["Alice"]


def test_document_node_subtree() -> None:
    doc = _sample_document()
    # First body node is a heading
    subtree = document_node_subtree(doc, "#/body/0")
    assert subtree["node_type"] == "heading"
    assert subtree["depth"] == 1


def test_document_node_subtree_missing_raises() -> None:
    doc = _sample_document()
    with pytest.raises(KeyError, match="not found"):
        document_node_subtree(doc, "#/body/999")


def test_document_to_summary_short() -> None:
    doc = ContentDocument(
        metadata=DocumentMetadata(title="Short"),
        body=(Paragraph(children=(Text(value="Hello world."),)),),
    )
    summary = document_to_summary(doc)
    assert "Short" in summary
    assert "Hello world" in summary
    assert "1 blocks" in summary


def test_document_to_summary_long() -> None:
    long_text = "word " * 200
    doc = ContentDocument(
        metadata=DocumentMetadata(title="Long"),
        body=(Paragraph(children=(Text(value=long_text),)),),
    )
    summary = document_to_summary(doc, max_length=100)
    assert summary.endswith("...")
    assert len(summary) < len(long_text)


def test_document_to_resource_views() -> None:
    from kaos_core.artifacts.models import ArtifactManifest
    from kaos_core.types.enums import ArtifactRole

    doc = _sample_document()
    manifest = ArtifactManifest(
        artifact_id="test-id",
        session_id="s1",
        context_id="s1",
        name="test-doc",
        uri="kaos://artifacts/test-id",
        role=ArtifactRole.BODY,
        mime_type="application/json",
        size=1000,
        path="documents/test-doc.json",
    )

    views = document_to_resource_views(doc, manifest)
    assert "metadata" in views
    assert "outline" in views
    assert "tables" in views
    assert "annotations" in views
    assert "definitions" in views
    assert "markdown" in views
    assert views["metadata"]["title"] == "Test Document"
    assert isinstance(views["markdown"], str)
    assert "Introduction" in views["markdown"]


# ---------------------------------------------------------------------------
# Parsed document round-trip through artifacts
# ---------------------------------------------------------------------------


async def test_parsed_markdown_store_load_roundtrip(tmp_path: Path) -> None:
    """Parse markdown → store as JSON artifact → load → verify."""
    runtime = _make_runtime(tmp_path)
    context = KaosContext.create(session_id="test", runtime=runtime)

    doc = parse_markdown(_sample_markdown())
    manifest = await store_document(doc, runtime, context, name="parsed-md")

    loaded = await load_document(manifest.artifact_id, runtime)
    assert loaded.metadata.title == "Sample Report"
    assert len(loaded.body) == len(doc.body)

    # Verify outline survives round-trip
    original_outline = document_outline(doc)
    loaded_outline = document_outline(loaded)
    assert len(original_outline) == len(loaded_outline)
    for orig, loaded_h in zip(original_outline, loaded_outline, strict=True):
        assert orig["depth"] == loaded_h["depth"]
        assert orig["text"] == loaded_h["text"]


# ---------------------------------------------------------------------------
# ArtifactManifest.to_tool_result integration with content
# ---------------------------------------------------------------------------


async def test_to_tool_result_with_content_document(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    context = KaosContext.create(session_id="test", runtime=runtime)
    doc = _sample_document()

    manifest = await store_document(doc, runtime, context, name="result-test")
    summary = document_to_summary(doc)

    result = manifest.to_tool_result(summary=summary)
    assert not result.isError
    # Small doc → summary + resource link
    assert len(result.content) == 2
    assert result.content[0].type == "text"
    assert result.content[1].type == "resource_link"


# ---------------------------------------------------------------------------
# unique_document_name
# ---------------------------------------------------------------------------


class TestUniqueDocumentName:
    """Tests for the unique_document_name helper."""

    def test_basic_sanitization(self) -> None:
        name = unique_document_name("Miller Canfield | Firm")
        # Should be lowercase, special chars replaced with hyphens
        assert name.islower() or name.replace("-", "").isalnum()
        assert "|" not in name
        assert " " not in name

    def test_uniqueness(self) -> None:
        a = unique_document_name("same-base")
        b = unique_document_name("same-base")
        assert a != b, "Two calls with the same base should produce different names"

    def test_url_sanitization(self) -> None:
        name = unique_document_name("https://www.example.com/page?q=1&foo=bar")
        assert "://" not in name
        assert "?" not in name
        assert "&" not in name

    def test_max_length(self) -> None:
        long_base = "a" * 200
        name = unique_document_name(long_base, max_length=50)
        assert len(name) <= 50

    def test_empty_base(self) -> None:
        name = unique_document_name("")
        assert name.startswith("document-")
        assert len(name) == len("document-") + 8

    def test_special_chars_only(self) -> None:
        name = unique_document_name("!!!@@@###")
        assert name.startswith("document-")

    def test_contains_uuid_suffix(self) -> None:
        name = unique_document_name("test")
        # Should end with 8-char hex suffix after a hyphen
        parts = name.rsplit("-", 1)
        assert len(parts) == 2
        assert len(parts[1]) == 8
        int(parts[1], 16)  # Should be valid hex
