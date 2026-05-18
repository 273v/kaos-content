"""MCP tool definitions for kaos-content document operations.

KaosTool implementations for parsing, serializing, chunking, searching,
and extracting from ContentDocument and TabularDocument ASTs.

Two annotation profiles are used:

- ``_QUERY_ANNOTATIONS`` — read-only, idempotent, local. Applied to
  search tools that only read existing artifacts.
- ``_ARTIFACT_WRITER_ANNOTATIONS`` — not read-only and not idempotent
  (each call writes a *new* artifact with a unique name) but still
  non-destructive and local-only. Applied to parse / chunk / extract /
  serialize-to-large which materialise new VFS artifacts.

Marking everything ``readOnlyHint=True`` would cause some agents
(e.g. Claude Code) to auto-approve calls that mutate the artifact
store. The audit (M3) flagged this; the split below is the fix.
"""

from __future__ import annotations

from typing import Any

from kaos_core.artifacts.models import INLINE_THRESHOLD
from kaos_core.base.context import KaosContext
from kaos_core.base.tool import KaosTool
from kaos_core.types.annotations import ToolAnnotations
from kaos_core.types.enums import ToolCapability, ToolCategory
from kaos_core.types.metadata import ToolMetadata
from kaos_core.types.parameters import ParameterSchema
from kaos_core.types.results import ToolResult

# Single-source the version: read from the package's _version module so
# a release bump (`_version.py`) automatically updates every tool's
# metadata without a manual edit here. Avoids the pre-KNT-602 pattern
# where this constant drifted ahead of / behind the package version.
from kaos_content._version import __version__ as _VERSION

_MODULE = "kaos-content"

# Hard cap on documents per kaos-content-dedup-semantic call. The MCP
# tool runs synchronously (with one asyncio.to_thread offload) and a
# misbehaving caller could otherwise embed millions of texts in one
# request. 5_000 is the same cap kaos-nlp-transformers used for the
# pre-KNT-602 version of this tool.
_MAX_DEDUP_DOCS: int = 5000

# Search and inspection tools — they read but do not mutate VFS.
_QUERY_ANNOTATIONS = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)

# Parse / chunk / extract / serialize-to-large tools — they create
# new VFS artifacts. Not read-only, not idempotent (each call
# materialises a fresh artifact under a unique name); but still
# non-destructive and local.
_ARTIFACT_WRITER_ANNOTATIONS = ToolAnnotations(
    readOnlyHint=False,
    destructiveHint=False,
    idempotentHint=False,
    openWorldHint=False,
)

_NO_CONTEXT_ERROR = (
    "No runtime context. This tool requires a running KAOS server with VFS. "
    "Register tools with a KaosRuntime, or expose them over MCP via the "
    "companion package kaos-mcp (ships separately)."
)

_LEVEL_ENUM = ["paragraph", "sentence"]
_FORMAT_ENUM = ["markdown", "html", "text"]


# ---------------------------------------------------------------------------
# 1. ParseMarkdownTool
# ---------------------------------------------------------------------------


class ParseMarkdownTool(KaosTool):
    """Parse markdown text into a ContentDocument and store as artifact."""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="kaos-content-parse-markdown",
            display_name="Parse Markdown",
            description=(
                "Parse markdown text into a ContentDocument AST and store as a VFS artifact. "
                "Returns a summary and artifact URI. Requires the [markdown] extra "
                "(markdown-it-py). Use kaos-content-serialize to convert back to another format."
            ),
            category=ToolCategory.DOCUMENT,
            capability=ToolCapability.TRANSFORM,
            module_name=_MODULE,
            version=_VERSION,
            annotations=_ARTIFACT_WRITER_ANNOTATIONS,
            input_schema=[
                ParameterSchema(
                    name="text",
                    type="string",
                    description="Markdown text to parse into a ContentDocument.",
                ),
                ParameterSchema(
                    name="title",
                    type="string",
                    description="Optional document title (overrides YAML front matter title).",
                    required=False,
                ),
                ParameterSchema(
                    name="source_url",
                    type="string",
                    description="Optional source URL for provenance tracking.",
                    required=False,
                ),
            ],
        )

    async def execute(
        self, inputs: dict[str, Any], context: KaosContext | None = None
    ) -> ToolResult:
        if context is None or context.runtime is None:
            return ToolResult.create_error(_NO_CONTEXT_ERROR)

        text = inputs.get("text", "")
        if not text or not text.strip():
            return ToolResult.create_error(
                "The 'text' parameter is required and must not be empty. "
                "Provide markdown text to parse."
            )

        title = inputs.get("title")
        source_url = inputs.get("source_url")

        try:
            from kaos_content.parsers.markdown import parse_markdown
        except ImportError:
            return ToolResult.create_error(
                "Markdown parser not available. Install the [markdown] extra: "
                "pip install 'kaos-content[markdown]'"
            )

        # Build source ref if URL provided
        source = None
        if source_url:
            from kaos_content.model.attr import SourceRef

            source = SourceRef(uri=source_url)

        try:
            doc = parse_markdown(text, source=source)
        except Exception as exc:
            return ToolResult.create_error(
                f"Markdown parsing failed: {exc}. Check that the input is valid markdown text."
            )

        # Override title if provided
        if title and doc.metadata.title != title:
            from kaos_content.model.metadata import DocumentMetadata

            doc = doc.model_copy(
                update={
                    "metadata": DocumentMetadata(
                        title=title,
                        **{k: v for k, v in doc.metadata.model_dump().items() if k != "title"},
                    )
                }
            )

        # Store as artifact
        from kaos_content.artifacts import document_to_summary, store_document, unique_document_name

        name = unique_document_name(title or "parsed-markdown")
        manifest = await store_document(
            doc,
            context.runtime,
            context,
            name=name,
            description=f"Parsed markdown: {title or 'untitled'}",
        )

        summary = document_to_summary(doc, max_length=300)

        return ToolResult.create_summary_with_resource(
            summary=f"Parsed markdown ({len(doc.body)} blocks). {summary}",
            uri=manifest.uri,
            name=manifest.name,
            mime_type="application/json",
            structured_content={
                "artifact_id": manifest.artifact_id,
                "artifact_uri": manifest.uri,
                "block_count": len(doc.body),
                "title": doc.metadata.title,
            },
        )


# ---------------------------------------------------------------------------
# 2. SerializeTool
# ---------------------------------------------------------------------------


class SerializeTool(KaosTool):
    """Serialize a stored ContentDocument to markdown, HTML, or plain text."""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="kaos-content-serialize",
            display_name="Serialize Document",
            description=(
                "Load a ContentDocument from a VFS artifact and serialize to markdown, "
                "HTML, or plain text. Returns the serialized text inline (or as artifact "
                "if large). Use kaos-content-parse-markdown first to create an artifact."
            ),
            category=ToolCategory.DOCUMENT,
            capability=ToolCapability.TRANSFORM,
            module_name=_MODULE,
            version=_VERSION,
            # May write a new artifact when the result exceeds INLINE_THRESHOLD.
            annotations=_ARTIFACT_WRITER_ANNOTATIONS,
            input_schema=[
                ParameterSchema(
                    name="artifact_id",
                    type="string",
                    description="Artifact ID of the stored ContentDocument.",
                ),
                ParameterSchema(
                    name="format",
                    type="string",
                    description="Output format: 'markdown', 'html', or 'text'.",
                    constraints={"enum": _FORMAT_ENUM},
                ),
                ParameterSchema(
                    name="include_provenance",
                    type="boolean",
                    description=(
                        "Include provenance data attributes in HTML output (default: true)."
                    ),
                    required=False,
                    default=True,
                ),
                ParameterSchema(
                    name="table_format",
                    type="string",
                    description=(
                        "Table rendering for text output: 'plain' (pipe-separated) "
                        "or 'csv' (comma-separated). Default: 'plain'."
                    ),
                    required=False,
                    default="plain",
                    constraints={"enum": ["plain", "csv"]},
                ),
            ],
        )

    async def execute(
        self, inputs: dict[str, Any], context: KaosContext | None = None
    ) -> ToolResult:
        if context is None or context.runtime is None:
            return ToolResult.create_error(_NO_CONTEXT_ERROR)

        artifact_id = inputs.get("artifact_id", "")
        fmt = inputs.get("format", "")
        include_provenance = inputs.get("include_provenance", True)
        table_format = inputs.get("table_format", "plain")

        if not artifact_id:
            return ToolResult.create_error(
                "The 'artifact_id' parameter is required. "
                "Use kaos-core-artifacts-list to find document artifact IDs."
            )

        if fmt not in _FORMAT_ENUM:
            return ToolResult.create_error(
                f"Invalid format '{fmt}'. Must be one of: {', '.join(_FORMAT_ENUM)}. "
                "Use 'markdown' for round-trip fidelity, 'text' for plain text, "
                "'html' for semantic HTML5."
            )

        from kaos_content.artifacts import load_document

        try:
            doc = await load_document(artifact_id, context.runtime)
        except Exception as exc:
            return ToolResult.create_error(
                f"Failed to load document artifact '{artifact_id}': {exc}. "
                "Verify the artifact_id with kaos-core-artifacts-list. "
                "Only JSON-format ContentDocument artifacts are supported."
            )

        try:
            if fmt == "markdown":
                from kaos_content.serializers.markdown import serialize_markdown

                result_text = serialize_markdown(doc)
            elif fmt == "html":
                from kaos_content.serializers.html import serialize_html

                result_text = serialize_html(doc, include_provenance=include_provenance)
            else:
                from kaos_content.serializers.text import serialize_text

                result_text = serialize_text(doc, table_format=table_format)
        except Exception as exc:
            return ToolResult.create_error(
                f"Serialization to '{fmt}' failed: {exc}. "
                "The document may have unexpected structure. Try a different format."
            )

        # Inline small results, store large ones as artifacts
        if len(result_text.encode("utf-8")) <= INLINE_THRESHOLD:
            return ToolResult.create_success(
                output={
                    "format": fmt,
                    "content": result_text,
                    "size": len(result_text),
                },
                summary=f"Serialized to {fmt} ({len(result_text)} chars)",
            )

        # Large result: store as artifact
        from kaos_content.artifacts import unique_document_name

        name = unique_document_name(f"serialized-{fmt}")
        ext_map = {"markdown": "md", "html": "html", "text": "txt"}
        mime_map = {"markdown": "text/markdown", "html": "text/html", "text": "text/plain"}

        vfs_path = f"serialized/{name}.{ext_map[fmt]}"
        ctx_path = context.get_vfs_path(vfs_path)
        await ctx_path.write_bytes(result_text.encode("utf-8"))

        manifest = await context.runtime.artifacts.create_from_path(
            vfs_path,
            context_id=context.session_id,
            session_id=context.session_id,
            name=name,
            description=f"Serialized ContentDocument ({fmt})",
            mime_type=mime_map[fmt],
        )

        return ToolResult.create_summary_with_resource(
            summary=(
                f"Serialized to {fmt} ({len(result_text)} chars, stored as artifact "
                f"— too large for inline). Read with kaos-core-vfs-read."
            ),
            uri=manifest.uri,
            name=manifest.name,
            mime_type=mime_map[fmt],
            structured_content={
                "format": fmt,
                "artifact_id": manifest.artifact_id,
                "artifact_uri": manifest.uri,
                "size": len(result_text),
            },
        )


# ---------------------------------------------------------------------------
# 3. ChunkDocumentTool
# ---------------------------------------------------------------------------


class ChunkDocumentTool(KaosTool):
    """Chunk a ContentDocument into smaller documents at heading boundaries."""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="kaos-content-chunk-document",
            display_name="Chunk Document",
            description=(
                "Split a ContentDocument into smaller chunks at heading boundaries, "
                "suitable for LLM processing. Each chunk is stored as a separate artifact. "
                "Returns chunk metadata (artifact IDs, character counts). "
                "Use kaos-content-parse-markdown or a PDF/Office tool first to create "
                "the source artifact."
            ),
            category=ToolCategory.DOCUMENT,
            capability=ToolCapability.TRANSFORM,
            module_name=_MODULE,
            version=_VERSION,
            annotations=_ARTIFACT_WRITER_ANNOTATIONS,
            input_schema=[
                ParameterSchema(
                    name="artifact_id",
                    type="string",
                    description="Artifact ID of the stored ContentDocument to chunk.",
                ),
                ParameterSchema(
                    name="max_chars",
                    type="integer",
                    description="Maximum characters per chunk (default: 8000). 0 = no limit.",
                    required=False,
                    default=8000,
                ),
                ParameterSchema(
                    name="split_depth",
                    type="integer",
                    description=(
                        "Heading depth at which to split (default: 2). "
                        "split_depth=2 splits at h1 and h2 but not h3+."
                    ),
                    required=False,
                    default=2,
                    constraints={"minimum": 1, "maximum": 6},
                ),
                ParameterSchema(
                    name="overlap_paragraphs",
                    type="integer",
                    description=(
                        "Number of trailing paragraphs from previous chunk to repeat "
                        "at the start of the next chunk for context overlap (default: 0)."
                    ),
                    required=False,
                    default=0,
                    constraints={"minimum": 0},
                ),
            ],
        )

    async def execute(
        self, inputs: dict[str, Any], context: KaosContext | None = None
    ) -> ToolResult:
        if context is None or context.runtime is None:
            return ToolResult.create_error(_NO_CONTEXT_ERROR)

        artifact_id = inputs.get("artifact_id", "")
        max_chars = inputs.get("max_chars", 8000)
        split_depth = inputs.get("split_depth", 2)
        overlap_paragraphs = inputs.get("overlap_paragraphs", 0)

        if not artifact_id:
            return ToolResult.create_error(
                "The 'artifact_id' parameter is required. "
                "Use kaos-core-artifacts-list to find document artifact IDs."
            )

        from kaos_content.artifacts import load_document, store_document, unique_document_name
        from kaos_content.chunking.section_chunker import SectionChunker
        from kaos_content.traversal.visitor import extract_text

        try:
            doc = await load_document(artifact_id, context.runtime)
        except Exception as exc:
            return ToolResult.create_error(
                f"Failed to load document artifact '{artifact_id}': {exc}. "
                "Verify the artifact_id with kaos-core-artifacts-list."
            )

        chunker = SectionChunker(
            max_chars=max_chars,
            split_depth=split_depth,
            overlap_paragraphs=overlap_paragraphs,
        )

        try:
            chunks = chunker.chunk(doc)
        except Exception as exc:
            return ToolResult.create_error(
                f"Chunking failed: {exc}. "
                "The document may have an unsupported structure. "
                "Try adjusting split_depth or max_chars."
            )

        # Store each chunk as a separate artifact
        chunk_infos: list[dict[str, Any]] = []
        for i, chunk in enumerate(chunks):
            name = unique_document_name(f"chunk-{i}")
            char_count = sum(len(extract_text(b)) for b in chunk.body)
            manifest = await store_document(
                chunk,
                context.runtime,
                context,
                name=name,
                description=f"Chunk {i + 1}/{len(chunks)}",
            )
            chunk_infos.append(
                {
                    "chunk_index": i,
                    "chunk_total": len(chunks),
                    "artifact_id": manifest.artifact_id,
                    "artifact_uri": manifest.uri,
                    "char_count": char_count,
                    "block_count": len(chunk.body),
                }
            )

        return ToolResult.create_success(
            output={
                "source_artifact_id": artifact_id,
                "chunk_count": len(chunks),
                "chunks": chunk_infos,
            },
            summary=(
                f"Split document into {len(chunks)} chunk(s). "
                f"Use kaos-content-serialize with each chunk artifact_id to read content."
            ),
        )


# ---------------------------------------------------------------------------
# 4. SearchDocumentTool
# ---------------------------------------------------------------------------


class SearchDocumentTool(KaosTool):
    """Search within a ContentDocument using BM25 or term frequency."""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="kaos-content-search-document",
            display_name="Search Document",
            description=(
                "Search within a stored ContentDocument by text query. Uses BM25 via "
                "kaos-nlp-core when available, falls back to term frequency scoring. "
                "Returns results with AST block_refs, page numbers, and section context. "
                "Sentence-level search requires kaos-nlp-core."
            ),
            category=ToolCategory.DOCUMENT,
            capability=ToolCapability.QUERY,
            module_name=_MODULE,
            version=_VERSION,
            annotations=_QUERY_ANNOTATIONS,
            input_schema=[
                ParameterSchema(
                    name="artifact_id",
                    type="string",
                    description="Artifact ID of the stored ContentDocument to search.",
                ),
                ParameterSchema(
                    name="query",
                    type="string",
                    description="Search query text.",
                ),
                ParameterSchema(
                    name="top_k",
                    type="integer",
                    description="Maximum number of results to return (default: 10).",
                    required=False,
                    default=10,
                    constraints={"minimum": 1, "maximum": 100},
                ),
                ParameterSchema(
                    name="level",
                    type="string",
                    description=(
                        "Search granularity: 'paragraph' or 'sentence'. "
                        "Sentence-level requires kaos-nlp-core. Default: 'paragraph'."
                    ),
                    required=False,
                    default="paragraph",
                    constraints={"enum": _LEVEL_ENUM},
                ),
                ParameterSchema(
                    name="preview_length",
                    type="integer",
                    description=(
                        "Maximum characters per result preview (default: 200). 0 = full text."
                    ),
                    required=False,
                    default=200,
                    constraints={"minimum": 0},
                ),
            ],
        )

    async def execute(
        self, inputs: dict[str, Any], context: KaosContext | None = None
    ) -> ToolResult:
        if context is None or context.runtime is None:
            return ToolResult.create_error(_NO_CONTEXT_ERROR)

        artifact_id = inputs.get("artifact_id", "")
        query = inputs.get("query", "")
        top_k = inputs.get("top_k", 10)
        level = inputs.get("level", "paragraph")
        preview_length = inputs.get("preview_length", 200)

        if not artifact_id:
            return ToolResult.create_error(
                "The 'artifact_id' parameter is required. "
                "Use kaos-core-artifacts-list to find document artifact IDs."
            )

        if not query or not query.strip():
            return ToolResult.create_error(
                "The 'query' parameter is required and must not be empty."
            )

        from kaos_content.artifacts import load_document
        from kaos_content.search import search_document

        try:
            doc = await load_document(artifact_id, context.runtime)
        except Exception as exc:
            return ToolResult.create_error(
                f"Failed to load document artifact '{artifact_id}': {exc}. "
                "Verify the artifact_id with kaos-core-artifacts-list."
            )

        try:
            results = search_document(
                doc,
                query,
                top_k=top_k,
                preview_length=preview_length,
                level=level,
            )
        except ImportError as exc:
            return ToolResult.create_error(
                f"Search failed: {exc}. "
                "Sentence-level search requires kaos-nlp-core. "
                "Use level='paragraph' or install kaos-nlp-core."
            )
        except ValueError as exc:
            return ToolResult.create_error(f"Search failed: {exc}")

        result_dicts = [
            {
                "text": r.text,
                "score": r.score,
                "block_ref": r.block_ref,
                "page": r.page,
                "section_ref": r.section_ref,
                "section_title": r.section_title,
                # Full structural breadcrumb (root-first, INCLUDING the
                # immediate section). Empty list signals "no enclosing
                # heading" — agents must NOT invent section identifiers
                # for hits with empty path. See
                # ``SearchResult.path`` docstring.
                "path": list(r.path),
            }
            for r in results.results
        ]

        return ToolResult.create_success(
            output={
                "query": results.query,
                "total_matches": results.total_matches,
                "has_more": results.has_more,
                "results": result_dicts,
            },
            summary=(
                f"Found {results.total_matches} match(es) for '{query}', "
                f"returning top {len(result_dicts)}."
            ),
        )


# ---------------------------------------------------------------------------
# 5. SearchTableTool
# ---------------------------------------------------------------------------


class SearchTableTool(KaosTool):
    """Search within a TabularDocument by text query."""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="kaos-content-search-table",
            display_name="Search Table",
            description=(
                "Search within a stored TabularDocument by text query. "
                "Performs case-insensitive substring matching across cell values. "
                "Can be scoped to a specific table and/or column. "
                "Use kaos-tabular tools or kaos-office-extract-xlsx first to create "
                "a TabularDocument artifact."
            ),
            category=ToolCategory.DATA,
            capability=ToolCapability.QUERY,
            module_name=_MODULE,
            version=_VERSION,
            annotations=_QUERY_ANNOTATIONS,
            input_schema=[
                ParameterSchema(
                    name="artifact_id",
                    type="string",
                    description="Artifact ID of the stored TabularDocument.",
                ),
                ParameterSchema(
                    name="query",
                    type="string",
                    description="Search query text (case-insensitive substring match).",
                ),
                ParameterSchema(
                    name="table_name",
                    type="string",
                    description="Restrict search to a specific table name.",
                    required=False,
                ),
                ParameterSchema(
                    name="column",
                    type="string",
                    description="Restrict search to a specific column name.",
                    required=False,
                ),
                ParameterSchema(
                    name="top_k",
                    type="integer",
                    description="Maximum number of results to return (default: 10).",
                    required=False,
                    default=10,
                    constraints={"minimum": 1, "maximum": 100},
                ),
            ],
        )

    async def execute(
        self, inputs: dict[str, Any], context: KaosContext | None = None
    ) -> ToolResult:
        if context is None or context.runtime is None:
            return ToolResult.create_error(_NO_CONTEXT_ERROR)

        artifact_id = inputs.get("artifact_id", "")
        query = inputs.get("query", "")
        table_name = inputs.get("table_name")
        column = inputs.get("column")
        top_k = inputs.get("top_k", 10)

        if not artifact_id:
            return ToolResult.create_error(
                "The 'artifact_id' parameter is required. "
                "Use kaos-core-artifacts-list to find tabular artifact IDs."
            )

        if not query or not query.strip():
            return ToolResult.create_error(
                "The 'query' parameter is required and must not be empty."
            )

        from kaos_content.artifacts import load_tabular
        from kaos_content.search import search_tabular

        try:
            doc = await load_tabular(artifact_id, context.runtime)
        except Exception as exc:
            return ToolResult.create_error(
                f"Failed to load tabular artifact '{artifact_id}': {exc}. "
                "Verify the artifact_id with kaos-core-artifacts-list. "
                "Only JSON-format TabularDocument artifacts are supported."
            )

        try:
            results = search_tabular(
                doc,
                query,
                table_name=table_name,
                column=column,
                top_k=top_k,
            )
        except ValueError as exc:
            return ToolResult.create_error(f"Search failed: {exc}")

        result_dicts = [
            {
                "text": r.text,
                "score": r.score,
                "block_ref": r.block_ref,
                "section_ref": r.section_ref,
                "section_title": r.section_title,
            }
            for r in results.results
        ]

        return ToolResult.create_success(
            output={
                "query": results.query,
                "total_matches": results.total_matches,
                "has_more": results.has_more,
                "results": result_dicts,
            },
            summary=(
                f"Found {results.total_matches} match(es) for '{query}', "
                f"returning top {len(result_dicts)}."
            ),
        )


# ---------------------------------------------------------------------------
# 6. ExtractSectionTool
# ---------------------------------------------------------------------------


class ExtractSectionTool(KaosTool):
    """Extract a section from a ContentDocument by heading ref."""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="kaos-content-extract-section",
            display_name="Extract Section",
            description=(
                "Extract a section from a stored ContentDocument by its heading "
                "JSON pointer ref (e.g. '#/body/5'). The section and its subsections "
                "are extracted as a standalone document and stored as a new artifact. "
                "Use kaos-content-search-document to find section_ref values, or "
                "kaos-core-artifacts-inspect to get the document outline."
            ),
            category=ToolCategory.DOCUMENT,
            capability=ToolCapability.EXTRACT,
            module_name=_MODULE,
            version=_VERSION,
            annotations=_ARTIFACT_WRITER_ANNOTATIONS,
            input_schema=[
                ParameterSchema(
                    name="artifact_id",
                    type="string",
                    description="Artifact ID of the stored ContentDocument.",
                ),
                ParameterSchema(
                    name="section_ref",
                    type="string",
                    description=(
                        "JSON pointer ref of the section heading "
                        "(e.g. '#/body/5'). Find these via search results or document outline."
                    ),
                ),
            ],
        )

    async def execute(
        self, inputs: dict[str, Any], context: KaosContext | None = None
    ) -> ToolResult:
        if context is None or context.runtime is None:
            return ToolResult.create_error(_NO_CONTEXT_ERROR)

        artifact_id = inputs.get("artifact_id", "")
        section_ref = inputs.get("section_ref", "")

        if not artifact_id:
            return ToolResult.create_error(
                "The 'artifact_id' parameter is required. "
                "Use kaos-core-artifacts-list to find document artifact IDs."
            )

        if not section_ref:
            return ToolResult.create_error(
                "The 'section_ref' parameter is required. "
                "Use kaos-content-search-document to find section_ref values, "
                "or examine the document outline."
            )

        from kaos_content.artifacts import load_document, store_document, unique_document_name
        from kaos_content.views.document_view import DocumentView

        try:
            doc = await load_document(artifact_id, context.runtime)
        except Exception as exc:
            return ToolResult.create_error(
                f"Failed to load document artifact '{artifact_id}': {exc}. "
                "Verify the artifact_id with kaos-core-artifacts-list."
            )

        view = DocumentView(doc)
        sv = view.section_by_ref(section_ref)

        if sv is None:
            headings = [s.heading_ref for s in view.flat_sections if s.heading_ref]
            return ToolResult.create_error(
                f"Section '{section_ref}' not found in the document. "
                "Use kaos-content-search-document to find valid section_ref values. "
                f"Available sections: {headings}"
            )

        # Extract section as markdown
        try:
            section_md = view.section_as_markdown(section_ref)
        except Exception as exc:
            return ToolResult.create_error(f"Failed to extract section '{section_ref}': {exc}.")

        # Build a standalone document from the section
        from kaos_content.model.document import ContentDocument
        from kaos_content.model.metadata import DocumentMetadata

        section_blocks = view.collect_section_blocks(sv)
        section_doc = ContentDocument(
            metadata=DocumentMetadata(
                title=sv.heading_text or doc.metadata.title,
            ),
            body=tuple(section_blocks),
        )

        name = unique_document_name(f"section-{sv.heading_text or 'extract'}")
        manifest = await store_document(
            section_doc,
            context.runtime,
            context,
            name=name,
            description=f"Extracted section: {sv.heading_text or section_ref}",
        )

        return ToolResult.create_summary_with_resource(
            summary=(
                f"Extracted section '{sv.heading_text or section_ref}' "
                f"({len(section_blocks)} blocks).\n\n{section_md}"
            ),
            uri=manifest.uri,
            name=manifest.name,
            mime_type="application/json",
            structured_content={
                "artifact_id": manifest.artifact_id,
                "artifact_uri": manifest.uri,
                "section_ref": section_ref,
                "heading_text": sv.heading_text,
                "block_count": len(section_blocks),
                "markdown": section_md,
            },
        )


# ---------------------------------------------------------------------------
# 7. ExtractPageTool
# ---------------------------------------------------------------------------


class ExtractPageTool(KaosTool):
    """Extract a page from a ContentDocument by 1-based page number."""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="kaos-content-extract-page",
            display_name="Extract Page",
            description=(
                "Extract a single page from a stored ContentDocument by 1-based "
                "page number. The page blocks are extracted as a standalone document "
                "and stored as a new artifact. Only works with documents that have "
                "page provenance (e.g., from PDF extraction). "
                "Use kaos-content-search-document to find page numbers."
            ),
            category=ToolCategory.DOCUMENT,
            capability=ToolCapability.EXTRACT,
            module_name=_MODULE,
            version=_VERSION,
            annotations=_ARTIFACT_WRITER_ANNOTATIONS,
            input_schema=[
                ParameterSchema(
                    name="artifact_id",
                    type="string",
                    description="Artifact ID of the stored ContentDocument.",
                ),
                ParameterSchema(
                    name="page_number",
                    type="integer",
                    description="1-based page number to extract.",
                    constraints={"minimum": 1},
                ),
            ],
        )

    async def execute(
        self, inputs: dict[str, Any], context: KaosContext | None = None
    ) -> ToolResult:
        if context is None or context.runtime is None:
            return ToolResult.create_error(_NO_CONTEXT_ERROR)

        artifact_id = inputs.get("artifact_id", "")
        page_number = inputs.get("page_number", 0)

        if not artifact_id:
            return ToolResult.create_error(
                "The 'artifact_id' parameter is required. "
                "Use kaos-core-artifacts-list to find document artifact IDs."
            )

        if page_number < 1:
            return ToolResult.create_error(
                "The 'page_number' parameter must be >= 1 (1-based page numbering)."
            )

        from kaos_content.artifacts import load_document, store_document, unique_document_name
        from kaos_content.views.document_view import DocumentView

        try:
            doc = await load_document(artifact_id, context.runtime)
        except Exception as exc:
            return ToolResult.create_error(
                f"Failed to load document artifact '{artifact_id}': {exc}. "
                "Verify the artifact_id with kaos-core-artifacts-list."
            )

        view = DocumentView(doc)

        if not view.has_pages:
            return ToolResult.create_error(
                "This document does not have page provenance. "
                "Page extraction requires documents from PDF or Office extraction "
                "that include page numbers in their provenance. "
                "Use kaos-content-extract-section for section-based extraction instead."
            )

        try:
            page_md = view.page_as_markdown(page_number)
        except KeyError:
            available = sorted(p.page_number for p in view.pages)
            return ToolResult.create_error(
                f"Page {page_number} not found. "
                f"Available pages: {available}. "
                "Check the page number and try again."
            )

        # Build standalone document from page blocks
        pv = view.page(page_number)

        from kaos_content.model.document import ContentDocument
        from kaos_content.model.metadata import DocumentMetadata

        page_doc = ContentDocument(
            metadata=DocumentMetadata(
                title=f"{doc.metadata.title or 'Document'} - Page {page_number}",
            ),
            body=pv.blocks,
        )

        name = unique_document_name(f"page-{page_number}")
        manifest = await store_document(
            page_doc,
            context.runtime,
            context,
            name=name,
            description=f"Page {page_number} extracted",
        )

        return ToolResult.create_summary_with_resource(
            summary=(f"Extracted page {page_number} ({len(pv.blocks)} blocks).\n\n{page_md}"),
            uri=manifest.uri,
            name=manifest.name,
            mime_type="application/json",
            structured_content={
                "artifact_id": manifest.artifact_id,
                "artifact_uri": manifest.uri,
                "page_number": page_number,
                "block_count": len(pv.blocks),
                "markdown": page_md,
            },
        )


# ---------------------------------------------------------------------------
# 8. ContextWindowTool — grep -A/-B style context expansion
# ---------------------------------------------------------------------------


class ContextWindowTool(KaosTool):
    """Expand around a node_ref with N blocks before and after.

    Solves a common agent failure mode: retrieval surfaces a paragraph
    that mentions the right keywords, but the actual answer is in the
    next or previous paragraph (definitions, list continuations,
    cross-reference tails). The agent should be able to "gesture" at
    a region and ask "show me the context around this".

    Mental model: ``grep -A N -B N`` for an AST. Returns the matching
    block + its N predecessors + its N successors as a coherent
    snippet, optionally expanded to the full enclosing section when
    the window crosses a heading boundary.
    """

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="kaos-content-context-window",
            display_name="Context Window",
            description=(
                "Show the document context around an AST node_ref — "
                "N blocks before and after the target — like ``grep -A -B``. "
                "Use after a search hit when you suspect the answer is "
                "*near* the matching paragraph (a definition above, a "
                "continuation list below, etc.). Returns the windowed "
                "blocks as plain text + structured refs. Set "
                "``expand_to_section=true`` to fall back to the full "
                "enclosing section when the window would cross a heading "
                "boundary. Find a node_ref via kaos-content-search-document "
                "or from the citations of a previous tool result."
            ),
            category=ToolCategory.DOCUMENT,
            capability=ToolCapability.QUERY,
            module_name=_MODULE,
            version=_VERSION,
            annotations=_QUERY_ANNOTATIONS,
            input_schema=[
                ParameterSchema(
                    name="artifact_id",
                    type="string",
                    description="Artifact ID of the stored ContentDocument.",
                ),
                ParameterSchema(
                    name="node_ref",
                    type="string",
                    description=(
                        "JSON pointer ref of the target block "
                        "(e.g. '#/body/12'). Typically obtained from a "
                        "search result's block_ref field."
                    ),
                ),
                ParameterSchema(
                    name="before_blocks",
                    type="integer",
                    description="Number of body blocks before the target. Default 2.",
                    required=False,
                    default=2,
                    constraints={"minimum": 0, "maximum": 50},
                ),
                ParameterSchema(
                    name="after_blocks",
                    type="integer",
                    description="Number of body blocks after the target. Default 2.",
                    required=False,
                    default=2,
                    constraints={"minimum": 0, "maximum": 50},
                ),
                ParameterSchema(
                    name="expand_to_section",
                    type="boolean",
                    description=(
                        "When true, if the window would cross a heading "
                        "boundary, expand to the full enclosing section "
                        "instead. Default false (strict block window)."
                    ),
                    required=False,
                    default=False,
                ),
            ],
        )

    async def execute(
        self, inputs: dict[str, Any], context: KaosContext | None = None
    ) -> ToolResult:
        if context is None or context.runtime is None:
            return ToolResult.create_error(_NO_CONTEXT_ERROR)

        artifact_id = inputs.get("artifact_id", "")
        node_ref = inputs.get("node_ref", "")
        before_blocks = int(inputs.get("before_blocks", 2))
        after_blocks = int(inputs.get("after_blocks", 2))
        expand_to_section = bool(inputs.get("expand_to_section", False))

        if not artifact_id:
            return ToolResult.create_error(
                "The 'artifact_id' parameter is required. "
                "Use kaos-core-artifacts-list to find document artifact IDs."
            )
        if not node_ref:
            return ToolResult.create_error(
                "The 'node_ref' parameter is required (e.g. '#/body/12'). "
                "Use kaos-content-search-document and pass the resulting "
                "block_ref, or read it from a previous citation's metadata."
            )

        from kaos_content.artifacts import load_document
        from kaos_content.serializers.text import serialize_text
        from kaos_content.views.document_view import DocumentView

        try:
            doc = await load_document(artifact_id, context.runtime)
        except Exception as exc:
            return ToolResult.create_error(
                f"Failed to load document artifact '{artifact_id}': {exc}. "
                "Verify the artifact_id with kaos-core-artifacts-list."
            )

        # Locate the target block index in the body. We restrict to the
        # top-level body for the window walk — sub-block refs (children
        # inside lists / tables) get mapped to their containing top-level
        # block first.
        body_idx = _resolve_body_index(node_ref)
        if body_idx is None or body_idx >= len(doc.body):
            return ToolResult.create_error(
                f"Could not locate '{node_ref}' as a top-level body block. "
                "Window walks operate on body blocks; sub-block refs are "
                "supported by mapping to their containing body block first. "
                f"Document has {len(doc.body)} body blocks."
            )

        # Compute the [start, end] block window, clamped to body bounds.
        start = max(0, body_idx - before_blocks)
        end = min(len(doc.body) - 1, body_idx + after_blocks)

        # Optional expansion to the enclosing section when the window
        # crosses a heading boundary. Also used to compute the
        # structural breadcrumb for the target node — we always need a
        # view to derive ``path`` from the target's containing section,
        # so the view construction below is shared between the expand
        # logic and the path computation.
        view = DocumentView(doc)
        expanded_section: bool = False
        section_heading_ref: str | None = None
        if expand_to_section:
            target_section = None
            for sv in view.flat_sections:
                if sv.heading_ref is None:
                    continue
                # Heading ref looks like "#/body/N" — parse N to get the
                # section's body index range.
                sec_start = _resolve_body_index(sv.heading_ref)
                if sec_start is None:
                    continue
                sec_end = sec_start + len(sv.blocks) - 1
                if sec_start <= body_idx <= sec_end:
                    target_section = (sec_start, sec_end, sv.heading_ref)
                    break
            if target_section is not None:
                sec_start, sec_end, sec_ref = target_section
                # Only expand if the strict window crosses out of this section.
                if start < sec_start or end > sec_end:
                    start = max(start, sec_start)
                    end = min(end, sec_end)
                    expanded_section = True
                    section_heading_ref = sec_ref

        window_blocks = doc.body[start : end + 1]
        window_refs = [f"#/body/{i}" for i in range(start, end + 1)]

        # Render the window as plain text (block-separated) for the
        # agent to read directly. Each block gets a "▸ #/body/N" header
        # so the agent can re-cite individual blocks if needed.
        from kaos_content.model.document import ContentDocument
        from kaos_content.model.metadata import DocumentMetadata

        window_doc = ContentDocument(
            metadata=DocumentMetadata(title=doc.metadata.title or "window"),
            body=tuple(window_blocks),
        )
        window_text = serialize_text(window_doc)

        # Pretty-print: prefix each block with its ref.
        lines = []
        for ref, block in zip(window_refs, window_blocks, strict=True):
            marker = "▶" if ref == node_ref else "▸"
            block_text = serialize_text(ContentDocument(body=(block,))).strip()
            lines.append(f"{marker} {ref}\n{block_text}")
        formatted = "\n\n".join(lines)

        # Structural breadcrumb for the target node — the chain of
        # enclosing heading texts. Empty list means the target lives in
        # the preamble (no heading); agents MUST NOT invent a section
        # identifier for an empty path. See ``DocumentView.block_path``.
        target_path = list(view.block_path(node_ref))

        structured = {
            "artifact_id": artifact_id,
            "target_node_ref": node_ref,
            "window_start_ref": window_refs[0] if window_refs else None,
            "window_end_ref": window_refs[-1] if window_refs else None,
            "window_block_refs": window_refs,
            "block_count": len(window_blocks),
            "before_blocks": before_blocks,
            "after_blocks": after_blocks,
            "expanded_to_section": expanded_section,
            "enclosing_section_ref": section_heading_ref,
            "path": target_path,
            "text": window_text,
        }
        return ToolResult.create_success(output=structured, summary=formatted)


def _resolve_body_index(node_ref: str) -> int | None:
    """Map any node_ref (top-level or descendant) to its top-level body
    index. Returns None if the ref doesn't start with ``#/body/`` or if
    the index segment isn't an integer.

    Examples:
        ``#/body/5``           → 5
        ``#/body/5/children/2`` → 5
        ``#/footnotes/x/0``   → None (not a body ref)
    """
    prefix = "#/body/"
    if not node_ref.startswith(prefix):
        return None
    tail = node_ref[len(prefix) :]
    head = tail.split("/", 1)[0]
    try:
        return int(head)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class DedupSemanticTool(KaosTool):
    """Semantic near-duplicate clustering via embedding cosine distance.

    KNT-602 Option A (kaos-content 0.1.0a3): moved here from
    kaos-nlp-transformers/tools.py. The previous tool name
    ``kaos-nlp-transformers-dedup-semantic`` is removed in
    kaos-nlp-transformers 0.2.0a3; callers should switch to
    ``kaos-content-dedup-semantic``.
    """

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="kaos-content-dedup-semantic",
            display_name="Semantic Deduplication",
            description=(
                "Cluster documents by embedding cosine distance using "
                "scipy hierarchical agglomerative clustering. Catches "
                "paraphrases and template variants that lexical "
                "dedup misses. Returns clusters with member doc_ids and "
                "mean intra-cluster similarity. Threshold guidance: "
                "0.02 = near-exact; 0.10 = same template; 0.20 = same "
                "topic (broader). Requires the [transformers] and "
                "[clustering] extras at execute time; missing extras "
                "surface as a friendly install hint. Hard cap: "
                f"{_MAX_DEDUP_DOCS} docs per call."
            ),
            category=ToolCategory.TEXT,
            capability=ToolCapability.ANALYZE,
            module_name=_MODULE,
            version=_VERSION,
            annotations=_QUERY_ANNOTATIONS,
            input_schema=[
                ParameterSchema(
                    name="documents",
                    type="array",
                    description=(
                        "List of objects with `doc_id` (string) and "
                        "`text` (string). Empty / whitespace-only texts "
                        "are skipped."
                    ),
                    constraints={
                        "minItems": 2,
                        "items": {
                            "type": "object",
                            "properties": {
                                "doc_id": {"type": "string"},
                                "text": {"type": "string"},
                            },
                            "required": ["doc_id", "text"],
                        },
                    },
                ),
                ParameterSchema(
                    name="distance_threshold",
                    type="number",
                    description=(
                        "Cosine-distance threshold for the cluster cut "
                        "(default 0.10). Must lie in [0.0, 2.0]."
                    ),
                    required=False,
                    default=0.10,
                    constraints={"minimum": 0.0, "maximum": 2.0},
                ),
                ParameterSchema(
                    name="max_chars",
                    type="integer",
                    description=(
                        "Truncate documents above this char count before embedding (default 8000)."
                    ),
                    required=False,
                    default=8000,
                    constraints={"minimum": 1},
                ),
                ParameterSchema(
                    name="model_id",
                    type="string",
                    description="Override the embedding model.",
                    required=False,
                    default=None,
                ),
            ],
        )

    async def execute(
        self, inputs: dict[str, Any], context: KaosContext | None = None
    ) -> ToolResult:
        import asyncio
        from typing import cast

        from kaos_content.dedup.levels.semantic import SemanticDedupLevel
        from kaos_content.dedup.types import DedupDocument

        documents = inputs.get("documents")
        if not isinstance(documents, list) or len(documents) < 2:
            return ToolResult.create_error(
                "Parameter 'documents' is required and must contain at "
                "least 2 entries. "
                'Fix: pass `{"documents": [{"doc_id": "a", '
                '"text": "…"}, …]}`. '
                "Alternative: with a single doc, dedup is a no-op — "
                "skip the call."
            )
        if len(documents) > _MAX_DEDUP_DOCS:
            return ToolResult.create_error(
                f"Too many documents: {len(documents)} (cap {_MAX_DEDUP_DOCS}). "
                "Fix: split the call into batches. "
                "Alternative: pre-filter with kaos-content's lexical "
                "dedup levels (binary hash, MinHash) before semantic dedup."
            )

        dedup_docs: list[DedupDocument] = []
        for idx, raw_item in enumerate(documents):
            if not isinstance(raw_item, dict):
                return ToolResult.create_error(
                    f"documents[{idx}] is not an object. "
                    'Fix: use `{"doc_id": "…", "text": "…"}`. '
                    "Alternative: wrap raw strings into the object form "
                    "client-side."
                )
            item = cast(dict[str, Any], raw_item)
            doc_id = item.get("doc_id")
            text = item.get("text")
            if not isinstance(doc_id, str) or not isinstance(text, str):
                return ToolResult.create_error(
                    f"documents[{idx}] must have string `doc_id` and `text`. "
                    "Fix: ensure both are strings (cast ints to strings "
                    "if needed). "
                    "Alternative: drop the malformed item client-side."
                )
            dedup_docs.append(DedupDocument(doc_id=doc_id, text=text))

        # Settings come from runtime context when available; otherwise
        # the level falls back to the package defaults.
        settings = None
        try:
            from kaos_nlp_transformers.settings import (  # type: ignore[import-not-found]
                KaosNLPTransformersSettings,
            )

            settings = KaosNLPTransformersSettings.from_context(context)
        except ImportError:
            pass  # Surfaced again with hint via SemanticDedupLevel.find_clusters

        model_id = inputs.get("model_id") or None
        distance_threshold = float(inputs.get("distance_threshold") or 0.10)
        max_chars = int(inputs.get("max_chars") or 8000)

        level = SemanticDedupLevel(
            model_id=model_id,
            distance_threshold=distance_threshold,
            max_chars=max_chars,
            settings=settings,
        )

        try:
            clusters = await asyncio.to_thread(level.find_clusters, dedup_docs)
        except ImportError as exc:
            # SemanticDedupLevel raises ImportError with the install
            # hint when scipy or kaos-nlp-transformers is missing — pass
            # it through verbatim.
            return ToolResult.create_error(str(exc))
        except Exception as exc:
            return ToolResult.create_error(
                f"Semantic dedup failed: {exc}. "
                "Fix: confirm the model is loadable and the [transformers] "
                "and [clustering] extras are installed. "
                "Alternative: lower the document count or simplify the "
                "input texts to isolate the failure."
            )

        payload = {
            "model_id": model_id,
            "distance_threshold": distance_threshold,
            "num_documents": len(dedup_docs),
            "clusters": [
                {
                    "cluster_id": c.cluster_id,
                    "canonical_doc_id": c.canonical_doc_id,
                    "member_doc_ids": list(c.member_doc_ids),
                    "size": len(c.member_doc_ids),
                    "level": c.level,
                    "similarity": c.similarity,
                }
                for c in clusters
            ],
        }
        n_clustered = sum(len(c.member_doc_ids) for c in clusters)
        return ToolResult.create_success(
            payload,
            summary=(f"{len(clusters)} cluster(s) covering {n_clustered}/{len(dedup_docs)} doc(s)"),
        )


# ---------------------------------------------------------------------------
# StatsTool
# ---------------------------------------------------------------------------


class StatsTool(KaosTool):
    """Per-document statistics — character / word / paragraph / table counts.

    Closes the aggregation-question gap (longest / shortest / largest /
    most-X) at the kaos-content boundary: produces a stable numerical
    summary of a single ContentDocument so callers can sort, filter, or
    compare across documents without re-tokenising the source.
    """

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="kaos-content-stats",
            display_name="Document Stats",
            description=(
                "Compute statistical summary of a stored ContentDocument: "
                "char_count, word_count, paragraph_count, heading_count, "
                "table_count, image_count, code_block_count, "
                "footnote_count, annotation_count, page_count (when "
                "provenance carries pages). Use for aggregation questions "
                "(longest doc, most tables, etc.) rather than retrieving "
                "passages and counting manually."
            ),
            category=ToolCategory.DOCUMENT,
            capability=ToolCapability.QUERY,
            module_name=_MODULE,
            version=_VERSION,
            annotations=_QUERY_ANNOTATIONS,
            input_schema=[
                ParameterSchema(
                    name="artifact_id",
                    type="string",
                    description="Artifact ID of the stored ContentDocument.",
                ),
            ],
        )

    async def execute(
        self, inputs: dict[str, Any], context: KaosContext | None = None
    ) -> ToolResult:
        if context is None or context.runtime is None:
            return ToolResult.create_error(_NO_CONTEXT_ERROR)

        artifact_id = inputs.get("artifact_id")
        if not artifact_id:
            return ToolResult.create_error(
                "Missing 'artifact_id'. Provide the ID returned by "
                "kaos-content-parse-markdown or any reader tool."
            )

        from kaos_content.artifacts import load_document
        from kaos_content.model.blocks import Paragraph
        from kaos_content.serializers.text import serialize_text
        from kaos_content.traversal import NodeIndex

        try:
            doc = await load_document(artifact_id, context.runtime)
        except Exception as exc:
            return ToolResult.create_error(
                f"Failed to load artifact {artifact_id!r}: {exc}. "
                "Verify the artifact exists in the runtime's VFS."
            )

        text = serialize_text(doc)
        char_count = len(text)
        word_count = len(text.split())

        index = NodeIndex(doc)
        paragraph_count = len(index.by_type(Paragraph))
        heading_count = len(index.headings)
        table_count = len(index.tables)
        image_count = len(index.images)
        code_block_count = len(index.code_blocks)
        footnote_count = sum(len(blocks) for blocks in doc.footnotes.values())
        annotation_count = len(doc.annotations)

        # Page count via provenance — only meaningful when the reader
        # (e.g. kaos-pdf) emitted provenance.page on its blocks.
        pages: set[int] = set()
        for block in doc.body:
            if block.provenance is not None and block.provenance.page is not None:
                pages.add(block.provenance.page)
        page_count = len(pages) if pages else None

        output = {
            "artifact_id": artifact_id,
            "char_count": char_count,
            "word_count": word_count,
            "paragraph_count": paragraph_count,
            "heading_count": heading_count,
            "table_count": table_count,
            "image_count": image_count,
            "code_block_count": code_block_count,
            "footnote_count": footnote_count,
            "annotation_count": annotation_count,
            "page_count": page_count,
        }
        summary = (
            f"{char_count:,} chars, {word_count:,} words, "
            f"{paragraph_count} paragraphs, {heading_count} headings, "
            f"{table_count} tables" + (f", {page_count} pages" if page_count is not None else "")
        )
        return ToolResult.create_success(output=output, summary=summary)


# ---------------------------------------------------------------------------
# Typed-entity sentence filters (K3)
#
# Five MCP tools — one per supported entity type — that surface the
# K2 kaos_content.views.entity_filters API to agents. Each tool takes
# a stored ContentDocument artifact id and returns the sentences (or
# optionally paragraphs) that contain at least one match of the
# target entity type, plus the typed match value.
#
# Implemented via a single base class parametrised by entity_type so
# adding a new type later (e.g. "parties" when the NER extractor
# lands) is one line, not 100.
# ---------------------------------------------------------------------------


class _EntityFilterToolBase(KaosTool):
    """Shared logic for the five entity-filter tools.

    Each concrete tool sets ``_ENTITY_TYPE`` and ``_DISPLAY_NAME``;
    everything else (metadata shape, parameter schema, execute body)
    lives here so the five tools stay in lockstep.
    """

    _ENTITY_TYPE: str = ""  # subclasses override
    _DISPLAY_NAME: str = ""  # human-friendly
    _DESCRIPTION_TAIL: str = ""  # one-line type-specific hint

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name=f"kaos-content-sentences-with-{self._ENTITY_TYPE}",
            display_name=f"Sentences with {self._DISPLAY_NAME}",
            description=(
                f"Return every sentence in a stored ContentDocument that "
                f"contains at least one {self._DISPLAY_NAME} match. "
                f"{self._DESCRIPTION_TAIL} "
                "Deterministic — pure regex + dictionary lookups, zero LLM "
                "cost. Each match carries the typed extracted value so "
                "agents can sort, threshold, or compare without re-parsing. "
                "Results carry a salience score in [0, 1] combining match "
                "density + document position + sentence length; the default "
                "select_by='salience' picks the load-bearing hits (effective "
                "date, signature lines) over early boilerplate. "
                "Pair with kaos-content-stats / kaos-content-summarize for "
                "corpus-scale triage."
            ),
            category=ToolCategory.DOCUMENT,
            capability=ToolCapability.QUERY,
            module_name=_MODULE,
            version=_VERSION,
            annotations=_QUERY_ANNOTATIONS,
            input_schema=[
                ParameterSchema(
                    name="artifact_id",
                    type="string",
                    description="Artifact ID of the stored ContentDocument.",
                ),
                ParameterSchema(
                    name="granularity",
                    type="string",
                    description=(
                        "One of 'sentence' (default) or 'paragraph'. "
                        "Sentence-level returns finer hits; paragraph-level "
                        "is appropriate when the answer needs more context."
                    ),
                    required=False,
                    default="sentence",
                ),
                ParameterSchema(
                    name="max_results",
                    type="integer",
                    description=(
                        "Cap on number of hits returned. Default 50; use a "
                        "lower number for triage, higher for diligence."
                    ),
                    required=False,
                    default=50,
                ),
                ParameterSchema(
                    name="select_by",
                    type="string",
                    description=(
                        "Ranking used when more hits exist than 'max_results'. "
                        "'salience' (default, PA9) picks the load-bearing hits "
                        "via the salience score documented on SentenceEntityHit "
                        "— match-density + heading-adjacent / front-and-back "
                        "position + length bell. 'position' preserves document "
                        "order (the pre-PA9 behaviour) — use when you need the "
                        "first N hits in reading order, not the most important."
                    ),
                    required=False,
                    default="salience",
                ),
            ],
        )

    async def execute(
        self, inputs: dict[str, Any], context: KaosContext | None = None
    ) -> ToolResult:
        if context is None or context.runtime is None:
            return ToolResult.create_error(_NO_CONTEXT_ERROR)

        artifact_id = inputs.get("artifact_id")
        if not artifact_id:
            return ToolResult.create_error(
                "Missing 'artifact_id'. Provide the ID returned by "
                "kaos-content-parse-markdown or any reader tool."
            )

        granularity = inputs.get("granularity", "sentence")
        if granularity not in ("sentence", "paragraph"):
            return ToolResult.create_error(
                f"Invalid granularity {granularity!r}. Must be 'sentence' or 'paragraph'."
            )
        # Use a None-aware default so an explicit 0 from the caller
        # surfaces as a validation error rather than silently being
        # replaced by 50 via ``or``.
        max_results_raw = inputs.get("max_results")
        max_results = 50 if max_results_raw is None else int(max_results_raw)
        if max_results < 1:
            return ToolResult.create_error("'max_results' must be >= 1.")

        select_by = inputs.get("select_by", "salience")
        if select_by not in ("salience", "position"):
            return ToolResult.create_error(
                f"Invalid select_by {select_by!r}. Must be 'salience' or 'position'. "
                "Fix: use 'salience' (default) to surface load-bearing hits, or "
                "'position' for document-order results (the pre-PA9 behaviour)."
            )

        from kaos_content.artifacts import load_document
        from kaos_content.views import DocumentView
        from kaos_content.views.entity_filters import (
            iter_paragraphs_with_entity,
            iter_sentences_with_entity,
        )

        try:
            doc = await load_document(artifact_id, context.runtime)
        except Exception as exc:
            return ToolResult.create_error(
                f"Failed to load artifact {artifact_id!r}: {exc}. "
                "Verify the artifact exists in the runtime's VFS."
            )

        # Sentence-level needs the segmenter; paragraph-level doesn't.
        if granularity == "sentence":
            from kaos_nlp_core._defaults import get_default_punkt_tokenizer

            view = DocumentView(doc, sentence_segmenter=get_default_punkt_tokenizer())
            hits = list(iter_sentences_with_entity(view, self._ENTITY_TYPE))
            # Sort: salience asks for highest first; position keeps doc
            # order. Both use enumerate-index as the stable tiebreaker
            # so equal-salience hits surface in reading order.
            ranked = _rank_hits(hits, select_by)
            matches_payload = [
                {
                    "block_ref": h.sentence.paragraph_ref,
                    "page": h.sentence.page,
                    "section_title": _section_title_for(view, h.sentence.section_ref),
                    "text": h.sentence.text,
                    "char_start": h.sentence.start,
                    "char_end": h.sentence.end,
                    "salience": h.salience,
                    "entities": [_match_payload(m) for m in h.matches],
                }
                for h in ranked[:max_results]
            ]
        else:
            view = DocumentView(doc)
            hits = list(iter_paragraphs_with_entity(view, self._ENTITY_TYPE))
            ranked = _rank_hits(hits, select_by)
            matches_payload = [
                {
                    "block_ref": h.paragraph.block_ref,
                    "page": h.paragraph.page,
                    "section_title": _section_title_for(view, h.paragraph.section_ref),
                    "text": h.paragraph.text,
                    "salience": h.salience,
                    "entities": [_match_payload(m) for m in h.matches],
                }
                for h in ranked[:max_results]
            ]

        total = len(hits)
        has_more = total > max_results
        output = {
            "artifact_id": artifact_id,
            "entity_type": self._ENTITY_TYPE,
            "granularity": granularity,
            "select_by": select_by,
            "matches": matches_payload,
            "total_matches": total,
            "has_more": has_more,
        }
        summary = (
            f"Found {total} {granularity}(s) with {self._DISPLAY_NAME} "
            f"in artifact {artifact_id} (ranked by {select_by})"
            + (f" (showing first {max_results})" if has_more else "")
        )
        return ToolResult.create_success(output=output, summary=summary)


def _section_title_for(view: Any, section_ref: str | None) -> str | None:
    """Resolve a section_ref to its heading text, or None."""
    if section_ref is None:
        return None
    sec = view.section_by_ref(section_ref)
    return sec.heading_text if sec is not None else None


def _rank_hits(hits: list[Any], select_by: str) -> list[Any]:
    """Order entity-filter hits for the MCP tool's top-K window.

    Args:
        hits: list of :class:`SentenceEntityHit` or
            :class:`ParagraphEntityHit` in document order.
        select_by: ``"salience"`` or ``"position"``.

    Returns:
        A new list ordered for top-K selection. ``"position"`` returns
        the input as-is (bit-for-bit pre-PA9 behaviour). ``"salience"``
        sorts by salience descending, with document index ascending as
        the stable tiebreaker. The original list is never mutated.
    """
    if select_by == "position":
        return list(hits)
    # Enumerate to capture the doc-order index as the tiebreaker.
    indexed = list(enumerate(hits))
    indexed.sort(key=lambda pair: (-pair[1].salience, pair[0]))
    return [h for _, h in indexed]


def _match_payload(match: Any) -> dict[str, Any]:
    """Convert an EntityMatch to a JSON-friendly dict."""
    # Avoid leaking domain objects (Decimal, datetime, MoneyMatch) —
    # serialise the typed value via repr() so the wire payload stays
    # JSON-clean. Callers needing the typed value should consume the
    # Python API, not the MCP surface.
    value_repr: Any
    val = match.value
    if val is None:
        value_repr = None
    elif hasattr(val, "amount") and hasattr(val, "currency"):
        # MoneyMatch shape
        value_repr = {"amount": str(val.amount), "currency": val.currency}
    elif hasattr(val, "quantity") and hasattr(val, "unit"):
        # DurationMatch shape
        value_repr = {"quantity": str(val.quantity), "unit": val.unit}
    elif hasattr(val, "isoformat"):
        # datetime / date
        value_repr = val.isoformat()
    else:
        # Decimal, int, float, str — pass through as string
        value_repr = str(val)
    return {
        "text": match.text,
        "value": value_repr,
        "char_start": match.start,
        "char_end": match.end,
    }


class SentencesWithDatesTool(_EntityFilterToolBase):
    """Find sentences containing at least one date."""

    _ENTITY_TYPE = "dates"
    _DISPLAY_NAME = "dates"
    _DESCRIPTION_TAIL = (
        "Matches both numeric ('Jan 1, 2026', '1/1/2026') and "
        "natural-language ('the first of January, 2026') date "
        "expressions. Returned typed value is the parsed datetime."
    )


class SentencesWithMoneyTool(_EntityFilterToolBase):
    """Find sentences containing at least one money amount."""

    _ENTITY_TYPE = "money"
    _DISPLAY_NAME = "money amounts"
    _DESCRIPTION_TAIL = (
        "Matches currency expressions ('$100,000', 'USD 50k', '€500'). "
        "Returned typed value carries amount (Decimal) and currency code."
    )


class SentencesWithPercentsTool(_EntityFilterToolBase):
    """Find sentences containing at least one percentage."""

    _ENTITY_TYPE = "percents"
    _DISPLAY_NAME = "percentages"
    _DESCRIPTION_TAIL = (
        "Matches numeric percents ('15%', '7.5 percent', 'fifteen percent'). "
        "Returned typed value is a Decimal between 0 and 1."
    )


class SentencesWithDurationsTool(_EntityFilterToolBase):
    """Find sentences containing at least one duration."""

    _ENTITY_TYPE = "durations"
    _DISPLAY_NAME = "durations"
    _DESCRIPTION_TAIL = (
        "Matches durations like '24 months', '7 days', 'two weeks'. "
        "Returned typed value carries quantity (Decimal), unit (str), "
        "and total_seconds (Decimal)."
    )


class SentencesWithNumbersTool(_EntityFilterToolBase):
    """Find sentences containing at least one numeric expression."""

    _ENTITY_TYPE = "numbers"
    _DISPLAY_NAME = "numbers"
    _DESCRIPTION_TAIL = (
        "Matches integer and decimal numbers ('1,000', '3.14', 'one "
        "thousand'). Broader than the typed money/percent extractors. "
        "Returned typed value is a Decimal."
    )


# ---------------------------------------------------------------------------
# Corpus summary + narrow (K4)
#
# Two tools that operate on a *corpus* (a list of stored ContentDocument
# artifacts) rather than a single document. The summary tool builds a
# cheap deterministic DocumentSummary per artifact; the narrow tool
# BM25-ranks summaries against a query to surface the most relevant
# subset. Together they implement the "uploaded 10K docs, work on the
# relevant 50" workflow from docs/design/findings-entities-summary.md
# proposal #3.
# ---------------------------------------------------------------------------


class CorpusSummarizeTool(KaosTool):
    """Build :class:`DocumentSummary` for each artifact in a corpus.

    Deterministic, zero-LLM. ~100 ms per typical NDA. Returns per-artifact
    summary previews; optionally re-stores each document with the
    summary attached when ``persist=True`` (the caller takes
    responsibility for the resulting fresh artifact IDs).
    """

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="kaos-content-corpus-summarize",
            display_name="Summarize Corpus",
            description=(
                "Build a deterministic, no-LLM DocumentSummary for each "
                "stored ContentDocument artifact in a corpus. Each summary "
                "carries head_tokens (first ~500 tokens verbatim), "
                "top_ngrams (50 most-frequent 1-3 grams after stopword "
                "removal), bottom_ngrams (50 rare-but-recurring n-grams — "
                "the distinctive fingerprint), entity_counts (per-type "
                "sentence counts), and basic statistics. Use to enable "
                "corpus-scale triage: BM25 over summaries is "
                "50-100x faster than BM25 over full bodies and gives "
                "comparable narrowing quality. Pair with "
                "kaos-content-corpus-narrow to actually run the search."
            ),
            category=ToolCategory.DOCUMENT,
            capability=ToolCapability.ANALYZE,
            module_name=_MODULE,
            version=_VERSION,
            annotations=_QUERY_ANNOTATIONS,
            input_schema=[
                ParameterSchema(
                    name="artifact_ids",
                    type="array",
                    description=(
                        "List of artifact IDs (stored ContentDocument) to "
                        "summarize. Build summaries in parallel order, "
                        "skipping artifacts whose summary is already "
                        "populated unless force_rebuild=True."
                    ),
                    constraints={"items": {"type": "string"}},
                ),
                ParameterSchema(
                    name="force_rebuild",
                    type="boolean",
                    description=(
                        "Rebuild summaries even when the artifact already "
                        "carries one. Default false."
                    ),
                    required=False,
                    default=False,
                ),
                ParameterSchema(
                    name="head_token_target",
                    type="integer",
                    description=(
                        "Approximate token count for head_tokens. Default "
                        "500 (matches build_document_summary)."
                    ),
                    required=False,
                    default=500,
                ),
                ParameterSchema(
                    name="with_entities",
                    type="boolean",
                    description=(
                        "Populate entity_counts via the alpha extractors. "
                        "Default true. Set false to skip extraction for "
                        "pure lexical signal at lower cost."
                    ),
                    required=False,
                    default=True,
                ),
            ],
        )

    async def execute(
        self, inputs: dict[str, Any], context: KaosContext | None = None
    ) -> ToolResult:
        if context is None or context.runtime is None:
            return ToolResult.create_error(_NO_CONTEXT_ERROR)

        raw_ids = inputs.get("artifact_ids")
        if not raw_ids or not isinstance(raw_ids, list):
            return ToolResult.create_error(
                "Missing 'artifact_ids' (list of strings). Provide the IDs "
                "of stored ContentDocument artifacts to summarize."
            )

        force_rebuild = bool(inputs.get("force_rebuild", False))
        head_token_target = int(inputs.get("head_token_target") or 500)
        with_entities = bool(inputs.get("with_entities", True))

        from kaos_content.artifacts import load_document
        from kaos_content.summarize import build_document_summary

        results: list[dict[str, Any]] = []
        failed: list[dict[str, Any]] = []
        built = 0
        skipped = 0
        for aid in raw_ids:
            aid_str = str(aid)
            try:
                doc = await load_document(aid_str, context.runtime)
            except Exception as exc:
                failed.append({"artifact_id": aid_str, "reason": repr(exc)})
                continue

            if doc.summary is not None and not force_rebuild:
                summary = doc.summary
                skipped += 1
            else:
                try:
                    summary = build_document_summary(
                        doc,
                        head_token_target=head_token_target,
                        with_entities=with_entities,
                    )
                except Exception as exc:
                    failed.append({"artifact_id": aid_str, "reason": repr(exc)})
                    continue
                built += 1

            results.append(
                {
                    "artifact_id": aid_str,
                    "head_snippet": summary.head_tokens[:200],
                    "top_ngrams": [
                        {"ngram": ng.ngram, "count": ng.count} for ng in summary.top_ngrams[:10]
                    ],
                    "bottom_ngrams": [
                        {"ngram": ng.ngram, "count": ng.count} for ng in summary.bottom_ngrams[:10]
                    ],
                    "char_length": summary.char_length,
                    "sentence_count": summary.sentence_count,
                    "paragraph_count": summary.paragraph_count,
                    "entity_counts": dict(summary.entity_counts),
                }
            )

        output = {
            "summaries": results,
            "built": built,
            "skipped": skipped,
            "failed": failed,
            "total_requested": len(raw_ids),
        }
        summary_text = (
            f"Built {built} summary/summaries, skipped {skipped} "
            f"(already populated), {len(failed)} failed"
        )
        return ToolResult.create_success(output=output, summary=summary_text)


class CorpusNarrowTool(KaosTool):
    """Rank corpus artifacts by relevance to a query using their summaries.

    BM25 over the concatenated (head_tokens + top_ngram_text +
    bottom_ngram_text) of each artifact's summary. Builds the summary
    on the fly when an artifact doesn't have one attached (no caching
    side-effects).

    Returns the top-K artifact IDs with scores and a short
    distinguishing-snippet. The caller uses this output as a triage
    hint — "process these N of K artifacts," not an answer.
    """

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="kaos-content-corpus-narrow",
            display_name="Narrow Corpus by Query",
            description=(
                "Given a query and a corpus of stored ContentDocument "
                "artifacts, return the top-K artifacts whose "
                "DocumentSummary best matches the query via BM25. Use to "
                "triage a large corpus down to a working subset before "
                "running expensive per-document operations (extraction, "
                "LLM Q&A). Builds summaries on demand for artifacts that "
                "don't have one attached. Pair with "
                "kaos-content-corpus-summarize to pre-build summaries "
                "for very large corpora (>100 artifacts)."
            ),
            category=ToolCategory.DOCUMENT,
            capability=ToolCapability.QUERY,
            module_name=_MODULE,
            version=_VERSION,
            annotations=_QUERY_ANNOTATIONS,
            input_schema=[
                ParameterSchema(
                    name="query",
                    type="string",
                    description=(
                        "The free-form query to rank artifacts against. "
                        "Typically the user's goal or current question."
                    ),
                ),
                ParameterSchema(
                    name="artifact_ids",
                    type="array",
                    description=(
                        "List of artifact IDs (stored ContentDocument) "
                        "forming the corpus to narrow."
                    ),
                    constraints={"items": {"type": "string"}},
                ),
                ParameterSchema(
                    name="top_k",
                    type="integer",
                    description=(
                        "Maximum number of artifacts to return. Default "
                        "10; use higher when the corpus is large and the "
                        "agent wants more breadth."
                    ),
                    required=False,
                    default=10,
                ),
            ],
        )

    async def execute(
        self, inputs: dict[str, Any], context: KaosContext | None = None
    ) -> ToolResult:
        if context is None or context.runtime is None:
            return ToolResult.create_error(_NO_CONTEXT_ERROR)

        query = inputs.get("query")
        if not query or not str(query).strip():
            return ToolResult.create_error(
                "Missing 'query'. Provide a non-empty string to rank artifacts against."
            )

        raw_ids = inputs.get("artifact_ids")
        if not raw_ids or not isinstance(raw_ids, list):
            return ToolResult.create_error("Missing 'artifact_ids' (list of strings).")

        top_k_raw = inputs.get("top_k")
        top_k = 10 if top_k_raw is None else int(top_k_raw)
        if top_k < 1:
            return ToolResult.create_error("'top_k' must be >= 1.")

        from kaos_content.artifacts import load_document
        from kaos_content.summarize import build_document_summary

        # Build the per-artifact summary text for BM25 indexing.
        # Searcher.from_documents wants integer doc_ids; keep a parallel
        # list mapping the integer slot back to the original artifact_id.
        records: list[dict[str, Any]] = []
        artifact_ids_in_order: list[str] = []
        artifact_summaries: dict[str, Any] = {}
        for aid in raw_ids:
            aid_str = str(aid)
            try:
                doc = await load_document(aid_str, context.runtime)
            except Exception:
                # Skip unloadable artifacts; the agent already knows
                # about them via the input list.
                continue
            summary = doc.summary
            if summary is None:
                try:
                    summary = build_document_summary(doc)
                except Exception:
                    continue
            artifact_summaries[aid_str] = summary
            search_text = _summary_search_text(summary)
            if not search_text.strip():
                continue
            records.append({"id": len(artifact_ids_in_order), "text": search_text})
            artifact_ids_in_order.append(aid_str)

        if not records:
            return ToolResult.create_success(
                output={"selected": [], "total_searched": 0},
                summary="No artifacts had usable summaries to rank.",
            )

        # Run BM25 over the per-artifact summary text.
        from kaos_nlp_core.search import Searcher

        searcher = Searcher.from_documents(records)
        hits = searcher.search(str(query), top_k=top_k)

        # Build selected payload, preserving the per-summary signal so
        # the caller can decide whether a hit is plausible without
        # re-fetching the artifact.
        selected: list[dict[str, Any]] = []
        for hit in hits:
            aid_str = artifact_ids_in_order[int(hit.doc_id)]
            summary = artifact_summaries[aid_str]
            selected.append(
                {
                    "artifact_id": aid_str,
                    "score": float(getattr(hit, "score", 0.0)),
                    "head_snippet": summary.head_tokens[:200],
                    "distinguishing_ngrams": [ng.ngram for ng in summary.bottom_ngrams[:5]],
                    "entity_counts": dict(summary.entity_counts),
                }
            )

        output = {
            "query": str(query),
            "selected": selected,
            "total_searched": len(records),
            "total_requested": len(raw_ids),
        }
        summary_text = (
            f"Narrowed {len(records)} artifacts to top {len(selected)} "
            f"for query {str(query)[:60]!r}"
        )
        return ToolResult.create_success(output=output, summary=summary_text)


def _summary_search_text(summary: Any) -> str:
    """Concatenate the searchable parts of a DocumentSummary.

    The triage-friendly BM25 corpus is the union of:
    - head_tokens (captures opening structure: parties, dates, recitals)
    - top_ngrams (thematic vocabulary)
    - bottom_ngrams (distinctive fingerprint)

    All joined with whitespace. Order doesn't matter for BM25; mixing
    gives the searcher access to both the thematic and distinctive
    signals.
    """
    parts: list[str] = []
    if summary.head_tokens:
        parts.append(summary.head_tokens)
    parts.extend(ng.ngram for ng in summary.top_ngrams)
    parts.extend(ng.ngram for ng in summary.bottom_ngrams)
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_content_tools(runtime: Any) -> int:
    """Register all kaos-content MCP tools with the runtime. Returns count."""
    tools: list[KaosTool] = [
        ParseMarkdownTool(),
        SerializeTool(),
        ChunkDocumentTool(),
        SearchDocumentTool(),
        SearchTableTool(),
        ExtractSectionTool(),
        ExtractPageTool(),
        ContextWindowTool(),
        DedupSemanticTool(),
        StatsTool(),
        # K3 (0.1.0a6) — typed-entity sentence/paragraph filters.
        SentencesWithDatesTool(),
        SentencesWithMoneyTool(),
        SentencesWithPercentsTool(),
        SentencesWithDurationsTool(),
        SentencesWithNumbersTool(),
        # K4 (0.1.0a6) — corpus-level summary + narrow.
        CorpusSummarizeTool(),
        CorpusNarrowTool(),
    ]
    for tool in tools:
        runtime.tools.register_tool(tool)
    return len(tools)
