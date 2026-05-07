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

_MODULE = "kaos-content"
# Keep in lockstep with [project.version] in pyproject.toml.
_VERSION = "0.1.0a1"

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
        # crosses a heading boundary.
        expanded_section: bool = False
        section_heading_ref: str | None = None
        if expand_to_section:
            view = DocumentView(doc)
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
    ]
    for tool in tools:
        runtime.tools.register_tool(tool)
    return len(tools)
