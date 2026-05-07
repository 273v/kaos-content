"""Bridge ContentDocument ↔ kaos-core VFS artifacts.

Provides store/load helpers for persisting documents as artifacts and
generating MCP-friendly resource views (outline, markdown, tables, etc.).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from kaos_core.artifacts.models import (
    ArtifactManifest,
    ArtifactRef,
)
from kaos_core.types.enums import ArtifactRetentionPolicy, ArtifactRole

from kaos_content.errors import ArtifactTooLargeError

if TYPE_CHECKING:
    from kaos_core.base.context import KaosContext
    from kaos_core.registry.container import KaosRuntime

    from kaos_content.model.document import ContentDocument
    from kaos_content.model.tabular import TabularDocument


# Default upper bound for ``load_document`` / ``load_tabular`` reads.
# 16 MiB is large enough for typical legal documents (contracts,
# regulations, filings) but small enough to cap a malicious payload.
# Callers handling routinely-larger artifacts (SEC EDGAR 10-K filings
# can exceed 50 MB) should pass an explicit ``max_bytes=...`` or
# ``max_bytes=None`` to opt out of the cap.
DEFAULT_LOAD_MAX_BYTES: int = 16 * 1024 * 1024


# ---------------------------------------------------------------------------
# Name helpers
# ---------------------------------------------------------------------------


def unique_document_name(base: str, *, max_length: int = 80) -> str:
    """Generate a unique VFS-safe document name from a base string.

    Sanitizes the base name (lowercase, non-alphanumeric → hyphens) and
    appends an 8-character UUID suffix to prevent collisions in batch
    operations.

    Args:
        base: Human-readable base name (URL, title, etc.).
        max_length: Maximum length of the returned name.

    Returns:
        A sanitized, unique name like ``"miller-canfield-firm-a1b2c3d4"``.
    """
    import re
    from uuid import uuid4

    sanitized = re.sub(r"[^a-z0-9]+", "-", base.lower()).strip("-")
    if not sanitized:
        sanitized = "document"
    suffix = uuid4().hex[:8]
    # Truncate base to fit suffix within max_length
    max_base = max_length - len(suffix) - 1  # -1 for the separator hyphen
    if max_base < 1:
        max_base = 1
    if len(sanitized) > max_base:
        sanitized = sanitized[:max_base].rstrip("-")
    return f"{sanitized}-{suffix}"


# ---------------------------------------------------------------------------
# Store / Load
# ---------------------------------------------------------------------------


async def store_document(
    document: ContentDocument,
    runtime: KaosRuntime,
    context: KaosContext,
    *,
    name: str = "document",
    format: Literal["json", "markdown"] = "json",
    description: str | None = None,
    retention_policy: ArtifactRetentionPolicy = ArtifactRetentionPolicy.SESSION,
    metadata: dict[str, Any] | None = None,
) -> ArtifactManifest:
    """Serialize a ContentDocument and store as a VFS artifact.

    Returns the ArtifactManifest for the stored document.
    """
    if format == "json":
        payload = document.model_dump_json(indent=2).encode("utf-8")
        mime_type = "application/json"
        ext = "json"
    else:
        from kaos_content.serializers import serialize_markdown

        payload = serialize_markdown(document).encode("utf-8")
        mime_type = "text/markdown"
        ext = "md"

    vfs_path = f"documents/{name}.{ext}"
    ctx_path = context.get_vfs_path(vfs_path)
    await ctx_path.write_bytes(payload)

    return await runtime.artifacts.create_from_path(
        vfs_path,
        context_id=context.session_id,
        session_id=context.session_id,
        name=name,
        description=description or f"ContentDocument ({format})",
        mime_type=mime_type,
        role=ArtifactRole.BODY,
        provenance={"format": format, "title": document.metadata.title},
        retention_policy=retention_policy,
        metadata=metadata or {},
    )


async def load_document(
    artifact_ref: ArtifactRef | str,
    runtime: KaosRuntime,
    *,
    max_bytes: int | None = DEFAULT_LOAD_MAX_BYTES,
) -> ContentDocument:
    """Load a ContentDocument from a VFS artifact (JSON format only).

    Sec-5 (security finding #5) added ``max_bytes`` to bound the read
    against denial-of-service via huge artifact payloads. Pre-fix this
    function called ``runtime.artifacts.read_text()`` unconditionally;
    a multi-gigabyte artifact would force a full read followed by JSON
    + Pydantic validation, OOMing the process.

    Args:
        artifact_ref: ArtifactRef or artifact id string to load.
        runtime: kaos-core runtime providing the artifact store.
        max_bytes: Maximum artifact size in bytes. Defaults to
            :data:`DEFAULT_LOAD_MAX_BYTES` (16 MiB). Pass ``None`` to
            disable the cap (caller has explicitly thought about it —
            e.g. SEC 10-K filings routinely exceed the default).

    Raises:
        ArtifactTooLargeError: If the artifact's manifest reports a
            size larger than ``max_bytes``.

    NOTE: A second OOM vector remains in
    :meth:`ContentDocument.model_validate_json` itself — a maliciously
    deep nested JSON can still blow the parser/validator stack even
    inside the ``max_bytes`` cap. Streaming/incremental validation is
    the proper fix; tracked separately. The cap here bounds the
    raw-bytes read, which is the dominant concern.
    """
    from kaos_content.model.document import ContentDocument

    artifact_id = artifact_ref if isinstance(artifact_ref, str) else artifact_ref.artifact_id
    if max_bytes is not None:
        manifest = runtime.artifacts.get(artifact_id)
        size = getattr(manifest, "size", None)
        if size is not None and size > max_bytes:
            raise ArtifactTooLargeError(
                "Artifact exceeds load_document max_bytes cap",
                artifact_id=artifact_id,
                size=size,
                max_bytes=max_bytes,
                hint=(
                    "Pass max_bytes=<larger> if you expect this artifact "
                    "to be legitimately large, or max_bytes=None to "
                    "disable the cap entirely."
                ),
            )
    text = await runtime.artifacts.read_text(artifact_id)
    return ContentDocument.model_validate_json(text)


# ---------------------------------------------------------------------------
# Resource views — generate MCP resource payloads from a ContentDocument
# ---------------------------------------------------------------------------


def document_outline(document: ContentDocument) -> list[dict[str, Any]]:
    """Extract heading hierarchy as a lightweight outline."""
    from kaos_content.traversal import NodeIndex

    index = NodeIndex(document)
    return [
        {
            "depth": h.depth,
            "text": _extract_heading_text(h),
            "ref": ref,
        }
        for ref, h in _headings_with_refs(index)
    ]


def document_tables_summary(document: ContentDocument) -> list[dict[str, Any]]:
    """List tables with row/col counts and optional caption text."""
    from kaos_content.traversal import NodeIndex

    index = NodeIndex(document)
    result = []
    for i, table in enumerate(index.tables):
        summary: dict[str, Any] = {"index": i}
        total_rows = 0
        if table.head:
            total_rows += len(table.head.rows)
        for body_section in table.bodies:
            total_rows += len(body_section.rows)
        if table.foot:
            total_rows += len(table.foot.rows)
        summary["rows"] = total_rows
        summary["cols"] = len(table.col_specs) if table.col_specs else 0
        if table.caption and table.caption.short:
            from kaos_content.traversal import extract_text

            # Extract text from short caption inlines
            caption_parts = [extract_text(inline) for inline in table.caption.short]
            summary["caption"] = " ".join(caption_parts)
        result.append(summary)
    return result


def document_annotations_by_type(
    document: ContentDocument,
    annotation_type: str | None = None,
) -> list[dict[str, Any]]:
    """List annotations, optionally filtered by type."""
    results = []
    for ann in document.annotations:
        if annotation_type and ann.type.value != annotation_type:
            continue
        results.append(
            {
                "id": ann.id,
                "type": ann.type.value,
                "targets": [
                    {
                        "node_ref": t.node_ref,
                        "start_offset": t.start_offset,
                        "end_offset": t.end_offset,
                    }
                    for t in ann.targets
                ],
                "body": ann.body,
            }
        )
    return results


def document_definitions(document: ContentDocument) -> dict[str, str]:
    """Return the document's definition dictionary."""
    return dict(document.definitions)


def document_node_subtree(
    document: ContentDocument,
    node_ref: str,
) -> dict[str, Any]:
    """Return a single node's JSON subtree by ref."""
    from kaos_content.traversal import NodeIndex

    index = NodeIndex(document)
    node = index.get(node_ref)
    if node is None:
        msg = f"Node not found: {node_ref}"
        raise KeyError(msg)
    return node.model_dump(mode="json")


def document_metadata(document: ContentDocument) -> dict[str, Any]:
    """Return document metadata as a dict."""
    return document.metadata.model_dump(mode="json")


def document_to_summary(document: ContentDocument, *, max_length: int = 500) -> str:
    """Generate a concise text summary for inline MCP results."""
    from kaos_content.serializers import serialize_text

    text = serialize_text(document)
    title = document.metadata.title or "Untitled"
    block_count = len(document.body)
    prefix = f"{title} ({block_count} blocks)"
    if len(text) <= max_length:
        return f"{prefix}\n\n{text}"
    return f"{prefix}\n\n{text[:max_length]}..."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_heading_text(heading: Any) -> str:
    """Extract plain text from a heading node."""
    from kaos_content.traversal import extract_text

    return extract_text(heading)


def _headings_with_refs(index: Any) -> list[tuple[str, Any]]:
    """Return (ref, heading) pairs from index."""
    from kaos_content.model.blocks import Heading

    result = []
    for ref, node in index._ref_map.items():
        if isinstance(node, Heading):
            result.append((ref, node))
    return result


def document_to_resource_views(
    document: ContentDocument,
    manifest: ArtifactManifest,
) -> dict[str, Any]:
    """Generate all MCP resource view payloads for a document.

    Returns a dict mapping resource template suffixes to their payloads:
    - "metadata" → document metadata dict
    - "outline" → heading outline list
    - "tables" → table summaries
    - "annotations" → all annotations
    - "definitions" → definition dict
    - "markdown" → serialized markdown string
    """
    from kaos_content.serializers import serialize_markdown

    return {
        "metadata": document_metadata(document),
        "outline": document_outline(document),
        "tables": document_tables_summary(document),
        "annotations": document_annotations_by_type(document),
        "definitions": document_definitions(document),
        "markdown": serialize_markdown(document),
    }


# ---------------------------------------------------------------------------
# TabularDocument artifact support
# ---------------------------------------------------------------------------


async def store_tabular(
    document: TabularDocument,
    runtime: KaosRuntime,
    context: KaosContext,
    *,
    name: str = "tabular",
    description: str | None = None,
    retention_policy: ArtifactRetentionPolicy = ArtifactRetentionPolicy.SESSION,
    metadata: dict[str, Any] | None = None,
) -> ArtifactManifest:
    """Serialize a TabularDocument and store as a VFS artifact (JSON).

    Returns the ArtifactManifest for the stored document.
    """

    payload = _tabular_to_json(document).encode("utf-8")
    mime_type = "application/json"

    vfs_path = f"tabular/{name}.json"
    ctx_path = context.get_vfs_path(vfs_path)
    await ctx_path.write_bytes(payload)

    return await runtime.artifacts.create_from_path(
        vfs_path,
        context_id=context.session_id,
        session_id=context.session_id,
        name=name,
        description=description or "TabularDocument (json)",
        mime_type=mime_type,
        role=ArtifactRole.BODY,
        provenance={
            "format": "json",
            "title": document.metadata.title,
            "table_count": len(document.tables),
        },
        retention_policy=retention_policy,
        metadata=metadata or {},
    )


async def load_tabular(
    artifact_ref: ArtifactRef | str,
    runtime: KaosRuntime,
    *,
    max_bytes: int | None = DEFAULT_LOAD_MAX_BYTES,
) -> TabularDocument:
    """Load a TabularDocument from a VFS artifact (JSON format).

    Sec-5 (security finding #5) added ``max_bytes`` — see
    :func:`load_document` for the full rationale. The same cap and
    same opt-out semantics apply.
    """
    artifact_id = artifact_ref if isinstance(artifact_ref, str) else artifact_ref.artifact_id
    if max_bytes is not None:
        manifest = runtime.artifacts.get(artifact_id)
        size = getattr(manifest, "size", None)
        if size is not None and size > max_bytes:
            raise ArtifactTooLargeError(
                "Artifact exceeds load_tabular max_bytes cap",
                artifact_id=artifact_id,
                size=size,
                max_bytes=max_bytes,
                hint=(
                    "Pass max_bytes=<larger> if you expect this artifact "
                    "to be legitimately large, or max_bytes=None to "
                    "disable the cap entirely."
                ),
            )
    text = await runtime.artifacts.read_text(artifact_id)
    return _tabular_from_json(text)


def tabular_summary(document: TabularDocument) -> dict[str, Any]:
    """Generate a summary dict for MCP resource views.

    Includes table names, dimensions, column types — enough for
    an agent to decide which table to query next.
    """
    tables_info = []
    for table in document.tables:
        tables_info.append(
            {
                "name": table.name,
                "row_count": table.row_count,
                "column_count": len(table.columns),
                "columns": [
                    {
                        "name": c.name,
                        "type": c.column_type.value,
                        "nullable": c.nullable,
                    }
                    for c in table.columns
                ],
            }
        )

    return {
        "title": document.metadata.title,
        "table_count": len(document.tables),
        "total_rows": sum(t.row_count for t in document.tables),
        "tables": tables_info,
    }


def tabular_schema(table: Any) -> dict[str, Any]:
    """Generate a detailed schema dict for a single Table.

    Includes column names, types, nullability, and metadata.
    Useful for MCP ``describe`` operations.
    """
    return {
        "name": table.name,
        "row_count": table.row_count,
        "columns": [
            {
                "name": c.name,
                "type": c.column_type.value,
                "nullable": c.nullable,
                "metadata": c.metadata if c.metadata else None,
            }
            for c in table.columns
        ],
        "metadata": table.metadata if table.metadata else None,
    }


# ---------------------------------------------------------------------------
# JSON serialization helpers for TabularDocument
# ---------------------------------------------------------------------------


def _tabular_to_json(document: TabularDocument) -> str:
    """Serialize TabularDocument to JSON string.

    Tables use dataclasses so we build the JSON dict manually.
    Cell values are converted to JSON-compatible types.
    """
    import json

    from kaos_content.serializers.tabular import _format_value_json

    doc_dict: dict[str, Any] = {
        "metadata": document.metadata.model_dump(mode="json"),
        "tables": [],
        "provenance": document.provenance.model_dump(mode="json") if document.provenance else None,
    }

    for table in document.tables:
        table_dict: dict[str, Any] = {
            "name": table.name,
            "columns": [
                {
                    "name": c.name,
                    "column_type": c.column_type.value,
                    "nullable": c.nullable,
                    "metadata": c.metadata if c.metadata else {},
                }
                for c in table.columns
            ],
            "rows": [[_format_value_json(v) for v in row] for row in table.rows],
            "row_count": table.row_count,
            "metadata": table.metadata if table.metadata else {},
        }
        doc_dict["tables"].append(table_dict)

    return json.dumps(doc_dict, indent=2, ensure_ascii=False, default=str)


def _tabular_from_json(text: str) -> TabularDocument:
    """Deserialize TabularDocument from JSON string."""
    import json

    from kaos_content.model.tabular import Column, ColumnType, Table, TabularDocument

    data = json.loads(text)

    tables = []
    for td in data.get("tables", []):
        columns = tuple(
            Column(
                name=cd["name"],
                column_type=ColumnType(cd["column_type"]),
                nullable=cd.get("nullable", True),
                metadata=cd.get("metadata", {}),
            )
            for cd in td.get("columns", [])
        )
        rows = tuple(tuple(v for v in row) for row in td.get("rows", []))
        tables.append(
            Table(
                name=td["name"],
                columns=columns,
                rows=rows,
                row_count=td.get("row_count", len(rows)),
                metadata=td.get("metadata", {}),
            )
        )

    metadata_dict = data.get("metadata", {})
    from kaos_content.model.metadata import DocumentMetadata

    prov = None
    if data.get("provenance"):
        from kaos_content.model.attr import Provenance

        prov = Provenance.model_validate(data["provenance"])

    return TabularDocument(
        metadata=DocumentMetadata.model_validate(metadata_dict),
        tables=tuple(tables),
        provenance=prov,
    )
