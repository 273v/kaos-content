"""StructureView: kaos-content wrapper around the kaos-nlp-core P7
structure layer.

Composes the per-line scorer (P7.1), Viterbi sequence decoder (P7.4),
and heading-stack inferencer (P7.6) with a ``ContentDocument`` /
``DocumentView`` to produce typed [``Annotation``] records of the four
new types ([``HEADING_CANDIDATE``], [``BOILERPLATE``], [``TABLE_ROW``],
[``METADATA``]).

This is the *AST attachment* layer per
``kaos-nlp-core/docs/INTEGRATION_BOUNDARIES.md``: the Rust core operates
on opaque strings and primitive types; this module is responsible for
turning the per-line label sequence into ``(node_ref, span,
AnnotationType, body)`` tuples that the rest of kaos-content
understands.

Falls back gracefully when the optional ``[nlp]`` extra (kaos-nlp-core)
is not installed: ``annotate_structure(doc)`` raises ``ImportError``
with a helpful message.

Coordinate model: each emitted Annotation targets exactly one block
(per-block character offsets), matching the existing ``Annotation``
contract. Multi-line headings are emitted as one annotation per line
record so the Viterbi label sequence stays 1-1 with the resulting
annotations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from kaos_content.model.annotation import (
    Annotation,
    AnnotationTarget,
    AnnotationType,
)
from kaos_content.model.document import ContentDocument
from kaos_content.serializers.text import serialize_text

if TYPE_CHECKING:
    pass

# ─── Public dataclasses ───────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class StructureLabeling:
    """Output of :func:`annotate_structure`.

    ``annotations`` is a tuple of new ``Annotation`` records to add to
    the document; ``label_counts`` reports how many lines fell into
    each of the seven label classes (useful for shape-of-document
    summaries).

    The ``annotations`` carry no ``id``s yet — callers choose how to
    namespace them (typically by adding them via
    :meth:`ContentDocument.with_extra_annotations` after pre-pending
    a stable prefix).
    """

    annotations: tuple[Annotation, ...]
    label_counts: dict[str, int]
    n_lines: int


# ─── Entry point ──────────────────────────────────────────────────────────


def annotate_structure(
    document: ContentDocument,
    *,
    enum_lexicon: str | None = None,
    heading_lexicon: str | None = None,
    hierarchy_lexicon: str | None = None,
    weights: dict[str, float] | None = None,
    threshold: float | None = None,
    decoder: dict[str, Any] | None = None,
    annotation_id_prefix: str = "structure",
) -> StructureLabeling:
    """Run the kaos-nlp-core P7 pipeline against ``document`` and emit
    typed structural annotations.

    Internally serializes the document to plain text (the same path
    used by :func:`kaos_content.serializers.serialize_text`), runs the
    pipeline, then maps each non-blank line back to the
    ``(node_ref, char_offset)`` that produced it. The text serializer
    is structure-preserving (one line per block roughly), so the
    line-index → block mapping is straightforward.

    Falls back to ``ImportError`` if the optional ``[nlp]`` extra is
    not installed.
    """
    try:
        from kaos_nlp_core.structure import label_lines
    except ImportError as exc:
        msg = (
            "kaos-content[nlp] extra is required for annotate_structure. "
            "Install with: pip install 'kaos-content[nlp]' or 'kaos-nlp-core'."
        )
        raise ImportError(msg) from exc

    # 1. Serialize the document to plain text. We rely on `serialize_text`
    #    producing one logical line per block — paragraphs collapse to one
    #    line, headings remain on their own line, and lists/tables emit
    #    one line per row. This matches the Rust scorer's "physical line"
    #    granularity.
    text = serialize_text(document)
    if not text or not text.strip():
        return StructureLabeling(
            annotations=(),
            label_counts={
                "blank": 0,
                "heading": 0,
                "body": 0,
                "list_item": 0,
                "table_row": 0,
                "metadata": 0,
                "boilerplate": 0,
            },
            n_lines=0,
        )

    # 2. Run the pipeline.
    scoring: dict[str, Any] = {}
    if heading_lexicon is not None:
        scoring["heading_lexicon"] = heading_lexicon
    if hierarchy_lexicon is not None:
        scoring["hierarchy_lexicon"] = hierarchy_lexicon
    if weights is not None:
        scoring["weights"] = weights
    if threshold is not None:
        scoring["threshold"] = threshold
    result = label_lines(
        text,
        enum_lexicon=enum_lexicon,
        scoring=scoring or None,
        decoder=decoder,
    )

    # 3. Map line indices → block targets. We walk the document in the
    #    same flat block order the text serializer used, building a
    #    per-line list of `node_ref`s (one per non-blank source line).
    line_to_node_ref = _map_lines_to_node_refs(document, text)

    # 4. Build heading-candidate annotations from the inferencer output,
    #    plus boilerplate / table_row / metadata annotations from the
    #    decoder labels.
    annotations: list[Annotation] = []
    label_counts = {
        "blank": 0,
        "heading": 0,
        "body": 0,
        "list_item": 0,
        "table_row": 0,
        "metadata": 0,
        "boilerplate": 0,
    }
    candidate_by_line = {c.line_index: c for c in result.candidates}
    hierarchy_lex_name = hierarchy_lexicon or "english_legal_us"

    for i, label in enumerate(result.labels):
        label_counts[label] = label_counts.get(label, 0) + 1
        ref_and_span = line_to_node_ref.get(i)
        if ref_and_span is None:
            continue
        node_ref, start_offset, end_offset = ref_and_span
        target = AnnotationTarget(
            node_ref=node_ref,
            start_offset=start_offset,
            end_offset=end_offset,
        )
        ann_id = f"{annotation_id_prefix}-{i}"
        if label == "heading":
            c = candidate_by_line.get(i)
            body = {
                "score": float(c.score) if c else float(result.features[i].score),
                "hierarchy_level": int(c.hierarchy_level) if c and c.hierarchy_level > 0 else None,
                "numeric_depth": int(c.numeric_depth) if c and c.numeric_depth > 0 else None,
                "atx_depth": int(c.atx_depth) if c and c.atx_depth > 0 else None,
                "enumerator_kind": c.enumerator_kind if c else None,
                "lexicon_used": hierarchy_lex_name,
            }
            annotations.append(
                Annotation(
                    id=ann_id,
                    type=AnnotationType.HEADING_CANDIDATE,
                    targets=(target,),
                    body=body,
                )
            )
        elif label == "boilerplate":
            annotations.append(
                Annotation(
                    id=ann_id,
                    type=AnnotationType.BOILERPLATE,
                    targets=(target,),
                    body={
                        "kind": "unknown",
                        "occurrences": 1,
                    },
                )
            )
        elif label == "table_row":
            annotations.append(
                Annotation(
                    id=ann_id,
                    type=AnnotationType.TABLE_ROW,
                    targets=(target,),
                    body={"row_index": i},
                )
            )
        elif label == "metadata":
            annotations.append(
                Annotation(
                    id=ann_id,
                    type=AnnotationType.METADATA,
                    targets=(target,),
                    body={"kind": "unknown"},
                )
            )
        # body / list_item / blank: no annotation emitted (default class).

    return StructureLabeling(
        annotations=tuple(annotations),
        label_counts=label_counts,
        n_lines=len(result.labels),
    )


# ─── Internal: line → node_ref mapping ────────────────────────────────────


def _block_ref_per_line(block: Any, base_ref: str) -> list[str]:
    """Return one node_ref per text-serialized line for a single block.

    Single-line blocks (paragraphs, headings) collapse to ``[base_ref]``.
    Multi-line blocks return as many entries as the block emits text
    lines, with sub-refs that descend into the AST when possible:

    * **Table** — each row line maps to the row's ref under whichever
      section (head / bodies[i] / foot) it belongs to. Cell-level refs
      are not used because the serializer joins cells with ``" | "`` on
      one line, so the line owns the row, not any single cell.
    * **List blocks** (bullet / ordered / definition) — N lines, all
      sharing ``base_ref``. The sub-list-item ref is reachable via the
      ``start_offset`` / ``end_offset`` on the AnnotationTarget plus the
      caller's NodeIndex if a cell-level anchor is needed.
    * **Code block / math block / raw block** — multi-line content all
      sharing ``base_ref``.

    Lines whose serialized form is purely whitespace (blank) get an
    empty string for that slot, signalling the caller to skip the line.

    The total length of the returned list matches
    ``rendered.split("\\n")`` for the block's standalone serialization.
    """
    rendered = _render_block_for_layout(block)
    if not rendered:
        return []
    line_count = rendered.count("\n") + 1

    nt = getattr(block, "node_type", None)
    if nt == "table":
        # Walk head → bodies[*] → foot in the same order serialize_text uses.
        refs: list[str] = []
        head = getattr(block, "head", None)
        if head is not None:
            for ri, _row in enumerate(getattr(head, "rows", ())):
                refs.append(f"{base_ref}/head/rows/{ri}")
        for bi, body_section in enumerate(getattr(block, "bodies", ())):
            for ri, _row in enumerate(getattr(body_section, "rows", ())):
                refs.append(f"{base_ref}/bodies/{bi}/rows/{ri}")
        foot = getattr(block, "foot", None)
        if foot is not None:
            for ri, _row in enumerate(getattr(foot, "rows", ())):
                refs.append(f"{base_ref}/foot/rows/{ri}")
        # Pad / truncate to the actual rendered line count. The serializer
        # may emit fewer lines (e.g. empty rows are skipped) — fall back to
        # the table's base_ref for any unmatched line slots.
        if len(refs) < line_count:
            refs.extend([base_ref] * (line_count - len(refs)))
        elif len(refs) > line_count:
            refs = refs[:line_count]
        return refs

    # Default: every emitted line shares the block's base ref.
    return [base_ref] * line_count


def _render_block_for_layout(block: Any) -> str:
    """Render a single block to its serialized text in isolation.

    Used to compute the line-count contribution of each block so the
    line-to-ref mapping can be assembled deterministically. Mirrors
    ``_TextContext.render_block`` but with a fresh context per call.
    """
    from kaos_content.serializers.text import _TextContext  # local import to avoid cycle

    ctx = _TextContext(
        block_separator="\n\n",
        heading_separator="\n",
        list_indent="  ",
        table_format="plain",
    )
    return ctx.render_block(block)


def _map_lines_to_node_refs(
    document: ContentDocument, serialized: str
) -> dict[int, tuple[str, int, int]]:
    """Build a `line_index -> (node_ref, start_offset, end_offset)` mapping.

    Walks ``document.body`` in tree order (and footnotes after), assigning
    each top-level block an inclusive range of serialized lines and
    descending into table rows / footnote-block contents for sub-block
    granularity. Per-line ``start_offset`` / ``end_offset`` are character
    offsets *within the owning ref's serialized text*.

    For a single-line block (paragraph, heading) the range is
    ``(0, len(line))``. For multi-line blocks (lists, code blocks,
    multi-row tables) the offsets accumulate across the block's lines.
    Tables additionally split refs per row so each table line points
    at its own ``rows/{j}`` node.

    Footnotes follow the body in the same order ``serialize_text``
    emits them: each footnote's ``[key]: <body>`` lines map to
    ``#/footnotes/{key}/{block_idx}`` refs.

    Returns ``{}`` for blank lines and any lines past the last
    serialized block.
    """
    out: dict[int, tuple[str, int, int]] = {}
    layout: list[tuple[str, int]] = []

    # Build the per-line layout: a list of (node_ref, char_offset_within_ref)
    # entries — one entry per **non-blank** serialized line, in the same
    # order the serializer emitted them. Char offsets reset whenever the
    # ref changes; trailing newlines emitted by heading / list separators
    # are not added to the layout but DO advance the offset within the ref.
    def _append_block_layout(block: Any, base_ref: str) -> None:
        per_line_refs = _block_ref_per_line(block, base_ref)
        if not per_line_refs:
            return
        rendered = _render_block_for_layout(block)
        rendered_lines = rendered.split("\n")
        char_offset_by_ref: dict[str, int] = {}
        for k, line_text in enumerate(rendered_lines):
            ref = per_line_refs[k] if k < len(per_line_refs) else per_line_refs[-1]
            offset = char_offset_by_ref.get(ref, 0)
            if line_text:  # only non-blank lines participate in the layout
                layout.append((ref, offset))
            # Advance the per-ref offset whether the line was blank or not,
            # so multi-line blocks (lists, tables) keep accurate offsets
            # across blank-line interruptions inside the block.
            char_offset_by_ref[ref] = offset + len(line_text) + 1

    for i, block in enumerate(document.body):
        _append_block_layout(block, f"#/body/{i}")

    for key, blocks in document.footnotes.items():
        for j, block in enumerate(blocks):
            _append_block_layout(block, f"#/footnotes/{key}/{j}")

    # Walk the serialized output, advancing through `layout` on every
    # non-blank line. Blank lines either separate blocks (skip them) or
    # appear inside the body (rare; also skipped — they map to nothing).
    layout_idx = 0
    for i, line in enumerate(serialized.split("\n")):
        if not line.strip():
            continue
        if layout_idx >= len(layout):
            break
        ref, start = layout[layout_idx]
        out[i] = (ref, start, start + len(line))
        layout_idx += 1
    return out


# ─── Convenience: attach annotations to the document ─────────────────────


def with_structure_annotations(
    document: ContentDocument,
    *,
    enum_lexicon: str | None = None,
    heading_lexicon: str | None = None,
    hierarchy_lexicon: str | None = None,
    weights: dict[str, float] | None = None,
    threshold: float | None = None,
    decoder: dict[str, Any] | None = None,
    annotation_id_prefix: str = "structure",
) -> ContentDocument:
    """Return a new ContentDocument with the P7 structural annotations
    attached. Convenience over :func:`annotate_structure` +
    :meth:`ContentDocument.with_extra_annotations`.
    """
    labeling = annotate_structure(
        document,
        enum_lexicon=enum_lexicon,
        heading_lexicon=heading_lexicon,
        hierarchy_lexicon=hierarchy_lexicon,
        weights=weights,
        threshold=threshold,
        decoder=decoder,
        annotation_id_prefix=annotation_id_prefix,
    )
    if not labeling.annotations:
        return document
    new_annotations = tuple(document.annotations) + labeling.annotations
    return document.model_copy(update={"annotations": new_annotations})


# ─── Heading promotion (T3c: SectionView integration) ───────────────────


def _resolve_heading_depth(body: dict[str, Any]) -> int:
    """Pick a 1-6 heading depth from a HeadingCandidateBody dict.

    Priority order matches the structural strength of each signal:

    1. ``atx_depth`` — explicit ``#``/``##`` markdown markers are the
       most authoritative depth source.
    2. ``hierarchy_level`` — keyword-driven (``Chapter``, ``Section``,
       ``Article`` from the configured hierarchy lexicon).
    3. ``numeric_depth`` — count of decimal segments in an enumerator
       (``1.`` → 1, ``1.2`` → 2, ``1.2.3`` → 3).
    4. ``1`` — fallback when no signal fired.

    Result is clamped to ``[1, 6]`` so it satisfies ``Heading``'s
    constructor invariant.
    """
    for field in ("atx_depth", "hierarchy_level", "numeric_depth"):
        value = body.get(field)
        if value is not None and value > 0:
            return max(1, min(6, int(value)))
    return 1


def with_inferred_structure(
    document: ContentDocument,
    *,
    enum_lexicon: str | None = None,
    heading_lexicon: str | None = None,
    hierarchy_lexicon: str | None = None,
    weights: dict[str, float] | None = None,
    threshold: float | None = None,
    decoder: dict[str, Any] | None = None,
    annotation_id_prefix: str = "structure",
) -> ContentDocument:
    """Promote P7 heading candidates to typed ``Heading`` blocks.

    Many documents arrive with no typed ``Heading`` blocks at all —
    PDFs and plain-text imports come through as a flat sequence of
    ``Paragraph`` blocks, and the structure layer's job is to identify
    which paragraphs are *really* section headings. Until those blocks
    are typed as ``Heading``, ``DocumentView.sections`` cannot compute
    a section tree (it only recognizes ``Heading`` blocks).

    This function runs :func:`annotate_structure`, then for every
    top-level body block whose ref is the target of a
    ``HEADING_CANDIDATE`` annotation, replaces the block with a
    ``Heading`` carrying:

    * the same ``children`` (inline content) — extracted via
      ``extract_text`` and wrapped as a single ``Text`` inline if the
      original block lacked an ``inline`` children field;
    * a ``depth`` resolved by :func:`_resolve_heading_depth`;
    * the same ``id`` and ``provenance`` (so downstream node_refs and
      page lookups are stable);
    * the original block's ``attr`` (so domain-specific classes like
      ``rev-ins`` are preserved).

    Blocks that are already ``Heading`` are left untouched (idempotent
    — re-running the function does not double-promote). Blocks whose
    candidate annotation targets a *sub-block* ref (e.g.
    ``#/body/3/children/0`` for a heading inside a div) are also left
    untouched in v1; only top-level body promotions are supported. This
    keeps the structure invariants (Block/Inline discipline, no
    Heading-inside-Heading) intact without a recursive transform.

    The full annotation payload (HEADING_CANDIDATE bodies, BOILERPLATE,
    TABLE_ROW, METADATA) is also attached so downstream consumers can
    still read score / lexicon / etc.
    """
    from kaos_content.model.annotation import AnnotationType
    from kaos_content.model.blocks import Heading, Paragraph
    from kaos_content.model.inlines import Text
    from kaos_content.traversal.visitor import extract_text

    labeling = annotate_structure(
        document,
        enum_lexicon=enum_lexicon,
        heading_lexicon=heading_lexicon,
        hierarchy_lexicon=hierarchy_lexicon,
        weights=weights,
        threshold=threshold,
        decoder=decoder,
        annotation_id_prefix=annotation_id_prefix,
    )

    # Index heading-candidate annotations by the top-level body ref they
    # target. Sub-block refs (containing "/children/", "/rows/", etc.)
    # are intentionally ignored — promoting a sub-block to Heading would
    # break the Block/Inline structural invariant.
    candidates_by_body_ref: dict[int, dict[str, Any]] = {}
    for ann in labeling.annotations:
        if ann.type != AnnotationType.HEADING_CANDIDATE:
            continue
        for tgt in ann.targets:
            ref = tgt.node_ref
            if not ref.startswith("#/body/"):
                continue
            tail = ref.removeprefix("#/body/")
            if "/" in tail:
                # Sub-block ref like "#/body/3/children/0" — skip.
                continue
            try:
                idx = int(tail)
            except ValueError:
                continue
            # First annotation per body block wins (deterministic).
            candidates_by_body_ref.setdefault(idx, dict(ann.body))

    if not candidates_by_body_ref:
        # Nothing to promote — still attach the labeling annotations so
        # callers see the same annotation payload as
        # `with_structure_annotations`.
        if not labeling.annotations:
            return document
        new_annotations = tuple(document.annotations) + labeling.annotations
        return document.model_copy(update={"annotations": new_annotations})

    new_body: list[Any] = []
    for i, block in enumerate(document.body):
        if i not in candidates_by_body_ref or isinstance(block, Heading):
            new_body.append(block)
            continue
        body_dict = candidates_by_body_ref[i]
        depth = _resolve_heading_depth(body_dict)
        # Build inline children: prefer the block's own inline children
        # when it's a Paragraph (preserves formatting marks); fall back
        # to flattened text for any other block shape (lists, tables,
        # blockquotes — rare but defensible).
        if isinstance(block, Paragraph):
            children: tuple[Any, ...] = block.children
        else:
            text_value = extract_text(block).strip()
            if not text_value:
                # Don't emit an empty heading — leave the original block.
                new_body.append(block)
                continue
            children = (Text(value=text_value),)
        promoted = Heading(
            id=block.id,
            depth=depth,
            children=children,
            provenance=block.provenance,
            attr=block.attr,
        )
        new_body.append(promoted)

    new_annotations = tuple(document.annotations) + labeling.annotations
    return document.model_copy(
        update={
            "body": tuple(new_body),
            "annotations": new_annotations,
        }
    )


# ─── Context windowing (P12) — grep -A/-B for AST refs ───────────────────


@dataclass(frozen=True, slots=True)
class ContextWindow:
    """A windowed slice of body blocks around a target node_ref.

    The result of expanding context around a search hit. ``blocks`` is
    the contiguous slice of top-level body blocks in source order;
    ``block_refs`` is the parallel list of JSON-pointer refs (so the
    agent can re-cite individual blocks). ``target_index`` is the
    position of the originally-requested node_ref within ``blocks``,
    or ``None`` when the ref didn't resolve to a single block (which
    should not happen for valid input — guarded at the API surface).
    """

    blocks: tuple[Any, ...]
    block_refs: tuple[str, ...]
    target_node_ref: str
    target_index: int | None
    expanded_to_section: bool = False
    enclosing_section_ref: str | None = None


def get_context_window(
    document: ContentDocument,
    node_ref: str,
    *,
    before_blocks: int = 2,
    after_blocks: int = 2,
    expand_to_section: bool = False,
) -> ContextWindow:
    """Pure-function counterpart to :class:`ContextWindowTool`.

    For library callers (the agent's retrieval-tools layer, custom
    pipelines, tests) that don't want to go through the MCP artifact
    round-trip. Same semantics as the tool: take ``node_ref``, return a
    grep-A/-B window over the body. Sub-block refs are resolved to
    their containing top-level body block.

    Raises ``ValueError`` for refs outside the body namespace or refs
    whose body index is out of range — the tool wrapper translates
    these into ``ToolResult.create_error`` for MCP consumers.
    """
    body_idx = _resolve_body_index_for_window(node_ref)
    if body_idx is None:
        msg = (
            f"node_ref {node_ref!r} does not address a top-level body block. "
            "Sub-block refs are mapped to their containing body block; "
            "footnote / metadata refs are not supported."
        )
        raise ValueError(msg)
    if body_idx >= len(document.body):
        msg = (
            f"node_ref {node_ref!r} resolves to body index {body_idx} "
            f"but the document only has {len(document.body)} body blocks."
        )
        raise ValueError(msg)
    if before_blocks < 0 or after_blocks < 0:
        msg = "before_blocks and after_blocks must be non-negative"
        raise ValueError(msg)

    start = max(0, body_idx - before_blocks)
    end = min(len(document.body) - 1, body_idx + after_blocks)
    expanded = False
    enclosing_ref: str | None = None
    if expand_to_section:
        from kaos_content.views.document_view import DocumentView

        view = DocumentView(document)
        for sv in view.flat_sections:
            if sv.heading_ref is None:
                continue
            sec_start = _resolve_body_index_for_window(sv.heading_ref)
            if sec_start is None:
                continue
            sec_end = sec_start + len(sv.blocks) - 1
            if sec_start <= body_idx <= sec_end:
                if start < sec_start or end > sec_end:
                    start = max(start, sec_start)
                    end = min(end, sec_end)
                    expanded = True
                    enclosing_ref = sv.heading_ref
                break

    blocks = tuple(document.body[start : end + 1])
    refs = tuple(f"#/body/{i}" for i in range(start, end + 1))
    target_index = body_idx - start if start <= body_idx <= end else None
    return ContextWindow(
        blocks=blocks,
        block_refs=refs,
        target_node_ref=node_ref,
        target_index=target_index,
        expanded_to_section=expanded,
        enclosing_section_ref=enclosing_ref,
    )


def _resolve_body_index_for_window(node_ref: str) -> int | None:
    """Same resolver as the ContextWindowTool — kept here for the
    pure-function path so kaos-content callers don't have to import
    the tool module just for the helper.
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


__all__ = [
    "ContextWindow",
    "StructureLabeling",
    "annotate_structure",
    "get_context_window",
    "with_inferred_structure",
    "with_structure_annotations",
]
