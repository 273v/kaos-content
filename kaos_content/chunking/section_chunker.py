"""AST-aware document chunking at heading boundaries.

Splits a ``ContentDocument`` into smaller chunks suitable for LLM processing,
respecting document structure. Each chunk is a valid ``ContentDocument``.
"""

from __future__ import annotations

from typing import Any

from kaos_core.logging import get_logger

from kaos_content.model.annotation import Annotation
from kaos_content.model.blocks import Heading
from kaos_content.model.document import ContentDocument
from kaos_content.model.metadata import DocumentMetadata
from kaos_content.traversal.index import NodeIndex
from kaos_content.traversal.visitor import extract_text

logger = get_logger(__name__)

# Block types that should never be split mid-content
_UNSPLITTABLE = frozenset({"table", "codeblock", "math_block"})


class SectionChunker:
    """Split a document at heading boundaries.

    Parameters
    ----------
    max_chars:
        Maximum character count per chunk (measured by ``extract_text``).
        When a section exceeds this limit, it is split at the next paragraph
        boundary. 0 means no limit.
    split_depth:
        Heading depth at which to split. ``split_depth=2`` splits at h1 and h2
        boundaries but not h3+.
    overlap_paragraphs:
        Number of trailing paragraphs from the previous chunk to repeat at the
        start of the next chunk, providing context overlap.
    """

    def __init__(
        self,
        max_chars: int = 8000,
        split_depth: int = 2,
        overlap_paragraphs: int = 0,
    ) -> None:
        self.max_chars = max_chars
        self.split_depth = split_depth
        self.overlap_paragraphs = overlap_paragraphs

    @classmethod
    def from_outline(
        cls,
        document: ContentDocument,
        *,
        max_chars: int = 8000,
        split_depth: int = 2,
        overlap_paragraphs: int = 0,
        enum_lexicon: str | None = None,
        heading_lexicon: str | None = None,
        hierarchy_lexicon: str | None = None,
        weights: dict[str, float] | None = None,
        threshold: float | None = None,
        decoder: dict[str, Any] | None = None,
        promote_inferred: bool = True,
    ) -> list[ContentDocument]:
        """Outline-aware chunking — runs structure inference before splitting.

        Many real documents (PDFs, plain-text imports, OCR output) arrive
        with no typed ``Heading`` blocks at all, so the standard
        :meth:`chunk` collapses them into a single huge chunk. This
        constructor first calls :func:`kaos_content.structure.with_inferred_structure`
        to promote P7 heading candidates to typed ``Heading`` blocks,
        then runs the same chunking pipeline on the promoted document.
        Each emitted chunk additionally carries a ``heading_path`` in
        ``metadata.extra`` — the list of ancestor heading texts forming
        the section breadcrumb, computed from the promoted Heading
        depth stack at the chunk's entry point.

        Set ``promote_inferred=False`` to skip the structure-inference
        step and chunk only on the document's existing typed Heading
        blocks. Useful when callers have already promoted manually or
        when they want strictly literal chunking.

        The structure-inference parameters (``enum_lexicon``,
        ``heading_lexicon``, ``hierarchy_lexicon``, ``weights``,
        ``threshold``, ``decoder``) match
        :func:`with_inferred_structure`'s signature exactly so callers
        can pass per-domain lexicons through.

        Returns the same chunk list shape as :meth:`chunk` — every chunk
        is a valid ``ContentDocument``. Raises ``ImportError`` if
        ``promote_inferred=True`` but the optional ``[nlp]`` extra is
        not installed.
        """
        if promote_inferred:
            from kaos_content.structure import with_inferred_structure

            document = with_inferred_structure(
                document,
                enum_lexicon=enum_lexicon,
                heading_lexicon=heading_lexicon,
                hierarchy_lexicon=hierarchy_lexicon,
                weights=weights,
                threshold=threshold,
                decoder=decoder,
            )
        chunker = cls(
            max_chars=max_chars,
            split_depth=split_depth,
            overlap_paragraphs=overlap_paragraphs,
        )
        chunks = chunker.chunk(document)
        return [_attach_heading_path(chunk, document, chunker.split_depth) for chunk in chunks]

    def chunk(self, document: ContentDocument) -> list[ContentDocument]:
        """Split *document* into chunks. Each chunk is a valid ContentDocument.

        Chunks inherit the parent document's metadata (with ``chunk_index``
        and ``chunk_total`` added to ``extra``). Annotations are partitioned
        to the chunk containing their target nodes. Footnotes referenced in
        a chunk are included in that chunk.
        """
        if not document.body:
            return [document]

        # Step 1: Split body into raw sections at heading boundaries
        raw_sections = self._split_at_headings(document)

        # Step 2: Enforce max_chars — further split oversized sections
        if self.max_chars > 0:
            sections = []
            for section in raw_sections:
                sections.extend(self._enforce_max_chars(section))
        else:
            sections = raw_sections

        if not sections:
            return [document]

        # Step 3: Apply overlap
        if self.overlap_paragraphs > 0:
            sections = self._apply_overlap(sections)

        # Step 4: Build chunk documents
        total = len(sections)
        index = NodeIndex(document)
        chunks: list[ContentDocument] = []

        for ci, section_blocks in enumerate(sections):
            # Collect all node IDs in this chunk
            chunk_node_ids = self._collect_node_ids(section_blocks)

            # Partition footnotes: include only those referenced in this chunk
            chunk_footnotes = self._partition_footnotes(section_blocks, document.footnotes)

            # Partition annotations: include only those targeting nodes in this chunk
            chunk_annotations = self._partition_annotations(
                document.annotations, chunk_node_ids, index
            )

            # Build metadata with chunk info
            extra = dict(document.metadata.extra) if document.metadata.extra else {}
            extra["chunk_index"] = ci
            extra["chunk_total"] = total

            meta = DocumentMetadata(
                title=document.metadata.title,
                authors=document.metadata.authors,
                date=document.metadata.date,
                language=document.metadata.language,
                source=document.metadata.source,
                document_type=document.metadata.document_type,
                extra=extra,
            )

            chunks.append(
                ContentDocument(
                    metadata=meta,
                    body=tuple(section_blocks),
                    footnotes=chunk_footnotes,
                    annotations=tuple(chunk_annotations),
                )
            )

        return chunks

    # ── Internal methods ──

    def _split_at_headings(self, document: ContentDocument) -> list[list]:
        """Split body blocks into sections at heading boundaries."""
        from kaos_content.model.blocks import Block

        sections: list[list[Block]] = []
        current: list[Block] = []

        for block in document.body:
            if isinstance(block, Heading) and block.depth <= self.split_depth:
                # Start a new section (save current if non-empty)
                if current:
                    sections.append(current)
                current = [block]
            else:
                current.append(block)

        if current:
            sections.append(current)

        return sections

    def _enforce_max_chars(self, blocks: list) -> list[list]:
        """Split a section to keep each chunk within ``max_chars``.

        The split discipline, in order of preference:

        1. **Between blocks** — if the current chunk plus the next block
           would exceed ``max_chars``, emit the current chunk and start a
           new one. Tables, code blocks, and math blocks are never split
           (they're listed in ``_UNSPLITTABLE``); a single oversized
           unsplittable block becomes its own chunk that goes over.
        2. **Between paragraphs** within a multi-block section — same
           rule, just at paragraph granularity.
        3. **Between sentences** within an oversized Paragraph — when a
           single ``Paragraph`` block exceeds ``max_chars`` (e.g. a 5000-
           char "WHEREAS..." run-on common in legal text), the paragraph
           is segmented at sentence boundaries via
           :func:`kaos_nlp_core.segmentation.segment_sentences`, and the
           sentences are repackaged into multiple ``Paragraph`` blocks
           each ≤ ``max_chars``. Provenance is carried forward to every
           sub-paragraph. Inline formatting marks within the original
           paragraph are flattened to plain text on this path — losing
           formatting is the price of preventing context overruns.
        4. **Within a sentence** — never. A single sentence longer than
           ``max_chars`` is emitted as one over-budget chunk rather than
           cut mid-clause; truncating a legal sentence mid-condition
           would corrupt meaning. Callers wanting hard caps must set a
           larger ``max_chars``.

        Sentence segmentation requires the optional ``[nlp]`` extra. When
        ``kaos_nlp_core`` is not installed, the function falls back to
        emitting oversized paragraphs whole (the legacy behavior) so the
        chunker still works without the extra.
        """
        total_chars = sum(len(extract_text(b)) for b in blocks)
        if total_chars <= self.max_chars:
            return [blocks]

        # First pass: split a too-large *single Paragraph* block into
        # multiple sentence-packed sub-paragraphs so the per-block
        # discipline below has bounded blocks to work with.
        expanded: list = []
        for block in blocks:
            if block.node_type == "paragraph" and len(extract_text(block)) > self.max_chars:
                expanded.extend(self._split_paragraph_at_sentences(block))
            else:
                expanded.append(block)

        result: list[list] = []
        current: list = []
        current_chars = 0

        for block in expanded:
            block_chars = len(extract_text(block))

            # Never split unsplittable blocks (tables, code blocks)
            if block.node_type in _UNSPLITTABLE:
                current.append(block)
                current_chars += block_chars
                continue

            # If adding this block would exceed the limit and we have content,
            # split at this boundary
            if (
                current_chars + block_chars > self.max_chars
                and current
                and block.node_type != "heading"
            ):
                result.append(current)
                current = [block]
                current_chars = block_chars
            else:
                current.append(block)
                current_chars += block_chars

        if current:
            result.append(current)

        return result

    def _split_paragraph_at_sentences(self, paragraph: Any) -> list[Any]:
        """Pack the sentences of an oversized paragraph into ≤``max_chars``
        sub-paragraphs. Returns a list of new ``Paragraph`` blocks; the
        original is not returned. Provenance is copied to every sub-block.

        Falls back to ``[paragraph]`` (legacy whole-paragraph emission)
        when ``kaos_nlp_core`` is not installed — the chunker stays
        functional without the optional ``[nlp]`` extra.
        """
        try:
            from kaos_nlp_core.segmentation import segment_sentences
        except ImportError:
            return [paragraph]

        from kaos_content.model.blocks import Paragraph
        from kaos_content.model.inlines import Text

        text = extract_text(paragraph)
        if not text or len(text) <= self.max_chars:
            return [paragraph]

        # segment_sentences returns spans; we just need the sentence text
        # in order to repack. Spans carry start/end byte offsets which we
        # don't use here because we're rebuilding inline content anyway.
        # On segmentation failure (model load issue, span-decoding bug on
        # adversarial input) emit a single warning per failure with the
        # paragraph length and traceback, then fall back to the unsplit
        # paragraph so chunking is best-effort, never fatal.
        try:
            spans = segment_sentences(text)
            sentences = [text[s.start : s.end].strip() for s in spans]
        except Exception:
            logger.warning(
                "Sentence segmentation failed; emitting paragraph unsplit",
                extra={"paragraph_chars": len(text)},
                exc_info=True,
            )
            return [paragraph]
        sentences = [s for s in sentences if s]
        if not sentences:
            return [paragraph]

        # Pack sentences into sub-paragraphs, never crossing max_chars
        # except for a single sentence that exceeds the cap on its own
        # (which becomes its own over-budget chunk by design).
        sub_paragraphs: list[Any] = []
        current_text = ""
        for sent in sentences:
            if not current_text:
                current_text = sent
                continue
            joiner = " " if not current_text.endswith((" ", "\n")) else ""
            candidate = current_text + joiner + sent
            if len(candidate) <= self.max_chars:
                current_text = candidate
            else:
                sub_paragraphs.append(
                    Paragraph(
                        children=(Text(value=current_text),),
                        provenance=paragraph.provenance,
                        attr=paragraph.attr,
                    )
                )
                current_text = sent
        if current_text:
            sub_paragraphs.append(
                Paragraph(
                    children=(Text(value=current_text),),
                    provenance=paragraph.provenance,
                    attr=paragraph.attr,
                )
            )
        return sub_paragraphs

    def _apply_overlap(self, sections: list[list]) -> list[list]:
        """Repeat trailing paragraphs from previous chunk at start of next."""
        if len(sections) <= 1:
            return sections

        result: list[list] = [sections[0]]
        for i in range(1, len(sections)):
            prev = sections[i - 1]
            # Find trailing paragraphs from previous section
            overlap_blocks = []
            for block in reversed(prev):
                if block.node_type == "paragraph":
                    overlap_blocks.insert(0, block)
                    if len(overlap_blocks) >= self.overlap_paragraphs:
                        break
                else:
                    break
            result.append([*overlap_blocks, *sections[i]])

        return result

    def _collect_node_ids(self, blocks: list) -> set[str]:
        """Collect all node UUIDs for blocks in this chunk."""
        from kaos_content.traversal.visitor import walk

        ids: set[str] = set()
        for block in blocks:
            for node in walk(block):
                ids.add(node.id)
        return ids

    def _partition_footnotes(
        self, blocks: list, all_footnotes: dict[str, tuple]
    ) -> dict[str, tuple]:
        """Include only footnotes referenced by FootnoteRef nodes in this chunk."""
        from kaos_content.model.inlines import FootnoteRef
        from kaos_content.traversal.visitor import walk

        referenced_ids: set[str] = set()
        for block in blocks:
            for node in walk(block):
                if isinstance(node, FootnoteRef):
                    referenced_ids.add(node.identifier)

        return {k: v for k, v in all_footnotes.items() if k in referenced_ids}

    def _partition_annotations(
        self,
        annotations: tuple[Annotation, ...],
        chunk_node_ids: set[str],
        index: NodeIndex,
    ) -> list[Annotation]:
        """Include annotations whose targets reference nodes in this chunk."""
        result: list[Annotation] = []

        for ann in annotations:
            # Include annotation if any target references a node in this chunk
            for target in ann.targets:
                node = index.get(target.node_ref)
                if node is not None and node.id in chunk_node_ids:
                    result.append(ann)
                    break

        return result


def _attach_heading_path(
    chunk: ContentDocument,
    full_document: ContentDocument,
    split_depth: int,
) -> ContentDocument:
    """Compute and attach a heading_path breadcrumb to a chunk.

    The breadcrumb is the list of ancestor heading texts at the chunk's
    entry point — computed by walking ``full_document.body`` in order,
    maintaining a depth-indexed stack of the most recent heading at each
    depth, and recording the stack snapshot when we hit the chunk's
    first block.

    Stored under ``chunk.metadata.extra['heading_path']`` as a list of
    strings, ordered shallowest-to-deepest. Empty list when the chunk
    contains the document's preamble (before any heading) or when the
    document has no headings.

    The split_depth bounds the *recorded* depth — only headings at
    depth ≤ ``split_depth`` are kept on the stack. Deeper headings are
    ignored for breadcrumb purposes (they live *inside* a chunk's
    content rather than naming the chunk).
    """
    if not chunk.body:
        return chunk
    chunk_first_id = chunk.body[0].id

    # Walk the source doc, build the stack, capture when we hit the chunk.
    stack: dict[int, str] = {}
    captured: dict[int, str] | None = None
    for block in full_document.body:
        if block.id == chunk_first_id:
            captured = dict(stack)
            break
        if isinstance(block, Heading) and block.depth <= split_depth:
            # Maintain stack invariant: drop any deeper entries when we
            # see a shallower heading.
            for d in list(stack):
                if d >= block.depth:
                    del stack[d]
            stack[block.depth] = extract_text(block).strip()
    if captured is None:
        # The chunk's entry point block isn't in the source doc body —
        # this happens when overlap_paragraphs duplicated a paragraph
        # from the previous section. Walk back via the chunk's blocks
        # until we find one that *is* in the source doc.
        body_ids = {b.id for b in full_document.body}
        for block in chunk.body:
            if block.id in body_ids:
                # Re-walk to find this block's stack.
                stack = {}
                for src_block in full_document.body:
                    if src_block.id == block.id:
                        captured = dict(stack)
                        break
                    if isinstance(src_block, Heading) and src_block.depth <= split_depth:
                        for d in list(stack):
                            if d >= src_block.depth:
                                del stack[d]
                        stack[src_block.depth] = extract_text(src_block).strip()
                if captured is not None:
                    break
        if captured is None:
            captured = {}

    heading_path = [captured[d] for d in sorted(captured)]
    extra = dict(chunk.metadata.extra) if chunk.metadata.extra else {}
    extra["heading_path"] = heading_path
    new_metadata = DocumentMetadata(
        title=chunk.metadata.title,
        authors=chunk.metadata.authors,
        date=chunk.metadata.date,
        language=chunk.metadata.language,
        source=chunk.metadata.source,
        document_type=chunk.metadata.document_type,
        extra=extra,
    )
    return chunk.model_copy(update={"metadata": new_metadata})
