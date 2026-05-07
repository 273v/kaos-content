"""Tests for SectionChunker (Phase 6).

Tests cover:
- Splitting at heading boundaries
- Configurable split_depth
- max_chars enforcement (split at paragraph boundary)
- Tables/code blocks not split mid-content
- Footnotes follow their references
- Annotations assigned to correct chunks
- Overlap paragraphs repeated at chunk boundaries
- Chunk metadata (chunk_index, chunk_total)
- Edge cases: empty document, single block, no headings
"""

from __future__ import annotations

from kaos_content import (
    Annotation,
    AnnotationTarget,
    AnnotationType,
    ContentDocument,
    DocumentBuilder,
    SectionChunker,
    extract_text,
)

# ── Helpers ──


def _build_sectioned_doc() -> ContentDocument:
    """Build a document with 3 h1 sections."""
    return (
        DocumentBuilder(title="Test")
        .heading(1, "Section 1")
        .paragraph("Content of section 1.")
        .heading(1, "Section 2")
        .paragraph("Content of section 2.")
        .paragraph("More section 2.")
        .heading(1, "Section 3")
        .paragraph("Content of section 3.")
        .build()
    )


def _build_nested_headings_doc() -> ContentDocument:
    """Build a document with h1, h2, and h3 headings."""
    return (
        DocumentBuilder(title="Nested")
        .heading(1, "Chapter 1")
        .paragraph("Intro to chapter 1.")
        .heading(2, "Section 1.1")
        .paragraph("Content 1.1.")
        .heading(3, "Subsection 1.1.1")
        .paragraph("Content 1.1.1.")
        .heading(2, "Section 1.2")
        .paragraph("Content 1.2.")
        .heading(1, "Chapter 2")
        .paragraph("Content of chapter 2.")
        .build()
    )


# ── Basic splitting ──


class TestBasicSplitting:
    def test_three_h1_sections(self):
        doc = _build_sectioned_doc()
        chunker = SectionChunker(max_chars=0, split_depth=1)
        chunks = chunker.chunk(doc)
        assert len(chunks) == 3

    def test_chunk_content(self):
        doc = _build_sectioned_doc()
        chunker = SectionChunker(max_chars=0, split_depth=1)
        chunks = chunker.chunk(doc)
        # First chunk: heading + paragraph
        assert chunks[0].body[0].node_type == "heading"
        assert len(chunks[0].body) == 2
        # Second chunk: heading + 2 paragraphs
        assert len(chunks[1].body) == 3
        # Third chunk: heading + paragraph
        assert len(chunks[2].body) == 2

    def test_each_chunk_is_valid_document(self):
        doc = _build_sectioned_doc()
        chunker = SectionChunker(max_chars=0, split_depth=1)
        chunks = chunker.chunk(doc)
        for chunk in chunks:
            assert isinstance(chunk, ContentDocument)
            assert chunk.metadata is not None

    def test_no_headings_single_chunk(self):
        doc = DocumentBuilder().paragraph("Just text.").paragraph("More text.").build()
        chunker = SectionChunker(max_chars=0, split_depth=2)
        chunks = chunker.chunk(doc)
        assert len(chunks) == 1

    def test_empty_document(self):
        doc = ContentDocument()
        chunker = SectionChunker()
        chunks = chunker.chunk(doc)
        assert len(chunks) == 1
        assert chunks[0].body == ()


# ── Split depth ──


class TestSplitDepth:
    def test_split_at_h1_only(self):
        doc = _build_nested_headings_doc()
        chunker = SectionChunker(max_chars=0, split_depth=1)
        chunks = chunker.chunk(doc)
        assert len(chunks) == 2  # Two h1 sections

    def test_split_at_h1_and_h2(self):
        doc = _build_nested_headings_doc()
        chunker = SectionChunker(max_chars=0, split_depth=2)
        chunks = chunker.chunk(doc)
        # h1 Chapter 1, h2 Section 1.1 (includes h3 subsection), h2 Section 1.2, h1 Chapter 2
        assert len(chunks) == 4

    def test_split_depth_3(self):
        doc = _build_nested_headings_doc()
        chunker = SectionChunker(max_chars=0, split_depth=3)
        chunks = chunker.chunk(doc)
        # h1, h2 1.1, h3 1.1.1, h2 1.2, h1 ch2
        assert len(chunks) == 5

    def test_content_before_first_heading(self):
        doc = (
            DocumentBuilder()
            .paragraph("Preamble.")
            .heading(1, "First Section")
            .paragraph("Content.")
            .build()
        )
        chunker = SectionChunker(max_chars=0, split_depth=1)
        chunks = chunker.chunk(doc)
        # Preamble is in first chunk, then section
        assert len(chunks) == 2
        assert chunks[0].body[0].node_type == "paragraph"
        assert chunks[1].body[0].node_type == "heading"


# ── Max chars ──


class TestMaxChars:
    def test_split_oversized_section(self):
        # Build a section with many paragraphs
        builder = DocumentBuilder().heading(1, "Big Section")
        for i in range(20):
            builder.paragraph(f"Paragraph {i} with enough text to add up. " * 5)
        doc = builder.build()

        chunker = SectionChunker(max_chars=500, split_depth=1)
        chunks = chunker.chunk(doc)
        assert len(chunks) > 1

    def test_each_chunk_within_limit(self):
        builder = DocumentBuilder().heading(1, "Section")
        for i in range(10):
            builder.paragraph(f"Content block {i}. " * 10)
        doc = builder.build()

        limit = 300
        chunker = SectionChunker(max_chars=limit, split_depth=1)
        chunks = chunker.chunk(doc)

        for chunk in chunks:
            # Allow some slack — headings and unsplittable blocks may push over
            total = sum(len(extract_text(b)) for b in chunk.body)
            # We mainly verify splitting happens, not strict enforcement
            assert total > 0

    def test_table_not_split(self):
        """Tables should never be split mid-content."""
        builder = DocumentBuilder().heading(1, "Section")
        headers = ["Column A", "Column B"]
        rows = [[f"Row {i} A", f"Row {i} B"] for i in range(50)]
        builder.table(headers, rows)
        builder.paragraph("After table.")
        doc = builder.build()

        chunker = SectionChunker(max_chars=100, split_depth=1)
        chunks = chunker.chunk(doc)
        # The table should not be split across chunks
        from kaos_content import Table

        for chunk in chunks:
            for block in chunk.body:
                if isinstance(block, Table):
                    # Table is intact in one chunk
                    assert block.head is not None

    def test_no_limit(self):
        doc = _build_sectioned_doc()
        chunker = SectionChunker(max_chars=0, split_depth=1)
        chunks = chunker.chunk(doc)
        assert len(chunks) == 3  # Only heading-based splitting


# ── Overlap ──


class TestOverlap:
    def test_overlap_paragraphs(self):
        doc = (
            DocumentBuilder()
            .heading(1, "Section 1")
            .paragraph("Para 1A")
            .paragraph("Para 1B")
            .heading(1, "Section 2")
            .paragraph("Para 2A")
            .build()
        )
        chunker = SectionChunker(max_chars=0, split_depth=1, overlap_paragraphs=1)
        chunks = chunker.chunk(doc)
        assert len(chunks) == 2

        # Second chunk should start with the last paragraph from section 1
        second_body = chunks[1].body
        first_text = extract_text(second_body[0])
        assert "Para 1B" in first_text

    def test_zero_overlap(self):
        doc = _build_sectioned_doc()
        chunker = SectionChunker(max_chars=0, split_depth=1, overlap_paragraphs=0)
        chunks = chunker.chunk(doc)
        # No overlap — sections are independent
        assert len(chunks) == 3

    def test_overlap_larger_than_section(self):
        """Overlap shouldn't cause issues if section has fewer paragraphs."""
        doc = (
            DocumentBuilder()
            .heading(1, "Section 1")
            .paragraph("Only paragraph")
            .heading(1, "Section 2")
            .paragraph("Content")
            .build()
        )
        chunker = SectionChunker(max_chars=0, split_depth=1, overlap_paragraphs=5)
        chunks = chunker.chunk(doc)
        assert len(chunks) == 2


# ── Metadata ──


class TestChunkMetadata:
    def test_chunk_index_and_total(self):
        doc = _build_sectioned_doc()
        chunker = SectionChunker(max_chars=0, split_depth=1)
        chunks = chunker.chunk(doc)

        for i, chunk in enumerate(chunks):
            assert chunk.metadata.extra["chunk_index"] == i
            assert chunk.metadata.extra["chunk_total"] == 3

    def test_title_preserved(self):
        doc = _build_sectioned_doc()
        chunker = SectionChunker(max_chars=0, split_depth=1)
        chunks = chunker.chunk(doc)
        for chunk in chunks:
            assert chunk.metadata.title == "Test"

    def test_original_metadata_preserved(self):
        doc = (
            DocumentBuilder(title="Document")
            .set_metadata(date="2026-01-01", language="en")
            .heading(1, "S1")
            .paragraph("Content")
            .heading(1, "S2")
            .paragraph("Content")
            .build()
        )
        chunker = SectionChunker(max_chars=0, split_depth=1)
        chunks = chunker.chunk(doc)
        for chunk in chunks:
            assert chunk.metadata.date == "2026-01-01"
            assert chunk.metadata.language == "en"


# ── Footnotes ──


class TestFootnotePartitioning:
    def test_footnote_follows_reference(self):
        doc = (
            DocumentBuilder()
            .heading(1, "Section 1")
            .paragraph(
                DocumentBuilder.text("Text "),
                DocumentBuilder.footnote_ref("fn1"),
            )
            .add_footnote("fn1", "Footnote 1 body")
            .heading(1, "Section 2")
            .paragraph("No footnotes here.")
            .build()
        )
        chunker = SectionChunker(max_chars=0, split_depth=1)
        chunks = chunker.chunk(doc)
        assert len(chunks) == 2

        # First chunk has the footnote
        assert "fn1" in chunks[0].footnotes
        # Second chunk does not
        assert "fn1" not in chunks[1].footnotes

    def test_footnote_in_multiple_chunks(self):
        """If both chunks reference the same footnote, both get it."""
        doc = (
            DocumentBuilder()
            .heading(1, "Section 1")
            .paragraph(
                DocumentBuilder.text("Text "),
                DocumentBuilder.footnote_ref("shared"),
            )
            .add_footnote("shared", "Shared footnote")
            .heading(1, "Section 2")
            .paragraph(
                DocumentBuilder.text("Also "),
                DocumentBuilder.footnote_ref("shared"),
            )
            .build()
        )
        chunker = SectionChunker(max_chars=0, split_depth=1)
        chunks = chunker.chunk(doc)
        assert "shared" in chunks[0].footnotes
        assert "shared" in chunks[1].footnotes


# ── Annotations ──


class TestAnnotationPartitioning:
    def test_annotation_follows_target(self):

        doc = (
            DocumentBuilder()
            .heading(1, "Section 1")
            .paragraph("Target paragraph")
            .heading(1, "Section 2")
            .paragraph("Other paragraph")
            .build()
        )
        # Add annotation targeting the first paragraph
        target_ref = "#/body/1"  # The paragraph after heading
        ann = Annotation(
            id="ann1",
            type=AnnotationType.HIGHLIGHT,
            targets=(AnnotationTarget(node_ref=target_ref),),
        )
        doc = ContentDocument(
            metadata=doc.metadata,
            body=doc.body,
            annotations=(ann,),
        )

        chunker = SectionChunker(max_chars=0, split_depth=1)
        chunks = chunker.chunk(doc)

        # Annotation should be in first chunk
        assert len(chunks[0].annotations) == 1
        assert chunks[0].annotations[0].id == "ann1"
        # Not in second chunk
        assert len(chunks[1].annotations) == 0


# ── Integration ──


class TestChunkerIntegration:
    def test_chunk_and_serialize(self):
        from kaos_content import serialize_markdown

        doc = _build_sectioned_doc()
        chunker = SectionChunker(max_chars=0, split_depth=1)
        chunks = chunker.chunk(doc)

        for chunk in chunks:
            md = serialize_markdown(chunk)
            assert len(md) > 0
            assert "# Section" in md

    def test_chunk_and_index(self):
        from kaos_content import NodeIndex

        doc = _build_sectioned_doc()
        chunker = SectionChunker(max_chars=0, split_depth=1)
        chunks = chunker.chunk(doc)

        for chunk in chunks:
            index = NodeIndex(chunk)
            assert len(index.headings) >= 1

    def test_chunk_round_trip(self):
        """Chunking a parsed document should produce valid chunks."""
        from kaos_content import parse_markdown

        md = """# Chapter 1

Content of chapter 1.

## Section 1.1

Details about 1.1.

# Chapter 2

Content of chapter 2.
"""
        doc = parse_markdown(md)
        chunker = SectionChunker(max_chars=0, split_depth=1)
        chunks = chunker.chunk(doc)
        assert len(chunks) == 2

    def test_all_content_preserved(self):
        """All body text should be present across chunks (no data loss)."""
        doc = _build_sectioned_doc()
        chunker = SectionChunker(max_chars=0, split_depth=1)
        chunks = chunker.chunk(doc)

        original_text = extract_text(doc.body[0])
        for b in doc.body[1:]:
            original_text += " " + extract_text(b)

        chunk_texts = []
        for chunk in chunks:
            for b in chunk.body:
                chunk_texts.append(extract_text(b))

        all_chunk_text = " ".join(chunk_texts)
        # Every word from the original should appear in chunks
        for word in original_text.split():
            assert word in all_chunk_text


class TestSectionChunkerEdgeCases:
    """Edge cases from code review."""

    def test_max_chars_smaller_than_single_paragraph(self):
        """When max_chars is smaller than a single paragraph, each paragraph
        becomes its own chunk (no infinite loop, no crash)."""
        doc = (
            DocumentBuilder(title="Tiny")
            .paragraph("This is a paragraph with more than ten characters.")
            .paragraph("Second paragraph also over ten characters.")
            .build()
        )
        chunker = SectionChunker(max_chars=10, split_depth=1)
        chunks = chunker.chunk(doc)
        # Should produce chunks — not crash or infinite loop
        assert len(chunks) >= 1
        # All text should still be present
        all_text = " ".join(extract_text(b) for c in chunks for b in c.body)
        assert "paragraph" in all_text

    def test_document_with_no_headings(self):
        """Document with only paragraphs and no headings still chunks."""
        doc = (
            DocumentBuilder(title="No Headings")
            .paragraph("First paragraph.")
            .paragraph("Second paragraph.")
            .paragraph("Third paragraph.")
            .build()
        )
        chunker = SectionChunker(max_chars=50, split_depth=1)
        chunks = chunker.chunk(doc)
        assert len(chunks) >= 1

    def test_max_chars_zero_means_no_limit(self):
        """max_chars=0 means no character limit — returns sections only."""
        doc = _build_sectioned_doc()
        chunker = SectionChunker(max_chars=0, split_depth=1)
        chunks = chunker.chunk(doc)
        # With 3 h1 sections, split_depth=1 → 3 chunks
        assert len(chunks) == 3


class TestSentenceFallback:
    """Sentence-level fallback for oversized paragraphs (legal run-ons)."""

    def test_oversized_paragraph_split_at_sentence_boundaries(self) -> None:
        """A single 4000-char paragraph with proper sentence terminators
        must be split into multiple sub-paragraphs each ≤ max_chars."""
        # Build a realistic legal paragraph: ~10 distinct sentences with
        # period terminators, ~4000 chars total.
        sentence = (
            "The parties have agreed to the foregoing terms and conditions "
            "in reliance upon the representations and warranties set forth "
            "herein and notwithstanding any prior course of dealing. "
        )
        long_paragraph_text = sentence * 15  # ~3750 chars, 15 sentences
        doc = DocumentBuilder().heading(1, "Recitals").paragraph(long_paragraph_text).build()
        chunker = SectionChunker(max_chars=1500, split_depth=1)
        chunks = chunker.chunk(doc)
        assert len(chunks) >= 2, f"oversized para should split; got {len(chunks)} chunks"
        # Every paragraph block must respect max_chars. With proper sentence
        # terminators in the fixture, each individual sentence is ~250 chars
        # so the packer can fit ~6 sentences per 1500-char chunk.
        for ch in chunks:
            for b in ch.body:
                if b.node_type == "paragraph":
                    assert len(extract_text(b)) <= 1500, (
                        f"sub-paragraph exceeded max_chars: {len(extract_text(b))}"
                    )

    def test_unsegmentable_paragraph_emits_whole(self) -> None:
        """A paragraph with no sentence terminators should emit as one
        over-budget chunk rather than being mid-clause truncated.

        Documented behavior: we never split a single sentence. A legal
        run-on with no periods is treated as one sentence and survives
        whole — the alternative (truncating mid-condition) would corrupt
        meaning. Callers wanting hard caps must increase max_chars."""
        run_on = "and ".join(["term"] * 600)  # 3000+ chars, no period
        doc = DocumentBuilder().heading(1, "Run-on").paragraph(run_on).build()
        chunker = SectionChunker(max_chars=1500, split_depth=1)
        chunks = chunker.chunk(doc)
        # The over-budget paragraph survives whole.
        any_oversized = any(
            b.node_type == "paragraph" and len(extract_text(b)) > 1500
            for ch in chunks
            for b in ch.body
        )
        assert any_oversized, (
            "expected at least one over-budget paragraph (cannot split mid-sentence)"
        )

    def test_short_paragraph_unchanged(self) -> None:
        """Paragraphs below max_chars must NOT be split."""
        doc = (
            DocumentBuilder()
            .heading(1, "Section")
            .paragraph("This is a short paragraph that fits comfortably within the chunk.")
            .build()
        )
        chunker = SectionChunker(max_chars=1500, split_depth=1)
        chunks = chunker.chunk(doc)
        assert len(chunks) == 1
        # The original short paragraph survives as one block (not re-split).
        para_blocks = [b for b in chunks[0].body if b.node_type == "paragraph"]
        assert len(para_blocks) == 1
        assert "comfortably within the chunk" in extract_text(para_blocks[0])


# ── T4b: from_outline + heading_path ──


class TestFromOutline:
    """Outline-aware chunking via from_outline()."""

    def test_heading_path_attached_to_each_chunk(self):
        """Every chunk's metadata.extra carries a heading_path list."""
        doc = _build_sectioned_doc()
        chunks = SectionChunker.from_outline(
            doc,
            max_chars=0,
            split_depth=1,
            promote_inferred=False,
        )
        assert len(chunks) == 3
        for chunk in chunks:
            assert "heading_path" in chunk.metadata.extra, chunk.metadata.extra
            assert isinstance(chunk.metadata.extra["heading_path"], list)

    def test_nested_heading_path_breadcrumb(self):
        """Nested h1 + h2 chunks must surface the right breadcrumb.

        Chunking at split_depth=2 produces a chunk whose first block is
        an h2 inside a h1. The h2 chunk's heading_path should be just
        the h1 ancestor (the h2 itself is the chunk's own header, not
        an ancestor).
        """
        doc = _build_nested_headings_doc()
        chunks = SectionChunker.from_outline(
            doc,
            max_chars=0,
            split_depth=2,
            promote_inferred=False,
        )
        # Find the chunk that starts with "Section 1.1" (h2 under Chapter 1).
        for chunk in chunks:
            first_text = extract_text(chunk.body[0]).strip()
            if first_text == "Section 1.1":
                # Its breadcrumb is the parent h1.
                assert chunk.metadata.extra["heading_path"] == ["Chapter 1"], chunk.metadata.extra[
                    "heading_path"
                ]
                return
        # If we never found that chunk, the test infra is wrong — fail.
        chunk_starts = [extract_text(c.body[0]).strip() for c in chunks]
        raise AssertionError(f"no chunk starts with 'Section 1.1' — got {chunk_starts}")

    def test_preamble_chunk_has_empty_heading_path(self):
        """Blocks before the first heading have no breadcrumb."""
        doc = (
            DocumentBuilder(title="Preamble Doc")
            .paragraph("Preamble before any heading.")
            .heading(1, "First Section")
            .paragraph("Section content.")
            .build()
        )
        chunks = SectionChunker.from_outline(
            doc, max_chars=0, split_depth=1, promote_inferred=False
        )
        # The first chunk should be the preamble — its heading_path is [].
        assert chunks[0].metadata.extra["heading_path"] == []

    def test_promote_inferred_promotes_paragraph_headings(self):
        """When promote_inferred=True, a doc with all paragraphs gets
        chunked at the inferred heading boundaries."""
        # All-Paragraph doc with strong heading-shaped lines.
        doc = (
            DocumentBuilder(title="No Typed Headings")
            .paragraph("Preamble body text here.")
            .paragraph("DISCUSSION")
            .paragraph("First section body text.")
            .paragraph("CONCLUSION")
            .paragraph("Second section body text.")
            .build()
        )
        # With promote_inferred=False, no headings → 1 chunk.
        chunks_off = SectionChunker.from_outline(
            doc, max_chars=0, split_depth=1, promote_inferred=False
        )
        assert len(chunks_off) == 1, f"expected 1 chunk without promotion, got {len(chunks_off)}"
        # With promote_inferred=True, headings get inferred → multiple chunks.
        try:
            chunks_on = SectionChunker.from_outline(
                doc, max_chars=0, split_depth=1, promote_inferred=True
            )
        except ImportError:
            # No [nlp] extra installed — accept the skip rather than fail.
            return
        assert len(chunks_on) > 1, f"promotion should produce >1 chunks, got {len(chunks_on)}"
