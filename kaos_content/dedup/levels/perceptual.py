"""Level 6 (bonus): Perceptual page hash — scanned/OCR'd document dedup.

For scanned PDFs where text extraction failed or OCR quality is poor,
comparing extracted text is unreliable. This level renders page
images and computes perceptual hashes (dHash by default) to detect
visually similar pages across documents.

Strategy: hash the first N pages of each document, group documents
whose page hashes overlap (Hamming distance below threshold).

Requires the ``[dedup-perceptual]`` extra (imagehash, BSD-2) and the
``[images]`` extra (Pillow) which is already a kaos-content dep.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, ClassVar, Literal

from kaos_core.logging import get_logger

from kaos_content.dedup.types import DedupCluster, DedupDocument, DedupLevel

logger = get_logger(__name__)


class PerceptualHashLevel(DedupLevel):
    """Perceptual hash on rendered page images."""

    name: ClassVar[str] = "perceptual"

    def __init__(
        self,
        *,
        algorithm: Literal["dhash", "phash", "ahash"] = "dhash",
        hash_size: int = 16,
        max_hamming_distance: int = 5,
        pages_to_hash: int = 3,
        min_page_overlap: int = 1,
    ) -> None:
        """
        Args:
            algorithm: Perceptual hash algorithm. "dhash" (gradient-
                based, fast) or "phash" (DCT, more robust to
                compression).
            hash_size: Bits per side. 16 → 256-bit hash.
            max_hamming_distance: Max Hamming distance between two
                page hashes to consider them "same page." 5 is
                conservative for clean scans; 10 for noisy OCR.
            pages_to_hash: Number of leading pages to hash per
                document (cover page + first few content pages).
            min_page_overlap: Minimum number of matching pages
                required to cluster two documents.
        """
        self._algorithm = algorithm
        self._hash_size = hash_size
        self._max_hamming = max_hamming_distance
        self._pages_to_hash = pages_to_hash
        self._min_overlap = min_page_overlap

    def find_clusters(
        self,
        documents: list[DedupDocument],
    ) -> list[DedupCluster]:
        try:
            import imagehash  # type: ignore[import-not-found]
        except ImportError:
            logger.warning(
                "PerceptualHashLevel requires imagehash [dedup-perceptual] extra. "
                "Skipping perceptual hash dedup."
            )
            return []

        hash_fn = {
            "dhash": imagehash.dhash,
            "phash": imagehash.phash,
            "ahash": imagehash.average_hash,
        }.get(self._algorithm, imagehash.dhash)

        # Hash leading pages of each document.
        doc_hashes: list[tuple[DedupDocument, list[Any]]] = []
        for doc in documents:
            if not doc.page_images:
                continue
            pages = doc.page_images[: self._pages_to_hash]
            hashes = []
            for img in pages:
                # `page_images` is documented as tuple[KaosImage, ...]; KaosImage
                # exposes the underlying PIL Image via the public `pil` property.
                # `getattr` (not direct attribute access) keeps the level resilient
                # against caller types that might predate the [images] extra.
                pil = getattr(img, "pil", None)
                if pil is None:
                    continue
                try:
                    h = hash_fn(pil, hash_size=self._hash_size)
                except Exception:  # nosec B112
                    # imagehash + PIL can throw on corrupt scans, mode
                    # mismatches, or backend-specific errors. A single
                    # bad page should not abort dedup of the remaining
                    # pages or the rest of the corpus — we drop the
                    # page and continue. The doc is dropped below if
                    # zero pages succeeded (`if hashes:`).
                    continue
                hashes.append(h)
            if hashes:
                doc_hashes.append((doc, hashes))

        if len(doc_hashes) < 2:
            return []

        # O(n^2) pairwise — acceptable for page-hash level since
        # it's only used on scanned-doc corpora (typically <10K).
        adj: dict[int, set[int]] = defaultdict(set)
        for i in range(len(doc_hashes)):
            for j in range(i + 1, len(doc_hashes)):
                overlap = _count_page_overlap(
                    doc_hashes[i][1],
                    doc_hashes[j][1],
                    self._max_hamming,
                )
                if overlap >= self._min_overlap:
                    adj[i].add(j)
                    adj[j].add(i)

        return _build_clusters(doc_hashes, adj, level_name=self.name)


def _count_page_overlap(
    hashes_a: list[Any],
    hashes_b: list[Any],
    max_hamming: int,
) -> int:
    """Count how many pages in A have a near-match in B."""
    matches = 0
    for ha in hashes_a:
        for hb in hashes_b:
            if ha - hb <= max_hamming:
                matches += 1
                break
    return matches


def _build_clusters(
    doc_hashes: list[tuple[DedupDocument, list[Any]]],
    adj: dict[int, set[int]],
    *,
    level_name: str,
) -> list[DedupCluster]:
    """Connected-component clustering from adjacency."""
    visited: set[int] = set()
    clusters: list[DedupCluster] = []

    for i in range(len(doc_hashes)):
        if i in visited or i not in adj:
            continue
        component: list[int] = []
        stack = [i]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            component.append(node)
            for neighbor in adj.get(node, ()):
                if neighbor not in visited:
                    stack.append(neighbor)

        if len(component) < 2:
            continue
        component.sort()
        members = [doc_hashes[idx][0] for idx in component]
        clusters.append(
            DedupCluster(
                cluster_id=f"perceptual_{members[0].doc_id}",
                canonical_doc_id=members[0].doc_id,
                member_doc_ids=tuple(m.doc_id for m in members),
                level=level_name,
            )
        )
    return clusters


__all__ = ["PerceptualHashLevel"]
