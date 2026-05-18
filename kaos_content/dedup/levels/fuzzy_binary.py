"""Level 2: Fuzzy binary hash — re-saved / minor-edit file detection.

Uses kaos-nlp-core's CTPH (Context-Triggered Piecewise Hash) to detect
files that are byte-similar but not identical. Catches PDFs that were
re-saved, documents re-exported from a different tool, and minor binary
edits (metadata changes, re-stamped timestamps).

Requires the ``[nlp]`` extra (kaos-nlp-core) for the CTPH
implementation. Falls back to a no-op (empty clusters) when not
installed, so callers that only want exact + text hash can skip this
level without a hard dep.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, ClassVar

from kaos_core.logging import get_logger

from kaos_content.dedup.types import DedupCluster, DedupDocument, DedupLevel

logger = get_logger(__name__)


class FuzzyBinaryLevel(DedupLevel):
    """CTPH fuzzy hash on raw file bytes."""

    name: ClassVar[str] = "fuzzy_binary"

    def __init__(
        self,
        *,
        threshold: float = 0.7,
        window_size: int = 64,
        digest_size: int = 8,
        use_piece_similarity: bool = True,
    ) -> None:
        """
        Args:
            threshold: CTPH similarity threshold [0, 1]. Pairs above
                this are considered duplicates. 0.7 is a reasonable
                default for re-saved PDFs; lower catches more but risks
                false positives.
            window_size: Rolling hash window for CTPH boundary detection.
            digest_size: Bytes per piece in the digest.
            use_piece_similarity: If True, uses ``piece_similarity()``
                (more tolerant of insertions/reorderings) instead of
                ``similarity()`` (strict block-level Jaccard).
        """
        self._threshold = threshold
        self._window_size = window_size
        self._digest_size = digest_size
        self._use_piece = use_piece_similarity

    def find_clusters(
        self,
        documents: list[DedupDocument],
    ) -> list[DedupCluster]:
        try:
            from kaos_nlp_core.hashing import CTPH
        except ImportError:
            logger.warning(
                "FuzzyBinaryLevel requires kaos-nlp-core [nlp] extra. Skipping fuzzy binary dedup."
            )
            return []

        hasher = CTPH(
            window_size=self._window_size,
            digest_size=self._digest_size,
        )

        digests: list[tuple[DedupDocument, Any]] = []
        for doc in documents:
            if doc.file_path is None or not doc.file_path.exists():
                continue
            data = doc.file_path.read_bytes()
            digest = hasher.compute(data)
            digests.append((doc, digest))

        if len(digests) < 2:
            return []

        # O(n^2) pairwise — acceptable for <10,000 docs. For larger
        # corpora, CTPH doesn't have an LSH index; use MinHash instead.
        adj: dict[int, set[int]] = defaultdict(set)
        for i in range(len(digests)):
            for j in range(i + 1, len(digests)):
                d_i = digests[i][1]
                d_j = digests[j][1]
                sim = d_i.piece_similarity(d_j) if self._use_piece else d_i.similarity(d_j)
                if sim >= self._threshold:
                    adj[i].add(j)
                    adj[j].add(i)

        return _build_clusters_from_adjacency(digests, adj, level_name=self.name)


def _build_clusters_from_adjacency(
    digests: list[tuple[DedupDocument, Any]],
    adj: dict[int, set[int]],
    *,
    level_name: str,
) -> list[DedupCluster]:
    """Connected-component clustering from an adjacency map."""
    visited: set[int] = set()
    clusters: list[DedupCluster] = []

    for i in range(len(digests)):
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
        members = [digests[idx][0] for idx in component]
        clusters.append(
            DedupCluster(
                cluster_id=f"fuzzy_{members[0].doc_id}",
                canonical_doc_id=members[0].doc_id,
                member_doc_ids=tuple(m.doc_id for m in members),
                level=level_name,
                similarity=0.0,  # could compute mean pairwise but expensive
            )
        )
    return clusters


__all__ = ["FuzzyBinaryLevel"]
