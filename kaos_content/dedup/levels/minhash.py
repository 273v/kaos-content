"""Level 4: Near-duplicate text via MinHash + LSH.

The industry standard for web-scale text dedup (FineWeb, RedPajama,
Common Crawl). Delegates to kaos-nlp-core's Rust ``find_duplicates()``
which runs the entire pipeline — hashing, LSH indexing, candidate
querying, and connected-component clustering — in a single Rust call
with zero Python-loop overhead.

Handles up to ~1M documents in-memory (128 perms x 8 bytes x 1M =
~1 GB for signatures). The Rust implementation is O(n) for insert +
O(n * avg_candidates) for query, with LSH bands/rows auto-tuned from
the target Jaccard threshold.

Requires the ``[nlp]`` extra (kaos-nlp-core).
"""

from __future__ import annotations

from typing import ClassVar

from kaos_core.logging import get_logger

from kaos_content.dedup.types import DedupCluster, DedupDocument, DedupLevel

logger = get_logger(__name__)


class MinHashLevel(DedupLevel):
    """Near-duplicate text detection via MinHash + LSH.

    Delegates the heavy lifting to kaos-nlp-core's Rust
    ``find_duplicates(hasher, documents, shingle_size, threshold)``
    which does hashing + LSH + clustering in one call.
    """

    name: ClassVar[str] = "minhash"

    def __init__(
        self,
        *,
        shingle_size: int = 5,
        num_perms: int = 128,
        threshold: float = 0.8,
        use_tokens: bool = True,
    ) -> None:
        """
        Args:
            shingle_size: N-gram size for shingling. 5 for general text;
                13 for legal docs (high boilerplate reduces false
                positives from shared statutory language).
            num_perms: MinHash permutation count. 128 is standard;
                64 for speed, 256 for precision.
            threshold: Jaccard similarity threshold for "near-duplicate."
                0.8 matches the RedPajama/FineWeb standard. 0.5 for
                loose similarity. 0.95 for near-exact.
            use_tokens: If True, tokenizes on whitespace (token shingles).
                If False, uses character shingles. Token shingles are
                standard for document dedup; character shingles catch
                sub-word edits but produce more false positives.
        """
        self._shingle_size = shingle_size
        self._num_perms = num_perms
        self._threshold = threshold
        self._use_tokens = use_tokens

    def find_clusters(
        self,
        documents: list[DedupDocument],
    ) -> list[DedupCluster]:
        try:
            from kaos_nlp_core.hashing import (  # type: ignore[import-not-found]
                MinHasher,
                find_duplicates,
            )
        except ImportError:
            logger.warning(
                "MinHashLevel requires kaos-nlp-core [nlp] extra. Skipping MinHash dedup."
            )
            return []

        hasher = MinHasher(num_perm=self._num_perms)

        # Build the (doc_id_int, tokens) input that find_duplicates expects.
        # We use sequential integer IDs and maintain a mapping back to
        # DedupDocument.doc_id strings.
        rust_docs: list[tuple[int, list[str]]] = []
        int_to_doc_id: dict[int, str] = {}

        for doc in documents:
            if doc.text is None:
                continue
            text = doc.text.strip()
            if not text:
                continue

            if self._use_tokens:
                tokens = text.lower().split()
                if len(tokens) < self._shingle_size:
                    continue
            else:
                tokens = list(text.lower())
                if len(tokens) < self._shingle_size:
                    continue

            idx = len(rust_docs)
            int_to_doc_id[idx] = doc.doc_id
            rust_docs.append((idx, tokens))

        if len(rust_docs) < 2:
            return []

        # Single Rust call — hashing + LSH + clustering all in Rust.
        groups = find_duplicates(
            hasher,
            rust_docs,
            shingle_size=self._shingle_size,
            threshold=self._threshold,
        )

        clusters: list[DedupCluster] = []
        for group in groups:
            canonical_int = group.canonical_id
            canonical_doc_id = int_to_doc_id.get(canonical_int, str(canonical_int))

            all_ids = [canonical_doc_id]
            for dup_int, _sim in group.duplicates:
                dup_doc_id = int_to_doc_id.get(dup_int, str(dup_int))
                all_ids.append(dup_doc_id)

            if len(all_ids) < 2:
                continue

            clusters.append(
                DedupCluster(
                    cluster_id=f"minhash_{canonical_doc_id}",
                    canonical_doc_id=canonical_doc_id,
                    member_doc_ids=tuple(all_ids),
                    level=self.name,
                )
            )
        return clusters


__all__ = ["MinHashLevel"]
