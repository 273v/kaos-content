"""Level 1: Exact binary hash — byte-identical file detection.

The cheapest level. Reads each file, computes a cryptographic hash,
and groups files with identical hashes. O(n) in file count, I/O-bound
on disk read.

Catches: byte-identical copies across directories.
Misses: re-saved PDFs, format conversions, any content edit.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import ClassVar, Literal

from kaos_content.dedup.types import DedupCluster, DedupDocument, DedupLevel

_BLOCK_SIZE = 65536


class BinaryHashLevel(DedupLevel):
    """Exact binary hash on raw file bytes."""

    name: ClassVar[str] = "binary_hash"

    def __init__(
        self,
        *,
        algorithm: Literal["sha256", "blake2b", "md5"] = "sha256",
    ) -> None:
        self._algorithm = algorithm

    def find_clusters(
        self,
        documents: list[DedupDocument],
    ) -> list[DedupCluster]:
        groups: dict[str, list[DedupDocument]] = defaultdict(list)
        fingerprints: dict[str, str] = {}

        for doc in documents:
            if doc.file_path is None or not doc.file_path.exists():
                continue
            h = self._hash_file(doc)
            groups[h].append(doc)
            fingerprints[doc.doc_id] = h

        clusters: list[DedupCluster] = []
        for h, members in groups.items():
            if len(members) < 2:
                continue
            clusters.append(
                DedupCluster(
                    cluster_id=f"binary_{h[:16]}",
                    canonical_doc_id=members[0].doc_id,
                    member_doc_ids=tuple(m.doc_id for m in members),
                    level=self.name,
                    similarity=1.0,
                    fingerprints={m.doc_id: h for m in members},
                )
            )
        return clusters

    def _hash_file(self, doc: DedupDocument) -> str:
        # `is_applicable()` already filtered out docs without file_path,
        # so this is a type-narrower for the checker, not user input
        # validation. Bandit flags any `assert` (B101) since asserts are
        # stripped under `python -O`; here we accept that — losing the
        # narrowing in optimized builds at worst raises AttributeError
        # one line later, which is the same failure mode.
        assert doc.file_path is not None  # nosec B101
        h = hashlib.new(self._algorithm)
        with doc.file_path.open("rb") as f:
            while True:
                chunk = f.read(_BLOCK_SIZE)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()


__all__ = ["BinaryHashLevel"]
