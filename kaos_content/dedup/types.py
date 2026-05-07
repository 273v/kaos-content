"""Core types for the document dedup pipeline.

All types are frozen dataclasses (``slots=True``) for immutability and
cache-friendliness. The :class:`DedupLevel` ABC is the extension
point — each concrete level (binary hash, text hash, MinHash, etc.)
subclasses it and implements :meth:`find_clusters`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar


@dataclass(frozen=True, slots=True)
class DedupDocument:
    """Input to the dedup pipeline — one document to fingerprint.

    Not every field is populated for every document. Binary-level
    dedup uses ``file_path``; text-level uses ``text``; semantic uses
    ``embedding``. The pipeline reads whichever fields the active
    levels require and skips documents missing those fields.
    """

    doc_id: str
    file_path: Path | None = None
    text: str | None = None
    embedding: Any | None = None  # np.ndarray (d,) — lazy to avoid hard numpy dep
    page_images: tuple[Any, ...] = ()  # tuple[KaosImage, ...] — lazy import
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DedupCluster:
    """A group of documents identified as duplicates by one level.

    Attributes:
        cluster_id: Deterministic identifier (typically the canonical
            document's fingerprint hash).
        canonical_doc_id: Representative document — the first-seen
            member by input order.
        member_doc_ids: All documents in the cluster including the
            canonical (order-preserving).
        level: Name of the dedup level that detected this cluster
            (e.g. ``"binary_hash"``, ``"minhash"``).
        similarity: Mean pairwise similarity within the cluster.
            1.0 for exact-hash clusters.
        fingerprints: Optional per-doc fingerprints for diagnostics.
    """

    cluster_id: str
    canonical_doc_id: str
    member_doc_ids: tuple[str, ...]
    level: str
    similarity: float = 1.0
    fingerprints: dict[str, str] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.member_doc_ids)

    @property
    def duplicate_doc_ids(self) -> tuple[str, ...]:
        """Member IDs excluding the canonical (the ones to drop)."""
        return tuple(d for d in self.member_doc_ids if d != self.canonical_doc_id)


@dataclass(frozen=True, slots=True)
class DedupReport:
    """Complete output of a pipeline run.

    Attributes:
        clusters: All detected duplicate clusters across all levels.
        singletons: Doc IDs that weren't part of any cluster.
        per_level_stats: ``{level_name: {"clusters": N, "docs_deduped": M}}``.
        total_input: Number of documents fed to the pipeline.
        total_unique: Singletons + one canonical per cluster.
    """

    clusters: tuple[DedupCluster, ...]
    singletons: tuple[str, ...]
    per_level_stats: dict[str, dict[str, int]]
    total_input: int
    total_unique: int

    @property
    def total_duplicates(self) -> int:
        return self.total_input - self.total_unique

    @property
    def dedup_rate(self) -> float:
        """Fraction of input that was duplicate (0.0 = no dups, 1.0 = all dups)."""
        if self.total_input == 0:
            return 0.0
        return self.total_duplicates / self.total_input


class DedupLevel(ABC):
    """Abstract base for a single dedup level.

    Each level implements :meth:`find_clusters` which receives a list
    of documents (potentially already filtered by prior levels) and
    returns clusters it detected. The pipeline orchestrator calls
    levels in sequence and removes clustered documents from the
    remaining set.
    """

    name: ClassVar[str]
    """Machine identifier for this level (e.g. ``"binary_hash"``).
    Used in :attr:`DedupCluster.level` and :attr:`DedupReport.per_level_stats`."""

    @abstractmethod
    def find_clusters(
        self,
        documents: list[DedupDocument],
    ) -> list[DedupCluster]:
        """Identify duplicate clusters within ``documents``.

        Args:
            documents: Candidate documents that haven't been clustered
                by a prior level. The level should handle missing
                fields gracefully (e.g. binary-hash skips docs without
                ``file_path``).

        Returns:
            List of :class:`DedupCluster` — one per group of >=2
            duplicate documents. Singletons are NOT returned here;
            the pipeline infers them from the remainder.
        """


__all__ = [
    "DedupCluster",
    "DedupDocument",
    "DedupLevel",
    "DedupReport",
]
