"""Pipeline orchestrator — chains dedup levels with short-circuit.

The pipeline runs each level in sequence. Documents clustered at
level N are removed from the candidate set for level N+1 (they're
already accounted for). Singletons after all levels are the truly
unique documents.

The short-circuit is optional: ``short_circuit=False`` runs every
level on every document (useful for auditing overlap between levels).
"""

from __future__ import annotations

from dataclasses import dataclass

from kaos_content.dedup.types import (
    DedupCluster,
    DedupDocument,
    DedupLevel,
    DedupReport,
)


@dataclass(frozen=True, slots=True)
class DedupPipelineConfig:
    """Immutable pipeline configuration.

    Attributes:
        levels: Ordered sequence of dedup levels to run.
        short_circuit: When True, documents clustered at level N
            are excluded from level N+1's input. When False, all
            documents are fed to every level (union of all clusters).
    """

    levels: tuple[DedupLevel, ...] = ()
    short_circuit: bool = True


class DedupPipeline:
    """Orchestrates multi-level dedup on a document set.

    Usage::

        from kaos_content.dedup import DedupPipeline, DedupDocument
        from kaos_content.dedup.levels import BinaryHashLevel, TextHashLevel

        config = DedupPipelineConfig(levels=(
            BinaryHashLevel(),
            TextHashLevel(),
        ))
        pipeline = DedupPipeline(config)
        report = pipeline.run([
            DedupDocument(doc_id="d1", file_path=Path("a.pdf"), text="..."),
            DedupDocument(doc_id="d2", file_path=Path("b.pdf"), text="..."),
        ])
    """

    def __init__(self, config: DedupPipelineConfig) -> None:
        self._config = config

    def run(self, documents: list[DedupDocument] | tuple[DedupDocument, ...]) -> DedupReport:
        """Execute the full pipeline and return a :class:`DedupReport`."""
        all_docs = list(documents)
        total_input = len(all_docs)

        all_clusters: list[DedupCluster] = []
        per_level_stats: dict[str, dict[str, int]] = {}
        clustered_ids: set[str] = set()

        for level in self._config.levels:
            if self._config.short_circuit:
                candidates = [d for d in all_docs if d.doc_id not in clustered_ids]
            else:
                candidates = all_docs

            if not candidates:
                per_level_stats[level.name] = {"clusters": 0, "docs_deduped": 0}
                continue

            clusters = level.find_clusters(candidates)
            docs_deduped = 0
            for cluster in clusters:
                all_clusters.append(cluster)
                for doc_id in cluster.duplicate_doc_ids:
                    clustered_ids.add(doc_id)
                    docs_deduped += 1
                clustered_ids.add(cluster.canonical_doc_id)

            per_level_stats[level.name] = {
                "clusters": len(clusters),
                "docs_deduped": docs_deduped,
            }

        # Singletons: docs not in ANY cluster (not even as canonical).
        all_clustered = set()
        for cluster in all_clusters:
            all_clustered.update(cluster.member_doc_ids)
        singletons = tuple(d.doc_id for d in all_docs if d.doc_id not in all_clustered)

        # Unique = singletons + one canonical per cluster.
        canonical_ids = {c.canonical_doc_id for c in all_clusters}
        total_unique = len(singletons) + len(canonical_ids)

        return DedupReport(
            clusters=tuple(all_clusters),
            singletons=singletons,
            per_level_stats=per_level_stats,
            total_input=total_input,
            total_unique=total_unique,
        )


__all__ = ["DedupPipeline", "DedupPipelineConfig"]
