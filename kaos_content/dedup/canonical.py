"""Canonical-record (survivor) selection for dedup clusters.

Every :class:`~kaos_content.dedup.types.DedupLevel` reports a cluster's
canonical as the *first-seen* member by input order — fine as a default,
but document/legal dedup usually wants the **most complete** record kept,
not the earliest. This module re-picks the survivor of each cluster
*after* the pipeline runs, leaving cluster membership untouched.

Strategies (``CanonicalStrategy``):

- ``"first"`` — first member by input order (the pipeline default; a no-op
  here, so :func:`recanonicalize` returns the report unchanged).
- ``"longest"`` / ``"shortest"`` — by ``len(text)``; ties resolve to the
  earliest member (deterministic).
- ``"medoid"`` — the member nearest the cluster centroid in embedding
  space (the *most representative* record). Requires every member to
  carry an ``embedding``. For L2-normalized embeddings (what
  ``EmbeddingModel.embed`` produces) centroid-nearest equals the pairwise
  medoid (``argmax`` of mean cosine to the other members) and is computed
  in O(n) per cluster.
- a ``Callable[[list[DedupDocument]], str]`` — return the chosen member's
  ``doc_id``; raises if the return value is not a member.

The survivorship choice is independent of how the cluster was formed, so
this composes with any preset/level combination.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Literal

from kaos_content.dedup.types import DedupDocument, DedupReport

CanonicalStrategy = (
    Literal["first", "longest", "shortest", "medoid"] | Callable[[list[DedupDocument]], str]
)
"""How to pick a cluster's surviving (canonical) record.

A string names a built-in strategy; a callable receives the cluster's
members (in input order) and returns the ``doc_id`` to keep."""


def _medoid_doc_id(members: list[DedupDocument]) -> str:
    """Return the ``doc_id`` of the member nearest the cluster centroid.

    Centroid-nearest over L2-normalized rows == the cosine medoid (argmax
    mean cosine to the others). Ties resolve to the lowest index
    (``argmax`` returns the first maximum), i.e. the earliest member.
    """
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - exercised only without numpy
        msg = (
            "canonical='medoid' requires numpy. "
            "Fix: pip install kaos-content[clustering] (pulls numpy) or "
            "pip install numpy>=2.0. Alternative: use canonical='longest' "
            "for a numpy-free 'most complete record' survivor."
        )
        raise ImportError(msg) from exc

    rows: list[np.ndarray] = []
    for m in members:
        if m.embedding is None:
            msg = (
                f"canonical='medoid' needs an embedding on every cluster member, "
                f"but document {m.doc_id!r} has embedding=None. "
                "Fix: populate DedupDocument.embedding (e.g. from "
                "EmbeddingModel.embed) before dedup, or use canonical='longest'."
            )
            raise ValueError(msg)
        rows.append(np.asarray(m.embedding, dtype=np.float64).reshape(-1))

    matrix = np.vstack(rows)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    unit = matrix / np.where(norms == 0.0, 1.0, norms)
    centroid = unit.mean(axis=0)
    sims = unit @ centroid
    best = int(np.argmax(sims))  # first max → earliest on ties
    return members[best].doc_id


def select_canonical(members: list[DedupDocument], strategy: CanonicalStrategy) -> str:
    """Choose the surviving ``doc_id`` for one cluster's ``members``.

    Args:
        members: the cluster's documents, in input order. Must be
            non-empty.
        strategy: a :data:`CanonicalStrategy`.

    Returns:
        The ``doc_id`` of the chosen survivor.

    Raises:
        ValueError: empty ``members``, an unknown strategy name, a
            ``medoid`` request on members missing embeddings, or a
            callable returning a non-member ``doc_id``.
    """
    if not members:
        raise ValueError("select_canonical requires a non-empty member list")

    if callable(strategy):
        chosen = strategy(members)
        valid = {m.doc_id for m in members}
        if chosen not in valid:
            msg = (
                f"canonical callable returned {chosen!r}, which is not a member "
                f"of the cluster {sorted(valid)}."
            )
            raise ValueError(msg)
        return chosen

    if strategy == "first":
        return members[0].doc_id
    if strategy == "longest":
        # max() keeps the first maximal element → earliest among the longest.
        return max(members, key=lambda m: len(m.text or "")).doc_id
    if strategy == "shortest":
        return min(members, key=lambda m: len(m.text or "")).doc_id
    if strategy == "medoid":
        return _medoid_doc_id(members)

    msg = (
        f"unknown canonical strategy {strategy!r}. "
        "Expected one of 'first', 'longest', 'shortest', 'medoid', or a callable."
    )
    raise ValueError(msg)


def recanonicalize(
    report: DedupReport,
    documents: list[DedupDocument] | tuple[DedupDocument, ...],
    strategy: CanonicalStrategy,
) -> DedupReport:
    """Return a copy of ``report`` with each cluster's survivor re-chosen.

    Cluster membership, level provenance, similarity, and counts are all
    preserved — only ``DedupCluster.canonical_doc_id`` changes. ``"first"``
    short-circuits to the original report (it is already the pipeline's
    behavior).

    Args:
        report: a :class:`DedupReport` from :class:`DedupPipeline`.
        documents: the documents fed to the pipeline (to resolve member
            ``doc_id`` → :class:`DedupDocument` for text/embedding access).
        strategy: a :data:`CanonicalStrategy`.

    Returns:
        A new :class:`DedupReport` (the input is not mutated).
    """
    if strategy == "first":
        return report

    by_id = {d.doc_id: d for d in documents}
    new_clusters = []
    for cluster in report.clusters:
        members = [by_id[mid] for mid in cluster.member_doc_ids if mid in by_id]
        if not members:
            new_clusters.append(cluster)
            continue
        new_clusters.append(replace(cluster, canonical_doc_id=select_canonical(members, strategy)))
    return replace(report, clusters=tuple(new_clusters))


__all__ = ["CanonicalStrategy", "recanonicalize", "select_canonical"]
