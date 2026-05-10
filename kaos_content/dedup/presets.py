"""Pre-built pipeline configurations for common use cases.

Import and pass to :class:`DedupPipeline`::

    from kaos_content.dedup import DedupPipeline
    from kaos_content.dedup.presets import STANDARD

    pipeline = DedupPipeline(STANDARD)

Semantic clustering (``SemanticDedupLevel``) ships in this package
since 0.1.0a3 (KNT-602 Option A) but its run-time deps —
``kaos-nlp-transformers`` and ``scipy`` — are gated behind the
``[transformers]`` and ``[clustering]`` extras. The ``COMPREHENSIVE``
and ``OCR_AWARE`` presets pick up the semantic level whenever both
deps are importable; when either is missing, the presets degrade to
their lexical-only form so :class:`DedupPipeline.run` never crashes
mid-pipeline on a missing-extra ImportError. Same plugin shape
kaos-content uses for the ``[nlp]`` BM25 path.
"""

from __future__ import annotations

from kaos_content.dedup.levels.binary_hash import BinaryHashLevel
from kaos_content.dedup.levels.fuzzy_binary import FuzzyBinaryLevel
from kaos_content.dedup.levels.minhash import MinHashLevel
from kaos_content.dedup.levels.perceptual import PerceptualHashLevel
from kaos_content.dedup.levels.semantic import SemanticDedupLevel
from kaos_content.dedup.levels.text_hash import TextHashLevel
from kaos_content.dedup.pipeline import DedupPipelineConfig
from kaos_content.dedup.types import DedupLevel


def _semantic_available() -> bool:
    """True if the optional deps for SemanticDedupLevel are importable.

    Checks ``kaos-nlp-transformers`` (the embedding model) AND ``scipy``
    (the clustering algorithm) — both are required at ``find_clusters``
    time. Module-import time check so the preset's level-list reflects
    actual capability.
    """
    try:
        import kaos_nlp_transformers  # noqa: F401  # type: ignore[import-not-found]
        import scipy  # noqa: F401  # type: ignore[import-not-found]
    except ImportError:
        return False
    return True


_SEMANTIC: SemanticDedupLevel | None = (
    SemanticDedupLevel(distance_threshold=0.10) if _semantic_available() else None
)


def _maybe_with_semantic(*lexical: DedupLevel) -> tuple[DedupLevel, ...]:
    if _SEMANTIC is None:
        return lexical
    return (*lexical, _SEMANTIC)


FAST = DedupPipelineConfig(
    levels=(
        BinaryHashLevel(algorithm="sha256"),
        TextHashLevel(lowercase=True),
    ),
    short_circuit=True,
)
"""Binary hash + text hash only. Catches exact and format-variant dups.
No fuzzy or semantic detection. Good for quick pass on small corpora."""

STANDARD = DedupPipelineConfig(
    levels=(
        BinaryHashLevel(algorithm="sha256"),
        FuzzyBinaryLevel(threshold=0.7),
        TextHashLevel(lowercase=True),
        MinHashLevel(shingle_size=5, threshold=0.8),
    ),
    short_circuit=True,
)
"""All four non-semantic levels. Binary → fuzzy binary → text hash →
MinHash near-dup. Catches identical files, re-saved files, format
variants, and near-duplicate text."""

COMPREHENSIVE = DedupPipelineConfig(
    levels=_maybe_with_semantic(
        BinaryHashLevel(algorithm="sha256"),
        FuzzyBinaryLevel(threshold=0.7),
        TextHashLevel(lowercase=True),
        MinHashLevel(shingle_size=13, threshold=0.8),
    ),
    short_circuit=True,
)
"""Lexical levels (binary → fuzzy binary → text hash → MinHash with
13-word shingles) plus semantic embedding clustering when the
``[transformers]`` and ``[clustering]`` extras are installed.
Without them the preset degrades to the four lexical levels.
Requires the ``[nlp]`` extra for MinHash."""

OCR_AWARE = DedupPipelineConfig(
    levels=_maybe_with_semantic(
        BinaryHashLevel(algorithm="sha256"),
        PerceptualHashLevel(algorithm="dhash", max_hamming_distance=5),
        MinHashLevel(shingle_size=5, threshold=0.5),
    ),
    short_circuit=True,
)
"""For corpora with scanned PDFs. Binary → perceptual page hash →
MinHash (low threshold for OCR-noisy text), plus semantic embedding
clustering when the ``[transformers]`` and ``[clustering]`` extras
are installed. Requires the ``[nlp]`` and ``[dedup-perceptual]``
extras for the lexical / image levels."""

LEGAL = DedupPipelineConfig(
    levels=(
        BinaryHashLevel(algorithm="sha256"),
        TextHashLevel(lowercase=True, strip_punctuation=False),
        MinHashLevel(shingle_size=13, num_perms=128, threshold=0.8),
    ),
    short_circuit=True,
)
"""Legal corpus preset: binary + text + MinHash with 13-word shingles.
Skips fuzzy binary (legal PDFs rarely get re-saved byte-similar)."""


__all__ = ["COMPREHENSIVE", "FAST", "LEGAL", "OCR_AWARE", "STANDARD"]
