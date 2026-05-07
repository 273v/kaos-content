"""Pre-built pipeline configurations for common use cases.

Import and pass to :class:`DedupPipeline`::

    from kaos_content.dedup import DedupPipeline
    from kaos_content.dedup.presets import STANDARD

    pipeline = DedupPipeline(STANDARD)

Semantic clustering (``SemanticDedupLevel``) lives in
``kaos-nlp-transformers``. When that package is installed, the
``COMPREHENSIVE`` and ``OCR_AWARE`` presets pick it up automatically;
when it isn't, those presets degrade gracefully to the lexical levels.
Same plugin shape kaos-content uses for the ``[nlp]`` BM25 path.
"""

from __future__ import annotations

from kaos_content.dedup.levels.binary_hash import BinaryHashLevel
from kaos_content.dedup.levels.fuzzy_binary import FuzzyBinaryLevel
from kaos_content.dedup.levels.minhash import MinHashLevel
from kaos_content.dedup.levels.perceptual import PerceptualHashLevel
from kaos_content.dedup.levels.text_hash import TextHashLevel
from kaos_content.dedup.pipeline import DedupPipelineConfig
from kaos_content.dedup.types import DedupLevel

_SEMANTIC: DedupLevel | None
try:
    # kaos-nlp-transformers is a Wave 3 sibling — not on PyPI at v0.1.0a1,
    # so ty can't resolve the import statically. Runtime guarded by the
    # try/except below; install kaos-nlp-transformers to enable semantic
    # dedup.
    from kaos_nlp_transformers.clustering import (  # type: ignore[import-not-found]  # ty: ignore[unresolved-import]
        SemanticDedupLevel,
    )

    _SEMANTIC = SemanticDedupLevel(distance_threshold=0.10)
except ImportError:
    _SEMANTIC = None


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
13-word shingles) plus semantic embedding clustering when
``kaos-nlp-transformers`` is installed. Without that package the
preset degrades to the four lexical levels. Requires the ``[nlp]``
extra for MinHash."""

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
clustering when ``kaos-nlp-transformers`` is installed. Requires the
``[nlp]`` and ``[dedup-perceptual]`` extras."""

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
