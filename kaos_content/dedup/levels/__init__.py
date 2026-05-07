"""Concrete dedup levels — one module per algorithm family.

Semantic embedding clustering lives in kaos-nlp-transformers
(``kaos_nlp_transformers.clustering.SemanticDedupLevel``) because it
requires running an embedding model at inference time. kaos-content
owns the ``DedupLevel`` Protocol; kaos-nlp-transformers registers an
implementation against it.
"""

from __future__ import annotations

from kaos_content.dedup.levels.binary_hash import BinaryHashLevel
from kaos_content.dedup.levels.fuzzy_binary import FuzzyBinaryLevel
from kaos_content.dedup.levels.minhash import MinHashLevel
from kaos_content.dedup.levels.perceptual import PerceptualHashLevel
from kaos_content.dedup.levels.text_hash import TextHashLevel

__all__ = [
    "BinaryHashLevel",
    "FuzzyBinaryLevel",
    "MinHashLevel",
    "PerceptualHashLevel",
    "TextHashLevel",
]
