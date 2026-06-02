"""Concrete dedup levels — one module per algorithm family.

KNT-602 Option A (kaos-content 0.1.0a3): ``SemanticDedupLevel`` now
lives here too, alongside the lexical levels. The level lazy-imports
``kaos-nlp-transformers`` and ``scipy`` (the optional embedding +
clustering deps) so the module is importable without them; install
``kaos-content[transformers,clustering]`` to actually run the level.

Pre-0.1.0a3 the level lived in ``kaos_nlp_transformers.clustering`` —
that path is removed in kaos-nlp-transformers 0.2.0a3. Update imports
to ``from kaos_content.dedup.levels.semantic import SemanticDedupLevel``.
"""

from __future__ import annotations

from kaos_content.dedup.levels.binary_hash import BinaryHashLevel
from kaos_content.dedup.levels.fuzzy_binary import FuzzyBinaryLevel
from kaos_content.dedup.levels.minhash import MinHashLevel
from kaos_content.dedup.levels.perceptual import PerceptualHashLevel
from kaos_content.dedup.levels.semantic import SemanticDedupLevel
from kaos_content.dedup.levels.semantic_graph import SemanticGraphDedupLevel
from kaos_content.dedup.levels.text_hash import TextHashLevel

__all__ = [
    "BinaryHashLevel",
    "FuzzyBinaryLevel",
    "MinHashLevel",
    "PerceptualHashLevel",
    "SemanticDedupLevel",
    "SemanticGraphDedupLevel",
    "TextHashLevel",
]
