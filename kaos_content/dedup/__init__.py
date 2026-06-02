"""Document deduplication pipeline — composable, multi-level.

Five levels from cheapest to most expensive:

1. **Exact binary hash** — byte-identical files (SHA-256/BLAKE2b)
2. **Fuzzy binary hash** — re-saved files (CTPH via kaos-nlp-core)
3. **Exact text hash** — format-variant dups (same text in PDF + DOCX)
4. **Near-dup text** — MinHash + LSH (kaos-nlp-core, token shingles)
5. **Semantic clustering** — template families (embedding + cosine,
   agglomerative)
6. **Semantic reachability** — embedding similarity graph + connected
   components; merges transitive paraphrase chains (``A ~ B ~ C``). Wired
   through ``dedup(embedder=...)``.

Each level is a standalone class implementing :class:`DedupLevel`. The
:class:`DedupPipeline` chains them in order with short-circuit: documents
clustered at level N skip level N+1.

Quick start::

    from kaos_content.dedup import DedupPipeline, DedupDocument, presets

    docs = [DedupDocument(doc_id="d1", file_path=Path("a.pdf")),
            DedupDocument(doc_id="d2", file_path=Path("b.pdf"))]
    pipeline = DedupPipeline(presets.STANDARD)
    report = pipeline.run(docs)
    print(report.total_unique, "unique out of", report.total_input)
"""

from __future__ import annotations

from kaos_content.dedup.api import dedup
from kaos_content.dedup.canonical import (
    CanonicalStrategy,
    recanonicalize,
    select_canonical,
)
from kaos_content.dedup.pipeline import DedupPipeline
from kaos_content.dedup.types import (
    DedupCluster,
    DedupDocument,
    DedupLevel,
    DedupReport,
)

__all__ = [
    "CanonicalStrategy",
    "DedupCluster",
    "DedupDocument",
    "DedupLevel",
    "DedupPipeline",
    "DedupReport",
    "dedup",
    "recanonicalize",
    "select_canonical",
]
