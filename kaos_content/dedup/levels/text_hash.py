"""Level 3: Exact text hash — format-variant duplicate detection.

Hashes the extracted text after normalization via the shared
:mod:`kaos_content.normalize` module so all KAOS dedup/search/
extraction consumers produce the same canonical form for identical
content.

Catches: same content in PDF + DOCX; minor whitespace drift.
Misses: any word-level edit, OCR variation, paraphrasing.
"""

from __future__ import annotations

from collections import defaultdict
from typing import ClassVar

from kaos_content.dedup.types import DedupCluster, DedupDocument, DedupLevel
from kaos_content.normalize import canonical_hash, normalize_text


class TextHashLevel(DedupLevel):
    """Exact hash on normalized extracted text."""

    name: ClassVar[str] = "text_hash"

    def __init__(
        self,
        *,
        lowercase: bool = True,
        strip_punctuation: bool = False,
        unicode_normalize: bool = True,
    ) -> None:
        self._lowercase = lowercase
        self._strip_punctuation = strip_punctuation
        self._unicode_normalize = unicode_normalize

    def find_clusters(
        self,
        documents: list[DedupDocument],
    ) -> list[DedupCluster]:
        groups: dict[str, list[DedupDocument]] = defaultdict(list)
        fingerprints: dict[str, str] = {}

        for doc in documents:
            if doc.text is None:
                continue
            normalized = normalize_text(
                doc.text,
                lowercase=self._lowercase,
                unicode_nfkc=self._unicode_normalize,
                strip_punctuation=self._strip_punctuation,
            )
            if not normalized:
                continue
            h = canonical_hash(normalized)
            groups[h].append(doc)
            fingerprints[doc.doc_id] = h

        clusters: list[DedupCluster] = []
        for h, members in groups.items():
            if len(members) < 2:
                continue
            clusters.append(
                DedupCluster(
                    cluster_id=f"text_{h[:16]}",
                    canonical_doc_id=members[0].doc_id,
                    member_doc_ids=tuple(m.doc_id for m in members),
                    level=self.name,
                    similarity=1.0,
                    fingerprints={m.doc_id: h for m in members},
                )
            )
        return clusters


__all__ = ["TextHashLevel"]
