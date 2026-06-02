"""One-call convenience over the dedup pipeline.

:func:`dedup` wraps the :class:`~kaos_content.dedup.pipeline.DedupPipeline`
+ presets + canonical selection into a single call that accepts plain
strings or :class:`~kaos_content.dedup.types.DedupDocument` objects. For
full control (custom level ordering, per-level config) construct a
``DedupPipeline`` directly; this is the ergonomic entry point for the
common case.
"""

from __future__ import annotations

from collections.abc import Sequence

from kaos_content.dedup.canonical import CanonicalStrategy, recanonicalize
from kaos_content.dedup.pipeline import DedupPipeline, DedupPipelineConfig
from kaos_content.dedup.types import DedupDocument, DedupLevel, DedupReport

_PRESET_NAMES = ("fast", "standard", "comprehensive", "legal", "ocr_aware")


def _resolve_preset(name: str) -> DedupPipelineConfig:
    """Look up a named preset.

    Imported lazily (inside the call) so ``import kaos_content.dedup`` does
    not eagerly trigger ``presets``' optional-extra probing (it attempts
    ``kaos_nlp_transformers`` / ``scipy`` imports to decide whether the
    semantic level is available).
    """
    key = name.lower()
    if key not in _PRESET_NAMES:
        msg = (
            f"unknown preset {name!r}. Expected one of "
            f"{sorted(_PRESET_NAMES)}, or pass levels=... explicitly."
        )
        raise ValueError(msg)
    from kaos_content.dedup import presets

    return getattr(presets, key.upper())


def _coerce_documents(items: Sequence[str | DedupDocument]) -> list[DedupDocument]:
    """Coerce inputs to :class:`DedupDocument`.

    Plain strings become text-only documents with their positional index
    as ``doc_id`` (``"0"``, ``"1"``, …). :class:`DedupDocument` instances
    pass through unchanged.
    """
    docs: list[DedupDocument] = []
    for i, item in enumerate(items):
        if isinstance(item, DedupDocument):
            docs.append(item)
        elif isinstance(item, str):
            docs.append(DedupDocument(doc_id=str(i), text=item))
        else:
            msg = f"dedup() items must be str or DedupDocument; item {i} is {type(item).__name__}."
            raise TypeError(msg)
    return docs


def dedup(
    items: Sequence[str | DedupDocument],
    *,
    preset: str = "standard",
    levels: Sequence[DedupLevel] | None = None,
    short_circuit: bool = True,
    canonical: CanonicalStrategy = "first",
) -> DedupReport:
    """Deduplicate ``items`` in one call.

    Args:
        items: documents to dedup — plain strings (text-only, ``doc_id`` =
            positional index) or :class:`DedupDocument` objects (use these
            to supply ``file_path``, ``embedding``, etc.).
        preset: named pipeline — ``"fast"``, ``"standard"`` (default),
            ``"comprehensive"``, ``"legal"``, or ``"ocr_aware"``. Ignored
            when ``levels`` is given. The semantic level in
            ``comprehensive`` / ``ocr_aware`` activates only when the
            ``[transformers]`` + ``[clustering]`` extras are installed.
        levels: explicit level sequence; overrides ``preset`` when given.
        short_circuit: when ``True`` (default), documents clustered by an
            earlier level skip later levels.
        canonical: how to pick each cluster's survivor — ``"first"``
            (default; input order), ``"longest"`` / ``"shortest"``,
            ``"medoid"`` (centroid-nearest; needs member embeddings), or a
            ``Callable[[list[DedupDocument]], str]``. See
            :data:`~kaos_content.dedup.canonical.CanonicalStrategy`.

    Returns:
        A :class:`DedupReport`.

    Raises:
        TypeError: an item is neither ``str`` nor :class:`DedupDocument`.
        ValueError: an unknown ``preset`` name (when ``levels`` is None).
    """
    docs = _coerce_documents(items)

    if levels is not None:
        config = DedupPipelineConfig(levels=tuple(levels), short_circuit=short_circuit)
    else:
        # Rebuild so an explicit short_circuit override is honored while
        # keeping the preset's level list.
        base = _resolve_preset(preset)
        config = DedupPipelineConfig(levels=base.levels, short_circuit=short_circuit)

    report = DedupPipeline(config).run(docs)
    return recanonicalize(report, docs, canonical)


__all__ = ["dedup"]
