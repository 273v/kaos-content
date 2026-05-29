"""Generate tracked changes by comparing two ``ContentDocument`` trees.

This is the "redline engine": given an *original* and a *revised*
document, :func:`compare_documents` returns a new ``ContentDocument`` in
which the differences are expressed as tracked-change wrappers — the same
``rev-*`` ``Span`` / ``Div`` nodes the DOCX reader produces and the DOCX
writer serializes. The result round-trips through
:mod:`kaos_content.revision` (``accept_all`` yields ``revised``,
``reject_all`` yields ``original``) and, via ``kaos-office``, writes out a
Word document with native tracked changes.

The engine is AST-level and format-agnostic: any two ``ContentDocument``
values can be compared, so DOCX↔DOCX, PDF↔PDF, and HTML↔HTML redlines all
flow through the same code path.

Algorithm (MVP):

1. **Block alignment.** Align ``original.body`` and ``revised.body`` with
   :class:`difflib.SequenceMatcher` over normalized block text.
2. **Equal** blocks are kept from the revised document unchanged.
3. **Replace** windows are paired positionally; a similar paragraph pair
   is word-diffed into mixed plain / ``rev-ins`` / ``rev-del`` inline
   content, while a dissimilar pair (or a non-paragraph block) is emitted
   as a block deletion followed by a block insertion.
4. **Delete** / **insert** blocks become block-level ``rev-del`` /
   ``rev-ins`` wrappers.
5. **Move detection** (optional) pairs a deleted block with a
   high-similarity inserted block elsewhere and re-tags the pair as
   ``rev-move-from`` / ``rev-move-to`` sharing a move name.

Known MVP limitations: a word-diffed paragraph is rebuilt from its plain
text, so inline run formatting (bold, links) inside a *changed* paragraph
is not preserved; unchanged paragraphs keep full fidelity. Block
alignment keys on text only, so two blocks with identical text but
different types align as equal. Move detection runs over whole
delete/insert blocks, not replace-internal leftovers.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from typing import TYPE_CHECKING

from kaos_content.revision import (
    make_block_deletion,
    make_block_insertion,
    make_block_move_from,
    make_block_move_to,
    make_inline_deletion,
    make_inline_insertion,
)
from kaos_content.traversal.visitor import extract_text

if TYPE_CHECKING:
    from kaos_content.model.blocks import Block
    from kaos_content.model.document import ContentDocument
    from kaos_content.model.inlines import Inline

# Block node types whose ``children`` are inline content and so can be
# word-diffed in place.
_INLINE_CONTENT_BLOCKS = frozenset({"paragraph", "heading"})

# Split text into runs of whitespace and runs of non-whitespace so the
# word-level diff aligns on word boundaries while preserving spacing.
_WORD_RE = re.compile(r"\s+|\S+")

# Move detection pairs every deleted block against every inserted block,
# which is O(deleted x inserted) similarity ratios. On a document with
# hundreds of insertions and deletions that product explodes, so we cap
# the number of candidate pairs and, when exceeded, skip move detection
# entirely (relocations then surface as delete + insert). The cap is
# logged, never silent.
_MOVE_PAIR_BUDGET = 50_000


def _logger() -> logging.Logger:
    """Lazily fetch the package logger (kept out of import-time work)."""
    from kaos_core.logging import get_logger

    return get_logger(__name__)


@dataclass(slots=True)
class _IdGen:
    """Monotonic source of unique numeric ``rev:id`` strings.

    OOXML requires ``w:id`` be unique within a document; Word uses small
    integers, so we mirror that with ``"0"``, ``"1"``, ...
    """

    _next: int = 0

    def next(self) -> str:
        rid = str(self._next)
        self._next += 1
        return rid


@dataclass(slots=True)
class _MoveGen:
    """Monotonic source of unique move names shared by a move pair."""

    _next: int = 0

    def next(self) -> str:
        name = f"move{self._next}"
        self._next += 1
        return name


def _norm(text: str) -> str:
    """Collapse runs of whitespace so alignment ignores formatting noise."""
    return " ".join(text.split())


def _ratio(a: str, b: str) -> float:
    """Similarity ratio in [0, 1] between two normalized strings."""
    if not a and not b:
        return 1.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


@dataclass(frozen=True, slots=True)
class _Settings:
    """Resolved knobs threaded through the comparison."""

    author: str
    date: datetime | None
    detect_moves: bool
    similarity_threshold: float
    move_threshold: float


def compare_documents(
    original: ContentDocument,
    revised: ContentDocument,
    *,
    author: str = "Reviewer",
    date: datetime | None = None,
    detect_moves: bool = True,
    similarity_threshold: float = 0.5,
    move_threshold: float = 0.8,
) -> ContentDocument:
    """Compare two documents and return a redlined ``ContentDocument``.

    Args:
        original: The baseline document.
        revised: The edited document.
        author: Author name recorded on every generated revision.
        date: Timestamp recorded on every generated revision (``None`` =
            undated).
        detect_moves: When True, a deleted block that closely matches an
            inserted block elsewhere is re-tagged as a move pair.
        similarity_threshold: Minimum text-similarity ratio for a
            replaced paragraph pair to be word-diffed in place rather than
            emitted as a full delete + insert. In ``[0, 1]``.
        move_threshold: Minimum text-similarity ratio for a delete/insert
            pair to be treated as a move. In ``[0, 1]``.

    Returns:
        A new ``ContentDocument`` (metadata, footnotes, and annotations
        carried from ``revised``) whose ``body`` expresses the diff as
        ``rev-*`` wrappers.
    """
    settings = _Settings(
        author=author,
        date=date,
        detect_moves=detect_moves,
        similarity_threshold=similarity_threshold,
        move_threshold=move_threshold,
    )
    ids = _IdGen()
    moves = _MoveGen()

    orig_blocks = tuple(original.body)
    rev_blocks = tuple(revised.body)
    orig_keys = [_norm(extract_text(b)) for b in orig_blocks]
    rev_keys = [_norm(extract_text(b)) for b in rev_blocks]

    opcodes = SequenceMatcher(None, orig_keys, rev_keys, autojunk=False).get_opcodes()

    move_from, move_to = (
        _plan_moves(opcodes, orig_keys, rev_keys, moves, settings)
        if settings.detect_moves
        else ({}, {})
    )

    body: list[Block] = []
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            body.extend(rev_blocks[j1:j2])
        elif tag == "delete":
            for oi in range(i1, i2):
                body.append(_emit_deleted(orig_blocks[oi], oi, ids, move_from, settings))
        elif tag == "insert":
            for rj in range(j1, j2):
                body.append(_emit_inserted(rev_blocks[rj], rj, ids, move_to, settings))
        else:  # replace
            body.extend(
                _emit_replace(
                    orig_blocks[i1:i2],
                    rev_blocks[j1:j2],
                    orig_keys[i1:i2],
                    rev_keys[j1:j2],
                    ids,
                    settings,
                )
            )

    return revised.model_copy(update={"body": tuple(body)})


def _plan_moves(
    opcodes: Sequence[tuple[str, int, int, int, int]],
    orig_keys: list[str],
    rev_keys: list[str],
    moves: _MoveGen,
    settings: _Settings,
) -> tuple[dict[int, str], dict[int, str]]:
    """Greedily pair whole deleted blocks with whole inserted blocks.

    Returns ``(move_from, move_to)`` index→move-name maps. Only ``delete``
    and ``insert`` opcodes participate; replace windows are in-place edits.
    """
    deleted = [oi for tag, i1, i2, _, _ in opcodes if tag == "delete" for oi in range(i1, i2)]
    inserted = [rj for tag, _, _, j1, j2 in opcodes if tag == "insert" for rj in range(j1, j2)]

    move_from: dict[int, str] = {}
    move_to: dict[int, str] = {}
    if not deleted or not inserted:
        return move_from, move_to

    # Bound the quadratic candidate space. When too large, skip move
    # detection rather than stall — relocations remain correct, just shown
    # as delete + insert. Logged so the downgrade is never silent.
    if len(deleted) * len(inserted) > _MOVE_PAIR_BUDGET:
        _logger().warning(
            "compare_documents: move detection skipped — %d deleted x %d inserted "
            "candidates exceed the %d-pair budget; relocations will appear as "
            "delete + insert.",
            len(deleted),
            len(inserted),
            _MOVE_PAIR_BUDGET,
        )
        return move_from, move_to

    threshold = settings.move_threshold
    used_inserts: set[int] = set()
    for oi in deleted:
        a = orig_keys[oi]
        best_rj = -1
        best_ratio = threshold
        for rj in inserted:
            if rj in used_inserts:
                continue
            b = rev_keys[rj]
            # Cheap length prefilter: difflib's ratio is bounded above by
            # 2*min/(len_a+len_b), so skip pairs that can't clear the bar
            # before paying for the full O(len) comparison.
            la, lb = len(a), len(b)
            if la and lb and 2 * min(la, lb) / (la + lb) < best_ratio:
                continue
            r = _ratio(a, b)
            if r >= best_ratio:
                best_ratio = r
                best_rj = rj
        if best_rj >= 0:
            name = moves.next()
            move_from[oi] = name
            move_to[best_rj] = name
            used_inserts.add(best_rj)

    return move_from, move_to


def _emit_deleted(
    block: Block,
    oi: int,
    ids: _IdGen,
    move_from: dict[int, str],
    settings: _Settings,
) -> Block:
    """Wrap a removed original block as a deletion or a move-from."""
    name = move_from.get(oi)
    if name is not None:
        return make_block_move_from(
            block,
            author=settings.author,
            move_name=name,
            date=settings.date,
            revision_id=ids.next(),
        )
    return make_block_deletion(
        block, author=settings.author, date=settings.date, revision_id=ids.next()
    )


def _emit_inserted(
    block: Block,
    rj: int,
    ids: _IdGen,
    move_to: dict[int, str],
    settings: _Settings,
) -> Block:
    """Wrap an added revised block as an insertion or a move-to."""
    name = move_to.get(rj)
    if name is not None:
        return make_block_move_to(
            block,
            author=settings.author,
            move_name=name,
            date=settings.date,
            revision_id=ids.next(),
        )
    return make_block_insertion(
        block, author=settings.author, date=settings.date, revision_id=ids.next()
    )


def _emit_replace(
    orig_blocks: tuple[Block, ...],
    rev_blocks: tuple[Block, ...],
    orig_keys: list[str],
    rev_keys: list[str],
    ids: _IdGen,
    settings: _Settings,
) -> list[Block]:
    """Emit blocks for a replace window by pairing originals with reviseds.

    Positionally paired blocks are word-diffed in place when both are
    paragraph-like and similar enough; otherwise the pair becomes a block
    deletion followed by a block insertion. Unpaired leftovers fall back
    to a deletion (extra originals) or insertion (extra reviseds).
    """
    out: list[Block] = []
    paired = min(len(orig_blocks), len(rev_blocks))
    for k in range(paired):
        ob, rb = orig_blocks[k], rev_blocks[k]
        if (
            ob.node_type in _INLINE_CONTENT_BLOCKS
            and rb.node_type in _INLINE_CONTENT_BLOCKS
            and _ratio(orig_keys[k], rev_keys[k]) >= settings.similarity_threshold
        ):
            out.append(_word_diff_block(ob, rb, ids, settings))
        else:
            out.append(
                make_block_deletion(
                    ob, author=settings.author, date=settings.date, revision_id=ids.next()
                )
            )
            out.append(
                make_block_insertion(
                    rb, author=settings.author, date=settings.date, revision_id=ids.next()
                )
            )

    for k in range(paired, len(orig_blocks)):
        out.append(
            make_block_deletion(
                orig_blocks[k], author=settings.author, date=settings.date, revision_id=ids.next()
            )
        )
    for k in range(paired, len(rev_blocks)):
        out.append(
            make_block_insertion(
                rev_blocks[k], author=settings.author, date=settings.date, revision_id=ids.next()
            )
        )
    return out


def _word_diff_block(original: Block, revised: Block, ids: _IdGen, settings: _Settings) -> Block:
    """Rebuild ``revised`` with word-level ``rev-ins`` / ``rev-del`` inlines.

    The returned block keeps the revised block's identity (type, depth,
    numbering label, attr) and replaces its inline children with the
    word-diff. When the two texts are identical the result is a single
    plain ``Text`` (no wrappers).
    """
    children = _word_diff_inlines(extract_text(original), extract_text(revised), ids, settings)
    return revised.model_copy(update={"children": children})


def _word_diff_inlines(
    original_text: str, revised_text: str, ids: _IdGen, settings: _Settings
) -> tuple[Inline, ...]:
    """Word-level diff of two strings into a tuple of inline nodes."""
    from kaos_content.model.inlines import Text

    a = _WORD_RE.findall(original_text)
    b = _WORD_RE.findall(revised_text)
    out: list[Inline] = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            out.append(Text(value="".join(b[j1:j2])))
            continue
        if tag in ("delete", "replace"):
            deleted = "".join(a[i1:i2])
            if deleted:
                out.append(
                    make_inline_deletion(
                        Text(value=deleted),
                        author=settings.author,
                        date=settings.date,
                        revision_id=ids.next(),
                    )
                )
        if tag in ("insert", "replace"):
            inserted = "".join(b[j1:j2])
            if inserted:
                out.append(
                    make_inline_insertion(
                        Text(value=inserted),
                        author=settings.author,
                        date=settings.date,
                        revision_id=ids.next(),
                    )
                )
    return tuple(out)
