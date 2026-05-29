"""Property/fuzz coverage for the redline engine's round-trip invariant.

For many randomly generated (original, revised) pairs — built by applying
random keep/edit/delete/insert/move operations to a base document — the
core invariant must hold: ``accept_all`` reproduces the revised text and
``reject_all`` reproduces the original text. Seeds are fixed so failures
are reproducible.
"""

from __future__ import annotations

import random

import pytest

from kaos_content import compare_documents
from kaos_content.model.blocks import Paragraph
from kaos_content.model.document import ContentDocument, DocumentMetadata
from kaos_content.model.inlines import Text
from kaos_content.revision import Revisions, accept_all, reject_all
from kaos_content.traversal.visitor import extract_text

_VOCAB = [
    "alpha", "beta", "gamma", "delta", "epsilon", "the", "party", "shall",
    "pay", "a", "fee", "under", "this", "clause", "term", "notice",
    "governing", "law", "venue", "indemnify", "within", "ten", "days",
]  # fmt: skip


def _doc(paras: list[str]) -> ContentDocument:
    return ContentDocument(
        metadata=DocumentMetadata(title=""),
        body=tuple(Paragraph(children=(Text(value=p),)) for p in paras),
    )


def _body_text(doc: ContentDocument) -> str:
    return "\n".join(extract_text(b) for b in doc.body)


def _rand_para(rng: random.Random) -> str:
    return " ".join(rng.choice(_VOCAB) for _ in range(rng.randint(3, 10)))


def _mutate(rng: random.Random, paras: list[str]) -> list[str]:
    """Apply random keep / edit / delete / insert ops, preserving alignment."""
    out: list[str] = []
    for p in paras:
        roll = rng.random()
        if roll < 0.15:  # delete
            continue
        if roll < 0.40:  # edit one or two words
            words = p.split()
            for _ in range(rng.randint(1, 2)):
                if words:
                    words[rng.randrange(len(words))] = rng.choice(_VOCAB)
            out.append(" ".join(words))
        else:  # keep
            out.append(p)
        if rng.random() < 0.15:  # insert a new paragraph after
            out.append(_rand_para(rng))
    # Occasionally relocate the first paragraph to the end (a move).
    if len(out) > 2 and rng.random() < 0.3:
        out.append(out.pop(0))
    return out


@pytest.mark.parametrize("seed", range(40))
@pytest.mark.parametrize("detect_moves", [True, False])
def test_random_pair_round_trips(seed: int, detect_moves: bool) -> None:
    rng = random.Random(seed)
    original = [_rand_para(rng) for _ in range(rng.randint(0, 12))]
    revised = _mutate(rng, original)

    odoc, rdoc = _doc(original), _doc(revised)
    redline = compare_documents(odoc, rdoc, author="Fuzz", detect_moves=detect_moves)

    assert _body_text(accept_all(redline)) == _body_text(rdoc), f"accept seed={seed}"
    assert _body_text(reject_all(redline)) == _body_text(odoc), f"reject seed={seed}"


@pytest.mark.parametrize("seed", range(20))
def test_random_pair_all_revisions_resolve(seed: int) -> None:
    """After accept_all or reject_all, no tracked changes remain."""
    rng = random.Random(1000 + seed)
    original = [_rand_para(rng) for _ in range(rng.randint(1, 10))]
    revised = _mutate(rng, original)
    redline = compare_documents(_doc(original), _doc(revised), author="Fuzz")
    assert len(Revisions.from_document(accept_all(redline))) == 0
    assert len(Revisions.from_document(reject_all(redline))) == 0
