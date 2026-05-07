"""Shared helpers for tracked-changes view modes in serializers.

Serializers accept a ``view`` parameter with three values:

- ``"final"`` (default) — render the accepted version: skip ``rev-del`` and
  ``rev-move-from`` nodes, unwrap ``rev-ins`` and ``rev-move-to`` nodes.
- ``"original"`` — render the version before changes: skip ``rev-ins`` and
  ``rev-move-to``, unwrap ``rev-del`` and ``rev-move-from``.
- ``"markup"`` — render both with visual differentiation.

This module is internal; serializers import the helpers directly.
"""

from __future__ import annotations

from typing import Any, Literal

ViewMode = Literal["final", "original", "markup"]

# Class names emitted by the DOCX reader's track_changes=True mode.
_REV_INS = "rev-ins"
_REV_DEL = "rev-del"
_REV_MOVE_FROM = "rev-move-from"
_REV_MOVE_TO = "rev-move-to"

_REV_CLASSES: frozenset[str] = frozenset({_REV_INS, _REV_DEL, _REV_MOVE_FROM, _REV_MOVE_TO})


def revision_class(node: Any) -> str | None:
    """Return the ``rev-*`` class on a node's Attr, or None.

    Checks the first matching class; documents that mix multiple revision
    classes on a single node are malformed.
    """
    attr = getattr(node, "attr", None)
    if attr is None:
        return None
    for cls in getattr(attr, "classes", ()) or ():
        if cls in _REV_CLASSES:
            return cls
    return None


def should_skip_revision(rev_class: str, view: ViewMode) -> bool:
    """Whether a node with ``rev_class`` should be entirely omitted in ``view``."""
    if view == "final":
        return rev_class in (_REV_DEL, _REV_MOVE_FROM)
    if view == "original":
        return rev_class in (_REV_INS, _REV_MOVE_TO)
    return False  # markup view renders everything


def wrap_text_markup(content: str, rev_class: str) -> str:
    """Wrap content for text view="markup" with ASCII markers."""
    if not content:
        return content
    if rev_class == _REV_INS:
        return "{+" + content + "+}"
    if rev_class == _REV_DEL:
        return "{-" + content + "-}"
    if rev_class == _REV_MOVE_FROM:
        return "{<<" + content + "<<}"
    if rev_class == _REV_MOVE_TO:
        return "{>>" + content + ">>}"
    return content


def wrap_markdown_markup(content: str, rev_class: str) -> str:
    """Wrap content for markdown view="markup" using HTML5 ins/del elements."""
    if not content:
        return content
    if rev_class == _REV_INS:
        return f"<ins>{content}</ins>"
    if rev_class == _REV_DEL:
        return f"<del>{content}</del>"
    if rev_class == _REV_MOVE_FROM:
        return f'<del class="rev-move-from">{content}</del>'
    if rev_class == _REV_MOVE_TO:
        return f'<ins class="rev-move-to">{content}</ins>'
    return content


def wrap_html_markup(content: str, rev_class: str) -> str:
    """Wrap content for HTML view="markup" using semantic ins/del elements."""
    if not content:
        return content
    if rev_class == _REV_INS:
        return f'<ins class="{rev_class}">{content}</ins>'
    if rev_class == _REV_DEL:
        return f'<del class="{rev_class}">{content}</del>'
    if rev_class == _REV_MOVE_FROM:
        return f'<del class="{rev_class}">{content}</del>'
    if rev_class == _REV_MOVE_TO:
        return f'<ins class="{rev_class}">{content}</ins>'
    return content
