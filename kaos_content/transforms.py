"""Document transform protocol and composition.

A ``DocumentTransform`` is anything callable that takes a
``ContentDocument`` and returns a new ``ContentDocument``. Most transforms
already in the codebase conform (``revision.accept_all``, ``reject_all``,
``at_time``, etc.); this module gives them a shared protocol and a
``compose`` helper so agent pipelines can chain them without ceremony.

Example::

    from kaos_content.transforms import compose
    from kaos_content.revision import accept_by_author

    pipeline = compose(
        accept_by_author("Alice"),
        lambda d: d,  # any DocumentTransform-compatible callable
    )
    result = pipeline(doc)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from kaos_content.model.document import ContentDocument


@runtime_checkable
class DocumentTransform(Protocol):
    """A callable that maps ``ContentDocument`` → ``ContentDocument``.

    The transform MUST NOT mutate its input. ``kaos-content`` uses
    frozen Pydantic v2 models; returning a new document via
    ``model_copy(update=...)`` or reconstructing the body tuple is the
    idiomatic pattern.
    """

    def __call__(self, doc: ContentDocument) -> ContentDocument: ...


def compose(*transforms: DocumentTransform) -> DocumentTransform:
    """Compose ``transforms`` into a single left-to-right pipeline.

    ``compose()`` with no args returns the identity transform.
    ``compose(f)`` is equivalent to ``f``.
    ``compose(f, g, h)(doc)`` == ``h(g(f(doc)))``.
    """
    if not transforms:
        return _identity
    if len(transforms) == 1:
        return transforms[0]

    def _composed(doc: ContentDocument) -> ContentDocument:
        for t in transforms:
            doc = t(doc)
        return doc

    return _composed


def _identity(doc: ContentDocument) -> ContentDocument:
    """The identity transform — returns its input unchanged."""
    return doc


def apply(
    doc: ContentDocument,
    *transforms: DocumentTransform,
) -> ContentDocument:
    """Apply a sequence of transforms to ``doc`` in order.

    Equivalent to ``compose(*transforms)(doc)`` but reads more naturally
    when there's a document and a transform list at hand.
    """
    for t in transforms:
        doc = t(doc)
    return doc
