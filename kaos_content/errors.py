"""Exception hierarchy for kaos-content.

All errors subclass KaosContentError -> KaosCoreError, carrying structured
details for agent-friendly error messages and middleware decision-making.
"""

from __future__ import annotations

from kaos_core.exceptions import KaosCoreError


class KaosContentError(KaosCoreError):
    """Base error for kaos-content operations."""


class SerializationError(KaosContentError):
    """Error during document serialization or deserialization."""


class SearchError(KaosContentError):
    """Error during document search operations."""


class ImageError(KaosContentError):
    """Error during image loading, decoding, or processing."""


class ImageDecompressionBombError(ImageError):
    """Image exceeded the configured pixel budget.

    Raised when a loaded image's pixel count is above
    ``kaos_content.images.model.MAX_IMAGE_PIXELS``. This is a defence
    against decompression-bomb attacks where a small compressed file
    expands to gigabytes of pixels in memory.
    """


class ImageSizeError(ImageError):
    """Image bytes exceeded the configured maximum size on load."""


class ArtifactTooLargeError(KaosContentError):
    """Artifact size exceeded the configured maximum on load.

    Raised by :func:`kaos_content.artifacts.load_document` and
    :func:`kaos_content.artifacts.load_tabular` (Sec-5, security
    finding #5) when the artifact's manifest reports a size above
    the caller's ``max_bytes`` cap.

    Mitigation: pass ``max_bytes=<larger>`` if the artifact is known
    to be legitimately large (e.g. SEC EDGAR 10-K filings routinely
    exceed 50 MB), or ``max_bytes=None`` to disable the cap entirely
    when the caller has explicitly thought about the trade-off.
    """


class ArtifactMimeTypeError(KaosContentError):
    """Artifact's mime type is incompatible with the requested loader.

    Raised by :func:`kaos_content.artifacts.load_document` when the
    artifact's manifest reports a ``mime_type`` that is not a
    serialized :class:`~kaos_content.model.document.ContentDocument`
    (i.e. not ``application/json`` / ``application/x-ndjson``). Also
    raised when the mime type is unknown but the body's first byte
    is ``<`` — a strong signal of an HTML/XML payload that would
    otherwise produce an opaque pydantic JSON-decode error.

    The message is agent-friendly: it names the offending mime type
    and points at the right preparatory tool
    (``kaos-content-parse-html`` for HTML,
    ``kaos-content-parse-markdown`` for Markdown/text) so the LLM
    can self-correct on its next turn instead of guessing.
    """
