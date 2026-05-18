"""Sec-5 regression tests: unbounded artifact reads (#5).

Pre-fix ``load_document`` and ``load_tabular`` called
``runtime.artifacts.read_text()`` unconditionally. A multi-gigabyte
artifact would force a full read followed by JSON + Pydantic
validation — OOMing the process. This is a denial-of-service vector,
not RCE, but real.

Fix: both functions accept ``max_bytes`` (default 16 MiB,
``None`` to disable). When the manifest reports a size above the cap,
``ArtifactTooLargeError`` is raised before any bytes are read.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from kaos_content.artifacts import (
    DEFAULT_LOAD_MAX_BYTES,
    load_document,
    load_tabular,
)
from kaos_content.errors import ArtifactTooLargeError


def _mock_runtime(
    *,
    manifest_size: int,
    payload: str,
    mime_type: str | None = "application/json",
) -> MagicMock:
    """Build a runtime stub with an artifact store that returns a
    manifest with ``manifest_size`` and a body of ``payload``.

    ``mime_type`` defaults to ``application/json`` so the Stage 5.2
    mime guard in :func:`load_document` is satisfied; tests that
    want to exercise the guard pass an explicit mime.
    """
    runtime = MagicMock()
    manifest = MagicMock()
    manifest.size = manifest_size
    manifest.mime_type = mime_type
    runtime.artifacts.get = MagicMock(return_value=manifest)
    runtime.artifacts.read_text = AsyncMock(return_value=payload)
    return runtime


_MIN_DOC_JSON = '{"metadata": {}, "body": []}'
_MIN_TABULAR_JSON = '{"metadata": {}, "tables": []}'


# ----- load_document --------------------------------------------------------


class TestLoadDocumentSizeCap:
    @pytest.mark.asyncio
    async def test_default_cap_blocks_oversized(self) -> None:
        # Manifest claims 100 MiB — well above the 16 MiB default.
        runtime = _mock_runtime(manifest_size=100 * 1024 * 1024, payload=_MIN_DOC_JSON)
        with pytest.raises(ArtifactTooLargeError) as exc_info:
            await load_document("artifact-id", runtime)
        # Error message names the cap and the actual size.
        assert "max_bytes" in str(exc_info.value).lower() or hasattr(exc_info.value, "context")
        # read_text was NEVER called — the cap rejected before reading.
        runtime.artifacts.read_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_default_cap_allows_small(self) -> None:
        # 1 KiB artifact — well under the 16 MiB default.
        runtime = _mock_runtime(manifest_size=1024, payload=_MIN_DOC_JSON)
        doc = await load_document("artifact-id", runtime)
        assert doc is not None
        runtime.artifacts.read_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_explicit_max_bytes_override(self) -> None:
        # 50 MiB artifact — bigger than default, but caller explicitly
        # raised the cap (think: SEC EDGAR 10-K filing).
        runtime = _mock_runtime(manifest_size=50 * 1024 * 1024, payload=_MIN_DOC_JSON)
        doc = await load_document("artifact-id", runtime, max_bytes=200 * 1024 * 1024)
        assert doc is not None

    @pytest.mark.asyncio
    async def test_max_bytes_none_disables_cap(self) -> None:
        # Even at 1 GiB, max_bytes=None means "I've thought about this".
        runtime = _mock_runtime(manifest_size=1024 * 1024 * 1024, payload=_MIN_DOC_JSON)
        doc = await load_document("artifact-id", runtime, max_bytes=None)
        assert doc is not None
        # Manifest IS still consulted for the Stage 5.2 mime guard,
        # even though the size cap is opted out.
        runtime.artifacts.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_default_max_bytes_constant(self) -> None:
        # The default is 16 MiB.
        assert DEFAULT_LOAD_MAX_BYTES == 16 * 1024 * 1024


# ----- load_tabular --------------------------------------------------------


class TestLoadTabularSizeCap:
    @pytest.mark.asyncio
    async def test_default_cap_blocks_oversized(self) -> None:
        runtime = _mock_runtime(manifest_size=100 * 1024 * 1024, payload=_MIN_TABULAR_JSON)
        with pytest.raises(ArtifactTooLargeError):
            await load_tabular("artifact-id", runtime)
        runtime.artifacts.read_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_default_cap_allows_small(self) -> None:
        runtime = _mock_runtime(manifest_size=1024, payload=_MIN_TABULAR_JSON)
        # load_tabular calls _tabular_from_json which validates; the
        # mock _MIN_TABULAR_JSON may not pass validation, but if it
        # raises something OTHER than ArtifactTooLargeError, that's
        # fine — we only care that the size check was bypassed and
        # read_text was reached.
        try:
            await load_tabular("artifact-id", runtime)
        except ArtifactTooLargeError:
            pytest.fail("Cap should not have triggered for 1 KiB artifact")
        except Exception:
            pass  # validation error from minimal JSON — OK
        runtime.artifacts.read_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_max_bytes_none_disables_cap(self) -> None:
        runtime = _mock_runtime(manifest_size=1024 * 1024 * 1024, payload=_MIN_TABULAR_JSON)
        try:
            await load_tabular("artifact-id", runtime, max_bytes=None)
        except ArtifactTooLargeError:
            pytest.fail("max_bytes=None must skip the cap")
        except Exception:
            pass  # validation error from minimal JSON — OK
        # Manifest IS still consulted for the load_tabular mime guard,
        # even though the size cap is opted out — parity with the
        # load_document equivalent above.
        runtime.artifacts.get.assert_called_once()


# ----- ArtifactTooLargeError class -----------------------------------------


class TestArtifactTooLargeError:
    def test_is_kaos_content_error(self) -> None:
        from kaos_content.errors import KaosContentError

        assert issubclass(ArtifactTooLargeError, KaosContentError)
