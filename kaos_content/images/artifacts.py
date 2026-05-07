"""Store and load KaosImage instances as kaos-core VFS artifacts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from kaos_core.artifacts.models import ArtifactManifest, ArtifactRef
from kaos_core.types.enums import ArtifactRetentionPolicy, ArtifactRole

from kaos_content.errors import ImageSizeError

if TYPE_CHECKING:
    from kaos_core.base.context import KaosContext
    from kaos_core.registry.container import KaosRuntime

    from kaos_content.images.model import KaosImage


# Default cap on the number of bytes ``load_image`` will read from a
# VFS artifact before refusing. 50 MB covers virtually all real-world
# raster images and prevents accidentally streaming a multi-GB blob
# into memory. Pass ``max_bytes=None`` to disable (NOT recommended).
DEFAULT_LOAD_IMAGE_MAX_BYTES: int = 50_000_000


async def store_image(
    image: KaosImage,
    runtime: KaosRuntime,
    context: KaosContext,
    *,
    name: str = "image",
    format: str = "png",
    quality: int = 95,
    description: str | None = None,
    retention_policy: ArtifactRetentionPolicy = ArtifactRetentionPolicy.SESSION,
    metadata: dict[str, Any] | None = None,
) -> ArtifactManifest:
    """Serialize a KaosImage and store as a VFS artifact.

    Returns the ArtifactManifest for the stored image.
    """
    from kaos_content.images.model import _FORMAT_TO_MIME

    data = image.to_bytes(format=format, quality=quality)
    mime_type = _FORMAT_TO_MIME.get(format, f"image/{format}")
    ext = format.lower()

    vfs_path = f"images/{name}.{ext}"
    ctx_path = context.get_vfs_path(vfs_path)
    await ctx_path.write_bytes(data)

    img_metadata = {
        "width": image.width,
        "height": image.height,
        "mode": image.mode,
        "dpi": list(image.dpi),
        "format": format,
        **(metadata or {}),
    }

    return await runtime.artifacts.create_from_path(
        vfs_path,
        context_id=context.session_id,
        session_id=context.session_id,
        name=name,
        description=description or f"KaosImage ({image.width}x{image.height}, {format})",
        mime_type=mime_type,
        role=ArtifactRole.BODY,
        provenance={"format": format, "mode": image.mode},
        retention_policy=retention_policy,
        metadata=img_metadata,
    )


async def load_image(
    artifact_ref: ArtifactRef | str,
    runtime: KaosRuntime,
    *,
    max_bytes: int | None = DEFAULT_LOAD_IMAGE_MAX_BYTES,
) -> KaosImage:
    """Load a KaosImage from a VFS artifact.

    Args:
        artifact_ref: Artifact id or ``ArtifactRef`` to load.
        runtime: KAOS runtime for VFS access.
        max_bytes: Hard cap on the number of bytes read from the
            artifact body. Defaults to ``DEFAULT_LOAD_IMAGE_MAX_BYTES``
            (50 MB). Pass ``None`` to disable (NOT recommended for
            untrusted artifacts).

    Raises:
        ImageSizeError: if the artifact body is larger than
            ``max_bytes``.
        ImageDecompressionBombError: if the decoded image's pixel count
            exceeds ``MAX_IMAGE_PIXELS``.
    """
    from kaos_content.images.model import KaosImage

    artifact_id = artifact_ref if isinstance(artifact_ref, str) else artifact_ref.artifact_id
    manifest = await runtime.artifacts._resolve_async(artifact_id)

    if max_bytes is not None and manifest.size is not None and manifest.size > max_bytes:
        msg = (
            f"Image artifact {artifact_id} is {manifest.size} bytes, "
            f"exceeds max_bytes={max_bytes}. Pass a larger max_bytes if "
            f"this is intentional."
        )
        raise ImageSizeError(msg)

    data = await runtime.artifacts.read_body(artifact_id, max_bytes=max_bytes)

    if max_bytes is not None and len(data) > max_bytes:
        msg = (
            f"Image artifact {artifact_id} body is {len(data)} bytes, "
            f"exceeds max_bytes={max_bytes}."
        )
        raise ImageSizeError(msg)

    img_meta = manifest.metadata or {}
    dpi = tuple(img_meta.get("dpi", [72, 72]))

    return KaosImage.from_bytes(
        data,
        dpi=(dpi[0], dpi[1]) if len(dpi) >= 2 else (72, 72),
        metadata=img_meta,
    )
