"""Base AST node types."""

from __future__ import annotations

import sys

from pydantic import BaseModel, ConfigDict, Field

from kaos_content.model.attr import Attr, Provenance

if sys.version_info >= (3, 14):
    from uuid import uuid7 as _uuid7
else:
    from uuid6 import uuid7 as _uuid7


def _generate_node_id() -> str:
    """Generate a UUID v7 hex string for node identity."""
    return _uuid7().hex


class BaseNode(BaseModel):
    """Base for all AST nodes."""

    model_config = ConfigDict(frozen=True)
    id: str = Field(default_factory=_generate_node_id)
    attr: Attr = Attr()
    provenance: Provenance | None = None


class BaseBlock(BaseNode):
    """Base for all block-level nodes."""


class BaseInline(BaseNode):
    """Base for all inline-level nodes."""
