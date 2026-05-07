"""kaos-content parsers — plain text, markdown, and HTML to ContentDocument AST.

``parse_plain_text`` is always available. ``parse_markdown`` requires
the ``[markdown]`` extra and is re-exported only when its dependency
is installed, so importing this package does not fail in minimal
environments.
"""

import contextlib

from kaos_content.parsers.plain import parse_plain_text

__all__ = ["parse_plain_text"]

# Re-export parse_markdown only if the [markdown] extra is installed.
with contextlib.suppress(ImportError):
    from kaos_content.parsers.markdown import parse_markdown  # noqa: F401

    __all__.append("parse_markdown")

# parse_html is intentionally NOT auto-imported even when [html] is
# installed — historically users have imported it directly to keep the
# lxml import explicit. To use it:
#
#     from kaos_content.parsers.html import parse_html
