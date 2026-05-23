"""Pin the base-install honesty contract for ``kaos_content.dedup.levels``.

audit-04 F-001: ``kaos_content/dedup/levels/__init__.py:3-7`` and the
``semantic.py:9-14`` docstring both promise that the module imports
without ``numpy`` / ``scipy`` / ``kaos-nlp-transformers`` installed,
and that ``find_clusters`` raises an actionable ``ImportError`` when
they are actually needed. The promise was false before this test
landed because ``semantic.py`` imported ``numpy`` at module scope,
so any ``from kaos_content.dedup.levels import ...`` blew up on a
base install (numpy is only in the ``[images]`` / ``[layout]`` /
``[clustering]`` extras).

This test simulates the missing-numpy install by routing ``numpy``
through a meta-path finder that refuses imports, then exercises the
public import surface that the audit reproducer used. ``find_clusters``
is intentionally NOT called — the test only pins the import-time
contract; ``find_clusters`` correctness with numpy installed is
covered elsewhere.
"""

from __future__ import annotations

import importlib
import importlib.abc
import sys


class _BlockNumpy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "numpy" or fullname.startswith("numpy."):
            raise ImportError(f"blocked {fullname}")
        return None


def test_dedup_levels_import_without_numpy() -> None:
    """``import kaos_content.dedup.levels`` must succeed without numpy.

    Reproduces the audit-04 F-001 evidence block. Closes the
    documented-but-broken lazy-import promise in
    ``kaos_content/dedup/levels/__init__.py:3-7``.
    """
    # Snapshot + isolate the import state so other tests aren't affected
    # by the meta-path block.
    blocker = _BlockNumpy()
    sys.meta_path.insert(0, blocker)
    # Drop cached modules so the import path re-executes under the block.
    for mod in [
        name
        for name in sys.modules
        if name.startswith("kaos_content.dedup.levels") or name == "numpy"
    ]:
        del sys.modules[mod]

    try:
        levels = importlib.import_module("kaos_content.dedup.levels")
        # The promised public exports must all be importable.
        assert hasattr(levels, "MinHashLevel")
        assert hasattr(levels, "SemanticDedupLevel")
        # The constructor must not touch numpy either.
        instance = levels.SemanticDedupLevel()
        assert instance is not None
    finally:
        if blocker in sys.meta_path:
            sys.meta_path.remove(blocker)
        # Let the next test re-import cleanly with numpy available.
        for mod in [
            name
            for name in sys.modules
            if name.startswith("kaos_content.dedup.levels")
        ]:
            del sys.modules[mod]
