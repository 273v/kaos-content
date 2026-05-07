"""Hierarchical document views: pages, sections, paragraphs, sentences.

Computed dynamically from the flat ContentDocument AST. Adapts to what
the document actually contains (pages, headings, etc.).
"""

from kaos_content.views.document_view import DocumentView
from kaos_content.views.models import PageView, ParagraphView, SectionView, SentenceView
from kaos_content.views.tabular_view import ColumnStats, TableInfo, TabularView

__all__ = [
    "ColumnStats",
    "DocumentView",
    "PageView",
    "ParagraphView",
    "SectionView",
    "SentenceView",
    "TableInfo",
    "TabularView",
]
