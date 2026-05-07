"""kaos-content: Abstract document AST for KAOS."""

from kaos_content._version import __version__
from kaos_content.artifacts import (
    document_annotations_by_type,
    document_definitions,
    document_metadata,
    document_node_subtree,
    document_outline,
    document_tables_summary,
    document_to_resource_views,
    document_to_summary,
    load_document,
    load_tabular,
    store_document,
    store_tabular,
    tabular_schema,
    tabular_summary,
)
from kaos_content.builders import DocumentBuilder
from kaos_content.chunking import SectionChunker
from kaos_content.corpus import ContentDocumentCorpus, ContentPassage, Corpus, Passage
from kaos_content.dedup import (
    DedupCluster,
    DedupDocument,
    DedupLevel,
    DedupPipeline,
    DedupReport,
)
from kaos_content.errors import KaosContentError, SearchError, SerializationError
from kaos_content.model import (
    Admonition,
    Alignment,
    Annotation,
    AnnotationTarget,
    AnnotationType,
    Attr,
    BaseBlock,
    BaseInline,
    BaseNode,
    Block,
    BlockQuote,
    BoundingBox,
    BulletList,
    Caption,
    Cell,
    Citation,
    Code,
    CodeBlock,
    ColSpec,
    Column,
    ColumnType,
    ContentDocument,
    CoordOrigin,
    DefinitionItem,
    DefinitionList,
    Div,
    DocumentMetadata,
    Emphasis,
    ExtractionCell,
    ExtractionCitation,
    ExtractionError,
    ExtractionErrorCode,
    Figure,
    FootnoteRef,
    Heading,
    Image,
    Inline,
    LineBreak,
    Link,
    ListItem,
    Math,
    MathBlock,
    OrderedList,
    PageBreak,
    Paragraph,
    Provenance,
    RawBlock,
    RawInline,
    Row,
    SoftBreak,
    SourceRef,
    Span,
    Strikethrough,
    Strong,
    Subscript,
    Superscript,
    Table,
    TableSection,
    TabularDocument,
    TabularTable,
    Text,
    ThematicBreak,
    Underline,
    column_type_from_python,
    infer_column_type,
    normalize_bbox,
)
from kaos_content.normalize import (
    canonical_hash,
    normalize_date,
    normalize_entity,
    normalize_number,
    normalize_text,
)
from kaos_content.parsers import parse_plain_text
from kaos_content.revision import Revision, Revisions, RevisionType
from kaos_content.search import SearchResult, SearchResults, search_document, search_tabular
from kaos_content.serializers import (
    serialize_csv,
    serialize_html,
    serialize_json_records,
    serialize_markdown,
    serialize_markdown_table,
    serialize_tabular_markdown,
    serialize_tabular_summary,
    serialize_text,
    serialize_tsv,
)
from kaos_content.tools import register_content_tools
from kaos_content.transforms import DocumentTransform, apply, compose
from kaos_content.traversal import (
    NodeIndex,
    content_hash,
    extract_text,
    find,
    find_annotations_of_type,
    find_by_class,
    find_by_kv,
    find_by_type,
    find_first,
    find_footnote_refs,
    find_headings,
    find_images,
    find_links,
    find_tables,
    walk,
    walk_blocks,
    walk_inlines,
)
from kaos_content.units import (
    ParagraphUnit,
    SentenceUnit,
    iter_paragraph_units,
    iter_sentence_units,
)
from kaos_content.views import (
    ColumnStats,
    DocumentView,
    PageView,
    ParagraphView,
    SectionView,
    SentenceView,
    TableInfo,
    TabularView,
)

# ─── Conditional imports for optional extras ─────────────────────────────
#
# Each block catches `ImportError` (not `ModuleNotFoundError`) because
# the missing-extra case manifests differently per submodule:
#
#   - [layout]: numpy missing → `ModuleNotFoundError("No module named 'numpy'")`
#   - [markdown]: markdown-it-py missing → `kaos_content.parsers/__init__.py`
#     uses `contextlib.suppress(ImportError)` to skip binding `parse_markdown`,
#     so re-importing it here raises `ImportError: cannot import name
#     'parse_markdown'` — that is **not** a ModuleNotFoundError.
#   - [nlp]: chained import error from `kaos-nlp-core` or its transitive deps.
#
# Since `ModuleNotFoundError` is a subclass of `ImportError`, catching the
# parent covers all three. (A 2026-05-07 attempt to tighten to
# `ModuleNotFoundError` broke the min-deps lane — see git log.)
#
# Names imported here are re-exported via __all__ below — see the
# `if _<X>_AVAILABLE:` extends right before the bottom of the file.

# [layout] extra requires numpy.
_LAYOUT_AVAILABLE = False
try:
    from kaos_content.layout import (
        BreaksResult,  # noqa: F401
        ClusterResult,  # noqa: F401
        ColumnResult,  # noqa: F401
        FontSizeClassification,  # noqa: F401
        LineGroup,  # noqa: F401
        ModeResult,  # noqa: F401
        TextBlock,  # noqa: F401
        ThresholdResult,  # noqa: F401
        Valley,  # noqa: F401
        classify_font_sizes,  # noqa: F401
        cluster_1d,  # noqa: F401
        detect_columns,  # noqa: F401
        detect_headers_footers,  # noqa: F401
        detect_paragraph_breaks,  # noqa: F401
        detect_table_regions,  # noqa: F401
        find_modes,  # noqa: F401
        find_valleys,  # noqa: F401
        find_widest_valley,  # noqa: F401
        group_into_lines,  # noqa: F401
        jenks_breaks,  # noqa: F401
        otsu_threshold,  # noqa: F401
        projection_profile,  # noqa: F401
        xy_cut,  # noqa: F401
    )

    _LAYOUT_AVAILABLE = True
except ImportError:
    from kaos_core.logging import get_logger as _get_logger

    _get_logger(__name__).debug(
        "kaos-content[layout] not available (requires numpy). "
        "Install with: pip install kaos-content[layout]"
    )

# [markdown] extra (markdown-it-py).
_MARKDOWN_AVAILABLE = False
try:
    from kaos_content.parsers import parse_markdown  # noqa: F401

    _MARKDOWN_AVAILABLE = True
except ImportError:
    pass

# [nlp] extra — SearchableDocument needs kaos-nlp-core.
_INDEXING_AVAILABLE = False
try:
    from kaos_content.indexing import SearchableDocument  # noqa: F401

    _INDEXING_AVAILABLE = True
except ImportError:
    pass

# Always-available exports. The conditional extras (layout/numpy,
# markdown/markdown-it-py, nlp/kaos-nlp-core) append to __all__ only when
# their imports succeeded — see the bottom of this module. Without that,
# `from kaos_content import *` would lie about what's actually importable.
__all__ = [
    "Admonition",
    "Alignment",
    "Annotation",
    "AnnotationTarget",
    "AnnotationType",
    "Attr",
    "BaseBlock",
    "BaseInline",
    "BaseNode",
    "Block",
    "BlockQuote",
    "BoundingBox",
    "BulletList",
    "Caption",
    "Cell",
    "Citation",
    "Code",
    "CodeBlock",
    "ColSpec",
    "Column",
    "ColumnStats",
    "ColumnType",
    "ContentDocument",
    "ContentDocumentCorpus",
    "ContentPassage",
    "CoordOrigin",
    "Corpus",
    "DedupCluster",
    "DedupDocument",
    "DedupLevel",
    "DedupPipeline",
    "DedupReport",
    "DefinitionItem",
    "DefinitionList",
    "Div",
    "DocumentBuilder",
    "DocumentMetadata",
    "DocumentTransform",
    "DocumentView",
    "Emphasis",
    "ExtractionCell",
    "ExtractionCitation",
    "ExtractionError",
    "ExtractionErrorCode",
    "Figure",
    "FootnoteRef",
    "Heading",
    "Image",
    "Inline",
    "KaosContentError",
    "LineBreak",
    "Link",
    "ListItem",
    "Math",
    "MathBlock",
    "NodeIndex",
    "OrderedList",
    "PageBreak",
    "PageView",
    "Paragraph",
    "ParagraphUnit",
    "ParagraphView",
    "Passage",
    "Provenance",
    "RawBlock",
    "RawInline",
    "Revision",
    "RevisionType",
    "Revisions",
    "Row",
    "SearchError",
    "SearchResult",
    "SearchResults",
    "SectionChunker",
    "SectionView",
    "SentenceUnit",
    "SentenceView",
    "SerializationError",
    "SoftBreak",
    "SourceRef",
    "Span",
    "Strikethrough",
    "Strong",
    "Subscript",
    "Superscript",
    "Table",
    "TableInfo",
    "TableSection",
    "TabularDocument",
    "TabularTable",
    "TabularView",
    "Text",
    "ThematicBreak",
    "Underline",
    "__version__",
    "apply",
    "canonical_hash",
    "column_type_from_python",
    "compose",
    "content_hash",
    "document_annotations_by_type",
    "document_definitions",
    "document_metadata",
    "document_node_subtree",
    "document_outline",
    "document_tables_summary",
    "document_to_resource_views",
    "document_to_summary",
    "extract_text",
    "find",
    "find_annotations_of_type",
    "find_by_class",
    "find_by_kv",
    "find_by_type",
    "find_first",
    "find_footnote_refs",
    "find_headings",
    "find_images",
    "find_links",
    "find_tables",
    "infer_column_type",
    "iter_paragraph_units",
    "iter_sentence_units",
    "load_document",
    "load_tabular",
    "normalize_bbox",
    "normalize_date",
    "normalize_entity",
    "normalize_number",
    "normalize_text",
    "parse_plain_text",
    "register_content_tools",
    "search_document",
    "search_tabular",
    "serialize_csv",
    "serialize_html",
    "serialize_json_records",
    "serialize_markdown",
    "serialize_markdown_table",
    "serialize_tabular_markdown",
    "serialize_tabular_summary",
    "serialize_text",
    "serialize_tsv",
    "store_document",
    "store_tabular",
    "tabular_schema",
    "tabular_summary",
    "walk",
    "walk_blocks",
    "walk_inlines",
]

# [layout] extra — append only if numpy was importable above.
if _LAYOUT_AVAILABLE:
    __all__.extend(
        [
            "BreaksResult",
            "ClusterResult",
            "ColumnResult",
            "FontSizeClassification",
            "LineGroup",
            "ModeResult",
            "TextBlock",
            "ThresholdResult",
            "Valley",
            "classify_font_sizes",
            "cluster_1d",
            "detect_columns",
            "detect_headers_footers",
            "detect_paragraph_breaks",
            "detect_table_regions",
            "find_modes",
            "find_valleys",
            "find_widest_valley",
            "group_into_lines",
            "jenks_breaks",
            "otsu_threshold",
            "projection_profile",
            "xy_cut",
        ]
    )

# [markdown] extra — append only if markdown-it-py was importable.
if _MARKDOWN_AVAILABLE:
    __all__.append("parse_markdown")

# [nlp] extra — SearchableDocument needs kaos-nlp-core.
if _INDEXING_AVAILABLE:
    __all__.append("SearchableDocument")
