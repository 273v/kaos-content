"""kaos-content AST model types."""

from kaos_content.model.annotation import Annotation, AnnotationTarget, AnnotationType
from kaos_content.model.attr import (
    Alignment,
    Attr,
    BoundingBox,
    Caption,
    ColSpec,
    CoordOrigin,
    Provenance,
    SourceRef,
)
from kaos_content.model.blocks import (
    Admonition,
    Block,
    BlockQuote,
    BulletList,
    CodeBlock,
    DefinitionItem,
    DefinitionList,
    Div,
    Figure,
    Heading,
    ListItem,
    MathBlock,
    OrderedList,
    PageBreak,
    Paragraph,
    RawBlock,
    Table,
    ThematicBreak,
)
from kaos_content.model.document import ContentDocument
from kaos_content.model.extraction import (
    CellStatus,
    ExtractionCell,
    ExtractionCitation,
    ExtractionError,
    ExtractionErrorCode,
    normalize_bbox,
)
from kaos_content.model.inlines import (
    Citation,
    Code,
    Emphasis,
    FootnoteRef,
    Image,
    Inline,
    LineBreak,
    Link,
    Math,
    RawInline,
    SoftBreak,
    Span,
    Strikethrough,
    Strong,
    Subscript,
    Superscript,
    Text,
    Underline,
)
from kaos_content.model.metadata import DocumentMetadata
from kaos_content.model.node import BaseBlock, BaseInline, BaseNode
from kaos_content.model.table import (
    Cell,
    Row,
    TableSection,
)
from kaos_content.model.tabular import (
    Column,
    ColumnType,
    TabularDocument,
    column_type_from_python,
    infer_column_type,
)
from kaos_content.model.tabular import (
    Table as TabularTable,
)

# Rebuild models with cross-module forward references now that all types are imported.
# Caption references Block/Inline from attr.py via TYPE_CHECKING.
# Cell references Block from table.py via TYPE_CHECKING.
# Block types that contain list[Block] children also need rebuilding.
Caption.model_rebuild()
Cell.model_rebuild()
Row.model_rebuild()
TableSection.model_rebuild()
BlockQuote.model_rebuild()
OrderedList.model_rebuild()
BulletList.model_rebuild()
ListItem.model_rebuild()
DefinitionList.model_rebuild()
DefinitionItem.model_rebuild()
Table.model_rebuild()
Figure.model_rebuild()
Div.model_rebuild()
Admonition.model_rebuild()
ContentDocument.model_rebuild()

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
    "CellStatus",
    "Citation",
    "Code",
    "CodeBlock",
    "ColSpec",
    "Column",
    "ColumnType",
    "ContentDocument",
    "CoordOrigin",
    "DefinitionItem",
    "DefinitionList",
    "Div",
    "DocumentMetadata",
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
    "LineBreak",
    "Link",
    "ListItem",
    "Math",
    "MathBlock",
    "OrderedList",
    "PageBreak",
    "Paragraph",
    "Provenance",
    "RawBlock",
    "RawInline",
    "Row",
    "SoftBreak",
    "SourceRef",
    "Span",
    "Strikethrough",
    "Strong",
    "Subscript",
    "Superscript",
    "Table",
    "TableSection",
    "TabularDocument",
    "TabularTable",
    "Text",
    "ThematicBreak",
    "Underline",
    "column_type_from_python",
    "infer_column_type",
    "normalize_bbox",
]
