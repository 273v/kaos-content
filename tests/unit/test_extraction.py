"""Unit tests for extraction primitives — ExtractionCell, ExtractionCitation,
ExtractionError, normalize_bbox, and TabularDocument.from_cells.

Covers WS-TR.PR-0 acceptance: status invariants, reviewer overlay, citation
shape, bbox normalization, ColumnType extensions, AnnotationType extension,
and TabularDocument.from_cells pivot semantics.
"""

from __future__ import annotations

import datetime
import hashlib
from typing import Any, cast

import pytest
from pydantic import ValidationError

from kaos_content import (
    Annotation,
    AnnotationTarget,
    AnnotationType,
    BoundingBox,
    ColumnType,
    ExtractionCell,
    ExtractionCitation,
    ExtractionError,
    TabularDocument,
    normalize_bbox,
)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _good_citation(
    block_ref: str = "#/body/0",
    snippet: str = "the parties hereto",
    page: int = 1,
    confidence: float = 0.95,
    parent_block_ref: str | None = None,
) -> ExtractionCitation:
    return ExtractionCitation(
        block_ref=block_ref,
        page=page,
        bbox=None,
        char_span=(0, len(snippet)),
        snippet=snippet,
        snippet_sha256=_hash(snippet),
        confidence=confidence,
        parent_block_ref=parent_block_ref,
    )


class TestExtractionCitation:
    def test_minimal(self) -> None:
        cit = _good_citation()
        assert cit.block_ref == "#/body/0"
        assert cit.page == 1
        assert cit.confidence == 0.95
        assert len(cit.snippet_sha256) == 64

    def test_page_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            ExtractionCitation(
                block_ref="#/body/0",
                page=0,
                char_span=(0, 5),
                snippet="hello",
                snippet_sha256=_hash("hello"),
                confidence=1.0,
            )

    def test_confidence_bounded(self) -> None:
        for bad in (-0.01, 1.01, 1.5):
            with pytest.raises(ValidationError):
                ExtractionCitation(
                    block_ref="#/body/0",
                    page=1,
                    char_span=(0, 5),
                    snippet="hello",
                    snippet_sha256=_hash("hello"),
                    confidence=bad,
                )

    def test_sha256_must_be_64_chars(self) -> None:
        with pytest.raises(ValidationError):
            ExtractionCitation(
                block_ref="#/body/0",
                page=1,
                char_span=(0, 5),
                snippet="hello",
                snippet_sha256="too_short",
                confidence=1.0,
            )

    def test_with_bbox_and_parent(self) -> None:
        bbox = BoundingBox(left=10, top=20, right=110, bottom=40)
        cit = _good_citation(parent_block_ref="#/body/0").model_copy(update={"bbox": bbox})
        assert cit.bbox is not None
        assert cit.bbox.left == 10
        assert cit.parent_block_ref == "#/body/0"

    def test_round_trip(self) -> None:
        cit = _good_citation(parent_block_ref="#/body/0/children/0")
        restored = ExtractionCitation.model_validate_json(cit.model_dump_json())
        assert restored == cit


class TestExtractionError:
    def test_minimal(self) -> None:
        err = ExtractionError(code="model_api_error", message="500 from provider")
        assert err.attempt == 1
        assert err.retry_recommended is False

    def test_attempt_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            ExtractionError(code="other", message="x", attempt=0)

    def test_unknown_code_rejected(self) -> None:
        # cast bypasses static-check; the runtime should still reject via Pydantic.
        with pytest.raises(ValidationError):
            ExtractionError(code=cast(Any, "something_undocumented"), message="x")


class TestExtractionCellExtractedStatus:
    def test_extracted_requires_value_and_confidence(self) -> None:
        with pytest.raises(ValidationError, match="ai_value"):
            ExtractionCell(
                doc_id="doc-1",
                column_id="effective_date",
                schema_version=1,
                status="extracted",
                ai_value=None,
                confidence=0.9,
            )
        with pytest.raises(ValidationError, match="confidence"):
            ExtractionCell(
                doc_id="doc-1",
                column_id="effective_date",
                schema_version=1,
                status="extracted",
                ai_value="2024-01-01",
                confidence=None,
            )

    def test_extracted_happy_path(self) -> None:
        cit = _good_citation()
        cell = ExtractionCell(
            doc_id="doc-1",
            column_id="effective_date",
            schema_version=1,
            status="extracted",
            ai_value="2024-01-01",
            rationale="The contract states 'as of January 1, 2024'.",
            citations=(cit,),
            confidence=0.92,
            model="anthropic:claude-haiku-4-5",
            cost_usd=0.0001,
            tokens_total=512,
            latency_ms=850.0,
        )
        assert cell.ai_value == "2024-01-01"
        assert len(cell.citations) == 1
        assert cell.cost_usd == pytest.approx(0.0001)


class TestExtractionCellRefusalStatuses:
    def test_not_in_document_clean(self) -> None:
        cell = ExtractionCell(
            doc_id="doc-1",
            column_id="termination_for_convenience",
            schema_version=1,
            status="not_in_document",
            rationale="No termination-for-convenience clause was found.",
        )
        assert cell.ai_value is None
        assert cell.citations == ()

    def test_not_in_document_rejects_value(self) -> None:
        with pytest.raises(ValidationError, match="ai_value"):
            ExtractionCell(
                doc_id="doc-1",
                column_id="x",
                schema_version=1,
                status="not_in_document",
                ai_value="something",
            )

    def test_not_in_document_rejects_citations(self) -> None:
        with pytest.raises(ValidationError, match="citations"):
            ExtractionCell(
                doc_id="doc-1",
                column_id="x",
                schema_version=1,
                status="not_in_document",
                citations=(_good_citation(),),
            )

    def test_unclear_requires_confidence(self) -> None:
        with pytest.raises(ValidationError, match="confidence"):
            ExtractionCell(
                doc_id="doc-1",
                column_id="x",
                schema_version=1,
                status="unclear",
                ai_value="maybe",
                confidence=None,
            )

    def test_unclear_happy_path(self) -> None:
        cell = ExtractionCell(
            doc_id="doc-1",
            column_id="governing_law",
            schema_version=1,
            status="unclear",
            ai_value="Delaware",
            confidence=0.4,
            considerations="Document says 'New York' once and 'Delaware' twice.",
        )
        assert cell.confidence == 0.4

    def test_error_requires_error_record(self) -> None:
        with pytest.raises(ValidationError, match="error record"):
            ExtractionCell(
                doc_id="doc-1",
                column_id="x",
                schema_version=1,
                status="error",
                error=None,
            )

    def test_error_rejects_value(self) -> None:
        with pytest.raises(ValidationError, match="ai_value"):
            ExtractionCell(
                doc_id="doc-1",
                column_id="x",
                schema_version=1,
                status="error",
                ai_value="oops",
                error=ExtractionError(code="timeout", message="exceeded 60s"),
            )

    def test_error_happy_path(self) -> None:
        cell = ExtractionCell(
            doc_id="doc-1",
            column_id="x",
            schema_version=1,
            status="error",
            error=ExtractionError(
                code="model_api_error",
                message="provider returned 503",
                retry_recommended=True,
                attempt=1,
            ),
        )
        assert cell.error is not None
        assert cell.error.retry_recommended is True


class TestExtractionCellReviewerOverlay:
    def test_reviewer_value_does_not_overwrite_ai(self) -> None:
        cell = ExtractionCell(
            doc_id="doc-1",
            column_id="effective_date",
            schema_version=1,
            status="extracted",
            ai_value="2024-01-01",
            confidence=0.9,
            reviewed_value="2024-01-15",
            reviewer_id="reviewer-7",
            reviewed_at=datetime.datetime(2026, 4, 14, 12, 0),
            review_note="Original date was the signing date; effective date is 2 weeks later.",
        )
        assert cell.ai_value == "2024-01-01"
        assert cell.reviewed_value == "2024-01-15"
        assert cell.reviewer_id == "reviewer-7"

    def test_reviewer_value_requires_reviewer_id_and_timestamp(self) -> None:
        with pytest.raises(ValidationError, match="reviewer overlay"):
            ExtractionCell(
                doc_id="doc-1",
                column_id="x",
                schema_version=1,
                status="not_in_document",
                reviewed_value="something",
            )

    def test_review_note_alone_requires_reviewer_metadata(self) -> None:
        with pytest.raises(ValidationError, match="reviewer overlay"):
            ExtractionCell(
                doc_id="doc-1",
                column_id="x",
                schema_version=1,
                status="not_in_document",
                review_note="Note without metadata",
            )


class TestExtractionCellRoundTrip:
    def test_full_cell_round_trip(self) -> None:
        cit = _good_citation(parent_block_ref="#/body/0")
        cell = ExtractionCell(
            doc_id="doc-1",
            column_id="parties",
            schema_version=2,
            status="extracted",
            ai_value=["Acme Corp", "Beta LLC"],
            rationale="The preamble names two parties.",
            considerations="No third party is referenced elsewhere.",
            citations=(cit,),
            confidence=0.97,
            flags=("verified",),
            model="anthropic:claude-haiku-4-5",
            cost_usd=0.0002,
            tokens_total=900,
            latency_ms=1200.5,
        )
        restored = ExtractionCell.model_validate_json(cell.model_dump_json())
        assert restored == cell


class TestNormalizeBbox:
    def test_basic_normalization(self) -> None:
        bbox = BoundingBox(left=100, top=50, right=300, bottom=150)
        out = normalize_bbox(bbox, page_width=1000, page_height=500)
        assert out["left"] == pytest.approx(0.1)
        assert out["top"] == pytest.approx(0.1)
        assert out["width"] == pytest.approx(0.2)
        assert out["height"] == pytest.approx(0.2)

    def test_full_page_box(self) -> None:
        bbox = BoundingBox(left=0, top=0, right=800, bottom=600)
        out = normalize_bbox(bbox, page_width=800, page_height=600)
        assert out == {"left": 0.0, "top": 0.0, "width": 1.0, "height": 1.0}

    def test_invalid_dimensions(self) -> None:
        bbox = BoundingBox(left=0, top=0, right=10, bottom=10)
        with pytest.raises(ValueError, match="positive"):
            normalize_bbox(bbox, page_width=0, page_height=100)
        with pytest.raises(ValueError, match="positive"):
            normalize_bbox(bbox, page_width=100, page_height=-1)


class TestColumnTypeExtensions:
    def test_extraction_types_present(self) -> None:
        assert ColumnType.VERBATIM_QUOTE.value == "verbatim_quote"
        assert ColumnType.MONEY.value == "money"
        assert ColumnType.SCORE.value == "score"
        assert ColumnType.ENTITY_ROLE.value == "entity_role"

    def test_total_count(self) -> None:
        # Tier 1 (8) + Tier 2 (3) + Tier 3 (2) + Tier 4 (4) = 17
        assert len(list(ColumnType)) == 17

    def test_legacy_types_preserved(self) -> None:
        # Spot-check a few existing types still work — backwards-compat.
        assert ColumnType.TEXT.value == "text"
        assert ColumnType.DECIMAL.value == "decimal"
        assert ColumnType.STRUCT.value == "struct"


class TestAnnotationTypeExtraction:
    def test_extracted_cell_type_present(self) -> None:
        assert AnnotationType.EXTRACTED_CELL.value == "extracted_cell"

    def test_attach_extracted_cell_annotation(self) -> None:
        annot = Annotation(
            id="ann-1",
            type=AnnotationType.EXTRACTED_CELL,
            targets=(AnnotationTarget(node_ref="#/body/0", start_offset=0, end_offset=18),),
            body={
                "column_id": "parties",
                "schema_version": 1,
                "cell_ref": "doc-1::parties",
                "doc_id": "doc-1",
            },
        )
        assert annot.type == AnnotationType.EXTRACTED_CELL
        assert annot.body["column_id"] == "parties"


class TestTabularDocumentFromCells:
    def _cell(
        self,
        doc_id: str,
        column_id: str,
        value: object,
        *,
        status: str = "extracted",
        confidence: float = 0.9,
    ) -> ExtractionCell:
        return ExtractionCell(
            doc_id=doc_id,
            column_id=column_id,
            schema_version=1,
            status=cast(Any, status),
            ai_value=value if status == "extracted" else None,
            confidence=confidence if status in ("extracted", "unclear") else None,
        )

    def test_pivots_into_one_row_per_doc(self) -> None:
        cells = [
            self._cell("doc-1", "effective_date", "2024-01-01"),
            self._cell("doc-1", "governing_law", "Delaware"),
            self._cell("doc-2", "effective_date", "2024-02-15"),
            self._cell("doc-2", "governing_law", "New York"),
        ]
        td = TabularDocument.from_cells(
            cells,
            column_specs=(
                ("effective_date", ColumnType.DATE),
                ("governing_law", ColumnType.TEXT),
            ),
            table_name="contracts",
        )
        assert len(td.tables) == 1
        table = td.tables[0]
        assert table.name == "contracts"
        assert table.column_names() == ("doc_id", "effective_date", "governing_law")
        assert table.row_count == 2
        assert table.rows[0] == ("doc-1", "2024-01-01", "Delaware")
        assert table.rows[1] == ("doc-2", "2024-02-15", "New York")

    def test_refusal_becomes_none(self) -> None:
        cells = [
            self._cell("doc-1", "effective_date", "2024-01-01"),
            self._cell("doc-1", "termination", None, status="not_in_document", confidence=0.0),
        ]
        td = TabularDocument.from_cells(
            cells,
            column_specs=(
                ("effective_date", ColumnType.DATE),
                ("termination", ColumnType.TEXT),
            ),
        )
        assert td.tables[0].rows[0] == ("doc-1", "2024-01-01", None)

    def test_reviewed_value_overrides_ai_value(self) -> None:
        cit = _good_citation()
        cell = ExtractionCell(
            doc_id="doc-1",
            column_id="effective_date",
            schema_version=1,
            status="extracted",
            ai_value="2024-01-01",
            confidence=0.9,
            citations=(cit,),
            reviewed_value="2024-01-15",
            reviewer_id="r-1",
            reviewed_at=datetime.datetime(2026, 4, 14),
        )
        td = TabularDocument.from_cells(
            [cell],
            column_specs=(("effective_date", ColumnType.DATE),),
        )
        assert td.tables[0].rows[0] == ("doc-1", "2024-01-15")

    def test_missing_cells_are_none(self) -> None:
        # doc-1 has both columns; doc-2 only has one.
        cells = [
            self._cell("doc-1", "a", "v1"),
            self._cell("doc-1", "b", "v2"),
            self._cell("doc-2", "a", "v3"),
        ]
        td = TabularDocument.from_cells(
            cells,
            column_specs=(("a", ColumnType.TEXT), ("b", ColumnType.TEXT)),
        )
        rows = sorted(td.tables[0].rows)
        assert rows == [("doc-1", "v1", "v2"), ("doc-2", "v3", None)]

    def test_empty_column_specs_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            TabularDocument.from_cells([], column_specs=())

    def test_duplicate_column_specs_rejected(self) -> None:
        with pytest.raises(ValueError, match="Duplicate"):
            TabularDocument.from_cells(
                [],
                column_specs=(("a", ColumnType.TEXT), ("a", ColumnType.INTEGER)),
            )

    def test_non_extraction_cell_rejected(self) -> None:
        with pytest.raises(TypeError, match="ExtractionCell"):
            TabularDocument.from_cells(
                [{"doc_id": "x"}],  # type: ignore[list-item]
                column_specs=(("a", ColumnType.TEXT),),
            )

    def test_extraction_column_types_preserved(self) -> None:
        cells = [
            self._cell("doc-1", "deal_value", {"amount": "1000000.00", "currency": "USD"}),
            self._cell("doc-1", "verbatim_clause", "as of the date first written above"),
            self._cell("doc-1", "relevance_score", 4),
        ]
        td = TabularDocument.from_cells(
            cells,
            column_specs=(
                ("deal_value", ColumnType.MONEY),
                ("verbatim_clause", ColumnType.VERBATIM_QUOTE),
                ("relevance_score", ColumnType.SCORE),
            ),
        )
        types = {c.name: c.column_type for c in td.tables[0].columns}
        assert types["deal_value"] == ColumnType.MONEY
        assert types["verbatim_clause"] == ColumnType.VERBATIM_QUOTE
        assert types["relevance_score"] == ColumnType.SCORE


class TestExtractionPolarsBridge:
    """Polars bridge accepts the new Tier-4 ColumnTypes without KeyError."""

    def test_extraction_table_to_polars(self) -> None:
        pl = pytest.importorskip("polars")
        from kaos_content.bridges.polars import table_to_polars

        cells = [
            ExtractionCell(
                doc_id="doc-1",
                column_id="party",
                schema_version=1,
                status="extracted",
                ai_value={"name": "Acme Corp", "role": "buyer", "entity_type": None},
                confidence=0.95,
            ),
            ExtractionCell(
                doc_id="doc-1",
                column_id="quote",
                schema_version=1,
                status="extracted",
                ai_value="as of the date first written above",
                confidence=0.99,
            ),
            ExtractionCell(
                doc_id="doc-1",
                column_id="score",
                schema_version=1,
                status="extracted",
                ai_value=4,
                confidence=0.85,
            ),
        ]
        td = TabularDocument.from_cells(
            cells,
            column_specs=(
                ("party", ColumnType.ENTITY_ROLE),
                ("quote", ColumnType.VERBATIM_QUOTE),
                ("score", ColumnType.SCORE),
            ),
        )
        df = table_to_polars(td.tables[0])
        # MONEY/ENTITY_ROLE serialize to JSON strings; SCORE stays Int64.
        assert df.schema["party"] == pl.String
        assert df.schema["quote"] == pl.String
        assert df.schema["score"] == pl.Int64
        assert df.row(0)[df.columns.index("score")] == 4
