"""
app/comparison/excel.py

Renders a ComparisonResult into a five-sheet Excel workbook, in memory.

    1. Executive Summary            plain-language scorecard + a reading
                                     guide explaining every term used
                                     later in the workbook, once
    2. Findings                     the ranked, filterable working list,
                                     in plain language - no M+/M-,
                                     no raw document codes
    3. Category Overview            where the job stands, dimension by
                                     dimension
    4. Appendix A - Full Comparison every document side by side, verbatim
    5. Appendix B - Audit Detail    every concept, every count, every
                                     document code, every degradation

Sheets 1-3 are what a business reader opens. Sheets 4-5 are labelled
"Appendix" on purpose: nothing on them is required reading, but nothing
that traces a sheet-1 number back to a named document is removed either
- it just no longer sits in the way of a first read. Same results, same
traceability, less to read to get the decision.
"""

from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from typing import Any, Dict, List, Optional, Sequence

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.worksheet import Worksheet

from app.comparison_engine.extract import CATEGORY_ACCESSOR, document_label
from app.comparison_engine.models import (
    BUCKET_EXPLAINER,
    BUCKET_LABEL,
    CATEGORY_ORDER,
    BenchmarkStrength,
    Bucket,
    ComparisonResult,
    ComparisonStatusV2,
    Direction,
)

# ==========================================================================
# Design tokens
# ==========================================================================

FONT_NAME = "Calibri"

NAVY = "1F3864"
SLATE = "44546A"
LIGHT = "EEF2F8"
BORDER_GREY = "BFBFBF"

BUCKET_COLOR: Dict[Bucket, str] = {
    Bucket.CRITICAL_GAP: "C00000",
    Bucket.NOTABLE_GAP: "C55A11",
    Bucket.STANDARD_GAP: "2E75B6",
    Bucket.ADDITIONAL: "7030A0",
    Bucket.MARKET_VARIATION: "7F7F7F",
    Bucket.ALIGNED: "548235",
}

BUCKET_FILL: Dict[Bucket, str] = {
    Bucket.CRITICAL_GAP: "FCE4E4",
    Bucket.NOTABLE_GAP: "FDEADA",
    Bucket.STANDARD_GAP: "E4EEF8",
    Bucket.ADDITIONAL: "F0E6F6",
    Bucket.MARKET_VARIATION: "F2F2F2",
    Bucket.ALIGNED: "E8F2E2",
}

NOT_SPECIFIED = "Not specified"

_THIN = Side(style="thin", color=BORDER_GREY)
BORDER = Border(left=_THIN, right=_THIN, top=_THIN, bottom=_THIN)

WRAP_TOP = Alignment(horizontal="left", vertical="top", wrap_text=True)
CENTER = Alignment(horizontal="center", vertical="center", wrap_text=True)
LEFT_MID = Alignment(horizontal="left", vertical="center", wrap_text=True)


def _font(
    size: int = 11,
    bold: bool = False,
    color: str = "000000",
    italic: bool = False,
) -> Font:
    return Font(
        name=FONT_NAME, size=size, bold=bold, color=color, italic=italic
    )


def _fill(color: str) -> PatternFill:
    return PatternFill(fill_type="solid", fgColor=color)


def _set(
    sheet: Worksheet,
    row: int,
    column: int,
    value: Any,
    font: Optional[Font] = None,
    fill: Optional[str] = None,
    alignment: Optional[Alignment] = None,
    border: bool = True,
):
    cell = sheet.cell(row=row, column=column, value=value)
    cell.font = font or _font()
    if fill:
        cell.fill = _fill(fill)
    cell.alignment = alignment or WRAP_TOP
    if border:
        cell.border = BORDER
    return cell


def _header_row(
    sheet: Worksheet,
    row: int,
    headers: Sequence[str],
    start_column: int = 1,
) -> None:
    for offset, title in enumerate(headers):
        _set(
            sheet,
            row,
            start_column + offset,
            title,
            font=_font(bold=True, color="FFFFFF"),
            fill=NAVY,
            alignment=CENTER,
        )
    sheet.row_dimensions[row].height = 28


def _widths(sheet: Worksheet, widths: Sequence[float]) -> None:
    for index, width in enumerate(widths, start=1):
        sheet.column_dimensions[get_column_letter(index)].width = width


def _autofit_height(
    sheet: Worksheet,
    row: int,
    columns: int,
    per_line: int = 15,
    minimum: int = 20,
    maximum: int = 160,
) -> None:
    lines = 1
    for column in range(1, columns + 1):
        value = sheet.cell(row=row, column=column).value
        if value is None:
            continue
        text = str(value)
        lines = max(lines, text.count("\n") + 1)
    sheet.row_dimensions[row].height = max(
        minimum, min(maximum, per_line * lines + 6)
    )


def _bullets(values: Sequence[str], limit: int = 6) -> str:
    cleaned = [str(v).strip() for v in values if v and str(v).strip()]
    if not cleaned:
        return NOT_SPECIFIED
    if len(cleaned) == 1:
        return cleaned[0]
    shown = cleaned[:limit]
    text = "\n".join(f"• {v}" for v in shown)
    if len(cleaned) > limit:
        text += f"\n… and {len(cleaned) - limit} more"
    return text


def _percent(value: Optional[float]) -> str:
    return "—" if value is None else f"{value * 100:.0f}%"


def _doc_list(indexes: Sequence[int]) -> str:
    return ", ".join(f"D{i}" for i in indexes) if indexes else "—"


# ==========================================================================
# SHEET 1 — Executive Summary
# ==========================================================================


def _sheet_summary(workbook: Workbook, result: ComparisonResult) -> None:
    sheet = workbook.active
    sheet.title = "Executive Summary"
    sheet.sheet_view.showGridLines = False
    _widths(sheet, [30, 26, 26, 26, 30, 30])

    # --- Title band ----------------------------------------------------
    sheet.merge_cells("A1:F2")
    title = sheet.cell(row=1, column=1, value="Job Description Benchmark Report")
    title.font = _font(size=18, bold=True, color="FFFFFF")
    title.fill = _fill(NAVY)
    title.alignment = Alignment(
        horizontal="left", vertical="center", indent=1
    )
    sheet.row_dimensions[1].height = 26
    sheet.row_dimensions[2].height = 20

    generated = datetime.now(timezone.utc).strftime("%d %b %Y, %H:%M UTC")

    facts = [
        ("Job code", result.new_job_code or NOT_SPECIFIED),
        ("Job title", result.new_job_title or NOT_SPECIFIED),
        ("Company", result.new_company_code or NOT_SPECIFIED),
        ("Benchmark documents", str(result.reference_jobs_count)),
        ("Generated", generated),
    ]

    row = 4
    for label, value in facts:
        _set(
            sheet, row, 1, label,
            font=_font(bold=True), fill=LIGHT, alignment=LEFT_MID,
        )
        sheet.merge_cells(
            start_row=row, start_column=2, end_row=row, end_column=3
        )
        _set(sheet, row, 2, value, alignment=LEFT_MID)
        for column in (3,):
            sheet.cell(row=row, column=column).border = BORDER
        sheet.row_dimensions[row].height = 20
        row += 1

    # --- New job code, nothing to compare ------------------------------
    if result.status == ComparisonStatusV2.NEW_JOB_CODE:
        row += 1
        sheet.merge_cells(
            start_row=row, start_column=1, end_row=row + 2, end_column=6
        )
        _set(
            sheet, row, 1, result.overall_summary,
            font=_font(size=12), fill="FFF4CE",
        )
        return

    # --- How to read this report -----------------------------------------
    #
    # Explained ONCE, here, in plain language. The colors and category
    # names below are used throughout the rest of the workbook without
    # re-explaining them every time - a reader who reads this box first
    # never has to guess what "Notable Gap" means when they hit it three
    # sheets later.
    row += 1
    sheet.merge_cells(
        start_row=row, start_column=1, end_row=row, end_column=6
    )
    _set(
        sheet, row, 1, "HOW TO READ THIS REPORT",
        font=_font(size=12, bold=True, color="FFFFFF"),
        fill=SLATE, alignment=LEFT_MID,
    )
    sheet.row_dimensions[row].height = 22
    row += 1

    guide_order = [
        Bucket.CRITICAL_GAP,
        Bucket.NOTABLE_GAP,
        Bucket.STANDARD_GAP,
        Bucket.ADDITIONAL,
        Bucket.MARKET_VARIATION,
        Bucket.ALIGNED,
    ]

    for bucket in guide_order:
        color = BUCKET_COLOR[bucket]
        tint = BUCKET_FILL[bucket]

        _set(
            sheet, row, 1, BUCKET_LABEL[bucket],
            font=_font(bold=True, color=color), fill=tint,
            alignment=LEFT_MID,
        )
        sheet.merge_cells(
            start_row=row, start_column=2, end_row=row, end_column=6
        )
        _set(
            sheet, row, 2, BUCKET_EXPLAINER[bucket],
            fill=tint, alignment=LEFT_MID,
        )
        for column in range(2, 7):
            sheet.cell(row=row, column=column).border = BORDER
        sheet.row_dimensions[row].height = 20
        row += 1

    row += 1

    # --- Scorecard -----------------------------------------------------
    row += 1
    sheet.merge_cells(
        start_row=row, start_column=1, end_row=row, end_column=6
    )
    _set(
        sheet, row, 1, "SCORECARD",
        font=_font(size=12, bold=True, color="FFFFFF"),
        fill=SLATE, alignment=LEFT_MID,
    )
    sheet.row_dimensions[row].height = 22
    row += 1

    cards = [
        ("Benchmark alignment", _percent(result.alignment_score), NAVY),
        (
            "Critical gaps",
            str(result.count_of(Bucket.CRITICAL_GAP)),
            BUCKET_COLOR[Bucket.CRITICAL_GAP],
        ),
        (
            "Notable gaps",
            str(result.count_of(Bucket.NOTABLE_GAP)),
            BUCKET_COLOR[Bucket.NOTABLE_GAP],
        ),
        (
            "Standard gaps",
            str(result.count_of(Bucket.STANDARD_GAP)),
            BUCKET_COLOR[Bucket.STANDARD_GAP],
        ),
        (
            "Extra requirements",
            str(result.count_of(Bucket.ADDITIONAL)),
            BUCKET_COLOR[Bucket.ADDITIONAL],
        ),
        ("Report confidence", _percent(result.confidence), SLATE),
    ]

    for column, (label, value, color) in enumerate(cards, start=1):
        _set(
            sheet, row, column, label,
            font=_font(size=9, bold=True, color="FFFFFF"),
            fill=color, alignment=CENTER,
        )
        _set(
            sheet, row + 1, column, value,
            font=_font(size=20, bold=True, color=color),
            fill="FFFFFF", alignment=CENTER,
        )

    sheet.row_dimensions[row].height = 20
    sheet.row_dimensions[row + 1].height = 34
    row += 3

    # --- Narrative -----------------------------------------------------
    sheet.merge_cells(
        start_row=row, start_column=1, end_row=row + 1, end_column=6
    )
    _set(sheet, row, 1, result.overall_summary, font=_font(size=11))
    sheet.row_dimensions[row].height = 20
    sheet.row_dimensions[row + 1].height = 20
    row += 3

    # --- Years of experience -------------------------------------------
    benchmark = result.experience_benchmark
    if benchmark is not None:
        sheet.merge_cells(
            start_row=row, start_column=1, end_row=row, end_column=6
        )
        _set(
            sheet, row, 1, "YEARS OF EXPERIENCE (calculated, not inferred)",
            font=_font(size=12, bold=True, color="FFFFFF"),
            fill=SLATE, alignment=LEFT_MID,
        )
        sheet.row_dimensions[row].height = 22
        row += 1

        _set(
            sheet, row, 1, "Verdict",
            font=_font(bold=True), fill=LIGHT, alignment=LEFT_MID,
        )
        sheet.merge_cells(
            start_row=row, start_column=2, end_row=row, end_column=6
        )
        _set(sheet, row, 2, benchmark.verdict, alignment=LEFT_MID)
        for column in range(2, 7):
            sheet.cell(row=row, column=column).border = BORDER
        row += 1

        _set(
            sheet, row, 1, "Basis",
            font=_font(bold=True), fill=LIGHT, alignment=LEFT_MID,
        )
        sheet.merge_cells(
            start_row=row, start_column=2, end_row=row, end_column=6
        )
        _set(sheet, row, 2, benchmark.detail, alignment=LEFT_MID)
        for column in range(2, 7):
            sheet.cell(row=row, column=column).border = BORDER
        sheet.row_dimensions[row].height = 30
        row += 2

    # --- Top priorities -------------------------------------------------
    priorities = [
        f
        for f in result.findings
        if f.bucket in (Bucket.CRITICAL_GAP, Bucket.NOTABLE_GAP)
    ][:8]

    sheet.merge_cells(
        start_row=row, start_column=1, end_row=row, end_column=6
    )
    _set(
        sheet, row, 1, "WHAT TO ACT ON FIRST",
        font=_font(size=12, bold=True, color="FFFFFF"),
        fill=SLATE, alignment=LEFT_MID,
    )
    sheet.row_dimensions[row].height = 22
    row += 1

    if not priorities:
        sheet.merge_cells(
            start_row=row, start_column=1, end_row=row, end_column=6
        )
        _set(
            sheet, row, 1,
            "No critical or notable gaps were identified against this "
            "benchmark.",
            font=_font(italic=True), fill=BUCKET_FILL[Bucket.ALIGNED],
        )
        return

    _header_row(
        sheet, row,
        ["#", "Category", "Finding", "Benchmark support", "Why it matters", "Action"],
    )
    row += 1

    for position, finding in enumerate(priorities, start=1):
        color = BUCKET_COLOR[finding.bucket]
        tint = BUCKET_FILL[finding.bucket]

        _set(sheet, row, 1, position, alignment=CENTER, fill=tint,
             font=_font(bold=True, color=color))
        _set(sheet, row, 2, finding.category.value, fill=tint)
        _set(sheet, row, 3, finding.label, font=_font(bold=True), fill=tint)
        _set(sheet, row, 4, finding.prevalence_text, alignment=CENTER,
             fill=tint)
        _set(
            sheet, row, 5,
            finding.business_impact or BUCKET_LABEL[finding.bucket],
            fill=tint,
        )
        _set(
            sheet, row, 6, finding.action_text,
            font=_font(bold=True, color=color), fill=tint, alignment=CENTER,
        )

        _autofit_height(sheet, row, 6)
        row += 1

    sheet.freeze_panes = "A4"


# ==========================================================================
# SHEET 2 — Findings
# ==========================================================================

FINDING_HEADERS = [
    "Priority",
    "Type",
    "Category",
    "Finding",
    "Status",
    "How common in other jobs",
    "Importance",
    "Why it matters",
    "What similar roles say",
    "What your JD says",
]

_DIRECTION_TEXT = {
    Direction.M_MINUS: "Missing from your JD",
    Direction.M_PLUS: "Additional vs. the benchmark",
    Direction.ALIGNED: "Matches the benchmark",
}


def _prevalence_sentence(finding) -> str:
    """
    Replaces the old two-column pair (Support "4/10", Prevalence "40%")
    with one sentence a non-technical reader can act on without doing
    arithmetic in their head.
    """
    if finding.applicable_reference_count <= 0:
        return "No benchmark data for this item"

    pct = _percent(finding.prevalence)
    return (
        f"{finding.support_count} of "
        f"{finding.applicable_reference_count} similar roles ({pct})"
    )


def _sheet_findings(workbook: Workbook, result: ComparisonResult) -> None:
    sheet = workbook.create_sheet("Findings")
    sheet.sheet_view.showGridLines = False
    _widths(
        sheet,
        [9, 17, 20, 38, 20, 24, 13, 40, 40, 40],
    )

    sheet.merge_cells("A1:J1")
    _set(
        sheet, 1, 1,
        "Every finding, ranked by what needs attention first. See "
        "\"How to read this report\" on the Executive Summary sheet for "
        "what each Type means. Full document-by-document evidence is on "
        "the Audit Trail sheet.",
        font=_font(size=11, bold=True, color="FFFFFF"),
        fill=NAVY, alignment=LEFT_MID,
    )
    sheet.row_dimensions[1].height = 30

    _header_row(sheet, 2, FINDING_HEADERS)

    row = 3

    if not result.findings:
        _set(sheet, row, 1, "No comparable content was extracted.")
        return

    for position, finding in enumerate(result.findings, start=1):
        color = BUCKET_COLOR[finding.bucket]
        tint = BUCKET_FILL[finding.bucket]

        values = [
            position,
            BUCKET_LABEL[finding.bucket],
            finding.category.value,
            finding.label,
            _DIRECTION_TEXT[finding.direction],
            _prevalence_sentence(finding),
            finding.tier.value.title(),
            finding.business_impact or "—",
            _bullets(finding.reference_texts),
            _bullets(finding.new_job_texts),
        ]

        for column, value in enumerate(values, start=1):
            bold = column in (2, 4)
            alignment = CENTER if column in (1, 5, 7) else WRAP_TOP
            _set(
                sheet, row, column, value,
                font=_font(
                    bold=bold,
                    color=color if column == 2 else "000000",
                ),
                fill=tint,
                alignment=alignment,
            )

        _autofit_height(sheet, row, len(values), maximum=130)
        row += 1

    sheet.auto_filter.ref = f"A2:J{row - 1}"
    sheet.freeze_panes = "D3"


# ==========================================================================
# SHEET 3 — Category Overview
# ==========================================================================


def _sheet_categories(workbook: Workbook, result: ComparisonResult) -> None:
    sheet = workbook.create_sheet("Category Overview")
    sheet.sheet_view.showGridLines = False
    _widths(sheet, [28, 16, 14, 14, 14, 16, 18, 34])

    sheet.merge_cells("A1:H1")
    _set(
        sheet, 1, 1, "Alignment by category",
        font=_font(size=12, bold=True, color="FFFFFF"),
        fill=NAVY, alignment=LEFT_MID,
    )
    sheet.row_dimensions[1].height = 24

    _header_row(
        sheet, 2,
        [
            "Category",
            "Alignment",
            "Matches",
            "Missing",
            "Extra",
            "Total items",
            "Data available",
            "Note",
        ],
    )

    row = 3
    summary_by_category = {
        s.category: s for s in result.category_summaries
    }

    for category in CATEGORY_ORDER:
        summary = summary_by_category.get(category)
        if summary is None:
            continue

        if summary.alignment_score is None:
            color, tint = "7F7F7F", "F2F2F2"
        elif summary.alignment_score >= 0.8:
            color, tint = "548235", "E8F2E2"
        elif summary.alignment_score >= 0.5:
            color, tint = "BF8F00", "FFF4CE"
        else:
            color, tint = "C00000", "FCE4E4"

        note = ""
        if summary.clustering_used_fallback:
            note = (
                "Exact-wording matching was used here; differences may be "
                "over-reported."
            )
        elif summary.benchmark_strength == BenchmarkStrength.NONE:
            note = "No benchmark document specifies this category."
        elif summary.benchmark_strength == BenchmarkStrength.WEAK:
            note = "Fewer than 3 documents specify this; indicative only."

        _set(sheet, row, 1, category.value, font=_font(bold=True), fill=tint)
        _set(
            sheet, row, 2, _percent(summary.alignment_score),
            font=_font(size=13, bold=True, color=color),
            fill=tint, alignment=CENTER,
        )
        _set(sheet, row, 3, summary.aligned_count, fill=tint, alignment=CENTER)
        _set(sheet, row, 4, summary.m_minus_count, fill=tint, alignment=CENTER)
        _set(sheet, row, 5, summary.m_plus_count, fill=tint, alignment=CENTER)
        _set(sheet, row, 6, summary.concept_count, fill=tint, alignment=CENTER)
        _set(
            sheet, row, 7, summary.benchmark_strength.value.title(),
            fill=tint, alignment=CENTER,
        )
        _set(sheet, row, 8, note or "—", fill=tint, font=_font(italic=True))

        _autofit_height(sheet, row, 8)
        row += 1

    row += 1
    sheet.merge_cells(
        start_row=row, start_column=1, end_row=row, end_column=8
    )
    _set(
        sheet, row, 1,
        "Alignment = concepts the submitted job shares with the benchmark, "
        "as a share of everything the benchmark asks for. Additional "
        "requirements (M+) do not reduce it.",
        font=_font(italic=True, size=10), fill=LIGHT,
    )
    sheet.freeze_panes = "A3"


# ==========================================================================
# SHEET 4 — Comparison Matrix
# ==========================================================================


def _sheet_matrix(
    workbook: Workbook,
    result: ComparisonResult,
    new_job: Any,
    reference_jobs: Sequence[Any],
) -> None:
    sheet = workbook.create_sheet("Appendix A - Full Comparison")
    sheet.sheet_view.showGridLines = False

    all_jobs = [new_job] + list(reference_jobs or [])

    _set(
        sheet, 1, 1, "Criteria",
        font=_font(bold=True, color="FFFFFF"), fill=NAVY, alignment=CENTER,
    )

    for offset, job in enumerate(all_jobs):
        label = document_label(job, offset)
        _set(
            sheet, 1, offset + 2, label,
            font=_font(bold=True, color="FFFFFF"),
            fill=(NAVY if offset else "0F6FC6"),
            alignment=CENTER,
        )

    sheet.row_dimensions[1].height = 32

    rows: List[tuple] = [
        (
            "Job code",
            lambda j: _first_text(j, ("job_information", "job_code"), ("job_code",)),
        ),
        (
            "Job title",
            lambda j: _first_text(j, ("job_information", "job_title"), ("job_title",)),
        ),
    ]

    for category in CATEGORY_ORDER:
        accessor = CATEGORY_ACCESSOR[category]
        rows.append((category.value, accessor))

    row = 2
    for label, accessor in rows:
        _set(
            sheet, row, 1, label,
            font=_font(bold=True), fill=LIGHT, alignment=LEFT_MID,
        )

        for offset, job in enumerate(all_jobs):
            try:
                raw = accessor(job)
            except Exception:
                raw = None

            if isinstance(raw, (list, tuple, set)):
                text = _bullets([str(v) for v in raw], limit=20)
            else:
                text = (
                    str(raw).strip()
                    if raw is not None and str(raw).strip()
                    else NOT_SPECIFIED
                )

            _set(
                sheet, row, offset + 2, text,
                fill=("FFFFFF" if offset else "F7FBFF"),
            )

        _autofit_height(sheet, row, len(all_jobs) + 1, maximum=220)
        row += 1

    sheet.column_dimensions["A"].width = 26
    for offset in range(len(all_jobs)):
        sheet.column_dimensions[
            get_column_letter(offset + 2)
        ].width = 42

    sheet.freeze_panes = "B2"


def _first_text(job: Any, *paths) -> str:
    for path in paths:
        current = job
        for attribute in path:
            current = getattr(current, attribute, None)
            if current is None:
                break
        if current and str(current).strip():
            return str(current).strip()
    return NOT_SPECIFIED


# ==========================================================================
# SHEET 5 — Audit Trail
# ==========================================================================


def _sheet_audit(workbook: Workbook, result: ComparisonResult) -> None:
    sheet = workbook.create_sheet("Appendix B - Audit Detail")
    sheet.sheet_view.showGridLines = False
    _widths(sheet, [16, 22, 36, 12, 10, 12, 20, 20, 22, 14])

    sheet.merge_cells("A1:J1")
    _set(
        sheet, 1, 1,
        "Every concept, every count. Sheet 1 and 2 are derived from this "
        "table and nothing else.",
        font=_font(size=11, bold=True, color="FFFFFF"),
        fill=NAVY, alignment=LEFT_MID,
    )
    sheet.row_dimensions[1].height = 24

    # --- Document key ---------------------------------------------------
    _header_row(sheet, 2, ["Ref", "Company", "Job code", "Job title"])
    row = 3
    for document in result.documents:
        tint = "E4EEF8" if document.is_new_job else "FFFFFF"
        _set(
            sheet, row, 1,
            "NEW" if document.is_new_job else f"D{document.doc_index}",
            font=_font(bold=True), fill=tint, alignment=CENTER,
        )
        _set(sheet, row, 2, document.company_code or NOT_SPECIFIED, fill=tint)
        _set(sheet, row, 3, document.job_code or NOT_SPECIFIED, fill=tint)
        _set(sheet, row, 4, document.job_title or NOT_SPECIFIED, fill=tint)
        row += 1

    row += 1

    # --- Concept evidence ------------------------------------------------
    _header_row(
        sheet, row,
        [
            "Concept id",
            "Category",
            "Concept",
            "In new job",
            "Support",
            "Applicable",
            "Prevalence",
            "Present in",
            "Absent from",
            "Silent",
        ],
    )
    row += 1

    for evidence in result.evidence:
        _set(sheet, row, 1, evidence.concept_id, font=_font(size=9))
        _set(sheet, row, 2, evidence.category.value)
        _set(sheet, row, 3, evidence.label)
        _set(
            sheet, row, 4, "Yes" if evidence.new_job_has else "No",
            alignment=CENTER,
            font=_font(
                bold=True,
                color="548235" if evidence.new_job_has else "C00000",
            ),
        )
        _set(sheet, row, 5, evidence.support_count, alignment=CENTER)
        _set(
            sheet, row, 6, evidence.applicable_reference_count,
            alignment=CENTER,
        )
        _set(
            sheet, row, 7, _percent(evidence.prevalence), alignment=CENTER
        )
        _set(sheet, row, 8, _doc_list(evidence.present_in))
        _set(sheet, row, 9, _doc_list(evidence.absent_in))
        _set(sheet, row, 10, _doc_list(evidence.not_specified_in))

        _autofit_height(sheet, row, 10, maximum=60)
        row += 1

    # --- Diagnostics ------------------------------------------------------
    row += 1
    diagnostics = result.diagnostics

    sheet.merge_cells(
        start_row=row, start_column=1, end_row=row, end_column=10
    )
    _set(
        sheet, row, 1, "RUN DIAGNOSTICS",
        font=_font(size=12, bold=True, color="FFFFFF"),
        fill=SLATE, alignment=LEFT_MID,
    )
    row += 1

    entries = [
        ("Model calls", str(diagnostics.llm_calls)),
        ("Model failures recovered", str(diagnostics.llm_failures)),
        (
            "Categories using offline clustering",
            ", ".join(diagnostics.categories_using_fallback) or "None",
        ),
        (
            "Unsupported references discarded",
            str(len(diagnostics.dropped_unknown_ids)),
        ),
        (
            "Requirements recovered as standalone concepts",
            str(len(diagnostics.recovered_orphan_ids)),
        ),
        (
            "Gaps tiered without model input",
            str(len(diagnostics.materiality_missing_for)),
        ),
        ("Report confidence", _percent(result.confidence)),
    ]

    for label, value in entries:
        _set(
            sheet, row, 1, label,
            font=_font(bold=True), fill=LIGHT, alignment=LEFT_MID,
        )
        sheet.merge_cells(
            start_row=row, start_column=2, end_row=row, end_column=10
        )
        _set(sheet, row, 2, value, alignment=LEFT_MID)
        for column in range(2, 11):
            sheet.cell(row=row, column=column).border = BORDER
        row += 1

    for warning in diagnostics.warnings:
        sheet.merge_cells(
            start_row=row, start_column=1, end_row=row, end_column=10
        )
        _set(
            sheet, row, 1, f"⚠  {warning}",
            font=_font(italic=True), fill="FFF4CE",
        )
        row += 1


# ==========================================================================
# Public API
# ==========================================================================


def generate_comparison_excel(
    result: ComparisonResult,
    new_job: Any,
    reference_jobs: Sequence[Any],
) -> bytes:
    """Render the workbook in memory and return its bytes."""

    if result is None:
        raise ValueError("result must not be None")

    workbook = Workbook()

    _sheet_summary(workbook, result)

    if result.status != ComparisonStatusV2.NEW_JOB_CODE:
        _sheet_findings(workbook, result)
        _sheet_categories(workbook, result)

    _sheet_matrix(workbook, result, new_job, reference_jobs)

    if result.status != ComparisonStatusV2.NEW_JOB_CODE:
        _sheet_audit(workbook, result)

    for sheet in workbook.worksheets:
        sheet.page_setup.orientation = "landscape"
        sheet.page_setup.fitToWidth = 1
        sheet.sheet_properties.pageSetUpPr.fitToPage = True

    buffer = BytesIO()
    workbook.save(buffer)
    buffer.seek(0)
    payload = buffer.getvalue()
    workbook.close()
    buffer.close()

    if not payload:
        raise RuntimeError("Excel generation produced an empty workbook.")

    return payload


__all__ = ["generate_comparison_excel"]
