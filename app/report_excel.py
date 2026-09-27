"""
app/report_excel.py

Excel report generation for JobComparisonAI.

IMPORTANT DESIGN:
- GeminiComparisonReport is the single source of truth for comparison.
- Excel does NOT perform semantic comparison.
- Excel does NOT calculate majority.
- Excel does NOT classify M+ / M-.
- Excel does NOT infer decisions from change_type.
- Excel does NOT infer decisions from semantic_relationship.
- Excel only renders Gemini's explicit decisions:
    is_m_plus
    is_m_minus
- Excel only renders Gemini's display_summary / structured evidence.
- Every Gemini semantic change remains independent.
"""

from __future__ import annotations

from io import BytesIO
from typing import List, Sequence, Tuple

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

from app.schemas import (
    GeminiComparisonReport,
    SemanticChange,
)


# ============================================================
# Constants
# ============================================================

CATEGORY_ORDER = [
    "Years of Experience",
    "Field of Experience",
    "Education Level",
    "Language",
    "Computer",
    "Soft Skills",
    "Duties & Responsibilities",
]


CATEGORY_ALIASES = {
    "years of experience": "Years of Experience",
    "experience years": "Years of Experience",

    "field of experience": "Field of Experience",
    "experience field": "Field of Experience",

    "education": "Education Level",
    "education level": "Education Level",

    "language": "Language",
    "languages": "Language",

    "computer": "Computer",
    "technical skills": "Computer",
    "technical skill": "Computer",
    "computer skills": "Computer",

    "soft skills": "Soft Skills",
    "soft skill": "Soft Skills",

    "duties": "Duties & Responsibilities",
    "responsibilities": "Duties & Responsibilities",
    "duties & responsibilities": "Duties & Responsibilities",
    "duties and responsibilities": "Duties & Responsibilities",
}


# ============================================================
# Generic helpers
# ============================================================

def _clean_text(value) -> str:
    """
    Convert a value to clean display text.

    Presentation helper only.
    No semantic interpretation is performed.
    """

    if value is None:
        return ""

    text = str(value).strip()

    if not text:
        return ""

    return text


def _canonical_category(category: str) -> str:
    """
    Normalize a category name for Excel grouping.

    This is only category-label normalization.
    It does not perform semantic comparison.
    """

    cleaned = _clean_text(category)

    if not cleaned:
        return ""

    return CATEGORY_ALIASES.get(
        cleaned.lower(),
        cleaned,
    )


# ============================================================
# Gemini semantic changes
# ============================================================

def _changes_for_category(
    report: GeminiComparisonReport,
    canonical_label: str,
) -> List[SemanticChange]:
    """
    Return Gemini's top-level semantic changes for one category.

    IMPORTANT:
    `report.semantic_changes` is the authoritative source.

    Excel does NOT merge:
        sections[].changes

    into:
        semantic_changes

    because doing so could introduce semantic decisions that are
    not present in Gemini's authoritative output.
    """

    semantic_changes = getattr(
        report,
        "semantic_changes",
        None,
    )

    if not semantic_changes:
        return []

    result: List[SemanticChange] = []

    for change in semantic_changes:

        change_category = _canonical_category(
            getattr(
                change,
                "category",
                "",
            )
        )

        if change_category == canonical_label:
            result.append(change)

    return result


# ============================================================
# Gemini -> Excel presentation
# ============================================================

def _gemini_decision_label(
    change: SemanticChange,
) -> str:
    """
    Render the comparison decision.

    Excel does NOT infer a decision from:

        - change_type
        - semantic_relationship
        - explanation
        - historical_requirements
        - new_requirements

    The decision comes only from `change.classification`, which is
    computed deterministically (in app/schemas.py) from Gemini's own
    per-document evidence and its own significance rating - never
    re-judged here.

    Rules:

        M_PLUS
            -> M+

        M_MINUS
            -> M-

        SIGNIFICANT_MINORITY_GAP
            -> M- (Notable Minority)
            Present in a MINORITY of historical jobs (<=50%) but
            missing from the NEW JOB, and Gemini rated it HIGH or
            CRITICAL importance. A plain majority cutoff would have
            silently dropped this - it is surfaced here on purpose,
            distinguished from an ordinary M- so the reader knows the
            statistical basis is different.

        anything else
            -> Matching
    """

    classification = getattr(
        change,
        "classification",
        None,
    )

    label = getattr(classification, "value", classification)

    if label == "m_plus":
        return "M+"

    if label == "m_minus":
        return "M-"

    if label == "significant_minority_gap":
        return "M- (Notable Minority)"

    return "Matching"


def _change_display_text(
    change: SemanticChange,
) -> str:
    """
    Render the Gemini evidence required for the final Excel report.

    Excel does not perform semantic comparison or classification.

    The final report always displays:

        - new_requirements
        - historical_requirements
        - explanation

    matched_reference_count, total_reference_count, and
    semantic_relationship stay out of the Excel presentation - they
    are audit detail, not something the reader needs to act on.

    significance and business_impact are shown ONLY for a
    Significant Minority Gap (change.classification ==
    "significant_minority_gap"). For every other row they are left
    out to avoid clutter, since a plain M+/M-/Matching row is already
    self-explanatory. For a Significant Minority Gap they are the
    entire reason the row is being shown at all - hiding them would
    turn an explained exception back into an unexplained one.
    """

    sections: List[str] = []

    classification = getattr(
        change,
        "classification",
        None,
    )

    classification_label = getattr(
        classification,
        "value",
        classification,
    )

    is_significant_minority = (
        classification_label == "significant_minority_gap"
    )

    # --------------------------------------------------------
    # NEW REQUIREMENTS
    # --------------------------------------------------------

    new_requirements = [
        _clean_text(item)
        for item in (
            getattr(
                change,
                "new_requirements",
                None,
            )
            or []
        )
        if _clean_text(item)
    ]

    if new_requirements:
        sections.append(
            "NEW REQUIREMENT(S):\n"
            + "\n".join(
                f"• {item}"
                for item in new_requirements
            )
        )

    # --------------------------------------------------------
    # HISTORICAL REQUIREMENTS
    # --------------------------------------------------------

    historical_requirements = [
        _clean_text(item)
        for item in (
            getattr(
                change,
                "historical_requirements",
                None,
            )
            or []
        )
        if _clean_text(item)
    ]

    if historical_requirements:
        sections.append(
            "HISTORICAL REQUIREMENT(S):\n"
            + "\n".join(
                f"• {item}"
                for item in historical_requirements
            )
        )

    # --------------------------------------------------------
    # WHY THIS IS FLAGGED DESPITE BEING A MINORITY
    #
    # Only rendered for Significant Minority Gap rows. This is the
    # information that would otherwise have been silently dropped by
    # a plain 50% cutoff.
    # --------------------------------------------------------

    if is_significant_minority:

        significance = getattr(
            change,
            "significance",
            None,
        )

        significance_label = getattr(
            significance,
            "value",
            significance,
        )

        business_impact = _clean_text(
            getattr(
                change,
                "business_impact",
                None,
            )
        )

        support_ratio = getattr(
            change,
            "support_ratio",
            None,
        )

        # Denominator = documents that actually specify this category
        # (falls back to all documents for older report objects).
        support_denominator = getattr(
            change,
            "applicable_reference_count",
            getattr(change, "total_reference_count", 0),
        )

        support_text = (
            f"Present in {getattr(change, 'matched_reference_count', 0)} "
            f"of {support_denominator} historical "
            "documents"
            + (
                f" ({support_ratio:.0%})"
                if support_ratio is not None
                else ""
            )
        )

        why_flagged_lines = [
            f"Significance: {significance_label}",
            support_text
            + " - below majority, flagged anyway due to importance.",
        ]

        if business_impact:
            why_flagged_lines.append(
                f"Why it matters: {business_impact}"
            )

        sections.append(
            "WHY THIS IS FLAGGED:\n"
            + "\n".join(why_flagged_lines)
        )

    # --------------------------------------------------------
    # HISTORICAL BASIS
    #
    # Shown only when some historical documents say nothing at all
    # about this category, so the reader knows how thin the benchmark
    # is (e.g. "1 of 2" or "no benchmark") instead of assuming all
    # documents were compared.
    # --------------------------------------------------------

    applicable_count = getattr(
        change,
        "applicable_reference_count",
        None,
    )

    total_count = getattr(
        change,
        "total_reference_count",
        None,
    )

    if (
        applicable_count is not None
        and total_count
        and applicable_count < total_count
    ):
        if applicable_count == 0:
            basis_text = (
                f"None of the {total_count} historical documents "
                "specify this category, so there is no historical "
                "benchmark."
            )
        else:
            basis_text = (
                f"Only {applicable_count} of {total_count} historical "
                "document(s) specify this category."
            )

        sections.append(
            "HISTORICAL BASIS:\n"
            + basis_text
        )

    # --------------------------------------------------------
    # EXPLANATION
    # --------------------------------------------------------

    explanation = _clean_text(
        getattr(
            change,
            "explanation",
            None,
        )
    )

    if explanation:
        sections.append(
            "EXPLANATION:\n"
            + explanation
        )

    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    if not sections:

        display_summary = _clean_text(
            getattr(
                change,
                "display_summary",
                None,
            )
        )

        if display_summary:
            return display_summary

    return "\n\n".join(sections)


def _format_change_line(
    change: SemanticChange,
) -> str:
    """
    Render ONE complete Gemini semantic change.

    The decision label is taken only from Gemini's explicit
    is_m_plus / is_m_minus fields.

    All remaining content is rendered from Gemini's structured
    evidence fields.
    """

    decision = _gemini_decision_label(
        change
    )

    description = _change_display_text(
        change
    )

    if description:
        return f"{decision}\n{description}"

    return decision


def _category_decision_and_description(
    changes: Sequence[SemanticChange],
    category: str,
) -> Tuple[str, str]:
    """
    Render ALL Gemini changes for a category.

    A category can contain multiple independent decisions.

    Example:

        M+ — AWS
        M+ — Azure
        M- — Linux
        M+ — Terraform

    We MUST NOT collapse these into a single category-level
    decision such as only M+ or only M-.

    Excel performs NO:

        - semantic comparison
        - majority calculation
        - M+ classification
        - M- classification
        - change_type inference
        - semantic relationship inference
    """

    if not changes:
        return "", ""

    rendered_lines: List[str] = []

    for change in changes:

        line = _format_change_line(
            change
        )

        if line:
            rendered_lines.append(
                line
            )

    if not rendered_lines:
        return "", ""

    return "", "\n".join(
        rendered_lines
    )


# ============================================================
# Excel category renderer
# ============================================================

def _smart_category_cell_text(
    report: GeminiComparisonReport,
    category: str,
) -> str:
    """
    Render every Gemini semantic decision belonging to a category.

    IMPORTANT:

    This function does NOT produce a category-level M+ / M-.

    Every individual Gemini decision is preserved.

    If Gemini returns no semantic changes for the category,
    Excel displays:

        No semantic change reported by Gemini

    This is ONLY a presentation fallback.

    Excel does NOT calculate or infer this semantic result.
    """

    canonical_category = _canonical_category(
        category
    )

    changes = _changes_for_category(
        report=report,
        canonical_label=canonical_category,
    )

    if not changes:
        return "No semantic change reported by Gemini"

    rendered_lines: List[str] = []

    for change in changes:

        line = _format_change_line(
            change
        )

        if line:
            rendered_lines.append(
                line
            )

    if not rendered_lines:
        return "No semantic change reported by Gemini"

    return "\n".join(
        rendered_lines
    )


# ============================================================
# Legacy compatibility helpers
# ============================================================

def _matching_or_mn_mp_text(
    historical_requirements,
    new_requirements,
    change_type=None,
) -> str:
    """
    Legacy compatibility helper.

    This function is intentionally NOT used by the current
    Gemini-driven report renderer.

    Kept to avoid breaking older imports/tests.
    """

    historical = [
        _clean_text(x)
        for x in (historical_requirements or [])
        if _clean_text(x)
    ]

    new = [
        _clean_text(x)
        for x in (new_requirements or [])
        if _clean_text(x)
    ]

    if new and not historical:
        return "M+"

    if historical and not new:
        return "M-"

    if historical and new:
        return "Matching"

    return ""


def _experience_years_text(
    historical_requirements,
    new_requirements,
) -> str:
    """
    Legacy compatibility helper.

    Current Excel generation does NOT use this function for
    semantic decisions.
    """

    historical = [
        _clean_text(x)
        for x in (historical_requirements or [])
        if _clean_text(x)
    ]

    new = [
        _clean_text(x)
        for x in (new_requirements or [])
        if _clean_text(x)
    ]

    if new:
        return "\n".join(new)

    if historical:
        return "\n".join(historical)

    return ""


# ============================================================
# Styling
# ============================================================

def _thin_border() -> Border:
    """
    Standard thin Excel border.
    """

    side = Side(
        style="thin",
    )

    return Border(
        left=side,
        right=side,
        top=side,
        bottom=side,
    )


def _header_fill() -> PatternFill:
    """
    Header fill.
    """

    return PatternFill(
        fill_type="solid",
        fgColor="1F4E78",
    )


def _section_fill() -> PatternFill:
    """
    Section/category fill.
    """

    return PatternFill(
        fill_type="solid",
        fgColor="D9EAF7",
    )


def _label_font() -> Font:
    return Font(
        bold=True,
        size=11,
    )


def _header_font() -> Font:
    return Font(
        bold=True,
        color="FFFFFF",
        size=12,
    )


def _normal_font() -> Font:
    return Font(
        size=10,
    )


def _apply_cell_style(
    cell,
    *,
    bold: bool = False,
    fill: PatternFill | None = None,
    font_color: str | None = None,
    horizontal: str = "left",
    vertical: str = "top",
    wrap_text: bool = True,
):
    """
    Apply presentation-only styling.
    """

    color = font_color

    if bold or color:
        cell.font = Font(
            bold=bold,
            color=color,
            size=10,
        )
    else:
        cell.font = _normal_font()

    if fill is not None:
        cell.fill = fill

    cell.alignment = Alignment(
        horizontal=horizontal,
        vertical=vertical,
        wrap_text=wrap_text,
    )

    cell.border = _thin_border()


# ============================================================
# Job metadata extraction
# ============================================================

def _get_job_information(
    report: GeminiComparisonReport,
):
    """
    Safely retrieve job information from the report when available.

    This function is presentation-oriented and intentionally tolerant
    of schema variations.
    """

    job_information = getattr(
        report,
        "job_information",
        None,
    )

    if job_information is not None:
        return job_information

    return None


def _get_report_value(
    report: GeminiComparisonReport,
    name: str,
    default: str = "",
) -> str:
    """
    Safely retrieve a report attribute.
    """

    value = getattr(
        report,
        name,
        None,
    )

    if value is None:
        return default

    return _clean_text(value)


# ============================================================
# Fixed-layout report
# ============================================================

def generate_fixed_layout_excel_report(
    report: GeminiComparisonReport,
    *,
    new_job=None,
    reference_jobs=None,
) -> bytes:
    """
    Generate the fixed-layout Excel report.

    Layout:

        Job Comparison Report

        Department
        Job Code
        Company Code
        Job Title

        Category | Job Comparison Result

    The workbook is generated entirely in memory.

    Returns:
        bytes: Excel workbook bytes.

    IMPORTANT:
        No .xlsx file is written to disk.
    """

    workbook = Workbook()

    worksheet = workbook.active
    worksheet.title = "Job Comparison"

    # ========================================================
    # General worksheet configuration
    # ========================================================

    worksheet.sheet_view.showGridLines = False

    # Only two columns are used.
    worksheet.column_dimensions["A"].width = 28
    worksheet.column_dimensions["B"].width = 90

    # ========================================================
    # Title
    # ========================================================

    worksheet.merge_cells(
        "A1:B1"
    )

    title_cell = worksheet["A1"]

    title_cell.value = (
        "Job Comparison Report"
    )

    title_cell.font = Font(
        bold=True,
        size=16,
        color="FFFFFF",
    )

    title_cell.fill = _header_fill()

    title_cell.alignment = Alignment(
        horizontal="center",
        vertical="center",
    )

    title_cell.border = _thin_border()

    worksheet.row_dimensions[1].height = 28

    # ========================================================
    # Metadata
    # ========================================================

    metadata_rows = []

    # --------------------------------------------------------
    # Department
    # --------------------------------------------------------

    department = ""

    if new_job is not None:

        department = _clean_text(
            getattr(
                new_job,
                "department",
                None,
            )
        )

        if not department:

            job_information = getattr(
                new_job,
                "job_information",
                None,
            )

            if job_information is not None:

                department = _clean_text(
                    getattr(
                        job_information,
                        "department",
                        None,
                    )
                )

    if not department:

        department = _get_report_value(
            report,
            "department",
        )

    # --------------------------------------------------------
    # Job code
    # --------------------------------------------------------

    job_code = ""

    if new_job is not None:

        job_code = _clean_text(
            getattr(
                new_job,
                "job_code",
                None,
            )
        )

        if not job_code:

            job_information = getattr(
                new_job,
                "job_information",
                None,
            )

            if job_information is not None:

                job_code = _clean_text(
                    getattr(
                        job_information,
                        "job_code",
                        None,
                    )
                )

    if not job_code:

        job_code = _get_report_value(
            report,
            "job_code",
        )

    # --------------------------------------------------------
    # Company code
    # --------------------------------------------------------

    company_code = ""

    if new_job is not None:

        # Primary source:
        # company code extracted by Gemini from the job text
        # (JobDescription.company.company_code).
        company = getattr(
            new_job,
            "company",
            None,
        )

        if company is not None:

            company_code = _clean_text(
                getattr(
                    company,
                    "company_code",
                    None,
                )
            )

        # Fallback:
        # direct attribute on the job itself.
        if not company_code:

            company_code = _clean_text(
                getattr(
                    new_job,
                    "company_code",
                    None,
                )
            )

        # Fallback:
        # nested job_information.
        if not company_code:

            job_information = getattr(
                new_job,
                "job_information",
                None,
            )

            if job_information is not None:

                company_code = _clean_text(
                    getattr(
                        job_information,
                        "company_code",
                        None,
                    )
                )

    # Final fallback:
    # Gemini report metadata, if the schema contains it.
    if not company_code:

        company_code = _get_report_value(
            report,
            "company_code",
        )

    # --------------------------------------------------------
    # Job title
    # --------------------------------------------------------

    job_title = ""

    if new_job is not None:

        job_title = _clean_text(
            getattr(
                new_job,
                "job_title",
                None,
            )
        )

        if not job_title:

            job_information = getattr(
                new_job,
                "job_information",
                None,
            )

            if job_information is not None:

                job_title = _clean_text(
                    getattr(
                        job_information,
                        "job_title",
                        None,
                    )
                )

    if not job_title:

        job_title = _get_report_value(
            report,
            "job_title",
        )

    # --------------------------------------------------------
    # Metadata rows
    # --------------------------------------------------------

    metadata_rows.extend(
        [
            (
                "Department",
                department,
            ),
            (
                "Job Code",
                job_code,
            ),
            (
                "Company Code",
                company_code,
            ),
            (
                "Job Title",
                job_title,
            ),
        ]
    )

    # Title is row 1.
    # Metadata starts at row 3.
    current_row = 3

    # ========================================================
    # Metadata rendering
    # ========================================================

    for label, value in metadata_rows:

        label_cell = worksheet.cell(
            row=current_row,
            column=1,
            value=label,
        )

        _apply_cell_style(
            label_cell,
            bold=True,
            fill=_section_fill(),
        )

        value_cell = worksheet.cell(
            row=current_row,
            column=2,
            value=value,
        )

        _apply_cell_style(
            value_cell,
        )

        current_row += 1

    # ========================================================
    # Comparison header
    # ========================================================

    worksheet.cell(
        row=current_row,
        column=1,
        value="Category",
    )

    worksheet.cell(
        row=current_row,
        column=2,
        value="Job Comparison Result",
    )

    # Only columns A and B.
    for column in range(1, 3):

        cell = worksheet.cell(
            row=current_row,
            column=column,
        )

        cell.font = _header_font()
        cell.fill = _header_fill()

        cell.alignment = Alignment(
            horizontal="center",
            vertical="center",
            wrap_text=True,
        )

        cell.border = _thin_border()

    worksheet.row_dimensions[
        current_row
    ].height = 24

    # Save the actual comparison header row.
    comparison_header_row = current_row

    current_row += 1

    # ========================================================
    # Category rows
    # ========================================================

    for category in CATEGORY_ORDER:

        canonical_category = _canonical_category(
            category
        )

        comparison_text = _smart_category_cell_text(
            report=report,
            category=canonical_category,
        )

        # ----------------------------------------------------
        # Category label
        # ----------------------------------------------------

        category_cell = worksheet.cell(
            row=current_row,
            column=1,
            value=category,
        )

        category_cell.font = _label_font()
        category_cell.fill = _section_fill()

        category_cell.alignment = Alignment(
            horizontal="left",
            vertical="top",
            wrap_text=True,
        )

        category_cell.border = _thin_border()

        # ----------------------------------------------------
        # Gemini comparison result
        # ----------------------------------------------------

        result_cell = worksheet.cell(
            row=current_row,
            column=2,
            value=comparison_text,
        )

        _apply_cell_style(
            result_cell,
        )

        # ----------------------------------------------------
        # Row height
        # ----------------------------------------------------

        line_count = (
            len(
                comparison_text.splitlines()
            )
            if comparison_text
            else 1
        )

        worksheet.row_dimensions[
            current_row
        ].height = max(
            30,
            min(
                20 * line_count,
                150,
            ),
        )

        current_row += 1

    # ========================================================
    # Freeze panes
    # ========================================================

    # Freeze everything above the comparison rows.
    #
    # With:
    #   row 1  = title
    #   rows 3-6 = metadata
    #   row 7  = comparison header
    #
    # Category rows start at row 8.
    worksheet.freeze_panes = "A8"

    # ========================================================
    # Print configuration
    # ========================================================

    worksheet.page_setup.orientation = (
        "landscape"
    )

    worksheet.page_setup.fitToWidth = 1
    worksheet.page_setup.fitToHeight = 0

    worksheet.sheet_properties.pageSetUpPr.fitToPage = True

    # Repeat title + metadata + comparison header.
    #
    # Rows:
    #   1 = title
    #   2 = blank
    #   3 = Department
    #   4 = Job Code
    #   5 = Company Code
    #   6 = Job Title
    #   7 = comparison header
    worksheet.print_title_rows = (
        f"1:{comparison_header_row}"
    )

    worksheet.page_margins.left = 0.25
    worksheet.page_margins.right = 0.25
    worksheet.page_margins.top = 0.5
    worksheet.page_margins.bottom = 0.5

    # ========================================================
    # Save to memory ONLY
    # ========================================================

    excel_buffer = BytesIO()

    workbook.save(excel_buffer)

    excel_buffer.seek(0)

    return excel_buffer.getvalue()


# ============================================================
# Public Excel report API
# ============================================================

def generate_excel_report(
    report: GeminiComparisonReport,
    *,
    new_job=None,
    reference_jobs=None,
) -> bytes:
    """
    Public Excel report generation function.

    The fixed layout is the canonical renderer.

    Returns:
        bytes: Excel workbook bytes.

    No Excel file is written to disk.
    """

    return generate_fixed_layout_excel_report(
        report=report,
        new_job=new_job,
        reference_jobs=reference_jobs,
    )