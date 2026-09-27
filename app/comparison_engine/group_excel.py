"""Excel overview for an all-jobs job-code comparison."""

from __future__ import annotations

from io import BytesIO
from typing import Iterable, Optional

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from app.comparison_engine.group import JobGroupComparison
from app.comparison_engine.models import (
    BUCKET_LABEL,
    CATEGORY_ORDER,
    Bucket,
    Direction,
)


NAVY = "1F3864"
LIGHT_BLUE = "D9EAF7"
LIGHT_RED = "FCE4E4"
LIGHT_PURPLE = "F0E6F6"
LIGHT_GREEN = "E2F0D9"
LIGHT_YELLOW = "FFF2CC"
LIGHT_GREY = "E7E6E6"


_GAP_RANK = {
    Bucket.CRITICAL_GAP: 0,
    Bucket.NOTABLE_GAP: 1,
    Bucket.STANDARD_GAP: 2,
    Bucket.MARKET_VARIATION: 3,
}

_ACTION_FILL = {
    Bucket.CRITICAL_GAP: "F4CCCC",
    Bucket.NOTABLE_GAP: "FCE4D6",
    Bucket.STANDARD_GAP: LIGHT_YELLOW,
    Bucket.MARKET_VARIATION: LIGHT_GREY,
}

_ACTION_TEXT = {
    Bucket.CRITICAL_GAP: "Validate with the job owner and add if mandatory.",
    Bucket.NOTABLE_GAP: "Review applicability with the job owner.",
    Bucket.STANDARD_GAP: "Consider aligning with the standard job template.",
    Bucket.MARKET_VARIATION: (
        "Confirm whether this is an intentional company-specific difference."
    ),
}


def _header(sheet, row: int, labels: Iterable[str]) -> None:
    for column, label in enumerate(labels, start=1):
        cell = sheet.cell(row=row, column=column, value=label)
        cell.font = Font(name="Calibri", bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor=NAVY)
        cell.alignment = Alignment(
            horizontal="center", vertical="center", wrap_text=True
        )


def _percent(value: Optional[float]) -> str:
    return "Not measurable" if value is None else f"{value * 100:.0f}%"


def _fit(sheet, widths) -> None:
    for column, width in enumerate(widths, start=1):
        sheet.column_dimensions[get_column_letter(column)].width = width
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions


def _direction_summary(report, direction: Direction) -> tuple[str, int]:
    """Render one readable, category-grouped direction into a single cell."""

    findings = [
        finding
        for finding in report.findings
        if finding.direction == direction
    ]

    if not findings:
        message = (
            "No missing requirements compared with the other jobs."
            if direction == Direction.M_MINUS
            else "No additional requirements compared with the other jobs."
        )
        return message, 1

    lines = []
    for category in CATEGORY_ORDER:
        category_findings = [
            finding
            for finding in findings
            if finding.category == category
        ]
        if not category_findings:
            continue

        lines.append(category.value.upper())
        for finding in category_findings:
            benchmark_prefix = (
                "seen in"
                if direction == Direction.M_MINUS
                else "also seen in"
            )
            benchmark = (
                f"{benchmark_prefix} {finding.support_count} of "
                f"{finding.applicable_reference_count} other jobs"
                if finding.applicable_reference_count
                else "benchmark availability not measurable"
            )
            priority = BUCKET_LABEL[finding.bucket]
            lines.append(f"   • {finding.label}")
            lines.append(f"      {priority} · {benchmark}")

    # Excel cells have a 32,767-character limit.  Extremely large groups
    # retain the complete row-level detail in "Per-job Differences".
    text = "\n".join(lines)
    if len(text) > 32_000:
        text = (
            text[:31_900].rsplit("\n", 1)[0]
            + "\n… More items are available in Per-job Differences."
        )

    return text, text.count("\n") + 1


def _company_label(report, index: int) -> str:
    return report.new_company_code or f"Job {index}"


def _action_plan_rows(group: JobGroupComparison) -> list[dict]:
    """Aggregate repeated per-company gaps into one row per concept."""

    companies = [
        _company_label(report, index)
        for index, report in enumerate(group.reports, start=1)
    ]
    evidence_lookups = [
        {item.concept_id: item for item in report.evidence}
        for report in group.reports
    ]
    missing_finding_lookups = [
        {
            finding.concept_id: finding
            for finding in report.findings
            if finding.direction == Direction.M_MINUS
        }
        for report in group.reports
    ]

    concepts = []
    seen = set()
    for report in group.reports:
        for evidence in report.evidence:
            if evidence.concept_id not in seen:
                seen.add(evidence.concept_id)
                concepts.append(evidence)

    rows = []
    for concept in concepts:
        present_companies = []
        missing_companies = []
        missing_findings = []

        for company, evidence_lookup, finding_lookup in zip(
            companies,
            evidence_lookups,
            missing_finding_lookups,
        ):
            evidence = evidence_lookup.get(concept.concept_id)
            if evidence is None:
                continue
            if evidence.new_job_has:
                present_companies.append(company)
            else:
                missing_companies.append(company)
                finding = finding_lookup.get(concept.concept_id)
                if finding is not None:
                    missing_findings.append(finding)

        # Fully aligned concepts need no standardisation decision.
        if not present_companies or not missing_companies:
            continue

        missing_findings.sort(
            key=lambda finding: (
                _GAP_RANK.get(finding.bucket, 99),
                -finding.priority_score,
            )
        )
        leading = missing_findings[0] if missing_findings else None
        bucket = leading.bucket if leading is not None else Bucket.MARKET_VARIATION
        impact = next(
            (
                finding.business_impact
                for finding in missing_findings
                if finding.business_impact
            ),
            "No additional business-impact statement was generated.",
        )
        total = len(present_companies) + len(missing_companies)
        coverage = len(present_companies) / total if total else None

        rows.append(
            {
                "bucket": bucket,
                "category": concept.category,
                "requirement": concept.label,
                "missing": missing_companies,
                "present": present_companies,
                "coverage": coverage,
                "impact": impact,
                "action": _ACTION_TEXT[bucket],
            }
        )

    category_rank = {
        category: index
        for index, category in enumerate(CATEGORY_ORDER)
    }
    rows.sort(
        key=lambda item: (
            _GAP_RANK.get(item["bucket"], 99),
            -len(item["missing"]),
            category_rank.get(item["category"], 99),
            item["requirement"].casefold(),
        )
    )
    return rows


def _heatmap_fill(score: Optional[float]) -> str:
    if score is None:
        return LIGHT_GREY
    if score >= 0.8:
        return LIGHT_GREEN
    if score >= 0.5:
        return LIGHT_YELLOW
    return LIGHT_RED


def generate_group_summary_excel(group: JobGroupComparison) -> bytes:
    """Create a compact decision-first workbook for the whole group."""

    workbook = Workbook()
    overview = workbook.active
    overview.title = "Group Overview"
    overview.sheet_view.showGridLines = False

    _header(
        overview,
        1,
        [
            "Company",
            "Job title",
            "Job code",
            "Compared with",
            "Alignment",
            "Missing requirements",
            "Additional requirements",
            "Critical gaps",
            "Notable gaps",
            "Confidence",
        ],
    )

    for row, report in enumerate(group.reports, start=2):
        missing = sum(
            1 for finding in report.findings
            if finding.direction == Direction.M_MINUS
        )
        additional = sum(
            1 for finding in report.findings
            if finding.direction == Direction.M_PLUS
        )
        values = [
            report.new_company_code or "Not specified",
            report.new_job_title or "Not specified",
            report.new_job_code or "Not specified",
            report.reference_jobs_count,
            _percent(report.alignment_score),
            missing,
            additional,
            report.count_of(Bucket.CRITICAL_GAP),
            report.count_of(Bucket.NOTABLE_GAP),
            _percent(report.confidence),
        ]
        for column, value in enumerate(values, start=1):
            cell = overview.cell(row=row, column=column, value=value)
            cell.alignment = Alignment(vertical="top", wrap_text=True)
            if column == 6 and missing:
                cell.fill = PatternFill("solid", fgColor=LIGHT_RED)
            elif column == 7 and additional:
                cell.fill = PatternFill("solid", fgColor=LIGHT_PURPLE)

    _fit(overview, [18, 28, 18, 14, 14, 20, 21, 14, 14, 14])

    company_summary = workbook.create_sheet("Company Comparison")
    company_summary.sheet_view.showGridLines = False
    _header(
        company_summary,
        1,
        [
            "Company",
            "Job title",
            "Job code",
            "Compared with",
            "Missing count",
            "Missing compared with other companies",
            "Additional count",
            "Additional compared with other companies",
        ],
    )
    company_summary.row_dimensions[1].height = 32

    for row, report in enumerate(group.reports, start=2):
        missing_summary, missing_lines = _direction_summary(
            report,
            Direction.M_MINUS,
        )
        additional_summary, additional_lines = _direction_summary(
            report,
            Direction.M_PLUS,
        )
        missing_count = sum(
            1
            for finding in report.findings
            if finding.direction == Direction.M_MINUS
        )
        additional_count = sum(
            1
            for finding in report.findings
            if finding.direction == Direction.M_PLUS
        )
        values = [
            report.new_company_code or "Not specified",
            report.new_job_title or "Not specified",
            report.new_job_code or "Not specified",
            report.reference_jobs_count,
            missing_count,
            missing_summary,
            additional_count,
            additional_summary,
        ]

        for column, value in enumerate(values, start=1):
            cell = company_summary.cell(row=row, column=column, value=value)
            cell.alignment = Alignment(
                vertical="top",
                horizontal="left",
                wrap_text=True,
            )

        company_summary.cell(row=row, column=1).font = Font(
            name="Calibri",
            bold=True,
            color=NAVY,
        )
        company_summary.cell(row=row, column=5).alignment = Alignment(
            horizontal="center",
            vertical="top",
        )
        company_summary.cell(row=row, column=6).fill = PatternFill(
            "solid",
            fgColor=LIGHT_RED if missing_count else LIGHT_GREEN,
        )
        company_summary.cell(row=row, column=7).alignment = Alignment(
            horizontal="center",
            vertical="top",
        )
        company_summary.cell(row=row, column=8).fill = PatternFill(
            "solid",
            fgColor=LIGHT_PURPLE if additional_count else LIGHT_BLUE,
        )
        company_summary.row_dimensions[row].height = min(
            360,
            max(36, max(missing_lines, additional_lines) * 15),
        )

    _fit(company_summary, [18, 30, 18, 16, 15, 78, 16, 78])

    action_plan = workbook.create_sheet("Standardization Action Plan")
    action_plan.sheet_view.showGridLines = False
    _header(
        action_plan,
        1,
        [
            "Rank",
            "Priority",
            "Category",
            "Requirement",
            "Companies missing it",
            "Companies containing it",
            "Group coverage",
            "Why it matters",
            "Recommended decision",
            "Decision",
            "Owner",
            "Due date",
            "Notes",
        ],
    )
    action_plan.row_dimensions[1].height = 34

    action_rows = _action_plan_rows(group)
    for row, item in enumerate(action_rows, start=2):
        bucket = item["bucket"]
        fill = _ACTION_FILL[bucket]
        values = [
            row - 1,
            BUCKET_LABEL[bucket],
            item["category"].value,
            item["requirement"],
            "\n".join(f"• {company}" for company in item["missing"]),
            "\n".join(f"• {company}" for company in item["present"]),
            (
                f"{_percent(item['coverage'])} "
                f"({len(item['present'])} of "
                f"{len(item['present']) + len(item['missing'])})"
            ),
            item["impact"],
            item["action"],
            "",
            "",
            "",
            "",
        ]
        for column, value in enumerate(values, start=1):
            cell = action_plan.cell(row=row, column=column, value=value)
            cell.alignment = Alignment(
                vertical="top",
                horizontal="center" if column in (1, 7, 12) else "left",
                wrap_text=True,
            )
            cell.fill = PatternFill(
                "solid",
                fgColor=LIGHT_YELLOW if column >= 10 else fill,
            )
            if column in (2, 4):
                cell.font = Font(name="Calibri", bold=True)

        line_count = max(len(item["missing"]), len(item["present"]), 2)
        action_plan.row_dimensions[row].height = min(
            120,
            max(36, line_count * 16),
        )

    if not action_rows:
        action_plan.cell(
            row=2,
            column=1,
            value="No standardization differences were identified.",
        )

    _fit(
        action_plan,
        [8, 23, 24, 38, 25, 25, 18, 42, 42, 22, 20, 16, 32],
    )
    action_plan.freeze_panes = "D2"

    heatmap = workbook.create_sheet("Company Category Heatmap")
    heatmap.sheet_view.showGridLines = False
    _header(
        heatmap,
        1,
        [
            "Company",
            "Job title",
            "Overall alignment",
            *[category.value for category in CATEGORY_ORDER],
            "Missing total",
            "Critical gaps",
        ],
    )
    heatmap.row_dimensions[1].height = 42

    for row, report in enumerate(group.reports, start=2):
        company = _company_label(report, row - 1)
        missing_total = sum(
            1
            for finding in report.findings
            if finding.direction == Direction.M_MINUS
        )
        category_lookup = {
            summary.category: summary
            for summary in report.category_summaries
        }

        heatmap.cell(row=row, column=1, value=company)
        heatmap.cell(
            row=row,
            column=2,
            value=report.new_job_title or "Not specified",
        )
        overall = heatmap.cell(
            row=row,
            column=3,
            value=_percent(report.alignment_score),
        )
        overall.fill = PatternFill(
            "solid",
            fgColor=_heatmap_fill(report.alignment_score),
        )

        for offset, category in enumerate(CATEGORY_ORDER, start=4):
            summary = category_lookup.get(category)
            score = summary.alignment_score if summary is not None else None
            if summary is None:
                value = "N/A"
            else:
                value = (
                    f"{_percent(score)}\n"
                    f"{summary.m_minus_count} missing · "
                    f"{summary.m_plus_count} extra"
                )
            cell = heatmap.cell(row=row, column=offset, value=value)
            cell.fill = PatternFill(
                "solid",
                fgColor=_heatmap_fill(score),
            )

        missing_cell = heatmap.cell(
            row=row,
            column=4 + len(CATEGORY_ORDER),
            value=missing_total,
        )
        critical_cell = heatmap.cell(
            row=row,
            column=5 + len(CATEGORY_ORDER),
            value=report.count_of(Bucket.CRITICAL_GAP),
        )
        if missing_total:
            missing_cell.fill = PatternFill("solid", fgColor=LIGHT_RED)
        if critical_cell.value:
            critical_cell.fill = PatternFill("solid", fgColor="F4CCCC")

        for column in range(1, 6 + len(CATEGORY_ORDER)):
            cell = heatmap.cell(row=row, column=column)
            cell.alignment = Alignment(
                horizontal="center" if column != 2 else "left",
                vertical="center",
                wrap_text=True,
            )
        heatmap.cell(row=row, column=1).font = Font(
            name="Calibri",
            bold=True,
            color=NAVY,
        )
        heatmap.row_dimensions[row].height = 44

    _fit(
        heatmap,
        [18, 30, 18, *([22] * len(CATEGORY_ORDER)), 15, 15],
    )
    heatmap.freeze_panes = "C2"

    differences = workbook.create_sheet("Per-job Differences")
    differences.sheet_view.showGridLines = False
    _header(
        differences,
        1,
        [
            "Company",
            "Category",
            "Difference",
            "Requirement",
            "Other jobs containing it",
            "Other applicable jobs",
            "Prevalence",
            "Priority",
            "Recommended action",
        ],
    )
    row = 2
    for report in group.reports:
        # Keep all missing rows together, followed by all additional rows.
        # The engine's importance order remains intact within each block.
        ordered_findings = [
            finding
            for direction in (Direction.M_MINUS, Direction.M_PLUS)
            for finding in report.findings
            if finding.direction == direction
        ]
        for finding in ordered_findings:
            difference = (
                "Missing from this job"
                if finding.direction == Direction.M_MINUS
                else "Additional in this job"
            )
            values = [
                report.new_company_code or "Not specified",
                finding.category.value,
                difference,
                finding.label,
                finding.support_count,
                finding.applicable_reference_count,
                finding.prevalence_text,
                finding.bucket.value.replace("_", " ").title(),
                finding.action_text,
            ]
            for column, value in enumerate(values, start=1):
                cell = differences.cell(row=row, column=column, value=value)
                cell.alignment = Alignment(vertical="top", wrap_text=True)
                cell.fill = PatternFill(
                    "solid",
                    fgColor=(
                        LIGHT_RED
                        if finding.direction == Direction.M_MINUS
                        else LIGHT_PURPLE
                    ),
                )
                if column == 3:
                    cell.font = Font(
                        name="Calibri",
                        bold=True,
                        color=(
                            "9C0006"
                            if finding.direction == Direction.M_MINUS
                            else "7030A0"
                        ),
                    )
            row += 1
    if row == 2:
        differences.cell(row=2, column=1, value="No differences found.")
    _fit(differences, [18, 24, 23, 42, 20, 20, 17, 20, 28])

    matrix = workbook.create_sheet("Concept Coverage")
    matrix.sheet_view.showGridLines = False
    companies = [
        report.new_company_code or f"Job {index}"
        for index, report in enumerate(group.reports, start=1)
    ]
    _header(matrix, 1, ["Category", "Concept", *companies])

    evidence_by_report = [
        {evidence.concept_id: evidence for evidence in report.evidence}
        for report in group.reports
    ]
    concept_order = []
    seen = set()
    for report in group.reports:
        for evidence in report.evidence:
            if evidence.concept_id not in seen:
                seen.add(evidence.concept_id)
                concept_order.append(evidence)

    for row, concept in enumerate(concept_order, start=2):
        matrix.cell(row=row, column=1, value=concept.category.value)
        matrix.cell(row=row, column=2, value=concept.label)
        for report_index, lookup in enumerate(evidence_by_report, start=3):
            evidence = lookup.get(concept.concept_id)
            present = bool(evidence and evidence.new_job_has)
            cell = matrix.cell(
                row=row,
                column=report_index,
                value="Present" if present else "Missing",
            )
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.fill = PatternFill(
                "solid", fgColor=LIGHT_BLUE if present else LIGHT_RED
            )

    _fit(matrix, [25, 45, *([18] * len(companies))])

    buffer = BytesIO()
    workbook.save(buffer)
    workbook.close()
    payload = buffer.getvalue()
    buffer.close()
    if not payload:
        raise RuntimeError("Group Excel generation produced no data.")
    return payload


__all__ = ["generate_group_summary_excel"]
