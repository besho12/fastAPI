"""
report_builder.py
-----------------
"""

from __future__ import annotations

from typing import List

from app.schemas import (
    ComparisonStatus,
    ComparisonSummary,
    AggregatedListItem,
)


# ---------------------------------------------------------------------
# Report Models
# ---------------------------------------------------------------------

class ReportRow:
    """
    One row in the final comparison report.

    Example:

        label = "Education Level"
        content = "Your JD requires more Years of Experience (1 year)"
    """

    def __init__(
        self,
        label: str,
        content: str,
    ) -> None:
        self.label = label
        self.content = content


class ReportData:
    """
    Complete structured report before PDF rendering.
    """

    def __init__(
        self,
        title: str,
        job_code: str,
        reference_jobs_count: int,
        rows: List[ReportRow],
    ) -> None:
        self.title = title
        self.job_code = job_code
        self.reference_jobs_count = reference_jobs_count
        self.rows = rows


# ---------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------

def _format_percentage(value: float) -> str:
    """
    Format percentage without unnecessary decimal zeros.
    """
    if value == int(value):
        return f"{int(value)}%"

    return f"{value:.2f}%"


def _format_item(item: AggregatedListItem) -> str:
    """
    Format one aggregated list item.

    Example:
        Python (80%)
    """
    return f"{item.value} ({_format_percentage(item.percentage)})"


def _format_items(
    items: List[AggregatedListItem],
) -> str:
    """
    Convert aggregated items into bullet-point text.
    """

    if not items:
        return ""

    return "\n".join(
        f"- {_format_item(item)}"
        for item in items
    )


def _matching_text(
    match_percentage: float,
) -> str:
    """
    Human-readable scalar comparison result.
    """

    if match_percentage >= 100:
        return "All Participants are Matching"

    if match_percentage <= 0:
        return "No Participants are Matching"

    return (
        f"{_format_percentage(match_percentage)} "
        f"of Participants are Matching"
    )


# ---------------------------------------------------------------------
# List field formatter
# ---------------------------------------------------------------------

def _build_list_content(
    common: List[AggregatedListItem],
    new_only: List[AggregatedListItem],
    reference_only: List[AggregatedListItem],
) -> str:

    sections: List[str] = []

    # -------------------------------------------------------------
    # Common
    # -------------------------------------------------------------

    if common:
        sections.append(
            "All Participants are Matching:\n"
            + _format_items(common)
        )

    # -------------------------------------------------------------
    # New-job-only = M+
    # -------------------------------------------------------------

    if new_only:
        sections.append(
            "M+\n"
            "Your JD has additional requirements:\n"
            + _format_items(new_only)
        )

    # -------------------------------------------------------------
    # Reference-only = M-
    # -------------------------------------------------------------

    if reference_only:
        sections.append(
            "M-\n"
            "Your JD is missing the following:\n"
            + _format_items(reference_only)
        )

    if not sections:
        return "No comparison data available"

    return "\n\n".join(sections)


# ---------------------------------------------------------------------
# Experience
# ---------------------------------------------------------------------

def _build_experience_content(
    summary: ComparisonSummary,
) -> str:

    experience = summary.experience

    sections: List[str] = []

    # -------------------------------------------------------------
    # Numeric experience
    # -------------------------------------------------------------

    if (
        experience.new_years is not None
        and experience.reference_years_average is not None
    ):

        difference = experience.years_difference_average

        if difference is not None:

            if difference > 0:
                sections.append(
                    "Your JD requires more Years of Experience "
                    f"({difference:g} year(s) more)"
                )

            elif difference < 0:
                sections.append(
                    "Your JD requires fewer Years of Experience "
                    f"({abs(difference):g} year(s) less)"
                )

            else:
                sections.append(
                    "Years of Experience are Matching"
                )

        sections.append(
            f"Your JD: {experience.new_years:g} years"
        )

        sections.append(
            "Reference average: "
            f"{experience.reference_years_average:g} years"
        )

    # -------------------------------------------------------------
    # Text-based comparison
    # -------------------------------------------------------------

    list_content = _build_list_content(
        common=experience.common,
        new_only=experience.new_only,
        reference_only=experience.reference_only,
    )

    if list_content:
        sections.append(list_content)

    if not sections:
        return "No experience comparison data available"

    return "\n\n".join(sections)


# ---------------------------------------------------------------------
# Job Identity
# ---------------------------------------------------------------------

def _build_identity_rows(
    summary: ComparisonSummary,
) -> List[ReportRow]:

    rows: List[ReportRow] = []

    identity = summary.job_identity

    rows.append(
        ReportRow(
            "Department",
            _matching_text(
                identity.department.match_percentage
            ),
        )
    )

    rows.append(
        ReportRow(
            "Job Code",
            _matching_text(
                identity.job_code.match_percentage
            ),
        )
    )

    rows.append(
        ReportRow(
            "Position",
            _matching_text(
                identity.job_title.match_percentage
            ),
        )
    )

    return rows


# ---------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------

def build_report_data(
    summary: ComparisonSummary,
) -> ReportData:
    """
    Convert ComparisonSummary into structured report data.
    """

    if summary is None:
        raise ValueError(
            "summary must not be None"
        )

    if not isinstance(summary, ComparisonSummary):
        raise TypeError(
            "summary must be a ComparisonSummary instance"
        )

    # -------------------------------------------------------------
    # New Job Group
    # -------------------------------------------------------------

    if summary.status == ComparisonStatus.NEW_JOB_GROUP:

        return ReportData(
            title="Job Comparison Report",
            job_code=summary.new_job_code or "N/A",
            reference_jobs_count=0,
            rows=[
                ReportRow(
                    "Status",
                    "This is a new job group. "
                    "No historical benchmark comparison is available.",
                )
            ],
        )

    # -------------------------------------------------------------
    # Normal comparison
    # -------------------------------------------------------------

    rows: List[ReportRow] = []

    # Identity
    rows.extend(
        _build_identity_rows(summary)
    )

    # -------------------------------------------------------------
    # Job Purpose
    # -------------------------------------------------------------

    rows.append(
        ReportRow(
            "Job Purpose",
            _matching_text(
                summary.job_purpose.match_percentage
            ),
        )
    )

    # -------------------------------------------------------------
    # Knowledge / Experience
    # -------------------------------------------------------------

    rows.append(
        ReportRow(
            "Knowledge",
            "Years of Experience\n\n"
            + _build_experience_content(summary),
        )
    )

    # -------------------------------------------------------------
    # Education
    # -------------------------------------------------------------

    rows.append(
        ReportRow(
            "Education Level",
            _build_list_content(
                common=summary.education.common,
                new_only=summary.education.new_only,
                reference_only=summary.education.reference_only,
            ),
        )
    )

    # -------------------------------------------------------------
    # Skills
    # -------------------------------------------------------------

    rows.append(
        ReportRow(
            "Skills",
            _build_list_content(
                common=summary.skills.common,
                new_only=summary.skills.new_only,
                reference_only=summary.skills.reference_only,
            ),
        )
    )

    # -------------------------------------------------------------
    # Responsibilities
    # -------------------------------------------------------------

    rows.append(
        ReportRow(
            "Duties & Responsibilities",
            _build_list_content(
                common=summary.responsibilities.common,
                new_only=summary.responsibilities.new_only,
                reference_only=summary.responsibilities.reference_only,
            ),
        )
    )

    return ReportData(
        title="Job Comparison Report",
        job_code=summary.new_job_code or "N/A",
        reference_jobs_count=summary.reference_jobs_count,
        rows=rows,
    )


__all__ = [
    "ReportRow",
    "ReportData",
    "build_report_data",
]