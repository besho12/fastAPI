
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_user
from app.database.connection import get_db
from app.database.models import JobComparison

from app.history.repository import HistoryRepository
from app.history.schemas import (
    ComparisonHistoryDeleteResponse,
    ComparisonHistoryDetail,
    ComparisonHistoryItem,
    JobCodeListItem,
    JobHistoryDeleteResponse,
    JobHistoryDetail,
    JobHistoryItem,
)


router = APIRouter(
    prefix="/api/history",
    tags=["History"],
)


# ============================================================
# COMPARISON HELPERS
# ============================================================

def _get_report(
    comparison: JobComparison,
):
    """
    Return the first report belonging to this comparison.
    """

    if not comparison.reports:
        return None

    return comparison.reports[0]


def _build_history_item(
    comparison: JobComparison,
) -> ComparisonHistoryItem:

    job = comparison.new_job

    reference_jobs = (
        comparison.reference_jobs
        if isinstance(comparison.reference_jobs, list)
        else []
    )

    return ComparisonHistoryItem(
        comparison_id=comparison.id,

        job_code=(
            comparison.job_code
            or job.job_code
            or ""
        ),

        company_code=(
            job.company_code
            or ""
        ),

        job_title=job.job_title,

        department=job.department,

        reference_jobs_count=len(reference_jobs),

        status=comparison.status,

        created_at=comparison.created_at,
    )


def _build_history_detail(
    comparison: JobComparison,
) -> ComparisonHistoryDetail:

    job = comparison.new_job

    reference_jobs = (
        comparison.reference_jobs
        if isinstance(comparison.reference_jobs, list)
        else []
    )

    report = _get_report(comparison)

    return ComparisonHistoryDetail(
        comparison_id=comparison.id,

        job_code=(
            comparison.job_code
            or job.job_code
            or ""
        ),

        company_code=(
            job.company_code
            or ""
        ),

        job_title=job.job_title,

        department=job.department,

        reference_jobs_count=len(reference_jobs),

        status=comparison.status,

        created_at=comparison.created_at,

        updated_at=comparison.updated_at,

        reference_jobs=reference_jobs,

        report_id=(
            report.id
            if report is not None
            else None
        ),

        report_type=(
            report.report_type
            if report is not None
            else None
        ),

        report_status=(
            report.status
            if report is not None
            else None
        ),

        confidence=(
            report.confidence
            if report is not None
            else None
        ),

        is_substantive_role_change=(
            report.is_substantive_role_change
            if report is not None
            else None
        ),

        report_data=(
            report.report_data
            if report is not None
            else None
        ),
    )


# ============================================================
# JOB HELPERS
# ============================================================

def _build_job_history_item(
    job,
) -> JobHistoryItem:

    return JobHistoryItem(
        job_id=job.job_id,

        job_code=job.job_code,

        company_code=job.company_code,

        job_title=job.job_title,

        department=job.department,

        original_file_name=job.original_file_name,

        input_type=job.input_type,

        language=job.language,

        created_at=job.created_at,
    )


def _build_job_history_detail(
    job,
) -> JobHistoryDetail:

    return JobHistoryDetail(
        job_id=job.job_id,

        job_code=job.job_code,

        company_code=job.company_code,

        job_title=job.job_title,

        department=job.department,

        grade=job.grade,

        reports_to=job.reports_to,

        number_of_job_holders=job.number_of_job_holders,

        number_of_direct_reports=job.number_of_direct_reports,

        creation_date=job.creation_date,

        company_industry=job.company_industry,

        input_type=job.input_type,

        original_file_name=job.original_file_name,

        language=job.language,

        job_purpose=job.job_purpose,

        raw_text=job.raw_text,

        created_at=job.created_at,

        record_created_at=job.record_created_at,

        record_updated_at=job.record_updated_at,
    )


# ============================================================
# COMPARISON HISTORY
# ============================================================

@router.get(
    "/comparisons",
    response_model=list[ComparisonHistoryItem],
)
async def get_comparison_history(
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Return comparison operations only.
    """

    repository = HistoryRepository(db)

    comparisons = repository.get_all_comparisons()

    return [
        _build_history_item(comparison)
        for comparison in comparisons
    ]


# ============================================================
# ONE COMPARISON
# ============================================================

@router.get(
    "/comparisons/{comparison_id}",
    response_model=ComparisonHistoryDetail,
)
async def get_comparison_history_detail(
    comparison_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Return the complete stored result of one comparison.
    """

    repository = HistoryRepository(db)

    comparison = repository.get_comparison(
        comparison_id
    )

    if comparison is None:
        raise HTTPException(
            status_code=404,
            detail="Comparison history not found.",
        )

    return _build_history_detail(
        comparison
    )


# ============================================================
# DELETE COMPARISON
# ============================================================

@router.delete(
    "/comparisons/{comparison_id}",
    response_model=ComparisonHistoryDeleteResponse,
)
async def delete_comparison_history(
    comparison_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Delete ONE comparison operation.

    The underlying Job remains in the database.
    """

    repository = HistoryRepository(db)

    success = repository.delete_comparison(
        comparison_id
    )

    if not success:
        raise HTTPException(
            status_code=404,
            detail="Comparison history not found.",
        )

    return ComparisonHistoryDeleteResponse(
        success=True,

        comparison_id=comparison_id,

        message="Comparison history deleted successfully.",
    )


# ============================================================
# JOB CODES
# ============================================================

@router.get(
    "/jobs/job-codes",
    response_model=list[JobCodeListItem],
)
async def get_job_codes(
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Return all unique Job Codes.

    Used by the Job Code dropdown in the new History screen.
    """

    repository = HistoryRepository(db)

    job_codes = repository.get_all_job_codes()

    return [
        JobCodeListItem(
            job_code=job_code
        )
        for job_code in job_codes
    ]


# ============================================================
# JOBS BY JOB CODE
# ============================================================

@router.get(
    "/jobs-by-code/{job_code}",
    response_model=list[JobHistoryItem],
)

async def get_jobs_by_job_code(
    job_code: str,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Return all Jobs belonging to one Job Code.

    The Flutter History screen will use this endpoint
    to display the company/job cards.
    """

    repository = HistoryRepository(db)

    jobs = repository.get_jobs_by_job_code(
        job_code
    )

    return [
        _build_job_history_item(job)
        for job in jobs
    ]


# ============================================================
# ONE JOB DETAIL
# ============================================================

@router.get(
    "/jobs/detail/{job_id}",
    response_model=JobHistoryDetail,
)
async def get_job_detail(
    job_id: str,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Return the complete details of one Job.

    Includes the original raw text.
    """

    repository = HistoryRepository(db)

    job = repository.get_job(
        job_id
    )

    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Job not found.",
        )

    return _build_job_history_detail(
        job
    )


# ============================================================
# DELETE JOB
# ============================================================

@router.delete(
    "/jobs/{job_id}",
    response_model=JobHistoryDeleteResponse,
)
async def delete_job(
    job_id: str,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Delete one Job completely.

    Because the Job model has cascade relationships,
    related requirements, responsibilities, additional
    information, comparisons and reports may also be deleted.
    """

    repository = HistoryRepository(db)

    success = repository.delete_job(
        job_id
    )

    if not success:
        raise HTTPException(
            status_code=404,
            detail="Job not found.",
        )

    return JobHistoryDeleteResponse(
        success=True,

        job_id=job_id,

        message="Job deleted successfully.",
    )

