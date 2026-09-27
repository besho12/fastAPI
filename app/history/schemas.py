
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel


# ============================================================
# COMPARISON HISTORY LIST ITEM
# ============================================================

class ComparisonHistoryItem(BaseModel):
    comparison_id: int

    job_code: str

    company_code: str

    job_title: Optional[str] = None

    department: Optional[str] = None

    reference_jobs_count: int

    status: str

    created_at: datetime


# ============================================================
# COMPARISON HISTORY DETAIL
# ============================================================

class ComparisonHistoryDetail(BaseModel):
    comparison_id: int

    job_code: str

    company_code: str

    job_title: Optional[str] = None

    department: Optional[str] = None

    reference_jobs_count: int

    status: str

    created_at: datetime

    updated_at: datetime

    reference_jobs: list[Any]

    report_id: Optional[int] = None

    report_type: Optional[str] = None

    report_status: Optional[str] = None

    confidence: Optional[float] = None

    is_substantive_role_change: Optional[bool] = None

    report_data: Optional[dict[str, Any]] = None


# ============================================================
# DELETE COMPARISON RESPONSE
# ============================================================

class ComparisonHistoryDeleteResponse(BaseModel):
    success: bool

    comparison_id: int

    message: str


# ============================================================
# JOB CODE LIST
# ============================================================

class JobCodeListItem(BaseModel):
    job_code: str


# ============================================================
# JOB HISTORY LIST ITEM
# ============================================================

class JobHistoryItem(BaseModel):
    """
    One company/job card shown under a selected Job Code.
    """

    job_id: str

    job_code: str

    company_code: str

    job_title: Optional[str] = None

    department: Optional[str] = None

    original_file_name: Optional[str] = None

    input_type: Optional[str] = None

    language: Optional[str] = None

    created_at: datetime


# ============================================================
# JOB HISTORY DETAIL
# ============================================================

class JobHistoryDetail(BaseModel):
    """
    Full details shown after opening one Job/Company card.
    """

    job_id: str

    job_code: str

    company_code: str

    job_title: Optional[str] = None

    department: Optional[str] = None

    grade: Optional[str] = None

    reports_to: Optional[str] = None

    number_of_job_holders: Optional[int] = None

    number_of_direct_reports: Optional[int] = None

    creation_date: Optional[Any] = None

    company_industry: Optional[str] = None

    input_type: Optional[str] = None

    original_file_name: Optional[str] = None

    language: Optional[str] = None

    job_purpose: Optional[str] = None

    raw_text: str

    created_at: datetime

    record_created_at: datetime

    record_updated_at: datetime


# ============================================================
# DELETE JOB RESPONSE
# ============================================================

class JobHistoryDeleteResponse(BaseModel):
    success: bool

    job_id: str

    message: str

