
from __future__ import annotations

import asyncio
import os

from io import BytesIO

from typing import List, Optional

from fastapi import Depends, FastAPI, HTTPException

from fastapi.middleware.cors import CORSMiddleware

from fastapi.responses import StreamingResponse

from pydantic import BaseModel, Field

from sqlalchemy.orm import Session

from app.auth.router import router as auth_router

from app.auth.dependencies import get_current_user

from app.history.router import router as history_router

from app.updator.updator import router as update_router
from app.text_extraction.router import router as text_extraction_router

from app.pipeline import (
    JobComparisonPipeline,
    JobIngestionResult,
)

from app.schemas import InputType

from app.extraction import ExtractionService

from app.preprocessing import Preprocessor

from app.duplicate_detection import DuplicateDetectionService

from app.database.connection import SessionLocal, get_db

from app.database.repositories import (
    JobRepository,
    ComparisonRepository,
    ReportRepository,
)


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Job Comparison AI API",
    version="1.0.0",
)


# ============================================================
# AUTHENTICATION ROUTES
# ============================================================

app.include_router(auth_router)

app.include_router(history_router)


# ============================================================
# APPLICATION UPDATE ROUTE
# ============================================================

app.include_router(update_router)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ROOT / HEALTH
# ============================================================

@app.get("/")
async def root():
    return {
        "status": "ok",
        "service": "Job Comparison AI API",
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
    }
# ============================================================
# extract text
# ============================================================

app.include_router(text_extraction_router)

# ============================================================
# REQUEST MODELS
# ============================================================

class JobInput(BaseModel):
    """
    One Job Description already converted to text
    by the upstream backend.
    """

    raw_text: str = Field(
        ...,
        min_length=1,
        description="Job Description text",
    )

    input_type: InputType = InputType.TEXT

    original_file_name: Optional[str] = None


class BulkJobIngestionRequest(BaseModel):
    """
    Request used for bulk Job Description ingestion.

    The files are NOT uploaded here.

    The upstream backend has already converted
    each file into raw text.
    """

    jobs: List[JobInput] = Field(
        ...,
        min_length=1,
        description="List of Job Descriptions to ingest",
    )


class JobComparisonRequest(BaseModel):
    """
    Identifies the job-code group to compare.

    job_code:
        Defines the comparison group.

    company_code:
        Identifies the anchor Job inside that group.  It remains required
        for compatibility with the existing desktop application; the ZIP
        now contains an individual report for every Job in the group.
    """

    job_code: str = Field(
        ...,
        min_length=1,
    )

    company_code: str = Field(
        ...,
        min_length=1,
    )

    reference_company_codes: Optional[List[str]] = Field(
        default=None,
        description=(
            "Optional exact peer-company selection. When omitted, all other "
            "companies with the same job code are used."
        ),
    )

    include_discrepancy_report: bool = Field(
        default=True,
        description=(
            "Return the OpenAI-analyzed target-company discrepancy workbook "
            "as the only file in the ZIP."
        ),
    )


class JobAnalysisRequest(BaseModel):
    """
    Legacy single-job endpoint request.

    Kept for backward compatibility.
    """

    raw_text: str = Field(
        ...,
        min_length=1,
    )

    input_type: InputType = InputType.TEXT

    original_file_name: Optional[str] = None


# ============================================================
# SERVICES
# ============================================================

preprocessor = Preprocessor()

extraction_service = ExtractionService()

duplicate_detection_service = DuplicateDetectionService()


def _bulk_ingestion_concurrency() -> int:
    """Return a conservative, configurable ingestion worker limit."""

    raw_value = os.getenv("BULK_INGESTION_CONCURRENCY", "4")

    try:
        return max(1, int(raw_value))
    except (TypeError, ValueError):
        return 4


# ============================================================
# PIPELINE FACTORY
# ============================================================

def create_pipeline(db: Session) -> JobComparisonPipeline:
    """
    Create a JobComparisonPipeline using the current
    PostgreSQL database session.
    """

    job_repository = JobRepository(db)

    comparison_repository = ComparisonRepository(db)

    report_repository = ReportRepository(db)

    return JobComparisonPipeline(
        extraction_service=extraction_service,
        preprocessor=preprocessor,
        db_session=db,
        job_repository=job_repository,
        comparison_repository=comparison_repository,
        report_repository=report_repository,
        duplicate_detection_service=duplicate_detection_service,
    )


def _ingest_job_with_own_session(item: JobInput) -> JobIngestionResult:
    """Ingest one document in a worker-owned SQLAlchemy session.

    A Session must never be shared between the worker threads used by the
    bulk endpoint.  Reusing ``ingest_jobs`` for a one-item list also keeps
    the endpoint's existing per-file error response contract.
    """

    with SessionLocal() as worker_db:
        pipeline = create_pipeline(worker_db)
        return pipeline.ingest_jobs(
            jobs=[
                (
                    item.raw_text,
                    item.input_type,
                    item.original_file_name,
                )
            ]
        )[0]


# ============================================================
# BULK INGESTION
# ============================================================

@app.post("/api/jobs/bulk-upload")
async def bulk_upload_jobs(
    request: BulkJobIngestionRequest,
    current_user=Depends(get_current_user),
):
    """
    Bulk Job Description ingestion.

    Flow:

        raw_text
            ↓
        preprocessing
            ↓
        Gemini extraction
            ↓
        job_code + company_code
            ↓
        duplicate detection
            ↓
        PostgreSQL

    IMPORTANT:

    This endpoint DOES NOT perform comparison.

    It also DOES NOT generate Excel or ZIP files.
    """

    try:
        # Extraction is the expensive part of this endpoint (one Gemini
        # request per document).  Running independent documents serially
        # made a three-file save take roughly three times as long and could
        # exceed the reverse proxy timeout, surfacing as HTTP 504.  Bound the
        # concurrency to respect provider limits, and give every worker its
        # own SQLAlchemy Session because Session is not thread-safe.
        semaphore = asyncio.Semaphore(
            min(
                _bulk_ingestion_concurrency(),
                len(request.jobs),
            )
        )

        async def ingest_one(item: JobInput) -> JobIngestionResult:
            async with semaphore:
                return await asyncio.to_thread(
                    _ingest_job_with_own_session,
                    item,
                )

        results: List[JobIngestionResult] = list(
            await asyncio.gather(
                *(ingest_one(item) for item in request.jobs)
            )
        )

        stored_count = sum(
            1
            for result in results
            if result.status == "stored"
        )

        duplicate_count = sum(
            1
            for result in results
            if result.status == "duplicate"
        )

        failed_count = sum(
            1
            for result in results
            if result.status in {"failed", "error"}
        )

        return {
            "status": "completed",
            "total": len(results),
            "stored": stored_count,
            "duplicates": duplicate_count,
            "errors": failed_count,
            "results": [
                result.model_dump()
                for result in results
            ],
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


# ============================================================
# COMPARE SELECTED JOB
# ============================================================

@app.post("/api/jobs/compare")
async def compare_selected_job(
    request: JobComparisonRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Compare every Job having the requested job_code against all the others.

    Selection:

        job_code + company_code

    Comparison scope:

        same job_code

    The selected company remains the first report for backward-compatible
    history metadata.  The returned ZIP also contains a group overview and
    one complete workbook per Job.
    """

    try:
        job_code = request.job_code.strip().upper()

        company_code = request.company_code.strip().upper()

        if not job_code:
            raise HTTPException(
                status_code=400,
                detail="job_code cannot be empty.",
            )

        if not company_code:
            raise HTTPException(
                status_code=400,
                detail="company_code cannot be empty.",
            )

        pipeline = create_pipeline(db)

        result = pipeline.compare_selected_job(
            job_code=job_code,
            company_code=company_code,
            reference_company_codes=request.reference_company_codes,
            include_discrepancy_report=request.include_discrepancy_report,
        )

        if not result.zip_bytes:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Comparison finished but ZIP "
                    "was not generated."
                ),
            )

        zip_filename = (
            result.zip_filename
            or "job_comparison.zip"
        )

        return StreamingResponse(
            BytesIO(result.zip_bytes),
            media_type="application/zip",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="{zip_filename}"'
                )
            },
        )

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


# ============================================================
# LEGACY SINGLE-JOB ANALYZE ENDPOINT
# ============================================================

@app.post("/api/analyze")
async def analyze_document(
    request: JobAnalysisRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Legacy endpoint.

    Keeps the old behavior:

        raw_text
            ↓
        ingestion
            ↓
        comparison
            ↓
        Excel
            ↓
        ZIP

    New frontend integrations should use:

        POST /api/jobs/bulk-upload

    followed by:

        POST /api/jobs/compare
    """

    try:
        pipeline = create_pipeline(db)

        result = pipeline.run(
            raw_text=request.raw_text,
            input_type=request.input_type,
            original_file_name=request.original_file_name,
        )

        if not result.zip_bytes:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Pipeline finished but ZIP "
                    "was not generated."
                ),
            )

        zip_filename = (
            result.zip_filename
            or "job_comparison.zip"
        )

        return StreamingResponse(
            BytesIO(result.zip_bytes),
            media_type="application/zip",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="{zip_filename}"'
                )
            },
        )

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc
