# app/comparison.py
#
# MAINTAINER NOTE: as of this review, app/pipeline.py calls
# LLMService.compare_and_generate_report() directly and does not go
# through compare()/compare_pair() in this module. This file's two
# bugs (an undefined `report` variable, and a missing report=report
# in the success return) were therefore never hit in production -
# they are fixed here anyway so this module is correct if it is ever
# wired back in, and so it stops being a trap for the next person who
# reads it and assumes it is the live path.

from __future__ import annotations

import logging
import re
import unicodedata
from typing import List, Optional

from app.exceptions import (
    ComparisonError,
    InvalidInputError,
    JobAIException,
)
from app.llm import LLMService
from app.schemas import (
    ComparisonResult,
    ComparisonSet,
    ComparisonStatus,
    GeminiComparisonReport,
    JobDescription,
)

logger = logging.getLogger(__name__)

_YEAR_PATTERNS = [
    re.compile(
        r"(?<!\d)(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?|yr)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?<!\d)(\d+(?:\.\d+)?)\s*(?:سنوات|سنة|سنين)",
        re.IGNORECASE,
    ),
]


# ==========================================================================
# TEXT UTILITIES
# ==========================================================================


def normalize_text(value: object) -> str:
    if value is None:
        return ""

    text = str(value).strip()

    if not text:
        return ""

    text = unicodedata.normalize("NFKD", text)

    text = "".join(
        char
        for char in text
        if not unicodedata.combining(char)
    )

    text = (
        text.replace("أ", "ا")
        .replace("إ", "ا")
        .replace("آ", "ا")
        .replace("ٱ", "ا")
    )

    return " ".join(text.split()).casefold()


def extract_years(
    value: Optional[str],
) -> Optional[float]:
    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    matches: List[float] = []

    for pattern in _YEAR_PATTERNS:
        for match in pattern.findall(text):
            try:
                matches.append(float(match))
            except (TypeError, ValueError):
                continue

    unique_values = set(matches)

    if len(unique_values) != 1:
        return None

    return next(iter(unique_values))


def extract_years_from_list(
    values: Optional[List[str]],
) -> Optional[float]:
    if not values:
        return None

    extracted: List[float] = []

    for value in values:
        years = extract_years(value)

        if years is not None:
            extracted.append(years)

    unique_values = set(extracted)

    if len(unique_values) != 1:
        return None

    return next(iter(unique_values))


# ==========================================================================
# VALIDATION
# ==========================================================================


def _validate_job(
    job: object,
    field_name: str,
) -> None:
    if job is None:
        raise InvalidInputError(
            f"{field_name} must not be None",
            details={
                "stage": "comparison",
                "field": field_name,
            },
        )

    if not isinstance(job, JobDescription):
        raise InvalidInputError(
            f"{field_name} must be a JobDescription instance",
            details={
                "stage": "comparison",
                "field": field_name,
                "type": type(job).__name__,
            },
        )

    if not job.job_id or not str(job.job_id).strip():
        raise InvalidInputError(
            f"{field_name}.job_id must be a non-empty string",
            details={
                "stage": "comparison",
                "field": field_name,
            },
        )


def _validate_same_group(
    new_job: JobDescription,
    reference_job: JobDescription,
) -> None:
    new_code = new_job.job_information.job_code
    reference_code = reference_job.job_information.job_code

    if not new_code or not reference_code:
        return

    if new_code != reference_code:
        raise ComparisonError(
            "reference_job belongs to a different job group than new_job.",
            details={
                "new_job_id": new_job.job_id,
                "new_job_code": new_code,
                "reference_job_id": reference_job.job_id,
                "reference_job_code": reference_code,
            },
        )


def _validate_reference_jobs(
    new_job: JobDescription,
    reference_jobs: List[JobDescription],
) -> None:
    """
    Validate all historical/reference jobs before sending the complete
    population to Gemini.

    This function performs structural validation only.

    It does NOT:
        - calculate semantic similarity
        - calculate majority
        - classify M+
        - classify M-
        - compare job requirements
    """

    for index, reference_job in enumerate(reference_jobs, start=1):
        _validate_job(
            reference_job,
            f"reference_jobs[{index - 1}]",
        )

        _validate_same_group(
            new_job,
            reference_job,
        )


# ==========================================================================
# LEGACY SINGLE-PAIR ENTRY POINT
# ==========================================================================


def compare_pair(
    new_job: JobDescription,
    reference_job: JobDescription,
    llm_service: Optional[LLMService] = None,
) -> ComparisonResult:
    """
    Legacy single-reference comparison.

    The main consolidated comparison flow should use `compare()`.

    This function is intentionally kept for backward compatibility with
    any existing callers that still require a single pair.
    """

    _validate_job(new_job, "new_job")
    _validate_job(reference_job, "reference_job")
    _validate_same_group(new_job, reference_job)

    logger.info(
        "Gemini semantic pair comparison: %s <-> %s",
        new_job.job_id,
        reference_job.job_id,
    )

    service = llm_service or LLMService()

    try:
        result = service.compare_jobs(
            new_job=new_job,
            reference_job=reference_job,
        )
    except JobAIException:
        raise
    except Exception as exc:
        raise ComparisonError(
            "Gemini semantic comparison failed.",
            details={
                "stage": "comparison",
                "new_job_id": new_job.job_id,
                "reference_job_id": reference_job.job_id,
                "error": str(exc),
            },
        ) from exc

    result.new_job_id = new_job.job_id
    result.new_job_code = (
        new_job.job_information.job_code or ""
    )
    result.reference_job_id = reference_job.job_id
    result.reference_job_code = (
        reference_job.job_information.job_code or ""
    )
    result.status = ComparisonStatus.COMPARED

    return result


# ==========================================================================
# CONSOLIDATED COMPARISON
# ==========================================================================


def compare(
    new_job: JobDescription,
    reference_jobs: List[JobDescription],
    llm_service: Optional[LLMService] = None,
) -> ComparisonSet:
    """
    Compare ONE new job against the COMPLETE historical population.

    IMPORTANT:
        Gemini receives the new job and ALL historical jobs in a single
        consolidated comparison.

    Gemini is responsible for:
        - semantic understanding
        - semantic equivalence
        - historical concept discovery
        - M+ detection
        - M- detection
        - strict-majority reasoning
        - document-level evidence
        - Arabic/English semantic matching

    Python is responsible only for:
        - input validation
        - group validation
        - calling Gemini
        - returning Gemini's structured result

    Python does NOT perform semantic reasoning.
    """

    _validate_job(new_job, "new_job")

    if reference_jobs is None:
        raise InvalidInputError(
            "reference_jobs must not be None",
            details={
                "stage": "comparison",
                "new_job_id": new_job.job_id,
            },
        )

    if not isinstance(reference_jobs, list):
        raise InvalidInputError(
            "reference_jobs must be a list",
            details={
                "stage": "comparison",
                "new_job_id": new_job.job_id,
                "type": type(reference_jobs).__name__,
            },
        )

    new_job_code = (
        new_job.job_information.job_code
        or ""
    )

    # ------------------------------------------------------------------
    # NO HISTORICAL REFERENCES
    # ------------------------------------------------------------------

    if not reference_jobs:
        logger.info(
            "No historical reference jobs for new job %s.",
            new_job.job_id,
        )

        # BUGFIX: this branch previously referenced a `report`
        # variable that was never defined here (NameError on every
        # call with an empty reference_jobs list). Build an explicit
        # empty report instead, matching the shape
        # pipeline.py._compare_and_generate_report already uses for
        # the same situation.
        empty_report = GeminiComparisonReport(
            status=ComparisonStatus.NEW_JOB_CODE,
            reference_document_count=0,
            sections=[],
            semantic_changes=[],
            overall_summary=(
                "This job_code has no historical benchmark jobs "
                "available for comparison yet."
            ),
        )

        return ComparisonSet(
            status=ComparisonStatus.COMPARED,
            new_job_id=new_job.job_id,
            new_job_code=new_job_code,
            reference_jobs_count=len(reference_jobs),
            report=empty_report,
            comparisons=[],
        )

    # ------------------------------------------------------------------
    # VALIDATE COMPLETE HISTORICAL POPULATION
    # ------------------------------------------------------------------

    _validate_reference_jobs(
        new_job=new_job,
        reference_jobs=reference_jobs,
    )

    service = llm_service or LLMService()

    logger.info(
        "Starting consolidated Gemini comparison. "
        "new_job_id=%s | job_code=%s | historical_count=%s",
        new_job.job_id,
        new_job_code,
        len(reference_jobs),
    )

    # ------------------------------------------------------------------
    # SEND ALL HISTORICAL JOBS TO GEMINI AT ONCE
    # ------------------------------------------------------------------

    try:
        report = service.compare_and_generate_report(
            new_job=new_job,
            reference_jobs=reference_jobs,
        )

    except JobAIException:
        raise

    except Exception as exc:
        raise ComparisonError(
            "Gemini consolidated semantic comparison failed.",
            details={
                "stage": "comparison",
                "new_job_id": new_job.job_id,
                "new_job_code": new_job_code,
                "reference_jobs_count": len(reference_jobs),
                "error": str(exc),
            },
        ) from exc

    # ------------------------------------------------------------------
    # RETURN THE CONSOLIDATED GEMINI REPORT
    # ------------------------------------------------------------------

    logger.info(
        "Consolidated Gemini comparison completed. "
        "new_job_id=%s | references=%s | semantic_changes=%s",
        new_job.job_id,
        len(reference_jobs),
        len(report.semantic_changes),
    )

    # BUGFIX: `report` (the actual Gemini comparison result computed
    # a few lines above) was previously never attached to the
    # returned ComparisonSet - report defaults to None on the model,
    # so every caller of compare() got status/counts back but the
    # entire semantic comparison was silently dropped.
    return ComparisonSet(
        status=ComparisonStatus.COMPARED,
        new_job_id=new_job.job_id,
        new_job_code=new_job_code,
        reference_jobs_count=len(reference_jobs),
        report=report,
        comparisons=[],
    )


__all__ = [
    "normalize_text",
    "extract_years",
    "extract_years_from_list",
    "compare_pair",
    "compare",
]