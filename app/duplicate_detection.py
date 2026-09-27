"""
app/duplicate_detection.py

Duplicate detection utilities for JobComparisonAI.

Business rules:
- Exact duplicate identity is scoped by:
      job_code + company_code + content_hash
  OR:
      job_code + company_code + raw_content_hash

- job_code is the comparison grouping key.
- company_code identifies the selected company's job.
- company_code does NOT restrict comparison references.
- The same job content can exist for different companies.
- Near-duplicate detection is kept only for backward compatibility.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

from pydantic import BaseModel

from app.config import settings
from app.schemas import JobDescription


if TYPE_CHECKING:
    from app.database.repositories import JobRepository
    from app.qdrant_service import QdrantService


logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_DUPLICATE_SIMILARITY_THRESHOLD: float = (
    settings.DUPLICATE_SIMILARITY_THRESHOLD
)


# ============================================================================
# Constants
# ============================================================================

_REQUIREMENT_CATEGORIES: tuple[str, ...] = (
    "education",
    "experience",
    "skills",
    "language",
    "computer",
    "soft_skills",
    "field_of_experience",
)

_ARABIC_DIACRITICS_PATTERN = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670"
    r"\u06D6-\u06DC\u06DF-\u06E8"
    r"\u06EA-\u06ED\u0640]"
)

_WHITESPACE_PATTERN = re.compile(r"\s+")


# ============================================================================
# Text normalization
# ============================================================================

def _normalize_arabic(text: str) -> str:
    """
    Normalize common Arabic character variations.
    """

    text = _ARABIC_DIACRITICS_PATTERN.sub("", text)

    text = re.sub(r"[إأآ]", "ا", text)

    text = text.replace("ى", "ي")
    text = text.replace("ة", "ه")
    text = text.replace("ؤ", "و")
    text = text.replace("ئ", "ي")

    return text


def _normalize_text(value: Optional[Any]) -> str:
    """
    Normalize a text value for deterministic hashing.
    """

    if value is None:
        return ""

    text = str(value)

    text = unicodedata.normalize(
        "NFKC",
        text,
    )

    text = _normalize_arabic(text)

    text = text.lower()

    text = _WHITESPACE_PATTERN.sub(
        " ",
        text,
    )

    return text.strip()


def _normalize_value(value: Any) -> Any:
    """
    Recursively normalize JSON-like values.

    Strings:
        normalized as text.

    Dictionaries:
        keys and values are normalized and keys are sorted.

    Lists:
        items are normalized and sorted so list ordering does not affect
        the resulting hash.

    Other values:
        returned unchanged.
    """

    if isinstance(value, str):
        return _normalize_text(value)

    if isinstance(value, dict):
        return {
            _normalize_text(str(key)): _normalize_value(val)
            for key, val in sorted(
                value.items(),
                key=lambda kv: str(kv[0]),
            )
        }

    if isinstance(value, list):
        normalized_items = [
            _normalize_value(item)
            for item in value
        ]

        return sorted(
            normalized_items,
            key=lambda item: json.dumps(
                item,
                sort_keys=True,
                ensure_ascii=False,
            ),
        )

    return value


# ============================================================================
# Canonical representation
# ============================================================================

def _build_canonical_dict(job: JobDescription) -> dict:
    """
    Build the normalized structured representation used to calculate
    content_hash.

    company_code is intentionally NOT included here.

    Why?

    Because content_hash represents the job content itself.

    The database determines duplicate identity using:

        job_code + company_code + content_hash

    Therefore the same content can have the same hash for two companies
    without being considered a duplicate.
    """

    requirements = job.requirements

    return {
        "company_name": _normalize_text(
            job.company.name
            if job.company
            else None
        ),

        "job_title": _normalize_text(
            job.job_information.job_title
            if job.job_information
            else None
        ),

        "department": _normalize_text(
            job.job_information.department
            if job.job_information
            else None
        ),

        "grade": _normalize_text(
            job.job_information.grade
            if job.job_information
            else None
        ),

        "job_purpose": _normalize_text(
            job.job_purpose
        ),

        "responsibilities": [
            _normalize_text(item)
            for item in (job.responsibilities or [])
        ],

        "requirements": {
            category: sorted(
                _normalize_text(item)
                for item in (
                    getattr(
                        requirements,
                        category,
                        None,
                    )
                    or []
                )
            )
            for category in _REQUIREMENT_CATEGORIES
        },

        "additional_information": _normalize_value(
            job.additional_information or {}
        ),
    }


# ============================================================================
# Hash generation
# ============================================================================

def compute_content_hash(job: JobDescription) -> str:
    """
    Compute a deterministic SHA-256 hash from the normalized structured job.

    The hash itself does not include:
        - job_code
        - company_code
        - job_id

    Duplicate identity is enforced by the database using:

        job_code + company_code + content_hash
    """

    canonical = _build_canonical_dict(job)

    canonical_json = json.dumps(
        canonical,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        canonical_json.encode("utf-8")
    ).hexdigest()


def compute_raw_content_hash(cleaned_text: str) -> str:
    """
    Compute a deterministic SHA-256 hash from preprocessed source text.
    """

    text = cleaned_text or ""

    text = unicodedata.normalize(
        "NFKC",
        text,
    )

    text = text.lower()

    text = _WHITESPACE_PATTERN.sub(
        " ",
        text,
    )

    text = text.strip()

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


# ============================================================================
# Duplicate types
# ============================================================================

class DuplicateType(str, Enum):
    """
    Supported duplicate classifications.
    """

    NONE = "none"
    EXACT = "exact"
    NEAR = "near"


# ============================================================================
# Duplicate detection result
# ============================================================================

class DuplicateDetectionResult(BaseModel):
    """
    Result returned by duplicate detection operations.
    """

    duplicate_type: DuplicateType

    is_duplicate: bool

    content_hash: Optional[str] = None

    raw_content_hash: Optional[str] = None

    existing_job_id: Optional[str] = None

    similarity: Optional[float] = None

    reason: Optional[str] = None


# ============================================================================
# Duplicate Detection Service
# ============================================================================

class DuplicateDetectionService:
    """
    Service responsible for exact and near duplicate detection.

    Exact duplicate:

        job_code + company_code + hash

    Near duplicate:

        Qdrant embedding similarity.

    Exact duplicate detection is the mechanism used for the new bulk
    ingestion workflow.
    """

    def __init__(
        self,
        similarity_threshold: Optional[float] = None,
    ) -> None:

        self.similarity_threshold: float = (
            similarity_threshold
            if similarity_threshold is not None
            else DEFAULT_DUPLICATE_SIMILARITY_THRESHOLD
        )

    # ========================================================================
    # Raw content duplicate
    # ========================================================================

    def check_raw_duplicate(
        self,
        raw_content_hash: str,
        job_repository: "JobRepository",
        job_code: Optional[str] = None,
        company_code: Optional[str] = None,
    ) -> DuplicateDetectionResult:
        """
        Check whether the raw source content already exists.

        Duplicate scope:

            job_code + company_code + raw_content_hash
        """

        existing = job_repository.get_job_by_raw_content_hash(
            raw_content_hash=raw_content_hash,
            job_code=job_code,
            company_code=company_code,
        )

        if existing is not None:

            logger.info(
                "Exact duplicate detected via raw_content_hash: "
                "existing_job_id=%s, "
                "raw_content_hash=%s, "
                "job_code=%s, "
                "company_code=%s",
                existing.job_id,
                raw_content_hash,
                job_code,
                company_code,
            )

            return DuplicateDetectionResult(
                duplicate_type=DuplicateType.EXACT,
                is_duplicate=True,
                content_hash=None,
                raw_content_hash=raw_content_hash,
                existing_job_id=existing.job_id,
                similarity=1.0,
                reason=(
                    "Identical raw_content_hash already stored for "
                    f"job_code={job_code}, "
                    f"company_code={company_code}"
                ),
            )

        return DuplicateDetectionResult(
            duplicate_type=DuplicateType.NONE,
            is_duplicate=False,
            content_hash=None,
            raw_content_hash=raw_content_hash,
        )

    # ========================================================================
    # Exact duplicate
    # ========================================================================

    def check_exact_duplicate(
        self,
        job: JobDescription,
        job_repository: "JobRepository",
        raw_content_hash: Optional[str] = None,
    ) -> DuplicateDetectionResult:
        """
        Check for an exact duplicate.

        Two checks are performed:

        1. raw_content_hash
        2. content_hash

        Both are scoped by:

            job_code + company_code
        """

        content_hash = compute_content_hash(job)

        job_code = (
            job.job_information.job_code
            if job.job_information
            else None
        )

        company_code = (
            job.company.company_code
            if job.company
            else None
        )

        # --------------------------------------------------------------------
        # Check raw content hash first
        # --------------------------------------------------------------------

        if raw_content_hash is not None:

            existing_by_raw = (
                job_repository.get_job_by_raw_content_hash(
                    raw_content_hash=raw_content_hash,
                    job_code=job_code,
                    company_code=company_code,
                )
            )

            if existing_by_raw is not None:

                logger.info(
                    "Exact duplicate detected via raw_content_hash: "
                    "job_id=%s matches existing job_id=%s "
                    "(job_code=%s, company_code=%s)",
                    job.job_id,
                    existing_by_raw.job_id,
                    job_code,
                    company_code,
                )

                return DuplicateDetectionResult(
                    duplicate_type=DuplicateType.EXACT,
                    is_duplicate=True,
                    content_hash=content_hash,
                    raw_content_hash=raw_content_hash,
                    existing_job_id=existing_by_raw.job_id,
                    similarity=1.0,
                    reason=(
                        "Identical raw_content_hash already stored "
                        f"for job_code={job_code}, "
                        f"company_code={company_code}"
                    ),
                )

        # --------------------------------------------------------------------
        # Check normalized structured content hash
        # --------------------------------------------------------------------

        existing = (
            job_repository.get_job_by_content_hash(
                content_hash=content_hash,
                job_code=job_code,
                company_code=company_code,
            )
        )

        if existing is not None:

            logger.info(
                "Exact duplicate detected via content_hash: "
                "job_id=%s matches existing job_id=%s "
                "(job_code=%s, company_code=%s)",
                job.job_id,
                existing.job_id,
                job_code,
                company_code,
            )

            return DuplicateDetectionResult(
                duplicate_type=DuplicateType.EXACT,
                is_duplicate=True,
                content_hash=content_hash,
                raw_content_hash=raw_content_hash,
                existing_job_id=existing.job_id,
                similarity=1.0,
                reason=(
                    "Identical normalized content_hash already stored "
                    f"for job_code={job_code}, "
                    f"company_code={company_code}"
                ),
            )

        # --------------------------------------------------------------------
        # No exact duplicate
        # --------------------------------------------------------------------

        return DuplicateDetectionResult(
            duplicate_type=DuplicateType.NONE,
            is_duplicate=False,
            content_hash=content_hash,
            raw_content_hash=raw_content_hash,
        )

    # ========================================================================
    # Near duplicate
    # ========================================================================

    def check_near_duplicate(
        self,
        job: JobDescription,
        embedding: list[float],
        qdrant_service: "QdrantService",
        content_hash: str,
    ) -> DuplicateDetectionResult:
        """
        Check for a semantically similar job using Qdrant.

        This method is retained for backward compatibility.

        It is NOT part of the exact duplicate identity.

        Exact duplicate:

            job_code + company_code + hash

        Near duplicate:

            embedding similarity >= configured threshold
        """

        candidates = qdrant_service.search_similar_jobs(
            embedding=embedding,
            limit=1,
            score_threshold=self.similarity_threshold,
        )

        candidates = [
            candidate
            for candidate in candidates
            if candidate.get("job_id") != job.job_id
        ]

        if candidates:

            best = candidates[0]

            similarity = best.get("score")

            logger.info(
                "Near duplicate detected: "
                "job_id=%s is similar to existing job_id=%s "
                "(similarity=%s, threshold=%s)",
                job.job_id,
                best.get("job_id"),
                similarity,
                self.similarity_threshold,
            )

            return DuplicateDetectionResult(
                duplicate_type=DuplicateType.NEAR,
                is_duplicate=True,
                content_hash=content_hash,
                existing_job_id=best.get("job_id"),
                similarity=similarity,
                reason=(
                    f"Embedding similarity {similarity} meets/exceeds "
                    f"DUPLICATE_SIMILARITY_THRESHOLD="
                    f"{self.similarity_threshold}"
                ),
            )

        return DuplicateDetectionResult(
            duplicate_type=DuplicateType.NONE,
            is_duplicate=False,
            content_hash=content_hash,
        )


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    "DuplicateType",
    "DuplicateDetectionResult",
    "DuplicateDetectionService",
    "compute_content_hash",
    "compute_raw_content_hash",
    "DEFAULT_DUPLICATE_SIMILARITY_THRESHOLD",
]