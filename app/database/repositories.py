# app/database/repositories.py

from __future__ import annotations

from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session, selectinload

from app.database.models import (
    Job,
    JobAdditionalInformation,
    JobComparison,
    JobReport,
    JobRequirement,
    JobResponsibility,
    User,
)
from app.schemas import (
    CompanyInfo,
    InputType,
    JobDescription,
    JobInformation,
    Requirements,
    SourceInfo,
)

__all__ = [
    "JobRepository",
    "ComparisonRepository",
    "ReportRepository",
    "UserRepository",
    "JobAlreadyExistsError",
    "JobContentHashConflictError",
    "job_row_to_job_description",
]


# ==========================================================================
# EXCEPTIONS
# ==========================================================================


class JobAlreadyExistsError(Exception):

    def __init__(self, job_id: str) -> None:
        self.job_id = job_id

        super().__init__(
            f"Job with job_id={job_id!r} already exists."
        )


class JobContentHashConflictError(Exception):

    def __init__(self, conflicting_hash: str) -> None:
        self.content_hash = conflicting_hash
        self.conflicting_hash = conflicting_hash

        super().__init__(
            f"Job with hash={conflicting_hash!r} was inserted "
            "concurrently by another request (race condition)."
        )


# ==========================================================================
# CONSTANTS
# ==========================================================================


_REQUIREMENT_CATEGORIES: tuple[str, ...] = (
    "education",
    "experience",
    "skills",
    "language",
    "computer",
    "soft_skills",
    "field_of_experience",
)


# ==========================================================================
# NORMALIZATION HELPERS
# ==========================================================================


def _normalize_job_code(
    job_code: Optional[str],
) -> Optional[str]:
    """
    Normalize job_code consistently across the application.

    Examples:
        "hr_job"    -> "HR_JOB"
        " HR_JOB "  -> "HR_JOB"
        None        -> None
    """

    if job_code is None:
        return None

    normalized = str(job_code).strip().upper()

    return normalized or None


def _normalize_company_code(
    company_code: Optional[str],
) -> Optional[str]:
    """
    Normalize company_code consistently across the application.

    Examples:
        "comp001"    -> "COMP001"
        " COMP001 "  -> "COMP001"
        None         -> None
    """

    if company_code is None:
        return None

    normalized = str(company_code).strip().upper()

    return normalized or None


# ==========================================================================
# ORM ROW -> JobDescription RECONSTRUCTION
# ==========================================================================


def job_row_to_job_description(
    db_job: Job,
) -> JobDescription:
    """
    Reconstruct a full JobDescription domain object from a persisted
    Job ORM row plus its related child rows.
    """

    grouped_requirements: dict[
        str,
        list[tuple[int, str]],
    ] = {
        category: []
        for category in _REQUIREMENT_CATEGORIES
    }

    for req in db_job.requirements:

        if req.category in grouped_requirements:

            grouped_requirements[
                req.category
            ].append(
                (
                    req.position
                    if req.position is not None
                    else 0,
                    req.value,
                )
            )

    requirements_kwargs = {
        category: [
            value
            for _position, value in sorted(
                items,
                key=lambda item: item[0],
            )
        ]
        for category, items in grouped_requirements.items()
    }

    responsibilities = [
        resp.value
        for resp in db_job.responsibilities
    ]

    additional_information = {
        info.key: info.value
        for info in db_job.additional_information
    }

    raw_input_type = db_job.input_type

    try:

        input_type = (
            InputType(raw_input_type)
            if raw_input_type
            else InputType.UNKNOWN
        )

    except ValueError:

        input_type = InputType.UNKNOWN

    return JobDescription(
        job_id=db_job.job_id,
        created_at=db_job.created_at,

        source=SourceInfo(
            language=db_job.language,
            input_type=input_type,
            original_file_name=db_job.original_file_name,
        ),

        company=CompanyInfo(
            name=None,
            industry=db_job.company_industry,
            company_code=db_job.company_code,
        ),

        job_information=JobInformation(
            job_title=db_job.job_title,
            job_code=db_job.job_code,
            department=db_job.department,
            grade=db_job.grade,
            reports_to=db_job.reports_to,
            number_of_job_holders=(
                db_job.number_of_job_holders
            ),
            number_of_direct_reports=(
                db_job.number_of_direct_reports
            ),
            creation_date=db_job.creation_date,
        ),

        job_purpose=db_job.job_purpose,

        responsibilities=responsibilities,

        requirements=Requirements(
            **requirements_kwargs
        ),

        additional_information=additional_information,

        raw_text=db_job.raw_text,
    )


# ==========================================================================
# JOB REPOSITORY
# ==========================================================================


class JobRepository:

    def __init__(
        self,
        db: Session,
    ) -> None:

        self.db = db

    # ======================================================================
    # CREATE JOB
    # ======================================================================

    def create_job(
        self,
        job: JobDescription,
        content_hash: Optional[str] = None,
        raw_content_hash: Optional[str] = None,
    ) -> Job:
        """
        Add a new Job row plus its child rows.

        Duplicate identity is scoped by:

            job_code + company_code + content_hash

        OR:

            job_code + company_code + raw_content_hash

        Race-condition handling:
        -------------------------
        Two concurrent requests can both pass the initial duplicate
        lookup and then attempt the same INSERT.

        PostgreSQL will allow only one request to win because of the
        unique constraint.

        The losing request receives IntegrityError.

        We handle that INSERT inside a SAVEPOINT (`begin_nested()`),
        so the IntegrityError rolls back only the savepoint instead
        of poisoning the outer transaction.

        After the savepoint rollback, we query PostgreSQL again and
        return the Job inserted by the concurrent request.
        """

        # ------------------------------------------------------------------
        # Job ID duplicate
        # ------------------------------------------------------------------

        if self.job_exists(job.job_id):
            raise JobAlreadyExistsError(job.job_id)

        # ------------------------------------------------------------------
        # Normalize identity fields
        # ------------------------------------------------------------------

        normalized_job_code = _normalize_job_code(
            job.job_information.job_code
        )

        normalized_company_code = _normalize_company_code(
            job.company.company_code
        )

        # ------------------------------------------------------------------
        # Build ORM row
        # ------------------------------------------------------------------

        db_job = Job(
            job_id=job.job_id,

            created_at=job.created_at,

            content_hash=content_hash,

            raw_content_hash=raw_content_hash,

            job_code=normalized_job_code,

            company_code=normalized_company_code,

            job_title=job.job_information.job_title,

            department=job.job_information.department,

            grade=job.job_information.grade,

            reports_to=job.job_information.reports_to,

            number_of_job_holders=(
                job.job_information.number_of_job_holders
            ),

            number_of_direct_reports=(
                job.job_information.number_of_direct_reports
            ),

            creation_date=(
                job.job_information.creation_date
            ),

            company_industry=job.company.industry,

            input_type=(
                job.source.input_type.value
                if hasattr(
                    job.source.input_type,
                    "value",
                )
                else job.source.input_type
            ),

            original_file_name=(
                job.source.original_file_name
            ),

            language=job.source.language,

            job_purpose=job.job_purpose,

            raw_text=job.raw_text,
        )

        # ------------------------------------------------------------------
        # Child rows
        # ------------------------------------------------------------------

        db_job.requirements = (
            self._build_requirement_rows(
                job.requirements
            )
        )

        db_job.responsibilities = (
            self._build_responsibility_rows(
                job.responsibilities
            )
        )

        db_job.additional_information = (
            self._build_additional_info_rows(
                job.additional_information
            )
        )

        # ------------------------------------------------------------------
        # INSERT using SAVEPOINT
        # ------------------------------------------------------------------

        try:

            with self.db.begin_nested():

                self.db.add(db_job)

                # Force PostgreSQL to evaluate unique constraints now.
                self.db.flush()

            return db_job

        except IntegrityError as exc:

            orig_message = str(
                getattr(exc, "orig", exc)
            ).lower()

            # --------------------------------------------------------------
            # New unique constraint names
            # --------------------------------------------------------------

            is_raw_hash_conflict = (
                raw_content_hash is not None
                and (
                    "uq_jobs_job_code_company_code_raw_content_hash"
                    in orig_message
                    or "raw_content_hash"
                    in orig_message
                )
            )

            is_content_hash_conflict = (
                content_hash is not None
                and (
                    "uq_jobs_job_code_company_code_content_hash"
                    in orig_message
                    or "content_hash"
                    in orig_message
                )
            )

            # --------------------------------------------------------------
            # RACE CONDITION RECOVERY
            # --------------------------------------------------------------

            if (
                is_raw_hash_conflict
                or is_content_hash_conflict
            ):

                # SAVEPOINT has already rolled back.
                # Outer transaction remains valid.

                existing_job: Optional[Job] = None

                # ----------------------------------------------------------
                # raw_content_hash is checked first
                # ----------------------------------------------------------

                if (
                    raw_content_hash is not None
                    and normalized_job_code is not None
                    and normalized_company_code is not None
                ):

                    existing_job = (
                        self.get_job_by_raw_content_hash(
                            raw_content_hash=raw_content_hash,
                            job_code=normalized_job_code,
                            company_code=normalized_company_code,
                        )
                    )

                # ----------------------------------------------------------
                # If raw hash didn't find the winner,
                # try normalized content hash.
                # ----------------------------------------------------------

                if (
                    existing_job is None
                    and content_hash is not None
                    and normalized_job_code is not None
                    and normalized_company_code is not None
                ):

                    existing_job = (
                        self.get_job_by_content_hash(
                            content_hash=content_hash,
                            job_code=normalized_job_code,
                            company_code=normalized_company_code,
                        )
                    )

                # ----------------------------------------------------------
                # Concurrent winner found
                # ----------------------------------------------------------

                if existing_job is not None:
                    return existing_job

                # ----------------------------------------------------------
                # PostgreSQL reported conflict but row wasn't found.
                # Don't silently hide the problem.
                # ----------------------------------------------------------

                conflicting_hash = (
                    raw_content_hash
                    if (
                        is_raw_hash_conflict
                        and raw_content_hash is not None
                    )
                    else content_hash
                )

                raise JobContentHashConflictError(
                    conflicting_hash or "unknown"
                ) from exc

            # ----------------------------------------------------------------
            # Unrelated IntegrityError
            # ----------------------------------------------------------------

            raise

    # ======================================================================
    # REQUIREMENTS
    # ======================================================================

    @staticmethod
    def _build_requirement_rows(
        requirements: Any,
    ) -> list[JobRequirement]:

        rows: list[JobRequirement] = []

        for category in _REQUIREMENT_CATEGORIES:

            values = (
                getattr(
                    requirements,
                    category,
                    None,
                )
                or []
            )

            for position, value in enumerate(values):

                rows.append(
                    JobRequirement(
                        category=category,
                        value=value,
                        position=position,
                    )
                )

        return rows

    # ======================================================================
    # RESPONSIBILITIES
    # ======================================================================

    @staticmethod
    def _build_responsibility_rows(
        responsibilities: list[str],
    ) -> list[JobResponsibility]:

        return [
            JobResponsibility(
                value=value,
                position=position,
            )
            for position, value in enumerate(
                responsibilities or []
            )
        ]

    # ======================================================================
    # ADDITIONAL INFORMATION
    # ======================================================================

    @staticmethod
    def _build_additional_info_rows(
        additional_information: dict[str, Any],
    ) -> list[JobAdditionalInformation]:

        return [
            JobAdditionalInformation(
                key=key,
                value=value,
            )
            for key, value in (
                additional_information or {}
            ).items()
        ]

    # ======================================================================
    # GET JOB BY ID
    # ======================================================================

    def get_job(
        self,
        job_id: str,
    ) -> Optional[Job]:

        stmt = (
            select(Job)
            .where(
                Job.job_id == job_id
            )
            .options(
                selectinload(Job.requirements),

                selectinload(
                    Job.responsibilities
                ),

                selectinload(
                    Job.additional_information
                ),

                selectinload(
                    Job.comparisons
                ),

                selectinload(
                    Job.reports
                ),
            )
        )

        return self.db.execute(
            stmt
        ).scalar_one_or_none()

    # ======================================================================
    # GET ALL JOBS BY JOB CODE
    # ======================================================================

    def get_jobs_by_job_code(
        self,
        job_code: str,
    ) -> list[Job]:
        """
        Return ALL jobs sharing the same normalized job_code.

        IMPORTANT:
        company_code is intentionally NOT used here.

        job_code defines the comparison group.

        company_code only identifies which specific job/company
        the user selected.

        Example:

            HR_JOB + COMP001
            HR_JOB + COMP002
            HR_JOB + COMP003

        are all historical references for HR_JOB.
        """

        normalized_job_code = _normalize_job_code(
            job_code
        )

        if normalized_job_code is None:
            return []

        stmt = (
            select(Job)
            .where(
                Job.job_code
                == normalized_job_code
            )
            .options(
                selectinload(Job.requirements),

                selectinload(
                    Job.responsibilities
                ),

                selectinload(
                    Job.additional_information
                ),
            )
        )

        return list(
            self.db.execute(
                stmt
            ).scalars().all()
        )

    # ======================================================================
    # GET JOB BY JOB CODE + COMPANY CODE
    # ======================================================================

    def get_job_by_job_code_and_company_code(
        self,
        job_code: str,
        company_code: str,
    ) -> Optional[Job]:
        """
        Return the specific job identified by:

            job_code + company_code

        This is used when the user selects which company/job
        should be compared.

        IMPORTANT:
        This method is for selecting the NEW/target job.

        It is NOT used for retrieving comparison references.
        """

        normalized_job_code = _normalize_job_code(
            job_code
        )

        normalized_company_code = _normalize_company_code(
            company_code
        )

        if (
            normalized_job_code is None
            or normalized_company_code is None
        ):
            return None

        stmt = (
            select(Job)
            .where(
                Job.job_code
                == normalized_job_code
            )
            .where(
                Job.company_code
                == normalized_company_code
            )
            .options(
                selectinload(Job.requirements),

                selectinload(
                    Job.responsibilities
                ),

                selectinload(
                    Job.additional_information
                ),

                selectinload(
                    Job.comparisons
                ),

                selectinload(
                    Job.reports
                ),
            )
        )

        return self.db.execute(
            stmt
        ).scalar_one_or_none()

    # ======================================================================
    # JOB EXISTS
    # ======================================================================

    def job_exists(
        self,
        job_id: str,
    ) -> bool:

        stmt = (
            select(Job.job_id)
            .where(
                Job.job_id == job_id
            )
        )

        return (
            self.db.execute(stmt)
            .scalar_one_or_none()
            is not None
        )

    # ======================================================================
    # GET BY CONTENT HASH
    # ======================================================================

    def get_job_by_content_hash(
        self,
        content_hash: str,
        job_code: Optional[str] = None,
        company_code: Optional[str] = None,
    ) -> Optional[Job]:
        """
        Exact duplicate lookup by content_hash.

        Duplicate identity:

            job_code
            + company_code
            + content_hash
        """

        stmt = select(Job).where(
            Job.content_hash
            == content_hash
        )

        normalized_job_code = (
            _normalize_job_code(job_code)
        )

        normalized_company_code = (
            _normalize_company_code(company_code)
        )

        if normalized_job_code is not None:

            stmt = stmt.where(
                Job.job_code
                == normalized_job_code
            )

        if normalized_company_code is not None:

            stmt = stmt.where(
                Job.company_code
                == normalized_company_code
            )

        return self.db.execute(
            stmt
        ).scalar_one_or_none()

    # ======================================================================
    # GET BY RAW CONTENT HASH
    # ======================================================================

    def get_job_by_raw_content_hash(
        self,
        raw_content_hash: str,
        job_code: Optional[str] = None,
        company_code: Optional[str] = None,
    ) -> Optional[Job]:
        """
        Exact duplicate lookup by raw_content_hash.

        Duplicate identity:

            job_code
            + company_code
            + raw_content_hash
        """

        stmt = select(Job).where(
            Job.raw_content_hash
            == raw_content_hash
        )

        normalized_job_code = (
            _normalize_job_code(job_code)
        )

        normalized_company_code = (
            _normalize_company_code(company_code)
        )

        if normalized_job_code is not None:

            stmt = stmt.where(
                Job.job_code
                == normalized_job_code
            )

        if normalized_company_code is not None:

            stmt = stmt.where(
                Job.company_code
                == normalized_company_code
            )

        return self.db.execute(
            stmt
        ).scalar_one_or_none()

    # ======================================================================
    # UPDATE JOB CODE
    # ======================================================================

    def update_job_code(
        self,
        job_id: str,
        job_code: str,
    ) -> Optional[Job]:

        normalized_job_code = (
            _normalize_job_code(job_code)
        )

        try:

            stmt = (
                select(Job)
                .where(
                    Job.job_id == job_id
                )
            )

            db_job = self.db.execute(
                stmt
            ).scalar_one_or_none()

            if db_job is None:
                return None

            db_job.job_code = (
                normalized_job_code
            )

            self.db.commit()

            self.db.refresh(db_job)

            return db_job

        except SQLAlchemyError:

            self.db.rollback()

            raise

    # ======================================================================
    # DELETE JOB
    # ======================================================================

    def delete_job(
        self,
        job_id: str,
    ) -> bool:

        try:

            stmt = (
                select(Job)
                .where(
                    Job.job_id == job_id
                )
            )

            db_job = self.db.execute(
                stmt
            ).scalar_one_or_none()

            if db_job is None:
                return False

            self.db.delete(db_job)

            self.db.commit()

            return True

        except SQLAlchemyError:

            self.db.rollback()

            raise


# ==========================================================================
# COMPARISON REPOSITORY
# ==========================================================================


class ComparisonRepository:

    def __init__(
        self,
        db: Session,
    ) -> None:

        self.db = db

    # ======================================================================
    # CREATE COMPARISON
    # ======================================================================

    def create_comparison(
        self,
        new_job_id: str,
        job_code: Optional[str],
        status: str,
        reference_jobs: list[dict],
    ) -> JobComparison:

        db_comparison = JobComparison(
            new_job_id=new_job_id,

            job_code=_normalize_job_code(
                job_code
            ),

            status=status,

            reference_jobs=reference_jobs,
        )

        self.db.add(db_comparison)

        self.db.flush()

        return db_comparison

    # ======================================================================
    # GET COMPARISON
    # ======================================================================

    def get_comparison(
        self,
        comparison_id: int,
    ) -> Optional[JobComparison]:

        stmt = (
            select(JobComparison)
            .where(
                JobComparison.id
                == comparison_id
            )
        )

        return self.db.execute(
            stmt
        ).scalar_one_or_none()

    # ======================================================================
    # GET COMPARISONS FOR JOB
    # ======================================================================

    def get_comparisons_for_job(
        self,
        new_job_id: str,
    ) -> list[JobComparison]:

        stmt = (
            select(JobComparison)
            .where(
                JobComparison.new_job_id
                == new_job_id
            )
            .order_by(
                JobComparison.created_at.desc()
            )
        )

        return list(
            self.db.execute(
                stmt
            ).scalars().all()
        )


# ==========================================================================
# REPORT REPOSITORY
# ==========================================================================


class ReportRepository:

    def __init__(
        self,
        db: Session,
    ) -> None:

        self.db = db

    # ======================================================================
    # CREATE REPORT
    # ======================================================================

    def create_report(
        self,
        job_id: str,
        job_code: Optional[str],
        comparison_id: Optional[int],
        report_type: Optional[str],
        status: Optional[str],
        confidence: Optional[float],
        is_substantive_role_change: Optional[bool],
        report_data: dict,
    ) -> JobReport:

        db_report = JobReport(
            job_id=job_id,

            job_code=_normalize_job_code(
                job_code
            ),

            comparison_id=comparison_id,

            report_type=report_type,

            status=status,

            confidence=confidence,

            is_substantive_role_change=(
                is_substantive_role_change
            ),

            report_data=report_data,
        )

        self.db.add(db_report)

        self.db.flush()

        return db_report

    # ======================================================================
    # GET REPORT
    # ======================================================================

    def get_report(
        self,
        report_id: int,
    ) -> Optional[JobReport]:

        stmt = (
            select(JobReport)
            .where(
                JobReport.id == report_id
            )
        )

        return self.db.execute(
            stmt
        ).scalar_one_or_none()

    # ======================================================================
    # GET REPORTS FOR JOB
    # ======================================================================

    def get_reports_for_job(
        self,
        job_id: str,
    ) -> list[JobReport]:

        stmt = (
            select(JobReport)
            .where(
                JobReport.job_id == job_id
            )
            .order_by(
                JobReport.created_at.desc()
            )
        )

        return list(
            self.db.execute(
                stmt
            ).scalars().all()
        )

    # ======================================================================
    # GET LATEST REPORT
    # ======================================================================

    def get_latest_report_for_job(
        self,
        job_id: str,
    ) -> Optional[JobReport]:

        stmt = (
            select(JobReport)
            .where(
                JobReport.job_id == job_id
            )
            .order_by(
                JobReport.created_at.desc(),
                JobReport.id.desc(),
            )
            .limit(1)
        )

        return self.db.execute(
            stmt
        ).scalar_one_or_none()


# ==========================================================================
# USER REPOSITORY
# ==========================================================================


class UserRepository:
    """
    Repository for user-related database operations.
    """

    def __init__(
        self,
        db: Session,
    ):
        self.db = db

    # ======================================================================
    # GET BY EMAIL
    # ======================================================================

    def get_by_email(
        self,
        email: str,
    ) -> Optional[User]:

        return (
            self.db.query(User)
            .filter(
                User.email == email
            )
            .first()
        )

    # ======================================================================
    # GET BY USERNAME
    # ======================================================================

    def get_by_username(
        self,
        username: str,
    ) -> Optional[User]:

        return (
            self.db.query(User)
            .filter(
                User.username == username
            )
            .first()
        )

    # ======================================================================
    # GET BY ID
    # ======================================================================

    def get_by_id(
        self,
        user_id: int,
    ) -> Optional[User]:

        return (
            self.db.query(User)
            .filter(
                User.id == user_id
            )
            .first()
        )

    # ======================================================================
    # CREATE USER
    # ======================================================================

    def create(
        self,
        username: str,
        email: str,
        password_hash: str,
    ) -> User:

        user = User(
            username=username,
            email=email,
            password_hash=password_hash,
            role="user",
            is_verified=False,
            is_active=True,
        )

        self.db.add(user)

        self.db.flush()

        return user