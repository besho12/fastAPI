from __future__ import annotations

import logging
import re
import uuid
import zipfile

from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from io import BytesIO
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Tuple,
    TypeVar,
)

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from app.database.repositories import (
    ComparisonRepository,
    JobAlreadyExistsError,
    JobContentHashConflictError,
    JobRepository,
    ReportRepository,
    job_row_to_job_description,
)
from app.database.unit_of_work import PostgresUnitOfWork
from app.duplicate_detection import (
    DuplicateDetectionService,
    DuplicateType,
    compute_content_hash,
    compute_raw_content_hash,
)
from app.exceptions import (
    JobAIException,
    PipelineError,
)
from app.extraction import ExtractionService
from app.comparison_engine import (
    ComparisonEngine,
    GeminiGateway,
    GroupComparisonEngine,
    JobGroupComparison,
)
from app.comparison_engine import generate_comparison_excel as _generate_comparison_excel_v2
from app.comparison_engine import generate_group_summary_excel
from app.comparison_engine.models import ComparisonResult as _ComparisonEngineResult
from app.config import settings
from app.discrepancy_excel import generate_discrepancy_excel
from app.discrepancy_report import (
    JobDescriptionDiscrepancyReport,
    analyze_discrepancies,
)
from app.openai_gateway import OpenAIGateway
from app.preprocessing import (
    PreprocessResult,
    Preprocessor,
)
from app.schemas import (
    ComparisonStatus,
    GeminiComparisonReport,
    InputType,
    JobDescription,
)


logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

T = TypeVar("T")


# ==========================================================================
# RESULT MODELS
# ==========================================================================


class PipelineResult(BaseModel):
    """
    Result of the comparison pipeline.

    Used when a selected Job is compared against
    historical Jobs with the same job_code.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )

    new_job: JobDescription
    job_code: str
    reference_jobs_count: int

    report: _ComparisonEngineResult

    comparison: Optional[
        _ComparisonEngineResult
    ] = None

    # Every job in the job-code group, each compared with all the others.
    group_reports: List[_ComparisonEngineResult] = Field(default_factory=list)

    # Kept for compatibility.
    # Excel files are NOT written to disk.
    excel_path: Optional[str] = None
    matrix_excel_path: Optional[str] = None

    # ZIP is kept entirely in memory.
    zip_bytes: Optional[bytes] = None
    zip_filename: Optional[str] = None

    comparison_id: Optional[int] = None
    report_id: Optional[int] = None

    duplicate_type: DuplicateType = (
        DuplicateType.NONE
    )

    duplicate_of_job_id: Optional[str] = None

    duplicate_similarity: Optional[float] = None

    content_hash: Optional[str] = None
    raw_content_hash: Optional[str] = None


class JobIngestionResult(BaseModel):
    """
    Result of storing one Job Description.

    This is intentionally separate from PipelineResult
    because bulk ingestion DOES NOT perform comparison,
    Excel generation, or ZIP generation.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )

    file_name: Optional[str] = None

    job_id: Optional[str] = None

    job_code: Optional[str] = None

    company_code: Optional[str] = None

    job_title: Optional[str] = None

    status: str

    duplicate_type: DuplicateType = (
        DuplicateType.NONE
    )

    duplicate_of_job_id: Optional[str] = None

    content_hash: Optional[str] = None

    raw_content_hash: Optional[str] = None

    error: Optional[str] = None


class _CanonicalJobResolution(
    NamedTuple
):
    job: JobDescription
    content_hash: str
    is_existing_job: bool
    existing_job_id: Optional[str]
    job_code: str
    company_code: str


# ==========================================================================
# PIPELINE
# ==========================================================================


class JobComparisonPipeline:
    """
    Main orchestration service.

    New architecture:

    1. ingest_job()
       raw_text
       -> preprocessing
       -> extraction
       -> job_code + company_code
       -> duplicate check
       -> PostgreSQL

       NO comparison.
       NO Excel.
       NO ZIP.

    2. compare_selected_job()
       job_code + company_code (backward-compatible anchor)
       -> all Jobs with same job_code
       -> one shared semantic analysis
       -> each Job compared against all remaining Jobs
       -> PostgreSQL comparison/report for every Job
       -> group overview + one Excel workbook per Job
       -> ZIP

    Legacy run() is kept for backward compatibility.
    """

    def __init__(
        self,
        extraction_service: ExtractionService,
        llm_service: Optional[Any] = None,
        preprocessor: Optional[
            Preprocessor
        ] = None,
        db_session: Optional[
            Session
        ] = None,
        job_repository: Optional[
            JobRepository
        ] = None,
        comparison_repository: Optional[
            ComparisonRepository
        ] = None,
        report_repository: Optional[
            ReportRepository
        ] = None,
        duplicate_detection_service: Optional[
            DuplicateDetectionService
        ] = None,
    ) -> None:

        self._extraction = (
            extraction_service
        )

        self._llm = llm_service

        # --------------------------------------------------------------
        # NEW COMPARISON ENGINE (app/comparison_engine/)
        #
        # Deterministic counting + narrowly-scoped LLM calls, replacing
        # app.llm.LLMService for the comparison step only. `self._llm`
        # above is kept untouched for backward compatibility; nothing
        # else in this class still calls it.
        #
        # Named `comparison_engine`, not `comparison`, because
        # app/comparison.py already exists in this project as a
        # separate (currently unused) module -- see that file's own
        # maintainer note. Keeping the names distinct avoids a
        # module-vs-package collision on `app.comparison`.
        #
        # If Gemini is not configured (missing key, network unavailable),
        # the engine still runs in fully deterministic offline mode
        # instead of raising -- see app/comparison_engine/gemini.py.
        # --------------------------------------------------------------

        try:
            if settings.LLM_PROVIDER == "gemini":
                gateway = GeminiGateway()
            else:
                gateway = OpenAIGateway()
            self._model_gateway = gateway
            self._comparison_engine = ComparisonEngine(gateway=gateway)
            self._group_comparison_engine = GroupComparisonEngine(
                gateway=gateway,
                use_model_materiality=not (
                    settings.LLM_PROVIDER == "openai"
                    and settings.OPENAI_FAST_COMPARISON
                ),
            )
        except Exception:
            logger.warning(
                "%s gateway unavailable; comparison engine will run "
                "in deterministic offline mode.",
                settings.LLM_PROVIDER,
                exc_info=True,
            )
            self._model_gateway = None
            self._comparison_engine = ComparisonEngine(gateway=None)
            self._group_comparison_engine = GroupComparisonEngine(
                gateway=None
            )

        self._preprocessor = (
            preprocessor
            if preprocessor is not None
            else Preprocessor()
        )

        self._db_session = db_session

        self._job_repository = (
            job_repository
            if job_repository is not None
            else (
                JobRepository(
                    db_session
                )
                if db_session is not None
                else None
            )
        )

        self._comparison_repository = (
            comparison_repository
        )

        self._report_repository = (
            report_repository
        )

        self._duplicate_detection = (
            duplicate_detection_service
        )

    # ======================================================================
    # NEW FLOW #1
    # BULK INGESTION / STORE ONLY
    # ======================================================================

    def ingest_job(
        self,
        raw_text: str,
        input_type: InputType = InputType.TEXT,
        original_file_name: Optional[
            str
        ] = None,
    ) -> JobIngestionResult:
        """
        Process and store ONE Job Description.

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
        This method does NOT:
        - compare Jobs
        - call Gemini comparison
        - generate Excel
        - generate ZIP
        """

        try:

            if not raw_text or not raw_text.strip():

                raise PipelineError(
                    "Job Description raw_text "
                    "cannot be empty.",
                    details={
                        "stage": "bulk_ingestion_validation"
                    },
                )

            # --------------------------------------------------------------
            # PREPROCESS
            # --------------------------------------------------------------

            preprocess_result = (
                self._preprocess(
                    raw_text,
                    input_type=input_type,
                )
            )

            cleaned_text = (
                preprocess_result[
                    "cleaned_text"
                ]
            )

            language = (
                preprocess_result[
                    "language"
                ]
            )

            # --------------------------------------------------------------
            # RAW HASH
            # --------------------------------------------------------------

            raw_content_hash = (
                compute_raw_content_hash(
                    raw_text
                )
            )

            # --------------------------------------------------------------
            # EXTRACT + DUPLICATE CHECK
            # --------------------------------------------------------------

            canonical = (
                self._get_or_create_canonical_job(
                    raw_content_hash=(
                        raw_content_hash
                    ),
                    cleaned_text=cleaned_text,
                    language=language,
                    input_type=input_type,
                    original_file_name=(
                        original_file_name
                    ),
                )
            )

            new_job = canonical.job

            # --------------------------------------------------------------
            # DUPLICATE
            # --------------------------------------------------------------

            if canonical.is_existing_job:

                return JobIngestionResult(
                    file_name=(
                        original_file_name
                    ),
                    job_id=(
                        canonical.existing_job_id
                    ),
                    job_code=(
                        canonical.job_code
                    ),
                    company_code=(
                        canonical.company_code
                    ),
                    job_title=(
                        new_job
                        .job_information
                        .job_title
                    ),
                    status="duplicate",
                    duplicate_type=(
                        DuplicateType.EXACT
                    ),
                    duplicate_of_job_id=(
                        canonical.existing_job_id
                    ),
                    content_hash=(
                        canonical.content_hash
                    ),
                    raw_content_hash=(
                        raw_content_hash
                    ),
                )

            # --------------------------------------------------------------
            # SAVE NEW JOB ONLY
            # --------------------------------------------------------------

            persisted_job_id = (
                self._persist_job_only(
                    new_job=new_job,
                    content_hash=(
                        canonical.content_hash
                    ),
                    raw_content_hash=(
                        raw_content_hash
                    ),
                )
            )

            return JobIngestionResult(
                file_name=(
                    original_file_name
                ),
                job_id=persisted_job_id,
                job_code=(
                    canonical.job_code
                ),
                company_code=(
                    canonical.company_code
                ),
                job_title=(
                    new_job
                    .job_information
                    .job_title
                ),
                status="stored",
                duplicate_type=(
                    DuplicateType.NONE
                ),
                duplicate_of_job_id=None,
                content_hash=(
                    canonical.content_hash
                ),
                raw_content_hash=(
                    raw_content_hash
                ),
            )

        except JobAIException:

            raise

        except Exception as exc:

            logger.exception(
                "Bulk ingestion failed "
                "(file=%s)",
                original_file_name,
            )

            raise PipelineError(
                "Bulk Job ingestion failed "
                f"unexpectedly: {exc}",
                details={
                    "stage": "bulk_ingestion",
                    "file_name": (
                        original_file_name
                    ),
                },
            ) from exc

    # ======================================================================
    # BULK INGESTION HELPER
    # ======================================================================

    def ingest_jobs(
        self,
        jobs: List[
            Tuple[
                str,
                InputType,
                Optional[str],
            ]
        ],
    ) -> List[JobIngestionResult]:
        """
        Convenience method for processing multiple Jobs.

        Each Job is processed independently.

        A failure in one Job does not automatically
        stop processing the remaining Jobs.

        Expected tuple:

            (
                raw_text,
                input_type,
                original_file_name
            )
        """

        results: List[
            JobIngestionResult
        ] = []

        for (
            raw_text,
            input_type,
            original_file_name,
        ) in jobs:

            try:

                result = self.ingest_job(
                    raw_text=raw_text,
                    input_type=input_type,
                    original_file_name=(
                        original_file_name
                    ),
                )

                results.append(result)

            except JobAIException as exc:

                details = getattr(
                    exc,
                    "details",
                    None,
                )

                error_message = str(exc)

                if isinstance(
                    details,
                    dict,
                ):

                    error_message = (
                        details.get(
                            "error",
                            error_message,
                        )
                        or error_message
                    )

                logger.warning(
                    "Bulk ingestion failed (file=%s): %s",
                    original_file_name,
                    error_message,
                )

                results.append(
                    JobIngestionResult(
                        file_name=(
                            original_file_name
                        ),
                        status="failed",
                        error=error_message,
                    )
                )

            except Exception as exc:

                logger.warning(
                    "Bulk ingestion failed unexpectedly (file=%s): %s",
                    original_file_name,
                    exc,
                )

                results.append(
                    JobIngestionResult(
                        file_name=(
                            original_file_name
                        ),
                        status="failed",
                        error=str(exc),
                    )
                )

        return results

    # ======================================================================
    # NEW FLOW #2
    # COMPARE SELECTED JOB
    # ======================================================================

    def compare_selected_job(
        self,
        job_code: str,
        company_code: str,
        reference_company_codes: Optional[List[str]] = None,
        include_discrepancy_report: bool = False,
    ) -> PipelineResult:
        """
        Compare the complete job-code group anchored by:

            job_code + company_code

        IMPORTANT:

        company_code identifies the first/selected Job ONLY and remains
        required for compatibility with the desktop client.

        Comparison scope is:

            every Job with the same job_code, each against all others.
        """

        # --------------------------------------------------------------
        # VALIDATION
        # --------------------------------------------------------------

        normalized_job_code = (
            self._normalize_job_code(
                job_code
            )
        )

        normalized_company_code = (
            self._normalize_company_code(
                company_code
            )
        )

        if not normalized_job_code:

            raise PipelineError(
                "job_code is required "
                "for comparison.",
                details={
                    "stage": (
                        "selected_job_validation"
                    ),
                },
            )

        if not normalized_company_code:

            raise PipelineError(
                "company_code is required "
                "for comparison.",
                details={
                    "stage": (
                        "selected_job_validation"
                    ),
                },
            )

        # --------------------------------------------------------------
        # DATABASE REPOSITORY REQUIRED
        # --------------------------------------------------------------

        if self._job_repository is None:

            raise PipelineError(
                "JobRepository is required "
                "to compare a selected Job.",
                details={
                    "stage": (
                        "selected_job_lookup"
                    ),
                    "job_code": (
                        normalized_job_code
                    ),
                    "company_code": (
                        normalized_company_code
                    ),
                },
            )

        # --------------------------------------------------------------
        # FIND EXACT SELECTED JOB
        # --------------------------------------------------------------

        selected_db_job = (
            self._run_stage(
                stage=(
                    "selected_job_lookup"
                ),
                action=lambda: (
                    self._job_repository
                    .get_job_by_job_code_and_company_code(
                        normalized_job_code,
                        normalized_company_code,
                    )
                ),
                error_message=(
                    "Selected Job lookup "
                    "failed unexpectedly"
                ),
                details={
                    "job_code": (
                        normalized_job_code
                    ),
                    "company_code": (
                        normalized_company_code
                    ),
                },
            )
        )

        if selected_db_job is None:

            raise PipelineError(
                "Selected Job was not found.",
                details={
                    "stage": (
                        "selected_job_lookup"
                    ),
                    "job_code": (
                        normalized_job_code
                    ),
                    "company_code": (
                        normalized_company_code
                    ),
                },
            )

        selected_job = (
            job_row_to_job_description(
                selected_db_job
            )
        )

        # --------------------------------------------------------------
        # GET ALL JOBS WITH SAME JOB_CODE
        # --------------------------------------------------------------

        reference_jobs = (
            self._retrieve_reference_jobs_by_job_code(
                job_code=(
                    normalized_job_code
                ),
                new_job_id=(
                    selected_db_job.job_id
                ),
                existing_job_id=None,
            )
        )

        if reference_company_codes is not None:
            requested_codes: List[str] = []
            for value in reference_company_codes:
                normalized = self._normalize_company_code(value)
                if normalized and normalized not in requested_codes:
                    requested_codes.append(normalized)

            if not requested_codes:
                raise PipelineError(
                    "reference_company_codes cannot be empty when supplied.",
                    details={"stage": "reference_company_selection"},
                )
            if normalized_company_code in requested_codes:
                raise PipelineError(
                    "The target company cannot also be a reference company.",
                    details={"stage": "reference_company_selection"},
                )

            references_by_company = {
                self._normalize_company_code(job.company.company_code): job
                for job in reference_jobs
            }
            missing_codes = [
                code for code in requested_codes if code not in references_by_company
            ]
            if missing_codes:
                raise PipelineError(
                    "One or more requested reference companies were not found "
                    f"for job code {normalized_job_code}: "
                    + ", ".join(missing_codes),
                    details={
                        "stage": "reference_company_selection",
                        "missing_company_codes": missing_codes,
                    },
                )
            reference_jobs = [
                references_by_company[code] for code in requested_codes
            ]

        # --------------------------------------------------------------
        # LOG REFERENCE JOBS
        # --------------------------------------------------------------

        self._log_reference_jobs(
            new_job=selected_job,
            reference_jobs=reference_jobs,
        )

        # --------------------------------------------------------------
        # GROUP COMPARISON
        #
        # The selected job stays first for backward compatibility, but
        # every job receives its own report against all remaining jobs.
        # Semantic clustering and materiality are performed once for the
        # complete group, so reports cannot disagree merely because the
        # model was called again in a different order.
        # --------------------------------------------------------------

        group_jobs = [selected_job, *reference_jobs]

        # The target discrepancy report depends only on the selected source
        # documents, not on the group comparison result.  Running it beside
        # semantic clustering/materiality removes one full model round trip
        # from the user-visible critical path while preserving both outputs.
        discrepancy_executor: Optional[ThreadPoolExecutor] = None
        discrepancy_future: Optional[
            Future[JobDescriptionDiscrepancyReport]
        ] = None
        if include_discrepancy_report and self._model_gateway is not None:
            discrepancy_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="discrepancy-analysis",
            )
            discrepancy_future = discrepancy_executor.submit(
                self._analyze_discrepancy_report,
                selected_job,
                reference_jobs,
            )

        try:
            # The installed client only consumes the discrepancy workbook.
            # Build the compatibility/history report deterministically so the
            # only provider round trip on this path is the requested report.
            comparison_engine = (
                GroupComparisonEngine(gateway=None)
                if include_discrepancy_report
                else self._group_comparison_engine
            )
            group_comparison = self._run_stage(
                stage="group_comparison_engine",
                action=lambda: comparison_engine.run(group_jobs),
                error_message="Group comparison engine failed",
                details={
                    "job_code": normalized_job_code,
                    "job_count": len(group_jobs),
                },
            )
            discrepancy_report = (
                discrepancy_future.result()
                if discrepancy_future is not None
                else None
            )
        finally:
            if discrepancy_executor is not None:
                discrepancy_executor.shutdown(wait=True, cancel_futures=True)
        report = group_comparison.reports[0]

        # --------------------------------------------------------------
        # HASHES
        # --------------------------------------------------------------

        content_hash = (
            selected_db_job.content_hash
        )

        raw_content_hash = (
            selected_db_job.raw_content_hash
        )

        # --------------------------------------------------------------
        # PERSIST ONE COMPARISON + REPORT FOR EVERY JOB
        # --------------------------------------------------------------

        (
            comparison_id,
            report_id,
        ) = self._persist_group_reports(
            group=group_comparison,
            selected_job_id=selected_db_job.job_id,
            job_code=normalized_job_code,
        )

        # --------------------------------------------------------------
        # EXCEL + ZIP
        # --------------------------------------------------------------

        (
            zip_bytes,
            zip_filename,
        ) = self._generate_group_comparison_package(
            job_code=(
                normalized_job_code
            ),
            group=group_comparison,
            include_discrepancy_report=include_discrepancy_report,
            discrepancy_report=discrepancy_report,
        )

        return PipelineResult(
            new_job=selected_job,
            job_code=normalized_job_code,
            reference_jobs_count=(
                len(reference_jobs)
            ),
            report=report,
            comparison=report,
            group_reports=group_comparison.reports,
            excel_path=None,
            matrix_excel_path=None,
            zip_bytes=zip_bytes,
            zip_filename=zip_filename,
            comparison_id=comparison_id,
            report_id=report_id,
            duplicate_type=(
                DuplicateType.NONE
            ),
            duplicate_of_job_id=None,
            duplicate_similarity=None,
            content_hash=(
                content_hash
            ),
            raw_content_hash=(
                raw_content_hash
            ),
        )

    # ======================================================================
    # LEGACY MAIN PIPELINE
    # ======================================================================

    def run(
        self,
        raw_text: str,
        input_type: InputType = InputType.TEXT,
        original_file_name: Optional[
            str
        ] = None,
        excel_output_path: Optional[
            str
        ] = None,
        matrix_excel_output_path: Optional[
            str
        ] = None,
    ) -> PipelineResult:
        """
        LEGACY FLOW.

        Kept for backward compatibility.

        New endpoints should prefer:

            ingest_job()

        and:

            compare_selected_job()

        This legacy method still performs:
            ingestion + comparison + Excel + ZIP
        """

        preprocess_result = (
            self._preprocess(
                raw_text,
                input_type=input_type,
            )
        )

        cleaned_text = (
            preprocess_result[
                "cleaned_text"
            ]
        )

        language = (
            preprocess_result[
                "language"
            ]
        )

        raw_content_hash = (
            compute_raw_content_hash(
                raw_text
            )
        )

        canonical = (
            self._get_or_create_canonical_job(
                raw_content_hash=(
                    raw_content_hash
                ),
                cleaned_text=cleaned_text,
                language=language,
                input_type=input_type,
                original_file_name=(
                    original_file_name
                ),
            )
        )

        new_job = canonical.job

        content_hash = (
            canonical.content_hash
        )

        is_existing_job = (
            canonical.is_existing_job
        )

        existing_job_id = (
            canonical.existing_job_id
        )

        job_code = canonical.job_code

        # --------------------------------------------------------------
        # Historical references come ONLY from PostgreSQL.
        # --------------------------------------------------------------

        reference_jobs = (
            self._retrieve_reference_jobs_by_job_code(
                job_code=job_code,
                new_job_id=new_job.job_id,
                existing_job_id=existing_job_id,
            )
        )

        self._log_reference_jobs(
            new_job=new_job,
            reference_jobs=reference_jobs,
        )

        # --------------------------------------------------------------
        # Gemini comparison
        # --------------------------------------------------------------

        report = (
            self._compare_and_generate_report(
                new_job=new_job,
                reference_jobs=reference_jobs,
            )
        )

        # --------------------------------------------------------------
        # PostgreSQL persistence
        # --------------------------------------------------------------

        (
            persisted_job_id,
            comparison_id,
            report_id,
        ) = self._persist_postgres_transaction(
            new_job=new_job,
            is_existing_job=is_existing_job,
            existing_job_id=existing_job_id,
            content_hash=content_hash,
            raw_content_hash=raw_content_hash,
            job_code=job_code,
            reference_jobs=reference_jobs,
            report=report,
        )

        # --------------------------------------------------------------
        # EXCEL + ZIP
        # --------------------------------------------------------------

        (
            zip_bytes,
            zip_filename,
        ) = self._generate_comparison_package(
            job_code=job_code,
            new_job=new_job,
            reference_jobs=reference_jobs,
            report=report,
        )

        if is_existing_job:

            duplicate_type = (
                DuplicateType.EXACT
            )

            duplicate_of_job_id = (
                existing_job_id
            )

            duplicate_similarity = 1.0

        else:

            duplicate_type = (
                DuplicateType.NONE
            )

            duplicate_of_job_id = None

            duplicate_similarity = None

        return PipelineResult(
            new_job=new_job,
            job_code=job_code,
            reference_jobs_count=(
                len(reference_jobs)
            ),
            report=report,
            comparison=report,
            excel_path=None,
            matrix_excel_path=None,
            zip_bytes=zip_bytes,
            zip_filename=zip_filename,
            comparison_id=comparison_id,
            report_id=report_id,
            duplicate_type=duplicate_type,
            duplicate_of_job_id=(
                duplicate_of_job_id
            ),
            content_hash=content_hash,
            raw_content_hash=(
                raw_content_hash
            ),
            duplicate_similarity=(
                duplicate_similarity
            ),
        )

    # ======================================================================
    # GENERIC STAGE RUNNER
    # ======================================================================

    def _run_stage(
        self,
        stage: str,
        action: Callable[[], T],
        error_message: str,
        details: Optional[
            Dict[str, Any]
        ] = None,
    ) -> T:

        try:

            return action()

        except JobAIException:

            raise

        except Exception as exc:

            raise PipelineError(
                f"{error_message}: {exc}",
                details={
                    "stage": stage,
                    **(details or {}),
                },
            ) from exc

    # ======================================================================
    # NORMALIZATION
    # ======================================================================

    @staticmethod
    def _normalize_job_code(
        job_code: Optional[str],
    ) -> str:

        if job_code is None:

            return ""

        return (
            str(job_code)
            .strip()
            .upper()
        )

    @staticmethod
    def _normalize_company_code(
        company_code: Optional[str],
    ) -> str:

        if company_code is None:

            return ""

        return (
            str(company_code)
            .strip()
            .upper()
        )

    # ======================================================================
    # PREPROCESSING
    # ======================================================================

    def _preprocess(
        self,
        raw_text: Optional[str],
        input_type: InputType,
    ) -> PreprocessResult:

        return self._run_stage(
            stage="preprocessing",
            action=lambda: (
                self._preprocessor.process(
                    raw_text,
                    input_type=input_type,
                )
            ),
            error_message=(
                "Preprocessing stage failed unexpectedly"
            ),
        )

    # ======================================================================
    # CANONICAL JOB
    # ======================================================================

    def _get_or_create_canonical_job(
        self,
        raw_content_hash: str,
        cleaned_text: str,
        language: str,
        input_type: InputType,
        original_file_name: Optional[
            str
        ],
    ) -> _CanonicalJobResolution:

        new_job = self._extract(
            cleaned_text=cleaned_text,
            language=language,
            input_type=input_type,
            original_file_name=(
                original_file_name
            ),
        )

        # --------------------------------------------------------------
        # JOB CODE
        # --------------------------------------------------------------

        job_code = (
            new_job
            .job_information
            .job_code
        )

        if (
            not job_code
            or not str(job_code).strip()
        ):

            raise PipelineError(
                "Extracted job description does not "
                "contain a valid job_code.",
                details={
                    "stage": "job_identity",
                    "file_name": (
                        original_file_name
                    ),
                },
            )

        job_code = (
            self._normalize_job_code(
                job_code
            )
        )

        # --------------------------------------------------------------
        # COMPANY CODE
        # --------------------------------------------------------------

        company_code = None

        if new_job.company is not None:

            company_code = (
                new_job
                .company
                .company_code
            )

        company_code = (
            self._normalize_company_code(
                company_code
            )
        )

        if not company_code:

            raise PipelineError(
                "Extracted job description does not "
                "contain a valid company_code.",
                details={
                    "stage": "company_identity",
                    "job_code": job_code,
                    "file_name": (
                        original_file_name
                    ),
                },
            )

        # --------------------------------------------------------------
        # Keep normalized values inside JobDescription
        # --------------------------------------------------------------

        new_job.job_information.job_code = (
            job_code
        )

        new_job.company.company_code = (
            company_code
        )

        # --------------------------------------------------------------
        # CONTENT HASH
        # --------------------------------------------------------------

        content_hash = (
            compute_content_hash(
                new_job
            )
        )

        # --------------------------------------------------------------
        # EXACT DUPLICATE
        # --------------------------------------------------------------

        existing_job_id = (
            self._check_exact_duplicate(
                job_code=job_code,
                company_code=company_code,
                content_hash=content_hash,
                raw_content_hash=(
                    raw_content_hash
                ),
            )
        )

        if existing_job_id is not None:

            logger.info(
                "Exact duplicate detected: "
                "job_code=%s, company_code=%s, "
                "existing_job_id=%s",
                job_code,
                company_code,
                existing_job_id,
            )

            return _CanonicalJobResolution(
                job=new_job,
                content_hash=content_hash,
                is_existing_job=True,
                existing_job_id=(
                    existing_job_id
                ),
                job_code=job_code,
                company_code=company_code,
            )

        return _CanonicalJobResolution(
            job=new_job,
            content_hash=content_hash,
            is_existing_job=False,
            existing_job_id=None,
            job_code=job_code,
            company_code=company_code,
        )

    # ======================================================================
    # DUPLICATE CHECK
    # ======================================================================

    def _check_exact_duplicate(
        self,
        job_code: str,
        company_code: str,
        content_hash: str,
        raw_content_hash: str,
    ) -> Optional[str]:

        if self._job_repository is None:

            return None

        # --------------------------------------------------------------
        # RAW CONTENT HASH
        # --------------------------------------------------------------

        existing = self._run_stage(
            stage=(
                "duplicate_detection_raw_hash"
            ),
            action=lambda: (
                self._job_repository
                .get_job_by_raw_content_hash(
                    raw_content_hash,
                    job_code=job_code,
                    company_code=company_code,
                )
            ),
            error_message=(
                "Duplicate detection "
                "(raw content hash) failed "
                "unexpectedly"
            ),
            details={
                "job_code": job_code,
                "company_code": company_code,
            },
        )

        if existing is not None:

            return existing.job_id

        # --------------------------------------------------------------
        # CLEANED / STRUCTURED CONTENT HASH
        # --------------------------------------------------------------

        existing = self._run_stage(
            stage=(
                "duplicate_detection_content_hash"
            ),
            action=lambda: (
                self._job_repository
                .get_job_by_content_hash(
                    content_hash,
                    job_code=job_code,
                    company_code=company_code,
                )
            ),
            error_message=(
                "Duplicate detection "
                "(content hash) failed "
                "unexpectedly"
            ),
            details={
                "job_code": job_code,
                "company_code": company_code,
            },
        )

        if existing is not None:

            return existing.job_id

        return None

    # ======================================================================
    # EXTRACTION
    # ======================================================================

    def _extract(
        self,
        cleaned_text: str,
        language: str,
        input_type: InputType,
        original_file_name: Optional[
            str
        ],
    ) -> JobDescription:

        return self._run_stage(
            stage="extraction",
            action=lambda: (
                self._extraction.extract(
                    cleaned_text=cleaned_text,
                    language=language,
                    input_type=input_type,
                    original_file_name=(
                        original_file_name
                    ),
                )
            ),
            error_message=(
                "Extraction stage failed unexpectedly"
            ),
        )

    # ======================================================================
    # REFERENCE JOBS
    # ======================================================================

    def _retrieve_reference_jobs_by_job_code(
        self,
        job_code: str,
        new_job_id: str,
        existing_job_id: Optional[
            str
        ] = None,
    ) -> List[JobDescription]:

        if self._job_repository is None:

            return []

        normalized_job_code = (
            self._normalize_job_code(
                job_code
            )
        )

        db_jobs = self._run_stage(
            stage=(
                "job_code_reference_lookup"
            ),
            action=lambda: (
                self._job_repository
                .get_jobs_by_job_code(
                    normalized_job_code
                )
            ),
            error_message=(
                "Job code reference lookup "
                "failed unexpectedly"
            ),
            details={
                "job_code": (
                    normalized_job_code
                )
            },
        )

        reference_jobs = [
            job_row_to_job_description(
                db_job
            )
            for db_job in db_jobs
            if db_job.job_id
            != new_job_id
            and db_job.job_id
            != existing_job_id
        ]

        return reference_jobs

    # ======================================================================
    # LOG REFERENCE JOBS
    # ======================================================================

    @staticmethod
    def _log_reference_jobs(
        new_job: JobDescription,
        reference_jobs: List[
            JobDescription
        ],
    ) -> None:
        # Do not print full requirements/responsibilities: those may contain
        # sensitive company data and large groups made stdout unnecessarily
        # expensive.  IDs and counts are enough for operational diagnostics.
        logger.info(
            "Comparison group prepared: selected_job_id=%s job_code=%s "
            "reference_count=%d reference_job_ids=%s",
            new_job.job_id,
            new_job.job_information.job_code,
            len(reference_jobs),
            [job.job_id for job in reference_jobs],
        )

    # ======================================================================
    # GEMINI COMPARISON
    # ======================================================================

    def _compare_and_generate_report(
        self,
        new_job: JobDescription,
        reference_jobs: List[
            JobDescription
        ],
    ) -> _ComparisonEngineResult:

        # The engine itself returns a clean NEW_JOB_CODE result when
        # reference_jobs is empty, so the early-return that used to live
        # here is no longer needed.

        return self._run_stage(
            stage=(
                "comparison_engine"
            ),
            action=lambda: (
                self._comparison_engine.run(
                    new_job=new_job,
                    reference_jobs=(
                        reference_jobs
                    ),
                )
            ),
            error_message=(
                "Comparison engine failed"
            ),
            details={
                "new_job_id": new_job.job_id,
                "reference_jobs_count": (
                    len(reference_jobs)
                ),
            },
        )

    # ======================================================================
    # PERSIST JOB ONLY
    # ======================================================================

    def _persist_job_only(
        self,
        new_job: JobDescription,
        content_hash: str,
        raw_content_hash: str,
    ) -> str:

        if self._job_repository is None:

            raise PipelineError(
                "JobRepository is required "
                "for Job ingestion.",
                details={
                    "stage": (
                        "bulk_job_persistence"
                    ),
                    "job_id": new_job.job_id,
                },
            )

        if self._db_session is None:

            raise PipelineError(
                "PostgreSQL JobRepository is "
                "configured but no db_session "
                "was provided.",
                details={
                    "stage": (
                        "bulk_job_persistence"
                    ),
                    "job_id": new_job.job_id,
                },
            )

        try:

            with PostgresUnitOfWork(
                self._db_session
            ):

                persisted_job_id = (
                    self._persist_postgres_job(
                        new_job=new_job,
                        content_hash=(
                            content_hash
                        ),
                        raw_content_hash=(
                            raw_content_hash
                        ),
                    )
                )

                if not persisted_job_id:

                    raise PipelineError(
                        "Bulk Job persistence did "
                        "not resolve a persisted "
                        "job_id.",
                        details={
                            "stage": (
                                "bulk_job_persistence"
                            ),
                            "job_id": new_job.job_id,
                        },
                    )

                logger.info(
                    "Bulk Job persisted successfully. "
                    "job_id=%s, job_code=%s, "
                    "company_code=%s",
                    persisted_job_id,
                    new_job.job_information.job_code,
                    new_job.company.company_code,
                )

                return persisted_job_id

        except JobAIException:

            raise

        except Exception as exc:

            logger.warning(
                "Bulk Job persistence "
                "transaction ROLLED BACK "
                "(job_id=%s): %s",
                new_job.job_id,
                exc,
            )

            raise PipelineError(
                "Bulk Job persistence "
                f"failed unexpectedly: {exc}",
                details={
                    "stage": (
                        "bulk_job_persistence"
                    ),
                    "job_id": new_job.job_id,
                    "error": str(exc),
                    "transaction": (
                        "rolled_back"
                    ),
                },
            ) from exc

    # ======================================================================
    # POSTGRES TRANSACTION
    # ======================================================================

    def _persist_group_reports(
        self,
        group: JobGroupComparison,
        selected_job_id: str,
        job_code: str,
    ) -> Tuple[Optional[int], Optional[int]]:
        """Atomically persist one comparison/report row for every job."""

        if (
            self._comparison_repository is None
            and self._report_repository is None
        ):
            return None, None

        if self._db_session is None:
            raise PipelineError(
                "PostgreSQL repositories are configured but no db_session "
                "was provided.",
                details={"stage": "group_report_persistence"},
            )

        selected_comparison_id: Optional[int] = None
        selected_report_id: Optional[int] = None

        try:
            with PostgresUnitOfWork(self._db_session):
                for target_job, report in zip(group.jobs, group.reports):
                    target_job_id = str(target_job.job_id)
                    references = [
                        job
                        for job in group.jobs
                        if str(job.job_id) != target_job_id
                    ]
                    comparison_id = self._persist_postgres_comparison(
                        new_job=target_job,
                        persisted_job_id=target_job_id,
                        job_code=job_code,
                        reference_jobs=references,
                        report=report,
                    )
                    report_id = self._persist_postgres_report(
                        new_job=target_job,
                        persisted_job_id=target_job_id,
                        job_code=job_code,
                        comparison_id=comparison_id,
                        report=report,
                    )
                    if target_job_id == str(selected_job_id):
                        selected_comparison_id = comparison_id
                        selected_report_id = report_id
        except JobAIException:
            raise
        except Exception as exc:
            raise PipelineError(
                f"Group report persistence failed unexpectedly: {exc}",
                details={
                    "stage": "group_report_persistence",
                    "job_code": job_code,
                    "job_count": len(group.jobs),
                    "transaction": "rolled_back",
                },
            ) from exc

        return selected_comparison_id, selected_report_id

    def _persist_postgres_transaction(
        self,
        new_job: JobDescription,
        is_existing_job: bool,
        existing_job_id: Optional[str],
        content_hash: str,
        raw_content_hash: str,
        job_code: str,
        reference_jobs: List[
            JobDescription
        ],
        report: GeminiComparisonReport,
    ) -> Tuple[
        Optional[str],
        Optional[int],
        Optional[int],
    ]:

        if (
            self._job_repository is None
            and self._comparison_repository
            is None
            and self._report_repository
            is None
        ):

            persisted_job_id = (
                existing_job_id
                if is_existing_job
                else new_job.job_id
            )

            return (
                persisted_job_id,
                None,
                None,
            )

        if self._db_session is None:

            raise PipelineError(
                "PostgreSQL repositories are "
                "configured but no db_session "
                "was provided.",
                details={
                    "stage": (
                        "postgres_persistence_transaction"
                    ),
                    "job_id": new_job.job_id,
                },
            )

        if (
            is_existing_job
            and not existing_job_id
        ):

            raise PipelineError(
                "Existing Job flow requires "
                "existing_job_id.",
                details={
                    "stage": (
                        "postgres_persistence_transaction"
                    ),
                    "job_id": new_job.job_id,
                },
            )

        persisted_job_id: Optional[
            str
        ] = None

        comparison_id: Optional[
            int
        ] = None

        report_id: Optional[
            int
        ] = None

        try:

            with PostgresUnitOfWork(
                self._db_session
            ):

                # ------------------------------------------------------
                # CASE 1:
                # Normal new Job.
                # ------------------------------------------------------

                if not is_existing_job:

                    persisted_job_id = (
                        self._persist_postgres_job(
                            new_job=new_job,
                            content_hash=(
                                content_hash
                            ),
                            raw_content_hash=(
                                raw_content_hash
                            ),
                        )
                    )

                    logger.info(
                        "Job persistence resolved "
                        "to job_id=%s",
                        persisted_job_id,
                    )

                # ------------------------------------------------------
                # CASE 2:
                # Existing Job.
                # ------------------------------------------------------

                else:

                    persisted_job_id = (
                        existing_job_id
                    )

                    logger.info(
                        "Skipping PostgreSQL Job "
                        "INSERT. Using existing "
                        "job_id=%s.",
                        persisted_job_id,
                    )

                # ------------------------------------------------------
                # Safety check
                # ------------------------------------------------------

                if not persisted_job_id:

                    raise PipelineError(
                        "PostgreSQL persistence did "
                        "not resolve a persisted job_id.",
                        details={
                            "stage": (
                                "postgres_job_persistence"
                            ),
                            "job_id": new_job.job_id,
                        },
                    )

                # ------------------------------------------------------
                # Comparison
                # ------------------------------------------------------

                comparison_id = (
                    self._persist_postgres_comparison(
                        new_job=new_job,
                        persisted_job_id=(
                            persisted_job_id
                        ),
                        job_code=job_code,
                        reference_jobs=(
                            reference_jobs
                        ),
                        report=report,
                    )
                )

                # ------------------------------------------------------
                # Report
                # ------------------------------------------------------

                report_id = (
                    self._persist_postgres_report(
                        new_job=new_job,
                        persisted_job_id=(
                            persisted_job_id
                        ),
                        job_code=job_code,
                        comparison_id=(
                            comparison_id
                        ),
                        report=report,
                    )
                )

        except JobAIException as exc:

            logger.warning(
                "PostgreSQL persistence "
                "transaction ROLLED BACK "
                "(job_id=%s): %s",
                new_job.job_id,
                exc,
            )

            details = getattr(
                exc,
                "details",
                None,
            )

            if isinstance(
                details,
                dict,
            ):

                details.setdefault(
                    "transaction",
                    "rolled_back",
                )

            raise

        except Exception as exc:

            logger.warning(
                "PostgreSQL persistence "
                "transaction ROLLED BACK "
                "(job_id=%s): %s",
                new_job.job_id,
                exc,
            )

            raise PipelineError(
                "PostgreSQL persistence "
                "transaction failed unexpectedly: "
                f"{exc}",
                details={
                    "stage": (
                        "postgres_persistence_transaction"
                    ),
                    "job_id": new_job.job_id,
                    "error": str(exc),
                    "transaction": (
                        "rolled_back"
                    ),
                },
            ) from exc

        logger.info(
            "PostgreSQL persistence "
            "transaction COMMIT "
            "(persisted_job_id=%s, "
            "comparison_id=%s, "
            "report_id=%s)",
            persisted_job_id,
            comparison_id,
            report_id,
        )

        return (
            persisted_job_id,
            comparison_id,
            report_id,
        )

    # ======================================================================
    # POSTGRES JOB
    # ======================================================================

    def _persist_postgres_job(
        self,
        new_job: JobDescription,
        content_hash: Optional[str] = None,
        raw_content_hash: Optional[str] = None,
    ) -> str:

        if self._job_repository is None:

            raise PipelineError(
                "PostgreSQL job_repository not "
                "configured; cannot persist Job.",
                details={
                    "stage": (
                        "postgres_job_persistence"
                    ),
                    "job_id": new_job.job_id,
                },
            )

        logger.info(
            "Pipeline stage: PostgreSQL job "
            "persistence "
            "(job_id=%s, job_code=%s, "
            "company_code=%s)",
            new_job.job_id,
            new_job.job_information.job_code,
            new_job.company.company_code,
        )

        try:

            db_job = (
                self._job_repository.create_job(
                    new_job,
                    content_hash=(
                        content_hash
                    ),
                    raw_content_hash=(
                        raw_content_hash
                    ),
                )
            )

            if (
                db_job.job_id
                != new_job.job_id
            ):

                logger.warning(
                    "PostgreSQL race-condition "
                    "recovery succeeded. "
                    "Using existing job_id=%s "
                    "instead of generated job_id=%s.",
                    db_job.job_id,
                    new_job.job_id,
                )

            return db_job.job_id

        except JobContentHashConflictError as exc:

            raise PipelineError(
                "PostgreSQL job persistence failed "
                "after a concurrent hash conflict "
                "because the conflicting Job could "
                "not be retrieved.",
                details={
                    "stage": (
                        "postgres_job_persistence"
                    ),
                    "job_id": new_job.job_id,
                    "job_code": (
                        new_job
                        .job_information
                        .job_code
                    ),
                    "company_code": (
                        new_job
                        .company
                        .company_code
                    ),
                    "content_hash": content_hash,
                    "raw_content_hash": (
                        raw_content_hash
                    ),
                    "conflicting_hash": (
                        exc.conflicting_hash
                    ),
                    "reason": (
                        "race_condition_recovery_failed"
                    ),
                },
            ) from exc

        except JobAlreadyExistsError as exc:

            raise PipelineError(
                "PostgreSQL job persistence failed: "
                f"job_id={new_job.job_id} already exists",
                details={
                    "stage": (
                        "postgres_job_persistence"
                    ),
                    "job_id": new_job.job_id,
                    "reason": (
                        "job_already_exists"
                    ),
                },
            ) from exc

        except JobAIException:

            raise

        except Exception as exc:

            raise PipelineError(
                "PostgreSQL job persistence "
                f"failed unexpectedly: {exc}",
                details={
                    "stage": (
                        "postgres_job_persistence"
                    ),
                    "job_id": new_job.job_id,
                    "error": str(exc),
                },
            ) from exc

    # ======================================================================
    # POSTGRES COMPARISON
    # ======================================================================

    def _persist_postgres_comparison(
        self,
        new_job: JobDescription,
        persisted_job_id: str,
        job_code: str,
        reference_jobs: List[
            JobDescription
        ],
        report: GeminiComparisonReport,
    ) -> Optional[int]:

        if (
            self._comparison_repository
            is None
        ):

            return None

        reference_jobs_payload = [
            job.job_id
            for job in reference_jobs
        ]

        status = self._status_value(
            report.status
        )

        db_comparison = (
            self._run_stage(
                stage=(
                    "postgres_comparison_persistence"
                ),
                action=lambda: (
                    self._comparison_repository
                    .create_comparison(
                        new_job_id=(
                            persisted_job_id
                        ),
                        job_code=job_code,
                        status=status,
                        reference_jobs=(
                            reference_jobs_payload
                        ),
                    )
                ),
                error_message=(
                    "PostgreSQL comparison "
                    "persistence failed "
                    "unexpectedly"
                ),
                details={
                    "job_id": new_job.job_id,
                    "persisted_job_id": (
                        persisted_job_id
                    ),
                    "job_code": job_code,
                    "reference_jobs_count": (
                        len(reference_jobs)
                    ),
                },
            )
        )

        return db_comparison.id

    # ======================================================================
    # POSTGRES REPORT
    # ======================================================================

    def _persist_postgres_report(
        self,
        new_job: JobDescription,
        persisted_job_id: str,
        job_code: str,
        comparison_id: Optional[int],
        report: GeminiComparisonReport,
    ) -> Optional[int]:

        if (
            self._report_repository
            is None
        ):

            return None

        status = self._status_value(
            report.status
        )

        report_data = (
            report.model_dump(
                mode="json"
            )
        )

        db_report = (
            self._run_stage(
                stage=(
                    "postgres_report_persistence"
                ),
                action=lambda: (
                    self._report_repository
                    .create_report(
                        job_id=(
                            persisted_job_id
                        ),
                        job_code=job_code,
                        comparison_id=(
                            comparison_id
                        ),
                        report_type=(
                            "comparison"
                        ),
                        status=status,
                        confidence=(
                            report.confidence
                        ),
                        is_substantive_role_change=(
                            report
                            .is_substantive_role_change
                        ),
                        report_data=(
                            report_data
                        ),
                    )
                ),
                error_message=(
                    "PostgreSQL report "
                    "persistence failed "
                    "unexpectedly"
                ),
                details={
                    "new_job_id": new_job.job_id,
                    "persisted_job_id": (
                        persisted_job_id
                    ),
                    "job_code": job_code,
                    "comparison_id": (
                        comparison_id
                    ),
                },
            )
        )

        return db_report.id

    # ======================================================================
    # HELPERS
    # ======================================================================

    @staticmethod
    def _status_value(
        status: Any,
    ) -> str:

        return (
            status.value
            if hasattr(
                status,
                "value",
            )
            else str(status)
        )

    # ======================================================================
    # COMPARISON PACKAGE
    # ======================================================================

    @staticmethod
    def _safe_filename_component(value: Optional[str], fallback: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
        return cleaned.strip("._-") or fallback

    def _generate_group_comparison_package(
        self,
        job_code: str,
        group: JobGroupComparison,
        include_discrepancy_report: bool = False,
        discrepancy_report: Optional[
            JobDescriptionDiscrepancyReport
        ] = None,
    ) -> Tuple[bytes, str]:
        """Generate a group overview and one full workbook per job."""

        unique_id = uuid.uuid4().hex[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_code_safe = self._safe_filename_component(job_code, "job_group")

        excel_files: List[Tuple[str, bytes]] = []

        if include_discrepancy_report:
            if self._model_gateway is None or not hasattr(
                self._model_gateway, "generate_model"
            ):
                raise PipelineError(
                    "The discrepancy report was requested, but the configured "
                    f"{settings.LLM_PROVIDER} gateway is unavailable. Check "
                    "the provider API key and model settings.",
                    details={"stage": "discrepancy_report"},
                )

            target_job = group.jobs[0]
            reference_jobs = list(group.jobs[1:])
            discrepancy = (
                discrepancy_report
                if discrepancy_report is not None
                else self._analyze_discrepancy_report(
                    target_job,
                    reference_jobs,
                )
            )
            target_code = self._safe_filename_component(
                target_job.company.company_code,
                "target_company",
            )
            excel_files.append(
                (
                    "Company_job_description_discrepancy_report.xlsx",
                    self._run_stage(
                        stage="discrepancy_excel",
                        action=lambda: generate_discrepancy_excel(discrepancy),
                        error_message="Discrepancy Excel generation failed",
                        details={"target_company_code": target_code},
                    ),
                )
            )

        else:
            excel_files.append(
                (
                    f"{job_code_safe}_group_summary.xlsx",
                    self._run_stage(
                        stage="group_summary_excel",
                        action=lambda: generate_group_summary_excel(group),
                        error_message="Group summary Excel generation failed",
                        details={"job_count": len(group.jobs)},
                    ),
                )
            )
            for index, (target_job, report) in enumerate(
                zip(group.jobs, group.reports), start=1
            ):
                references = [
                    job
                    for job in group.jobs
                    if str(job.job_id) != str(target_job.job_id)
                ]
                company_code = self._safe_filename_component(
                    target_job.company.company_code,
                    f"job_{index}",
                )
                job_id = self._safe_filename_component(
                    str(target_job.job_id), f"job_{index}"
                )
                excel_files.append(
                    (
                        f"jobs/{index:02d}_{company_code}_{job_id}_comparison.xlsx",
                        self._generate_excel(
                            report=report,
                            new_job=target_job,
                            reference_jobs=references,
                        ),
                    )
                )

        zip_bytes = self._generate_zip(excel_files=excel_files)
        zip_filename = (
            "Company_job_description_discrepancy_report.zip"
            if include_discrepancy_report
            else (
                f"{job_code_safe}_{timestamp}_{unique_id}_"
                "all_jobs_comparison.zip"
            )
        )
        return zip_bytes, zip_filename

    def _analyze_discrepancy_report(
        self,
        target_job: JobDescription,
        reference_jobs: List[JobDescription],
    ) -> JobDescriptionDiscrepancyReport:
        """Run and consistently wrap the provider-backed discrepancy call."""

        return self._run_stage(
            stage="discrepancy_analysis",
            action=lambda: analyze_discrepancies(
                target_job=target_job,
                reference_jobs=reference_jobs,
                gateway=self._model_gateway,
            ),
            error_message=(
                f"{settings.LLM_PROVIDER.capitalize()} discrepancy "
                "analysis failed"
            ),
            details={
                "target_company_code": target_job.company.company_code,
                "reference_company_codes": [
                    job.company.company_code for job in reference_jobs
                ],
            },
        )

    def _generate_comparison_package(
        self,
        job_code: str,
        new_job: JobDescription,
        reference_jobs: List[
            JobDescription
        ],
        report: _ComparisonEngineResult,
    ) -> Tuple[
        bytes,
        str,
    ]:

        unique_id = (
            uuid.uuid4().hex[:8]
        )

        timestamp = (
            datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )
        )

        job_code_safe = (
            job_code
            .replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
        )

        comparison_excel_filename = (
            f"{job_code_safe}_"
            f"{timestamp}_"
            f"{unique_id}_"
            "comparison.xlsx"
        )

        zip_filename = (
            f"{job_code_safe}_"
            f"{timestamp}_"
            f"{unique_id}_"
            "comparison.zip"
        )

        # --------------------------------------------------------------
        # Comparison Excel
        #
        # NOTE: this workbook (app/comparison_engine/excel.py) already
        # contains a "Comparison Matrix" sheet with every document
        # side by side. A second, standalone matrix.xlsx used to be
        # generated here (via app.report_matrix_excel, a leftover from
        # before comparison_engine existed) and zipped alongside it -
        # duplicating the same side-by-side view as a whole extra
        # workbook, roughly doubling the ZIP's size and handing the
        # client two files to reconcile instead of one. Removed.
        # --------------------------------------------------------------

        excel_bytes = (
            self._generate_excel(
                report=report,
                new_job=new_job,
                reference_jobs=(
                    reference_jobs
                ),
            )
        )

        # --------------------------------------------------------------
        # ZIP
        # --------------------------------------------------------------

        zip_bytes = self._generate_zip(
            excel_files=[
                (
                    comparison_excel_filename,
                    excel_bytes,
                ),
            ]
        )

        return (
            zip_bytes,
            zip_filename,
        )

    # ======================================================================
    # EXCEL - IN MEMORY
    # ======================================================================

    def _generate_excel(
        self,
        report: _ComparisonEngineResult,
        new_job: JobDescription,
        reference_jobs: List[
            JobDescription
        ],
    ) -> bytes:

        result = self._run_stage(
            stage="report_excel",
            action=lambda: (
                _generate_comparison_excel_v2(
                    result=report,
                    new_job=new_job,
                    reference_jobs=(
                        reference_jobs
                    ),
                )
            ),
            error_message=(
                "Excel generation failed "
                "unexpectedly"
            ),
            details={},
        )

        if not result:

            raise PipelineError(
                "Comparison Excel generator "
                "returned empty bytes.",
                details={
                    "stage": "report_excel"
                },
            )

        if not isinstance(
            result,
            bytes,
        ):

            raise PipelineError(
                "Comparison Excel generator "
                "must return bytes.",
                details={
                    "stage": "report_excel",
                    "actual_type": type(
                        result
                    ).__name__,
                },
            )

        return result

    # ======================================================================
    # ZIP - IN MEMORY
    # ======================================================================

    def _generate_zip(
        self,
        excel_files: List[
            Tuple[str, bytes]
        ],
    ) -> bytes:

        def _create_zip() -> bytes:

            if not excel_files:

                raise PipelineError(
                    "No Excel files were "
                    "provided to create ZIP.",
                    details={
                        "stage": "report_zip",
                    },
                )

            zip_buffer = BytesIO()

            try:

                with zipfile.ZipFile(
                    zip_buffer,
                    mode="w",
                    compression=zipfile.ZIP_DEFLATED,
                    allowZip64=True,
                ) as zip_file:

                    for (
                        filename,
                        file_bytes,
                    ) in excel_files:

                        if not filename:

                            continue

                        if not file_bytes:

                            raise PipelineError(
                                (
                                    "Excel file "
                                    "is empty: "
                                    f"{filename}"
                                ),
                                details={
                                    "stage": (
                                        "report_zip"
                                    ),
                                    "file_name": (
                                        filename
                                    ),
                                },
                            )

                        if not isinstance(
                            file_bytes,
                            bytes,
                        ):

                            raise PipelineError(
                                (
                                    "Excel file "
                                    "must be bytes: "
                                    f"{filename}"
                                ),
                                details={
                                    "stage": (
                                        "report_zip"
                                    ),
                                    "file_name": (
                                        filename
                                    ),
                                    "actual_type": (
                                        type(
                                            file_bytes
                                        ).__name__
                                    ),
                                },
                            )

                        zip_file.writestr(
                            filename,
                            file_bytes,
                        )

                zip_buffer.seek(0)

                zip_bytes = (
                    zip_buffer.getvalue()
                )

                if not zip_bytes:

                    raise PipelineError(
                        "Generated ZIP is empty.",
                        details={
                            "stage": "report_zip",
                        },
                    )

                # ------------------------------------------------------
                # VALIDATE ZIP IN MEMORY
                # ------------------------------------------------------

                with zipfile.ZipFile(
                    BytesIO(zip_bytes),
                    mode="r",
                ) as zip_file:

                    bad_file = (
                        zip_file.testzip()
                    )

                    if bad_file is not None:

                        raise PipelineError(
                            (
                                "ZIP contains a "
                                "corrupted file: "
                                f"{bad_file}"
                            ),
                            details={
                                "stage": (
                                    "report_zip_validation"
                                ),
                                "bad_file": bad_file,
                            },
                        )

                    zip_contents = (
                        zip_file.namelist()
                    )

                    if not zip_contents:

                        raise PipelineError(
                            (
                                "ZIP was created "
                                "but contains no files."
                            ),
                            details={
                                "stage": (
                                    "report_zip_validation"
                                ),
                            },
                        )

                    logger.info(
                        "ZIP created successfully "
                        "in memory. contents=%s "
                        "size=%d",
                        zip_contents,
                        len(zip_bytes),
                    )

                return zip_bytes

            finally:

                zip_buffer.close()

        return self._run_stage(
            stage="report_zip",
            action=_create_zip,
            error_message=(
                "ZIP report generation failed "
                "unexpectedly"
            ),
            details={
                "excel_file_count": len(
                    excel_files
                ),
            },
        )


__all__ = [
    "PipelineResult",
    "JobIngestionResult",
    "JobComparisonPipeline",
]
