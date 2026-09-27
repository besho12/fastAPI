"""
app/llm.py

Gemini is the sole semantic comparison engine.

Python responsibilities:
- Build the Gemini request.
- Send NEW JOB + ALL REFERENCE JOBS to Gemini.
- Validate the returned structure.
- Validate reference counts.
- Validate reference evidence structure.
- Audit the returned Gemini decisions.
- Return Gemini's report unchanged.

Python does NOT:
- Calculate semantic similarity.
- Calculate majority.
- Decide M+.
- Decide M-.
- Reclassify Gemini decisions.
- Merge Gemini semantic changes.
- Create missing semantic changes.
- Create UNCHANGED decisions.
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional

from google import genai
from google.genai import types

from app.checklists import build_category_checklists
from app.config import settings
from app.exceptions import InvalidInputError, LLMError
from app.prompts import build_consolidated_comparison_messages
from app.schemas import (
    ChangeClassification,
    ComparisonStatus,
    GeminiComparisonReport,
    JobDescription,
    ReferenceEvidence,
    ReferenceEvidenceStatus,
    SemanticChange,
    SemanticRelationship,
)

logger = logging.getLogger(__name__)

# Fixed seed: same input -> (as far as Gemini allows) same output.
GEMINI_SEED = 7


# ============================================================================
# CANONICAL CATEGORIES
# ============================================================================

CANONICAL_CATEGORIES = [
    "Years of Experience",
    "Field of Experience",
    "Education Level",
    "Language",
    "Computer",
    "Soft Skills",
    "Duties & Responsibilities",
]


# ============================================================================
# GEMINI REPORT VALIDATION
# ============================================================================


def _validate_semantic_change_structure(
    change: SemanticChange,
    reference_jobs: List[JobDescription],
) -> None:
    """
    Validate structural consistency only.

    IMPORTANT:
    Gemini remains the sole semantic decision-maker.

    This function MUST NOT:
        - calculate semantic similarity
        - discover semantic concepts
        - decide M+
        - decide M-
        - calculate majority
        - merge changes
        - split changes

    It only checks whether Gemini's returned structure is internally
    consistent with the supplied schema and reference population.
    """

    reference_jobs_count = len(reference_jobs)

    # ----------------------------------------------------------------------
    # Category
    # ----------------------------------------------------------------------

    if change.category not in CANONICAL_CATEGORIES:
        raise LLMError(
            f"Gemini returned an invalid category: {change.category!r}"
        )

    # ----------------------------------------------------------------------
    # Reference counts
    # ----------------------------------------------------------------------

    if change.matched_reference_count < 0:
        raise LLMError(
            f"Gemini returned negative matched_reference_count "
            f"for category '{change.category}'."
        )

    if change.historical_support_count < 0:
        raise LLMError(
            f"Gemini returned negative historical_support_count "
            f"for category '{change.category}'."
        )

    if change.total_reference_count < 0:
        raise LLMError(
            f"Gemini returned negative total_reference_count "
            f"for category '{change.category}'."
        )

    # ----------------------------------------------------------------------
    # Counts cannot exceed total population
    # ----------------------------------------------------------------------

    if change.matched_reference_count > change.total_reference_count:
        raise LLMError(
            f"Gemini returned matched_reference_count greater than "
            f"total_reference_count for '{change.category}'."
        )

    if change.historical_support_count > change.total_reference_count:
        raise LLMError(
            f"Gemini returned historical_support_count greater than "
            f"total_reference_count for '{change.category}'."
        )

    # ----------------------------------------------------------------------
    # Total population MUST equal actual supplied references
    # ----------------------------------------------------------------------

    if change.total_reference_count != reference_jobs_count:
        raise LLMError(
            f"Gemini returned total_reference_count="
            f"{change.total_reference_count} for '{change.category}', "
            f"but {reference_jobs_count} reference jobs were provided."
        )

    if change.matched_reference_count > reference_jobs_count:
        raise LLMError(
            f"Gemini returned matched_reference_count="
            f"{change.matched_reference_count}, but only "
            f"{reference_jobs_count} reference jobs were provided."
        )

    if change.historical_support_count > reference_jobs_count:
        raise LLMError(
            f"Gemini returned historical_support_count="
            f"{change.historical_support_count}, but only "
            f"{reference_jobs_count} reference jobs were provided."
        )

    # ----------------------------------------------------------------------
    # Reference evidence
    # ----------------------------------------------------------------------

    evidence = change.reference_evidence

    if reference_jobs_count == 0:

        if evidence:
            raise LLMError(
                f"Gemini returned reference evidence for "
                f"'{change.category}' even though there are no "
                f"reference jobs."
            )

    else:

        if len(evidence) != reference_jobs_count:
            raise LLMError(
                f"Gemini returned {len(evidence)} reference evidence "
                f"records for '{change.category}', but "
                f"{reference_jobs_count} reference jobs were provided."
            )

        # ------------------------------------------------------------------
        # Reference indexes MUST be 1..N and in exact order
        # ------------------------------------------------------------------

        expected_indexes = list(
            range(1, reference_jobs_count + 1)
        )

        actual_indexes = [
            item.reference_index
            for item in evidence
        ]

        if actual_indexes != expected_indexes:
            raise LLMError(
                f"Gemini returned invalid reference indexes for "
                f"'{change.category}'. "
                f"Expected {expected_indexes}, "
                f"received {actual_indexes}."
            )

        # ------------------------------------------------------------------
        # Validate company_code / job_code identity
        # ------------------------------------------------------------------

        for index, item in enumerate(evidence, start=1):

            reference_job = reference_jobs[index - 1]

            expected_company_code = (
                reference_job.company.company_code
                if reference_job.company is not None
                else None
            )

            expected_job_code = (
                reference_job.job_information.job_code
                if reference_job.job_information is not None
                else None
            )

            if (
                item.company_code is not None
                and expected_company_code is not None
                and item.company_code != expected_company_code
            ):
                raise LLMError(
                    f"Gemini returned an incorrect company_code "
                    f"for reference #{index} in '{change.category}'. "
                    f"Expected {expected_company_code!r}, "
                    f"received {item.company_code!r}."
                )

            if (
                item.job_code is not None
                and expected_job_code is not None
                and item.job_code != expected_job_code
            ):
                raise LLMError(
                    f"Gemini returned an incorrect job_code "
                    f"for reference #{index} in '{change.category}'. "
                    f"Expected {expected_job_code!r}, "
                    f"received {item.job_code!r}."
                )

        # ------------------------------------------------------------------
        # MATCH evidence count
        # ------------------------------------------------------------------

        evidence_match_count = sum(
            1
            for item in evidence
            if item.status == ReferenceEvidenceStatus.MATCH
        )

        if evidence_match_count != change.matched_reference_count:
            raise LLMError(
                f"Gemini returned matched_reference_count="
                f"{change.matched_reference_count} for "
                f"'{change.category}', but reference evidence "
                f"contains {evidence_match_count} MATCH records."
            )

        if evidence_match_count != change.historical_support_count:
            raise LLMError(
                f"Gemini returned historical_support_count="
                f"{change.historical_support_count} for "
                f"'{change.category}', but reference evidence "
                f"contains {evidence_match_count} MATCH records."
            )

        if (
            change.matched_reference_count
            != change.historical_support_count
        ):
            raise LLMError(
                f"Gemini returned inconsistent support counts for "
                f"'{change.category}': "
                f"matched_reference_count="
                f"{change.matched_reference_count}, "
                f"historical_support_count="
                f"{change.historical_support_count}."
            )

    # ----------------------------------------------------------------------
    # M+ / M- contradiction
    # ----------------------------------------------------------------------

    if change.is_m_plus and change.is_m_minus:
        raise LLMError(
            f"Gemini marked the same semantic change as both "
            f"M+ and M- in category '{change.category}'."
        )

    # ----------------------------------------------------------------------
    # M+ structural validation
    # ----------------------------------------------------------------------

    if change.is_m_plus:

        if change.historical_requirements:
            # The historical text already lives in reference_evidence.
            logger.warning(
                "M+ change in '%s' had historical_requirements; "
                "clearing them.",
                change.category,
            )
            change.historical_requirements = []

        if not change.new_requirements:
            raise LLMError(
                f"Gemini marked '{change.category}' as M+ but "
                f"new_requirements is empty."
            )

        if not change.new_job_has_semantic_equivalent:
            raise LLMError(
                f"Gemini marked '{change.category}' as M+ but "
                f"new_job_has_semantic_equivalent is false."
            )

        # Structural consistency only.
        #
        # Gemini already made the M+ decision.
        if change.applicable_reference_count > 0 and (
            change.historical_support_count
            > change.applicable_reference_count / 2
        ):
            raise LLMError(
                f"Gemini marked '{change.category}' as M+ but "
                f"historical_support_count="
                f"{change.historical_support_count} is greater "
                f"than half of applicable_reference_count="
                f"{change.applicable_reference_count}."
            )

    # ----------------------------------------------------------------------
    # M- structural validation
    # ----------------------------------------------------------------------

    if change.is_m_minus:

        if change.new_requirements:
            raise LLMError(
                f"Gemini marked '{change.category}' as M- but "
                f"new_requirements is not empty."
            )

        if not change.historical_requirements:
            recovered = []
            for item in change.reference_evidence:
                text = (item.historical_requirement or "").strip()
                if text and text not in recovered:
                    recovered.append(text)
            if not recovered:
                raise LLMError(
                    f"Gemini marked '{change.category}' as M- but "
                    f"historical_requirements is empty."
                )
            change.historical_requirements = recovered[:1]

        if change.new_job_has_semantic_equivalent:
            raise LLMError(
                f"Gemini marked '{change.category}' as M- but "
                f"new_job_has_semantic_equivalent is true."
            )

        # Structural consistency only.
        #
        # Gemini already made the M- decision.
        if change.historical_support_count <= (
            change.applicable_reference_count / 2
        ):
            raise LLMError(
                f"Gemini marked '{change.category}' as M- but "
                f"historical_support_count="
                f"{change.historical_support_count} is not greater "
                f"than half of applicable_reference_count="
                f"{change.applicable_reference_count}."
            )


def _validate_gemini_report(
    report: GeminiComparisonReport,
    reference_jobs: List[JobDescription],
) -> None:
    """
    Validate the complete Gemini report.

    This function validates structure only.
    Gemini remains responsible for all semantic decisions.
    """

    if not isinstance(report, GeminiComparisonReport):
        raise LLMError(
            "Gemini did not return a valid GeminiComparisonReport."
        )

    reference_jobs_count = len(reference_jobs)

    # ----------------------------------------------------------------------
    # Report-level reference count
    # ----------------------------------------------------------------------

    if (
        report.reference_document_count is not None
        and report.reference_document_count != reference_jobs_count
    ):
        raise LLMError(
            f"Gemini returned reference_document_count="
            f"{report.reference_document_count}, but "
            f"{reference_jobs_count} reference jobs were provided."
        )

    # ----------------------------------------------------------------------
    # Top-level semantic changes
    # ----------------------------------------------------------------------

    for change in report.semantic_changes:

        _validate_semantic_change_structure(
            change=change,
            reference_jobs=reference_jobs,
        )

    # ----------------------------------------------------------------------
    # Section-level changes
    # ----------------------------------------------------------------------

    for section in report.sections:

        if section.category not in CANONICAL_CATEGORIES:
            raise LLMError(
                f"Gemini returned an invalid report section category: "
                f"{section.category!r}"
            )

        for change in section.changes:

            _validate_semantic_change_structure(
                change=change,
                reference_jobs=reference_jobs,
            )

    # ----------------------------------------------------------------------
    # Required report-level fields
    # ----------------------------------------------------------------------

    if report.evolution_analysis is None:
        raise LLMError(
            "Gemini did not return evolution_analysis."
        )

    if report.confidence is not None:

        if not 0.0 <= report.confidence <= 1.0:
            raise LLMError(
                "Gemini returned an invalid confidence value."
            )


# ============================================================================
# GEMINI AUDIT LOGGING
# ============================================================================


def _audit_gemini_report(
    report: GeminiComparisonReport,
) -> None:
    """
    Audit Gemini's returned semantic decisions.

    This function only logs.

    It does NOT:
        - modify Gemini decisions
        - calculate M+
        - calculate M-
        - merge changes
        - create changes
    """

    logger.info(
        "============================================================"
    )
    logger.info("GEMINI COMPARISON AUDIT")
    logger.info(
        "============================================================"
    )

    logger.info(
        "Reference document count: %s",
        report.reference_document_count,
    )

    logger.info(
        "Top-level semantic_changes count: %d",
        len(report.semantic_changes),
    )

    top_level_m_plus = 0
    top_level_m_minus = 0
    top_level_matching = 0

    for index, change in enumerate(
        report.semantic_changes,
        start=1,
    ):

        if change.is_m_plus:
            decision = "M+"
            top_level_m_plus += 1

        elif change.is_m_minus:
            decision = "M-"
            top_level_m_minus += 1

        else:
            decision = "Matching"
            top_level_matching += 1

        logger.info(
            (
                "TOP LEVEL CHANGE #%d | category=%s | decision=%s | "
                "change_type=%s | matched=%s/%s | historical_support=%s | "
                "new_equivalent=%s | new=%s | historical=%s | summary=%s"
            ),
            index,
            change.category,
            decision,
            change.change_type,
            change.matched_reference_count,
            change.total_reference_count,
            change.historical_support_count,
            change.new_job_has_semantic_equivalent,
            change.new_requirements,
            change.historical_requirements,
            change.display_summary,
        )

        for evidence in change.reference_evidence:

            logger.info(
                (
                    "REFERENCE EVIDENCE | category=%s | "
                    "reference_index=%s | company_code=%s | "
                    "job_code=%s | status=%s | "
                    "semantic_relationship=%s | "
                    "historical_requirement=%s | explanation=%s"
                ),
                change.category,
                evidence.reference_index,
                evidence.company_code,
                evidence.job_code,
                evidence.status,
                evidence.semantic_relationship,
                evidence.historical_requirement,
                evidence.explanation,
            )

    logger.info(
        "TOP LEVEL DECISION COUNTS | M+=%d | M-=%d | Matching=%d",
        top_level_m_plus,
        top_level_m_minus,
        top_level_matching,
    )

    logger.info(
        "Gemini returned %d report sections.",
        len(report.sections),
    )

    section_m_plus = 0
    section_m_minus = 0
    section_changes = 0

    for section in report.sections:

        logger.info(
            "SECTION | category=%s | changes=%d",
            section.category,
            len(section.changes),
        )

        for index, change in enumerate(
            section.changes,
            start=1,
        ):

            section_changes += 1

            if change.is_m_plus:
                decision = "M+"
                section_m_plus += 1

            elif change.is_m_minus:
                decision = "M-"
                section_m_minus += 1

            else:
                decision = "Matching"

            logger.info(
                (
                    "SECTION CHANGE #%d | category=%s | decision=%s | "
                    "change_type=%s | matched=%s/%s | "
                    "historical_support=%s | new_equivalent=%s | "
                    "new=%s | historical=%s | summary=%s"
                ),
                index,
                section.category,
                decision,
                change.change_type,
                change.matched_reference_count,
                change.total_reference_count,
                change.historical_support_count,
                change.new_job_has_semantic_equivalent,
                change.new_requirements,
                change.historical_requirements,
                change.display_summary,
            )

            for evidence in change.reference_evidence:

                logger.info(
                    (
                        "SECTION REFERENCE EVIDENCE | "
                        "category=%s | reference_index=%s | "
                        "company_code=%s | job_code=%s | "
                        "status=%s | semantic_relationship=%s"
                    ),
                    section.category,
                    evidence.reference_index,
                    evidence.company_code,
                    evidence.job_code,
                    evidence.status,
                    evidence.semantic_relationship,
                )

    logger.info(
        "SECTION DECISION COUNTS | changes=%d | M+=%d | M-=%d",
        section_changes,
        section_m_plus,
        section_m_minus,
    )

    logger.info(
        "============================================================"
    )


# ============================================================================
# LLM SERVICE
# ============================================================================


# ============================================================================
# RECONCILIATION (deterministic repair of facts Python already knows)
# ============================================================================

# Gemini sometimes labels a category with a near-synonym ("Technical
# Skills"). report_excel already accepts these, but the validation below
# used to reject them and fail the whole request with a 500.
_CATEGORY_ALIASES = {
    "years of experience": "Years of Experience",
    "experience years": "Years of Experience",
    "field of experience": "Field of Experience",
    "experience field": "Field of Experience",
    "education": "Education Level",
    "education level": "Education Level",
    "language": "Language",
    "languages": "Language",
    "computer": "Computer",
    "computer skills": "Computer",
    "technical skills": "Computer",
    "technical skill": "Computer",
    "soft skills": "Soft Skills",
    "soft skill": "Soft Skills",
    "duties": "Duties & Responsibilities",
    "responsibilities": "Duties & Responsibilities",
    "duties & responsibilities": "Duties & Responsibilities",
    "duties and responsibilities": "Duties & Responsibilities",
}

# Only these classifications go into report.semantic_changes (what the
# Excel renders). MATCHING = present in both, or a low-importance minority
# concept: nothing to show. It stays available in sections[].changes.
_SURFACED_CLASSIFICATIONS = frozenset(
    {
        ChangeClassification.M_PLUS,
        ChangeClassification.M_MINUS,
        ChangeClassification.SIGNIFICANT_MINORITY_GAP,
    }
)


def _canonical_category(category: str) -> str:
    cleaned = (category or "").strip()
    return _CATEGORY_ALIASES.get(cleaned.lower(), cleaned)


def _reconcile_change(
    change: SemanticChange,
    reference_jobs: List[JobDescription],
    checklists: dict,
) -> None:
    """
    Fix per-document evidence using facts that do not need semantics:

    - exactly one record per document, indexes 1..N (missing -> a
      conservative "no equivalent" record, never a MATCH)
    - company_code / job_code taken from the real documents
    - NOT_SPECIFIED <=> the document has NO data in this category
      (taken from the checklist, not from the model's opinion)
    """

    unspecified = set(
        (checklists.get(change.category) or {}).get(
            "unspecified_document_indexes", []
        )
    )

    by_index = {}
    for item in change.reference_evidence:
        if 1 <= item.reference_index <= len(reference_jobs):
            by_index.setdefault(item.reference_index, item)

    fixed: List[ReferenceEvidence] = []

    for index, job in enumerate(reference_jobs, start=1):

        item = by_index.get(index)

        if item is None:
            logger.warning(
                "Gemini returned no evidence for reference #%d in "
                "'%s'; treating it as not supporting.",
                index,
                change.category,
            )
            item = ReferenceEvidence(
                reference_index=index,
                status=ReferenceEvidenceStatus.NO_SEMANTIC_EQUIVALENT,
                semantic_relationship=(
                    SemanticRelationship.NO_SEMANTIC_EQUIVALENT
                ),
                explanation=(
                    "No verdict was returned for this document; "
                    "treated as not supporting."
                ),
            )

        item.company_code = (
            job.company.company_code if job.company else None
        )
        item.job_code = (
            job.job_information.job_code
            if job.job_information
            else None
        )

        if index in unspecified and item.status != ReferenceEvidenceStatus.MATCH:
            item.status = ReferenceEvidenceStatus.NOT_SPECIFIED
            item.historical_requirement = None
            item.semantic_relationship = None

        elif index in unspecified:
            # The checklist says this document has no data in the
            # category, yet the model quoted text from it (e.g. the text
            # sits in a field the checklist does not map, such as
            # requirements.skills). An evidenced MATCH is kept.
            logger.warning(
                "Reference #%d has no '%s' data in the checklist but "
                "the model returned a MATCH; keeping the MATCH.",
                index,
                change.category,
            )

        elif item.status == ReferenceEvidenceStatus.NOT_SPECIFIED:
            # The document DOES have data in this category; it just has
            # nothing equivalent to this requirement.
            item.status = ReferenceEvidenceStatus.NO_SEMANTIC_EQUIVALENT
            if item.semantic_relationship is None:
                item.semantic_relationship = (
                    SemanticRelationship.NO_SEMANTIC_EQUIVALENT
                )

        fixed.append(item)

    change.reference_evidence = fixed


def reconcile_report(
    report: GeminiComparisonReport,
    new_job: JobDescription,
    reference_jobs: List[JobDescription],
) -> GeminiComparisonReport:
    """
    Make the report internally consistent WITHOUT making any semantic
    decision.

    - repairs per-document evidence (see _reconcile_change)
    - semantic_changes is DERIVED from sections[].changes (Gemini used to
      fill both lists by hand and they disagreed run to run: 7 items in one
      run, 2 in the next). It keeps only M+, M- and Significant Minority
      Gap, because report_excel renders every item in it. Everything else
      (MATCHING) stays in sections[].changes. If a response has no sections
      at all, Gemini's own top-level list is used instead.
    """

    checklists = build_category_checklists(
        new_job=new_job,
        reference_jobs=reference_jobs,
    )

    for section in report.sections:
        section.category = _canonical_category(section.category)
        for change in section.changes:
            change.category = _canonical_category(change.category)

    for change in report.semantic_changes:
        change.category = _canonical_category(change.category)

    section_changes: List[SemanticChange] = [
        change
        for section in report.sections
        for change in section.changes
    ]

    # Same source of truth as before: sections, or (if there are none)
    # the model's own top-level list.
    candidates = section_changes or list(report.semantic_changes)

    for change in candidates:
        _reconcile_change(change, reference_jobs, checklists)

    report.semantic_changes = [
        change
        for change in candidates
        if change.classification in _SURFACED_CLASSIFICATIONS
    ]

    return report


class LLMService:

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        client_factory: Optional[Callable[[str], object]] = None,
    ) -> None:

        self.api_key = api_key or settings.GEMINI_API_KEY

        self.model = model or getattr(
            settings,
            "GEMINI_MODEL",
            "gemini-3.5-flash-lite",
        )

        logger.info(
            "Gemini model configured: %s",
            self.model,
        )

        if not self.api_key:
            raise LLMError(
                "GEMINI_API_KEY is not configured."
            )

        try:

            if client_factory is not None:
                self.client = client_factory(self.api_key)

            else:
                self.client = genai.Client(
                    api_key=self.api_key
                )

        except Exception as exc:

            raise LLMError(
                "Failed to initialize Gemini client."
            ) from exc

    def _call_gemini(
        self,
        new_job: JobDescription,
        reference_jobs: List[JobDescription],
    ) -> GeminiComparisonReport:

        # ==================================================================
        # BUILD PROMPT
        # ==================================================================

        try:

            system_instruction, user_content = (
                build_consolidated_comparison_messages(
                    new_job=new_job,
                    reference_jobs=reference_jobs,
                )
            )

        except Exception as exc:

            raise LLMError(
                "Failed to build Gemini comparison prompt."
            ) from exc

        if not system_instruction:
            raise LLMError(
                "Gemini system instruction is empty."
            )

        if not user_content:
            raise LLMError(
                "Gemini user content is empty."
            )

        # ==================================================================
        # CALL GEMINI + PARSE RESPONSE
        #
        # Gemini is the ONLY semantic comparison engine.
        #
        # Python does NOT:
        # - compare requirements
        # - calculate semantic similarity
        # - calculate semantic majority
        # - discover M+
        # - discover M-
        # - reclassify Gemini decisions
        #
        # RETRY POLICY:
        # A genuine API/network failure (timeout, auth error, service
        # outage) is NOT retried here - retrying those blindly can
        # multiply cost and latency without fixing anything, so that
        # stays single-shot exactly as before.
        #
        # A PARSING/VALIDATION failure (Gemini's JSON did not satisfy
        # GeminiComparisonReport - e.g. it missed a reference_evidence
        # record for one historical document) IS retried, ONCE. This
        # is a real, previously-unhandled failure mode: on any such
        # mismatch the whole comparison used to fail outright with no
        # second chance. One bounded retry catches the common case of
        # a single slip without the cost/latency risk of unlimited
        # retries.
        #
        # Automatic Function Calling is explicitly disabled because
        # this comparison request does not use tools/functions.
        # ==================================================================

        max_parse_attempts = 2
        report: Optional[GeminiComparisonReport] = None
        request_content = user_content

        for attempt in range(1, max_parse_attempts + 1):

            try:

                logger.info(
                    "Sending job comparison request to Gemini model: "
                    "%s (attempt %s/%s)",
                    self.model,
                    attempt,
                    max_parse_attempts,
                )

                response = self.client.models.generate_content(
                    model=self.model,
                    contents=request_content,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.15,
                        seed=GEMINI_SEED,
                        response_mime_type="application/json",
                        response_schema=GeminiComparisonReport,
                        automatic_function_calling=(
                            types.AutomaticFunctionCallingConfig(
                                disable=True
                            )
                        ),
                    ),
                )

            except Exception as exc:

                logger.exception(
                    "Gemini comparison request failed."
                )

                raise LLMError(
                    "Gemini comparison request failed."
                ) from exc

            # ==============================================================
            # TEMPORARY DEBUG PRINTS - RAW GEMINI RESPONSE
            # ==============================================================

            print("\n" + "=" * 100)
            print(f"GEMINI RESPONSE RECEIVED (attempt {attempt})")
            print("=" * 100)

            raw_text = getattr(
                response,
                "text",
                None,
            )

            parsed_response = getattr(
                response,
                "parsed",
                None,
            )

            print("RAW RESPONSE TEXT:")
            print(raw_text)

            print("\nPARSED RESPONSE:")
            print(parsed_response)

            print("=" * 100 + "\n")

            # ==============================================================
            # PARSE RESPONSE
            # ==============================================================

            try:

                parsed = getattr(
                    response,
                    "parsed",
                    None,
                )

                if parsed is not None:

                    if isinstance(
                        parsed,
                        GeminiComparisonReport,
                    ):
                        report = parsed

                    else:
                        report = (
                            GeminiComparisonReport.model_validate(
                                parsed
                            )
                        )

                else:

                    response_text = getattr(
                        response,
                        "text",
                        None,
                    )

                    if not response_text:
                        raise LLMError(
                            "Gemini returned an empty response."
                        )

                    report = (
                        GeminiComparisonReport.model_validate_json(
                            response_text
                        )
                    )

                # Parsed and validated successfully.
                break

            except Exception as exc:

                if attempt < max_parse_attempts:
                    logger.warning(
                        "Gemini response failed schema validation "
                        "on attempt %s/%s - retrying once. error=%s",
                        attempt,
                        max_parse_attempts,
                        exc,
                    )
                    # Tell the model what was wrong. Re-sending the exact
                    # same request just reproduces the same mistake.
                    request_content = (
                        user_content
                        + "\n\n======================================\n"
                        + "YOUR PREVIOUS RESPONSE WAS REJECTED\n"
                        + "======================================\n"
                        + str(exc)[:2000]
                        + "\n\nReturn the full corrected JSON. status=match "
                        + "requires semantic_relationship exact_equivalent, "
                        + "semantic_equivalent or same_core_requirement. "
                        + "A related-but-not-equivalent statement must use "
                        + "status=no_semantic_equivalent."
                    )
                    continue

                logger.exception(
                    "Failed to parse Gemini comparison response "
                    "after %s attempt(s).",
                    max_parse_attempts,
                )

                raise LLMError(
                    "Failed to parse Gemini comparison response "
                    f"after {max_parse_attempts} attempt(s)."
                ) from exc

        assert report is not None  # loop always breaks or raises

        report = reconcile_report(
            report=report,
            new_job=new_job,
            reference_jobs=reference_jobs,
        )

        # ==================================================================
        # DEBUG PRINTS - PARSED SEMANTIC COMPARISON
        # ==================================================================

        print("\n" + "=" * 100)
        print("GEMINI SEMANTIC COMPARISON RESULT")
        print("=" * 100)

        print(
            f"REFERENCE DOCUMENT COUNT: "
            f"{report.reference_document_count}"
        )

        print(
            f"TOTAL SEMANTIC CHANGES: "
            f"{len(report.semantic_changes)}"
        )

        for change_index, change in enumerate(
            report.semantic_changes,
            start=1,
        ):

            print("\n" + "-" * 100)
            print(f"CHANGE #{change_index}")
            print("-" * 100)

            print(
                f"CATEGORY: "
                f"{change.category}"
            )

            print(
                f"CHANGE TYPE: "
                f"{change.change_type}"
            )

            if change.is_m_plus:
                print("DECISION: M+")

            elif change.is_m_minus:
                print("DECISION: M-")

            else:
                print(
                    "DECISION: NOT M+ / NOT M-"
                )

            print(
                f"MATCHED REFERENCE COUNT: "
                f"{change.matched_reference_count}"
            )

            print(
                f"HISTORICAL SUPPORT COUNT: "
                f"{change.historical_support_count}"
            )

            print(
                f"TOTAL REFERENCE COUNT: "
                f"{change.total_reference_count}"
            )

            print(
                f"NEW JOB HAS SEMANTIC EQUIVALENT: "
                f"{change.new_job_has_semantic_equivalent}"
            )

            print(
                f"COVERAGE: "
                f"{change.matched_reference_count}/"
                f"{change.applicable_reference_count} "
                f"(of {change.total_reference_count} documents; "
                f"low_sample={change.low_sample})"
            )

            print(
                f"NEW REQUIREMENTS: "
                f"{change.new_requirements}"
            )

            print(
                f"HISTORICAL REQUIREMENTS: "
                f"{change.historical_requirements}"
            )

            print(
                "\nREFERENCE-BY-REFERENCE "
                "SEMANTIC ANALYSIS:"
            )

            for evidence in change.reference_evidence:

                print(
                    f"\n  Reference #"
                    f"{evidence.reference_index}"
                )

                print(
                    f"  Company Code: "
                    f"{evidence.company_code}"
                )

                print(
                    f"  Job Code: "
                    f"{evidence.job_code}"
                )

                print(
                    f"  Status: "
                    f"{evidence.status}"
                )

                print(
                    f"  Semantic Relationship: "
                    f"{evidence.semantic_relationship}"
                )

                print(
                    f"  Historical Requirement: "
                    f"{evidence.historical_requirement}"
                )

                print(
                    f"  Explanation: "
                    f"{evidence.explanation}"
                )

        print("\n" + "=" * 100)
        print(
            "END GEMINI SEMANTIC COMPARISON RESULT"
        )
        print("=" * 100 + "\n")

        return report

    def compare_and_generate_report(
        self,
        new_job: JobDescription,
        reference_jobs: Optional[
            List[JobDescription]
        ] = None,
    ) -> GeminiComparisonReport:

        # ==================================================================
        # INPUT VALIDATION
        # ==================================================================

        if not isinstance(
            new_job,
            JobDescription,
        ):
            raise InvalidInputError(
                "new_job must be a JobDescription."
            )

        if reference_jobs is None:
            reference_jobs = []

        if not isinstance(
            reference_jobs,
            list,
        ):
            raise InvalidInputError(
                "reference_jobs must be a list."
            )

        for job in reference_jobs:

            if not isinstance(
                job,
                JobDescription,
            ):
                raise InvalidInputError(
                    "Every reference job must be a JobDescription."
                )

        reference_jobs_count = len(
            reference_jobs
        )

        # ==================================================================
        # NO HISTORICAL JOBS
        # ==================================================================

        if reference_jobs_count == 0:

            return GeminiComparisonReport(
                status=ComparisonStatus.NEW_JOB_CODE,
                reference_document_count=0,
                sections=[],
                semantic_changes=[],
                overall_summary=(
                    "No historical reference jobs were available "
                    "for comparison."
                ),
                overall_assessment=(
                    "The job is treated as a new job because no "
                    "historical reference jobs were provided."
                ),
                is_substantive_role_change=False,
                confidence=1.0,
            )

        # ==================================================================
        # CONSOLIDATED GEMINI COMPARISON
        # ==================================================================

        report = self._call_gemini(
            new_job=new_job,
            reference_jobs=reference_jobs,
        )

        # ==================================================================
        # AUDIT
        # ==================================================================

        _audit_gemini_report(
            report
        )

        # ==================================================================
        # STRUCTURAL VALIDATION
        # ==================================================================

        _validate_gemini_report(
            report=report,
            reference_jobs=reference_jobs,
        )

        # ==================================================================
        # FINAL METADATA
        # ==================================================================

        report.reference_document_count = (
            reference_jobs_count
        )

        report.status = ComparisonStatus.COMPARED

        return report


__all__ = [
    "LLMService",
    "CANONICAL_CATEGORIES",
]