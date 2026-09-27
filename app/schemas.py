"""
app/schemas.py

Unified schemas for JobComparisonAI.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)


# ==========================================================================
# ENUMS
# ==========================================================================


class InputType(str, Enum):
    PDF = "pdf"
    TEXT = "text"
    DOCX = "docx"
    UNKNOWN = "unknown"


# ==========================================================================
# API REQUEST SCHEMAS
# ==========================================================================


class TextRequest(BaseModel):
    jd_text: str


# ==========================================================================
# SOURCE
# ==========================================================================


class SourceInfo(BaseModel):
    input_type: InputType = InputType.UNKNOWN
    original_file_name: Optional[str] = None
    language: Optional[str] = "en"

    @field_validator("language")
    @classmethod
    def normalize_language(
        cls,
        value: Optional[str],
    ) -> Optional[str]:
        if value is None:
            return None

        return value.strip().lower()


# ==========================================================================
# COMPANY
# ==========================================================================


class CompanyInfo(BaseModel):
    name: Optional[str] = None
    industry: Optional[str] = None
    company_code: Optional[str] = None


# ==========================================================================
# JOB INFORMATION
# ==========================================================================


class JobInformation(BaseModel):
    job_title: Optional[str] = None
    job_code: Optional[str] = None
    department: Optional[str] = None
    grade: Optional[str] = None
    reports_to: Optional[str] = None
    number_of_job_holders: Optional[int] = None
    number_of_direct_reports: Optional[int] = None
    creation_date: Optional[date] = None


# ==========================================================================
# REQUIREMENTS
# ==========================================================================


class Requirements(BaseModel):
    education: List[str] = Field(default_factory=list)
    experience: List[str] = Field(default_factory=list)
    skills: List[str] = Field(default_factory=list)
    language: List[str] = Field(default_factory=list)
    computer: List[str] = Field(default_factory=list)
    soft_skills: List[str] = Field(default_factory=list)
    field_of_experience: List[str] = Field(default_factory=list)


# ==========================================================================
# ADDITIONAL INFORMATION
# ==========================================================================


class AdditionalInfoItem(BaseModel):
    key: str
    value: str


# ==========================================================================
# EXTRACTION RESULT
# ==========================================================================


class JobExtractionResult(BaseModel):
    company: CompanyInfo = Field(default_factory=CompanyInfo)

    job_information: JobInformation = Field(
        default_factory=JobInformation
    )

    job_purpose: Optional[str] = None

    responsibilities: List[str] = Field(
        default_factory=list
    )

    requirements: Requirements = Field(
        default_factory=Requirements
    )

    additional_information: List[AdditionalInfoItem] = Field(
        default_factory=list
    )


# ==========================================================================
# JOB DESCRIPTION
# ==========================================================================


class JobDescription(BaseModel):
    job_id: str = Field(
        default_factory=lambda: f"job_{uuid.uuid4().hex[:12]}"
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    source: SourceInfo = Field(
        default_factory=SourceInfo
    )

    company: CompanyInfo = Field(
        default_factory=CompanyInfo
    )

    job_information: JobInformation = Field(
        default_factory=JobInformation
    )

    job_purpose: Optional[str] = None

    responsibilities: List[str] = Field(
        default_factory=list
    )

    requirements: Requirements = Field(
        default_factory=Requirements
    )

    additional_information: Dict[str, Any] = Field(
        default_factory=dict
    )

    raw_text: str

    def is_complete(self) -> bool:
        return bool(
            self.job_information.job_title
            and self.job_purpose
            and self.responsibilities
            and (
                self.requirements.skills
                or self.requirements.experience
                or self.requirements.education
                or self.requirements.language
                or self.requirements.computer
                or self.requirements.soft_skills
                or self.requirements.field_of_experience
            )
        )

    def to_embedding_text(self) -> str:
        parts: List[str] = []

        if self.job_information.job_title:
            parts.append(
                f"Job Title: {self.job_information.job_title}"
            )

        if self.job_information.department:
            parts.append(
                f"Department: {self.job_information.department}"
            )

        if self.job_purpose:
            parts.append(
                f"Purpose: {self.job_purpose}"
            )

        if self.responsibilities:
            parts.append(
                "Responsibilities: "
                + " | ".join(self.responsibilities)
            )

        if self.requirements.education:
            parts.append(
                "Education: "
                + " | ".join(self.requirements.education)
            )

        if self.requirements.experience:
            parts.append(
                "Experience: "
                + " | ".join(self.requirements.experience)
            )

        if self.requirements.skills:
            parts.append(
                "Skills: "
                + " | ".join(self.requirements.skills)
            )

        if self.requirements.language:
            parts.append(
                "Language: "
                + " | ".join(self.requirements.language)
            )

        if self.requirements.computer:
            parts.append(
                "Computer: "
                + " | ".join(self.requirements.computer)
            )

        if self.requirements.soft_skills:
            parts.append(
                "Soft Skills: "
                + " | ".join(self.requirements.soft_skills)
            )

        if self.requirements.field_of_experience:
            parts.append(
                "Field of Experience: "
                + " | ".join(
                    self.requirements.field_of_experience
                )
            )

        return "\n".join(parts) if parts else self.raw_text

    model_config = ConfigDict(
        use_enum_values=True
    )


# ==========================================================================
# COMPARISON STATUS
# ==========================================================================


class ComparisonStatus(str, Enum):
    COMPARED = "compared"
    NEW_JOB_CODE = "new_job_code"
    DUPLICATE = "duplicate"


# ==========================================================================
# GEMINI SEMANTIC COMPARISON
# ==========================================================================


class SemanticChangeType(str, Enum):
    UNCHANGED = "unchanged"
    REFINED = "refined"
    EXPANDED = "expanded"
    REDUCED = "reduced"
    NEW = "new"
    REMOVED = "removed"


class SemanticRelationship(str, Enum):
    EXACT_EQUIVALENT = "exact_equivalent"
    SEMANTIC_EQUIVALENT = "semantic_equivalent"
    SAME_CORE_REQUIREMENT = "same_core_requirement"
    SAME_CORE_WITH_ADDITIONAL_SCOPE = (
        "same_core_with_additional_scope"
    )
    SAME_CORE_WITH_REDUCED_SCOPE = (
        "same_core_with_reduced_scope"
    )
    SAME_CORE_MORE_SPECIFIC = (
        "same_core_more_specific"
    )
    NO_SEMANTIC_EQUIVALENT = (
        "no_semantic_equivalent"
    )


class ChangeSignificance(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ==========================================================================
# REFERENCE EVIDENCE
# ==========================================================================


class ReferenceEvidenceStatus(str, Enum):
    MATCH = "match"
    NO_SEMANTIC_EQUIVALENT = "no_semantic_equivalent"
    NOT_SPECIFIED = "not_specified"


# Only these relationships count as "this document supports the candidate".
# The prompt already says PARTIAL MATCH = NO MATCH, so a related-but-not-
# complete statement (same_core_with_additional_scope, ...reduced_scope,
# ...more_specific) is recorded as evidence but is NOT counted as support.
# Tune this set, not call sites, if the business rule changes.
_SUPPORTING_RELATIONSHIPS = frozenset(
    {
        SemanticRelationship.EXACT_EQUIVALENT,
        SemanticRelationship.SEMANTIC_EQUIVALENT,
        SemanticRelationship.SAME_CORE_REQUIREMENT,
    }
)

# Denominator policy.
#   False -> a document that has NO data at all in the category (status
#            NOT_SPECIFIED) is left out of the ratio: silence is "unknown",
#            not "disagrees". (recommended)
#   True  -> old behaviour: every historical document is always in the
#            denominator, so one sparse document caps support at (N-1)/N.
COUNT_UNSPECIFIED_IN_DENOMINATOR = False


class ReferenceEvidence(BaseModel):
    """
    Semantic evidence for ONE historical/reference document.

    Gemini is responsible for determining the semantic relationship.
    Python only validates structural consistency.

    A reference document may be:

        MATCH
            Genuine semantic equivalent exists.

        NO_SEMANTIC_EQUIVALENT
            Document was reviewed and no complete semantic equivalent
            exists.

        NOT_SPECIFIED
            Gemini could not determine a reliable relationship.

    IMPORTANT:
        Exactly one ReferenceEvidence record must exist for every
        historical/reference document. This is validated at the
        SemanticChange level because only SemanticChange knows N.
    """

    reference_index: int = Field(
        ge=1
    )

    company_code: Optional[str] = None

    job_code: Optional[str] = None

    status: ReferenceEvidenceStatus

    historical_requirement: Optional[str] = None

    explanation: str = ""

    semantic_relationship: Optional[
        SemanticRelationship
    ] = None

    @model_validator(mode="after")
    def normalize_evidence(
        self,
    ) -> "ReferenceEvidence":
        """
        Make status / semantic_relationship coherent instead of failing.

        Gemini often finds a RELATED statement and reports it as
        status=match + relationship=no_semantic_equivalent (or a partial
        relationship). Rejecting the whole report for that made the API
        return 500. Now the record is repaired CONSERVATIVELY: anything
        that is not a complete, evidenced equivalent is downgraded to
        NO_SEMANTIC_EQUIVALENT, so it can never inflate support.
        """

        if self.status == ReferenceEvidenceStatus.NOT_SPECIFIED:
            self.historical_requirement = None
            self.semantic_relationship = None
            return self

        if self.status == ReferenceEvidenceStatus.MATCH:

            has_text = bool(
                (self.historical_requirement or "").strip()
            )

            if self.semantic_relationship is None and has_text:
                self.semantic_relationship = (
                    SemanticRelationship.SEMANTIC_EQUIVALENT
                )

            if (
                not has_text
                or self.semantic_relationship
                not in _SUPPORTING_RELATIONSHIPS
            ):
                self.status = (
                    ReferenceEvidenceStatus.NO_SEMANTIC_EQUIVALENT
                )

        if (
            self.status
            == ReferenceEvidenceStatus.NO_SEMANTIC_EQUIVALENT
            and self.semantic_relationship is None
        ):
            self.semantic_relationship = (
                SemanticRelationship.NO_SEMANTIC_EQUIVALENT
            )

        return self

    @property
    def counts_as_support(self) -> bool:
        return (
            self.status == ReferenceEvidenceStatus.MATCH
            and self.semantic_relationship
            in _SUPPORTING_RELATIONSHIPS
        )


# ==========================================================================
# SEMANTIC CHANGE
# ==========================================================================


class ChangeClassification(str, Enum):
    """
    The final, PYTHON-COMPUTED outcome for one semantic change.

    This is NOT a semantic decision. Every value here is derived by
    counting Gemini's own per-document verdicts (reference_evidence)
    and reading Gemini's own significance judgement. Python performs
    no semantic reasoning to produce this - it only aggregates
    decisions Gemini already made, the same way a spreadsheet SUM()
    does not "understand" the numbers it adds.

    M_PLUS
        In the NEW JOB, not held by the historical majority (<=50%).

    M_MINUS
        Held by the historical majority (>50%), absent from NEW JOB.

    SIGNIFICANT_MINORITY_GAP
        Absent from NEW JOB, held by <=50% of historical jobs, BUT
        Gemini rated it HIGH or CRITICAL significance. This is the
        category that a strict 50% cutoff would silently discard.
        It must never be dropped - only ranked below M_MINUS.

    MATCHING
        Present in both, or absent from both, or a low-significance
        minority concept not worth surfacing.

    NOT_APPLICABLE
        No historical population to compare against (N = 0).
    """

    M_PLUS = "m_plus"
    M_MINUS = "m_minus"
    SIGNIFICANT_MINORITY_GAP = "significant_minority_gap"
    MATCHING = "matching"
    NOT_APPLICABLE = "not_applicable"


# Significance levels that are strong enough to rescue a sub-50%
# historical concept from being discarded as noise. Tune this set,
# not the individual call sites, if the bar should move.
_SIGNIFICANT_MINORITY_LEVELS = frozenset(
    {
        ChangeSignificance.HIGH,
        ChangeSignificance.CRITICAL,
    }
)


class SemanticChange(BaseModel):
    """
    One semantic change discovered by Gemini.

    RESPONSIBILITY SPLIT
    ---------------------
    Gemini is responsible ONLY for judgements that require
    understanding meaning:
        - semantic matching, per historical document
          (reference_evidence[i].status)
        - whether the NEW JOB has a genuine equivalent
          (new_job_has_semantic_equivalent)
        - how important the concept is, independent of how common it
          is (significance, business_impact)

    Python is responsible ONLY for arithmetic on top of those
    judgements:
        - counting how many reference_evidence records are MATCH
        - dividing by how many there are
        - applying the 50% rule and the significant-minority rule

    Gemini therefore NEVER sets is_m_plus, is_m_minus, or any support
    count directly. Those are computed on `classification` below.
    This removes an entire class of failures where Gemini's own
    arithmetic disagreed with its own evidence and the whole response
    was rejected - the counts can no longer disagree with the
    evidence, because they are no longer a separate opinion.
    """

    category: str = Field(
        description=(
            "Canonical reporting category: Years of Experience, "
            "Field of Experience, Education Level, Language, "
            "Computer, Soft Skills, or Duties & Responsibilities."
        )
    )

    change_type: SemanticChangeType = Field(
        description="Semantic classification of this change."
    )

    semantic_relationship: Optional[
        SemanticRelationship
    ] = Field(
        default=None,
        description=(
            "Semantic relationship between the candidate "
            "requirement and its historical evidence."
        ),
    )

    historical_requirements: List[str] = Field(
        default_factory=list,
        description=(
            "Complete historical requirement evidence for a "
            "historical-side semantic concept (used when the "
            "concept is absent from the NEW JOB)."
        ),
    )

    new_requirements: List[str] = Field(
        default_factory=list,
        description=(
            "Complete NEW JOB requirement evidence for a "
            "NEW-JOB-side semantic concept (used when the concept "
            "is present in the NEW JOB)."
        ),
    )

    new_job_has_semantic_equivalent: bool = Field(
        default=False,
        description=(
            "Gemini's semantic decision indicating whether "
            "the NEW JOB contains a genuine semantic equivalent "
            "of the candidate concept."
        ),
    )

    # ------------------------------------------------------------------
    # DOCUMENT-BY-DOCUMENT EVIDENCE
    #
    # This is the ONLY place support is recorded. matched_reference_
    # count, historical_support_count and total_reference_count are
    # computed properties below, derived from this list - Gemini does
    # not report them separately.
    # ------------------------------------------------------------------

    reference_evidence: List[
        ReferenceEvidence
    ] = Field(
        default_factory=list,
        description=(
            "Exactly one evidence record per historical/reference "
            "document."
        ),
    )

    # ------------------------------------------------------------------
    # EXPLANATION / IMPORTANCE
    #
    # significance and business_impact are REQUIRED, not decorative.
    # They are what lets a sub-50% historical concept still reach the
    # final report as a Significant Minority Gap instead of being
    # silently dropped. Every candidate concept must get an honest
    # significance rating, regardless of how common it is.
    # ------------------------------------------------------------------

    explanation: str = Field(
        default=""
    )

    business_impact: Optional[str] = Field(
        default=None,
        description=(
            "Why this concept matters in practice, independent of "
            "how many historical documents contain it. Required "
            "whenever significance is HIGH or CRITICAL."
        ),
    )

    significance: ChangeSignificance = Field(
        default=ChangeSignificance.NONE,
        description=(
            "Gemini's honest judgement of how much this concept "
            "matters, assessed independently of its frequency in "
            "the historical population. A concept held by only 2 "
            "of 10 historical jobs can still be CRITICAL (e.g. a "
            "legal/compliance/safety requirement) - frequency and "
            "importance are different questions."
        ),
    )

    display_summary: Optional[str] = Field(
        default=None
    )

    # ------------------------------------------------------------------
    # COMPUTED - NOT SET BY GEMINI
    #
    # Every one of these is decorated with @computed_field, not just
    # @property. Plain @property attributes are invisible to Pydantic's
    # model_dump()/model_dump_json() - they exist in memory but do NOT
    # get written out. pipeline.py persists the report with
    # `report.model_dump(mode="json")` into JobReport.report_data
    # (JSONB), so without @computed_field, classification/is_m_plus/
    # is_m_minus/support counts would compute correctly in memory
    # during this request, then silently vanish from what is actually
    # stored - anything reading report_data back later (the history
    # endpoint, an admin export, a re-generated Excel from a past
    # report) would see none of it. @computed_field makes these part
    # of the serialized JSON, exactly like a normal field.
    # ------------------------------------------------------------------

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_reference_count(self) -> int:
        return len(self.reference_evidence)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def matched_reference_count(self) -> int:
        return sum(
            1
            for evidence in self.reference_evidence
            if evidence.counts_as_support
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def historical_support_count(self) -> int:
        return self.matched_reference_count

    @computed_field  # type: ignore[prop-decorator]
    @property
    def support_ratio(self) -> Optional[float]:
        total = self.total_reference_count
        if total == 0:
            return None
        denominator = self.applicable_reference_count
        if denominator == 0:
            # Nobody has any data in this category: nothing supports it.
            return 0.0
        return self.matched_reference_count / denominator

    @computed_field  # type: ignore[prop-decorator]
    @property
    def applicable_reference_count(self) -> int:
        """
        How many historical documents actually say something in this
        category. This is the denominator of support_ratio (see
        COUNT_UNSPECIFIED_IN_DENOMINATOR).
        """
        if COUNT_UNSPECIFIED_IN_DENOMINATOR:
            return self.total_reference_count
        return sum(
            1
            for evidence in self.reference_evidence
            if evidence.status != ReferenceEvidenceStatus.NOT_SPECIFIED
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def low_sample(self) -> bool:
        """True when the ratio rests on fewer than 2 documents."""
        return (
            self.total_reference_count > 0
            and self.applicable_reference_count < 2
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_historical_majority(self) -> Optional[bool]:
        ratio = self.support_ratio
        if ratio is None:
            return None
        return ratio > 0.5

    @computed_field  # type: ignore[prop-decorator]
    @property
    def classification(self) -> ChangeClassification:

        is_majority = self.is_historical_majority

        if is_majority is None:
            return ChangeClassification.NOT_APPLICABLE

        has_equivalent = self.new_job_has_semantic_equivalent

        # Held by the majority, missing from the NEW JOB.
        if is_majority and not has_equivalent:
            return ChangeClassification.M_MINUS

        # In the NEW JOB, not held by the majority.
        if (not is_majority) and has_equivalent:
            return ChangeClassification.M_PLUS

        # Missing from the NEW JOB, held by a minority - the case a
        # strict 50% cutoff would erase. Rescued here if, and only
        # if, Gemini itself rated it HIGH/CRITICAL importance.
        if (
            (not is_majority)
            and (not has_equivalent)
            and self.significance in _SIGNIFICANT_MINORITY_LEVELS
        ):
            return ChangeClassification.SIGNIFICANT_MINORITY_GAP

        return ChangeClassification.MATCHING

    # Backward-compatible read-only aliases so existing callers
    # (report_excel.py, report_builder.py, dashboards, etc.) that do
    # `getattr(change, "is_m_plus", False)` keep working unchanged -
    # and, same as above, so these survive persistence too.
    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_m_plus(self) -> bool:
        return self.classification == ChangeClassification.M_PLUS

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_m_minus(self) -> bool:
        return self.classification == ChangeClassification.M_MINUS

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_significant_minority_gap(self) -> bool:
        return (
            self.classification
            == ChangeClassification.SIGNIFICANT_MINORITY_GAP
        )

    # ------------------------------------------------------------------
    # STRUCTURAL VALIDATION
    #
    # What's left here checks that Gemini's OWN evidence is complete
    # and internally coherent. Nothing here can fail because of a
    # threshold - there is no threshold field left for Gemini to get
    # wrong.
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def validate_semantic_change(
        self,
    ) -> "SemanticChange":

        # ==============================================================
        # EVIDENCE COMPLETENESS
        #
        # There MUST be exactly one evidence record per historical
        # document, covering every index from 1..N with no gaps and
        # no duplicates. This is the one place we still hard-fail,
        # because a missing/duplicated document silently corrupts the
        # denominator for every category - there is no safe way to
        # proceed without it.
        # ==============================================================

        if self.reference_evidence:

            # Keep the first record for each index and sort. Gaps are
            # filled (conservatively) by reconcile_report() in llm.py,
            # which knows N. Duplicates/disorder are no reason to fail.
            unique = {}
            for evidence in self.reference_evidence:
                unique.setdefault(evidence.reference_index, evidence)

            self.reference_evidence = [
                unique[index] for index in sorted(unique)
            ]

        # ==============================================================
        # EVIDENCE QUALITY FOR M+ / M- SIDED CONTENT
        #
        # These check that the requirement TEXT Gemini wrote is on
        # the right side of the story it is telling - not that the
        # classification threshold was met, since that is no longer
        # something Gemini reports.
        # ==============================================================

        if (
            self.new_job_has_semantic_equivalent
            and not self.new_requirements
        ):
            raise ValueError(
                "new_job_has_semantic_equivalent=true requires at "
                "least one item in new_requirements."
            )

        if (
            not self.new_job_has_semantic_equivalent
            and self.new_requirements
        ):
            raise ValueError(
                "new_requirements must be empty when "
                "new_job_has_semantic_equivalent=false."
            )

        # ==============================================================
        # BUSINESS IMPACT REQUIRED FOR HIGH-STAKES SIGNIFICANCE
        #
        # If Gemini claims a concept is HIGH/CRITICAL, it must say
        # why. This is what makes a Significant Minority Gap
        # trustworthy to a reader instead of an unexplained flag.
        # ==============================================================

        if (
            self.significance in _SIGNIFICANT_MINORITY_LEVELS
            and not (self.business_impact or "").strip()
        ):
            # Repair instead of failing the whole report. Do NOT
            # downgrade significance: that would silently hide exactly the
            # rare-but-important items this field exists to rescue.
            self.business_impact = (
                (self.explanation or "").strip()
                or (self.display_summary or "").strip()
                or "Rated high significance by the model."
            )

        return self


# ==========================================================================
# EVOLUTION ANALYSIS
# ==========================================================================


class EvolutionAnalysis(BaseModel):
    experience_evolution: Optional[str] = None
    skills_evolution: Optional[str] = None
    responsibilities_evolution: Optional[str] = None
    education_evolution: Optional[str] = None
    seniority_evolution: Optional[str] = None
    technology_evolution: Optional[str] = None
    overall_evolution: Optional[str] = None
    substantive_role_change: bool = False
    explanation: Optional[str] = None


# ==========================================================================
# REPORT SECTION
# ==========================================================================


class ReportSection(BaseModel):
    category: str

    investigation: str

    result: str = ""

    changes: List[SemanticChange] = Field(
        default_factory=list
    )

    section_summary: Optional[str] = None


# ==========================================================================
# GEMINI FINAL REPORT
# ==========================================================================


class GeminiComparisonReport(BaseModel):
    status: ComparisonStatus

    reference_document_count: int = Field(
        default=0,
        ge=0,
    )

    sections: List[ReportSection] = Field(
        default_factory=list
    )

    semantic_changes: List[SemanticChange] = Field(
        default_factory=list
    )

    evolution_analysis: EvolutionAnalysis = Field(
        default_factory=EvolutionAnalysis
    )

    overall_summary: Optional[str] = None

    overall_assessment: Optional[str] = None

    is_substantive_role_change: bool = False

    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )


# ==========================================================================
# LEGACY COMPARISON SCHEMAS
# ==========================================================================


class FieldComparison(BaseModel):
    new_value: Optional[str] = None
    reference_value: Optional[str] = None
    match: bool = False
    reason: Optional[str] = None


class ListComparison(BaseModel):
    common: List[str] = Field(default_factory=list)
    new_only: List[str] = Field(default_factory=list)
    reference_only: List[str] = Field(default_factory=list)


class ExperienceComparison(ListComparison):
    new_years: Optional[float] = None
    reference_years: Optional[float] = None
    years_difference: Optional[float] = None


class JobIdentityComparison(BaseModel):
    job_title: FieldComparison
    department: FieldComparison
    grade: FieldComparison
    job_code: FieldComparison


class ComparisonResult(BaseModel):
    status: ComparisonStatus
    new_job_id: str
    new_job_code: str
    reference_job_id: str
    reference_job_code: str
    overall_match: bool
    reason: Optional[str] = None
    job_identity: JobIdentityComparison
    job_purpose: FieldComparison
    experience: ExperienceComparison
    education: ListComparison
    skills: ListComparison
    responsibilities: ListComparison
    field_comparisons: List[FieldComparison] = Field(
        default_factory=list
    )


class ComparisonSet(BaseModel):
    status: ComparisonStatus
    new_job_id: str
    new_job_code: Optional[str] = None
    reference_jobs_count: int

    # Gemini's complete semantic comparison report
    report: Optional[GeminiComparisonReport] = None

    comparisons: List[ComparisonResult] = Field(
        default_factory=list
    )


# ==========================================================================
# LEGACY SUMMARY SCHEMAS
# ==========================================================================


class AggregatedListItem(BaseModel):
    value: str
    count: int
    percentage: float


class FieldSummary(BaseModel):
    matches: int = 0
    mismatches: int = 0
    match_percentage: float = 0.0


class ListSummary(BaseModel):
    common: List[AggregatedListItem] = Field(
        default_factory=list
    )

    new_only: List[AggregatedListItem] = Field(
        default_factory=list
    )

    reference_only: List[AggregatedListItem] = Field(
        default_factory=list
    )


class ExperienceSummary(BaseModel):
    common: List[AggregatedListItem] = Field(
        default_factory=list
    )

    new_only: List[AggregatedListItem] = Field(
        default_factory=list
    )

    reference_only: List[AggregatedListItem] = Field(
        default_factory=list
    )

    new_years: Optional[float] = None
    reference_years_min: Optional[float] = None
    reference_years_max: Optional[float] = None
    reference_years_average: Optional[float] = None
    years_difference_min: Optional[float] = None
    years_difference_max: Optional[float] = None
    years_difference_average: Optional[float] = None
    comparable_references: int = 0


class JobIdentitySummary(BaseModel):
    job_title: FieldSummary
    department: FieldSummary
    grade: FieldSummary
    job_code: FieldSummary


class ComparisonSummary(BaseModel):
    status: ComparisonStatus
    new_job_id: str
    new_job_code: Optional[str] = None
    reference_jobs_count: int
    job_identity: JobIdentitySummary
    job_purpose: FieldSummary
    experience: ExperienceSummary
    education: ListSummary
    skills: ListSummary
    responsibilities: ListSummary


# ==========================================================================
# EXPORTS
# ==========================================================================


__all__ = [
    "TextRequest",
    "InputType",
    "SourceInfo",
    "CompanyInfo",
    "JobInformation",
    "Requirements",
    "AdditionalInfoItem",
    "JobExtractionResult",
    "JobDescription",
    "ComparisonStatus",
    "SemanticChangeType",
    "SemanticRelationship",
    "ChangeSignificance",
    "ReferenceEvidenceStatus",
    "ReferenceEvidence",
    "ChangeClassification",
    "SemanticChange",
    "EvolutionAnalysis",
    "ReportSection",
    "GeminiComparisonReport",
    "FieldComparison",
    "ListComparison",
    "ExperienceComparison",
    "JobIdentityComparison",
    "ComparisonResult",
    "ComparisonSet",
    "AggregatedListItem",
    "FieldSummary",
    "ListSummary",
    "ExperienceSummary",
    "JobIdentitySummary",
    "ComparisonSummary",
]