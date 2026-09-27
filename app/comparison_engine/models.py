"""
app/comparison/models.py

Schemas for the deterministic comparison engine.

DESIGN RULE
-----------
Anything that is a NUMBER lives here as a computed, Python-owned field.
Anything that is a JUDGEMENT is explicitly marked as LLM-owned.

The LLM never returns a count, a ratio, a percentage or a classification.
It returns (a) concept clusters and (b) a materiality tier. Nothing else.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ==========================================================================
# CATEGORIES
# ==========================================================================


class Category(str, Enum):
    """The seven comparable dimensions of a job description."""

    YEARS_OF_EXPERIENCE = "Years of Experience"
    FIELD_OF_EXPERIENCE = "Field of Experience"
    EDUCATION = "Education Level"
    LANGUAGE = "Language"
    COMPUTER = "Computer Skills"
    SOFT_SKILLS = "Soft Skills"
    RESPONSIBILITIES = "Duties & Responsibilities"


# Short, stable prefixes used to build human-traceable item ids.
CATEGORY_CODE: Dict[Category, str] = {
    Category.YEARS_OF_EXPERIENCE: "EXP",
    Category.FIELD_OF_EXPERIENCE: "FLD",
    Category.EDUCATION: "EDU",
    Category.LANGUAGE: "LNG",
    Category.COMPUTER: "CMP",
    Category.SOFT_SKILLS: "SFT",
    Category.RESPONSIBILITIES: "DTY",
}

CATEGORY_ORDER: List[Category] = [
    Category.YEARS_OF_EXPERIENCE,
    Category.FIELD_OF_EXPERIENCE,
    Category.EDUCATION,
    Category.LANGUAGE,
    Category.COMPUTER,
    Category.SOFT_SKILLS,
    Category.RESPONSIBILITIES,
]


# ==========================================================================
# STAGE 0 — SOURCE ITEMS
# ==========================================================================


NEW_JOB_DOC_INDEX = 0


class SourceItem(BaseModel):
    """
    One atomic requirement line, extracted verbatim from one document.

    `item_id` is the unit of traceability for the whole engine. Every
    concept the LLM produces must point back to real item_ids, and any
    id that does not exist is rejected. This is what makes hallucinated
    findings structurally impossible to reach the report.
    """

    item_id: str
    doc_index: int  # 0 = new job, 1..N = reference documents
    doc_label: str
    category: Category
    text: str


class DocumentRef(BaseModel):
    """Identity of one document participating in the comparison."""

    doc_index: int
    label: str
    job_code: Optional[str] = None
    company_code: Optional[str] = None
    job_title: Optional[str] = None
    is_new_job: bool = False


# ==========================================================================
# STAGE 1 — CONCEPTS (LLM-owned, structurally validated)
# ==========================================================================


class Concept(BaseModel):
    """
    A canonical semantic concept: the same requirement expressed in many
    different wordings, collapsed into one unit of comparison.

    This is the single most important object in the system. If clustering
    is correct, every downstream number is correct by construction.
    """

    concept_id: str
    category: Category
    label: str
    member_item_ids: List[str] = Field(default_factory=list)
    from_fallback: bool = False  # produced by the deterministic fallback


class ConceptClusteringResult(BaseModel):
    category: Category
    concepts: List[Concept] = Field(default_factory=list)
    used_fallback: bool = False
    dropped_unknown_ids: List[str] = Field(default_factory=list)
    recovered_orphan_ids: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


# ==========================================================================
# STAGE 2 — EVIDENCE (100% Python, deterministic)
# ==========================================================================


class Direction(str, Enum):
    ALIGNED = "aligned"
    M_PLUS = "m_plus"  # new job has it, the benchmark largely does not
    M_MINUS = "m_minus"  # benchmark has it, the new job does not


class BenchmarkStrength(str, Enum):
    STRONG = "strong"  # >= 5 documents actually specify this category
    MODERATE = "moderate"  # 3-4 documents
    WEAK = "weak"  # 1-2 documents -> no statistical claim is made
    NONE = "none"  # nobody specified this category


class ConceptEvidence(BaseModel):
    """
    Pure arithmetic. Every field below is computed by Python from the
    concept membership map. No model output is trusted here.
    """

    concept_id: str
    category: Category
    label: str

    # --- Per-document verdicts (reference documents only) --------------
    present_in: List[int] = Field(default_factory=list)
    absent_in: List[int] = Field(default_factory=list)
    not_specified_in: List[int] = Field(default_factory=list)

    # --- Counts ---------------------------------------------------------
    total_reference_count: int = 0
    applicable_reference_count: int = 0  # present + absent
    support_count: int = 0  # == len(present_in)
    prevalence: Optional[float] = None  # support / applicable

    # --- New job --------------------------------------------------------
    new_job_has: bool = False
    new_job_texts: List[str] = Field(default_factory=list)
    reference_texts: List[str] = Field(default_factory=list)

    # --- Derived --------------------------------------------------------
    direction: Direction = Direction.ALIGNED
    benchmark_strength: BenchmarkStrength = BenchmarkStrength.NONE

    @property
    def prevalence_text(self) -> str:
        if self.applicable_reference_count == 0:
            return "No benchmark data"
        pct = 0.0 if self.prevalence is None else self.prevalence * 100.0
        return (
            f"{self.support_count} of {self.applicable_reference_count} "
            f"({pct:.0f}%)"
        )


# ==========================================================================
# STAGE 3 — MATERIALITY (LLM-owned judgement only)
# ==========================================================================


class Tier(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    STANDARD = "standard"
    LOW = "low"


TIER_WEIGHT: Dict[Tier, int] = {
    Tier.CRITICAL: 4,
    Tier.HIGH: 3,
    Tier.STANDARD: 2,
    Tier.LOW: 1,
}


class MaterialityVerdict(BaseModel):
    """What the LLM is allowed to say in stage 3. Nothing more."""

    concept_id: str
    tier: Tier = Tier.STANDARD
    business_impact: str = ""


# ==========================================================================
# STAGE 4 — FINDINGS
# ==========================================================================


class Bucket(str, Enum):
    """
    The 2x2 presentation matrix.

    prevalence (Python) x materiality (LLM) -> where a finding is shown.
    NOTHING is ever deleted. Buckets decide ORDER and PROMINENCE only.
    """

    CRITICAL_GAP = "critical_gap"  # missing + common + important
    NOTABLE_GAP = "notable_gap"  # missing + rare + important <- used to be lost
    STANDARD_GAP = "standard_gap"  # missing + common + ordinary
    ADDITIONAL = "additional"  # the new job asks for more than the benchmark
    MARKET_VARIATION = "market_variation"  # missing + rare + ordinary
    ALIGNED = "aligned"


BUCKET_LABEL: Dict[Bucket, str] = {
    Bucket.CRITICAL_GAP: "Critical Gap",
    Bucket.NOTABLE_GAP: "Notable Gap (uncommon but important)",
    Bucket.STANDARD_GAP: "Standard Gap",
    Bucket.ADDITIONAL: "Extra Requirement",
    Bucket.MARKET_VARIATION: "Minor Variation",
    Bucket.ALIGNED: "Matches the Benchmark",
}

# One-line, no-jargon explanation of each bucket, meant for a first-time
# reader who has never seen this report before. Shown once, in the
# Executive Summary's reading guide - never repeated on every row, since
# BUCKET_LABEL above is already readable on its own once explained once.
BUCKET_EXPLAINER: Dict[Bucket, str] = {
    Bucket.CRITICAL_GAP: (
        "Most similar roles ask for this, your JD does not - and it's "
        "important. Fix first."
    ),
    Bucket.NOTABLE_GAP: (
        "Only a few similar roles ask for this, but it's important where "
        "it applies. Worth a look."
    ),
    Bucket.STANDARD_GAP: (
        "Most similar roles ask for this, your JD does not - but it's "
        "routine, not urgent."
    ),
    Bucket.ADDITIONAL: (
        "Your JD asks for this and most similar roles don't. Confirm "
        "it's intentional."
    ),
    Bucket.MARKET_VARIATION: (
        "A few similar roles mention this; it's minor. For your "
        "information only."
    ),
    Bucket.ALIGNED: "Your JD already matches the other comparable jobs here.",
}

BUCKET_ORDER: List[Bucket] = [
    Bucket.CRITICAL_GAP,
    Bucket.NOTABLE_GAP,
    Bucket.STANDARD_GAP,
    Bucket.ADDITIONAL,
    Bucket.MARKET_VARIATION,
    Bucket.ALIGNED,
]


class Finding(BaseModel):
    """One actionable line of the report."""

    concept_id: str
    category: Category
    label: str

    direction: Direction
    bucket: Bucket
    tier: Tier

    support_count: int
    applicable_reference_count: int
    total_reference_count: int
    prevalence: Optional[float] = None
    benchmark_strength: BenchmarkStrength

    new_job_texts: List[str] = Field(default_factory=list)
    reference_texts: List[str] = Field(default_factory=list)
    present_in: List[int] = Field(default_factory=list)
    absent_in: List[int] = Field(default_factory=list)
    not_specified_in: List[int] = Field(default_factory=list)

    business_impact: str = ""
    priority_score: float = 0.0

    @property
    def prevalence_text(self) -> str:
        if self.applicable_reference_count == 0:
            return "No benchmark data"
        pct = 0.0 if self.prevalence is None else self.prevalence * 100.0
        return (
            f"{self.support_count} of {self.applicable_reference_count} "
            f"({pct:.0f}%)"
        )

    @property
    def action_text(self) -> str:
        if self.direction == Direction.M_MINUS:
            return "Consider adding"
        if self.direction == Direction.M_PLUS:
            return "Above benchmark — confirm intentional"
        return "No action"


# ==========================================================================
# NUMERIC EXPERIENCE BENCHMARK (fully deterministic, no LLM at all)
# ==========================================================================


class ExperienceBenchmark(BaseModel):
    new_job_years: Optional[float] = None
    reference_years: List[float] = Field(default_factory=list)
    reference_min: Optional[float] = None
    reference_max: Optional[float] = None
    reference_median: Optional[float] = None
    documents_with_years: int = 0
    total_reference_count: int = 0
    verdict: str = "Not comparable"
    detail: str = ""


# ==========================================================================
# CATEGORY + FINAL RESULT
# ==========================================================================


class CategorySummary(BaseModel):
    category: Category
    concept_count: int = 0
    aligned_count: int = 0
    m_plus_count: int = 0
    m_minus_count: int = 0
    critical_gap_count: int = 0
    notable_gap_count: int = 0
    alignment_score: Optional[float] = None  # 0..1, None = no benchmark
    benchmark_strength: BenchmarkStrength = BenchmarkStrength.NONE
    clustering_used_fallback: bool = False


class ComparisonStatusV2(str, Enum):
    COMPARED = "compared"
    NEW_JOB_CODE = "new_job_code"
    INSUFFICIENT_BENCHMARK = "insufficient_benchmark"


class EngineDiagnostics(BaseModel):
    """
    The trust layer. Everything that degraded instead of crashing is
    recorded here and rendered into the report's audit sheet.
    """

    started_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    finished_at: Optional[datetime] = None
    llm_calls: int = 0
    llm_failures: int = 0
    categories_using_fallback: List[str] = Field(default_factory=list)
    dropped_unknown_ids: List[str] = Field(default_factory=list)
    recovered_orphan_ids: List[str] = Field(default_factory=list)
    materiality_missing_for: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    def warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)


class ComparisonResult(BaseModel):
    """
    The complete, self-describing outcome of one comparison run.

    This object replaces GeminiComparisonReport as the pipeline's payload.
    It keeps `status`, `confidence` and `is_substantive_role_change` so the
    existing persistence layer needs no change.
    """

    status: ComparisonStatusV2 = ComparisonStatusV2.COMPARED
    overall_summary: str = ""

    new_job_code: Optional[str] = None
    new_job_title: Optional[str] = None
    new_company_code: Optional[str] = None

    documents: List[DocumentRef] = Field(default_factory=list)
    reference_jobs_count: int = 0

    findings: List[Finding] = Field(default_factory=list)
    category_summaries: List[CategorySummary] = Field(default_factory=list)
    experience_benchmark: Optional[ExperienceBenchmark] = None

    alignment_score: Optional[float] = None
    confidence: float = 0.0
    is_substantive_role_change: bool = False

    evidence: List[ConceptEvidence] = Field(default_factory=list)
    diagnostics: EngineDiagnostics = Field(default_factory=EngineDiagnostics)

    # ------------------------------------------------------------------
    # Convenience accessors used by the renderers
    # ------------------------------------------------------------------

    def findings_in(self, bucket: Bucket) -> List[Finding]:
        return [f for f in self.findings if f.bucket == bucket]

    @property
    def actionable_findings(self) -> List[Finding]:
        return [f for f in self.findings if f.bucket != Bucket.ALIGNED]

    def count_of(self, bucket: Bucket) -> int:
        return len(self.findings_in(bucket))


__all__ = [
    "BUCKET_EXPLAINER",
    "BUCKET_LABEL",
    "BUCKET_ORDER",
    "CATEGORY_CODE",
    "CATEGORY_ORDER",
    "NEW_JOB_DOC_INDEX",
    "TIER_WEIGHT",
    "BenchmarkStrength",
    "Bucket",
    "Category",
    "CategorySummary",
    "ComparisonResult",
    "ComparisonStatusV2",
    "Concept",
    "ConceptClusteringResult",
    "ConceptEvidence",
    "Direction",
    "DocumentRef",
    "EngineDiagnostics",
    "ExperienceBenchmark",
    "Finding",
    "MaterialityVerdict",
    "SourceItem",
    "Tier",
]
