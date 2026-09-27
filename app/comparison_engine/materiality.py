"""
app/comparison/materiality.py

STAGE 3 — Materiality, and STAGE 4 — Bucketing.

The model is asked exactly one question here: how much does this matter?
It is never asked whether the concept is present, how many documents have
it, or what to show. Those are already decided, deterministically.

Bucketing is then pure Python:

  For a MISSING concept (M-):

                     |  material (critical/high)  |  ordinary (standard/low)
    ---------------- + -------------------------- + ------------------------
    majority support |  CRITICAL GAP              |  STANDARD GAP
    minority support |  NOTABLE GAP               |  MARKET VARIATION

  An ADDITIONAL concept (M+) is not a gap and is never dressed up as one.
  It goes to ADDITIONAL REQUIREMENT, ranked by materiality.

Nothing is discarded. The 40%-but-important finding that the old
majority cutoff deleted lands in NOTABLE GAP, near the top of the report.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

from pydantic import BaseModel, Field

from app.comparison_engine.evidence import (
    MAJORITY_THRESHOLD,
    MIN_DOCS_FOR_STATISTICAL_CLAIM,
)
from app.comparison_engine.gemini import LLMUnavailable
from app.comparison_engine.models import (
    TIER_WEIGHT,
    BenchmarkStrength,
    Bucket,
    ConceptEvidence,
    Direction,
    Finding,
    MaterialityVerdict,
    Tier,
)
from app.comparison_engine.prompts import (
    MATERIALITY_SYSTEM_INSTRUCTION,
    build_materiality_user_content,
)

logger = logging.getLogger(__name__)

# Gaps are rated in batches so one oversized request cannot fail the lot.
MATERIALITY_BATCH_SIZE = 25


# ==========================================================================
# Response schema
# ==========================================================================


class _VerdictOut(BaseModel):
    concept_id: str
    tier: str = "standard"
    business_impact: str = ""


class _MaterialityResponse(BaseModel):
    verdicts: List[_VerdictOut] = Field(default_factory=list)


def _coerce_tier(value: str) -> Tier:
    try:
        return Tier(str(value).strip().casefold())
    except ValueError:
        return Tier.STANDARD


# ==========================================================================
# Heuristic fallback
# ==========================================================================

_CRITICAL_SIGNALS = (
    "licen", "certif", "regulat", "complian", "audit", "legal", "statutory",
    "safety", "risk", "budget", "p&l", "authoriz", "authoris", "accredit",
    "governance", "confidential", "aml", "ifrs", "gaap", "tax",
)

_HIGH_SIGNALS = (
    "manage", "supervis", "lead", "strategy", "strategic", "design",
    "develop", "analy", "forecast", "negotiat", "report", "sap", "oracle",
    "erp", "degree", "bachelor", "master", "engineer", "responsible for",
)


def _heuristic_tier(evidence: ConceptEvidence) -> Tier:
    """Used only when the model is unavailable. Deliberately cautious."""

    text = " ".join(
        [evidence.label] + evidence.reference_texts + evidence.new_job_texts
    ).casefold()

    if any(signal in text for signal in _CRITICAL_SIGNALS):
        return Tier.CRITICAL

    if any(signal in text for signal in _HIGH_SIGNALS):
        return Tier.HIGH

    return Tier.STANDARD


# ==========================================================================
# Stage 3
# ==========================================================================


def rate_materiality(
    gaps: Sequence[ConceptEvidence],
    gateway: Optional[object] = None,
) -> Dict[str, MaterialityVerdict]:
    """
    Return {concept_id: verdict}. Never raises; degrades to heuristics.
    """

    verdicts: Dict[str, MaterialityVerdict] = {}

    gaps = list(gaps)

    if not gaps:
        return verdicts

    if gateway is None:
        for gap in gaps:
            verdicts[gap.concept_id] = MaterialityVerdict(
                concept_id=gap.concept_id,
                tier=_heuristic_tier(gap),
                business_impact="",
            )
        return verdicts

    for start in range(0, len(gaps), MATERIALITY_BATCH_SIZE):
        batch = gaps[start : start + MATERIALITY_BATCH_SIZE]

        rows = [
            {
                "concept_id": gap.concept_id,
                "category": gap.category.value,
                "label": gap.label,
                "direction": (
                    "present in the benchmark, absent from the submitted job"
                    if gap.direction == Direction.M_MINUS
                    else "present in the submitted job, uncommon in the benchmark"
                ),
                "prevalence_text": gap.prevalence_text,
                "new_job_text": " | ".join(gap.new_job_texts[:2]),
                "reference_text": " | ".join(gap.reference_texts[:3]),
            }
            for gap in batch
        ]

        try:
            payload = gateway.generate_json(
                system_instruction=MATERIALITY_SYSTEM_INSTRUCTION,
                user_content=build_materiality_user_content(rows),
                response_schema=_MaterialityResponse,
                label="materiality",
            )

            parsed = _MaterialityResponse.model_validate(payload)

            valid_ids = {gap.concept_id for gap in batch}

            for out in parsed.verdicts:
                concept_id = str(out.concept_id).strip()

                # Grounding again: a verdict for a concept we never asked
                # about is discarded rather than displayed.
                if concept_id not in valid_ids:
                    continue

                verdicts[concept_id] = MaterialityVerdict(
                    concept_id=concept_id,
                    tier=_coerce_tier(out.tier),
                    business_impact=str(out.business_impact or "").strip()[
                        :300
                    ],
                )

        except (LLMUnavailable, Exception) as exc:  # noqa: B014
            logger.warning("Materiality batch failed: %s", exc)

    # Anything the model skipped still gets a tier, from the heuristics.
    for gap in gaps:
        if gap.concept_id not in verdicts:
            verdicts[gap.concept_id] = MaterialityVerdict(
                concept_id=gap.concept_id,
                tier=_heuristic_tier(gap),
                business_impact="",
            )

    return verdicts


# ==========================================================================
# Stage 4 — Bucketing (deterministic)
# ==========================================================================


def assign_bucket(
    evidence: ConceptEvidence,
    tier: Tier,
) -> Bucket:
    if evidence.direction == Direction.ALIGNED:
        return Bucket.ALIGNED

    # The new job asks for something the benchmark largely does not. That
    # is a deliberate stretch, not a deficiency, and mislabelling it as a
    # "gap" is exactly the kind of wrong output this rewrite exists to end.
    if evidence.direction == Direction.M_PLUS:
        return Bucket.ADDITIONAL

    material = tier in (Tier.CRITICAL, Tier.HIGH)

    # With fewer than three speaking documents there is no such thing as a
    # majority, so we never claim one. Materiality alone decides.
    if evidence.applicable_reference_count < MIN_DOCS_FOR_STATISTICAL_CLAIM:
        return Bucket.NOTABLE_GAP if material else Bucket.MARKET_VARIATION

    prevalence = evidence.prevalence or 0.0
    majority = prevalence >= MAJORITY_THRESHOLD

    if material and majority:
        return Bucket.CRITICAL_GAP
    if material:
        return Bucket.NOTABLE_GAP
    if majority:
        return Bucket.STANDARD_GAP
    return Bucket.MARKET_VARIATION


_BUCKET_RANK = {
    Bucket.CRITICAL_GAP: 0,
    Bucket.NOTABLE_GAP: 1,
    Bucket.STANDARD_GAP: 2,
    Bucket.ADDITIONAL: 3,
    Bucket.MARKET_VARIATION: 4,
    Bucket.ALIGNED: 5,
}

_STRENGTH_FACTOR = {
    BenchmarkStrength.STRONG: 1.0,
    BenchmarkStrength.MODERATE: 0.85,
    BenchmarkStrength.WEAK: 0.6,
    BenchmarkStrength.NONE: 0.4,
}


def build_findings(
    evidence_list: Sequence[ConceptEvidence],
    verdicts: Dict[str, MaterialityVerdict],
) -> List[Finding]:
    """Merge deterministic evidence with materiality into ranked findings."""

    findings: List[Finding] = []

    for evidence in evidence_list:
        verdict = verdicts.get(evidence.concept_id)
        tier = verdict.tier if verdict else Tier.STANDARD
        impact = verdict.business_impact if verdict else ""

        bucket = assign_bucket(evidence, tier)

        prevalence = evidence.prevalence or 0.0

        # Priority = materiality first, prevalence second, benchmark
        # reliability as a damping factor. Fully reproducible.
        score = (
            TIER_WEIGHT[tier] * 100.0
            + prevalence * 50.0
            + (10.0 if evidence.direction == Direction.M_MINUS else 0.0)
        ) * _STRENGTH_FACTOR[evidence.benchmark_strength]

        findings.append(
            Finding(
                concept_id=evidence.concept_id,
                category=evidence.category,
                label=evidence.label,
                direction=evidence.direction,
                bucket=bucket,
                tier=tier,
                support_count=evidence.support_count,
                applicable_reference_count=(
                    evidence.applicable_reference_count
                ),
                total_reference_count=evidence.total_reference_count,
                prevalence=evidence.prevalence,
                benchmark_strength=evidence.benchmark_strength,
                new_job_texts=evidence.new_job_texts,
                reference_texts=evidence.reference_texts,
                present_in=evidence.present_in,
                absent_in=evidence.absent_in,
                not_specified_in=evidence.not_specified_in,
                business_impact=impact,
                priority_score=round(score, 2),
            )
        )

    findings.sort(
        key=lambda f: (
            _BUCKET_RANK[f.bucket],
            -f.priority_score,
            f.category.value,
            f.label.casefold(),
        )
    )

    return findings


__all__ = [
    "assign_bucket",
    "build_findings",
    "rate_materiality",
]
