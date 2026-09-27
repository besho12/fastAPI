"""
app/comparison/evidence.py

STAGE 2 — Evidence and counting. 100% Python. Zero model involvement.

Everything here is a pure function of (concepts, items). Given the same
inputs it returns the same output, forever. This is where the old system's
entire class of "Gemini returned support_count=4 but evidence has 3
MATCH records" errors stops being possible: no count ever comes from the
model, so no count can ever disagree with the evidence.
"""

from __future__ import annotations

import re
import statistics
from typing import Dict, List, Optional, Sequence, Set

from app.comparison_engine.models import (
    NEW_JOB_DOC_INDEX,
    BenchmarkStrength,
    Category,
    Concept,
    ConceptEvidence,
    Direction,
    ExperienceBenchmark,
    SourceItem,
)

# A concept is treated as shared with the benchmark, rather than an
# addition, once at least this share of applicable documents carry it.
MAJORITY_THRESHOLD = 0.5

# Below this many applicable documents we refuse to make a statistical
# claim at all and label the benchmark weak.
MIN_DOCS_FOR_STATISTICAL_CLAIM = 3


def _benchmark_strength(applicable: int) -> BenchmarkStrength:
    if applicable <= 0:
        return BenchmarkStrength.NONE
    if applicable < MIN_DOCS_FOR_STATISTICAL_CLAIM:
        return BenchmarkStrength.WEAK
    if applicable < 5:
        return BenchmarkStrength.MODERATE
    return BenchmarkStrength.STRONG


def build_evidence(
    category: Category,
    concepts: Sequence[Concept],
    items: Sequence[SourceItem],
    total_reference_count: int,
) -> List[ConceptEvidence]:
    """
    Compute per-concept evidence for one category.

    Denominator rule
    ----------------
    `applicable_reference_count` counts only reference documents that said
    SOMETHING in this category. A document that is silent about education
    is not evidence that education is unimportant — it is no evidence at
    all, and folding it into the denominator is what made every finding
    look like a minority in the previous system.
    """

    items_by_id: Dict[str, SourceItem] = {i.item_id: i for i in items}

    category_items = [i for i in items if i.category == category]

    speaking_docs: Set[int] = {i.doc_index for i in category_items}

    applicable_docs: Set[int] = {
        d for d in speaking_docs if d != NEW_JOB_DOC_INDEX
    }

    silent_docs: List[int] = sorted(
        d
        for d in range(1, total_reference_count + 1)
        if d not in applicable_docs
    )

    applicable_count = len(applicable_docs)
    strength = _benchmark_strength(applicable_count)

    evidence: List[ConceptEvidence] = []

    for concept in concepts:
        members = [
            items_by_id[item_id]
            for item_id in concept.member_item_ids
            if item_id in items_by_id
        ]

        present_docs = {m.doc_index for m in members}

        new_job_has = NEW_JOB_DOC_INDEX in present_docs

        present_in = sorted(d for d in present_docs if d != NEW_JOB_DOC_INDEX)

        absent_in = sorted(applicable_docs - set(present_in))

        support_count = len(present_in)

        prevalence = (
            support_count / applicable_count
            if applicable_count > 0
            else None
        )

        # ------------------------------------------------------------------
        # Direction. Note there is NO filtering here — every concept gets a
        # direction and every concept survives to the report.
        # ------------------------------------------------------------------

        if not new_job_has:
            direction = Direction.M_MINUS
        elif prevalence is None:
            # New job has it and there is no benchmark to compare against.
            direction = Direction.M_PLUS
        elif prevalence >= MAJORITY_THRESHOLD:
            direction = Direction.ALIGNED
        else:
            direction = Direction.M_PLUS

        evidence.append(
            ConceptEvidence(
                concept_id=concept.concept_id,
                category=category,
                label=concept.label,
                present_in=present_in,
                absent_in=absent_in,
                not_specified_in=silent_docs,
                total_reference_count=total_reference_count,
                applicable_reference_count=applicable_count,
                support_count=support_count,
                prevalence=prevalence,
                new_job_has=new_job_has,
                new_job_texts=[
                    m.text for m in members if m.doc_index == NEW_JOB_DOC_INDEX
                ],
                reference_texts=_unique_texts(
                    m.text
                    for m in members
                    if m.doc_index != NEW_JOB_DOC_INDEX
                ),
                direction=direction,
                benchmark_strength=strength,
            )
        )

    # Stable, meaningful ordering: strongest support first.
    evidence.sort(
        key=lambda e: (-e.support_count, e.label.casefold())
    )

    return evidence


def _unique_texts(texts) -> List[str]:
    seen = set()
    out: List[str] = []
    for text in texts:
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out[:8]


# ==========================================================================
# NUMERIC EXPERIENCE BENCHMARK
# ==========================================================================

_YEARS_PATTERNS = [
    re.compile(r"(\d+(?:\.\d+)?)\s*(?:\+|plus)?\s*(?:years?|yrs?|سنة|سنوات)"),
    re.compile(r"(?:years?|yrs?)\s*(?:of\s+experience)?\s*[:\-]?\s*(\d+(?:\.\d+)?)"),
    re.compile(r"^\s*(\d+(?:\.\d+)?)\s*$"),
]


def parse_years(text: str) -> Optional[float]:
    """
    Extract a minimum year count from a free-text experience requirement.

    '5-7 years' -> 5.0 (the requirement is the floor, not the ceiling).
    """

    if not text:
        return None

    lowered = str(text).casefold()

    range_match = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:-|–|—|to|إلى)\s*(\d+(?:\.\d+)?)", lowered
    )
    if range_match:
        try:
            return min(
                float(range_match.group(1)), float(range_match.group(2))
            )
        except ValueError:
            pass

    for pattern in _YEARS_PATTERNS:
        match = pattern.search(lowered)
        if match:
            try:
                value = float(match.group(1))
            except (ValueError, IndexError):
                continue
            if 0 <= value <= 60:
                return value

    return None


def build_experience_benchmark(
    items: Sequence[SourceItem],
    total_reference_count: int,
) -> ExperienceBenchmark:
    """
    A separate, fully deterministic comparison for years of experience.

    Asking a language model whether 5 is more than 3 is a category error;
    this is arithmetic and belongs in arithmetic.
    """

    experience_items = [
        i for i in items if i.category == Category.YEARS_OF_EXPERIENCE
    ]

    new_values = [
        parse_years(i.text)
        for i in experience_items
        if i.doc_index == NEW_JOB_DOC_INDEX
    ]
    new_values = [v for v in new_values if v is not None]

    new_years = min(new_values) if new_values else None

    per_document: Dict[int, List[float]] = {}

    for item in experience_items:
        if item.doc_index == NEW_JOB_DOC_INDEX:
            continue
        value = parse_years(item.text)
        if value is not None:
            per_document.setdefault(item.doc_index, []).append(value)

    reference_years = sorted(min(values) for values in per_document.values())

    benchmark = ExperienceBenchmark(
        new_job_years=new_years,
        reference_years=reference_years,
        reference_min=reference_years[0] if reference_years else None,
        reference_max=reference_years[-1] if reference_years else None,
        reference_median=(
            statistics.median(reference_years) if reference_years else None
        ),
        documents_with_years=len(reference_years),
        total_reference_count=total_reference_count,
    )

    if new_years is None and not reference_years:
        benchmark.verdict = "Not specified"
        benchmark.detail = (
            "Neither the submitted job nor the benchmark documents state a "
            "number of years."
        )
        return benchmark

    if new_years is None:
        benchmark.verdict = "Missing in submitted job"
        benchmark.detail = (
            f"The benchmark states years of experience in "
            f"{len(reference_years)} of {total_reference_count} documents "
            f"(median {benchmark.reference_median:g}), but the submitted "
            "job does not state a figure."
        )
        return benchmark

    if not reference_years:
        benchmark.verdict = "No benchmark"
        benchmark.detail = (
            f"The submitted job requires {new_years:g} year(s); no benchmark "
            "document states a comparable figure."
        )
        return benchmark

    median = benchmark.reference_median or 0.0
    difference = new_years - median

    if abs(difference) < 0.5:
        benchmark.verdict = "In line with benchmark"
    elif difference > 0:
        benchmark.verdict = f"{difference:g} year(s) above benchmark"
    else:
        benchmark.verdict = f"{abs(difference):g} year(s) below benchmark"

    benchmark.detail = (
        f"Submitted job: {new_years:g} year(s). "
        f"Benchmark median {median:g} "
        f"(range {benchmark.reference_min:g}–{benchmark.reference_max:g}), "
        f"from {len(reference_years)} of {total_reference_count} documents."
    )

    return benchmark


__all__ = [
    "MAJORITY_THRESHOLD",
    "MIN_DOCS_FOR_STATISTICAL_CLAIM",
    "build_evidence",
    "build_experience_benchmark",
    "parse_years",
]
