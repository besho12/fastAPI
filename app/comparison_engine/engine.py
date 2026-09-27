"""
app/comparison/engine.py

The orchestrator.

    Stage 0  extract.py       deterministic   items with traceable ids
    Stage 1  concepts.py      LLM + guards    semantic clustering
    Stage 2  evidence.py      deterministic   counts, ratios, direction
    Stage 3  materiality.py   LLM + guards    tier + business impact
    Stage 4  materiality.py   deterministic   buckets, ranking

One rule holds across all of them: a model failure degrades the run, it
never ends it. A report that is honest about a weakened stage is far more
useful than an exception.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, List, Optional, Sequence

from app.comparison_engine.concepts import cluster_category
from app.comparison_engine.evidence import (
    MIN_DOCS_FOR_STATISTICAL_CLAIM,
    build_evidence,
    build_experience_benchmark,
)
from app.comparison_engine.extract import extract_corpus, items_by_category
from app.comparison_engine.materiality import build_findings, rate_materiality
from app.comparison_engine.models import (
    CATEGORY_ORDER,
    BenchmarkStrength,
    Bucket,
    Category,
    CategorySummary,
    ComparisonResult,
    ComparisonStatusV2,
    ConceptEvidence,
    Direction,
    EngineDiagnostics,
    Finding,
)

logger = logging.getLogger(__name__)


class ComparisonEngine:
    """
    Usage:

        engine = ComparisonEngine(gateway=GeminiGateway())
        result = engine.run(new_job, reference_jobs)

    `gateway` is optional. Without it the engine still produces a complete,
    deterministic report using exact-wording clustering and heuristic
    materiality — useful for tests, offline runs and outages.
    """

    def __init__(self, gateway: Optional[Any] = None) -> None:
        self.gateway = gateway

    # ------------------------------------------------------------------

    def run(
        self,
        new_job: Any,
        reference_jobs: Sequence[Any],
    ) -> ComparisonResult:
        diagnostics = EngineDiagnostics()

        references = list(reference_jobs or [])
        total_references = len(references)

        documents, items = extract_corpus(new_job, references)

        result = ComparisonResult(
            new_job_code=documents[0].job_code,
            new_job_title=documents[0].job_title,
            new_company_code=documents[0].company_code,
            documents=documents,
            reference_jobs_count=total_references,
            diagnostics=diagnostics,
        )

        # ------------------------------------------------------------------
        # No benchmark at all
        # ------------------------------------------------------------------

        if total_references == 0:
            result.status = ComparisonStatusV2.NEW_JOB_CODE
            result.overall_summary = (
                "This job code has no historical benchmark documents yet. "
                "The submitted job has been recorded and will serve as the "
                "first reference for future comparisons."
            )
            result.confidence = 0.0
            diagnostics.finished_at = datetime.now(timezone.utc)
            return result

        grouped = items_by_category(items)

        all_evidence: List[ConceptEvidence] = []
        summaries: List[CategorySummary] = []

        # ------------------------------------------------------------------
        # Stages 1 and 2, per category
        # ------------------------------------------------------------------

        for category in CATEGORY_ORDER:
            category_items = grouped.get(category, [])

            clustering = cluster_category(
                category=category,
                items=category_items,
                gateway=self.gateway,
            )

            if clustering.used_fallback:
                diagnostics.categories_using_fallback.append(category.value)

            diagnostics.dropped_unknown_ids.extend(
                clustering.dropped_unknown_ids
            )
            diagnostics.recovered_orphan_ids.extend(
                clustering.recovered_orphan_ids
            )

            for note in clustering.notes:
                diagnostics.warn(f"{category.value}: {note}")

            evidence = build_evidence(
                category=category,
                concepts=clustering.concepts,
                items=category_items,
                total_reference_count=total_references,
            )

            all_evidence.extend(evidence)

            summaries.append(
                self._summarise_category(
                    category=category,
                    evidence=evidence,
                    used_fallback=clustering.used_fallback,
                )
            )

        if self.gateway is not None:
            diagnostics.llm_calls = getattr(self.gateway, "call_count", 0)
            diagnostics.llm_failures = getattr(
                self.gateway, "failure_count", 0
            )

        # ------------------------------------------------------------------
        # Stage 3 — materiality, on gaps only
        # ------------------------------------------------------------------

        gaps = [
            e for e in all_evidence if e.direction != Direction.ALIGNED
        ]

        verdicts = rate_materiality(gaps, gateway=self.gateway)

        # A gap with no business_impact was tiered by the offline
        # heuristic rather than by the model. Record it so the audit sheet
        # can say so instead of presenting a guess as a judgement.
        diagnostics.materiality_missing_for = [
            gap.concept_id
            for gap in gaps
            if not (
                verdicts.get(gap.concept_id)
                and verdicts[gap.concept_id].business_impact
            )
        ]

        if self.gateway is not None:
            diagnostics.llm_calls = getattr(self.gateway, "call_count", 0)
            diagnostics.llm_failures = getattr(
                self.gateway, "failure_count", 0
            )

        # ------------------------------------------------------------------
        # Stage 4 — findings
        # ------------------------------------------------------------------

        findings = build_findings(all_evidence, verdicts)

        # ------------------------------------------------------------------
        # Assemble
        # ------------------------------------------------------------------

        result.evidence = all_evidence
        result.findings = findings
        result.category_summaries = summaries
        result.experience_benchmark = build_experience_benchmark(
            items, total_references
        )

        result.alignment_score = self._alignment_score(summaries)
        result.confidence = self._confidence(
            total_references, summaries, diagnostics
        )
        result.is_substantive_role_change = self._is_substantive(findings)

        weak_benchmark = all(
            s.benchmark_strength
            in (BenchmarkStrength.WEAK, BenchmarkStrength.NONE)
            for s in summaries
        )

        result.status = (
            ComparisonStatusV2.INSUFFICIENT_BENCHMARK
            if weak_benchmark
            else ComparisonStatusV2.COMPARED
        )

        result.overall_summary = self._summary_text(result)

        if total_references < MIN_DOCS_FOR_STATISTICAL_CLAIM:
            diagnostics.warn(
                f"Only {total_references} benchmark document(s) available. "
                "Prevalence figures are indicative, not statistical."
            )

        diagnostics.finished_at = datetime.now(timezone.utc)

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _summarise_category(
        category: Category,
        evidence: Sequence[ConceptEvidence],
        used_fallback: bool,
    ) -> CategorySummary:
        aligned = sum(
            1 for e in evidence if e.direction == Direction.ALIGNED
        )
        m_plus = sum(1 for e in evidence if e.direction == Direction.M_PLUS)
        m_minus = sum(1 for e in evidence if e.direction == Direction.M_MINUS)

        strength = (
            evidence[0].benchmark_strength
            if evidence
            else BenchmarkStrength.NONE
        )

        # Alignment is measured against what the benchmark actually asks
        # for: aligned concepts as a share of everything the benchmark
        # carries. Additions by the new job do not lower it.
        benchmark_concepts = aligned + m_minus

        score = (
            aligned / benchmark_concepts
            if benchmark_concepts > 0
            else None
        )

        return CategorySummary(
            category=category,
            concept_count=len(evidence),
            aligned_count=aligned,
            m_plus_count=m_plus,
            m_minus_count=m_minus,
            alignment_score=score,
            benchmark_strength=strength,
            clustering_used_fallback=used_fallback,
        )

    @staticmethod
    def _alignment_score(
        summaries: Sequence[CategorySummary],
    ) -> Optional[float]:
        scored = [
            s for s in summaries if s.alignment_score is not None
        ]
        if not scored:
            return None

        aligned = sum(s.aligned_count for s in scored)
        total = sum(s.aligned_count + s.m_minus_count for s in scored)

        return (aligned / total) if total else None

    @staticmethod
    def _confidence(
        total_references: int,
        summaries: Sequence[CategorySummary],
        diagnostics: EngineDiagnostics,
    ) -> float:
        """
        An honest number: how much should a reader trust this report?

        Driven by benchmark size and by how many stages had to degrade.
        """

        if total_references <= 0:
            return 0.0

        # Benchmark size, saturating at 8 documents.
        size_component = min(total_references / 8.0, 1.0)

        strong = sum(
            1
            for s in summaries
            if s.benchmark_strength
            in (BenchmarkStrength.STRONG, BenchmarkStrength.MODERATE)
        )
        coverage_component = strong / max(len(summaries), 1)

        # Degradation lowers confidence, but a fully offline run is still
        # a real comparison — the penalty is capped so the score stays
        # informative instead of collapsing to zero.
        penalty = min(
            0.35, 0.06 * len(set(diagnostics.categories_using_fallback))
        )
        penalty += 0.05 if diagnostics.llm_failures else 0.0

        score = (0.55 * size_component) + (0.45 * coverage_component) - penalty

        return round(max(0.0, min(1.0, score)), 2)

    @staticmethod
    def _is_substantive(findings: Sequence[Finding]) -> bool:
        critical = sum(
            1 for f in findings if f.bucket == Bucket.CRITICAL_GAP
        )
        notable = sum(1 for f in findings if f.bucket == Bucket.NOTABLE_GAP)

        responsibility_gaps = sum(
            1
            for f in findings
            if f.category == Category.RESPONSIBILITIES
            and f.bucket in (Bucket.CRITICAL_GAP, Bucket.NOTABLE_GAP)
        )

        return critical >= 2 or responsibility_gaps >= 3 or (
            critical >= 1 and notable >= 3
        )

    @staticmethod
    def _summary_text(result: ComparisonResult) -> str:
        critical = result.count_of(Bucket.CRITICAL_GAP)
        notable = result.count_of(Bucket.NOTABLE_GAP)
        standard = result.count_of(Bucket.STANDARD_GAP)
        additional = result.count_of(Bucket.ADDITIONAL)
        variation = result.count_of(Bucket.MARKET_VARIATION)

        alignment = (
            f"{result.alignment_score * 100:.0f}%"
            if result.alignment_score is not None
            else "not measurable"
        )

        parts = [
            f"Compared against {result.reference_jobs_count} benchmark "
            f"document(s) sharing job code "
            f"{result.new_job_code or 'N/A'}.",
            f"Overall alignment with the benchmark: {alignment}.",
        ]

        if critical or notable:
            parts.append(
                f"{critical} critical gap(s) and {notable} notable "
                "minority gap(s) require attention."
            )
        else:
            parts.append(
                "No critical or notable gaps were identified."
            )

        parts.append(
            f"{standard} standard gap(s), {additional} additional "
            f"requirement(s) beyond the benchmark and {variation} benchmark "
            "variation(s) are listed for completeness."
        )

        if result.confidence < 0.5:
            parts.append(
                "Confidence is limited; treat prevalence figures as "
                "indicative."
            )

        return " ".join(parts)


def compare(
    new_job: Any,
    reference_jobs: Sequence[Any],
    gateway: Optional[Any] = None,
) -> ComparisonResult:
    """Convenience entry point."""
    return ComparisonEngine(gateway=gateway).run(new_job, reference_jobs)


__all__ = ["ComparisonEngine", "compare"]
