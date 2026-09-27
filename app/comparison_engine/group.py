"""Group-wide, all-jobs-against-all-others comparison orchestration.

The semantic concepts are discovered once for the complete job-code group.
Each job is then rotated into the target position and all arithmetic is
recomputed against the remaining jobs.  This produces one independent,
traceable report per job without asking the model to rediscover the same
concepts for every target.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field

from app.comparison_engine.concepts import cluster_category
from app.comparison_engine.engine import ComparisonEngine
from app.comparison_engine.evidence import (
    MIN_DOCS_FOR_STATISTICAL_CLAIM,
    build_evidence,
    build_experience_benchmark,
)
from app.comparison_engine.extract import (
    build_document_ref,
    extract_corpus,
    items_by_category,
)
from app.comparison_engine.materiality import build_findings, rate_materiality
from app.comparison_engine.models import (
    CATEGORY_CODE,
    CATEGORY_ORDER,
    BenchmarkStrength,
    Bucket,
    Category,
    ComparisonResult,
    ComparisonStatusV2,
    Concept,
    ConceptClusteringResult,
    ConceptEvidence,
    Direction,
    EngineDiagnostics,
    MaterialityVerdict,
    SourceItem,
)


class JobGroupComparison(BaseModel):
    """All per-job reports produced from one shared semantic analysis."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    job_code: Optional[str] = None
    jobs: List[Any] = Field(default_factory=list, exclude=True)
    reports: List[ComparisonResult] = Field(default_factory=list)

    def report_for_job_id(self, job_id: str) -> Optional[ComparisonResult]:
        for job, report in zip(self.jobs, self.reports):
            if str(getattr(job, "job_id", "")) == str(job_id):
                return report
        return None


class GroupComparisonEngine:
    """Compare every job in a job-code group against all remaining jobs."""

    def __init__(self, gateway: Optional[Any] = None) -> None:
        self.gateway = gateway

    def run(self, jobs: Sequence[Any]) -> JobGroupComparison:
        group_jobs = list(jobs or [])
        if not group_jobs:
            raise ValueError("At least one job is required for group comparison.")

        # Stable corpus indexes and item ids are created once.  The first job
        # is not semantically special at this stage; extract_corpus is reused
        # only because it already provides the required complete flattening.
        _, corpus_items = extract_corpus(group_jobs[0], group_jobs[1:])
        grouped_items = items_by_category(corpus_items)

        shared_diagnostics = EngineDiagnostics()
        clusterings: Dict[Category, ConceptClusteringResult] = {}

        for category in CATEGORY_ORDER:
            clustering = cluster_category(
                category=category,
                items=grouped_items.get(category, []),
                gateway=self.gateway,
            )
            clusterings[category] = clustering
            if clustering.used_fallback:
                shared_diagnostics.categories_using_fallback.append(
                    category.value
                )
            shared_diagnostics.dropped_unknown_ids.extend(
                clustering.dropped_unknown_ids
            )
            shared_diagnostics.recovered_orphan_ids.extend(
                clustering.recovered_orphan_ids
            )
            for note in clustering.notes:
                shared_diagnostics.warn(f"{category.value}: {note}")

        target_views: List[
            Tuple[List[Any], List[SourceItem], List[ConceptEvidence], List[bool]]
        ] = []

        for target_index in range(len(group_jobs)):
            ordered_jobs, remapped_items, id_map = self._target_view(
                group_jobs, corpus_items, target_index
            )
            remapped_by_category = items_by_category(remapped_items)
            evidence: List[ConceptEvidence] = []
            fallback_flags: List[bool] = []

            for category in CATEGORY_ORDER:
                clustering = clusterings[category]
                concepts = self._remap_concepts(clustering.concepts, id_map)
                evidence.extend(
                    build_evidence(
                        category=category,
                        concepts=concepts,
                        items=remapped_by_category.get(category, []),
                        total_reference_count=len(group_jobs) - 1,
                    )
                )
                fallback_flags.append(clustering.used_fallback)

            target_views.append(
                (ordered_jobs, remapped_items, evidence, fallback_flags)
            )

        # Materiality is an intrinsic judgement about a concept.  Rate each
        # shared concept once, then reuse that verdict for every target.  The
        # per-target prevalence and missing/additional direction remain fully
        # deterministic and are never shared.
        materiality_inputs: Dict[str, ConceptEvidence] = {}
        for _, _, evidence, _ in target_views:
            for item in evidence:
                if item.direction == Direction.ALIGNED:
                    continue
                existing = materiality_inputs.get(item.concept_id)
                if existing is None or (
                    existing.direction == Direction.M_PLUS
                    and item.direction == Direction.M_MINUS
                ):
                    materiality_inputs[item.concept_id] = item

        verdicts: Dict[str, MaterialityVerdict] = rate_materiality(
            list(materiality_inputs.values()), gateway=self.gateway
        )

        if self.gateway is not None:
            shared_diagnostics.llm_calls = getattr(
                self.gateway, "call_count", 0
            )
            shared_diagnostics.llm_failures = getattr(
                self.gateway, "failure_count", 0
            )

        shared_diagnostics.materiality_missing_for = [
            concept_id
            for concept_id in materiality_inputs
            if not (
                verdicts.get(concept_id)
                and verdicts[concept_id].business_impact
            )
        ]
        shared_diagnostics.finished_at = datetime.now(timezone.utc)

        reports = [
            self._assemble_report(
                ordered_jobs=ordered_jobs,
                items=items,
                evidence=evidence,
                fallback_flags=fallback_flags,
                verdicts=verdicts,
                shared_diagnostics=shared_diagnostics,
            )
            for ordered_jobs, items, evidence, fallback_flags in target_views
        ]

        job_code = reports[0].new_job_code if reports else None
        return JobGroupComparison(
            job_code=job_code,
            jobs=group_jobs,
            reports=reports,
        )

    @staticmethod
    def _target_view(
        jobs: Sequence[Any],
        corpus_items: Sequence[SourceItem],
        target_index: int,
    ) -> Tuple[List[Any], List[SourceItem], Dict[str, str]]:
        order = [target_index] + [
            index for index in range(len(jobs)) if index != target_index
        ]
        old_to_new_doc = {
            old_index: new_index for new_index, old_index in enumerate(order)
        }
        ordered_jobs = [jobs[index] for index in order]

        counters: Dict[Tuple[int, Category], int] = {}
        id_map: Dict[str, str] = {}
        remapped_items: List[SourceItem] = []

        for item in corpus_items:
            new_doc_index = old_to_new_doc[item.doc_index]
            counter_key = (new_doc_index, item.category)
            position = counters.get(counter_key, 0) + 1
            counters[counter_key] = position
            new_id = (
                f"{CATEGORY_CODE[item.category]}"
                f"-D{new_doc_index:02d}-{position:02d}"
            )
            id_map[item.item_id] = new_id
            document = build_document_ref(
                ordered_jobs[new_doc_index], new_doc_index
            )
            remapped_items.append(
                SourceItem(
                    item_id=new_id,
                    doc_index=new_doc_index,
                    doc_label=document.label,
                    category=item.category,
                    text=item.text,
                )
            )

        return ordered_jobs, remapped_items, id_map

    @staticmethod
    def _remap_concepts(
        concepts: Sequence[Concept], id_map: Dict[str, str]
    ) -> List[Concept]:
        return [
            concept.model_copy(
                update={
                    "member_item_ids": [
                        id_map[item_id]
                        for item_id in concept.member_item_ids
                        if item_id in id_map
                    ]
                }
            )
            for concept in concepts
        ]

    @staticmethod
    def _assemble_report(
        ordered_jobs: Sequence[Any],
        items: Sequence[SourceItem],
        evidence: Sequence[ConceptEvidence],
        fallback_flags: Sequence[bool],
        verdicts: Dict[str, MaterialityVerdict],
        shared_diagnostics: EngineDiagnostics,
    ) -> ComparisonResult:
        references = list(ordered_jobs[1:])
        documents = [
            build_document_ref(job, index)
            for index, job in enumerate(ordered_jobs)
        ]
        diagnostics = shared_diagnostics.model_copy(deep=True)
        diagnostics.warn(
            "Semantic concepts and materiality were analysed once for the "
            "whole job-code group and reused consistently for every job."
        )

        result = ComparisonResult(
            new_job_code=documents[0].job_code,
            new_job_title=documents[0].job_title,
            new_company_code=documents[0].company_code,
            documents=documents,
            reference_jobs_count=len(references),
            diagnostics=diagnostics,
        )

        if not references:
            result.status = ComparisonStatusV2.NEW_JOB_CODE
            result.overall_summary = (
                "This job code has no other job descriptions to compare."
            )
            result.confidence = 0.0
            return result

        evidence_list = list(evidence)
        findings = build_findings(evidence_list, verdicts)
        summaries = []
        for index, category in enumerate(CATEGORY_ORDER):
            category_evidence = [
                item for item in evidence_list if item.category == category
            ]
            summary = ComparisonEngine._summarise_category(
                category,
                category_evidence,
                fallback_flags[index],
            )
            summary.critical_gap_count = sum(
                1
                for finding in findings
                if finding.category == category
                and finding.bucket == Bucket.CRITICAL_GAP
            )
            summary.notable_gap_count = sum(
                1
                for finding in findings
                if finding.category == category
                and finding.bucket == Bucket.NOTABLE_GAP
            )
            summaries.append(summary)

        result.evidence = evidence_list
        result.findings = findings
        result.category_summaries = summaries
        result.experience_benchmark = build_experience_benchmark(
            items, len(references)
        )
        result.alignment_score = ComparisonEngine._alignment_score(summaries)
        result.confidence = ComparisonEngine._confidence(
            len(references), summaries, diagnostics
        )
        result.is_substantive_role_change = ComparisonEngine._is_substantive(
            findings
        )

        weak_benchmark = all(
            summary.benchmark_strength
            in (BenchmarkStrength.WEAK, BenchmarkStrength.NONE)
            for summary in summaries
        )
        result.status = (
            ComparisonStatusV2.INSUFFICIENT_BENCHMARK
            if weak_benchmark
            else ComparisonStatusV2.COMPARED
        )
        result.overall_summary = ComparisonEngine._summary_text(result)
        if len(references) < MIN_DOCS_FOR_STATISTICAL_CLAIM:
            diagnostics.warn(
                f"Only {len(references)} benchmark document(s) available. "
                "Prevalence figures are indicative, not statistical."
            )
        return result


__all__ = ["GroupComparisonEngine", "JobGroupComparison"]
