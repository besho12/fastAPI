"""
tests/test_comparison_engine.py

These tests pin down the behaviours the old engine got wrong. Run them
before every deploy:

    pytest tests/test_comparison_engine.py -v

No network access is required: the engine runs fully offline when no
gateway is supplied, and a stub gateway covers the model-backed paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import pytest

from app.comparison_engine.concepts import cluster_category
from app.comparison_engine.evidence import (
    build_evidence,
    build_experience_benchmark,
    parse_years,
)
from app.comparison_engine.extract import extract_corpus, items_by_category
from app.comparison_engine.engine import ComparisonEngine
from app.comparison_engine.excel import generate_comparison_excel
from app.comparison_engine.materiality import assign_bucket
from app.comparison_engine.models import (
    BenchmarkStrength,
    Bucket,
    Category,
    Concept,
    ComparisonStatusV2,
    ConceptEvidence,
    Direction,
    SourceItem,
    Tier,
)


# ==========================================================================
# Fixtures
# ==========================================================================


@dataclass
class _Company:
    company_code: str = ""


@dataclass
class _JobInfo:
    job_code: str = ""
    job_title: str = ""


@dataclass
class _Requirements:
    experience: List[str] = field(default_factory=list)
    field_of_experience: List[str] = field(default_factory=list)
    education: List[str] = field(default_factory=list)
    language: List[str] = field(default_factory=list)
    computer: List[str] = field(default_factory=list)
    soft_skills: List[str] = field(default_factory=list)


@dataclass
class _Job:
    job_id: str = "J"
    company: _Company = field(default_factory=_Company)
    job_information: _JobInfo = field(default_factory=_JobInfo)
    requirements: _Requirements = field(default_factory=_Requirements)
    responsibilities: List[str] = field(default_factory=list)


def make_job(code="FIN-201", company="ORG", **kwargs) -> _Job:
    return _Job(
        job_id=company,
        company=_Company(company_code=company),
        job_information=_JobInfo(job_code=code, job_title="Accountant"),
        requirements=_Requirements(
            experience=kwargs.get("experience", []),
            field_of_experience=kwargs.get("field_of_experience", []),
            education=kwargs.get("education", []),
            language=kwargs.get("language", []),
            computer=kwargs.get("computer", []),
            soft_skills=kwargs.get("soft_skills", []),
        ),
        responsibilities=kwargs.get("responsibilities", []),
    )


@pytest.fixture
def corpus():
    new_job = make_job(
        company="NEW",
        education=["Bachelor's degree in Accounting"],
        responsibilities=["Prepare financial statements"],
    )
    references = [
        make_job(
            company=f"ORG{i}",
            education=["B.Sc. Accounting"],
            responsibilities=[
                "Prepare financial statements",
                "Ensure IFRS compliance",
            ],
        )
        for i in range(1, 6)
    ]
    return new_job, references


# ==========================================================================
# Stage 0 — extraction
# ==========================================================================


def test_extraction_assigns_unique_traceable_ids(corpus):
    new_job, references = corpus
    documents, items = extract_corpus(new_job, references)

    assert documents[0].is_new_job is True
    assert len(documents) == 6

    ids = [item.item_id for item in items]
    assert len(ids) == len(set(ids)), "item ids must be unique"
    assert all(item.doc_index <= 5 for item in items)


def test_extraction_splits_newline_joined_requirements():
    job = make_job(education=["Bachelor degree\nMaster degree"])
    _, items = extract_corpus(job, [])
    education = [i.text for i in items if i.category == Category.EDUCATION]
    assert education == ["Bachelor degree", "Master degree"]


# ==========================================================================
# Stage 1 — grounding guarantees
# ==========================================================================


class _HallucinatingGateway:
    """Returns one real id, one invented id, and forgets a third."""

    call_count = 0
    failure_count = 0

    def __init__(self, real_ids):
        self.real_ids = list(real_ids)

    def generate_json(self, system_instruction, user_content,
                      response_schema, label=""):
        self.call_count += 1
        if label.startswith("clustering"):
            return {
                "concepts": [
                    {
                        "label": "invented concept",
                        "member_item_ids": [
                            self.real_ids[0],
                            "EDU-D99-99",  # does not exist
                        ],
                    }
                ]
            }
        return {"verdicts": []}


def test_hallucinated_item_ids_are_discarded_and_orphans_recovered():
    items = [
        SourceItem(
            item_id=f"EDU-D{i:02d}-01",
            doc_index=i,
            doc_label=f"D{i}",
            category=Category.EDUCATION,
            text=f"Degree {i}",
        )
        for i in range(0, 3)
    ]

    gateway = _HallucinatingGateway([i.item_id for i in items])
    result = cluster_category(Category.EDUCATION, items, gateway)

    assert "EDU-D99-99" in result.dropped_unknown_ids

    covered = {
        item_id
        for concept in result.concepts
        for item_id in concept.member_item_ids
    }
    assert covered == {i.item_id for i in items}, (
        "every real requirement must survive clustering"
    )
    assert len(result.recovered_orphan_ids) == 2


def test_engine_survives_a_totally_broken_gateway(corpus):
    class _Broken:
        call_count = 0
        failure_count = 0

        def generate_json(self, *args, **kwargs):
            raise RuntimeError("model is down")

    new_job, references = corpus
    result = ComparisonEngine(gateway=_Broken()).run(new_job, references)

    assert result.status != ComparisonStatusV2.NEW_JOB_CODE
    assert result.findings, "a degraded run must still produce findings"
    assert result.diagnostics.categories_using_fallback


# ==========================================================================
# Stage 2 — arithmetic
# ==========================================================================


def test_counts_can_never_contradict_the_evidence(corpus):
    new_job, references = corpus
    result = ComparisonEngine().run(new_job, references)

    for evidence in result.evidence:
        assert evidence.support_count == len(evidence.present_in)
        assert evidence.applicable_reference_count == (
            len(evidence.present_in) + len(evidence.absent_in)
        )
        if evidence.prevalence is not None:
            assert 0.0 <= evidence.prevalence <= 1.0


def test_silent_documents_are_excluded_from_the_denominator():
    """
    Three documents speak about education, two say nothing at all. A
    concept held by two of the three speakers is 67%, not 40%.
    """
    items = []
    for doc_index, text in [
        (0, "Bachelor in Accounting"),
        (1, "Bachelor in Accounting"),
        (2, "Bachelor in Accounting"),
        (3, "Diploma"),
    ]:
        items.append(
            SourceItem(
                item_id=f"EDU-D{doc_index:02d}-01",
                doc_index=doc_index,
                doc_label=f"D{doc_index}",
                category=Category.EDUCATION,
                text=text,
            )
        )

    concept = Concept(
        concept_id="EDU-C001",
        category=Category.EDUCATION,
        label="Bachelor in Accounting",
        member_item_ids=["EDU-D00-01", "EDU-D01-01", "EDU-D02-01"],
    )

    evidence = build_evidence(
        category=Category.EDUCATION,
        concepts=[concept],
        items=items,
        total_reference_count=5,  # documents 4 and 5 are silent
    )[0]

    assert evidence.applicable_reference_count == 3
    assert evidence.support_count == 2
    assert evidence.prevalence == pytest.approx(2 / 3)
    assert evidence.not_specified_in == [4, 5]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("5-7 years of experience", 5.0),
        ("Minimum 8 years", 8.0),
        ("6 yrs", 6.0),
        ("no figure here", None),
        ("", None),
    ],
)
def test_years_parsing(text, expected):
    assert parse_years(text) == expected


def test_experience_benchmark_is_pure_arithmetic():
    new_job = make_job(company="NEW", experience=["4 years"])
    references = [
        make_job(company="A", experience=["6 years"]),
        make_job(company="B", experience=["8 years"]),
        make_job(company="C", experience=[]),
    ]
    _, items = extract_corpus(new_job, references)

    benchmark = build_experience_benchmark(items, 3)

    assert benchmark.new_job_years == 4.0
    assert benchmark.reference_median == 7.0
    assert benchmark.documents_with_years == 2
    assert "below benchmark" in benchmark.verdict


# ==========================================================================
# Stage 4 — the bug this rewrite exists to fix
# ==========================================================================


def _evidence(prevalence_numerator, applicable, direction):
    return ConceptEvidence(
        concept_id="X-C001",
        category=Category.RESPONSIBILITIES,
        label="Ensure IFRS compliance",
        present_in=list(range(1, prevalence_numerator + 1)),
        absent_in=list(range(prevalence_numerator + 1, applicable + 1)),
        total_reference_count=applicable,
        applicable_reference_count=applicable,
        support_count=prevalence_numerator,
        prevalence=prevalence_numerator / applicable,
        new_job_has=(direction != Direction.M_MINUS),
        direction=direction,
        benchmark_strength=BenchmarkStrength.STRONG,
    )


def test_forty_percent_but_critical_is_surfaced_not_deleted():
    """
    The old engine raised an LLMError for any M- below 50% support, so a
    regulatory duty present in 4 of 10 documents could never be reported.
    It must now appear as a NOTABLE GAP.
    """
    evidence = _evidence(4, 10, Direction.M_MINUS)
    assert assign_bucket(evidence, Tier.CRITICAL) == Bucket.NOTABLE_GAP


def test_forty_percent_and_ordinary_is_still_shown_just_lower():
    evidence = _evidence(4, 10, Direction.M_MINUS)
    assert assign_bucket(evidence, Tier.STANDARD) == Bucket.MARKET_VARIATION


def test_majority_and_critical_is_a_critical_gap():
    evidence = _evidence(8, 10, Direction.M_MINUS)
    assert assign_bucket(evidence, Tier.CRITICAL) == Bucket.CRITICAL_GAP


def test_an_addition_is_never_labelled_a_gap():
    evidence = _evidence(1, 10, Direction.M_PLUS)
    assert assign_bucket(evidence, Tier.CRITICAL) == Bucket.ADDITIONAL


def test_no_concept_is_ever_dropped_from_the_report(corpus):
    new_job, references = corpus
    result = ComparisonEngine().run(new_job, references)

    assert len(result.findings) == len(result.evidence), (
        "findings and evidence must be one-to-one: nothing is filtered"
    )


def test_thin_benchmark_makes_no_statistical_claim():
    evidence = _evidence(1, 2, Direction.M_MINUS)
    evidence.benchmark_strength = BenchmarkStrength.WEAK
    assert assign_bucket(evidence, Tier.CRITICAL) == Bucket.NOTABLE_GAP
    assert assign_bucket(evidence, Tier.LOW) == Bucket.MARKET_VARIATION


# ==========================================================================
# End to end
# ==========================================================================


def test_no_reference_jobs_returns_a_clean_new_job_result():
    result = ComparisonEngine().run(make_job(company="NEW"), [])
    assert result.status == ComparisonStatusV2.NEW_JOB_CODE
    assert result.findings == []
    assert result.confidence == 0.0


def test_determinism(corpus):
    new_job, references = corpus
    first = ComparisonEngine().run(new_job, references)
    second = ComparisonEngine().run(new_job, references)

    assert [f.concept_id for f in first.findings] == [
        f.concept_id for f in second.findings
    ]
    assert first.alignment_score == second.alignment_score


def test_excel_is_generated_for_every_status(corpus):
    new_job, references = corpus

    compared = ComparisonEngine().run(new_job, references)
    payload = generate_comparison_excel(compared, new_job, references)
    assert payload[:2] == b"PK" and len(payload) > 5000

    empty = ComparisonEngine().run(new_job, [])
    payload = generate_comparison_excel(empty, new_job, [])
    assert payload[:2] == b"PK"


def test_excel_contains_the_expected_sheets(corpus):
    from io import BytesIO

    from openpyxl import load_workbook

    new_job, references = corpus
    result = ComparisonEngine().run(new_job, references)
    workbook = load_workbook(
        BytesIO(generate_comparison_excel(result, new_job, references)),
        read_only=True,
    )

    assert workbook.sheetnames == [
        "Executive Summary",
        "Findings",
        "Category Overview",
        "Appendix A - Full Comparison",
        "Appendix B - Audit Detail",
    ]
    workbook.close()