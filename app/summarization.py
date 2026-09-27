"""
app/summarization.py
"""

from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Dict, Iterable, List, Optional, Set, Tuple

from app.comparison import normalize_text

from app.schemas import (
    AggregatedListItem,
    ComparisonResult,
    ComparisonSet,
    ComparisonStatus,
    ComparisonSummary,
    ExperienceSummary,
    FieldSummary,
    JobIdentitySummary,
    ListSummary,
)
_PERCENTAGE_DECIMAL_PLACES = 2

def _percentage(
    count: int,
    total: int,
) -> float:
    """
    Calculate a percentage safely.
    """

    if total <= 0:
        return 0.0

    return round(
        (count / total) * 100,
        _PERCENTAGE_DECIMAL_PLACES,
    )
def _empty_field_summary() -> FieldSummary:
    """
    Return an empty scalar field summary.
    """
    return FieldSummary(
        matches=0,
        mismatches=0,
        match_percentage=0.0,
    )
def _empty_list_summary() -> ListSummary:
    """
    Return an empty list summary.
    """

    return ListSummary(
        common=[],
        new_only=[],
        reference_only=[],
    )
def _validate_comparison_set(
    comparison_set: ComparisonSet,
) -> None:
    """
    Validate the structural consistency of ComparisonSet.
    """

    if comparison_set is None:
        raise ValueError(
            "comparison_set must not be None"
        )

    if not isinstance(
        comparison_set,
        ComparisonSet,
    ):
        raise TypeError(
            "comparison_set must be a ComparisonSet instance"
        )

    if comparison_set.reference_jobs_count < 0:
        raise ValueError(
            "reference_jobs_count cannot be negative"
        )

    comparison_count = len(
        comparison_set.comparisons
    )
    if (
        comparison_set.status
        == ComparisonStatus.NEW_JOB_GROUP
    ):
        if (
            comparison_set.reference_jobs_count
            != 0
        ):
            raise ValueError(
                "NEW_JOB_GROUP must have "
                "reference_jobs_count=0"
            )

        if comparison_count != 0:
            raise ValueError(
                "NEW_JOB_GROUP must have no comparisons"
            )

        return
    if (
        comparison_set.status
        == ComparisonStatus.COMPARED
    ):
        if (
            comparison_set.reference_jobs_count
            != comparison_count
        ):
            raise ValueError(
                "reference_jobs_count does not match "
                "the number of comparisons"
            )
def _summarize_field(
    comparisons: Iterable[ComparisonResult],
    field_name: str,
) -> FieldSummary:
    """
    Aggregate a scalar comparison field.

    Example:

        comparison.job_purpose
        comparison.job_title
        comparison.department

    The field is expected to be a FieldComparison-like object
    containing a boolean `match`.
    """

    matches = 0
    mismatches = 0

    for comparison in comparisons:

        field_comparison = getattr(
            comparison,
            field_name,
        )

        if field_comparison.match:
            matches += 1
        else:
            mismatches += 1

    total = matches + mismatches

    return FieldSummary(
        matches=matches,
        mismatches=mismatches,
        match_percentage=_percentage(
            matches,
            total,
        ),
    )
def _summarize_identity_field(
    comparisons: Iterable[ComparisonResult],
    field_name: str,
) -> FieldSummary:
    """
    Aggregate one Job Identity field.

    IMPORTANT:
        ComparisonResult in the current schema stores job identity
        fields directly.

    Therefore we use:

        comparison.job_title

    instead of:

        comparison.job_identity.job_title

    This keeps summarization aligned with ComparisonResult.
    """

    matches = 0
    mismatches = 0

    for comparison in comparisons:

        field_comparison = getattr(
            comparison,
            field_name,
        )

        if field_comparison.match:
            matches += 1
        else:
            mismatches += 1

    total = matches + mismatches

    return FieldSummary(
        matches=matches,
        mismatches=mismatches,
        match_percentage=_percentage(
            matches,
            total,
        ),
    )
def _summarize_job_identity(
    comparisons: List[ComparisonResult],
) -> JobIdentitySummary:
    """
    Aggregate all Job Identity fields.
    """

    return JobIdentitySummary(
        job_title=_summarize_identity_field(
            comparisons,
            "job_title",
        ),
        department=_summarize_identity_field(
            comparisons,
            "department",
        ),
        grade=_summarize_identity_field(
            comparisons,
            "grade",
        ),
        job_code=_summarize_identity_field(
            comparisons,
            "job_code",
        ),
    )

def _canonical_key(
    value: str,
) -> str:
    """
    Canonical aggregation key.

    The key is intentionally deterministic and based on the same
    normalization used by comparison.py.

    Original text is preserved separately.
    """

    return normalize_text(value)


def _merge_items(
    target: Dict[str, AggregatedListItem],
    value: str,
    count: int,
    percentage: float,
) -> None:
    """
    Merge an aggregated item into a dictionary.
    """

    key = _canonical_key(value)

    if not key:
        return

    existing = target.get(key)

    if existing is None:

        target[key] = AggregatedListItem(
            value=value,
            count=count,
            percentage=percentage,
        )

        return
    if count > existing.count:

        target[key] = AggregatedListItem(
            value=existing.value,
            count=count,
            percentage=percentage,
        )


def _aggregate_list_category(
    comparisons: List[ComparisonResult],
    attribute: str,
    category: str,
    total_references: int,
) -> Dict[str, AggregatedListItem]:
    """
    Aggregate one category of a list comparison.

    Each concept counts at most once per reference job.
    """

    result: Dict[
        str,
        AggregatedListItem,
    ] = {}

    for comparison in comparisons:

        list_comparison = getattr(
            comparison,
            attribute,
        )

        values = getattr(
            list_comparison,
            category,
        )

        # One value counts once per reference job.
        seen_in_reference: Set[str] = set()

        for value in values or []:

            if value is None:
                continue

            value = str(value).strip()

            if not value:
                continue

            key = _canonical_key(value)

            if not key:
                continue

            if key in seen_in_reference:
                continue

            seen_in_reference.add(key)

            if key not in result:

                result[key] = AggregatedListItem(
                    value=value,
                    count=1,
                    percentage=_percentage(
                        1,
                        total_references,
                    ),
                )

            else:

                existing = result[key]

                result[key] = AggregatedListItem(
                    value=existing.value,
                    count=existing.count + 1,
                    percentage=_percentage(
                        existing.count + 1,
                        total_references,
                    ),
                )

    return result
def _aggregate_list_field(
    comparisons: List[ComparisonResult],
    attribute: str,
    total_references: int,
) -> ListSummary:
    """
    Aggregate a complete list comparison field.

    COMMON always has priority over NEW_ONLY and REFERENCE_ONLY.
    """

    common = _aggregate_list_category(
        comparisons,
        attribute,
        "common",
        total_references,
    )

    new_only = _aggregate_list_category(
        comparisons,
        attribute,
        "new_only",
        total_references,
    )

    reference_only = _aggregate_list_category(
        comparisons,
        attribute,
        "reference_only",
        total_references,
    )
    common_keys = set(
        common.keys()
    )

    new_only = {
        key: item
        for key, item in new_only.items()
        if key not in common_keys
    }

    reference_only = {
        key: item
        for key, item in reference_only.items()
        if key not in common_keys
    }
    def _sort(
        items: Dict[
            str,
            AggregatedListItem,
        ],
    ) -> List[AggregatedListItem]:

        values = list(
            items.values()
        )

        values.sort(
            key=lambda item: (
                -item.count,
                item.value.casefold(),
            )
        )

        return values

    return ListSummary(
        common=_sort(common),
        new_only=_sort(new_only),
        reference_only=_sort(reference_only),
    )
def _statistics(
    values: Iterable[Optional[float]],
) -> Tuple[
    Optional[float],
    Optional[float],
    Optional[float],
]:
    """
    Calculate min, max and average for numeric values.
    """

    numeric_values = [
        value
        for value in values
        if value is not None
    ]

    if not numeric_values:
        return None, None, None

    return (
        min(numeric_values),
        max(numeric_values),
        round(
            mean(numeric_values),
            _PERCENTAGE_DECIMAL_PLACES,
        ),
    )


def _get_consistent_new_years(
    comparisons: Iterable[ComparisonResult],
) -> Optional[float]:
    """
    Return the new-job experience value when all comparisons
    agree on the same value.
    """

    values = {
        comparison.experience.new_years
        for comparison in comparisons
        if comparison.experience.new_years
        is not None
    }

    if not values:
        return None

    if len(values) == 1:
        return next(iter(values))

    return None


def _summarize_experience(
    comparisons: List[ComparisonResult],
    total_references: int,
) -> ExperienceSummary:
    """
    Aggregate experience comparisons across all references.
    """

    common = _aggregate_list_category(
        comparisons,
        "experience",
        "common",
        total_references,
    )

    new_only = _aggregate_list_category(
        comparisons,
        "experience",
        "new_only",
        total_references,
    )

    reference_only = _aggregate_list_category(
        comparisons,
        "experience",
        "reference_only",
        total_references,
    )
    common_keys = set(
        common.keys()
    )

    new_only = {
        key: value
        for key, value in new_only.items()
        if key not in common_keys
    }

    reference_only = {
        key: value
        for key, value in reference_only.items()
        if key not in common_keys
    }
    def _sort(
        items: Dict[
            str,
            AggregatedListItem,
        ],
    ) -> List[AggregatedListItem]:

        values = list(
            items.values()
        )

        values.sort(
            key=lambda item: (
                -item.count,
                item.value.casefold(),
            )
        )

        return values
    new_years = _get_consistent_new_years(
        comparisons
    )

    reference_years = [
        comparison.experience.reference_years
        for comparison in comparisons
    ]

    years_difference = [
        comparison.experience.years_difference
        for comparison in comparisons
    ]

    (
        reference_years_min,
        reference_years_max,
        reference_years_average,
    ) = _statistics(
        reference_years
    )

    (
        years_difference_min,
        years_difference_max,
        years_difference_average,
    ) = _statistics(
        years_difference
    )

    comparable_references = sum(
        1
        for comparison in comparisons
        if (
            comparison.experience.new_years
            is not None
            and comparison.experience.reference_years
            is not None
            and comparison.experience.years_difference
            is not None
        )
    )

    return ExperienceSummary(
        common=_sort(common),
        new_only=_sort(new_only),
        reference_only=_sort(reference_only),

        new_years=new_years,

        reference_years_min=reference_years_min,
        reference_years_max=reference_years_max,
        reference_years_average=reference_years_average,

        years_difference_min=years_difference_min,
        years_difference_max=years_difference_max,
        years_difference_average=years_difference_average,

        comparable_references=comparable_references,
    )
def summarize(
    comparison_set: ComparisonSet,
) -> ComparisonSummary:
    """
    Convert pair-level comparisons into one ComparisonSummary.

    This function is deterministic and does not call any LLM.
    """

    _validate_comparison_set(
        comparison_set
    )

    comparisons = list(
        comparison_set.comparisons
    )

    total_references = (
        comparison_set.reference_jobs_count
    )
    if (
        comparison_set.status
        == ComparisonStatus.NEW_JOB_GROUP
    ):

        empty_identity = JobIdentitySummary(
            job_title=_empty_field_summary(),
            department=_empty_field_summary(),
            grade=_empty_field_summary(),
            job_code=_empty_field_summary(),
        )

        empty_experience = ExperienceSummary(
            common=[],
            new_only=[],
            reference_only=[],

            new_years=None,

            reference_years_min=None,
            reference_years_max=None,
            reference_years_average=None,

            years_difference_min=None,
            years_difference_max=None,
            years_difference_average=None,

            comparable_references=0,
        )

        return ComparisonSummary(
            status=ComparisonStatus.NEW_JOB_GROUP,

            new_job_id=comparison_set.new_job_id,
            new_job_code=comparison_set.new_job_code,

            reference_jobs_count=0,

            job_identity=empty_identity,

            job_purpose=_empty_field_summary(),

            experience=empty_experience,

            education=_empty_list_summary(),
            skills=_empty_list_summary(),
            responsibilities=_empty_list_summary(),
        )
    return ComparisonSummary(
        status=ComparisonStatus.COMPARED,

        new_job_id=comparison_set.new_job_id,
        new_job_code=comparison_set.new_job_code,

        reference_jobs_count=total_references,
        job_identity=_summarize_job_identity(
            comparisons
        ),

        job_purpose=_summarize_field(
            comparisons,
            "job_purpose",
        ),

        experience=_summarize_experience(
            comparisons,
            total_references,
        ),

        education=_aggregate_list_field(
            comparisons,
            "education",
            total_references,
        ),

        skills=_aggregate_list_field(
            comparisons,
            "skills",
            total_references,
        ),

        responsibilities=_aggregate_list_field(
            comparisons,
            "responsibilities",
            total_references,
        ),
    )
__all__ = [
    "summarize",
]