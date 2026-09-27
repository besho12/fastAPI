"""
Checklist builder for consolidated Gemini job comparison.

Purpose:
- Expose every new-job requirement.
- Expose every historical/reference document for every category.
- Explicitly represent missing/unspecified values as NOT SPECIFIED.
- Never silently omit a reference document from a category.
- Do NOT perform semantic matching, majority voting, or M+/M- classification.
  Gemini remains the sole semantic comparison engine.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List

from app.schemas import JobDescription


# ---------------------------------------------------------------------------
# Explicit marker for missing / unspecified information
# ---------------------------------------------------------------------------

NOT_SPECIFIED = "NOT SPECIFIED"


# ---------------------------------------------------------------------------
# Category -> JobDescription field mapping
# ---------------------------------------------------------------------------

CATEGORY_FIELD_MAP: Dict[str, Callable[[JobDescription], Any]] = {
    "Years of Experience": lambda job: job.requirements.experience,
    "Field of Experience": lambda job: job.requirements.field_of_experience,
    "Education Level": lambda job: job.requirements.education,
    "Language": lambda job: job.requirements.language,
    "Computer": lambda job: job.requirements.computer,
    "Soft Skills": lambda job: job.requirements.soft_skills,
    "Duties & Responsibilities": lambda job: job.responsibilities,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _job_label(job: JobDescription, index: int) -> str:
    """
    Create a stable human-readable label for a reference job.

    Prefer:
        COMPANY_CODE / JOB_CODE

    and fall back safely if one of them is unavailable.
    """

    company_code = (
        getattr(getattr(job, "company", None), "company_code", None)
        or getattr(getattr(job, "job_information", None), "company_code", None)
        or ""
    )

    job_code = (
        getattr(getattr(job, "job_information", None), "job_code", None)
        or ""
    )

    company_code = str(company_code).strip()
    job_code = str(job_code).strip()

    if company_code and job_code:
        return f"{company_code} / {job_code}"

    if company_code:
        return company_code

    if job_code:
        return job_code

    return f"REFERENCE_DOCUMENT_{index + 1}"


def _normalize_items(value: Any) -> List[str]:
    """
    Convert a requirement field into a clean list of strings.

    Empty / missing / None values become [NOT_SPECIFIED].

    Important:
    This function only normalizes structure.
    It does NOT determine semantic equivalence.
    """

    if value is None:
        return [NOT_SPECIFIED]

    # Already a list / tuple / set
    if isinstance(value, (list, tuple, set)):
        items: List[str] = []

        for item in value:
            if item is None:
                continue

            text = str(item).strip()

            if text:
                items.append(text)

        return items if items else [NOT_SPECIFIED]

    # Single scalar value
    text = str(value).strip()

    if not text:
        return [NOT_SPECIFIED]

    return [text]


def _get_category_items(
    job: JobDescription,
    category: str,
) -> List[str]:
    """
    Extract and normalize items for one category from one job.
    """

    getter = CATEGORY_FIELD_MAP.get(category)

    if getter is None:
        raise ValueError(
            f"Unsupported comparison category: {category}"
        )

    try:
        value = getter(job)
    except Exception:
        # If the field cannot be read, represent it explicitly as unspecified.
        return [NOT_SPECIFIED]

    return _normalize_items(value)


# ---------------------------------------------------------------------------
# Main checklist builder
# ---------------------------------------------------------------------------

def build_category_checklists(
    new_job: JobDescription,
    reference_jobs: List[JobDescription],
) -> Dict[str, Dict[str, Any]]:
    """
    Build exhaustive category checklists for the consolidated Gemini prompt.

    For EVERY category:
        - Include NEW JOB items.
        - Include EVERY historical/reference document.
        - Never omit a historical document because its category is empty.
        - Represent missing information explicitly as NOT SPECIFIED.

    Example:

        "Education Level": {
            "new_job_items": ["NOT SPECIFIED"],
            "historical_items_by_document": {
                "ORG001 / JOB001": ["Bachelor of Commerce"],
                "ORG002 / JOB002": ["NOT SPECIFIED"],
                "ORG003 / JOB003": ["Bachelor of Accounting"],
            },
            "historical_document_count": 3,
        }

    This is intentionally NOT doing semantic comparison.
    Gemini remains responsible for:
        - semantic equivalence
        - matching
        - strict majority
        - M+
        - M-
        - significance
        - business impact
    """

    if new_job is None:
        raise ValueError("new_job cannot be None")

    if reference_jobs is None:
        reference_jobs = []

    checklists: Dict[str, Dict[str, Any]] = {}

    for category in CATEGORY_FIELD_MAP:
        # ---------------------------------------------------------------
        # NEW JOB
        # ---------------------------------------------------------------

        new_items = _get_category_items(
            job=new_job,
            category=category,
        )

        # ---------------------------------------------------------------
        # HISTORICAL / REFERENCE JOBS
        # ---------------------------------------------------------------
        #
        # IMPORTANT:
        # We iterate over ALL reference jobs.
        #
        # The previous implementation only added a reference document
        # when it had items:
        #
        #     if items:
        #         historical_items_by_document[...] = items
        #
        # That silently removed documents whose value was missing.
        #
        # Now every reference is represented explicitly.
        # ---------------------------------------------------------------

        historical_items_by_document: Dict[str, List[str]] = {}
        unspecified_document_indexes: List[int] = []

        for index, reference_job in enumerate(reference_jobs):
            label = _job_label(
                job=reference_job,
                index=index,
            )

            # Two references can share company_code / job_code. As a dict
            # key that silently overwrote one document, so make the key
            # unique while keeping the document order.
            if label in historical_items_by_document:
                label = f"{label} (#{index + 1})"

            items = _get_category_items(
                job=reference_job,
                category=category,
            )

            historical_items_by_document[label] = items

            # Deterministic, code-side fact: this document has NO data in
            # this category. 1-based, same numbering as the prompt.
            if items == [NOT_SPECIFIED]:
                unspecified_document_indexes.append(index + 1)

        # ---------------------------------------------------------------
        # STORE CHECKLIST
        # ---------------------------------------------------------------

        checklists[category] = {
            "new_job_items": new_items,
            "historical_items_by_document": historical_items_by_document,
            "historical_document_count": len(reference_jobs),
            "unspecified_document_indexes": unspecified_document_indexes,
        }

    return checklists


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

def render_checklists_text(
    checklists: Dict[str, Dict[str, Any]],
) -> str:
    """
    Render checklists into deterministic text for the Gemini prompt.

    Every category contains:
        1. NEW JOB
        2. ALL HISTORICAL DOCUMENTS

    Missing values are explicitly shown as:
        NOT SPECIFIED

    This function only renders data.
    It does NOT perform semantic comparison or majority logic.
    """

    if not checklists:
        return "No comparison checklist available."

    sections: List[str] = []

    for category, data in checklists.items():
        sections.append(f"### CATEGORY: {category}")

        # ---------------------------------------------------------------
        # NEW JOB
        # ---------------------------------------------------------------

        sections.append("NEW JOB:")

        new_items = data.get("new_job_items") or [NOT_SPECIFIED]

        for index, item in enumerate(new_items, start=1):
            sections.append(f"  {index}. {item}")

        # ---------------------------------------------------------------
        # HISTORICAL DOCUMENTS
        # ---------------------------------------------------------------

        sections.append("HISTORICAL / REFERENCE DOCUMENTS:")

        historical_items = data.get(
            "historical_items_by_document",
            {},
        )

        historical_count = data.get(
            "historical_document_count",
            0,
        )

        if historical_count == 0:
            sections.append("  NO HISTORICAL DOCUMENTS")
        else:
            # Explicitly iterate over every document.
            for document_index, (label, items) in enumerate(
                historical_items.items(),
                start=1,
            ):
                sections.append(
                    f"  DOCUMENT {document_index}: {label}"
                )

                normalized_items = items or [NOT_SPECIFIED]

                for item_index, item in enumerate(
                    normalized_items,
                    start=1,
                ):
                    sections.append(
                        f"    {item_index}. {item}"
                    )

        sections.append(
            f"HISTORICAL DOCUMENT COUNT: {historical_count}"
        )

        unspecified = data.get("unspecified_document_indexes") or []

        if unspecified:
            sections.append(
                "DOCUMENTS WITH NO DATA IN THIS CATEGORY "
                "(use NOT_SPECIFIED for them): "
                + ", ".join(str(i) for i in unspecified)
            )

        sections.append("")

    return "\n".join(sections).strip()