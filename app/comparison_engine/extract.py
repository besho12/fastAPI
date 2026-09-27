"""
app/comparison/extract.py

STAGE 0 — Extraction (100% deterministic, no LLM).

Turns JobDescription objects into flat, ID-tagged SourceItems.

Every downstream stage speaks only in item_ids. That is what lets us
validate the LLM's output against reality instead of against itself.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from app.comparison_engine.models import (
    CATEGORY_CODE,
    CATEGORY_ORDER,
    NEW_JOB_DOC_INDEX,
    Category,
    DocumentRef,
    SourceItem,
)

# --------------------------------------------------------------------------
# Field access — tolerant of missing attributes so a malformed job never
# takes the whole run down.
# --------------------------------------------------------------------------

CATEGORY_ACCESSOR: Dict[Category, Callable[[Any], Any]] = {
    Category.YEARS_OF_EXPERIENCE: lambda j: _req(j, "experience"),
    Category.FIELD_OF_EXPERIENCE: lambda j: _req(j, "field_of_experience"),
    Category.EDUCATION: lambda j: _req(j, "education"),
    Category.LANGUAGE: lambda j: _req(j, "language"),
    Category.COMPUTER: lambda j: _req(j, "computer"),
    Category.SOFT_SKILLS: lambda j: _req(j, "soft_skills"),
    Category.RESPONSIBILITIES: lambda j: getattr(j, "responsibilities", None),
}


def _req(job: Any, field: str) -> Any:
    requirements = getattr(job, "requirements", None)
    if requirements is None:
        return None
    return getattr(requirements, field, None)


def _nested(job: Any, *paths: Sequence[str]) -> Optional[str]:
    """Try several attribute paths and return the first non-empty string."""
    for path in paths:
        current: Any = job
        for attribute in path:
            current = getattr(current, attribute, None)
            if current is None:
                break
        if current is None:
            continue
        text = str(current).strip()
        if text:
            return text
    return None


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------

_BULLET_PREFIX = re.compile(r"^\s*(?:[-*\u2022\u25cf\u25aa]|\d+[.)])\s+")
_WHITESPACE = re.compile(r"\s+")


def _clean(text: Any) -> str:
    if text is None:
        return ""
    value = str(text).replace("\u00a0", " ")
    value = _BULLET_PREFIX.sub("", value)
    value = _WHITESPACE.sub(" ", value).strip()
    return value.strip(" ;،,")


def _flatten(value: Any) -> List[str]:
    """Turn any requirement field into a flat list of clean strings."""
    if value is None:
        return []

    if isinstance(value, (list, tuple, set)):
        out: List[str] = []
        for entry in value:
            out.extend(_flatten(entry))
        return out

    if isinstance(value, dict):
        out = []
        for entry in value.values():
            out.extend(_flatten(entry))
        return out

    # A single string holding several newline-separated requirements is a
    # very common extraction artefact, so split BEFORE collapsing
    # whitespace. Never split on commas: they legitimately appear inside a
    # single requirement ("Bachelor of Commerce, Accounting").
    raw = str(value).replace("\r\n", "\n").replace("\r", "\n")

    parts = [_clean(part) for part in raw.split("\n")]

    return [part for part in parts if part]


def _dedupe_preserving_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------


def document_label(job: Any, doc_index: int) -> str:
    company = _nested(
        job,
        ("company", "company_code"),
        ("job_information", "company_code"),
        ("company_code",),
    )
    code = _nested(
        job,
        ("job_information", "job_code"),
        ("job_code",),
    )

    if doc_index == NEW_JOB_DOC_INDEX:
        return f"NEW JOB ({company or code or 'submitted'})"

    if company and code:
        return f"D{doc_index} · {company} / {code}"
    if company:
        return f"D{doc_index} · {company}"
    if code:
        return f"D{doc_index} · {code}"
    return f"D{doc_index} · Reference {doc_index}"


def build_document_ref(job: Any, doc_index: int) -> DocumentRef:
    return DocumentRef(
        doc_index=doc_index,
        label=document_label(job, doc_index),
        job_code=_nested(job, ("job_information", "job_code"), ("job_code",)),
        company_code=_nested(
            job,
            ("company", "company_code"),
            ("job_information", "company_code"),
            ("company_code",),
        ),
        job_title=_nested(
            job, ("job_information", "job_title"), ("job_title",)
        ),
        is_new_job=(doc_index == NEW_JOB_DOC_INDEX),
    )


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def extract_items(
    job: Any,
    doc_index: int,
) -> List[SourceItem]:
    """Extract every atomic item of every category from one job."""

    label = document_label(job, doc_index)
    items: List[SourceItem] = []

    for category in CATEGORY_ORDER:
        accessor = CATEGORY_ACCESSOR[category]

        try:
            raw = accessor(job)
        except Exception:  # defensive: a broken field is "not specified"
            raw = None

        values = _dedupe_preserving_order(_flatten(raw))

        for position, text in enumerate(values, start=1):
            items.append(
                SourceItem(
                    item_id=(
                        f"{CATEGORY_CODE[category]}"
                        f"-D{doc_index:02d}-{position:02d}"
                    ),
                    doc_index=doc_index,
                    doc_label=label,
                    category=category,
                    text=text,
                )
            )

    return items


def extract_corpus(
    new_job: Any,
    reference_jobs: Sequence[Any],
) -> Tuple[List[DocumentRef], List[SourceItem]]:
    """
    Build the full corpus for one comparison run.

    Returns (documents, items). Document index 0 is always the new job.
    """

    if new_job is None:
        raise ValueError("new_job must not be None")

    references = list(reference_jobs or [])

    documents: List[DocumentRef] = [
        build_document_ref(new_job, NEW_JOB_DOC_INDEX)
    ]
    items: List[SourceItem] = extract_items(new_job, NEW_JOB_DOC_INDEX)

    for offset, reference in enumerate(references, start=1):
        documents.append(build_document_ref(reference, offset))
        items.extend(extract_items(reference, offset))

    return documents, items


def items_by_category(
    items: Sequence[SourceItem],
) -> Dict[Category, List[SourceItem]]:
    grouped: Dict[Category, List[SourceItem]] = {
        category: [] for category in CATEGORY_ORDER
    }
    for item in items:
        grouped[item.category].append(item)
    return grouped


def documents_speaking_in(
    items: Sequence[SourceItem],
    category: Category,
) -> set:
    """Doc indexes that said ANYTHING in this category."""
    return {
        item.doc_index for item in items if item.category == category
    }


__all__ = [
    "build_document_ref",
    "document_label",
    "documents_speaking_in",
    "extract_corpus",
    "extract_items",
    "items_by_category",
]
