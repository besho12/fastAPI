"""
app/comparison/concepts.py

STAGE 1 — Canonicalisation.

This is the only stage where the model is allowed to be creative, and the
only stage whose output can be wrong in an interesting way. So it is the
stage with the hardest guard rails:

  1. Every returned member id must exist in the corpus. Unknown ids are
     dropped and recorded — never silently accepted.
  2. Every corpus id must end up in exactly one concept. Duplicates are
     de-assigned; orphans are recovered as single-member concepts.
  3. If the model fails entirely, a deterministic string-normalisation
     clusterer takes over so the run still completes with honest,
     slightly coarser results.

Guarantee: |union of member ids| == |corpus ids|, always.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Dict, List, Optional, Sequence

from pydantic import BaseModel, Field

from app.comparison_engine.gemini import LLMUnavailable
from app.comparison_engine.models import (
    CATEGORY_CODE,
    Category,
    Concept,
    ConceptClusteringResult,
    SourceItem,
)
from app.comparison_engine.prompts import (
    CLUSTERING_SYSTEM_INSTRUCTION,
    build_clustering_user_content,
)

logger = logging.getLogger(__name__)


# ==========================================================================
# Response schema handed to Gemini
# ==========================================================================


class _ConceptOut(BaseModel):
    label: str
    member_item_ids: List[str] = Field(default_factory=list)


class _ClusteringResponse(BaseModel):
    concepts: List[_ConceptOut] = Field(default_factory=list)


# ==========================================================================
# Deterministic fallback clusterer
# ==========================================================================

_PUNCTUATION = re.compile(r"[^\w\s]", flags=re.UNICODE)
_SPACES = re.compile(r"\s+")

_STOPWORDS = {
    "a", "an", "and", "the", "of", "in", "on", "for", "to", "with", "or",
    "at", "by", "as", "is", "are", "be", "must", "should", "shall", "have",
    "has", "having", "minimum", "min", "least", "good", "strong", "very",
    "ability", "able", "skills", "skill", "knowledge", "experience",
    "degree", "excellent", "proven", "solid", "years", "year",
    "microsoft", "ms", "advanced", "basic", "intermediate", "proficiency",
    "proficient", "working", "preferred", "required", "familiarity",
    "understanding", "level", "etc", "related", "field", "relevant",
}

_SYNONYM_CANON = {
    "bsc": "bachelor", "bs": "bachelor", "ba": "bachelor",
    "bachelors": "bachelor", "bachelor's": "bachelor",
    "undergraduate": "bachelor", "licence": "bachelor",
    "msc": "master", "ms": "master", "masters": "master",
    "master's": "master", "mba": "master",
    "phd": "doctorate", "dphil": "doctorate",
    "excel": "excel", "msexcel": "excel", "spreadsheets": "excel",
    "msoffice": "office", "microsoftoffice": "office",
    "english": "english", "arabic": "arabic",
    "accounting": "accounting", "accountancy": "accounting",
    "finance": "finance", "financial": "finance",
    "communication": "communication", "communications": "communication",
    "leadership": "leadership", "leading": "leadership",
    "supervision": "supervision", "supervising": "supervision",
    "supervise": "supervision", "management": "management",
    "managing": "management", "manage": "management",
}


def _fingerprint(text: str) -> str:
    """Aggressive normalisation used only by the offline fallback."""
    lowered = unicodedata.normalize("NFKD", text).casefold()
    lowered = _PUNCTUATION.sub(" ", lowered)
    lowered = _SPACES.sub(" ", lowered).strip()

    tokens = []
    for token in lowered.split(" "):
        token = _SYNONYM_CANON.get(token, token)
        if token and token not in _STOPWORDS and not token.isdigit():
            tokens.append(token)

    if not tokens:
        tokens = lowered.split(" ")

    return " ".join(sorted(set(tokens)))


def _fallback_cluster(
    category: Category,
    items: Sequence[SourceItem],
) -> List[Concept]:
    """
    Group by normalised fingerprint. Conservative on purpose: it will
    under-merge rather than wrongly merge two different requirements.
    """

    buckets: Dict[str, List[SourceItem]] = {}

    for item in items:
        buckets.setdefault(_fingerprint(item.text), []).append(item)

    concepts: List[Concept] = []

    for index, (_, members) in enumerate(buckets.items(), start=1):
        # Prefer the shortest wording as the label: it is usually the
        # cleanest phrasing of the same requirement.
        label = min((m.text for m in members), key=len)

        concepts.append(
            Concept(
                concept_id=f"{CATEGORY_CODE[category]}-C{index:03d}",
                category=category,
                label=label[:160],
                member_item_ids=[m.item_id for m in members],
                from_fallback=True,
            )
        )

    return concepts


# ==========================================================================
# Validation / repair
# ==========================================================================


def _validate_and_repair(
    category: Category,
    items: Sequence[SourceItem],
    raw_concepts: Sequence[_ConceptOut],
) -> ConceptClusteringResult:
    """
    Enforce the grounding guarantee. This function is the reason a
    hallucinated requirement cannot reach the report: anything that does
    not map onto a real, extracted line is discarded here.
    """

    known: Dict[str, SourceItem] = {item.item_id: item for item in items}

    assigned: set = set()
    dropped_unknown: List[str] = []
    notes: List[str] = []

    concepts: List[Concept] = []
    counter = 0

    for raw in raw_concepts:
        members: List[str] = []

        for item_id in raw.member_item_ids or []:
            normalized = str(item_id).strip()

            if normalized not in known:
                dropped_unknown.append(normalized)
                continue

            if normalized in assigned:
                # The model put the same line in two concepts. Keep the
                # first assignment; a line has exactly one meaning.
                notes.append(
                    f"{normalized} was assigned more than once; kept the "
                    "first concept."
                )
                continue

            assigned.add(normalized)
            members.append(normalized)

        if not members:
            continue

        counter += 1
        label = str(raw.label or "").strip()

        if not label:
            label = known[members[0]].text

        concepts.append(
            Concept(
                concept_id=f"{CATEGORY_CODE[category]}-C{counter:03d}",
                category=category,
                label=label[:160],
                member_item_ids=members,
            )
        )

    # ------------------------------------------------------------------
    # Orphan recovery — the "never silently drop a requirement" rule.
    # ------------------------------------------------------------------

    orphans = [item for item in items if item.item_id not in assigned]

    for orphan in orphans:
        counter += 1
        concepts.append(
            Concept(
                concept_id=f"{CATEGORY_CODE[category]}-C{counter:03d}",
                category=category,
                label=orphan.text[:160],
                member_item_ids=[orphan.item_id],
                from_fallback=True,
            )
        )

    return ConceptClusteringResult(
        category=category,
        concepts=concepts,
        used_fallback=False,
        dropped_unknown_ids=dropped_unknown,
        recovered_orphan_ids=[o.item_id for o in orphans],
        notes=notes,
    )


# ==========================================================================
# Public API
# ==========================================================================


def cluster_category(
    category: Category,
    items: Sequence[SourceItem],
    gateway: Optional[object] = None,
) -> ConceptClusteringResult:
    """
    Canonicalise one category. Never raises for model-side problems.
    """

    items = list(items)

    if not items:
        return ConceptClusteringResult(category=category, concepts=[])

    # A single item cannot be clustered against anything.
    if len(items) == 1 or gateway is None:
        return ConceptClusteringResult(
            category=category,
            concepts=_fallback_cluster(category, items),
            used_fallback=gateway is None,
            notes=(
                []
                if gateway is not None
                else ["No LLM gateway supplied; used deterministic clustering."]
            ),
        )

    try:
        payload = gateway.generate_json(
            system_instruction=CLUSTERING_SYSTEM_INSTRUCTION,
            user_content=build_clustering_user_content(category, items),
            response_schema=_ClusteringResponse,
            label=f"clustering[{category.value}]",
        )

        parsed = _ClusteringResponse.model_validate(payload)

    except (LLMUnavailable, Exception) as exc:  # noqa: B014 - deliberate
        logger.warning(
            "Clustering fell back to deterministic mode for %s: %s",
            category.value,
            exc,
        )

        return ConceptClusteringResult(
            category=category,
            concepts=_fallback_cluster(category, items),
            used_fallback=True,
            notes=[
                "Semantic clustering was unavailable; results for this "
                "category use exact-wording matching and may over-report "
                "differences."
            ],
        )

    result = _validate_and_repair(category, items, parsed.concepts)

    # Sanity: the grounding guarantee must hold.
    covered = {
        item_id
        for concept in result.concepts
        for item_id in concept.member_item_ids
    }

    if covered != {item.item_id for item in items}:  # pragma: no cover
        logger.error(
            "Coverage invariant violated for %s; using fallback.",
            category.value,
        )
        return ConceptClusteringResult(
            category=category,
            concepts=_fallback_cluster(category, items),
            used_fallback=True,
            notes=["Coverage invariant failed; deterministic clustering used."],
        )

    return result


__all__ = ["cluster_category"]
