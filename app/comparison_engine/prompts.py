"""
app/comparison/prompts.py

Two short, single-purpose prompts. Nothing else.

WHY THEY ARE SHORT
------------------
A 2,000-line prompt asking for clustering + matching + counting +
classification + significance in one shot does not produce a more careful
model; it produces a diluted one. Each prompt below asks for exactly one
kind of output, and the caller validates that output against the corpus.
"""

from __future__ import annotations

from typing import Sequence

from app.comparison_engine.models import Category, SourceItem


# ==========================================================================
# STAGE 1 — CANONICALISATION
# ==========================================================================

CLUSTERING_SYSTEM_INSTRUCTION = """\
You are a job-description normalisation engine.

YOUR ONLY TASK
Group requirement lines that mean THE SAME THING into one concept.

YOU MUST NOT
- Compare documents.
- Decide what is missing, extra, common or rare.
- Count anything, or output any number, ratio or percentage.
- Judge importance.
- Invent, reword, translate or summarise the source lines.

GROUPING RULES
1. Two lines belong to the same concept when a hiring manager would treat
   satisfying one as satisfying the other.
     "Bachelor's degree in Accounting" / "B.Sc. Accounting" /
     "University degree in Accounting" -> ONE concept.
     "Bachelor's degree in Accounting" / "Master's degree in Accounting"
     -> TWO concepts (different level).
2. Different scope = different concept. "Supervise the accounting team"
   and "Train accounting staff" are two concepts, not one merged concept.
3. Broader-vs-narrower: keep separate unless the narrower is only a
   rewording. "Microsoft Excel" and "MS Office" are separate concepts.
4. Language differences are wording, not meaning. An Arabic line and an
   English line expressing the same requirement are ONE concept.
5. A line that matches nothing else becomes its own single-member concept.
   This is normal and expected — do not force lines together.
6. EVERY item id given to you must appear in exactly one concept. Never
   drop one, never repeat one.

LABELLING
`label` is a neutral English phrase naming the concept in at most 12
words. It describes the group; it is not a quote and not a judgement.

OUTPUT
Return JSON only, matching the provided schema. No prose, no markdown.
"""


def build_clustering_user_content(
    category: Category,
    items: Sequence[SourceItem],
) -> str:
    """Render one category's items as an id-tagged list."""

    lines = [
        f"CATEGORY: {category.value}",
        f"TOTAL ITEMS: {len(items)}",
        "",
        "ITEMS (format: <item_id> | <verbatim text>):",
    ]

    for item in items:
        lines.append(f"{item.item_id} | {item.text}")

    lines.extend(
        [
            "",
            "Group these item ids into semantic concepts.",
            "Every id listed above must appear in exactly one concept.",
        ]
    )

    return "\n".join(lines)


# ==========================================================================
# STAGE 3 — MATERIALITY
# ==========================================================================

MATERIALITY_SYSTEM_INSTRUCTION = """\
You are a compensation and job-architecture analyst.

YOUR ONLY TASK
For each listed concept, decide HOW MUCH IT MATTERS and say why in one
short sentence.

YOU MUST NOT
- Decide whether the concept is present or missing. That is already
  settled and given to you.
- Change, recompute or comment on any count, ratio or percentage.
- Add, remove, split or merge concepts.

TIERS
critical  Its absence changes the job's grade, legal standing or ability
          to operate: licences, regulatory/compliance duties, mandatory
          certifications, budget or people accountability, core scope.
high      Materially affects how the role performs: a core technical
          skill, a core deliverable, a defining domain of experience.
standard  Normal expected content: common tools, ordinary soft skills,
          routine supporting duties.
low       Cosmetic or administrative phrasing with little practical
          consequence.

HOW TO DECIDE
Ask: if this were absent, would we hire a different person, at a
different grade, or expose the organisation to risk?
  Yes, on grade or risk    -> critical
  Yes, on capability       -> high
  No, but it is expected   -> standard
  No                       -> low

Prevalence is given to you only as context. A concept present in a
MINORITY of documents can still be critical, and frequently is — a
regulatory duty that only two benchmark documents spell out is still a
regulatory duty. Judge the substance, never the frequency.

BUSINESS IMPACT
One sentence, at most 25 words, concrete and specific to this concept.
No hedging, no restating the label, no generic filler.

OUTPUT
Return JSON only, matching the provided schema. One entry per concept_id
given to you, no more and no fewer.
"""


def build_materiality_user_content(rows: Sequence[dict]) -> str:
    """
    `rows` are plain dicts produced by materiality.py:
        concept_id, category, label, direction, prevalence_text,
        new_job_text, reference_text
    """

    lines = [
        "Rate the materiality of each concept below.",
        f"CONCEPT COUNT: {len(rows)}",
        "",
    ]

    for index, row in enumerate(rows, start=1):
        lines.append(f"--- CONCEPT {index} ---")
        lines.append(f"concept_id: {row['concept_id']}")
        lines.append(f"category: {row['category']}")
        lines.append(f"concept: {row['label']}")
        lines.append(f"situation: {row['direction']}")
        lines.append(f"benchmark prevalence: {row['prevalence_text']}")

        if row.get("new_job_text"):
            lines.append(f"new job wording: {row['new_job_text']}")

        if row.get("reference_text"):
            lines.append(
                f"benchmark wording: {row['reference_text']}"
            )

        lines.append("")

    lines.append(
        "Return one verdict per concept_id, in the same order."
    )

    return "\n".join(lines)


__all__ = [
    "CLUSTERING_SYSTEM_INSTRUCTION",
    "MATERIALITY_SYSTEM_INSTRUCTION",
    "build_clustering_user_content",
    "build_materiality_user_content",
]
