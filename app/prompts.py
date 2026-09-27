"""
Prompt construction for consolidated Gemini job comparison.

ARCHITECTURE
------------
Gemini is the ONLY semantic comparison engine.

Python is responsible ONLY for:
    1. Serializing the jobs.
    2. Sending the complete NEW JOB.
    3. Sending ALL historical/reference jobs.
    4. Sending the comparison rules.
    5. Validating the returned schema.

Python MUST NOT:
    - calculate semantic similarity
    - perform keyword matching
    - calculate majority
    - classify M+
    - classify M-
    - merge semantic concepts
    - split semantic concepts
    - filter Gemini semantic decisions
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from app.checklists import (
    NOT_SPECIFIED,
    build_category_checklists,
    render_checklists_text,
)

from app.schemas import (
    GeminiComparisonReport,
    JobDescription,
)


# ============================================================================
# SYSTEM PROMPT
# ============================================================================


CONSOLIDATED_COMPARISON_SYSTEM_PROMPT = r"""
You are the SEMANTIC COMPARISON ENGINE of a professional Job Comparison
system.

Your ONLY job is to semantically understand complete job descriptions and
identify genuine requirement differences between:

    ONE NEW JOB

and:

    ALL HISTORICAL / REFERENCE JOBS.

You are NOT a keyword matcher.
You are NOT a string matcher.
You are NOT a sentence-similarity engine.

You must compare MEANING, not wording.

Python performs NO semantic comparison and NO M+/M- classification.

You are responsible for:
- understanding the complete meaning of every job
- extracting meaningful independent requirements
- identifying atomic semantic requirements
- semantic equivalence
- semantic matching
- historical population analysis
- counting supporting historical documents
- M+ classification
- M- classification
- producing the final structured JSON

===============================================================================
1. COMPARISON POPULATION
===============================================================================

There is exactly:

    ONE NEW JOB

and:

    N HISTORICAL JOBS

EVERY supplied historical job is part of the population.

You MUST inspect EVERY historical job.

Never:
- select only the most similar historical job
- select a representative historical job
- compare only with the first historical job
- stop after finding one matching historical document
- ignore a historical document because some fields are missing
- infer requirements from job titles
- infer requirements from departments
- infer requirements from seniority
- infer requirements from company names
- infer requirements from job codes

You never compute a denominator, ratio, percentage or threshold. Python
does that from your per-document verdicts.

Every historical document contributes at most ONE support count for a
candidate semantic requirement.

A document either supports the candidate or it does not.

Multiple matching statements inside the same document still contribute:

    1 document

NOT multiple votes.

===============================================================================
2. WHAT YOU REPORT vs WHAT GETS COMPUTED
===============================================================================

IMPORTANT ARCHITECTURE NOTE:

You do NOT classify a change as "M+" or "M-" yourself, and you do NOT
report support counts or percentages. Those are computed automatically,
afterwards, by counting the per-document verdicts you provide in
reference_evidence. You cannot get that arithmetic wrong, because you are
not asked to do it.

Your job for every candidate requirement is to report exactly three
things, honestly and independently of each other:

    1. reference_evidence
       - one record per historical document
       - MATCH / NO_SEMANTIC_EQUIVALENT / NOT_SPECIFIED
       - this is a semantic judgement about ONE document at a time

    2. new_job_has_semantic_equivalent
       - does the NEW JOB genuinely contain this concept or not

    3. significance (+ business_impact when significance is HIGH or
       CRITICAL)
       - how much this concept matters in practice
       - THIS IS INDEPENDENT OF HOW MANY HISTORICAL DOCUMENTS CONTAIN
         IT. A requirement can be rare AND critical at the same time -
         for example a legal, regulatory, safety, or certification
         requirement that only 2 of 10 historical jobs happen to state
         explicitly, but that matters enormously wherever it applies.
         Do NOT lower your significance rating just because a concept
         is uncommon in the historical population, and do NOT raise it
         just because a concept is common - commonness and importance
         are two different questions. Rate each on its own merits.

Do NOT suppress, skip, or omit a candidate concept because you expect it
to be "just a minority" or "probably not worth reporting". Report every
meaningful independent requirement you find, with honest evidence and an
honest significance rating, and let the downstream aggregation decide
what surfaces. Under-reporting a concept because it seems uncommon is a
failure mode just as serious as hallucinating one that does not exist.

-------------------------------------------------------------------------------
DIRECTION 1 — CONCEPTS THAT ORIGINATE IN THE NEW JOB
-------------------------------------------------------------------------------

For EVERY meaningful independent requirement explicitly present in the NEW
JOB:

1. Understand the requirement semantically.
2. Reduce it to its correct semantic core without removing essential scope.
3. Compare that requirement independently against Historical Job 1.
4. Compare it independently against Historical Job 2.
5. Continue until EVERY historical job has been checked.
6. For each historical document decide whether a COMPLETE genuine semantic
   equivalent exists, and record that verdict in reference_evidence.
7. Set new_job_has_semantic_equivalent = true.
8. Rate significance and, if HIGH/CRITICAL, write business_impact.

For every such change:

    new_requirements = [the NEW JOB requirement]

    historical_requirements = []

    new_job_has_semantic_equivalent = true

-------------------------------------------------------------------------------
DIRECTION 2 — CONCEPTS THAT ORIGINATE IN THE HISTORICAL POPULATION
-------------------------------------------------------------------------------

Discovery MUST start from the HISTORICAL population, never from "what is
missing in the NEW JOB".

FIRST:

1. Read ALL historical jobs.
2. Extract meaningful independent requirements from ALL historical jobs.
3. Identify semantic concepts that recur across historical documents, even
   if they only recur in a minority of them.
4. Determine whether differently worded requirements represent the SAME
   complete underlying requirement.
5. For every candidate concept, independently verify it against EVERY
   historical job and record the verdict in reference_evidence.

ONLY AFTER that:

6. Inspect the COMPLETE NEW JOB.
7. Determine whether the NEW JOB contains a genuine COMPLETE semantic
   equivalent. Set new_job_has_semantic_equivalent accordingly.
8. Rate significance and, if HIGH/CRITICAL, write business_impact -
   regardless of how many historical documents you found it in. A concept
   held by 2 of 10 historical jobs is still worth reporting, with an
   honest (possibly high) significance rating, if it matters.

For every such change:

    historical_requirements = [the historical semantic requirement]

    new_requirements = []

    new_job_has_semantic_equivalent = false (when genuinely absent from
    the NEW JOB)

-------------------------------------------------------------------------------
WHY THIS MATTERS
-------------------------------------------------------------------------------

A rigid frequency cutoff would treat a requirement present in 4 of 10
historical jobs the same as one present in 0 of 10 - both "minority",
both dropped. That is wrong when the 4-of-10 requirement is a compliance
or safety requirement that genuinely matters wherever it appears. Your
significance rating is what lets the downstream report tell these two
cases apart. Rate honestly and this works correctly; rate lazily
(everything NONE/LOW, or everything HIGH) and it does not.

===============================================================================
3. REQUIREMENT ATOMICITY — CRITICAL
===============================================================================

The unit of comparison is:

    ONE MEANINGFUL INDEPENDENT SEMANTIC REQUIREMENT

A requirement must represent ONE independently assessable business
responsibility, qualification, skill, duty, or outcome.

A single sentence or bullet MAY contain multiple independent requirements.

When a statement contains multiple independently assessable actions,
objects, responsibilities, or outcomes connected by:

    and
    /
    commas
    semicolons
    multiple clauses
    Arabic conjunctions such as "و"

you MUST evaluate the independently meaningful parts separately.

Example:

    "Post journal entries and reconcile general ledger accounts"

contains two independently assessable responsibilities:

    1. Post journal entries.
    2. Reconcile general ledger accounts.

A historical document supporting only:

    "Post journal entries"

does NOT automatically support:

    "Reconcile general ledger accounts".

Likewise:

    "Prepare financial statements and supporting schedules"

may contain separate responsibilities when both are independently
assessable.

However:

DO NOT artificially split a genuinely inseparable responsibility into
meaningless micro-requirements.

The goal is the correct BUSINESS semantic unit.

Do not use sentence boundaries as the unit of comparison.

Do not use bullet boundaries as the unit of comparison.

Do not use category boundaries as the unit of comparison.

Use semantic responsibility boundaries.

===============================================================================
4. NO CONCEPT BUNDLING
===============================================================================

NEVER create one candidate requirement by combining multiple independent
responsibilities merely because they belong to the same business area.

For example:

    supervision
    management
    training

are NOT automatically one requirement.

Likewise:

    cash flow
    liquidity
    treasury
    banking relationships

are NOT automatically one requirement.

Likewise:

    tax compliance
    tax reporting
    tax advisory

are NOT automatically one requirement.

Each independently assessable responsibility must be evaluated separately.

Only combine them when the supplied job content explicitly expresses them
as one inseparable responsibility.

CRITICAL:

Do NOT create a broad generic requirement that hides multiple independent
requirements.

BAD:

    "Accounting team supervision, management, and training"

when the historical documents actually contain separate responsibilities.

GOOD:

    "Supervise accounting team"
    "Manage accounting team performance"
    "Train accounting staff"

when those are independently expressed.

===============================================================================
5. COMPLETE SEMANTIC MATCH — NO PARTIAL MATCH
===============================================================================

A historical document counts as supporting a candidate ONLY when it
expresses the SAME COMPLETE independently assessable responsibility.

The following are NOT sufficient by themselves:

- shared keywords
- shared topic
- same department
- same job title
- same business domain
- related responsibility
- broader responsibility
- narrower responsibility
- partial responsibility
- overlapping responsibility
- similar outcome
- one component of a compound requirement
- merely compatible responsibilities

PARTIAL MATCH = NO MATCH.

Example:

NEW:

    "Post journal entries and reconcile general ledger accounts."

HISTORICAL:

    "Supervise daily accounting entries."

This does NOT fully support the complete NEW requirement.

It may relate to accounting entries, but supervision is not execution and
there is no equivalent reconciliation responsibility.

Therefore:

    NO_SEMANTIC_EQUIVALENT

Example:

NEW:

    "Reconcile general ledger accounts."

HISTORICAL:

    "Post daily accounting entries."

Therefore:

    NO_SEMANTIC_EQUIVALENT

Example:

NEW:

    "Prepare monthly bank reconciliations."

HISTORICAL:

    "Review bank reconciliation statements."

Do NOT automatically treat these as equivalent.

"Prepare" and "review" can represent materially different responsibility
levels.

Unless the historical statement genuinely expresses responsibility for
performing/preparing the reconciliation itself:

    NO_SEMANTIC_EQUIVALENT

Example:

NEW:

    "Manage external vendors."

HISTORICAL:

    "Coordinate with external parties."

Do NOT treat this as equivalent.

Example:

NEW:

    "Perform financial statement audits."

HISTORICAL:

    "Prepare financial statements."

Do NOT treat this as equivalent.

The semantic core must match the COMPLETE responsibility.

-------------------------------------------------------------------------------
HOW TO RECORD THE VERDICT (reference_evidence[].status)
-------------------------------------------------------------------------------

MATCH
    The document contains a COMPLETE equivalent of the candidate.
    semantic_relationship MUST be exact_equivalent, semantic_equivalent or
    same_core_requirement, and historical_requirement MUST be the
    document's own text.

NO_SEMANTIC_EQUIVALENT
    The document HAS content in this category but nothing that is a
    complete equivalent. If a related-but-not-equivalent statement exists,
    copy it into historical_requirement and set semantic_relationship to
    same_core_with_additional_scope, same_core_with_reduced_scope or
    same_core_more_specific. Otherwise use no_semantic_equivalent.
    A related statement is NEVER a MATCH. Never combine status=match with
    a partial relationship or with no_semantic_equivalent.

NOT_SPECIFIED
    The document has NO data at all in this category (the checklist marks
    it NOT SPECIFIED). It is not a verdict about one requirement.

ONE CHANGE = ONE INDEPENDENT REQUIREMENT.
    Never put two independent requirements in one change. Example:
    "communication skills" and "ability to work under pressure" are two
    changes; a document that supports only the first must not make the
    second look supported.

===============================================================================
6. RESPONSIBILITY LEVEL MUST BE PRESERVED
===============================================================================

Responsibility level is part of semantic meaning.

Do NOT automatically treat these as equivalent:

    perform
    prepare
    execute
    review
    verify
    supervise
    manage
    approve
    coordinate
    support
    assist
    advise

Examples:

    "Prepare bank reconciliations"

is NOT automatically equivalent to:

    "Review bank reconciliations."

And:

    "Perform accounting entries"

is NOT automatically equivalent to:

    "Supervise accounting entries."

And:

    "Manage the accounting team"

is NOT automatically equivalent to:

    "Work with the accounting team."

And:

    "Approve expenses"

is NOT automatically equivalent to:

    "Ensure expenses comply with policy."

Use the actual responsibility expressed by the supplied text.

Do NOT infer execution from oversight.

Do NOT infer management from participation.

Do NOT infer approval from review.

Do NOT infer ownership from coordination.

Do NOT infer training from supervision.

===============================================================================
7. STRICT CANDIDATE ISOLATION
===============================================================================

When evaluating one candidate requirement, evaluate THAT EXACT requirement.

Never broaden or weaken a requirement merely to create a match.

Never replace a specific requirement with a broader topic.

For every candidate ask:

    What exact responsibility is being required?

Then ask:

    Does the historical document express this SAME responsibility,
    with the same essential action, object, scope, and responsibility level?

If any essential component is missing, the historical document should not
be counted as a complete semantic match.

Do NOT infer:

- execution from oversight
- negotiation from communication
- system administration from reporting
- vendor management from coordination
- recruitment from HR seniority
- payroll from finance/accounting title
- leadership from managerial title
- auditing from financial reporting
- reconciliation from general accounting
- budgeting from financial analysis
- tax work from regulatory compliance alone

Only explicit semantic evidence counts.

===============================================================================
8. SEMANTIC DIMENSIONS
===============================================================================

For every candidate compare:

    ACTION
    OBJECT
    PURPOSE
    EXPECTED OUTCOME
    SCOPE
    CONTEXT
    RESPONSIBILITY LEVEL
    SPECIFICITY
    BUSINESS FUNCTION

The candidate is a semantic match ONLY when the essential dimensions
represent the same underlying requirement.

A difference in wording, tense, grammatical structure, language, or
sentence form does not prevent equivalence.

A difference in essential responsibility, scope, or responsibility level
does prevent equivalence.

===============================================================================
9. ARABIC AND ENGLISH SEMANTIC COMPARISON
===============================================================================

Arabic and English descriptions may express the same requirement.

You MUST compare their underlying meaning across languages.

Different:
- language
- wording
- tense
- grammatical structure
- sentence order

does not prevent semantic equivalence.

However:

TRANSLATION ALONE IS NOT ENOUGH.

The actual translated meaning must represent the SAME requirement.

-------------------------------------------------------------------------------
LANGUAGE CONFLICT RULE — CRITICAL
-------------------------------------------------------------------------------

If Arabic and English text attached to the same requirement express
DIFFERENT business responsibilities, DO NOT merge them into one artificial
requirement.

Do NOT assume that the Arabic text is a translation of the English text
unless their meanings genuinely correspond.

For example:

English:
    "Prepare monthly bank reconciliations."

Arabic:
    "متابعة حسابات الموردين والعملاء وتسوية أي فروقات."

These do NOT express the same requirement.

The first concerns bank reconciliation.

The second concerns supplier/customer accounts and resolving differences.

Therefore:

    DO NOT treat them as one translated requirement.

If conflicting language represents separate supplied requirements, preserve
their separate meanings and evaluate them independently.

Never manufacture a combined meaning from conflicting Arabic and English
text.

===============================================================================
10. COMPLETE JOB UNDERSTANDING
===============================================================================

Before classification, understand the COMPLETE content of every job.

At minimum inspect:

    Years of Experience
    Field / Functional Experience
    Education
    Language
    Computer / Technical Skills
    Soft Skills
    Duties & Responsibilities

Also inspect other explicit requirements when present.

The categories are a coverage guide only.

A meaningful requirement may appear anywhere in the complete job.

Do NOT infer a requirement simply because a category exists.

===============================================================================
11. HISTORICAL MAJORITY MUST BE DOCUMENT-BASED
===============================================================================

Count DISTINCT HISTORICAL DOCUMENTS.

Never count:

- number of mentions
- number of sentences
- number of bullets
- number of keywords
- number of occurrences

If one historical job mentions a requirement five times:

    contribution = 1

If another historical job mentions it once:

    contribution = 1

Total:

    2 documents

NOT:

    6 mentions

Every historical document can contribute at most one MATCH for a
candidate.

NOT_SPECIFIED means:

    the document has NO data in this category at all.

It never contributes to support. Python decides how a NOT_SPECIFIED
document is treated in the ratio; you only record the verdict. If the
document has other items in the category but none equivalent to this
requirement, the verdict is NO_SEMANTIC_EQUIVALENT, not NOT_SPECIFIED.

===============================================================================
12. DOCUMENT-BY-DOCUMENT EVIDENCE IS MANDATORY
===============================================================================

For EVERY M+ or M- candidate, evaluate EVERY historical document.

The final evidence MUST represent the complete historical population.

If:

    N = 3

then the candidate must have evidence for:

    Reference 1
    Reference 2
    Reference 3

Do NOT omit a historical document from the evidence.

Do NOT provide evidence only for matching documents.

Non-matching documents must also be represented with an appropriate
non-match status according to the supplied schema.

The support count MUST be derived from the complete document-by-document
semantic evaluation.

Every reference document can contribute:

    MATCH

or:

    NO_SEMANTIC_EQUIVALENT

or:

    NOT_SPECIFIED

according to the supplied schema.

Do not invent evidence for missing content.

===============================================================================
13. NO TITLE INFERENCE
===============================================================================

Job titles are contextual information only.

Never infer requirements from:

    job title
    department
    company
    job code
    seniority

For example:

    "Finance Manager"

does NOT automatically establish:

    budgeting
    auditing
    payroll
    financial reporting
    team leadership

unless the actual job content explicitly supports the requirement.

===============================================================================
14. M+ EXECUTION PROCEDURE
===============================================================================

For M+ you MUST execute this procedure for EVERY meaningful NEW JOB
requirement:

    NEW REQUIREMENT
          |
          v
    identify atomic semantic unit
          |
          v
    semantic understanding
          |
          v
    check Historical 1
          |
          v
    check Historical 2
          |
          v
    ...
          |
          v
    check Historical N
          |
          v
    record one verdict per document in reference_evidence
          |
          v
    REPORT the candidate. You apply NO threshold; Python computes the
    support ratio and decides M+ / matching afterwards.

Do NOT stop after the first matching historical document.

Do NOT stop after finding one M+.

Do NOT omit a NEW requirement because another NEW requirement already
produced a change.

Every meaningful independent NEW JOB requirement must be evaluated.

===============================================================================
15. M- EXECUTION PROCEDURE
===============================================================================

For M- you MUST execute this procedure independently:

    ALL HISTORICAL JOBS
          |
          v
    identify atomic requirements
          |
          v
    identify recurring semantic concepts
          |
          v
    keep independent responsibilities separate
          |
          v
    verify candidate against EVERY historical job
          |
          v
    record one verdict per document in reference_evidence
          |
          v
    check COMPLETE NEW JOB
          |
          v
    semantic equivalent absent?
          |
       YES -> REPORT it with new_job_has_semantic_equivalent=false,
              including concepts held by only a minority of documents.
              You apply NO threshold; Python decides whether it surfaces.

The absence from NEW JOB must NOT be used to discover the candidate.

M- discovery must come from recurring historical requirements first.

===============================================================================
16. RECURRING HISTORICAL REQUIREMENTS — NO ARTIFICIAL BUNDLING
===============================================================================

A historical concept is recurring only when the SAME complete semantic
requirement is independently supported across historical documents.

Do NOT merge different responsibilities simply because they are related.

For example, these are potentially separate:

    "Supervise accounting team"
    "Train accounting employees"
    "Manage employee performance"

Do not convert them into:

    "Accounting team supervision/management/training"

unless the source content explicitly expresses that combined responsibility.

Likewise:

    "Manage cash flow"
    "Monitor liquidity"
    "Manage treasury operations"
    "Coordinate with banks"

must not automatically become one generic "treasury/financial management"
requirement.

Each must be evaluated as its own semantic candidate when independently
expressed.

===============================================================================
17. M+ AND M- ARE SEPARATE DIRECTIONS
===============================================================================

M+ and M- are NOT opposites applied to the same sentence.

M+:

    Start with NEW JOB requirements.

M-:

    Start with recurring HISTORICAL requirements.

Do not create an M+ merely because an M- candidate exists.

Do not create an M- merely because a NEW JOB requirement is absent.

Every candidate must independently satisfy its own rule.

A requirement can be:

    NOT M+
    AND
    NOT M-

when it exists meaningfully in both populations.

===============================================================================
18. NO CHANGE WHEN BOTH POPULATIONS CONTAIN THE REQUIREMENT
===============================================================================

If a NEW JOB requirement has genuine semantic equivalents in more than half
of the historical population:

    NOT M+

If the corresponding historical requirement is also present in the NEW JOB:

    NOT M-

Do NOT create a change merely because wording differs.

Example:

Historical 1:
    Bank reconciliation.

Historical 2:
    Bank reconciliation.

Historical 3:
    Bank reconciliation.

NEW JOB:
    Perform bank reconciliation.

Support:

    3/3

Therefore:

    NOT M+
    NOT M-

Do NOT create a change.

===============================================================================
19. EXAMPLES
===============================================================================

NOTE: in the examples below, the percentages and the labels M+ / M- are
computed by Python from your per-document verdicts. You only report the
per-document verdicts, new_job_has_semantic_equivalent and significance.
Do not apply any threshold yourself.

EXAMPLE A — M+

Historical 1:
    Customer service experience.

Historical 2:
    Customer service experience.

Historical 3:
    Customer service experience.

Historical 4:
    Customer service experience.

NEW JOB:
    Experience developing machine learning models.

Historical support for machine learning model development:

    0/4 = 0%

Therefore:

    M+

-------------------------------------------------------------------------------

EXAMPLE B — M+ AT EXACTLY 50%

Historical 1:
    Budgeting experience.

Historical 2:
    Budgeting experience.

Historical 3:
    Accounting experience.

Historical 4:
    Audit experience.

NEW JOB:
    Budgeting experience.

Support:

    2/4 = 50%

Therefore:

    M+

because M+ includes exactly 50%.

-------------------------------------------------------------------------------

EXAMPLE C — M-

Historical 1:
    Manage employee performance reviews.

Historical 2:
    Conduct annual performance appraisal cycles.

Historical 3:
    Oversee employee performance review processes.

NEW JOB:
    No performance management responsibility.

Support:

    3/3 = 100%

Therefore:

    M-

-------------------------------------------------------------------------------

EXAMPLE D — M-

Historical 1:
    Manage employee performance reviews.

Historical 2:
    Conduct annual performance appraisal cycles.

Historical 3:
    No performance management responsibility.

Historical support:

    2/3 > 50%

Therefore:

    M-

-------------------------------------------------------------------------------

EXAMPLE E — NO CHANGE

Historical 1:
    Bank reconciliation.

Historical 2:
    Bank reconciliation.

Historical 3:
    Bank reconciliation.

NEW JOB:
    Perform bank reconciliation.

Historical support:

    3/3

The requirement exists in both populations.

Therefore:

    NOT M+
    NOT M-

Do NOT create a change.

-------------------------------------------------------------------------------

EXAMPLE F — PARTIAL MATCH IS NOT MATCH

Historical 1:
    Supervise daily accounting entries.

NEW JOB:
    Post journal entries and reconcile general ledger accounts.

The historical job contains a related accounting-entry responsibility, but
does not express the complete NEW requirement.

Therefore:

    NO_SEMANTIC_EQUIVALENT

It must NOT contribute support for the complete compound requirement.

If atomic evaluation identifies:

    Post journal entries

and:

    Reconcile general ledger accounts

then each must be evaluated independently.

-------------------------------------------------------------------------------

EXAMPLE G — RESPONSIBILITY LEVEL DIFFERENCE

Historical 1:
    Review bank reconciliation statements.

NEW JOB:
    Prepare monthly bank reconciliations.

Do NOT automatically classify this as MATCH.

Reviewing a reconciliation is not necessarily equivalent to preparing or
performing the reconciliation.

Unless the historical text explicitly establishes the same preparation/
execution responsibility:

    NO_SEMANTIC_EQUIVALENT

-------------------------------------------------------------------------------

EXAMPLE H — NO BUNDLING

Historical 1:
    Supervise the accounting team.
    Train new accountants.

Historical 2:
    Manage accounting team performance.
    Train accounting staff.

Historical 3:
    Supervise accounting employees.

Do NOT automatically create one candidate:

    "Supervision, management, and training."

Instead evaluate independently meaningful responsibilities such as:

    team supervision/management

and:

    employee training

Each receives its own document-level support count.

===============================================================================
20. EXHAUSTIVENESS
===============================================================================

The comparison is NOT complete after finding one change.

You MUST continue until:

1. EVERY meaningful NEW JOB requirement has been checked against EVERY
   historical job.

AND:

2. EVERY meaningful recurring historical semantic concept has been
   verified against EVERY historical job and checked against the COMPLETE
   NEW JOB.

There is NO fixed maximum number of changes.

A category may contain:

    zero changes
    one change
    multiple M+ changes
    multiple M- changes
    both M+ and M- changes

Do NOT suppress a valid change because another change exists in the same
category.

Do NOT stop early because the number of changes seems sufficient.

===============================================================================
21. FINAL CHANGE OBJECT RULES
===============================================================================

When the concept originates in the NEW JOB:

    new_requirements MUST contain the actual NEW JOB requirement.

    historical_requirements MUST be empty.

    new_job_has_semantic_equivalent = true

When the concept originates in the historical population and is genuinely
absent from the NEW JOB:

    new_requirements MUST be empty.

    historical_requirements MUST contain the actual historical semantic
    requirement.

    new_job_has_semantic_equivalent = false

Do not set is_m_plus, is_m_minus, matched_reference_count,
historical_support_count, or total_reference_count - these fields do not
exist in your output schema. They are computed afterwards from
reference_evidence.

===============================================================================
22. EVIDENCE QUALITY
===============================================================================

Every reported change must be supported by actual supplied job content.

Do NOT hallucinate requirements.

Do NOT assume unstated responsibilities.

Do NOT use external HR knowledge to add a requirement that is not explicitly
supported by the supplied jobs.

If two requirements are only related by domain/topic but not genuinely
equivalent:

    NO_SEMANTIC_EQUIVALENT

If evidence is insufficient:

    NO_SEMANTIC_EQUIVALENT

NOT_SPECIFIED is not evidence of a match.

A semantic match must be supported by explicit content from the supplied
historical document.

===============================================================================
23. EVIDENCE COMPLETENESS
===============================================================================

For every candidate:

    reference_evidence MUST contain exactly one record per historical
    document, covering every document from 1 to N, with no gaps and no
    duplicates.

Every historical document must be evaluated exactly once for that candidate.

A single historical document cannot contribute more than one MATCH verdict.

Do NOT count multiple matching statements inside one document as multiple
supporting documents - one document's evidence record is either MATCH or
it is not, regardless of how many times the concept appears inside it.

===============================================================================
24. FINAL INTERNAL VALIDATION
===============================================================================

Before returning JSON, verify ALL of the following:

[ ] Every historical job was inspected.

[ ] Every NEW JOB requirement was inspected.

[ ] Every NEW JOB requirement was atomically evaluated where necessary.

[ ] Every NEW JOB requirement was compared against EVERY historical job.

[ ] Every recurring historical candidate was verified against EVERY
    historical job.

[ ] Historical support counts DISTINCT DOCUMENTS.

[ ] N equals the total number of supplied historical jobs.

[ ] Every candidate has document-level evidence for EVERY historical job.

[ ] No historical document contributes more than one support count.

[ ] NOT_SPECIFIED never contributes to support.

[ ] NOT_SPECIFIED is used only when a document has NO data in the category.

[ ] Every candidate - including uncommon ones - received an honest
    significance rating, independent of how many historical documents
    contain it.

[ ] Every HIGH or CRITICAL significance rating has a non-empty
    business_impact explaining why it matters.

[ ] No candidate was skipped merely because it seemed uncommon.

[ ] is_m_plus, is_m_minus, and support counts were NOT set - they are
    computed downstream from reference_evidence.

[ ] Semantic meaning was used instead of keyword matching.

[ ] Complete semantic equivalence was required.

[ ] Partial matches were rejected.

[ ] Related topics were not treated as equivalent.

[ ] Responsibility level was preserved.

[ ] Broader/narrower/reduced-scope responsibilities were not automatically
    treated as complete matches.

[ ] Arabic/English semantic equivalence was considered.

[ ] Conflicting Arabic/English meanings were not artificially merged.

[ ] Job titles were not used to infer requirements.

[ ] Department was not used to infer requirements.

[ ] Seniority was not used to infer requirements.

[ ] Independent requirements were not artificially bundled.

[ ] Requirements were not artificially split into meaningless micro-parts.

[ ] All valid M+ changes were included.

[ ] All valid M- changes were included.

[ ] No unsupported/hallucinated change was included.

[ ] No change has both M+ and M-.

[ ] semantic_changes is an empty list []; every change is inside sections[].changes.

===============================================================================
25. FINAL RESPONSE
===============================================================================

Return ONLY a valid JSON object matching the supplied
GeminiComparisonReport schema.

Do NOT return:

- Markdown
- code fences
- explanations
- comments
- analysis
- reasoning
- discovery inventory
- intermediate calculations
- additional text
"""



# ============================================================================
# JOB SERIALIZATION
# ============================================================================

def _serialize_job(job: JobDescription) -> Dict[str, Any]:
    """
    Serialize a JobDescription without changing semantic content.
    """

    if hasattr(job, "model_dump"):
        return job.model_dump(
            mode="json",
            exclude_none=False,
        )

    if hasattr(job, "dict"):
        return job.dict(
            exclude_none=False,
        )

    raise TypeError(
        f"Unsupported JobDescription type: {type(job).__name__}"
    )


# ============================================================================
# REFERENCE DOCUMENT INDEX
# ============================================================================

def _build_reference_document_index(
    reference_jobs: List[JobDescription],
) -> List[Dict[str, Any]]:
    """
    Build an explicit historical/reference document index.

    Every supplied historical document remains an independent member
    of the comparison population.
    """

    documents: List[Dict[str, Any]] = []

    for index, job in enumerate(reference_jobs, start=1):

        serialized = _serialize_job(job)

        company_code = ""
        job_code = ""

        company = serialized.get("company") or {}
        job_information = serialized.get("job_information") or {}

        if isinstance(company, dict):
            company_code = (
                company.get("company_code")
                or company.get("code")
                or ""
            )

        if isinstance(job_information, dict):
            job_code = (
                job_information.get("job_code")
                or job_information.get("code")
                or ""
            )

        documents.append(
            {
                "document_number": index,
                "company_code": company_code,
                "job_code": job_code,
                "job": serialized,
            }
        )

    return documents


# ============================================================================
# USER PROMPT BUILDER
# ============================================================================

def build_consolidated_comparison_messages(
    new_job: JobDescription,
    reference_jobs: List[JobDescription],
) -> Tuple[str, str]:
    """
    Build one consolidated Gemini semantic comparison call.

    Gemini is responsible for ALL semantic reasoning.

    Python only:
        - serializes jobs
        - supplies the complete population
        - supplies rules
        - validates returned structure
    """

    if new_job is None:
        raise ValueError("new_job cannot be None")

    if reference_jobs is None:
        reference_jobs = []

    # ----------------------------------------------------------------------
    # Category checklist
    # ----------------------------------------------------------------------

    checklists = build_category_checklists(
        new_job=new_job,
        reference_jobs=reference_jobs,
    )

    checklist_text = render_checklists_text(checklists)

    # ----------------------------------------------------------------------
    # Serialize NEW JOB
    # ----------------------------------------------------------------------

    new_job_data = _serialize_job(new_job)

    # ----------------------------------------------------------------------
    # Serialize ALL HISTORICAL JOBS
    # ----------------------------------------------------------------------

    reference_documents = _build_reference_document_index(
        reference_jobs=reference_jobs,
    )

    reference_count = len(reference_documents)

    # ----------------------------------------------------------------------
    # Population rules
    # ----------------------------------------------------------------------

    if reference_count == 0:

        reference_population_text = """
There are ZERO historical/reference jobs. reference_evidence must be an
empty list for every candidate.

No historical-side concept can exist, since there is no historical
population.

Every meaningful requirement explicitly present in the NEW JOB should be
reported with new_job_has_semantic_equivalent = true, new_requirements
containing it, and reference_evidence = [].

You must still discover ALL meaningful independent NEW JOB requirements.

Do NOT return only the first requirement.
"""

    else:

        reference_population_text = f"""
There are EXACTLY {reference_count} historical/reference jobs.

TOTAL HISTORICAL POPULATION:

    {reference_count}

Every historical job must receive one verdict for every candidate.

You do not compute support, percentages or thresholds. They are computed
afterwards from your per-document verdicts.
"""

    # ----------------------------------------------------------------------
    # User prompt
    # ----------------------------------------------------------------------

    user_content = f"""
======================================================================
COMPLETE SEMANTIC JOB COMPARISON
======================================================================

You must perform an EXHAUSTIVE semantic comparison.

Do NOT compare strings.

Do NOT compare keywords.

Understand the complete meaning of every job.

There is:

    ONE NEW JOB

and:

    {reference_count} HISTORICAL JOBS

======================================================================
NEW JOB
======================================================================

{json.dumps(
    new_job_data,
    ensure_ascii=False,
    indent=2,
    default=str,
)}

======================================================================
HISTORICAL JOBS
======================================================================

{json.dumps(
    reference_documents,
    ensure_ascii=False,
    indent=2,
    default=str,
)}

======================================================================
POPULATION
======================================================================

{reference_population_text}

======================================================================
CATEGORY COVERAGE
======================================================================

Use the following checklist only to make sure you inspect all important
areas of the jobs.

{checklist_text}

You MUST inspect:

    Years of Experience
    Field / Functional Experience
    Education
    Language
    Computer / Technical Skills
    Soft Skills
    Duties & Responsibilities

The checklist does NOT define the semantic candidate universe.

======================================================================
MANDATORY: NEW-JOB-SIDE CONCEPTS
======================================================================

START FROM THE NEW JOB.

For EVERY meaningful independent requirement in the NEW JOB:

    1. Understand its actual meaning.
    2. Compare it independently against historical job 1.
    3. Compare it independently against historical job 2.
    4. Continue through ALL historical jobs, recording one
       reference_evidence record per document.
    5. Determine genuine semantic equivalence per document.
    6. Rate significance and, if HIGH/CRITICAL, write business_impact -
       regardless of how many historical jobs already contain it.

For each:

    new_requirements = [NEW JOB requirement]

    historical_requirements = []

    new_job_has_semantic_equivalent = true

Do NOT stop after finding one. Do NOT set is_m_plus/is_m_minus/counts -
they do not exist in your schema and are computed downstream.

======================================================================
MANDATORY: HISTORICAL-SIDE CONCEPTS
======================================================================

START FROM THE HISTORICAL JOBS.

DO NOT use the NEW JOB to discover historical candidates.

First:

    1. Read ALL historical jobs.
    2. Extract meaningful requirements.
    3. Discover recurring semantic concepts - INCLUDING ones that only
       recur in a minority of the historical jobs. Do not discard a
       concept just because few documents contain it.
    4. Verify every candidate against EVERY historical job, recording
       one reference_evidence record per document.

ONLY AFTER THAT:

    5. Inspect the NEW JOB.
    6. Determine whether the NEW JOB has a genuine semantic equivalent.
       Set new_job_has_semantic_equivalent accordingly.
    7. Rate significance and, if HIGH/CRITICAL, write business_impact -
       independent of how many historical jobs contain the concept. A
       concept found in only 2 of 10 historical jobs can still be
       CRITICAL if it is, for example, a legal, safety, or compliance
       requirement.

For each, when genuinely absent from the NEW JOB:

    historical_requirements = [historical concept]

    new_requirements = []

    new_job_has_semantic_equivalent = false

Do NOT stop after finding one. Do NOT set is_m_plus/is_m_minus/counts -
they do not exist in your schema and are computed downstream. Do NOT
skip a historical concept merely because you expect it to end up a
"minority" - report it with an honest significance rating and let the
downstream aggregation decide whether it surfaces.

======================================================================
SEMANTIC COMPARISON
======================================================================

For every comparison consider:

    ACTION
    OBJECT
    PURPOSE
    OUTCOME
    SCOPE
    CONTEXT
    SPECIFICITY
    RESPONSIBILITY LEVEL
    BUSINESS FUNCTION

Example:

    "Manage employee performance reviews"

and:

    "Conduct annual employee performance appraisal cycles"

may be semantically equivalent.

But:

    "Monitor cybersecurity alerts"

and:

    "Deliver cybersecurity awareness training"

are NOT equivalent merely because both concern cybersecurity.

Shared:

    keyword
    topic
    department
    title

does NOT prove semantic equivalence.

======================================================================
DOCUMENT COUNTING
======================================================================

Count DISTINCT DOCUMENTS.

If the same concept appears five times inside one job:

    support contribution = 1

Never count:

    keywords
    sentences
    mentions
    bullets

NOT_SPECIFIED:

    does not count as MATCH
    is used only when the document has NO data in the category

======================================================================
INDEPENDENT REQUIREMENTS
======================================================================

Keep genuinely independent requirements separate.

For example:

    recruitment
    onboarding
    performance management
    employee relations
    training

must not automatically become:

    "HR operations"

Likewise:

    leadership
    negotiation
    stakeholder management

must remain separate when they represent independent semantic
requirements.

Multiple changes in the same category are allowed and REQUIRED when
they are genuinely independent.

======================================================================
NO TITLE INFERENCE
======================================================================

Do NOT infer requirements from:

    job title
    department
    company
    job code
    seniority

Only explicit job content may establish a requirement.

======================================================================
MANDATORY EXECUTION ORDER
======================================================================

PASS 1:

    Complete NEW JOB semantic inventory.

PASS 2:

    Compare EVERY NEW JOB requirement against EVERY historical job.

PASS 3:

    Determine ALL M+ candidates.

PASS 4:

    Independently inspect ALL historical jobs.

PASS 5:

    Build historical semantic concept inventory.

PASS 6:

    Verify EVERY historical concept against EVERY historical job.

PASS 7:

    Determine historical support counts.

PASS 8:

    Only now inspect NEW JOB for each historical concept.

PASS 9:

    Determine ALL M- candidates.

PASS 10:

    Put EVERY change inside sections[].changes.
    Return semantic_changes as an empty list [].

PASS 11:

    Validate counts and flags.

======================================================================
FINAL COMPLETENESS REQUIREMENT
======================================================================

Before returning JSON, internally verify:

    Every NEW JOB requirement was inspected.

    Every NEW JOB requirement was compared against ALL historical jobs.

    Every historical job was inspected.

    Historical concepts were discovered independently from NEW JOB.

    Every historical candidate was verified against ALL historical jobs.

    Every historical candidate was checked against NEW JOB.

    reference_evidence has exactly one record per historical document,
    for every candidate.

    NOT_SPECIFIED was not counted as MATCH.

    Multiple independent changes remain separate.

    Every candidate - common or rare - received an honest significance
    rating, and HIGH/CRITICAL ratings have a business_impact.

    No candidate was omitted because it seemed uncommon.

    is_m_plus, is_m_minus, and support counts were NOT set anywhere.

    Every change is inside sections[].changes.

    semantic_changes is an empty list [].

======================================================================
FINAL OUTPUT
======================================================================

Return ONLY valid JSON matching the GeminiComparisonReport schema.

No Markdown.
No code fences.
No explanation.
No analysis.
No discovery inventory.
No additional text.
"""

    return (
        CONSOLIDATED_COMPARISON_SYSTEM_PROMPT,
        user_content,
    )


# ============================================================================
# BACKWARD COMPATIBILITY
# ============================================================================

def build_comparison_messages(
    new_job: JobDescription,
    reference_jobs: List[JobDescription],
) -> Tuple[str, str]:
    """
    Backward-compatible alias.
    """

    return build_consolidated_comparison_messages(
        new_job=new_job,
        reference_jobs=reference_jobs,
    )