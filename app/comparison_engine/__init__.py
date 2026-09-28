"""
app/comparison_engine

A four-stage job-description benchmark engine.

NAMING NOTE: this package is named `comparison_engine`, not `comparison`,
because `app/comparison.py` already exists in this project as a separate
(currently unused) module. Keeping both names distinct avoids a module
vs. package collision on `app.comparison`.

    from app.comparison_engine import ComparisonEngine, GeminiGateway
    from app.comparison_engine import generate_comparison_excel

    engine = ComparisonEngine(gateway=GeminiGateway())
    result = engine.run(new_job, reference_jobs)
    workbook_bytes = generate_comparison_excel(result, new_job, reference_jobs)

Design principles, in one place:

  1. The model clusters meaning and rates importance. It never counts,
     never classifies, never decides what appears in the report.
  2. Every number is computed in Python from grounded evidence, so no
     number can ever contradict the evidence behind it.
  3. Nothing is filtered away. Prevalence and materiality decide ORDER
     and PROMINENCE, never visibility. The important-but-uncommon
     requirement that a 50% cutoff used to delete is now a Notable Gap
     near the top of the report.
  4. Degrade, never crash. Any model failure falls back to deterministic
     behaviour and is disclosed on the audit sheet.
"""

from app.comparison_engine.engine import ComparisonEngine, compare
from app.comparison_engine.excel import generate_comparison_excel
from app.comparison_engine.group import (
    GroupComparisonEngine,
    JobGroupComparison,
)
from app.comparison_engine.group_excel import generate_group_summary_excel
from app.comparison_engine.gateway import LLMUnavailable
from app.comparison_engine.gemini import GeminiGateway
from app.openai_gateway import OpenAIGateway
from app.comparison_engine.models import (
    BUCKET_LABEL,
    BenchmarkStrength,
    Bucket,
    Category,
    ComparisonResult,
    ComparisonStatusV2,
    Direction,
    Finding,
    Tier,
)

__all__ = [
    "BUCKET_LABEL",
    "BenchmarkStrength",
    "Bucket",
    "Category",
    "ComparisonEngine",
    "ComparisonResult",
    "ComparisonStatusV2",
    "Direction",
    "Finding",
    "GeminiGateway",
    "GroupComparisonEngine",
    "JobGroupComparison",
    "LLMUnavailable",
    "OpenAIGateway",
    "Tier",
    "compare",
    "generate_comparison_excel",
    "generate_group_summary_excel",
]
