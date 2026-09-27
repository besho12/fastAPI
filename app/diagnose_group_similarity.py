"""
diagnose_group_similarity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.embedding import EmbeddingService  # noqa: E402


def cosine_similarity(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)

TITLE_PAIRS = [
    # -- Should be SAME group (paraphrases / seniority variants) --
    ("Backend Developer", "Senior Backend Engineer", "same"),
    ("Backend Developer", "Backend Engineer", "same"),
    ("Data Analyst", "Data Analytics Specialist", "same"),
    ("مطور", "Software Developer", "same"),

    # -- Should be DIFFERENT groups (different roles) --
    ("Backend Developer", "Frontend Developer", "different"),
    ("Backend Developer", "Data Analyst", "different"),
    ("Frontend Developer", "Data Analyst", "different"),
    ("HR Specialist", "Marketing Specialist", "different"),
    ("مطور", "مطور واجهات أمامية", "different"),
    ("Financial Analyst", "Data Analyst", "different"),
    ("Senior Python Backend Engineer", "Frontend Developer", "different"),
]


def main() -> None:

    print("Loading BGE-M3 (this may take a moment)...")
    service = EmbeddingService()
    service.load()

    print()
    print("=" * 90)
    print("TITLE-ONLY SIMILARITY DIAGNOSTIC")
    print("=" * 90)
    print()

    # Cache embeddings so identical titles are only embedded once.
    cache = {}

    def embed(title: str):
        if title not in cache:
            cache[title] = service.embed_text(title)
        return cache[title]

    results = []

    for title_a, title_b, expected in TITLE_PAIRS:
        vec_a = embed(title_a)
        vec_b = embed(title_b)
        score = cosine_similarity(vec_a, vec_b)
        results.append((title_a, title_b, expected, score))

    for title_a, title_b, expected, score in sorted(
        results, key=lambda r: r[3], reverse=True
    ):
        flag = ""
        print(
            f"{score:.4f}  [{expected:>9}]  "
            f"{title_a!r:45} vs {title_b!r}"
        )

    same_scores = [r[3] for r in results if r[2] == "same"]
    different_scores = [r[3] for r in results if r[2] == "different"]

    print()
    print("=" * 90)
    print("THRESHOLD ANALYSIS")
    print("=" * 90)

    if same_scores:
        min_same = min(same_scores)
        print(f"Lowest 'same'-pair score:       {min_same:.4f}")
    else:
        min_same = None
        print("No 'same' pairs provided.")

    if different_scores:
        max_different = max(different_scores)
        print(f"Highest 'different'-pair score: {max_different:.4f}")
    else:
        max_different = None
        print("No 'different' pairs provided.")

    print()

    if min_same is not None and max_different is not None:

        if min_same > max_different:
            suggested = (min_same + max_different) / 2
            print(
                "Clean separation exists. Suggested "
                f"JOB_IDENTITY_TITLE_THRESHOLD ≈ {suggested:.3f}"
            )
            print(
                "(strictly between the highest 'different' score "
                "and the lowest 'same' score)"
            )
        else:
            print(
                "WARNING: no threshold cleanly separates 'same' from "
                "'different' pairs for this title set. At least one "
                "'different' pair scores >= at least one 'same' pair.\n"
                "A pure title-embedding threshold cannot fully solve "
                "this on its own -- consider combining the title gate "
                "with a Department-level check, or reviewing whether "
                "BGE-M3 is the right model for short-text "
                "discrimination in your title style."
            )

    print()
    print(
        "Set the threshold in your .env file:\n"
        "  JOB_IDENTITY_TITLE_THRESHOLD=<value>"
    )


if __name__ == "__main__":
    main()