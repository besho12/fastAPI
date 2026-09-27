"""
    python diagnose_arabic_numbers.py path/to/job_11.pdf
"""
import re
import sys

from pypdf import PdfReader

ARABIC_NEAR_DIGIT = re.compile(
    r"[\u0600-\u06FF][^\n]{0,15}[+\-]?\d+[^\n]{0,15}"
    r"|[+\-]?\d+[^\n]{0,15}[\u0600-\u06FF]"
)


def main(path: str) -> None:
    reader = PdfReader(path)
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        for line in text.split("\n"):
            if ARABIC_NEAR_DIGIT.search(line):
                print(f"[page {i}] RAW: {line!r}")


if __name__ == "__main__":
    main(sys.argv[1])