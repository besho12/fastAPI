"""
preprocessing.py
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Optional, TypedDict, Union

from app.exceptions import InvalidInputError, PreprocessingError
from app.schemas import InputType


# ==========================================================================
# Configuration
# ==========================================================================

# Very short text is unlikely to represent a useful job description.
MIN_TEXT_LENGTH = 10


# ==========================================================================
# Unicode normalization
# ==========================================================================

_UNICODE_REPLACEMENTS = {
    "\u2018": "'",       # left single quotation mark
    "\u2019": "'",       # right single quotation mark
    "\u201c": '"',       # left double quotation mark
    "\u201d": '"',       # right double quotation mark
    "\u2013": "-",       # en dash
    "\u2014": "-",       # em dash
    "\u2212": "-",       # minus sign

    # Bullet characters
    "\u2022": "-",
    "\u25cf": "-",
    "\u25aa": "-",
    "\u25c6": "-",
    "\u2023": "-",
    "\u2219": "-",
    "\uf0b7": "-",

    # Spaces
    "\u00a0": " ",       # non-breaking space
}


# ==========================================================================
# Common PDF extraction artifacts
# ==========================================================================

def fix_common_pdf_artifacts(text: str) -> str:
    """
    Fix common artifacts produced by PDF text extraction.

    Examples:

        C .#       -> C#
        C. #       -> C#
        C . #      -> C#
        C  #       -> C#

        Java Script -> JavaScript
        REST  API   -> REST API

    This function is intentionally conservative and focuses on
    common technical/job-description artifacts.
    """

    # --------------------------------------------------------------
    # Programming language: C#
    # --------------------------------------------------------------
    text = re.sub(
        r"\bC\s*(?:\.\s*)?#",
        "C#",
        text,
        flags=re.IGNORECASE,
    )

    # --------------------------------------------------------------
    # Programming language: C++
    #
    # Examples:
    #   C ++
    #   C  ++
    #   C . ++
    # --------------------------------------------------------------
    text = re.sub(
        r"\bC\s*(?:\.\s*)?\+\s*\+",
        "C++",
        text,
        flags=re.IGNORECASE,
    )

    # --------------------------------------------------------------
    # Common technical terms
    # --------------------------------------------------------------

    # REST API / REST APIs
    text = re.sub(
        r"\bREST\s+APIs?\b",
        lambda m: m.group(0).replace(" ", " "),
        text,
        flags=re.IGNORECASE,
    )

    # Java Script -> JavaScript
    text = re.sub(
        r"\bJava\s+Script\b",
        "JavaScript",
        text,
        flags=re.IGNORECASE,
    )

    # Type Script -> TypeScript
    text = re.sub(
        r"\bType\s+Script\b",
        "TypeScript",
        text,
        flags=re.IGNORECASE,
    )

    # Node . js -> Node.js
    text = re.sub(
        r"\bNode\s*\.\s*js\b",
        "Node.js",
        text,
        flags=re.IGNORECASE,
    )

    # Next . js -> Next.js
    text = re.sub(
        r"\bNext\s*\.\s*js\b",
        "Next.js",
        text,
        flags=re.IGNORECASE,
    )

    # .NET / . Net / . NET
    text = re.sub(
        r"\.\s*N\s*E\s*T\b",
        ".NET",
        text,
        flags=re.IGNORECASE,
    )

    return text


# ==========================================================================
# Reversed Arabic numeric/symbol tokens (RTL extraction artifacts)
# ==========================================================================
#
# PDF is not bidi-aware: text is placed via positioning operators, not a
# logical reading-order stream. Many PDF producers writing Arabic (RTL)
# paragraphs that contain short embedded LTR runs (like "5+", a number)
# end up storing/emitting that run in an order that reads correctly only
# visually, not logically. Extraction libraries such as pypdf that don't
# apply bidi resolution then surface the run reversed, e.g. a source
# "5+ سنوات" (5+ years) comes out as "+5 سنوات".
#
# This fix is intentionally narrow: it only touches a leading +/- symbol
# immediately followed by digits, when that token sits right next to
# Arabic text AND is immediately followed by a recognizable
# years/experience word. This avoids corrupting unrelated numbers
# (phone numbers, dates, salaries, percentages, job codes, etc.).
#
# NOTE: this is a targeted patch for the specific artifact reported
# ("+5" instead of "5+"). If real documents show other reversed-number
# patterns (e.g. inside dates or ranges), extend
# `_ARABIC_REVERSED_NUMBER_PATTERNS` accordingly rather than broadening
# this regex indiscriminately.
# --------------------------------------------------------------------------

_ARABIC_YEAR_WORDS = (
    r"سن[ةه]|سنوات|سنين|عام|أعوام|عاماً|عامًا"
)

_ARABIC_REVERSED_NUMBER_PATTERNS = [
    # "+5 سنوات" (symbol before digits, Arabic year word right after)
    # -> "5+ سنوات"
    re.compile(
        r"(?<![\w])([+\-])(\d+)(?=\s*(?:" + _ARABIC_YEAR_WORDS + r"))"
    ),
]


def fix_reversed_arabic_numbers(text: str) -> str:
    """
    Fix PDF extraction artifacts where a short numeric+symbol token like
    "5+" gets reversed to "+5" when it appears next to Arabic
    years-of-experience wording (common with RTL PDF extraction that
    doesn't apply bidi resolution).

    Deliberately narrow in scope, see module comment above.
    """

    for pattern in _ARABIC_REVERSED_NUMBER_PATTERNS:
        text = pattern.sub(
            lambda m: f"{m.group(2)}{m.group(1)}",
            text,
        )

    return text


# ==========================================================================
# Bullet handling
# ==========================================================================

_BULLET_PATTERN = re.compile(
    r"^\s*(?:[•●▪♦‣∙*+-]|\d+[\.\)])\s+"
)


# ==========================================================================
# Heading / label detection
# ==========================================================================

_LABEL_VALUE_PATTERN = re.compile(
    r"^[A-Za-z\u0600-\u06FF]"
    r"[A-Za-z\u0600-\u06FF /&_-]{2,50}"
    r":\s*.+$"
)


_STANDALONE_HEADING_PATTERN = re.compile(
    r"^[A-Za-z\u0600-\u06FF /&_-]{3,60}$"
)


def _is_standalone_line(line: str) -> bool:
    """
    Check whether a line is likely to be a complete label or heading.

    Such lines should not be merged with the following line when fixing
    PDF line breaks.
    """

    return bool(
        _LABEL_VALUE_PATTERN.match(line)
        or _STANDALONE_HEADING_PATTERN.match(line)
    )


# ==========================================================================
# Result type
# ==========================================================================

class PreprocessResult(TypedDict):
    """
    Output of the preprocessing stage.
    """

    cleaned_text: str
    language: str


# ==========================================================================
# PDF text extraction
# ==========================================================================

def extract_text_from_pdf(file_path: Union[str, Path]) -> str:
    """
    Extract raw text from a PDF file.

    This function performs ONLY raw text extraction.
    Cleaning and normalization happen later.

    Raises:
        InvalidInputError:
            If the path doesn't exist or isn't a PDF.

        PreprocessingError:
            If the PDF cannot be opened/parsed or contains no extractable
            text.
    """

    try:
        from pypdf import PdfReader

    except ImportError as exc:
        raise PreprocessingError(
            "The 'pypdf' package is required for PDF extraction. "
            "Install it with: pip install pypdf",
            details={
                "missing_dependency": "pypdf"
            },
        ) from exc

    path = Path(file_path)

    # --------------------------------------------------------------
    # Validate path
    # --------------------------------------------------------------

    if not path.exists():
        raise InvalidInputError(
            f"PDF file not found: {path}",
            details={
                "reason": "file_not_found",
                "path": str(path),
            },
        )

    if path.suffix.lower() != ".pdf":
        raise InvalidInputError(
            f"File is not a PDF: {path}",
            details={
                "reason": "invalid_extension",
                "path": str(path),
            },
        )

    # --------------------------------------------------------------
    # Extract text
    # --------------------------------------------------------------

    try:
        reader = PdfReader(str(path))

        pages_text: list[str] = []

        for page in reader.pages:
            page_text = page.extract_text() or ""
            pages_text.append(page_text)

        text = "\n".join(pages_text)

    except Exception as exc:
        raise PreprocessingError(
            f"Failed to read/parse PDF file: {path}",
            details={
                "path": str(path),
                "error": str(exc),
            },
        ) from exc

    # --------------------------------------------------------------
    # Make sure something was extracted
    # --------------------------------------------------------------

    if not text.strip():
        raise PreprocessingError(
            "No extractable text found in PDF. "
            "The file may be scanned/image-only and require OCR.",
            details={
                "path": str(path),
            },
        )

    return text


# ==========================================================================
# Validation
# ==========================================================================

def validate_raw_text(text: Optional[str]) -> str:
    """
    Validate raw text before preprocessing.
    """

    if text is None:
        raise InvalidInputError(
            "Input text is None.",
            details={
                "reason": "missing_text"
            },
        )

    stripped = text.strip()

    if not stripped:
        raise InvalidInputError(
            "Input text is empty.",
            details={
                "reason": "empty_text"
            },
        )

    if len(stripped) < MIN_TEXT_LENGTH:
        raise InvalidInputError(
            f"Input text is too short ({len(stripped)} characters).",
            details={
                "min_length": MIN_TEXT_LENGTH,
                "actual_length": len(stripped),
            },
        )

    return stripped


# ==========================================================================
# Unicode normalization
# ==========================================================================

def normalize_unicode(text: str) -> str:
    """
    Normalize Unicode and replace common PDF extraction artifacts.
    """

    # NFKC makes visually equivalent Unicode representations consistent.
    text = unicodedata.normalize("NFKC", text)

    # Replace known Unicode artifacts.
    for old, new in _UNICODE_REPLACEMENTS.items():
        text = text.replace(old, new)

    # Fix technical/PDF-specific artifacts.
    text = fix_common_pdf_artifacts(text)

    # Fix reversed Arabic numeric/symbol tokens (e.g. "+5" -> "5+" next
    # to a years-of-experience word). Must run after the unicode/artifact
    # replacements above so quotation/dash normalization doesn't interfere
    # with the digit/symbol pattern.
    text = fix_reversed_arabic_numbers(text)

    return text


# ==========================================================================
# PDF line-break handling
# ==========================================================================

def fix_pdf_line_breaks(text: str) -> str:
    """
    Repair line breaks introduced by PDF layout.

    Example:

        Preparing and reviewing all kinds of Credit Investigation
        reports about Corporate Clients.

    becomes:

        Preparing and reviewing all kinds of Credit Investigation reports
        about Corporate Clients.

    We preserve:
        - bullets
        - headings
        - label:value lines
        - paragraph boundaries
    """

    lines = text.split("\n")

    fixed: list[str] = []
    buffer = ""

    for line in lines:

        stripped = line.strip()

        # ----------------------------------------------------------
        # Preserve blank lines as paragraph boundaries
        # ----------------------------------------------------------

        if not stripped:

            if buffer:
                fixed.append(buffer)
                buffer = ""

            fixed.append("")
            continue

        # ----------------------------------------------------------
        # Detect bullet
        # ----------------------------------------------------------

        is_bullet = bool(_BULLET_PATTERN.match(stripped))

        # ----------------------------------------------------------
        # Decide whether current line belongs to previous line
        # ----------------------------------------------------------

        should_join = (
            bool(buffer)
            and not is_bullet
            and not _is_standalone_line(stripped)
            and not _is_standalone_line(buffer)
            and not buffer.endswith(
                (
                    ".",
                    ":",
                    ";",
                    "!",
                    "?",
                    "؟",
                )
            )
        )

        if should_join:

            buffer = f"{buffer} {stripped}"

        else:

            if buffer:
                fixed.append(buffer)

            buffer = stripped

    # --------------------------------------------------------------
    # Flush remaining buffer
    # --------------------------------------------------------------

    if buffer:
        fixed.append(buffer)

    return "\n".join(fixed)


# ==========================================================================
# Bullet normalization
# ==========================================================================

def normalize_bullets(text: str) -> str:
    """
    Normalize different bullet styles into a single '- ' format.

    Examples:

        • Python
        ● FastAPI
        1. Docker
        2) PostgreSQL

    become:

        - Python
        - FastAPI
        - Docker
        - PostgreSQL
    """

    lines = text.split("\n")

    normalized: list[str] = []

    for line in lines:

        if _BULLET_PATTERN.match(line):

            content = _BULLET_PATTERN.sub("", line).strip()

            if content:
                normalized.append(f"- {content}")
            else:
                normalized.append("")

        else:
            normalized.append(line)

    return "\n".join(normalized)


# ==========================================================================
# Whitespace normalization
# ==========================================================================

def normalize_whitespace(text: str) -> str:
    """
    Normalize spaces, tabs, and excessive blank lines.

    Single newlines are preserved because they may represent:
        - headings
        - bullet items
        - responsibilities
        - requirements
    """

    # Replace tabs and repeated spaces.
    text = re.sub(r"[ \t]+", " ", text)

    # Remove spaces around line breaks.
    text = re.sub(r" *\n *", "\n", text)

    # Collapse 3+ consecutive newlines into 2.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ==========================================================================
# Language detection
# ==========================================================================

def _character_language_ratio(text: str) -> tuple[float, float]:
    """
    Calculate Arabic and Latin character ratios.

    Returns:
        (arabic_ratio, latin_ratio)
    """

    arabic_chars = len(
        re.findall(r"[\u0600-\u06FF]", text)
    )

    latin_chars = len(
        re.findall(r"[A-Za-z]", text)
    )

    total = arabic_chars + latin_chars

    if total == 0:
        return 0.0, 0.0

    return (
        arabic_chars / total,
        latin_chars / total,
    )


def detect_language(text: str) -> str:
    """
    Detect whether the job description is:

        ar      -> Arabic
        en      -> English
        mixed   -> Arabic + English
        unknown -> insufficient language evidence

    Character ratios are used because job descriptions often contain
    technical English terms inside Arabic text.
    """

    arabic_ratio, latin_ratio = _character_language_ratio(text)

    # No meaningful Arabic or Latin characters.
    if arabic_ratio == 0.0 and latin_ratio == 0.0:
        return "unknown"

    # Strongly Arabic.
    if arabic_ratio >= 0.70:
        return "ar"

    # Strongly English.
    if latin_ratio >= 0.70:
        return "en"

    # Both languages have meaningful presence.
    return "mixed"


# ==========================================================================
# Public preprocessing class
# ==========================================================================

class Preprocessor:
    """
    Main preprocessing service.

    Input:
        raw_text + input_type
        OR
        PDF file path

    Output:
        cleaned_text + detected language
    """

    def process(
        self,
        raw_text: Optional[str],
        input_type: InputType = InputType.TEXT,
    ) -> PreprocessResult:

        try:

            # ------------------------------------------------------
            # 1. Validate
            # ------------------------------------------------------

            text = validate_raw_text(raw_text)

            # ------------------------------------------------------
            # 2. Unicode + PDF artifact normalization (includes
            #    reversed-Arabic-number fix)
            # ------------------------------------------------------

            text = normalize_unicode(text)

            # ------------------------------------------------------
            # 3. PDF-specific line-break repair
            # ------------------------------------------------------

            if input_type == InputType.PDF:
                text = fix_pdf_line_breaks(text)

            # ------------------------------------------------------
            # 4. Normalize bullets
            # ------------------------------------------------------

            text = normalize_bullets(text)

            # ------------------------------------------------------
            # 5. Normalize whitespace
            # ------------------------------------------------------

            text = normalize_whitespace(text)

            # ------------------------------------------------------
            # 6. Detect language
            # ------------------------------------------------------

            language = detect_language(text)

            return {
                "cleaned_text": text,
                "language": language,
            }

        except InvalidInputError:
            # Keep specific validation errors.
            raise

        except Exception as exc:

            raise PreprocessingError(
                "Unexpected failure while preprocessing text.",
                details={
                    "input_type": str(input_type),
                    "error": str(exc),
                },
            ) from exc

    def process_pdf(
        self,
        file_path: Union[str, Path],
    ) -> PreprocessResult:
        """
        Extract text from a PDF and run preprocessing.
        """

        raw_text = extract_text_from_pdf(file_path)

        return self.process(
            raw_text,
            input_type=InputType.PDF,
        )


# ==========================================================================
# Quick self-test
# ==========================================================================

if __name__ == "__main__":

    # --------------------------------------------------------------
    # 1. English PDF-like text
    # --------------------------------------------------------------

    messy_pdf_text = """
    Job Purpose

    Preparing and reviewing all kinds of Credit Investigation
    reports about Corporate Clients.

    Responsibilities

    • Carry out Managerial responsibilities for the
    Credit Investigation Team.

    • Making visits to clients to verify financial issues.

    Requirements

    1. University degree
    2. Minimum 10 years experience
    """

    result = Preprocessor().process(
        messy_pdf_text,
        input_type=InputType.PDF,
    )

    print("=== English PDF Test ===")
    print("Detected language:", result["language"])
    print("---- Cleaned text ----")
    print(result["cleaned_text"])

    # --------------------------------------------------------------
    # 2. Arabic test
    # --------------------------------------------------------------

    arabic_text = """
    المسمى الوظيفي: مهندس برمجيات

    المسؤوليات:

    • تطوير تطبيقات الويب
    • تصميم واجهات برمجية API

    المتطلبات:

    • خبرة 3 سنوات
    • إجادة Python
    """

    result = Preprocessor().process(
        arabic_text,
        input_type=InputType.TEXT,
    )

    print("\n=== Arabic Test ===")
    print("Detected language:", result["language"])
    print("---- Cleaned text ----")
    print(result["cleaned_text"])

    # --------------------------------------------------------------
    # 3. Mixed language test
    # --------------------------------------------------------------

    mixed_text = """
    Senior Software Engineer

    المسؤوليات:

    • Develop backend APIs using Python
    • تصميم الأنظمة وقواعد البيانات

    Requirements:

    • 3 years of experience
    """

    result = Preprocessor().process(
        mixed_text,
        input_type=InputType.TEXT,
    )

    print("\n=== Mixed Language Test ===")
    print("Detected language:", result["language"])
    print("---- Cleaned text ----")
    print(result["cleaned_text"])

    # --------------------------------------------------------------
    # 4. C# PDF artifact test
    # --------------------------------------------------------------

    technical_text = """
    Skills:

    • Python
    • Java
    • C .#
    • SQL
    • C ++
    • Java Script
    • Node . js
    """

    result = Preprocessor().process(
        technical_text,
        input_type=InputType.PDF,
    )

    print("\n=== Technical PDF Artifact Test ===")
    print("---- Cleaned text ----")
    print(result["cleaned_text"])

    # --------------------------------------------------------------
    # 5. Reversed Arabic number test
    # --------------------------------------------------------------

    reversed_number_text = """
    المتطلبات:

    • خبرة +5 سنوات في المبيعات
    • خبرة +3 سنة في إدارة المشاريع
    """

    result = Preprocessor().process(
        reversed_number_text,
        input_type=InputType.PDF,
    )

    print("\n=== Reversed Arabic Number Test ===")
    print("---- Cleaned text ----")
    print(result["cleaned_text"])
    assert "5+ سنوات" in result["cleaned_text"], (
        "Expected '+5' to be corrected to '5+'"
    )
    assert "3+ سنة" in result["cleaned_text"], (
        "Expected '+3' to be corrected to '3+'"
    )
    print("Reversed-number fix verified.")

    # --------------------------------------------------------------
    # 6. Invalid input test
    # --------------------------------------------------------------

    try:

        Preprocessor().process("   ")

    except InvalidInputError as exc:

        print("\n=== Invalid Input Test ===")
        print("Correctly rejected:", exc)

    # --------------------------------------------------------------
    # 7. Optional real PDF test
    # --------------------------------------------------------------

    sample_pdf = Path("data/sample_jobs/sample.pdf")

    if sample_pdf.exists():

        result = Preprocessor().process_pdf(sample_pdf)

        print("\n=== Real PDF Test ===")
        print("Detected language:", result["language"])
        print("---- Cleaned text (first 1000 chars) ----")
        print(result["cleaned_text"][:1000])