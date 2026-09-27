"""
app/exceptions.py
-----------------

Application-level exceptions for JobComparisonAI.

These exceptions represent failures inside the application's
business logic, services, and pipeline.

They are intentionally independent of FastAPI/HTTP concerns.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class JobAIException(Exception):
    """
    Base exception for every application-level exception
    in JobComparisonAI.
    """

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | details={self.details}"
        return self.message

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the exception into a structured dictionary.

        Useful for:
        - structured logging
        - debugging
        - application-level error handling
        """
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
        }


# ============================================================================
# INPUT / FILE ERRORS
# ============================================================================


class InvalidInputError(JobAIException):
    """
    Raised when the raw input is missing, empty, or invalid.
    """

    pass


class UnsupportedFileTypeError(InvalidInputError):
    """
    Raised when a file type is not supported.

    Currently supported:
        - PDF
        - TXT
        - DOCX
    """

    pass


# ============================================================================
# TEXT PROCESSING ERRORS
# ============================================================================


class TextExtractionError(JobAIException):
    """
    Raised when raw text cannot be extracted from an input file.

    Examples:
        - corrupted PDF
        - scanned document without OCR
        - encoding problems
    """

    pass


class PreprocessingError(JobAIException):
    """
    Raised when preprocessing, cleaning, or normalization fails.

    Examples:
        - language detection failure
        - encoding issues
        - text normalization failure
    """

    pass


# ============================================================================
# EXTRACTION / VALIDATION ERRORS
# ============================================================================


class ExtractionError(JobAIException):
    """
    Raised when raw text cannot be converted into the
    unified JobDescription schema.
    """

    pass


class SchemaValidationError(JobAIException):
    """
    Raised when extracted data fails Pydantic validation.
    """

    pass


# ============================================================================
# EMBEDDING / VECTOR DATABASE ERRORS
# ============================================================================


class EmbeddingServiceError(JobAIException):
    """
    Raised when generating embeddings fails.

    Examples:
        - model/API failure
        - empty text
        - rate limit
        - embedding generation error
    """

    pass


class VectorDBError(JobAIException):
    """
    Base exception for vector database failures.
    """

    pass


class QdrantServiceError(VectorDBError):
    """
    Raised when a Qdrant operation fails.

    Covers:
        - connection failures
        - collection creation failures
        - invalid vector dimensions
        - upsert failures
        - similarity search failures
        - retrieval failures
        - deletion failures
    """

    pass


class VectorNotFoundError(VectorDBError):
    """
    Raised when a vector cannot be found for a requested job_id.
    """

    pass


class RetrievalError(JobAIException):
    """
    Raised when retrieval or similarity search fails unexpectedly.
    """

    pass


# ============================================================================
# JOB IDENTITY / COMPARISON ERRORS
# ============================================================================


class JobIdentityError(JobAIException):
    """
    Raised when the Job Identity / Grouping layer cannot
    safely determine a job's group identity.
    """

    pass


class ComparisonError(JobAIException):
    """
    Raised when semantic comparison between jobs fails.
    """

    pass


# ============================================================================
# LLM ERRORS
# ============================================================================


class PromptBuildError(JobAIException):
    """
    Raised when an LLM prompt cannot be built correctly.
    """

    pass


class LLMServiceError(JobAIException):
    """
    Raised when an LLM service operation fails.

    Examples:
        - timeout
        - API error
        - invalid response
        - empty response
        - content filtering block
    """

    pass


# Backward-compatible alias.
LLMError = LLMServiceError


class LLMResponseParsingError(LLMServiceError):
    """
    Raised when the LLM responds but its output cannot
    be parsed into the expected structure.
    """

    pass


# ============================================================================
# PIPELINE / CONFIGURATION ERRORS
# ============================================================================


class PipelineError(JobAIException):
    """
    Raised when pipeline orchestration fails and no more
    specific exception type applies.
    """

    pass


class ConfigurationError(JobAIException):
    """
    Raised when required application configuration is missing
    or invalid.

    Examples:
        - missing API key
        - missing database configuration
        - invalid environment configuration
    """

    pass