"""
config.py
---------
Centralized configuration for the Job Comparison & Analysis AI System.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

from app.exceptions import ConfigurationError

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL")

# LLM provider used by the active extraction and comparison pipelines.
# Gemini remains available as an explicit rollback option, but new
# deployments use OpenAI by default.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip().lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-6-astra")
OPENAI_REQUEST_TIMEOUT_SECONDS = int(
    os.getenv("OPENAI_REQUEST_TIMEOUT_SECONDS", "90")
)
OPENAI_MAX_OUTPUT_TOKENS = int(
    os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "16000")
)

QDRANT_PATH = os.getenv(
    "QDRANT_PATH",
    "qdrant_storage",
)

QDRANT_COLLECTION_NAME = os.getenv(
    "QDRANT_COLLECTION_NAME",
    "jobs",
)

QDRANT_VECTOR_SIZE = int(
    os.getenv(
        "QDRANT_VECTOR_SIZE",
        "1024",
    )
)

DATABASE_URL = os.getenv("DATABASE_URL")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "60"))

EMAIL_HOST = os.getenv("EMAIL_HOST")
EMAIL_PORT = int(os.getenv("EMAIL_PORT", "587"))
EMAIL_USERNAME = os.getenv("EMAIL_USERNAME")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_FROM = os.getenv("EMAIL_FROM")

DUPLICATE_SIMILARITY_THRESHOLD = float(
    os.getenv(
        "DUPLICATE_SIMILARITY_THRESHOLD",
        "0.95",
    )
)

class Settings:
    """
    Compatibility settings object.

    This allows modules such as app.llm to access configuration through:

        settings.GEMINI_API_KEY
        settings.GEMINI_MODEL

    while keeping the existing module-level configuration variables
    available for the rest of the project.
    """

    # Gemini
    GEMINI_API_KEY = GEMINI_API_KEY
    GEMINI_MODEL = GEMINI_MODEL

    # Active model provider / OpenAI
    LLM_PROVIDER = LLM_PROVIDER
    OPENAI_API_KEY = OPENAI_API_KEY
    OPENAI_MODEL = OPENAI_MODEL
    OPENAI_REQUEST_TIMEOUT_SECONDS = OPENAI_REQUEST_TIMEOUT_SECONDS
    OPENAI_MAX_OUTPUT_TOKENS = OPENAI_MAX_OUTPUT_TOKENS

    # Qdrant
    QDRANT_PATH = QDRANT_PATH
    QDRANT_COLLECTION_NAME = QDRANT_COLLECTION_NAME
    QDRANT_VECTOR_SIZE = QDRANT_VECTOR_SIZE

    # PostgreSQL
    DATABASE_URL = DATABASE_URL
    # JWT
    JWT_SECRET_KEY = JWT_SECRET_KEY
    JWT_ALGORITHM = JWT_ALGORITHM
    JWT_EXPIRE_MINUTES = JWT_EXPIRE_MINUTES
    # Email
    EMAIL_HOST = EMAIL_HOST
    EMAIL_PORT = EMAIL_PORT
    EMAIL_USERNAME = EMAIL_USERNAME
    EMAIL_PASSWORD = EMAIL_PASSWORD
    EMAIL_FROM = EMAIL_FROM
    # Duplicate detection
    DUPLICATE_SIMILARITY_THRESHOLD = DUPLICATE_SIMILARITY_THRESHOLD


settings = Settings()
