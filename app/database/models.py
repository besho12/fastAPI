"""
SQLAlchemy 2.x ORM models for JobComparisonAI.

"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Optional

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.connection import Base


class Job(Base):
    """
    A persisted job description.

    `job_id` is the primary key of the persisted job.
    `job_code` is extracted from the submitted job text and is the
    authoritative grouping key for comparison.

    Multiple historical jobs can have the same `job_code`.
    Jobs are compared only with other jobs having the same `job_code`.

    `company_code` identifies the company associated with the job.
    It does not determine comparison scope.

    `content_hash` and `raw_content_hash` are used for exact duplicate
    detection within the same `job_code`.
    """

    __tablename__ = "jobs"

    __table_args__ = (
        UniqueConstraint(
            "job_code",
            "company_code",
            "content_hash",
            name="uq_jobs_job_code_company_code_content_hash",
        ),
        UniqueConstraint(
            "job_code",
            "company_code",
            "raw_content_hash",
            name="uq_jobs_job_code_company_code_raw_content_hash",
        ),
    )

    # --- Identity -----------------------------------------------------
    job_id: Mapped[str] = mapped_column(String(64), primary_key=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    # --- Job identity / grouping ---------------------------------------
    job_code: Mapped[str] = mapped_column(
        String(64),
        index=True,
        nullable=False,
    )

    company_code: Mapped[str] = mapped_column(
        String(64),
        index=True,
        nullable=False,
    )

    content_hash: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
        index=True,
    )

    raw_content_hash: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
        index=True,
    )

    # --- Job information -------------------------------------------------
    job_title: Mapped[Optional[str]] = mapped_column(
        String(255), index=True, nullable=True
    )
    department: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    grade: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    reports_to: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    number_of_job_holders: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True
    )
    number_of_direct_reports: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True
    )
    creation_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)

    # --- Company info ----------------------------------------------------
    company_industry: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )

    # --- Source info -----------------------------------------------------
    # See "Notes on enums" above re: input_type being a plain string.
    input_type: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    original_file_name: Mapped[Optional[str]] = mapped_column(
        String(512), nullable=True
    )
    language: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)

    # --- Content -----------------------------------------------------------
    job_purpose: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)

    # --- Record bookkeeping (distinct from the JobDescription's own
    # created_at above) ---------------------------------------------------
    record_created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    record_updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # --- Relationships -----------------------------------------------------
    requirements: Mapped[list["JobRequirement"]] = relationship(
        back_populates="job",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    responsibilities: Mapped[list["JobResponsibility"]] = relationship(
        back_populates="job",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="JobResponsibility.position",
    )
    additional_information: Mapped[list["JobAdditionalInformation"]] = relationship(
        back_populates="job",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    comparisons: Mapped[list["JobComparison"]] = relationship(
        back_populates="new_job",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    reports: Mapped[list["JobReport"]] = relationship(
        back_populates="job",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging convenience
        return f"<Job job_id={self.job_id!r} job_code={self.job_code!r}>"


class JobRequirement(Base):
    """
    A single requirement entry belonging to a Job, grouped by category.

    Replaces 7 separate columns/tables (education, experience, skills,
    language, computer, soft_skills, field_of_experience) with one
    normalized (job_id, category, value, position) table.
    """

    __tablename__ = "job_requirements"
    __table_args__ = (
        Index("ix_job_requirements_job_id_category", "job_id", "category"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("jobs.job_id", ondelete="CASCADE"), nullable=False
    )

    category: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    position: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    job: Mapped["Job"] = relationship(back_populates="requirements")

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<JobRequirement job_id={self.job_id!r} "
            f"category={self.category!r} value={self.value!r}>"
        )


class JobResponsibility(Base):
    """A single ordered responsibility entry belonging to a Job."""

    __tablename__ = "job_responsibilities"
    __table_args__ = (
        Index("ix_job_responsibilities_job_id_position", "job_id", "position"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("jobs.job_id", ondelete="CASCADE"), nullable=False
    )
    value: Mapped[str] = mapped_column(Text, nullable=False)
    position: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    job: Mapped["Job"] = relationship(back_populates="responsibilities")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<JobResponsibility job_id={self.job_id!r} position={self.position!r}>"


class JobAdditionalInformation(Base):
    """
    A single free-form key/value entry for a Job's additional_information
    (originally Dict[str, Any] in JobDescription).

    `value` is stored as JSONB rather than plain text: additional_information
    is explicitly typed as Dict[str, Any] upstream, so a given value may be
    a string, number, boolean, list, or nested object. JSONB preserves that
    type information on round trip and still allows querying/indexing on
    the value if ever needed, whereas a plain TEXT column would force every
    value to be stringified and lose type fidelity.
    """

    __tablename__ = "job_additional_information"
    __table_args__ = (
        UniqueConstraint("job_id", "key", name="uq_job_additional_info_job_key"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("jobs.job_id", ondelete="CASCADE"), nullable=False
    )
    key: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    value: Mapped[Any] = mapped_column(JSONB, nullable=True)

    job: Mapped["Job"] = relationship(back_populates="additional_information")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<JobAdditionalInformation job_id={self.job_id!r} key={self.key!r}>"


class JobComparison(Base):
    """
    A single comparison OPERATION: one new job compared against
    potentially many historical reference jobs.

    Design choice (option B from the spec): rather than a separate
    ComparisonSet/ComparisonItem pair of tables, each comparison operation
    is one row. `reference_jobs` is a JSONB array capturing the reference
    job identifiers/metadata used in that operation (e.g.
    [{"job_id": "...", "job_code": "...", "similarity_score": 0.87}, ...]),
    which keeps the schema simple while still fully supporting a
    one-new-job-to-many-reference-jobs comparison.

    The full Gemini report content itself lives in JobReport.report_data;
    this table tracks the comparison operation's identity/status/inputs.
    """

    __tablename__ = "job_comparisons"
    __table_args__ = (
        Index("ix_job_comparisons_new_job_id", "new_job_id"),
        Index("ix_job_comparisons_job_code", "job_code"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    new_job_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("jobs.job_id", ondelete="CASCADE"), nullable=False
    )
    # NOTE: index=True intentionally removed here -- the explicit
    # Index("ix_job_comparisons_job_code", "job_code") above already
    # creates this index. Keeping both created a duplicate index with
    # the same name, which SQLAlchemy would try to CREATE INDEX twice.
    job_code: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True,
    )

    # See "Notes on enums" -- mirrors app.schemas.ComparisonStatus values.
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)

    reference_jobs: Mapped[list[Any]] = mapped_column(JSONB, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    new_job: Mapped["Job"] = relationship(back_populates="comparisons")
    reports: Mapped[list["JobReport"]] = relationship(
        back_populates="comparison",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<JobComparison id={self.id!r} new_job_id={self.new_job_id!r} "
            f"status={self.status!r}>"
        )


class JobReport(Base):
    """
    A persisted GeminiComparisonReport.

    Only the fields useful for filtering/listing without loading the full
    payload are promoted to real columns (status, confidence,
    is_substantive_role_change). Everything else -- sections,
    semantic_changes, evolution_analysis, overall_summary,
    overall_assessment, and any future additions -- is stored verbatim in
    `report_data` (JSONB) rather than flattened into dozens of columns.
    """

    __tablename__ = "job_reports"
    __table_args__ = (
        Index("ix_job_reports_job_id", "job_id"),
        Index("ix_job_reports_job_code", "job_code"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    job_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("jobs.job_id", ondelete="CASCADE"), nullable=False
    )
    # NOTE: index=True intentionally removed here -- the explicit
    # Index("ix_job_reports_job_code", "job_code") above already
    # creates this index. Keeping both created a duplicate index with
    # the same name, which SQLAlchemy would try to CREATE INDEX twice.
    job_code: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True,
    )
    comparison_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        ForeignKey("job_comparisons.id", ondelete="CASCADE"),
        nullable=True,
    )

    report_type: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    status: Mapped[Optional[str]] = mapped_column(
        String(32), nullable=True, index=True
    )
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    is_substantive_role_change: Mapped[Optional[bool]] = mapped_column(
        Boolean, nullable=True
    )

    report_data: Mapped[Any] = mapped_column(JSONB, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    job: Mapped["Job"] = relationship(back_populates="reports")
    comparison: Mapped[Optional["JobComparison"]] = relationship(
        back_populates="reports"
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<JobReport id={self.id!r} job_id={self.job_id!r} status={self.status!r}>"

class User(Base):
    

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    username: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        index=True,
        nullable=False,
    )

    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        index=True,
        nullable=False,
    )

    password_hash: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    # All newly registered users are normal users.
    role: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="user",
    )

    # New users must verify their email using OTP.
    is_verified: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
    )

    # OTP is stored as a hash, not as plain text.
    otp_hash: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    otp_expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    def __repr__(self) -> str:
        return (
            f"<User id={self.id!r} "
            f"email={self.email!r} "
            f"role={self.role!r}>"
        )