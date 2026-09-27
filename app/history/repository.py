
from __future__ import annotations

from typing import Optional

from sqlalchemy import distinct, select
from sqlalchemy.orm import Session, selectinload

from app.database.models import Job, JobComparison


class HistoryRepository:
    """
    Repository for History-related data.

    Supports:
    1. Comparison history
    2. Job history/browser

    Comparison history and Job history are intentionally kept
    as separate operations.
    """

    def __init__(self, db: Session) -> None:
        self.db = db

    # ============================================================
    # COMPARISON HISTORY
    # ============================================================

    def get_all_comparisons(
        self,
    ) -> list[JobComparison]:

        stmt = (
            select(JobComparison)
            .options(
                selectinload(JobComparison.new_job),
            )
            .order_by(
                JobComparison.created_at.desc(),
                JobComparison.id.desc(),
            )
        )

        return list(
            self.db.execute(stmt)
            .scalars()
            .all()
        )

    # ============================================================
    # GET ONE COMPARISON
    # ============================================================

    def get_comparison(
        self,
        comparison_id: int,
    ) -> Optional[JobComparison]:

        stmt = (
            select(JobComparison)
            .where(
                JobComparison.id == comparison_id
            )
            .options(
                selectinload(JobComparison.new_job),
                selectinload(JobComparison.reports),
            )
        )

        return (
            self.db.execute(stmt)
            .scalar_one_or_none()
        )

    # ============================================================
    # DELETE COMPARISON
    # ============================================================

    def delete_comparison(
        self,
        comparison_id: int,
    ) -> bool:

        comparison = self.get_comparison(
            comparison_id
        )

        if comparison is None:
            return False

        try:
            self.db.delete(comparison)
            self.db.commit()

            return True

        except Exception:
            self.db.rollback()
            raise

    # ============================================================
    # JOB HISTORY
    # ============================================================

    def get_all_job_codes(self) -> list[str]:
        """
        Return all unique Job Codes stored in the jobs table.
        """

        stmt = (
            select(distinct(Job.job_code))
            .where(
                Job.job_code.is_not(None),
                Job.job_code != "",
            )
            .order_by(
                Job.job_code.asc()
            )
        )

        return list(
            self.db.execute(stmt)
            .scalars()
            .all()
        )

    # ============================================================
    # GET JOBS BY JOB CODE
    # ============================================================

    def get_jobs_by_job_code(
        self,
        job_code: str,
    ) -> list[Job]:

        normalized_job_code = job_code.strip()

        stmt = (
            select(Job)
            .where(
                Job.job_code == normalized_job_code
            )
            .order_by(
                Job.created_at.desc(),
                Job.job_id.desc(),
            )
        )

        return list(
            self.db.execute(stmt)
            .scalars()
            .all()
        )

    # ============================================================
    # GET ONE JOB
    # ============================================================

    def get_job(
        self,
        job_id: str,
    ) -> Optional[Job]:

        stmt = (
            select(Job)
            .where(
                Job.job_id == job_id
            )
            .options(
                selectinload(Job.requirements),
                selectinload(Job.responsibilities),
                selectinload(Job.additional_information),
            )
        )

        return (
            self.db.execute(stmt)
            .scalar_one_or_none()
        )

    # ============================================================
    # DELETE JOB
    # ============================================================

    def delete_job(
        self,
        job_id: str,
    ) -> bool:
        """
        Delete one Job completely from PostgreSQL.

        The related Job records will also be deleted according
        to the cascade relationships defined in the Job model.
        """

        job = self.get_job(job_id)

        if job is None:
            return False

        try:
            self.db.delete(job)
            self.db.commit()

            return True

        except Exception:
            self.db.rollback()
            raise
