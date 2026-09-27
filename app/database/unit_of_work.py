"""
app/database/unit_of_work.py
"""
from __future__ import annotations

from sqlalchemy.orm import Session, SessionTransaction


class PostgresUnitOfWork:
    """
    Owns the PostgreSQL transaction boundary for one atomic unit of work
    (currently: Job + Comparison + Report persistence in a pipeline run).

    This exists so callers like JobComparisonPipeline never call
    Session.begin()/.rollback()/.in_transaction() themselves -- see
    app/database/repositories.py's "TRANSACTION-OWNERSHIP NOTE". The
    pipeline composes repository calls inside `with PostgresUnitOfWork(...)`;
    this class is the only thing that knows SQLAlchemy's transaction API.

    NOTE: `JobGroupRepository.get_or_create_job_group()` is intentionally
    NOT composed inside this unit of work -- it owns its own commit (see
    that method's docstring), the same way `JobRepository.update_job_code`
    and `delete_job` already do. Nothing about that decision required any
    change to this class.
    """

    def __init__(self, session: Session) -> None:
        self._session = session
        self._transaction: SessionTransaction | None = None

    def __enter__(self) -> "PostgresUnitOfWork":
        # Close out any pre-existing, read-only autobegin transaction left
        # open by earlier read-only lookups on this same Session (e.g. the
        # raw-hash duplicate check), so begin() below opens a clean,
        # self-owned transaction boundary. Nothing was ever written to
        # that pre-existing transaction, so this rollback has no effect
        # on any data.
        if self._session.in_transaction():
            self._session.rollback()

        self._transaction = self._session.begin()
        self._transaction.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        assert self._transaction is not None
        return self._transaction.__exit__(exc_type, exc, tb)


__all__ = ["PostgresUnitOfWork"]