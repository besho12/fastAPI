"""Current-version selection for job comparison repositories."""

from __future__ import annotations

from types import SimpleNamespace

from app.database.repositories import JobRepository


class _Scalars:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class _Result:
    def __init__(self, rows=None, selected=None):
        self._rows = rows or []
        self._selected = selected

    def scalars(self):
        return _Scalars(self._rows)

    def scalar_one_or_none(self):
        return self._selected


class _Session:
    def __init__(self, result):
        self.result = result
        self.statement = None

    def execute(self, statement):
        self.statement = statement
        return self.result


def test_job_code_group_keeps_only_newest_row_per_company():
    newest_a = SimpleNamespace(job_id="a-new", company_code="A")
    newest_b = SimpleNamespace(job_id="b-new", company_code="B")
    older_a = SimpleNamespace(job_id="a-old", company_code="A")
    session = _Session(_Result(rows=[newest_a, newest_b, older_a]))

    jobs = JobRepository(session).get_jobs_by_job_code("ROLE-1")

    assert [job.job_id for job in jobs] == ["a-new", "b-new"]
    sql = str(session.statement)
    assert "ORDER BY jobs.record_created_at DESC, jobs.job_id DESC" in sql


def test_selected_job_lookup_orders_newest_first_and_limits_to_one():
    selected = SimpleNamespace(job_id="current")
    session = _Session(_Result(selected=selected))

    result = JobRepository(session).get_job_by_job_code_and_company_code(
        "ROLE-1", "A"
    )

    assert result is selected
    sql = str(session.statement)
    assert "ORDER BY jobs.record_created_at DESC, jobs.job_id DESC" in sql
    assert "LIMIT" in sql
