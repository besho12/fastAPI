from __future__ import annotations

import asyncio
import threading
import time

import main
from app.extraction import ExtractionService, _gemini_timeout_ms
from app.pipeline import JobIngestionResult


class _DummySession:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


def _request(count: int) -> main.BulkJobIngestionRequest:
    return main.BulkJobIngestionRequest(
        jobs=[
            main.JobInput(
                raw_text=f"Job description {index}",
                original_file_name=f"job-{index}.pdf",
            )
            for index in range(count)
        ]
    )


def test_bulk_upload_processes_documents_concurrently_with_isolated_sessions(
    monkeypatch,
):
    lock = threading.Lock()
    active = 0
    maximum_active = 0
    session_ids: list[int] = []

    class _FakePipeline:
        def __init__(self, session):
            self.session = session

        def ingest_jobs(self, jobs):
            nonlocal active, maximum_active

            with lock:
                active += 1
                maximum_active = max(maximum_active, active)
                session_ids.append(id(self.session))

            time.sleep(0.05)

            with lock:
                active -= 1

            return [
                JobIngestionResult(
                    file_name=jobs[0][2],
                    status="stored",
                )
            ]

    monkeypatch.setattr(main, "SessionLocal", _DummySession)
    monkeypatch.setattr(
        main,
        "create_pipeline",
        lambda session: _FakePipeline(session),
    )
    monkeypatch.setenv("BULK_INGESTION_CONCURRENCY", "3")

    response = asyncio.run(
        main.bulk_upload_jobs(
            _request(3),
            current_user=object(),
        )
    )

    assert response["stored"] == 3
    assert response["errors"] == 0
    assert maximum_active == 3
    assert len(set(session_ids)) == 3


def test_bulk_upload_counts_failed_status_as_an_error(monkeypatch):
    class _FakePipeline:
        def ingest_jobs(self, jobs):
            return [
                JobIngestionResult(
                    file_name=jobs[0][2],
                    status="failed",
                    error="Gemini timed out",
                )
            ]

    monkeypatch.setattr(main, "SessionLocal", _DummySession)
    monkeypatch.setattr(
        main,
        "create_pipeline",
        lambda session: _FakePipeline(),
    )

    response = asyncio.run(
        main.bulk_upload_jobs(
            _request(1),
            current_user=object(),
        )
    )

    assert response["stored"] == 0
    assert response["errors"] == 1
    assert response["results"][0]["status"] == "failed"


def test_extraction_config_is_bounded_and_uses_minimal_gemini_3_thinking(
    monkeypatch,
):
    monkeypatch.setenv("GEMINI_REQUEST_TIMEOUT_SECONDS", "40")

    config = ExtractionService(
        model="gemini-3.5-flash-lite"
    )._generation_config()

    assert _gemini_timeout_ms() == 40_000
    assert config.http_options.timeout == 40_000
    assert config.automatic_function_calling.disable is True
    assert config.thinking_config.thinking_level.value == "MINIMAL"


def test_extraction_config_disables_thinking_for_gemini_25_flash():
    config = ExtractionService(
        model="gemini-2.5-flash"
    )._generation_config()

    assert config.thinking_config.thinking_budget == 0
