"""Regression tests for application imports in a clean Python process."""

from __future__ import annotations

import subprocess
import sys


def _assert_clean_import(module: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_openai_gateway_imports_in_clean_process():
    _assert_clean_import("app.openai_gateway")


def test_fastapi_application_imports_in_clean_process():
    _assert_clean_import("main")
