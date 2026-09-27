# RoleFit Job Comparison API

FastAPI backend for extracting structured job-description data and comparing
jobs that share a `job_code`.

## Comparison workflow

`POST /api/jobs/compare` keeps the desktop application's existing request:

```json
{
  "job_code": "CLERK-01",
  "company_code": "COMPANY-A"
}
```

The company identifies the anchor job, but comparison covers the complete
job-code group. The service performs semantic concept clustering once, then
compares every job against all remaining jobs. The returned ZIP contains:

- A group-summary workbook with per-job counts, all actionable differences,
  and a concept-coverage matrix.
- One detailed workbook per job showing missing, additional, and aligned
  requirements with benchmark evidence and an audit trail.

Counts, prevalence, missing/additional direction, and report buckets are
calculated in Python. Gemini is limited to semantic clustering and materiality
judgement, with deterministic fallbacks if it is unavailable.

## Tests

```powershell
python -m pytest -q
```
