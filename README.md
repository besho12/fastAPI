# RoleFit Job Comparison API

FastAPI backend for extracting structured job-description data and comparing
jobs that share a `job_code`.

## Comparison workflow

`POST /api/jobs/compare` remains compatible with the desktop application's
existing request:

```json
{
  "job_code": "CLERK-01",
  "company_code": "COMPANY-A"
}
```

It can also select an exact peer set and request the discrepancy report:

```json
{
  "job_code": "TCGNBFI010104-A",
  "company_code": "LSURV2602",
  "reference_company_codes": [
    "LSURV2604",
    "LSURV2608",
    "LSURV2609"
  ],
  "include_discrepancy_report": true
}
```

The company identifies the anchor job, but comparison covers the complete
job-code group. The service performs semantic concept clustering once, then
compares every job against all remaining jobs. The returned ZIP contains:

When multiple stored versions share the same `job_code + company_code`, the
newest version is treated as current for comparison. Older versions remain in
the database as history and are not duplicated in the peer group.

- A group-summary workbook with per-job counts, all actionable differences,
  and a concept-coverage matrix.
- One detailed workbook per job showing missing, additional, and aligned
  requirements with benchmark evidence and an audit trail.
- A target-company discrepancy workbook with Executive Summary, Criteria
  Comparison, and Duty Gap Matrix sheets when requested.

Counts, prevalence, missing/additional direction, and report buckets are
calculated in Python. OpenAI is limited to structured extraction, semantic
clustering, materiality judgement, and the target-company narrative analysis;
the comparison engine retains deterministic fallbacks if semantic model calls
are unavailable.

The frontend request does not change. To switch the complete workflow,
including the discrepancy workbook, set `LLM_PROVIDER=openai` or
`LLM_PROVIDER=gemini` on the server and recreate the web container.

## OpenAI configuration

Copy the relevant values from `.env.example` into the deployment `.env`:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your_project_api_key
OPENAI_MODEL=gpt-6-astra
OPENAI_REQUEST_TIMEOUT_SECONDS=90
OPENAI_MAX_OUTPUT_TOKENS=12000
OPENAI_REASONING_EFFORT=low
OPENAI_MAX_RETRIES=0
OPENAI_FAST_COMPARISON=true
```

The API key must remain server-side and must not be committed to Git. After
changing the server environment, recreate the web container so Compose reloads
the values:

```bash
docker build -t jd-backend:latest .
docker compose up -d --force-recreate web
```

To switch the complete workflow back to Gemini without changing the frontend:

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=your_gemini_model
GEMINI_COMPARISON_TIMEOUT_SECONDS=90
```

Removing `LLM_PROVIDER` does not select Gemini; the default provider is OpenAI.

## Tests

```powershell
python -m pytest -q
```
