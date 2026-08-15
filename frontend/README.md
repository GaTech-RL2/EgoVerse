# EgoVerse Diversity Console

Comparison-first Next.js interface for the existing `track2` backend. The app
does not implement or alter diversity scoring.

## Execution boundary

`POST /api/jobs` starts the unchanged Track 2 command with a resolved Python:

```text
<resolved Python> -m track2.run_track2
```

The process runs from the EgoVerse repository root and inherits the web
server's environment. The app polls `GET /api/jobs/{jobId}` and, after a
successful exit, validates these unchanged backend outputs:

The bridge resolves Python in this order: `PYTHON_EXECUTABLE`, the repository's
`emimic/bin/python`, then `python` / `python3`. It rejects interpreters that are
missing the Track 2 runtime dependencies before a job starts.

```text
track2/results/final_dataset_summary.csv
track2/results/final_two_dataset_results.csv
```

The adapter writes a per-job JSON result under `frontend/data/jobs/{jobId}`.
Only one analysis may run at a time because the existing backend uses fixed CSV
paths.

## Run locally

From the EgoVerse repository root:

```bash
source emimic/bin/activate
source ~/.egoverse_env
cd frontend
npm install
npm run dev
```

If `python` is not the desired interpreter, set its absolute path before
starting Next.js:

```bash
export PYTHON_EXECUTABLE=/absolute/path/to/emimic/bin/python
```

`EGOVERSE_REPO_ROOT` can override the default parent-directory lookup.

## Validation

```bash
npm run typecheck
npm run lint
npm test
npm run build
```

## Product constraints reflected in the UI

- Fixed Dataset A / Mecka and Dataset B / Scale configuration
- Existing deterministic oldest-first, 120-hour subset preparation
- Existing Behavior, Context / Visual, and Embodiment dimensions
- Raw 0–1 scores converted to 0–100 only in the adapter
- Frontend-only weights that always rebalance to 100%
- Aggregate CSV evidence only; no PCA, distributions, or episode-level claims
