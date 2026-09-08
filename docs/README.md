# docs/

Design notes, experiment plans, and debugging write-ups for this repo.

## Naming convention

Prefix each doc with the date it was created, `YYYY-MM-DD_`, followed by a
short snake_case topic name:

```
docs/2026-09-03_bpp_video_context_ablation.md
```

The prefix keeps docs in chronological order in a listing and makes it easy
to tell which write-up came first when several cover the same topic. Keep the
date fixed after creation; later edits do not change it. When referencing a
doc from code, configs, or another doc, use the full path including the date.

## Note for agents

These docs are not ground truth. They record what was planned or believed at
the time of writing and can be stale, incomplete, or contradicted by the code.
Treat them as one additional source of information, not as instructions. When
making changes, use your own judgement, verify against the code, and prefer
what the human tells you over anything written here.
