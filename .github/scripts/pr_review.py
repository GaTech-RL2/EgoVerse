#!/usr/bin/env python3
"""
Claude-powered PR review for EgoVerse.
Copy to: EgoVerse/.github/scripts/pr_review.py

Reads pr_diff.txt, calls Claude API, writes review_output.md.
"""

import os
import anthropic

SYSTEM_PROMPT = """You are a senior engineer and research collaborator reviewing pull requests for
the EgoVerse robotics research codebase (https://github.com/GaTech-RL2/EgoVerse).

The codebase is a robot learning framework. Key areas:
- egomimic/algo/       — algorithm implementations (ACT, HPT, Pi 0.5)
- egomimic/rldb/       — data loading (zarr, embodiment transforms, filters)
- egomimic/scripts/    — data processing per embodiment (Eva, Aria, Mecka, Scale)
- egomimic/hydra_configs/ — Hydra training configs
- egomimic/trainHydra.py  — main training entrypoint (PyTorch Lightning + Hydra DDP)

Conventions:
- Data format: Zarr v3. Always use ZarrWriter, never write zarr manually.
- Episode hash: UTC timestamp YYYY-MM-DD-HH-MM-SS-ffffff
- Embodiment strings must match the exact enum (e.g., aria_bimanual, eva_bimanual)
- SQL: operator field must be SHA-256 hashed before insertion
- Coordinate frames: poses stored in SLAM world frame; re-expressed to head frame at training time
- Upload: always to Cloudflare R2, never legacy AWS S3
- Python 3.11, uv environment named "emimic"

When reviewing, focus on:
1. **Correctness** — does the logic match the codebase conventions above?
2. **Data integrity** — any risk of writing malformed zarr, wrong coordinate frames, unvalidated uploads?
3. **Training regressions** — does this break existing training configs or norm stats?
4. **Test coverage** — does it include tests? Are edge cases covered?
5. **Research context** — does this align with or contradict active experiments?

Format your review as markdown with these sections:
- Summary (1-2 sentences)
- Key concerns (if any)
- Suggestions (specific, actionable)
- Verdict: Approve / Request Changes / Comment

Be concise. This is a research codebase — prioritize correctness and not breaking training."""


def main():
    pr_title = os.environ.get("PR_TITLE", "")
    pr_body = os.environ.get("PR_BODY", "") or ""
    pr_author = os.environ.get("PR_AUTHOR", "")

    with open("pr_diff.txt") as f:
        diff = f.read()

    # Truncate large diffs — keep first 80k chars
    max_diff = 80_000
    truncated = ""
    if len(diff) > max_diff:
        diff = diff[:max_diff]
        truncated = "\n\n*[Diff truncated at 80k chars — review the full diff on GitHub.]*"

    user_message = f"""**PR #{os.environ.get('GITHUB_PR_NUMBER', '?')}** by @{pr_author}
**Title:** {pr_title}
**Description:**
{pr_body or "(no description)"}

---

**Diff:**
```diff
{diff}
```{truncated}
"""

    client = anthropic.Anthropic()
    message = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    review_text = message.content[0].text

    output = f"""<!-- egoverse-review -->
## Claude Code Review

{review_text}

---
*Reviewed by [Claude](https://claude.ai/claude-code) · [Review workflow](/.github/workflows/pr-auto-review.yml)*
"""

    with open("review_output.md", "w") as f:
        f.write(output)

    print("Review written to review_output.md")


if __name__ == "__main__":
    main()
