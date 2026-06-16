#!/usr/bin/env python3
"""
For every PR event (open / ready_for_review / close / merge):
  1. Saves raw PR metadata → raw/prs/pr-NNNN.json
  2. Creates/updates clean wiki stub → wiki/prs/pr-NNNN.md

Copy to: EgoVerse/.github/scripts/create_wiki_pr_page.py
"""

import base64
import json
import os
from datetime import datetime

import requests

VAULT_API = "https://api.github.com/repos/{repo}/contents/{path}"


def vault_headers(pat: str) -> dict:
    return {
        "Authorization": f"Bearer {pat}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def get_sha(repo: str, path: str, headers: dict) -> str | None:
    resp = requests.get(VAULT_API.format(repo=repo, path=path), headers=headers)
    return resp.json().get("sha") if resp.status_code == 200 else None


def push_file(repo: str, path: str, content: str, message: str,
              headers: dict, sha: str | None = None):
    payload = {
        "message": message,
        "content": base64.b64encode(content.encode()).decode(),
        "branch": "main",
    }
    if sha:
        payload["sha"] = sha
    resp = requests.put(VAULT_API.format(repo=repo, path=path),
                        headers=headers, json=payload)
    resp.raise_for_status()
    print(f"{'Updated' if sha else 'Created'}: {path}")


def build_raw_metadata(pr_number: int, title: str, author: str, status: str,
                       created_at: str, body: str) -> str:
    return json.dumps({
        "pr_number": pr_number,
        "title": title,
        "author": author,
        "body": body,
        "status": status,
        "github_url": f"https://github.com/GaTech-RL2/EgoVerse/pull/{pr_number}",
        "created_at": created_at,
        "updated_at": datetime.utcnow().isoformat(),
    }, indent=2)


def build_wiki_page(pr_number: int, title: str, author: str, status: str,
                    created_at: str, body: str) -> str:
    updated = datetime.utcnow().strftime("%Y-%m-%d")
    created = created_at[:10]
    github_url = f"https://github.com/GaTech-RL2/EgoVerse/pull/{pr_number}"

    # One-line description from first non-empty line of body
    brief = next(
        (line.strip() for line in (body or "").splitlines() if line.strip()),
        "(no description)"
    )[:200]

    sources = [f"- [[raw/prs/pr-{pr_number:04d}.json]]"]
    if status == "merged":
        sources.append(f"- [[raw/prs/pr-{pr_number:04d}-diff.json]]")

    checklist = """\
- [ ] Zarr writes use `ZarrWriter` (never manual)
- [ ] Embodiment strings match exact enum
- [ ] SQL `operator` field is SHA-256 hashed
- [ ] Poses in SLAM world frame (re-expressed at training time only)
- [ ] Upload targets Cloudflare R2, not AWS S3
- [ ] No absolute paths
- [ ] `dataset_name == embodiment` assumption not made worse"""

    return f"""# PR #{pr_number} — {title}

**Type**: pr
**Author**: {author}
**Status**: {status}
**GitHub**: {github_url}
**Created**: {created}
**Updated**: {updated}

{brief}

---

## Review Notes

## EgoVerse Checklist

{checklist}

## Contradictions & Open Questions

## See Also

## Sources
{chr(10).join(sources)}
"""


def main():
    pr_number = int(os.environ["PR_NUMBER"])
    title = os.environ.get("PR_TITLE", "")
    author = os.environ.get("PR_AUTHOR", "")
    status = os.environ.get("PR_STATUS", "open")
    created_at = os.environ.get("PR_CREATED_AT", datetime.utcnow().isoformat())
    body = os.environ.get("PR_BODY", "") or ""
    vault_repo = os.environ["OBSIDIAN_VAULT_REPO"]
    pat = os.environ["VAULT_PAT"]

    headers = vault_headers(pat)
    raw_path = f"raw/prs/pr-{pr_number:04d}.json"
    wiki_path = f"wiki/prs/pr-{pr_number:04d}.md"

    # 1. Raw metadata (all events)
    push_file(
        vault_repo, raw_path,
        build_raw_metadata(pr_number, title, author, status, created_at, body),
        f"raw: PR #{pr_number} metadata [{status}] — {title}",
        headers, get_sha(vault_repo, raw_path, headers),
    )

    # 2. Clean wiki stub
    push_file(
        vault_repo, wiki_path,
        build_wiki_page(pr_number, title, author, status, created_at, body),
        f"wiki: PR #{pr_number} [{status}] — {title}",
        headers, get_sha(vault_repo, wiki_path, headers),
    )


if __name__ == "__main__":
    main()
