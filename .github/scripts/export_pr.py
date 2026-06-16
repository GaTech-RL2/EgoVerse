#!/usr/bin/env python3
"""
On PR merge: exports full diff + review comments to raw/prs/pr-NNNN-diff.json
and pushes directly to the Obsidian vault via GitHub Contents API.

Copy to: EgoVerse/.github/scripts/export_pr.py
"""

import base64
import json
import os
import subprocess
from datetime import datetime

import requests

VAULT_API = "https://api.github.com/repos/{repo}/contents/{path}"


def gh_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}


def vault_headers(pat: str) -> dict:
    return {
        "Authorization": f"Bearer {pat}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def get_pr_comments(repo: str, pr_number: int, token: str) -> list[dict]:
    url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}/comments"
    resp = requests.get(url, headers=gh_headers(token))
    resp.raise_for_status()
    return [
        {"author": c["user"]["login"], "body": c["body"],
         "path": c["path"], "line": c.get("line"), "created_at": c["created_at"]}
        for c in resp.json()
    ]


def get_pr_reviews(repo: str, pr_number: int, token: str) -> list[dict]:
    url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}/reviews"
    resp = requests.get(url, headers=gh_headers(token))
    resp.raise_for_status()
    return [
        {"author": r["user"]["login"], "state": r["state"],
         "body": r["body"], "submitted_at": r["submitted_at"]}
        for r in resp.json()
    ]


def push_to_vault(repo: str, path: str, content: str, message: str, pat: str):
    headers = vault_headers(pat)
    url = VAULT_API.format(repo=repo, path=path)
    sha = None
    resp = requests.get(url, headers=headers)
    if resp.status_code == 200:
        sha = resp.json()["sha"]
    payload = {
        "message": message,
        "content": base64.b64encode(content.encode()).decode(),
        "branch": "main",
    }
    if sha:
        payload["sha"] = sha
    resp = requests.put(url, headers=headers, json=payload)
    resp.raise_for_status()
    print(f"Pushed: {path}")


def main():
    pr_number = int(os.environ["PR_NUMBER"])
    base_sha = os.environ["BASE_SHA"]
    head_sha = os.environ["HEAD_SHA"]
    token = os.environ["GH_TOKEN"]
    vault_pat = os.environ["VAULT_PAT"]
    vault_repo = os.environ["OBSIDIAN_VAULT_REPO"]
    repo = os.environ.get("GITHUB_REPOSITORY", "GaTech-RL2/EgoVerse")

    diff = subprocess.check_output(
        ["git", "diff", f"{base_sha}...{head_sha}"], text=True
    )
    if len(diff) > 200_000:
        diff = diff[:200_000] + "\n... [truncated]"

    export = {
        "pr_number": pr_number,
        "title": os.environ.get("PR_TITLE", ""),
        "body": os.environ.get("PR_BODY", ""),
        "author": os.environ.get("PR_AUTHOR", ""),
        "merged_at": os.environ.get("PR_MERGED_AT", ""),
        "base_ref": os.environ.get("BASE_REF", "main"),
        "head_sha": head_sha,
        "base_sha": base_sha,
        "repo": repo,
        "diff": diff,
        "review_comments": get_pr_comments(repo, pr_number, token),
        "reviews": get_pr_reviews(repo, pr_number, token),
        "exported_at": datetime.utcnow().isoformat(),
    }

    path = f"raw/prs/pr-{pr_number:04d}-diff.json"
    push_to_vault(
        vault_repo, path,
        json.dumps(export, indent=2),
        f"raw: export merged PR #{pr_number} diff — {export['title']}",
        vault_pat,
    )


if __name__ == "__main__":
    main()
