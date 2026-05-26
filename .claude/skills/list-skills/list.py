#!/usr/bin/env python3
"""List project + user skills with their frontmatter description."""

import os
import subprocess
import sys
from pathlib import Path


def parse_frontmatter(skill_md: Path) -> dict:
    """Return dict with at least 'name' and 'description' keys (may be empty)."""
    out = {"name": "", "description": ""}
    try:
        text = skill_md.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out
    if not text.startswith("---"):
        return out
    end = text.find("\n---", 3)
    if end < 0:
        return out
    body = text[3:end].strip("\n")
    current_key = None
    current_val_parts: list[str] = []
    for line in body.splitlines():
        if line and not line.startswith((" ", "\t")) and ":" in line:
            # Flush previous key
            if current_key in ("name", "description"):
                out[current_key] = " ".join(current_val_parts).strip()
            k, _, v = line.partition(":")
            current_key = k.strip()
            current_val_parts = [v.strip()]
        else:
            current_val_parts.append(line.strip())
    if current_key in ("name", "description"):
        out[current_key] = " ".join(current_val_parts).strip()
    return out


def find_repo_root() -> Path:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, check=True,
        )
        return Path(r.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return Path.cwd()


def list_skills(skills_dir: Path) -> list[tuple[str, str, Path]]:
    """Return [(name, description, skill_md_path), ...] sorted by name."""
    out = []
    if not skills_dir.is_dir():
        return out
    for entry in sorted(skills_dir.iterdir()):
        skill_md = entry / "SKILL.md"
        if skill_md.is_file():
            fm = parse_frontmatter(skill_md)
            name = fm["name"] or entry.name
            out.append((name, fm["description"], skill_md))
    return out


def print_section(title: str, src: Path, skills: list):
    print(f"## {title}")
    print(f"   {src}")
    print()
    if not skills:
        print("   (none)")
        print()
        return
    name_w = max(len(n) for n, _d, _p in skills)
    for name, desc, _ in skills:
        if desc:
            print(f"   - {name.ljust(name_w)}  {desc}")
        else:
            print(f"   - {name.ljust(name_w)}  (no description)")
    print()


def main():
    repo = find_repo_root()
    project_dir = repo / ".claude" / "skills"
    user_dir = Path.home() / ".claude" / "skills"

    project_skills = list_skills(project_dir)
    user_skills = list_skills(user_dir)

    print_section(
        f"Project skills ({len(project_skills)})", project_dir, project_skills
    )
    print_section(
        f"User skills ({len(user_skills)})", user_dir, user_skills
    )


if __name__ == "__main__":
    main()
