---
name: list-skills
description: List all project-level and user-level skills on this machine, with their one-line descriptions parsed from each SKILL.md's frontmatter. Use when the user asks what skills are available, what skills they have, or to summarize their custom skills.
---

# List Skills

Print a concise inventory of locally-installed skills (project + user), with each skill's `description:` from its SKILL.md frontmatter.

## Procedure

Run the bundled scanner and show its output verbatim:

```bash
python /coc/flash7/zhenyang/EgoVerse/.claude/skills/list-skills/list.py
```

That's it — no confirmation, no side effects. The script is read-only.

## Scope

- **Project skills**: `<repo>/.claude/skills/*/SKILL.md` (scanner finds the repo root via `git rev-parse --show-toplevel`, falling back to CWD).
- **User skills**: `~/.claude/skills/*/SKILL.md`.

Bundled / plugin skills (loop, verify, code-review, run, init, claude-api, etc.) are **not** listed — those live outside these two directories and are managed by Claude Code itself.
