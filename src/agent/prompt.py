import os
from pathlib import Path

CHAR_LIMIT = 40_000
PROMPT_FILENAME = "PROMPT.md"


def _load(path: Path) -> str | None:
    if not path.is_file():
        return None
    text = path.read_text(errors="replace")
    if len(text) > CHAR_LIMIT:
        text = text[:CHAR_LIMIT]
    return text.strip() or None


def _build_default_system(cwd: Path) -> str:
    skills_path = cwd / ".codebot" / "skills"
    memory_path = cwd / ".codebot" / "memory"
    memory_index = memory_path / "MEMORY.md"
    return f"""\
You are a helpful coding assistant operating inside the user's local project folder. \
For general question, you can try to answer them with your knowledge. \
For project-specific question, read files with your tools to gather result.
Do not ask the user to paste file contents or provide paths unless your tool search yields no results.

## Paths

- Skills: {skills_path}
- Memory: {memory_path}

## Memory

Persist useful context across sessions by writing topic `.md` files to the memory directory above. \
Use the memory tools (write_file / edit_file) directly on those paths.

### File format

```
---
name: feedback_no_summaries
description: User wants no trailing response summaries
type: feedback
---

Don't add trailing summaries after responses.

**Why:** User finds them redundant.
**How to apply:** Every response after code changes.
```

Filename: `{{type}}_{{short_slug}}.md`. The `name` field equals the filename without `.md`.

### Memory index — ALWAYS update on write

`{memory_index}` is the index. One line per file, no file content:
```
- [Title](filename.md) — one-line hook
```
Whenever you write or modify a memory file, update this index in the same operation.

### Memory types

| type | what to save |
|------|-------------|
| **user** | role, expertise, preferences — who the user is |
| **feedback** | behavioral rules: corrections AND confirmed non-obvious approaches |
| **project** | ongoing work, goals, deadlines not in code or git history |
| **reference** | pointers to external systems (URLs, dashboards, tickets) |

For **feedback** and **project** memories, include a `**Why:**` line (reason) and `**How to apply:**` line (when it kicks in).

### What NOT to save

Code patterns, architecture, file paths, git history, debugging steps, fix recipes, \
anything already in CLAUDE.md, or ephemeral task state. \
Even if the user asks — save only what is genuinely non-obvious and future-relevant.\
"""


def build_system(tools: list | None = None) -> list[dict]:
    """Return Bedrock-formatted system list. Always starts with default system prompt.
    Appends cwd/PROMPT.md if present, then tool ref, skill listing, agents listing."""
    from agent import info as _info
    from agent import skills as _skills

    cwd = Path(os.getcwd())
    parts: list[str] = [_build_default_system(cwd)]

    cwd_prompt = _load(cwd / PROMPT_FILENAME)
    if cwd_prompt:
        parts.append(cwd_prompt)

    if tools:
        parts.append(_info.format_tools_text(tools))

    skills_listing = _skills.format_skills_listing(_skills.scan_skills_meta())
    if skills_listing:
        parts.append(skills_listing)

    agents_text = _info.format_agents_text(_info.scan_agents())
    if agents_text:
        parts.append(agents_text)

    full = "\n\n---\n\n".join(parts)

    # debug dump — inspect assembled system prompt in cwd
    debug_path = Path(os.getcwd()) / "system_prompt.debug.md"
    debug_path.write_text(full)
    print(f"[debug] system prompt written to {debug_path} ({len(full)} chars, {len(parts)} blocks)")

    return [{"text": full}]
