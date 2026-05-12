"""Tool + skill + subagent context injection.

Tools — formatted once at startup, in system prompt.
Skills — listing in system prompt; bodies loaded on demand via load_skill tool, or pre-injected for subagents.
Subagents — name+description listed in system prompt so model knows what's spawnable.
"""
from dataclasses import dataclass, field
from pathlib import Path

SKILL_CHAR_LIMIT = 4_000
SKILLS_DIRNAME = "skills"

_AGENT_SKILLS_DIR = Path(__file__).parent / SKILLS_DIRNAME


@dataclass
class SkillSpec:
    name: str
    description: str
    when_to_use: str
    body: str
    source: str  # "global" | "local"


# ── tools ─────────────────────────────────────────────────────────────────────

def format_tools_text(tools: list) -> str:
    lines = ["# Available tools",
             "Call by emitting structured tool_use blocks. Each tool below shows: name(params) — description."]
    for t in tools:
        spec = t["toolSpec"]
        name = spec["name"]
        desc = spec["description"]
        schema = spec["inputSchema"]["json"]
        props = schema.get("properties", {})
        required = set(schema.get("required", []))
        params = []
        for pname, pinfo in props.items():
            ptype = pinfo.get("type", "any")
            mark = "" if pname in required else "?"
            params.append(f"{pname}{mark}:{ptype}")
        lines.append(f"- `{name}({', '.join(params)})` — {desc}")
    return "\n".join(lines)


# ── skills ────────────────────────────────────────────────────────────────────

def parse_skill_md(text: str, source: str, fallback_name: str) -> SkillSpec:
    """Parse a skill .md. Frontmatter optional; backward compatible with bare-body files."""
    name = fallback_name
    description = ""
    when_to_use = ""
    body = text.strip()

    if body.startswith("---"):
        parts = body.split("---", 2)
        if len(parts) >= 3:
            fm_block = parts[1].strip()
            body = parts[2].strip()
            for line in fm_block.splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    k, v = k.strip(), v.strip()
                    if k == "name" and v:
                        name = v
                    elif k == "description":
                        description = v
                    elif k == "when_to_use":
                        when_to_use = v

    if not description:
        for line in body.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                description = line[:120]
                break

    if len(body) > SKILL_CHAR_LIMIT:
        body = body[:SKILL_CHAR_LIMIT] + "\n... [truncated]"

    return SkillSpec(name=name, description=description, when_to_use=when_to_use,
                     body=body, source=source)


def _skill_dirs() -> list[tuple[Path, str]]:
    """Return (dir, source) pairs in lookup order: cwd first, then agent dir."""
    out = []
    local = Path.cwd() / ".codebot" / SKILLS_DIRNAME
    if local.is_dir() and local.resolve() != _AGENT_SKILLS_DIR.resolve():
        out.append((local, "local"))
    if _AGENT_SKILLS_DIR.is_dir():
        out.append((_AGENT_SKILLS_DIR, "global"))
    return out


def load_skill(name: str) -> str | None:
    """Return body of single skill. cwd/skills wins over agent/skills."""
    for d, source in _skill_dirs():
        f = d / f"{name}.md"
        if f.is_file():
            spec = parse_skill_md(f.read_text(errors="replace"), source, name)
            return spec.body
    return None


def load_skills(names: list[str]) -> dict[str, str]:
    """Bulk loader. Skips missing names silently."""
    out = {}
    for n in names:
        body = load_skill(n)
        if body is not None:
            out[n] = body
    return out


def scan_skills_meta() -> dict[str, SkillSpec]:
    """All skills with metadata. cwd entries override agent entries by stem."""
    out: dict[str, SkillSpec] = {}
    # iterate agent first, then local — local overwrites
    for d, source in reversed(_skill_dirs()):
        for f in sorted(d.glob("*.md")):
            spec = parse_skill_md(f.read_text(errors="replace"), source, f.stem)
            out[spec.name] = spec
    return out


def scan_skills() -> dict[str, str]:
    """Backward-compat helper for diff logic. Returns {prefixed_key: body}."""
    out: dict[str, str] = {}
    for d, source in _skill_dirs():
        for f in sorted(d.glob("*.md")):
            spec = parse_skill_md(f.read_text(errors="replace"), source, f.stem)
            key = f"{source}:{spec.name}"
            out[key] = spec.body
    return out


def format_skills_listing(meta: dict[str, SkillSpec]) -> str:
    """Bulleted listing for system prompt: name - when_to_use - description."""
    if not meta:
        return ""
    lines = ["# Available skills",
             "Use list_skills() to refresh, load_skill(name) to fetch full body."]
    for spec in meta.values():
        if spec.when_to_use:
            lines.append(f"- `{spec.name}` — {spec.when_to_use} — {spec.description}")
        else:
            lines.append(f"- `{spec.name}` — {spec.description}")
    return "\n".join(lines)


def format_skills_text(skills: dict[str, str]) -> str:
    """Full bodies. Used for pre-injection (subagent skills frontmatter, diff updates)."""
    if not skills:
        return ""
    lines = ["# Available skills",
             "These are extra instructions you can apply when relevant."]
    for name, content in skills.items():
        lines.append(f"\n## skill: `{name}`\n{content}")
    return "\n".join(lines)


def diff_skills(old: dict[str, str], new: dict[str, str]) -> tuple[dict[str, str], set[str], set[str]]:
    old_keys, new_keys = set(old), set(new)
    added_keys = new_keys - old_keys
    removed_keys = old_keys - new_keys
    modified_keys = {k for k in (old_keys & new_keys) if old[k] != new[k]}
    added = {k: new[k] for k in added_keys}
    return added, removed_keys, modified_keys


def format_skills_diff(added: dict[str, str], removed: set[str], modified: set[str], new_state: dict[str, str]) -> str | None:
    if not (added or removed or modified):
        return None
    parts = ["[skill state changed]"]
    for name, content in added.items():
        parts.append(f"+ ADDED skill `{name}`:\n{content}")
    for name in modified:
        parts.append(f"~ UPDATED skill `{name}`:\n{new_state[name]}")
    for name in removed:
        parts.append(f"- REMOVED skill `{name}` (no longer applies)")
    return "\n\n".join(parts)


# ── subagents ─────────────────────────────────────────────────────────────────

def scan_agents() -> dict:
    """Delegates. Lazy import — subagent imports info, avoid cycle."""
    from agent import subagent
    return subagent.load_agents()


def format_agents_text(agents: dict) -> str:
    if not agents:
        return ""
    lines = ["# Available subagents",
             "Spawn via spawn_agent(subagent_type, prompt, description). "
             "Multiple calls in one turn run in parallel."]
    for name, spec in agents.items():
        lines.append(f"- `{name}` — {spec.description}")
    return "\n".join(lines)
