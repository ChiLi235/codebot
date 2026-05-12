import os
from pathlib import Path

CHAR_LIMIT = 40_000
PROMPT_FILENAME = "PROMPT.md"

_AGENT_DIR = Path(__file__).parent.parent


def _load(path: Path) -> str | None:
    if not path.is_file():
        return None
    text = path.read_text(errors="replace")
    if len(text) > CHAR_LIMIT:
        text = text[:CHAR_LIMIT]
    return text.strip()


def build_system(tools: list | None = None) -> list[dict]:
    """Return Bedrock-formatted system list. Includes PROMPT.md, tool ref, skill listing, agents listing."""
    from agent import info as _info

    parts: list[str] = []

    system_prompt = _load(_AGENT_DIR / PROMPT_FILENAME)
    if system_prompt:
        parts.append(system_prompt)

    cwd_prompt = _load(Path(os.getcwd()) / PROMPT_FILENAME)
    if cwd_prompt:
        parts.append(cwd_prompt)

    if not parts:
        parts.append("You are a helpful coding assistant operating inside the user's local project folder. If the user asks about a file, automatically use your tools to read it from the current directory. Do not ask the user to paste file contents or provide paths unless your tool search yields no results.")

    if tools:
        parts.append(_info.format_tools_text(tools))

    skills_listing = _info.format_skills_listing(_info.scan_skills_meta())
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
