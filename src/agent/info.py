"""Tool + subagent context injection.

Tools — formatted once at startup, in system prompt.
Subagents — name+description listed in system prompt so model knows what's spawnable.
"""
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
