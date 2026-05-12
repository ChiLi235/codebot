"""Subagent system: load specs, resolve tools, run isolated subagent loop."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
import threading

from botocore.exceptions import ClientError, BotoCoreError

from agent import client, config, info, ui


# ── data ──────────────────────────────────────────────────────────────────────

@dataclass
class AgentSpec:
    name: str
    description: str
    prompt: str
    allowed: list[str] = field(default_factory=lambda: ["*"])
    disallowed: list[str] = field(default_factory=list)
    maxturn: int = 0
    skills: list[str] = field(default_factory=list)
    model: str | None = None
    source: str = "built-in"


_BUILTIN_DIR = Path(__file__).parent / "agents"

# active context set by main loop each turn
_active: dict = {"model_id": None, "model_key": None, "force_model_id": None}

# UI/approval serialization across parallel subagents
_ui_lock = threading.Lock()


def configure(model_id: str, model_key: str, force_model_id: str | None = None):
    _active["model_id"] = model_id
    _active["model_key"] = model_key
    _active["force_model_id"] = force_model_id


def clear_force_model():
    _active["force_model_id"] = None


# ── parser / loader ───────────────────────────────────────────────────────────

def _parse_list(value: str) -> list[str]:
    return [s.strip() for s in value.split(",") if s.strip()]


def parse_agent_md(text: str, source: str, fallback_name: str) -> AgentSpec | None:
    text = text.strip()
    if not text.startswith("---"):
        return None
    parts = text.split("---", 2)
    if len(parts) < 3:
        return None
    fm_block = parts[1].strip()
    body = parts[2].strip()

    fields_: dict = {}
    for line in fm_block.splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        fields_[k.strip()] = v.strip()

    name = fields_.get("name") or fallback_name
    description = fields_.get("description", "")
    if not name or not description or not body:
        return None

    allowed = _parse_list(fields_.get("allowed", "*")) or ["*"]
    if allowed == [""]:
        allowed = ["*"]
    disallowed = _parse_list(fields_.get("disallowed", ""))
    skills_l = _parse_list(fields_.get("skills", ""))
    model = fields_.get("model") or None

    try:
        maxturn = int(fields_.get("maxturn", "0"))
    except ValueError:
        maxturn = 0

    return AgentSpec(name=name, description=description, prompt=body,
                     allowed=allowed, disallowed=disallowed, maxturn=maxturn,
                     skills=skills_l, model=model, source=source)


def load_agents() -> dict[str, AgentSpec]:
    """Built-in first, custom overrides by name."""
    out: dict[str, AgentSpec] = {}
    if _BUILTIN_DIR.is_dir():
        for f in sorted(_BUILTIN_DIR.glob("*.md")):
            spec = parse_agent_md(f.read_text(errors="replace"), "built-in", f.stem)
            if spec:
                out[spec.name] = spec
    custom_dir = Path.cwd() / ".codebot" / "agents"
    if custom_dir.is_dir():
        for f in sorted(custom_dir.glob("*.md")):
            spec = parse_agent_md(f.read_text(errors="replace"), "custom", f.stem)
            if spec:
                out[spec.name] = spec
    return out


# ── tool resolution ───────────────────────────────────────────────────────────

def resolve_tools(spec: AgentSpec, remaining_depth: int) -> tuple[list, dict]:
    from agent.tools import TOOLS, TOOL_MAP

    available = set(TOOL_MAP.keys())
    if remaining_depth <= 0:
        available.discard("spawn_agent")
    available -= set(spec.disallowed)
    if spec.allowed != ["*"]:
        available &= set(spec.allowed)

    tools_list = [t for t in TOOLS if t["toolSpec"]["name"] in available]
    tool_map = {n: TOOL_MAP[n] for n in available}
    return tools_list, tool_map


# ── runner ────────────────────────────────────────────────────────────────────

def spawn_agent(subagent_type: str, prompt: str, description: str = "",
                _parent_depth: int | None = None) -> str:
    spec = load_agents().get(subagent_type)
    if spec is None:
        return f"Error: unknown subagent '{subagent_type}'. Use /agents to list."

    parent_depth = _parent_depth if _parent_depth is not None else 99
    effective_depth = min(spec.maxturn, parent_depth - 1)

    tools_list, tool_map = resolve_tools(spec, effective_depth)
    model_id = (_active.get("force_model_id")
                or config.AVAILABLE_MODELS.get(spec.model or "")
                or _active["model_id"])
    if not model_id:
        return "Error: no model_id resolved for subagent (subagent.configure not called?)"

    parts = [spec.prompt, info.format_tools_text(tools_list)]
    if spec.skills:
        injected = info.load_skills(spec.skills)
        if injected:
            parts.append(info.format_skills_text(injected))
    if "list_skills" in tool_map:
        idx = info.format_skills_listing(info.scan_skills_meta())
        if idx:
            parts.append(idx)
    system = [{"text": "\n\n---\n\n".join(parts)}]

    messages = [{"role": "user", "content": [{"text": prompt}]}]
    return _run_subagent_loop(messages, tools_list, tool_map, system, model_id,
                              spec.name, description, effective_depth)


def _format_api_error(e: Exception) -> tuple[str, str]:
    if isinstance(e, ClientError):
        return e.response["Error"]["Code"], e.response["Error"]["Message"]
    return type(e).__name__, str(e)


def _run_subagent_loop(messages: list, tools_list: list, tool_map: dict,
                       system: list, model_id: str, name: str,
                       description: str, effective_depth: int) -> str:
    from agent.tools.guard import check_valid
    from agent.tools.human_check import (
        check_human_eval, REJECT_MESSAGE, REJECT_MESSAGE_WITH_REASON_PREFIX,
    )
    from agent.tools.shell import categorize

    with _ui_lock:
        ui.print_subagent_header(name, description)

    final_text = ""
    iterations = 0
    text_buffer = ""
    tool_uses: list = []
    stop_reason = "end_turn"

    # initial call
    try:
        with _ui_lock:
            for event in client.converse_stream(messages, tools_list, system, model_id):
                if event["type"] == "text":
                    text_buffer += event["text"]
                elif event["type"] == "tool_use":
                    tool_uses.append(event["tool_use"])
                elif event["type"] == "done":
                    stop_reason = event["stop_reason"]
    except (ClientError, BotoCoreError, Exception) as e:
        code, msg = _format_api_error(e)
        return f"Error: subagent '{name}' failed: {code} {msg}"

    final_text = text_buffer

    assistant_content = []
    if text_buffer:
        assistant_content.append({"text": text_buffer})
    for tu in tool_uses:
        assistant_content.append({"toolUse": {
            "toolUseId": tu["toolUseId"], "name": tu["name"], "input": tu["input"],
        }})
    if assistant_content:
        messages.append({"role": "assistant", "content": assistant_content})

    while stop_reason == "tool_use" and tool_uses and iterations < config.SUBAGENT_MAX_ITERATIONS:
        iterations += 1
        tool_results = []
        user_denied = False

        for tu in tool_uses:
            tname = tu["name"]
            fn = tool_map.get(tname)
            if fn is None:
                result_text, status = f"Error: tool '{tname}' not available to subagent '{name}'", "error"
            else:
                guard_err = check_valid(tname, tu["input"].get("path"), tu["input"].get("command"))
                if guard_err:
                    result_text, status = guard_err, "error"
                else:
                    with _ui_lock:
                        verdict, instr = check_human_eval(tname, tu["input"])
                    if verdict == "denied":
                        result_text, status = REJECT_MESSAGE, "error"
                        user_denied = True
                    elif verdict == "instructed":
                        result_text, status = REJECT_MESSAGE_WITH_REASON_PREFIX + (instr or ""), "error"
                    else:
                        try:
                            if tname == "spawn_agent":
                                result_text = spawn_agent(_parent_depth=effective_depth, **tu["input"])
                            else:
                                result_text = fn(**tu["input"])
                            status = "success"
                        except Exception as e:
                            result_text, status = f"Error: {e}", "error"

            category = categorize(tu["input"].get("command", "")) if tname == "bash" else "unknown"
            preview = tu["input"].get("command") or tu["input"].get("path") or ""
            with _ui_lock:
                ui.print_tool_call(tname, preview[:80], category)

            tool_results.append({"toolResult": {
                "toolUseId": tu["toolUseId"], "content": [{"text": result_text}], "status": status,
            }})

        messages.append({"role": "user", "content": tool_results})

        if user_denied:
            break

        tool_uses = []
        stop_reason = "end_turn"
        text_buffer = ""

        try:
            with _ui_lock:
                for event in client.converse_stream(messages, tools_list, system, model_id):
                    if event["type"] == "text":
                        text_buffer += event["text"]
                    elif event["type"] == "tool_use":
                        tool_uses.append(event["tool_use"])
                    elif event["type"] == "done":
                        stop_reason = event["stop_reason"]
        except (ClientError, BotoCoreError, Exception) as e:
            code, msg = _format_api_error(e)
            return f"Error: subagent '{name}' failed mid-loop: {code} {msg}"

        if text_buffer:
            final_text = text_buffer

        assistant_content = []
        if text_buffer:
            assistant_content.append({"text": text_buffer})
        for tu in tool_uses:
            assistant_content.append({"toolUse": {
                "toolUseId": tu["toolUseId"], "name": tu["name"], "input": tu["input"],
            }})
        if assistant_content:
            messages.append({"role": "assistant", "content": assistant_content})

    with _ui_lock:
        ui.print_subagent_done(name)

    return final_text or "(subagent returned no text)"


# ── parallel batch dispatch ───────────────────────────────────────────────────

def dispatch_spawn_batch(spawn_tool_uses: list, parent_depth: int = 99) -> dict:
    """Run multiple spawn_agent tool_uses in parallel. Returns {toolUseId: (text, status)}."""
    if not spawn_tool_uses:
        return {}

    def _run_one(tu):
        try:
            text = spawn_agent(_parent_depth=parent_depth, **tu["input"])
            status = "error" if text.startswith("Error:") else "success"
            return tu["toolUseId"], (text, status)
        except Exception as e:
            return tu["toolUseId"], (f"Error: spawn dispatch failed: {e}", "error")

    if len(spawn_tool_uses) == 1:
        tuid, result = _run_one(spawn_tool_uses[0])
        return {tuid: result}

    workers = min(len(spawn_tool_uses), config.MAX_PARALLEL_SUBAGENTS)
    out: dict = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for tuid, result in ex.map(_run_one, spawn_tool_uses):
            out[tuid] = result
    return out
