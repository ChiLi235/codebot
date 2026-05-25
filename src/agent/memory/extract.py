"""Background memory extraction: fires after every turn, non-blocking."""
from __future__ import annotations

import threading
from pathlib import Path

from agent import client, config
from agent.messages.types import AssistantMessage, UserMessage, SystemMessage
from agent.memory.store import MemoryStore, MemoryCursor, _build_manifest
from agent.memory.prompts import EXTRACT_SYSTEM_PROMPT
from agent.memory import tools as memory_tools

# ── constants ──────────────────────────────────────────────────────────────────

EXTRACT_MODEL     = config.AVAILABLE_MODELS["haiku"]
MAX_EXTRACT_TURNS = 5
_WRITE_TOOLS      = {"write_file", "edit_file"}

# ── mutex + stash (module-level singletons) ────────────────────────────────────

_extract_lock: threading.Lock = threading.Lock()
_stash_lock:   threading.Lock = threading.Lock()
_pending_stash: dict | None   = None

# ── conversation helpers ───────────────────────────────────────────────────────

def _messages_since(messages: list, cursor_uuid: str | None) -> list:
    """Return messages after cursor_uuid. All non-system messages if cursor is None."""
    if cursor_uuid is None:
        return [m for m in messages if not isinstance(m, SystemMessage)]
    found = False
    result = []
    for m in messages:
        if found and not isinstance(m, SystemMessage):
            result.append(m)
        if hasattr(m, "uuid") and m.uuid == cursor_uuid:
            found = True
    return result


def _format_conversation(messages: list) -> str:
    """Convert message objects to readable [User]/[Assistant] text."""
    lines = []
    for m in messages:
        if isinstance(m, UserMessage):
            if m.isMeta or m.toolResult:
                continue
            content = m.message.get("content", [])
            text = " ".join(b.get("text", "") for b in content if "text" in b).strip()
            if text:
                lines.append(f"[User]: {text}")
        elif isinstance(m, AssistantMessage):
            content = m.message.get("content", [])
            text = " ".join(b.get("text", "") for b in content if "text" in b).strip()
            if text:
                lines.append(f"[Assistant]: {text}")
    return "\n\n".join(lines) if lines else "(no text content)"


def _main_agent_wrote_memory(
    messages: list, memory_dir: Path, cursor_uuid: str | None
) -> bool:
    """True if main agent used write_file/edit_file targeting memory_dir since cursor."""
    mem_resolved = memory_dir.resolve()
    found_cursor = cursor_uuid is None
    for m in messages:
        if not found_cursor:
            if hasattr(m, "uuid") and m.uuid == cursor_uuid:
                found_cursor = True
            continue
        if isinstance(m, AssistantMessage):
            for block in m.message.get("content", []):
                if "toolUse" not in block:
                    continue
                tu = block["toolUse"]
                if tu.get("name") not in _WRITE_TOOLS:
                    continue
                path_str = tu.get("input", {}).get("path", "")
                if not path_str:
                    continue
                try:
                    if Path(path_str).expanduser().resolve().is_relative_to(mem_resolved):
                        return True
                except Exception:
                    pass
    return False

# ── extraction core ────────────────────────────────────────────────────────────

def _do_extract(messages: list, cursor_uuid: str | None, memory_dir: Path) -> None:
    store = MemoryStore(memory_dir)
    store.ensure_dir()

    if _main_agent_wrote_memory(messages, memory_dir, cursor_uuid):
        return

    new_msgs = _messages_since(messages, cursor_uuid)
    if not new_msgs:
        return

    manifest  = _build_manifest(store.scan_files())
    conv_text = _format_conversation(new_msgs)
    n         = len(new_msgs)

    tool_specs = memory_tools.make_tool_specs()
    system     = [{"text": EXTRACT_SYSTEM_PROMPT}]
    llm_msgs: list = [{
        "role": "user",
        "content": [{
            "text": (
                f"Analyze these {n} new messages and update memory as needed.\n\n"
                f"## Conversation\n{conv_text}\n\n"
                f"## Existing memory files\n{manifest}"
            )
        }],
    }]

    for _ in range(MAX_EXTRACT_TURNS):
        try:
            response = client.converse(
                llm_msgs, tool_specs, system, EXTRACT_MODEL, max_tokens=1024
            )
        except Exception:
            break

        out_content = response.get("output", {}).get("message", {}).get("content", [])
        stop_reason = response.get("stopReason", "end_turn")
        llm_msgs.append({"role": "assistant", "content": out_content})

        if stop_reason != "tool_use":
            break

        tool_results = []
        for block in out_content:
            if "toolUse" not in block:
                continue
            tu     = block["toolUse"]
            result = memory_tools.dispatch(tu, store)
            status = "error" if result.startswith("Error") else "success"
            tool_results.append({
                "toolResult": {
                    "toolUseId": tu["toolUseId"],
                    "content":   [{"text": result}],
                    "status":    status,
                }
            })

        if not tool_results:
            break
        llm_msgs.append({"role": "user", "content": tool_results})


def _run_extract(messages: list, cursor_uuid: str | None, memory_dir: Path) -> None:
    global _pending_stash

    if not _extract_lock.acquire(blocking=False):
        with _stash_lock:
            _pending_stash = {
                "messages":    messages,
                "cursor_uuid": cursor_uuid,
                "memory_dir":  memory_dir,
            }
        return

    try:
        _do_extract(messages, cursor_uuid, memory_dir)
    except Exception:
        pass
    finally:
        _extract_lock.release()

    # trailing run if a newer context was stashed while we were running
    with _stash_lock:
        stash, _pending_stash = _pending_stash, None

    if stash:
        try:
            _do_extract(stash["messages"], stash["cursor_uuid"], stash["memory_dir"])
        except Exception:
            pass


# ── public ─────────────────────────────────────────────────────────────────────

def fire_extract(messages: list, cursor: MemoryCursor, memory_dir: Path) -> None:
    """Snapshot conversation, advance cursor, fire daemon thread. Non-blocking."""
    if not messages:
        return
    snapshot = list(messages)
    old_uuid = cursor.last_uuid
    cursor.last_uuid = messages[-1].uuid if hasattr(messages[-1], "uuid") else None
    cursor.save()
    threading.Thread(
        target=_run_extract,
        args=(snapshot, old_uuid, memory_dir),
        daemon=True,
    ).start()
