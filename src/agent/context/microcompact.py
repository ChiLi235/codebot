"""Microcompact: clear old bulky tool-result bodies when session goes idle > 5 min.

Keeps last MICROCOMPACT_KEEP_TURNS complete turns intact.
A "complete turn" = one non-meta human UserMessage + everything after it up
to (but not including) the next non-meta human UserMessage.

Target tools (bulky output): read_file, bash, grep, glob, edit_file, write_file.
NOT cleared: Task results, user text, assistant thinking.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from agent import config
from agent.messages.transcript import recordTranscript
from agent.messages.types import AnyMessage, SystemMessage, UserMessage

BULKY_TOOL_NAMES = {
    "read_file", "bash", "grep", "glob",
    "edit_file", "write_file",
}

_CLEARED = "[Old tool result content cleared]"


def _now() -> datetime:
    return datetime.now()


def _parse_ts(ts: str) -> datetime | None:
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def _last_user_message_time(mutableMessages: list[AnyMessage]) -> datetime | None:
    for msg in reversed(mutableMessages):
        if isinstance(msg, UserMessage):
            ts = _parse_ts(msg.timestamp)
            if ts:
                return ts
    return None


def _find_turn_boundaries(mutableMessages: list[AnyMessage]) -> list[int]:
    """Return positions (indices) of non-meta human UserMessages, in order."""
    return [
        i for i, m in enumerate(mutableMessages)
        if isinstance(m, UserMessage) and not m.isMeta and not m.toolResult
    ]


def _get_tool_name_for_result(tr: dict, mutableMessages: list[AnyMessage]) -> str | None:
    """Find the tool name for a given toolResult by matching toolUseId in assistant messages."""
    tool_use_id = tr.get("toolUseId", "")
    if not tool_use_id:
        return None
    from agent.messages.types import AssistantMessage
    for msg in mutableMessages:
        if not isinstance(msg, AssistantMessage):
            continue
        for block in msg.message.get("content", []):
            tu = block.get("toolUse")
            if tu and tu.get("toolUseId") == tool_use_id:
                return tu.get("name")
    return None


def maybe_microcompact(
    session_id: int,
    mutableMessages: list[AnyMessage],
    idle_minutes: int = 5,
) -> bool:
    """Run microcompact if conditions met. Returns True if compaction occurred."""
    last_ts = _last_user_message_time(mutableMessages)
    if last_ts is None:
        return False
    if (_now() - last_ts) < timedelta(minutes=idle_minutes):
        return False

    turn_positions = _find_turn_boundaries(mutableMessages)
    keep_turns = config.MICROCOMPACT_KEEP_TURNS

    if len(turn_positions) <= keep_turns:
        return False  # everything is within the keep window

    # messages before this index are in "old" turns
    cutoff_pos = turn_positions[-(keep_turns)]

    pre_tokens = _estimate_tokens(mutableMessages)
    compacted_ids: list[str] = []

    for msg in mutableMessages[:cutoff_pos]:
        if not isinstance(msg, UserMessage) or not msg.toolResult:
            continue
        for tr in msg.toolResult:
            tool_name = _get_tool_name_for_result(tr, mutableMessages)
            if tool_name not in BULKY_TOOL_NAMES:
                continue
            current = _get_tr_content(tr)
            if current == _CLEARED:
                continue
            tid = tr.get("toolUseId", "")
            if tid:
                compacted_ids.append(tid)
            _set_tr_content(tr, _CLEARED)

    if not compacted_ids:
        return False

    post_tokens = _estimate_tokens(mutableMessages)
    tokens_saved = pre_tokens - post_tokens

    boundary = SystemMessage(
        subtype="microcompact_boundary",
        content=f"Microcompact cleared {len(compacted_ids)} tool results",
        metadata={
            "trigger": "idle_5min",
            "preTokens": pre_tokens,
            "tokensSaved": tokens_saved,
            "compactedToolIds": compacted_ids,
        },
    )
    mutableMessages.append(boundary)
    recordTranscript(session_id, boundary)
    return True


def _get_tr_content(tr: dict) -> str:
    blocks = tr.get("content", [])
    if isinstance(blocks, list):
        return "".join(b.get("text", "") for b in blocks if isinstance(b, dict))
    return str(blocks)


def _set_tr_content(tr: dict, text: str) -> None:
    tr["content"] = [{"text": text}]


def _estimate_tokens(mutableMessages: list[AnyMessage]) -> int:
    total = 0
    for msg in mutableMessages:
        if isinstance(msg, UserMessage):
            content = msg.message.get("content", [])
            if isinstance(content, list):
                for block in content:
                    total += len(block.get("text", ""))
        from agent.messages.types import AssistantMessage
        if isinstance(msg, AssistantMessage):
            for block in msg.message.get("content", []):
                total += len(block.get("text", ""))
    return total // 4
