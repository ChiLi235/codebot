"""Append-only JSONL transcript per session.

Layout (Option B):
  .codebot/history/
    session-1.jsonl          ← one JSON line per message
    session-1/               ← offloaded tool-result bodies (Task 3+)
      {tool_use_id}.txt
    session-2.jsonl
    session-2/
"""
from __future__ import annotations
import json
from pathlib import Path

from .types import AnyMessage, SystemMessage
from .serde import to_dict, from_dict


def _history_dir() -> Path:
    return Path.cwd() / ".codebot" / "history"


def session_jsonl(session_id: int) -> Path:
    return _history_dir() / f"session-{session_id}.jsonl"


def offload_dir(session_id: int) -> Path:
    return _history_dir() / f"session-{session_id}"


def ensure_session(session_id: int) -> None:
    _history_dir().mkdir(parents=True, exist_ok=True)


def recordTranscript(session_id: int, message: AnyMessage) -> None:
    """Append one message as a JSON line. Never mutates existing lines."""
    ensure_session(session_id)
    line = json.dumps(to_dict(message), ensure_ascii=False)
    with session_jsonl(session_id).open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_transcript(session_id: int) -> list[AnyMessage]:
    """Read messages from the JSONL file. Skips corrupt lines silently.

    If a compact_boundary exists, only messages from the last boundary onward
    are returned — pre-compact history is intentionally discarded.
    """
    path = session_jsonl(session_id)
    if not path.exists():
        return []
    messages: list[AnyMessage] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            messages.append(from_dict(json.loads(raw)))
        except Exception:
            pass

    # find last compact_boundary — discard everything before it
    last_boundary = -1
    for i, m in enumerate(messages):
        if isinstance(m, SystemMessage) and getattr(m, "subtype", None) == "compact_boundary":
            last_boundary = i

    return messages[last_boundary:] if last_boundary >= 0 else messages
