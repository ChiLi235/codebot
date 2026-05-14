"""Session ID management and metadata.

Storage format is now JSONL (see messages/transcript.py).
One file per session: .codebot/history/session-{n}.jsonl
Offloaded tool bodies: .codebot/history/session-{n}/{tool_use_id}.txt
"""
import re
from pathlib import Path

from agent.messages.transcript import (
    session_jsonl,
    load_transcript,
    _history_dir,
)

HISTORY_DIRNAME = "history"
SESSION_PREFIX = "session-"
_SESSION_RE = re.compile(rf"^{SESSION_PREFIX}(\d+)\.jsonl$")


def history_dir() -> Path:
    return _history_dir()


def ensure_history() -> Path:
    d = history_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _list_session_ids() -> list[int]:
    d = history_dir()
    if not d.is_dir():
        return []
    ids = []
    for f in d.iterdir():
        m = _SESSION_RE.match(f.name)
        if m:
            ids.append(int(m.group(1)))
    return sorted(ids)


def latest_session_id() -> int | None:
    ids = _list_session_ids()
    return ids[-1] if ids else None


def next_session_id() -> int:
    ids = _list_session_ids()
    return (ids[-1] + 1) if ids else 1


def session_path(session_id: int) -> Path:
    return session_jsonl(session_id)


def load(session_id: int) -> list:
    """Return list of AnyMessage for the session."""
    return load_transcript(session_id)


def delete(session_id: int) -> bool:
    """Remove session JSONL and its offload dir. Returns True if anything was deleted."""
    import shutil
    deleted = False
    p = session_jsonl(session_id)
    if p.exists():
        p.unlink()
        deleted = True
    od = history_dir() / f"session-{session_id}"
    if od.is_dir():
        shutil.rmtree(od)
        deleted = True
    return deleted


def list_session_ids() -> list[int]:
    return _list_session_ids()


def list_all() -> list[dict]:
    """Return session metadata for /sessions display."""
    from agent.messages.types import AssistantMessage, SystemMessage
    out = []
    for sid in _list_session_ids():
        try:
            msgs = load_transcript(sid)
            p = session_jsonl(sid)
            updated_at = p.stat().st_mtime if p.exists() else None
            if updated_at:
                from datetime import datetime
                updated_at = datetime.fromtimestamp(updated_at).isoformat(timespec="seconds")
            model_key = None
            for m in msgs:
                if isinstance(m, AssistantMessage) and not m.apierror:
                    model_key = m.message.get("model_key")
                    if model_key:
                        break
            out.append({
                "session_id": sid,
                "model_key": model_key or "?",
                "updated_at": updated_at,
                "msg_count": sum(1 for m in msgs if not isinstance(m, SystemMessage)),
            })
        except Exception:
            out.append({"session_id": sid, "error": True})
    return out
