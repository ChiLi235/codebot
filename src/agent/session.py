"""Session persistence. One JSON per session in cwd/.codebot/history/."""
import json
import re
from datetime import datetime
from pathlib import Path

HISTORY_DIRNAME = "history"
SESSION_PREFIX = "session-"
_SESSION_RE = re.compile(rf"^{SESSION_PREFIX}(\d+)\.json$")


def history_dir() -> Path:
    return Path.cwd() / ".codebot" / HISTORY_DIRNAME


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
    return history_dir() / f"{SESSION_PREFIX}{session_id}.json"


def load(session_id: int) -> dict:
    """Return parsed session dict. Raises FileNotFoundError if missing."""
    return json.loads(session_path(session_id).read_text())


def save(session_id: int, messages: list, model_key: str) -> None:
    """Write session JSON. Creates history dir if needed."""
    ensure_history()
    p = session_path(session_id)
    now = datetime.now().isoformat(timespec="seconds")

    payload = {
        "session_id": session_id,
        "model_key": model_key,
        "updated_at": now,
        "messages": messages,
    }
    if not p.exists():
        payload["created_at"] = now
    else:
        try:
            existing = json.loads(p.read_text())
            payload["created_at"] = existing.get("created_at", now)
        except Exception:
            payload["created_at"] = now

    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def delete(session_id: int) -> bool:
    """Remove session file. Returns True if file existed and was deleted."""
    p = session_path(session_id)
    if not p.exists():
        return False
    p.unlink()
    return True


def list_session_ids() -> list[int]:
    return _list_session_ids()


def list_all() -> list[dict]:
    """Return list of session metadata for /sessions or similar."""
    out = []
    for sid in _list_session_ids():
        try:
            data = load(sid)
            out.append({
                "session_id": sid,
                "model_key": data.get("model_key"),
                "updated_at": data.get("updated_at"),
                "msg_count": len(data.get("messages", [])),
            })
        except Exception:
            out.append({"session_id": sid, "error": True})
    return out
