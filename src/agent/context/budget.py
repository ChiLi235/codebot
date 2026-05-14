"""applytoolresultbudget: offload large tool results to disk, replace with preview.

Per-result limit : DEFAULT_MAX_RESULT_SIZE_CHARS (50 000)
Per-message limit: PER_MSG_RESULT_BUDGET_CHARS   (200 000)

Decision map {tool_use_id: "keep"|"offload"} is kept in memory and passed
across turns so already-decided results are not re-evaluated.
"""
from __future__ import annotations

from agent import config
from agent.messages.transcript import offload_dir
from agent.messages.types import AnyMessage, UserMessage

_PREVIEW_SENTINEL = "<preview:"


def _get_content(tr: dict) -> str:
    blocks = tr.get("content", [])
    if isinstance(blocks, list):
        return "".join(b.get("text", "") for b in blocks if isinstance(b, dict))
    return str(blocks)


def _set_content(tr: dict, text: str) -> None:
    tr["content"] = [{"text": text}]


def _is_preview(content: str) -> bool:
    return content.startswith(_PREVIEW_SENTINEL)


def _make_preview(content: str, tool_use_id: str, session_id: int) -> str:
    raw = content.encode("utf-8")
    if len(raw) <= 2000:
        snippet = content
        has_more = False
    else:
        chunk = raw[:2000]
        nl_pos = chunk.rfind(b"\n")
        if nl_pos > 1000:
            snippet = chunk[:nl_pos].decode("utf-8", errors="replace")
        else:
            snippet = chunk.decode("utf-8", errors="replace")
        has_more = True

    rel_path = f"session-{session_id}/{tool_use_id}.txt"
    suffix = f"\n...\n[full content at: {rel_path}]" if has_more else f"\n[full content at: {rel_path}]"
    return f"{_PREVIEW_SENTINEL} {snippet}{suffix}>"


def _offload(tool_use_id: str, content: str, session_id: int) -> str:
    od = offload_dir(session_id)
    od.mkdir(parents=True, exist_ok=True)
    (od / f"{tool_use_id}.txt").write_text(content, encoding="utf-8")
    return _make_preview(content, tool_use_id, session_id)


def applytoolresultbudget(
    session_id: int,
    mutableMessages: list[AnyMessage],
    decision_map: dict[str, str],
) -> None:
    """Mutate mutableMessages in-place. Offloads results exceeding thresholds."""
    for msg in mutableMessages:
        if not isinstance(msg, UserMessage) or not msg.toolResult:
            continue

        tr_list = msg.toolResult

        # per-result threshold
        for tr in tr_list:
            tool_use_id = tr.get("toolUseId", "")
            if not tool_use_id or tool_use_id in decision_map:
                continue
            content = _get_content(tr)
            if len(content) > config.DEFAULT_MAX_RESULT_SIZE_CHARS:
                preview = _offload(tool_use_id, content, session_id)
                _set_content(tr, preview)
                decision_map[tool_use_id] = "offload"
            else:
                decision_map[tool_use_id] = "keep"

        # per-message total threshold: offload largest until under budget
        total = sum(len(_get_content(tr)) for tr in tr_list)
        if total <= config.PER_MSG_RESULT_BUDGET_CHARS:
            continue

        candidates = [
            tr for tr in tr_list
            if not _is_preview(_get_content(tr))
        ]
        candidates.sort(key=lambda t: len(_get_content(t)), reverse=True)

        for tr in candidates:
            if total <= config.PER_MSG_RESULT_BUDGET_CHARS:
                break
            tool_use_id = tr.get("toolUseId", "")
            content = _get_content(tr)
            total -= len(content)
            preview = _offload(tool_use_id, content, session_id)
            _set_content(tr, preview)
            if tool_use_id:
                decision_map[tool_use_id] = "offload"
            total += len(preview)


def rebuild_decision_map(mutableMessages: list[AnyMessage]) -> dict[str, str]:
    """Reconstruct decision_map from current mutableMessages state on session load."""
    dm: dict[str, str] = {}
    for msg in mutableMessages:
        if not isinstance(msg, UserMessage) or not msg.toolResult:
            continue
        for tr in msg.toolResult:
            tid = tr.get("toolUseId", "")
            if not tid:
                continue
            dm[tid] = "offload" if _is_preview(_get_content(tr)) else "keep"
    return dm
