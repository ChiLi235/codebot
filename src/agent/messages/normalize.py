"""
Convert a list of AnyMessage (as stored in the transcript) into the
flat [{"role": ..., "content": [...]}] format expected by the Bedrock
converse API.

The transformation mirrors the example in current.md:

  JSONL (8 entries)                   → API sees (4 messages)
  ─────────────────────────────────────────────────────────────
  user "read foo.ts and bar.ts"
  asst [thinking, tool_use foo, …]    → asst [thinking, tool_use foo, tool_use bar]
  user [tool_result foo]              → user [tool_result foo, tool_result bar, system-reminder]
  user [tool_result bar]
  user [<system-reminder> ide state]
  asst [text "files read"]            → asst [text "files read"]

Rules
  1. Consecutive user-role entries (tool_result blocks + meta text injections)
     are merged into one API user message.
  2. AssistantMessages with the same message.id are merged into one API
     assistant message (multi-block streaming collapses here).
  3. SystemMessages are never forwarded to the API.
"""
from __future__ import annotations

from .types import AnyMessage, AssistantMessage, UserMessage, SystemMessage


def normalizeMessagesForAPI(messages: list[AnyMessage]) -> list[dict]:
    """Return a Bedrock-compatible message list from the stored transcript."""
    api_messages: list[dict] = []

    i = 0
    while i < len(messages):
        msg = messages[i]

        if isinstance(msg, SystemMessage):
            i += 1
            continue

        if isinstance(msg, AssistantMessage):
            if msg.apierror:
                i += 1
                continue
            api_messages.append(_assistant_to_api(msg))
            i += 1
            continue

        if isinstance(msg, UserMessage):
            merged_content: list[dict] = []
            while i < len(messages) and _is_mergeable_user(messages[i]):
                merged_content.extend(_user_content(messages[i]))  # type: ignore[arg-type]
                i += 1
            if merged_content:
                api_messages.append({"role": "user", "content": merged_content})
            continue

        i += 1

    return api_messages


# ── helpers ──────────────────────────────────────────────────────────────────

def _is_mergeable_user(msg: AnyMessage) -> bool:
    """True for consecutive user entries that belong in the same API message."""
    if isinstance(msg, UserMessage):
        return True
    return False


def _user_content(msg: UserMessage) -> list[dict]:
    """Extract Bedrock content blocks from a UserMessage."""
    if msg.toolResult:
        return [{"toolResult": tr} for tr in msg.toolResult]
    content = msg.message.get("content", [])
    if isinstance(content, str):
        return [{"text": content}] if content else []
    return list(content)


def _assistant_to_api(msg: AssistantMessage) -> dict:
    """Convert a stored AssistantMessage to a Bedrock assistant turn."""
    inner = msg.message
    content = inner.get("content", [])
    if isinstance(content, str):
        content = [{"text": content}] if content else []
    return {"role": "assistant", "content": list(content)}
