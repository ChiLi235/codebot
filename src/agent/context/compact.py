"""Compact: summarize old context via LLM when token budget is exceeded.

Post-compact mutableMessages layout:
  SystemMessage(compact_boundary)
  UserMessage(isCompactSummary=True)   ← LLM summary + last N turns verbatim
  UserMessage(current query)           ← real query, merged with above by normalizeMessagesForAPI

If the post-compact estimate is still over threshold, reduce keep_turns and
retry until keep_turns reaches 0 (summary only). Summary is capped at
MAX_SUMMARY_CHARS to avoid runaway output.

Token estimate: use inputTokens from last AssistantMessage; fall back to char/4.
Summary model: config.COMPACT_MODEL (default "haiku"), configurable.
"""
from __future__ import annotations

from agent import config
from agent.messages.transcript import recordTranscript
from agent.messages.types import AnyMessage, AssistantMessage, SystemMessage, UserMessage

MAX_SUMMARY_CHARS = 40_000

_SUMMARY_SYSTEM = """\
You are summarizing a conversation to reduce context size.
Produce a structured summary using exactly these 9 sections.
Be specific and complete — future conversation depends solely on this summary.

## 1. Primary Request and Intent
What the user is trying to accomplish overall.

## 2. Key Technical Concepts
Languages, frameworks, APIs, patterns, constraints in play.

## 3. Files and Code Sections
Every file touched or discussed, with full relevant code snippets.

## 4. Errors and Fixes
Every error encountered, root cause, and the fix applied.

## 5. Problem Solving
Decisions made, approaches tried, trade-offs considered.

## 6. All User Messages (verbatim)
Every human message in order, word-for-word.

## 7. Pending Tasks
Work that was planned but not yet done.

## 8. Current Work
The task in progress at the moment of compaction.

## 9. Optional Next Step
The single most logical next action, if obvious.\
"""


def _estimate_tokens(mutableMessages: list[AnyMessage]) -> int:
    total = 0
    for msg in mutableMessages:
        if isinstance(msg, (UserMessage, AssistantMessage)):
            content = msg.message.get("content", [])
            if isinstance(content, list):
                for block in content:
                    total += len(block.get("text", ""))
    return total // 4


def _last_input_tokens(mutableMessages: list[AnyMessage]) -> int | None:
    for msg in reversed(mutableMessages):
        if isinstance(msg, AssistantMessage) and msg.inputTokens is not None:
            return msg.inputTokens
    return None


def current_token_estimate(mutableMessages: list[AnyMessage]) -> int:
    known = _last_input_tokens(mutableMessages)
    return known if known is not None else _estimate_tokens(mutableMessages)


def _find_turn_boundaries(mutableMessages: list[AnyMessage]) -> list[int]:
    return [
        i for i, m in enumerate(mutableMessages)
        if isinstance(m, UserMessage) and not m.isMeta and not m.toolResult
    ]


def _format_messages_as_text(messages: list[AnyMessage]) -> str:
    lines: list[str] = []
    for msg in messages:
        if isinstance(msg, UserMessage):
            if msg.isMeta or msg.isCompactSummary:
                continue
            if msg.toolResult:
                for tr in msg.toolResult:
                    content = "".join(
                        b.get("text", "") for b in tr.get("content", [])
                        if isinstance(b, dict)
                    )
                    lines.append(f"[tool_result {tr.get('toolUseId','')}]: {content[:500]}")
            else:
                content = msg.message.get("content", [])
                text = "".join(
                    b.get("text", "") for b in content
                    if isinstance(b, dict) and "text" in b
                )
                lines.append(f"User: {text}")
        elif isinstance(msg, AssistantMessage) and not msg.apierror:
            parts: list[str] = []
            for block in msg.message.get("content", []):
                if "text" in block:
                    parts.append(block["text"])
                elif "toolUse" in block:
                    tu = block["toolUse"]
                    parts.append(f"[tool_use {tu.get('name','')}]")
            if parts:
                lines.append(f"Assistant: {''.join(parts)}")
    return "\n".join(lines)


def _call_summary_llm(messages_text: str, model_key: str) -> str:
    """Call LLM to summarize. Retries up to 2 times if model hits max_tokens (cut off)."""
    from agent import client
    model_id = config.AVAILABLE_MODELS.get(model_key)
    if not model_id:
        return f"[Summary unavailable — model '{model_key}' not found]\n\n{messages_text[:2000]}"

    token_target = config.MAX_COMPACT_OUTPUT_TOKENS
    system_text  = _SUMMARY_SYSTEM + (
        f"\n\nTarget length: under {token_target} tokens. "
        "Be concise while preserving all critical technical detail."
    )

    current_input = messages_text
    max_attempts  = 3  # initial + 2 retries on cut-off

    for attempt in range(max_attempts):
        try:
            response   = client.converse(
                messages=[{"role": "user", "content": [{"text": current_input}]}],
                tools=[],
                system=[{"text": system_text}],
                model_id=model_id,
                max_tokens=token_target,
                thinking="disabled",
            )
            stop_reason = response.get("stopReason", "end_turn")
            output      = response.get("output", {}).get("message", {})
            parts       = [b.get("text", "") for b in output.get("content", []) if "text" in b]
            summary     = "".join(parts)

            if stop_reason != "max_tokens":
                return summary  # clean finish

            # model was cut off — retry with stricter instruction
            if attempt < max_attempts - 1:
                current_input = (
                    f"Your previous summary was cut off at {token_target} tokens. "
                    f"Rewrite much more concisely, completing all 9 sections within the limit.\n\n"
                    + messages_text
                )
            else:
                return summary + "\n[summary cut off — token limit reached after retries]"

        except Exception as e:
            return f"[Summary failed: {e}]\n\n{messages_text[:2000]}"

    return messages_text[:token_target * 4]


def _build_compact(
    session_id: int,
    mutableMessages: list[AnyMessage],
    keep_turns: int,
    pre_tokens: int,
    model_key: str,
) -> list[AnyMessage]:
    """Build compacted message list for a given keep_turns value."""
    turn_positions = _find_turn_boundaries(mutableMessages)

    if keep_turns > 0 and len(turn_positions) > keep_turns:
        keep_start = turn_positions[-keep_turns]
    else:
        keep_start = len(mutableMessages)  # summarize everything

    to_summarize = mutableMessages[:keep_start]
    to_keep      = mutableMessages[keep_start:]

    summary_input  = _format_messages_as_text(to_summarize)
    last_turns_text = _format_messages_as_text(to_keep) if to_keep else ""

    summary = _call_summary_llm(summary_input, model_key)
    combined = summary
    if last_turns_text:
        combined = f"{summary}\n\n=== Recent conversation ===\n{last_turns_text}"

    boundary = SystemMessage(
        subtype="compact_boundary",
        content=f"Compact: summarized {len(to_summarize)} messages, kept {len(to_keep)}",
        metadata={
            "preTokens": pre_tokens,
            "summarizedCount": len(to_summarize),
            "keptCount": len(to_keep),
            "compactModel": model_key,
            "keepTurns": keep_turns,
        },
    )
    summary_msg = UserMessage(
        message={"role": "user", "content": [{"text": combined}]},
        isCompactSummary=True,
    )
    return [boundary, summary_msg] + list(to_keep)


def compact_if_needed(
    session_id: int,
    mutableMessages: list[AnyMessage],
    compact_model: str | None = None,
    force: bool = False,
) -> bool:
    """Compact if token threshold exceeded or forced. Returns True if compaction ran.

    Iteratively reduces keep_turns until post-compact estimate is under threshold
    or keep_turns reaches 0 (summary-only fallback).
    """
    pre_tokens = current_token_estimate(mutableMessages)
    if not force and pre_tokens < config.COMPACT_TOKEN_THRESHOLD:
        return False

    model_key  = compact_model or config.COMPACT_MODEL
    keep_turns = config.COMPACT_KEEP_TURNS

    new_messages: list[AnyMessage] = []

    while keep_turns >= 0:
        new_messages = _build_compact(
            session_id, mutableMessages, keep_turns, pre_tokens, model_key
        )
        post_estimate = _estimate_tokens(new_messages)
        if post_estimate < config.COMPACT_TOKEN_THRESHOLD:
            break
        if keep_turns == 0:
            # can't reduce further — accept the result
            break
        keep_turns -= 1

    from agent import ui
    ui.console.print("[dim]compacting...[/dim]")

    # record boundary + summary to JSONL
    for msg in new_messages:
        if isinstance(msg, (SystemMessage, UserMessage)) and (
            getattr(msg, "isCompactSummary", False)
            or (isinstance(msg, SystemMessage) and msg.subtype == "compact_boundary")
        ):
            recordTranscript(session_id, msg)

    mutableMessages.clear()
    mutableMessages.extend(new_messages)
    return True
