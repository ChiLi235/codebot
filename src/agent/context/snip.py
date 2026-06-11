"""Snip: pipeline-internal context pruning via a dedicated LLM call.

Snip is NOT exposed to the main model. A separate haiku call reviews the
conversation history, tags non-meta human messages with sequential IDs, and
may call snip to remove stale turns.

Range semantics: snip(from_id, to_id) removes [from_id, to_id) — inclusive
start, exclusive end. Removes the from_id UserMessage + everything after it
up to (but not including) the to_id UserMessage.
"""
from __future__ import annotations

from agent.messages.transcript import recordTranscript
from agent.messages.types import AnyMessage, AssistantMessage, SystemMessage, UserMessage


def build_snip_id_map(mutableMessages: list[AnyMessage]) -> dict[int, str]:
    """Return {snip_id: uuid} for every non-meta human UserMessage."""
    snip_map: dict[int, str] = {}
    counter = 1
    for msg in mutableMessages:
        if isinstance(msg, UserMessage) and not msg.isMeta and not msg.toolResult:
            snip_map[counter] = msg.uuid
            counter += 1
    return snip_map


def inject_snip_tags(
    mutableMessages: list[AnyMessage],
    snip_map: dict[int, str],
) -> list[AnyMessage]:
    """Return shallow copy of mutableMessages with [msg:N] prepended to tagged messages.

    Does NOT mutate originals — returns new UserMessage instances only for tagged msgs.
    """
    uuid_to_snip = {v: k for k, v in snip_map.items()}
    result: list[AnyMessage] = []
    for msg in mutableMessages:
        if isinstance(msg, UserMessage) and msg.uuid in uuid_to_snip:
            snip_id = uuid_to_snip[msg.uuid]
            content = msg.message.get("content", [])
            if isinstance(content, list) and content:
                first = content[0]
                if "text" in first:
                    tagged_content = [{"text": f"[msg:{snip_id}] {first['text']}"}] + content[1:]
                    tagged_msg = UserMessage(
                        message={"role": "user", "content": tagged_content},
                        uuid=msg.uuid,
                        parentUuid=msg.parentUuid,
                        timestamp=msg.timestamp,
                        isMeta=msg.isMeta,
                        isCompactSummary=msg.isCompactSummary,
                        toolResult=msg.toolResult,
                        sourceToolAssistantUuid=msg.sourceToolAssistantUuid,
                    )
                    result.append(tagged_msg)
                    continue
        result.append(msg)
    return result


def handle_snip(
    from_id: int,
    to_id: int,
    mutableMessages: list[AnyMessage],
    snip_map: dict[int, str],
    session_id: int,
) -> str:
    """Execute snip: remove [from_id, to_id) range from mutableMessages."""
    from_uuid = snip_map.get(from_id)
    to_uuid   = snip_map.get(to_id)

    if from_uuid is None:
        return f"Error: snip_id {from_id} not found in current context"

    from_pos = next(
        (i for i, m in enumerate(mutableMessages)
         if isinstance(m, UserMessage) and m.uuid == from_uuid),
        None,
    )
    if from_pos is None:
        return f"Error: message {from_id} not found in mutableMessages"

    if to_uuid is not None:
        to_pos = next(
            (i for i, m in enumerate(mutableMessages)
             if isinstance(m, UserMessage) and m.uuid == to_uuid),
            len(mutableMessages),
        )
    else:
        to_pos = len(mutableMessages)

    removed = mutableMessages[from_pos:to_pos]
    del mutableMessages[from_pos:to_pos]

    removed_uuids = [m.uuid for m in removed if isinstance(m, (UserMessage, AssistantMessage))]
    boundary = SystemMessage(
        subtype="snip_boundary",
        content=f"Snipped {len(removed)} messages (snip_id {from_id}–{to_id - 1})",
        metadata={"removedUuids": removed_uuids, "fromSnipId": from_id, "toSnipId": to_id},
    )
    mutableMessages.insert(from_pos, boundary)
    recordTranscript(session_id, boundary)

    return f"Removed {len(removed)} messages (snip_id {from_id} to {to_id - 1} inclusive)"


# ── pipeline step ──────────────────────────────────────────────────────────────

_SNIP_TOOL = {
    "toolSpec": {
        "name": "snip",
        "description": (
            "Remove old conversation turns that are no longer relevant. "
            "Human messages are labelled [msg:N]. "
            "snip(from_id, to_id) removes turns [from_id, to_id) — inclusive start, exclusive end."
        ),
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "from_id": {"type": "integer"},
                    "to_id":   {"type": "integer"},
                },
                "required": ["from_id", "to_id"],
            }
        },
    }
}

_SNIP_SYSTEM = (
    "You are a context manager. Review the tagged conversation history. "
    "If there are old turns that are clearly no longer relevant to the current work, "
    "call snip to remove them. Be conservative — only prune if confident. "
    "If nothing needs pruning, respond with plain text only (do not call snip)."
)


def run_snip_pipeline_step(
    session_id: int,
    mutableMessages: list[AnyMessage],
    model_id: str,
) -> None:
    """Make a dedicated model call to prune stale history. Mutates mutableMessages."""
    from agent import client, config
    from agent.messages.normalize import normalizeMessagesForAPI

    token_estimate = sum(
        len(b.get("text", ""))
        for msg in mutableMessages
        if isinstance(msg, UserMessage)
        for block in (
            msg.message.get("content", [])
            if isinstance(msg.message.get("content"), list) else []
        )
        for b in [block]
    ) // 4

    if token_estimate < config.SNIP_TOKEN_THRESHOLD:
        return

    snip_map = build_snip_id_map(mutableMessages)
    if len(snip_map) < 3:
        return  # too few turns to be worth pruning

    tagged   = inject_snip_tags(mutableMessages, snip_map)
    api_msgs = normalizeMessagesForAPI(tagged)

    snip_model_id = config.AVAILABLE_MODELS.get(config.SNIP_MODEL, model_id)

    try:
        response = client.converse(
            messages=api_msgs,
            tools=[_SNIP_TOOL],
            system=[{"text": _SNIP_SYSTEM}],
            model_id=snip_model_id,
            thinking="disabled",
        )
        output  = response.get("output", {}).get("message", {})
        content = output.get("content", [])
        for block in content:
            tu = block.get("toolUse")
            if not tu or tu.get("name") != "snip":
                continue
            inp     = tu.get("input", {})
            from_id = inp.get("from_id")
            to_id   = inp.get("to_id")
            if from_id is not None and to_id is not None:
                handle_snip(int(from_id), int(to_id), mutableMessages, snip_map, session_id)
                # rebuild map after pruning in case of multiple snip calls
                snip_map = build_snip_id_map(mutableMessages)
    except Exception:
        pass  # snip step failures are non-fatal
