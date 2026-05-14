"""Context pipeline: prepare mutableMessages before each outer-turn API call.

Order: applytoolresultbudget → snip → microcompact → compact_if_needed

Returns clean, normalized list[dict] for the main model.
Main model never sees snip tags or the snip tool.
Inner tool-loop API calls bypass this pipeline and call normalizeMessagesForAPI directly.
"""
from __future__ import annotations

from agent.context.budget import applytoolresultbudget
from agent.context.compact import compact_if_needed
from agent.context.microcompact import maybe_microcompact
from agent.context.snip import run_snip_pipeline_step
from agent.messages.normalize import normalizeMessagesForAPI
from agent.messages.types import AnyMessage


def prepare_query(
    session_id: int,
    mutableMessages: list[AnyMessage],
    decision_map: dict[str, str],
    model_id: str,
    compact_model: str | None = None,
) -> list[dict]:
    """Run full pipeline. Mutates mutableMessages in-place. Returns API-ready messages."""
    applytoolresultbudget(session_id, mutableMessages, decision_map)
    run_snip_pipeline_step(session_id, mutableMessages, model_id)
    maybe_microcompact(session_id, mutableMessages)
    compact_if_needed(session_id, mutableMessages, compact_model=compact_model)
    return normalizeMessagesForAPI(mutableMessages)
