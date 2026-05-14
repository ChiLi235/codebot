"""Serialize / deserialize AnyMessage to/from plain dicts for JSONL storage."""
from __future__ import annotations
import dataclasses
from .types import AnyMessage, UserMessage, AssistantMessage, SystemMessage


def to_dict(msg: AnyMessage) -> dict:
    d = dataclasses.asdict(msg)
    exclude = getattr(type(msg), "__disk_exclude__", frozenset())
    for k in exclude:
        d.pop(k, None)
    return d


def from_dict(d: dict) -> AnyMessage:
    data = {k: v for k, v in d.items() if k != "type"}
    t = d.get("type")
    if t == "user":
        return UserMessage(**data)
    if t == "assistant":
        return AssistantMessage(**data)
    if t == "system":
        return SystemMessage(**data)
    raise ValueError(f"Unknown message type: {t!r}")
