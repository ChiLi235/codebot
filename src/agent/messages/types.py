from __future__ import annotations
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, ClassVar, Literal


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> str:
    return datetime.now().isoformat(timespec="milliseconds")


@dataclass
class UserMessage:
    """A human turn or a tool-result turn stored in the transcript."""
    type: Literal["user"] = field(default="user", init=False)
    message: dict = field(default_factory=dict)
    uuid: str = field(default_factory=_uuid)
    parentUuid: str | None = None
    timestamp: str = field(default_factory=_now)
    isMeta: bool = False
    isCompactSummary: bool = False
    toolResult: list[dict] | None = None
    sourceToolAssistantUuid: str | None = None


@dataclass
class AssistantMessage:
    """One assistant turn (may contain text, tool_use blocks, or both)."""
    type: Literal["assistant"] = field(default="assistant", init=False)
    uuid: str = field(default_factory=_uuid)
    parentUuid: str | None = None
    timestamp: str = field(default_factory=_now)
    message: dict = field(default_factory=dict)
    apierror: str | None = None
    error_detail: str | None = None
    # in-memory only — excluded from JSONL by serde
    inputTokens: int | None = None
    outputTokens: int | None = None

    __disk_exclude__: ClassVar[frozenset] = frozenset({"inputTokens", "outputTokens"})


@dataclass
class SystemMessage:
    """Internal boundary / event record; never sent to the model."""
    type: Literal["system"] = field(default="system", init=False)
    subtype: Literal[
        "compact_boundary",
        "microcompact_boundary",
        "snip_boundary",
    ] | None = None
    content: str = ""
    level: Literal["info", "warning", "error"] = "info"
    uuid: str = field(default_factory=_uuid)
    timestamp: str = field(default_factory=_now)
    metadata: dict[str, Any] = field(default_factory=dict)


AnyMessage = UserMessage | AssistantMessage | SystemMessage
