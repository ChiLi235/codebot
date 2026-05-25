"""Memory subsystem public API.

Layout: {cwd}/.codebot/memory/
  MEMORY.md      ← index (one line per topic file)
  cursor.json    ← last-processed message UUID
  {topic}.md     ← one file per memory topic

Submodules:
  store.py   — MemoryStore, MemoryCursor, frontmatter helpers
  prompts.py — system prompts for extraction + recall LLM calls
  tools.py   — tool specs + dispatch for the background extraction LLM
  extract.py — background extraction thread (fire_extract)
  recall.py  — pre-turn recall (recall_memories)
"""
from pathlib import Path

from agent.memory.store import MemoryStore, MemoryCursor
from agent.memory.extract import fire_extract as _fire_extract
from agent.memory.recall import recall_memories as _recall_memories


def get_memory_dir() -> Path:
    """Always project-relative: {cwd}/.codebot/memory."""
    return Path.cwd() / ".codebot" / "memory"


def fire_extract(messages: list, cursor: MemoryCursor) -> None:
    """Snapshot conversation, advance cursor, fire background daemon thread."""
    _fire_extract(messages, cursor, get_memory_dir())


def recall_memories(query: str) -> list[str]:
    """Return up to 5 memory file contents relevant to query."""
    return _recall_memories(query, get_memory_dir())


__all__ = [
    "get_memory_dir",
    "fire_extract",
    "recall_memories",
    "MemoryStore",
    "MemoryCursor",
]
