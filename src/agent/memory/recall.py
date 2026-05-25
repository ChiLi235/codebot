"""Pre-turn memory recall: selects relevant files from MEMORY.md index."""
from __future__ import annotations

import json
import re

from agent import client, config
from agent.memory.store import MemoryStore
from agent.memory.prompts import RECALL_SYSTEM_PROMPT

RECALL_MODEL     = config.AVAILABLE_MODELS["haiku"]
RECALL_MAX_FILES = 5

_EMPTY_INDEX = {"", "# Memory", "# Memory\n\n"}


def recall_memories(query: str, memory_dir) -> list[str]:
    """Return up to 5 memory file contents relevant to query.

    Reads MEMORY.md index only — no frontmatter scanning.
    Returns [] when index is empty or LLM call fails.
    """
    store = MemoryStore(memory_dir)
    index = store.read_index()
    if not index or index.strip() in _EMPTY_INDEX:
        return []

    try:
        response = client.converse(
            messages=[{
                "role": "user",
                "content": [{"text": f"Query: {query}\n\nMEMORY.md:\n{index}"}],
            }],
            tools=[],
            system=[{"text": RECALL_SYSTEM_PROMPT}],
            model_id=RECALL_MODEL,
            max_tokens=256,
        )
    except Exception:
        return []

    out_content = response.get("output", {}).get("message", {}).get("content", [])
    raw = " ".join(b.get("text", "") for b in out_content if "text" in b).strip()

    selected: list[str] = _parse_selected(raw)
    contents = []
    for filename in selected[:RECALL_MAX_FILES]:
        content = store.read_file(filename)
        if not content.startswith("Error"):
            contents.append(content)
    return contents


def _parse_selected(raw: str) -> list[str]:
    """Extract selected_memories list from raw LLM text (handles fenced JSON)."""
    try:
        return json.loads(raw).get("selected_memories", [])
    except Exception:
        pass
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group()).get("selected_memories", [])
        except Exception:
            pass
    return []
