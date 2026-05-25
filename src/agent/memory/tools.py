"""Tool specs and dispatch for the background extraction LLM.

These tools are NOT exposed to the main agent — they exist only for
the background haiku call inside extract.py.
"""
from __future__ import annotations

from agent.memory.store import MemoryStore


def make_tool_specs() -> list:
    """Return Bedrock toolSpec list for the three memory tools."""
    def _spec(name: str, desc: str, props: dict, required: list[str]) -> dict:
        return {
            "toolSpec": {
                "name": name,
                "description": desc,
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": props,
                        "required": required,
                    }
                },
            }
        }

    S = {"type": "string"}
    return [
        _spec(
            "read_memory_file",
            "Read an existing memory file by filename. Call before writing.",
            {"filename": S},
            ["filename"],
        ),
        _spec(
            "write_memory_file",
            "Create or overwrite a memory file. Use for new files or full rewrites.",
            {"filename": S, "content": S},
            ["filename", "content"],
        ),
        _spec(
            "edit_memory_file",
            "Replace old_str with new_str in a memory file. old_str must be unique.",
            {"filename": S, "old_str": S, "new_str": S},
            ["filename", "old_str", "new_str"],
        ),
        _spec(
            "delete_memory_file",
            "Delete a memory file. Use when renaming (write new file first, then delete old) "
            "or when a memory is fully superseded and no longer applies.",
            {"filename": S},
            ["filename"],
        ),
    ]


def dispatch(tu: dict, store: MemoryStore) -> str:
    """Route a tool_use block to the correct MemoryStore method."""
    name = tu.get("name")
    inp = tu.get("input", {})
    if name == "read_memory_file":
        return store.read_file(inp.get("filename", ""))
    if name == "write_memory_file":
        return store.write_file(inp.get("filename", ""), inp.get("content", ""))
    if name == "edit_memory_file":
        return store.edit_file(
            inp.get("filename", ""), inp.get("old_str", ""), inp.get("new_str", "")
        )
    if name == "delete_memory_file":
        return store.delete_file(inp.get("filename", ""))
    return f"Error: unknown tool '{name}'"
