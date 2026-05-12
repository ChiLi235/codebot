from agent.tools.files import read_file, write_file, edit_file, list_directory
from agent.tools.search import grep, glob_files
from agent.tools.shell import bash
from agent.tools.skill import load_skill, list_skills


def _spawn_agent(*args, **kwargs):
    """Lazy wrapper to break the subagent <-> tools import cycle."""
    from agent.subagent import spawn_agent
    return spawn_agent(*args, **kwargs)


TOOL_MAP: dict = {
    "read_file": read_file,
    "write_file": write_file,
    "edit_file": edit_file,
    "list_directory": list_directory,
    "grep": grep,
    "glob": glob_files,
    "bash": bash,
    "load_skill": load_skill,
    "list_skills": list_skills,
    "spawn_agent": _spawn_agent,
}


def _spec(name: str, sig: str, props: dict, required: list[str]) -> dict:
    return {
        "toolSpec": {
            "name": name,
            "description": sig,
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": props,
                    "required": required,
                }
            },
        }
    }


_S = {"type": "string"}
_I = {"type": "integer"}

TOOLS: list = [
    _spec("read_file",
          "read_file(path: str, start_line: int = None, end_line: int = None) -> str",
          {"path": _S, "start_line": _I, "end_line": _I},
          ["path"]),
    _spec("write_file",
          "write_file(path: str, content: str) -> str",
          {"path": _S, "content": _S},
          ["path", "content"]),
    _spec("edit_file",
          "edit_file(path: str, old_str: str, new_str: str) -> str  # old_str must be unique in file",
          {"path": _S, "old_str": _S, "new_str": _S},
          ["path", "old_str", "new_str"]),
    _spec("list_directory",
          "list_directory(path: str = '.') -> str",
          {"path": _S},
          []),
    _spec("grep",
          "grep(pattern: str, path: str = '.', glob: str = None) -> str  # regex search; returns path:line:content",
          {"pattern": _S, "path": _S, "glob": _S},
          ["pattern"]),
    _spec("glob",
          "glob(pattern: str) -> str  # e.g. '**/*.py'",
          {"pattern": _S},
          ["pattern"]),
    _spec("bash",
          "bash(command: str, timeout: int = 120) -> str  # cwd persists; output truncated at 100 head + 100 tail lines",
          {"command": _S, "timeout": _I},
          ["command"]),
    _spec("load_skill",
          "load_skill(name: str) -> str  # fetch full body of a named skill mid-conversation",
          {"name": _S},
          ["name"]),
    _spec("list_skills",
          "list_skills() -> str  # list every available skill: name - when_to_use - description",
          {},
          []),
    _spec("spawn_agent",
          "spawn_agent(subagent_type: str, prompt: str, description: str) -> str  "
          "# delegate scoped task to a fresh subagent with isolated history. "
          "Multiple calls in one turn run in parallel.",
          {"subagent_type": _S, "prompt": _S, "description": _S},
          ["subagent_type", "prompt", "description"]),
]
