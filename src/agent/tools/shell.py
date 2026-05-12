import os
import re
import subprocess
from pathlib import Path

# ── command category sets ──────────────────────────────────────────────────────
SEARCH_COMMANDS = {'find', 'grep', 'rg', 'ag', 'ack', 'locate', 'which', 'whereis'}
READ_COMMANDS   = {'cat', 'head', 'tail', 'less', 'more',
                   'wc', 'stat', 'file', 'strings',
                   'jq', 'awk', 'cut', 'sort', 'uniq', 'tr'}
LIST_COMMANDS   = {'ls', 'tree', 'du'}
NEUTRAL_COMMANDS = {'echo', 'printf', 'true', 'false', ':'}
SILENT_COMMANDS  = {'mv', 'cp', 'rm', 'mkdir', 'rmdir', 'chmod', 'chown',
                    'chgrp', 'touch', 'ln', 'cd', 'export', 'unset', 'wait'}

# category → display label for UI
CATEGORY_LABELS = {
    "search":  "Searched",
    "read":    "Read",
    "list":    "Listed",
    "write":   "Modified",
    "silent":  "Ran",
    "unknown": "Ran",
}

_TRUNCATE_HEAD = 100
_TRUNCATE_TAIL = 100

# module-level cwd — persists across bash() calls within one session
_cwd: str = os.getcwd()


def get_cwd() -> str:
    return _cwd


def _first_command(cmd: str) -> str:
    """Extract the first meaningful command token from a shell string."""
    # strip leading env-var assignments (FOO=bar cmd)
    cmd = cmd.strip()
    cmd = re.sub(r'^(\w+=\S+\s+)+', '', cmd)
    # skip sudo/env
    tokens = cmd.split()
    i = 0
    while i < len(tokens) and tokens[i] in {'sudo', 'env', 'time', 'nice', 'nohup', 'stdbuf'}:
        i += 1
    return tokens[i] if i < len(tokens) else ""


def categorize(command: str) -> str:
    """Return category string for a shell command."""
    # check each subcommand in a pipeline / compound
    subcmds = re.split(r'[|&;]', command)
    non_neutral = []
    for sub in subcmds:
        first = _first_command(sub)
        if first and first not in NEUTRAL_COMMANDS:
            non_neutral.append(first)

    if not non_neutral:
        return "unknown"

    # category of the first non-neutral command determines overall category
    first = non_neutral[0]
    if first in SEARCH_COMMANDS:
        return "search"
    if first in READ_COMMANDS:
        return "read"
    if first in LIST_COMMANDS:
        return "list"
    if first in SILENT_COMMANDS:
        return "silent"
    return "unknown"


def bash(command: str, timeout: int = 120) -> str:
    global _cwd

    result = subprocess.run(
        command,
        shell=True,
        executable="/bin/bash",
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=_cwd,
    )

    # detect cd — update _cwd if command was a bare cd
    if re.match(r'^\s*cd\s+', command) and result.returncode == 0:
        # ask bash where we ended up
        probe = subprocess.run(
            f"{command} && pwd",
            shell=True, executable="/bin/bash",
            capture_output=True, text=True, cwd=_cwd,
        )
        new_dir = probe.stdout.strip().splitlines()[-1] if probe.stdout.strip() else ""
        if new_dir and Path(new_dir).is_dir():
            _cwd = new_dir

    combined = result.stdout
    if result.stderr:
        combined = combined + ("\n" if combined else "") + result.stderr

    lines = combined.splitlines()
    if len(lines) > _TRUNCATE_HEAD + _TRUNCATE_TAIL:
        elided = len(lines) - _TRUNCATE_HEAD - _TRUNCATE_TAIL
        lines = (lines[:_TRUNCATE_HEAD]
                 + [f"... [{elided} lines elided] ..."]
                 + lines[-_TRUNCATE_TAIL:])

    output = "\n".join(lines).strip()
    if result.returncode != 0:
        output += f"\n[exit code {result.returncode}]"

    return output or "(no output)"
