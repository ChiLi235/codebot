import re
from pathlib import Path

from agent import ui
from agent.tools.shell import (
    SEARCH_COMMANDS, READ_COMMANDS, LIST_COMMANDS, NEUTRAL_COMMANDS,
)

REJECT_MESSAGE = (
    "The user doesn't want to proceed with this tool use. "
    "The tool use was rejected (eg. if it was a file edit, the new_string was NOT written to the file)."
)

REJECT_MESSAGE_WITH_REASON_PREFIX = (
    "The user doesn't want to proceed with this tool use. "
    "The tool use was rejected (eg. if it was a file edit, the new_string was NOT written to the file). "
    "To tell you how to proceed, the user said:\n"
)

WRITE_TOOLS = {"write_file", "edit_file"}
READ_TOOLS  = {"read_file", "list_directory", "grep"}

_SAFE_BASH = SEARCH_COMMANDS | READ_COMMANDS | LIST_COMMANDS | NEUTRAL_COMMANDS

# subcommands matching these names always require approval, even with
# auto-approve on — wrong path/flag here is irreversible
_ALWAYS_CONFIRM_BASH = {
    "rm", "rmdir", "dd", "mkfs", "shred", "truncate",
    "chmod", "chown", "kill", "pkill", "killall",
    "shutdown", "reboot", "halt", "poweroff",
}

_WRITE_REDIRECT_RE = re.compile(r'(?<![<>])(>|>>)(?!&)')

# global toggle set via /switch-access; default = always allow (TB 2.0 runs)
_auto_approve = True

# set once at startup for non-interactive runs (eg. Terminal-Bench harness) —
# bypasses ALL approval, including rm-class commands and outside-cwd reads,
# since ui.ask_approval() can't prompt without a tty
_headless = False


def get_auto_approve() -> bool:
    return _auto_approve


def set_auto_approve(value: bool) -> None:
    global _auto_approve
    _auto_approve = value


def get_headless() -> bool:
    return _headless


def set_headless(value: bool) -> None:
    global _headless
    _headless = value


def _first_token(sub: str) -> str:
    s = sub.strip()
    s = re.sub(r'^(\w+=\S+\s+)+', '', s)
    tokens = s.split()
    i = 0
    while i < len(tokens) and tokens[i] in {'sudo', 'env', 'time', 'nice', 'nohup', 'stdbuf'}:
        i += 1
    return tokens[i] if i < len(tokens) else ""


def _bash_needs_approval(cmd: str) -> str | None:
    """Approval needed unless every subcommand is read/search/list/neutral and no write redirects."""
    if _WRITE_REDIRECT_RE.search(cmd):
        return "shell write redirect"

    subcmds = [s for s in re.split(r'[|&;]', cmd) if s.strip()]
    if not subcmds:
        return "empty command"

    for sub in subcmds:
        first = _first_token(sub)
        if not first:
            continue
        if first not in _SAFE_BASH:
            return f"non-read command: {first}"

    return None


def _has_dangerous_command(cmd: str) -> bool:
    subcmds = [s for s in re.split(r'[|&;]', cmd) if s.strip()]
    return any(_first_token(s) in _ALWAYS_CONFIRM_BASH for s in subcmds)


def _is_outside_cwd(path: str) -> bool:
    cwd = Path.cwd().resolve()
    p = Path(path).expanduser()
    try:
        target = p.resolve()
    except OSError:
        return True
    try:
        target.relative_to(cwd)
        return False
    except ValueError:
        return True


def needs_approval(action: str, params: dict) -> str | None:
    """Return reason string if approval needed, None otherwise."""
    if _headless:
        return None

    if action == "bash":
        cmd = params.get("command", "")
        if _has_dangerous_command(cmd):
            return "potentially destructive command"
        reason = _bash_needs_approval(cmd)
        if reason is None:
            return None
        return None if _auto_approve else reason

    if action in WRITE_TOOLS:
        if _auto_approve:
            return None
        return f"write to {params.get('path', '?')}"

    if action in READ_TOOLS:
        path = params.get("path")
        if path and _is_outside_cwd(path):
            return f"read outside cwd: {path}"

    return None


def check_human_eval(action: str, params: dict) -> tuple[str, str | None]:
    """
    Return ('approved', None) | ('denied', None) | ('instructed', user_message).
    Blocks on user input. Caller halts conversation until this returns.
    """
    reason = needs_approval(action, params)
    if reason is None:
        return ("approved", None)

    diff_lines: list[str] = []
    diff_path: str = ""
    if action in WRITE_TOOLS:
        from agent.diff import compute_preview_diff
        diff_lines, diff_path = compute_preview_diff(action, params)

    detail = params.get("command") or params.get("path") or ""
    return ui.ask_approval(action, reason, detail, diff_lines, diff_path)
