import re
import stat
from pathlib import Path

_AGENT_DIR = Path(__file__).parent.parent.resolve()

_BLOCKED_WRITE_ROOTS = {
    p.resolve() for p in [
        Path("/dev"), Path("/sys"), Path("/proc"), Path("/etc"),
        Path("/bin"), Path("/sbin"), Path("/usr"), Path("/boot"),
    ] if p.exists()
}

_READ_ACTIONS  = {"read_file", "list_directory", "grep", "glob"}
_WRITE_ACTIONS = {"write_file", "edit_file"}

# patterns that are never safe to run
_BLOCKED_BASH_PATTERNS = [
    (r'rm\s+-[^\s]*r[^\s]*\s+/', "recursive delete from root blocked"),
    (r':\(\)\s*\{',               "fork bomb pattern blocked"),
    (r'dd\s+.*of=/dev/',          "dd to device blocked"),
    (r'>\s*/dev/sd',              "write to block device blocked"),
    (r'mkfs',                     "filesystem format blocked"),
    (r'fdisk|parted|diskutil\s+eraseDisk', "disk partition command blocked"),
    (r'curl\s+.*\|\s*(?:sudo\s+)?(?:ba)?sh', "curl|sh pattern blocked"),
    (r'wget\s+.*\|\s*(?:sudo\s+)?(?:ba)?sh', "wget|sh pattern blocked"),
]


def check_valid(action: str, path: str | None, command: str | None = None) -> str | None:
    """Return error string if blocked, None if allowed."""
    if action == "bash":
        return _check_bash(command or "")

    if not path:
        return None

    p = Path(path).expanduser()

    if action in _READ_ACTIONS:
        return _check_read(p)

    if action in _WRITE_ACTIONS:
        return _check_write(p)

    return None


def _check_bash(command: str) -> str | None:
    for pattern, reason in _BLOCKED_BASH_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            return f"Error: {reason}"
    return None


def _check_read(p: Path) -> str | None:
    if not p.exists():
        return None  # let the tool produce the "does not exist" error

    try:
        mode = p.stat().st_mode
        if stat.S_ISBLK(mode) or stat.S_ISCHR(mode) or stat.S_ISFIFO(mode) or stat.S_ISSOCK(mode):
            return f"Error: {p} is a device/special file — reading blocked"
    except OSError:
        pass

    if p.is_file() and _is_binary(p):
        return f"Error: {p} appears to be a binary file — reading blocked"

    return None


def _check_write(p: Path) -> str | None:
    resolved = p.resolve() if p.exists() else Path(str(p)).absolute()

    # block agent source
    try:
        resolved.relative_to(_AGENT_DIR)
        return f"Error: writes to agent source directory are blocked"
    except ValueError:
        pass

    # block .git anywhere
    if ".git" in resolved.parts:
        return "Error: writes to .git directories are blocked"

    # block system roots
    for root in _BLOCKED_WRITE_ROOTS:
        try:
            resolved.relative_to(root)
            return f"Error: writes to {root} are blocked"
        except ValueError:
            pass

    return None


def _is_binary(p: Path) -> bool:
    try:
        return b"\x00" in p.read_bytes()[:8192]
    except OSError:
        return False
