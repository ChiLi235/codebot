from pathlib import Path

SKIP_DIRS = {".git", "__pycache__", ".venv", "node_modules", ".mypy_cache", ".ruff_cache"}
MAX_READ_LINES = 2000


def read_file(path: str, start_line: int | None = None, end_line: int | None = None) -> str:
    p = Path(path).expanduser()
    if not p.exists():
        return f"Error: {path} does not exist"
    if not p.is_file():
        return f"Error: {path} is not a file"

    lines = p.read_text(errors="replace").splitlines()
    total = len(lines)

    lo = (start_line - 1) if start_line else 0
    hi = end_line if end_line else total
    lo = max(0, lo)
    hi = min(total, hi)
    window = lines[lo:hi]

    if len(window) > MAX_READ_LINES:
        half = MAX_READ_LINES // 2
        elided = len(window) - MAX_READ_LINES
        window = window[:half] + [f"... [{elided} lines elided] ..."] + window[-half:]

    numbered = [f"{lo + i + 1}\t{line}" for i, line in enumerate(window)]
    return "\n".join(numbered)


def write_file(path: str, content: str) -> str:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"Wrote {len(content)} chars to {path}"


def edit_file(path: str, old_str: str, new_str: str) -> str:
    p = Path(path).expanduser()
    if not p.exists():
        return f"Error: {path} does not exist"

    content = p.read_text(errors="replace")
    count = content.count(old_str)
    if count == 0:
        return "Error: old_str not found in file"
    if count > 1:
        return f"Error: old_str matches {count} locations — add more context to make it unique"

    p.write_text(content.replace(old_str, new_str, 1))
    return f"Edited {path}"


def list_directory(path: str = ".") -> str:
    p = Path(path).expanduser()
    if not p.exists():
        return f"Error: {path} does not exist"
    if not p.is_dir():
        return f"Error: {path} is not a directory"

    lines = []
    for entry in sorted(p.iterdir(), key=lambda e: (e.is_file(), e.name)):
        if entry.name in SKIP_DIRS:
            continue
        if entry.is_dir():
            lines.append(f"{entry.name}/")
        else:
            lines.append(f"{entry.name}  ({entry.stat().st_size} bytes)")
    return "\n".join(lines) if lines else "(empty)"
