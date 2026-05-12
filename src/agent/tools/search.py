import re
import shutil
import subprocess
from pathlib import Path

MAX_RESULTS = 100


def grep(pattern: str, path: str = ".", glob: str | None = None) -> str:
    """Search for regex pattern in files. Returns path:line:content triples."""
    if shutil.which("rg"):
        cmd = ["rg", "--line-number", "--no-heading", "--color=never", pattern, path]
        if glob:
            cmd += ["--glob", glob]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        lines = result.stdout.splitlines()
    else:
        lines = _python_grep(pattern, path, glob)

    if not lines:
        return "No matches found"
    if len(lines) > MAX_RESULTS:
        lines = lines[:MAX_RESULTS] + [f"... [{len(lines) - MAX_RESULTS} more results truncated]"]
    return "\n".join(lines)


def _python_grep(pattern: str, path: str, glob_pattern: str | None) -> list[str]:
    base = Path(path)
    file_glob = glob_pattern or "**/*"
    results = []
    try:
        rx = re.compile(pattern)
    except re.error as e:
        return [f"Error: invalid regex: {e}"]

    for fp in sorted(base.rglob(file_glob if glob_pattern else "*")):
        if not fp.is_file():
            continue
        if any(part.startswith(".") or part in {"__pycache__", "node_modules", ".venv"}
               for part in fp.parts):
            continue
        try:
            for i, line in enumerate(fp.read_text(errors="replace").splitlines(), 1):
                if rx.search(line):
                    results.append(f"{fp}:{i}:{line}")
                    if len(results) >= MAX_RESULTS:
                        return results
        except (OSError, UnicodeDecodeError):
            continue
    return results


def glob_files(pattern: str) -> str:
    """Expand a glob pattern. Returns one path per line."""
    results = sorted(Path(".").glob(pattern))
    if not results:
        return "No matches found"
    lines = [str(p) for p in results[:MAX_RESULTS]]
    if len(results) > MAX_RESULTS:
        lines.append(f"... [{len(results) - MAX_RESULTS} more truncated]")
    return "\n".join(lines)
