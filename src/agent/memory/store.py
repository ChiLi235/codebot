"""MemoryStore and MemoryCursor — disk I/O for the memory directory."""
from __future__ import annotations

import json
import re
from pathlib import Path

# ── frontmatter helpers ────────────────────────────────────────────────────────

_FM_RE = re.compile(r"^---\s*\n(.*?)\n---", re.DOTALL)


def _parse_frontmatter(text: str) -> dict:
    m = _FM_RE.match(text)
    if not m:
        return {}
    out: dict = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            out[k.strip()] = v.strip()
    return out


def _build_manifest(files: list[dict]) -> str:
    if not files:
        return "(no memory files yet)"
    return "\n".join(f"- {f['filename']} — {f['description']}" for f in files)


# ── MemoryStore ────────────────────────────────────────────────────────────────

class MemoryStore:
    def __init__(self, memory_dir: Path):
        self.memory_dir = memory_dir.resolve()

    def ensure_dir(self) -> None:
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        index = self.memory_dir / "MEMORY.md"
        if not index.exists():
            index.write_text("# Memory\n\n")

    def read_index(self) -> str:
        index = self.memory_dir / "MEMORY.md"
        return index.read_text(errors="replace") if index.exists() else ""

    def read_file(self, filename: str) -> str:
        p = self._safe_path(filename)
        if p is None:
            return f"Error: '{filename}' is outside memory directory"
        if not p.exists():
            return f"Error: '{filename}' not found"
        return p.read_text(errors="replace")

    def write_file(self, filename: str, content: str) -> str:
        p = self._safe_path(filename)
        if p is None:
            return f"Error: '{filename}' is outside memory directory"
        p.write_text(content)
        return f"Wrote {filename}"

    def edit_file(self, filename: str, old_str: str, new_str: str) -> str:
        p = self._safe_path(filename)
        if p is None:
            return f"Error: '{filename}' is outside memory directory"
        if not p.exists():
            return f"Error: '{filename}' not found"
        content = p.read_text(errors="replace")
        if old_str not in content:
            return "Error: old_str not found in file"
        if content.count(old_str) > 1:
            return "Error: old_str matches multiple locations — add more context"
        p.write_text(content.replace(old_str, new_str, 1))
        return f"Edited {filename}"

    def scan_files(self) -> list[dict]:
        """Return [{filename, description}] parsed from frontmatter of each .md file."""
        if not self.memory_dir.exists():
            return []
        results = []
        for p in sorted(self.memory_dir.glob("*.md")):
            if p.name == "MEMORY.md":
                continue
            fm = _parse_frontmatter(p.read_text(errors="replace"))
            results.append({"filename": p.name, "description": fm.get("description", "")})
        return results

    def _safe_path(self, filename: str) -> Path | None:
        """Resolve filename to memory_dir. Reject path traversal or separators."""
        if "/" in filename or "\\" in filename or filename.startswith("."):
            return None
        p = (self.memory_dir / filename).resolve()
        try:
            p.relative_to(self.memory_dir)
            return p
        except ValueError:
            return None


# ── MemoryCursor ───────────────────────────────────────────────────────────────

class MemoryCursor:
    def __init__(self, memory_dir: Path):
        self._path = Path(memory_dir) / "cursor.json"
        self.last_uuid: str | None = None

    def load(self) -> None:
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text())
                self.last_uuid = data.get("last_uuid")
            except Exception:
                self.last_uuid = None

    def save(self) -> None:
        try:
            self._path.write_text(json.dumps({"last_uuid": self.last_uuid}))
        except Exception:
            pass
