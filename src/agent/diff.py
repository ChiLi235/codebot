import difflib
from pathlib import Path
from rich.console import Console
from rich.text import Text
from rich.rule import Rule


def compute_preview_diff(action: str, params: dict) -> tuple[list[str], str]:
    """Compute what the diff will look like before the tool executes."""
    path = params.get("path", "")
    p = Path(path).expanduser()
    old = p.read_text(errors="replace") if p.exists() and p.is_file() else ""

    if action == "edit_file":
        new = old.replace(params.get("old_str", ""), params.get("new_str", ""), 1)
    elif action == "write_file":
        new = params.get("content", "")
    else:
        return [], path

    return compute_diff(old, new), path


def compute_diff(old: str, new: str) -> list[str]:
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    hunks = list(difflib.unified_diff(old_lines, new_lines, n=3))
    return hunks[2:] if len(hunks) >= 2 and hunks[0].startswith("---") else hunks


def _word_highlight(a: str, b: str) -> tuple[Text, Text]:
    matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
    del_t, add_t = Text(), Text()
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            del_t.append(a[i1:i2], style="red")
            add_t.append(b[j1:j2], style="green")
        else:
            if a[i1:i2]:
                del_t.append(a[i1:i2], style="bold on dark_red")
            if b[j1:j2]:
                add_t.append(b[j1:j2], style="bold on dark_green")
    return del_t, add_t


def render_diff(diff_lines: list[str], filepath: str, console: Console) -> None:
    if not diff_lines:
        return
    console.print()
    console.print(Rule(f"[dim]{filepath}[/dim]", style="bright_black"))

    i = 0
    while i < len(diff_lines):
        raw = diff_lines[i].rstrip("\n")
        if raw.startswith("@@"):
            console.print(Text(raw, style="cyan dim"))
            i += 1
        elif raw.startswith("-") and i + 1 < len(diff_lines) and diff_lines[i + 1].startswith("+"):
            a_line = raw[1:]
            b_line = diff_lines[i + 1].rstrip("\n")[1:]
            del_t, add_t = _word_highlight(a_line, b_line)
            del_line = Text("-", style="red")
            del_line.append_text(del_t)
            add_line = Text("+", style="green")
            add_line.append_text(add_t)
            console.print(del_line)
            console.print(add_line)
            i += 2
        elif raw.startswith("-"):
            t = Text()
            t.append("-" + raw[1:], style="red")
            console.print(t)
            i += 1
        elif raw.startswith("+"):
            t = Text()
            t.append("+" + raw[1:], style="green")
            console.print(t)
            i += 1
        else:
            console.print(Text(raw, style="dim"))
            i += 1

    console.print()
