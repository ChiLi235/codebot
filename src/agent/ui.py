import sys
from contextlib import contextmanager

from rich.console import Console
from rich.text import Text
from rich.rule import Rule
from rich.panel import Panel
from rich.markdown import Markdown
from rich.live import Live
from prompt_toolkit import PromptSession, Application
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style
from prompt_toolkit.layout import Layout, HSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.document import Document

console = Console(highlight=False)

MODEL_DISPLAY_NAMES = {
    "sonnet":    "Claude Sonnet",
    "haiku":     "Claude Haiku",
    "opus":      "Claude Opus",
    "nova-pro":  "Amazon Nova Pro",
    "nova-lite": "Amazon Nova Lite",
    "deepseek":  "DeepSeek",
    "chatgpt-120b": "ChatGPT OSS 120B",
    "chatgpt-20b":"ChatGPT OSS 20b",
}

_prompt_style = Style.from_dict({"prompt": "bold"})

_input_kb = KeyBindings()

@_input_kb.add("c-u")
def _clear_input(event):
    event.app.current_buffer.set_document(Document(), bypass_readonly=True)

@_input_kb.add("c-a")
def _go_line_start(event):
    event.app.current_buffer.cursor_position = 0

@_input_kb.add("c-e")
def _go_line_end(event):
    buf = event.app.current_buffer
    buf.cursor_position = len(buf.text)

_session = PromptSession(history=InMemoryHistory(), style=_prompt_style, key_bindings=_input_kb)


def get_display_name(model_key: str) -> str:
    return MODEL_DISPLAY_NAMES.get(model_key, model_key)


def print_header(model_key: str):
    name = get_display_name(model_key)
    console.print()
    console.print(Panel(
        f"[bold cyan]Coding Agent[/bold cyan]  [dim]·[/dim]  [bold]{name}[/bold]\n"
        "[dim]Type [white]/model <name>[/white] to switch · [white]exit[/white] to quit[/dim]",
        border_style="bright_black",
        padding=(0, 2),
    ))
    console.print()


def print_model_label(model_key: str):
    name = get_display_name(model_key)
    console.print(f"[bold cyan]{name}[/bold cyan]")


def print_response_chunk(text: str):
    console.print(text, end="", markup=False, highlight=False)


def print_response_end(full_text: str):
    """Re-render completed response as markdown."""
    console.print()


def render_history(messages: list, session_id: int | None = None):
    """Replay saved AnyMessage list. Tool results and meta messages skipped."""
    from agent.messages.types import UserMessage, AssistantMessage, SystemMessage
    for msg in messages:
        if isinstance(msg, SystemMessage):
            continue
        if isinstance(msg, UserMessage):
            if msg.isMeta or msg.toolResult or msg.isCompactSummary:
                continue
            content = msg.message.get("content", [])
            if isinstance(content, str):
                if content:
                    print_user_message(content, session_id)
            else:
                for block in content:
                    if "text" in block:
                        print_user_message(block["text"], session_id)
        elif isinstance(msg, AssistantMessage):
            if msg.apierror:
                continue
            content = msg.message.get("content", [])
            if isinstance(content, str):
                content = [{"text": content}] if content else []
            text_parts = [b["text"] for b in content if "text" in b]
            tool_parts = [b["toolUse"] for b in content if "toolUse" in b]
            if text_parts:
                console.print()
                console.print(Markdown("".join(text_parts)))
            for tu in tool_parts:
                inp = tu.get("input") or {}
                preview = inp.get("command") or inp.get("path") or ""
                console.print(f"[dim]  ⚙ {tu.get('name','?')} {str(preview)[:80]}[/dim]")
    console.print()


def clear_screen():
    console.clear()


@contextmanager
def stream_response():
    """Stream markdown using the line-commit strategy from opencode.

    How it works (per ui_stream_fix.md):
    - Incoming chunks are appended to a line buffer
    - When a newline arrives, the completed line is locked into scrollback as
      rendered Markdown and removed from the buffer — it is NEVER redrawn
    - The current partial (incomplete) line sits in a Rich Live region which
      redraws only that one line in-place as plain text
    - Code fences (``` blocks) are an exception: lines inside an open fence are
      buffered until the closing ``` arrives, then the whole block is committed
      at once as a single Markdown render
    - On stream end, whatever remains in the buffer is flushed as Markdown

    This avoids:
    - Re-rendering growing text from scratch on every chunk (old bug → flicker)
    - live.stop()/start() cycling (old bug → visual chaos)
    - Raw text leaking to scrollback (old bug → duplication)
    """
    line_buf = ""               # current partial line (not yet newline-terminated)
    block_buf: list[str] = []   # lines buffered for multi-line blocks
    in_fence = False             # inside a ``` code fence
    in_table = False             # inside a markdown table

    live = Live(
        Text(""),
        console=console,
        refresh_per_second=15,
        vertical_overflow="visible",
        transient=True,
        auto_refresh=False,   # we call live.refresh() manually after each update
    )
    live.start()

    def _is_table_line(line: str) -> bool:
        return line.strip().startswith("|")

    def _commit_block():
        """Flush block_buf to scrollback as a single Markdown render."""
        block = "\n".join(block_buf)
        block_buf.clear()
        if block.strip():
            console.print(Markdown(block))

    def _commit_line(line: str):
        """Print one standalone line as rendered Markdown to scrollback."""
        if line.strip():
            console.print(Markdown(line))
        else:
            console.print()  # preserve blank lines

    def write(chunk: str):
        nonlocal line_buf, in_fence, in_table

        for ch in chunk:
            if ch == "\n":
                completed = line_buf
                line_buf = ""
                stripped = completed.strip()

                # ── inside a code fence ──────────────────────────────────────
                if in_fence:
                    block_buf.append(completed)
                    if stripped.startswith("```"):
                        in_fence = False
                        _commit_block()

                # ── inside a table ───────────────────────────────────────────
                elif in_table:
                    if _is_table_line(completed):
                        block_buf.append(completed)   # keep buffering rows
                    else:
                        # table ended — commit the whole table, then this line
                        in_table = False
                        _commit_block()
                        _commit_line(completed)

                # ── normal context ───────────────────────────────────────────
                else:
                    if stripped.startswith("```"):
                        in_fence = True
                        block_buf.clear()
                        block_buf.append(completed)
                    elif _is_table_line(completed):
                        in_table = True
                        block_buf.clear()
                        block_buf.append(completed)
                    else:
                        _commit_line(completed)

                live.update(Text(""))
                live.refresh()
            else:
                line_buf += ch
                live.update(Text(line_buf))
                live.refresh()

    try:
        yield write
    except BaseException:
        live.stop()
        raise

    live.stop()

    # Flush anything remaining in the buffers
    if line_buf:
        block_buf.append(line_buf)
    if block_buf:
        _commit_block()


def print_model_switched(model_key: str):
    name = get_display_name(model_key)
    console.print(f"[dim]⇄ Switched to [bold]{name}[/bold][/dim]\n")


def print_interrupted():
    console.print("\n[dim]interrupted[/dim]\n")


def print_subagent_header(name: str, description: str):
    console.print()
    console.print(f"[bold magenta]┃ {name}[/bold magenta] [dim]{description}[/dim]")


def print_subagent_done(name: str):
    console.print(f"[dim magenta]┗ {name} done[/dim magenta]")


def print_tool_call(name: str, preview: str, category: str = "unknown"):
    from agent.tools.shell import CATEGORY_LABELS
    label = {
        "read_file":      "Read",
        "write_file":     "Wrote",
        "edit_file":      "Edited",
        "list_directory": "Listed",
        "grep":           "Searched",
        "glob":           "Globbed",
        "bash":           CATEGORY_LABELS.get(category, "Ran"),
    }.get(name, "Called")
    console.print(f"[dim]  ⚙ [bold]{label}[/bold] {preview}[/dim]")


def print_diff(diff_lines: list[str], filepath: str) -> None:
    from agent.diff import render_diff
    render_diff(diff_lines, filepath, console)


def print_error(msg: str):
    console.print(f"[red]Error:[/red] {msg}")


def print_api_error(code: str, message: str):
    console.print()
    console.print(Panel(
        f"[bold red]API error:[/bold red] {code}\n[white]{message}[/white]\n\n"
        f"[dim]Your message was kept. Type again to retry, or continue normally.[/dim]",
        border_style="red",
        padding=(0, 2),
    ))
    console.print()


def get_input(session_id: int | None = None) -> str:
    """Blocking prompt with history support. Raises EOFError/KeyboardInterrupt."""
    label = f"\n[{session_id}] > " if session_id is not None else "\n> "
    text = _session.prompt(label, style=_prompt_style)
    # erase all wrapped lines of the prompt before re-echoing via print_user_message
    visible_label = label.lstrip("\n")
    term_width = console.width or 80
    lines = max(1, -(-( len(visible_label) + len(text)) // term_width))  # ceiling div
    print("\033[F\033[K" * lines, end="", flush=True)
    return text


def print_user_message(text: str, session_id: int | None = None):
    """Re-echo submitted user input as a grey-background bar."""
    label = f"[{session_id}] >" if session_id is not None else ">"
    width = console.width
    lines = text.splitlines() or [""]
    console.print()
    first = f" {label} {lines[0]}".ljust(width)
    console.print(first, style="bold white on grey23")
    for ln in lines[1:]:
        console.print(f"   {ln}".ljust(width), style="bold white on grey23")


def _arrow_select(options: list[str]) -> int | None:
    """Inline arrow-key selector. Returns index or None on cancel."""
    selected = [0]

    def render():
        lines = []
        for i, opt in enumerate(options):
            if i == selected[0]:
                lines.append(("class:selected", f"  ❯ {opt}\n"))
            else:
                lines.append(("class:option", f"    {opt}\n"))
        return lines

    kb = KeyBindings()

    @kb.add("up")
    @kb.add("c-p")
    def _up(event):
        selected[0] = (selected[0] - 1) % len(options)

    @kb.add("down")
    @kb.add("c-n")
    def _down(event):
        selected[0] = (selected[0] + 1) % len(options)

    @kb.add("enter")
    def _enter(event):
        event.app.exit(result=selected[0])

    @kb.add("c-c")
    @kb.add("escape")
    def _cancel(event):
        event.app.exit(result=None)

    style = Style.from_dict({
        "selected": "bold fg:cyan",
        "option": "fg:#888888",
    })

    layout = Layout(HSplit([
        Window(FormattedTextControl(render), height=len(options), always_hide_cursor=True),
    ]))
    app = Application(layout=layout, key_bindings=kb, style=style, full_screen=False, mouse_support=False)
    return app.run()


def ask_approval(
    action: str,
    reason: str,
    detail: str,
    diff_lines: list[str] | None = None,
    diff_path: str = "",
) -> tuple[str, str | None]:
    """
    Show approval prompt with arrow-key selection.
    When diff_lines provided, renders an expandable diff above the selector.
    Returns ('approved', None) | ('denied', None) | ('instructed', text).
    """
    MAX_COLLAPSED = 8
    options = ["Yes — approve", "No — deny", "Tell something else"]

    console.print()

    if diff_lines:
        selected = [0]
        expanded = [False]
        has_more = len(diff_lines) > MAX_COLLAPSED

        def _render_diff():
            rows = []
            rows.append(("bold", f"  {diff_path}\n"))
            display = diff_lines if expanded[0] else diff_lines[:MAX_COLLAPSED]
            for raw in display:
                line = raw.rstrip("\n")
                if line.startswith("@@"):
                    rows.append(("ansicyan", f"  {line}\n"))
                elif line.startswith("-"):
                    rows.append(("ansired", f"  {line}\n"))
                elif line.startswith("+"):
                    rows.append(("ansigreen", f"  {line}\n"))
                else:
                    rows.append(("", f"  {line}\n"))
            if has_more:
                hidden = len(diff_lines) - MAX_COLLAPSED
                if not expanded[0]:
                    rows.append(("ansiyellow", f"  … {hidden} more lines\n"))
                else:
                    rows.append(("ansiyellow", "  [c to collapse]\n"))
            rows.append(("bold ansiyellow", f"\n  ⚠ {action}  "))
            rows.append(("ansiyellow", f"{reason}\n\n"))
            for i, opt in enumerate(options):
                if i == selected[0]:
                    rows.append(("bold ansicyan", f"  ❯ {opt}\n"))
                else:
                    rows.append(("#888888", f"    {opt}\n"))
            hint = "  ↑↓ navigate · enter select"
            if has_more:
                hint += " · e expand" if not expanded[0] else " · c collapse"
            hint += " · esc cancel\n"
            rows.append(("", hint))
            return rows

        kb = KeyBindings()

        @kb.add("up")
        @kb.add("c-p")
        def _up(event):
            selected[0] = (selected[0] - 1) % len(options)

        @kb.add("down")
        @kb.add("c-n")
        def _down(event):
            selected[0] = (selected[0] + 1) % len(options)

        @kb.add("e")
        def _expand(event):
            if has_more:
                expanded[0] = True

        @kb.add("c")
        def _collapse(event):
            expanded[0] = False

        @kb.add("enter")
        def _confirm(event):
            event.app.exit(result=selected[0])

        @kb.add("c-c")
        @kb.add("escape")
        def _cancel(event):
            event.app.exit(result=None)

        layout = Layout(HSplit([
            Window(FormattedTextControl(_render_diff), always_hide_cursor=True),
        ]))
        app = Application(layout=layout, key_bindings=kb, full_screen=False, mouse_support=False)
        idx = app.run()
    else:
        console.print(Panel(
            f"[bold yellow]⚠ Approval required[/bold yellow]\n"
            f"[bold]Tool:[/bold] {action}\n"
            f"[bold]Reason:[/bold] {reason}\n"
            f"[bold]Detail:[/bold]\n[white]{detail}[/white]",
            border_style="yellow",
            padding=(0, 2),
        ))
        console.print("[dim]↑↓ to navigate · enter to select · esc to cancel[/dim]")
        idx = _arrow_select(options)

    if idx is None or idx == 1:
        console.print("[red]denied[/red]\n")
        return ("denied", None)

    if idx == 0:
        console.print("[green]approved[/green]\n")
        return ("approved", None)

    # idx == 2: text input
    try:
        msg = _session.prompt("instruction > ", style=_prompt_style).strip()
    except (EOFError, KeyboardInterrupt):
        console.print("[red]denied[/red]\n")
        return ("denied", None)

    if not msg:
        console.print("[dim]empty instruction, treating as deny[/dim]\n")
        return ("denied", None)

    console.print(f"[cyan]instructed:[/cyan] {msg}\n")
    return ("instructed", msg)
