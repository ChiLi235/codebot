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
_session = PromptSession(history=InMemoryHistory(), style=_prompt_style)


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
    """Replay saved messages (user text, assistant text/tool calls). Tool results skipped."""
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", [])
        if role == "user":
            for block in content:
                if "text" in block:
                    print_user_message(block["text"], session_id)
        elif role == "assistant":
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
    """Stream chunks inside a bounded rich Live region (never scrolls into
    scrollback). On clean exit, Live clears its region (transient=True) and we
    print the full Markdown render. On exception, Live still clears via
    transient; we print raw text fallback.

    Why this design: pure-ANSI cursor save/restore cannot erase content that
    scrolled into terminal scrollback. Live keeps the stream view bounded
    within the viewport using vertical_overflow='ellipsis', so the streamed
    text never reaches scrollback and can be fully erased on exit.
    """
    buf: list[str] = []
    state = {"text": ""}

    with Live(
        Text(""),
        console=console,
        refresh_per_second=15,
        vertical_overflow="ellipsis",
        transient=True,
        auto_refresh=True,
    ) as live:
        def write(chunk: str):
            if not chunk:
                return
            buf.append(chunk)
            state["text"] += chunk
            live.update(Text(state["text"]))

        try:
            yield write
        except BaseException:
            raise

    # Live region has been cleared (transient=True). Print final markdown.
    text = "".join(buf)
    if text:
        console.print(Markdown(text))


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
    # erase the live-prompt line; we re-echo as a styled bar via print_user_message
    print("\033[F\033[K", end="", flush=True)
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


def ask_approval(action: str, reason: str, detail: str) -> tuple[str, str | None]:
    """
    Show approval prompt with arrow-key selection.
    Returns ('approved', None) | ('denied', None) | ('instructed', text).
    """
    console.print()
    console.print(Panel(
        f"[bold yellow]⚠ Approval required[/bold yellow]\n"
        f"[bold]Tool:[/bold] {action}\n"
        f"[bold]Reason:[/bold] {reason}\n"
        f"[bold]Detail:[/bold]\n[white]{detail}[/white]",
        border_style="yellow",
        padding=(0, 2),
    ))
    console.print("[dim]↑↓ to navigate · enter to select · esc to cancel[/dim]")

    options = ["Yes — approve", "No — deny", "Tell something else"]
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
