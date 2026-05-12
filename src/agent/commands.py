"""Slash command registry. Add new commands by decorating with @register."""
from dataclasses import dataclass, field
from typing import Callable

from agent import config, ui, session


@dataclass
class State:
    """Mutable state passed to command handlers."""
    model_key: str
    model_id: str
    session_id: int = 1
    messages: list = field(default_factory=list)
    force_subagent: bool = False
    force_subagent_model: str | None = None


_REGISTRY: dict[str, tuple[Callable, str]] = {}


def register(name: str, description: str = ""):
    def decorator(fn: Callable):
        _REGISTRY[name] = (fn, description)
        return fn
    return decorator


def handle(text: str, state: State) -> bool:
    """Run a slash command. Returns True if text was a slash command (handled or invalid)."""
    if not text.startswith("/"):
        return False

    parts = text[1:].split(maxsplit=1)
    if not parts:
        return False
    cmd = parts[0]
    args = parts[1].strip() if len(parts) > 1 else ""

    entry = _REGISTRY.get(cmd)
    if entry is None:
        ui.print_error(f"Unknown command: /{cmd}. Try /help.")
        return True

    handler, _ = entry
    handler(args, state)
    return True


# ── built-in commands ─────────────────────────────────────────────────────────

@register("model", "Switch model: /model <name>")
def _cmd_model(args: str, state: State):
    if not args:
        ui.print_error("Usage: /model <name>. See /model-list.")
        return
    if args not in config.AVAILABLE_MODELS:
        ui.print_error(f"Unknown model '{args}'. See /model-list.")
        return
    state.model_key = args
    state.model_id = config.AVAILABLE_MODELS[args]
    ui.print_model_switched(args)


@register("model-list", "List all available models")
def _cmd_model_list(args: str, state: State):
    ui.console.print()
    ui.console.print("[bold]Available models[/bold]")
    for key, mid in config.AVAILABLE_MODELS.items():
        marker = "[bold green]●[/bold green]" if key == state.model_key else "[dim]○[/dim]"
        ui.console.print(f"  {marker} [cyan]{key:12}[/cyan] [dim]{mid}[/dim]")
    ui.console.print()


@register("help", "Show this help")
def _cmd_help(args: str, state: State):
    ui.console.print()
    ui.console.print("[bold]Commands[/bold]")
    for name, (_, desc) in sorted(_REGISTRY.items()):
        ui.console.print(f"  [cyan]/{name:14}[/cyan] [dim]{desc}[/dim]")
    ui.console.print(f"  [cyan]{'exit':15}[/cyan] [dim]Quit codebot[/dim]")
    ui.console.print()


@register("checkout", "Switch sessions: /checkout <id>")
def _cmd_checkout(args: str, state: State):
    if not args.isdigit():
        ui.print_error("Usage: /checkout <session_id>")
        return
    target = int(args)
    if not session.session_path(target).exists():
        ui.print_error(f"session-{target}.json not found. Try /sessions.")
        return

    # save current before swap
    if state.messages:
        session.save(state.session_id, state.messages, state.model_key)

    data = session.load(target)
    state.session_id = target
    state.messages.clear()
    state.messages.extend(data.get("messages", []))
    ui.clear_screen()
    ui.console.print(f"[dim]checked out session {target} "
                     f"({len(state.messages)} messages)[/dim]")
    if state.messages:
        ui.render_history(state.messages)


@register("new-session", "Save current and start a fresh session")
def _cmd_new_session(args: str, state: State):
    if state.messages:
        session.save(state.session_id, state.messages, state.model_key)
    new_id = session.next_session_id()
    state.session_id = new_id
    state.messages.clear()
    session.save(new_id, [], state.model_key)
    ui.console.print(f"[dim]started session {new_id}[/dim]\n")


@register("delete", "Delete session: /delete <id> (one) | /delete (all except current)")
def _cmd_delete(args: str, state: State):
    if args:
        if not args.isdigit():
            ui.print_error("Usage: /delete <session_id>")
            return
        target = int(args)
        if target == state.session_id:
            ui.print_error("Cannot delete current session. Use /delete-all to wipe and reset.")
            return
        if not session.delete(target):
            ui.print_error(f"session-{target}.json not found.")
            return
        ui.console.print(f"[dim]deleted session {target}[/dim]\n")
        return

    # no args: delete all except current
    ids = [sid for sid in session.list_session_ids() if sid != state.session_id]
    if not ids:
        ui.console.print("[dim]nothing to delete[/dim]\n")
        return
    for sid in ids:
        session.delete(sid)
    ui.console.print(f"[dim]deleted {len(ids)} session(s); kept current ({state.session_id})[/dim]\n")


@register("delete-all", "Delete every session and start a new one")
def _cmd_delete_all(args: str, state: State):
    for sid in session.list_session_ids():
        session.delete(sid)
    state.messages.clear()
    state.session_id = 1
    session.save(1, [], state.model_key)
    ui.clear_screen()
    ui.console.print(f"[dim]all sessions deleted; started session 1[/dim]\n")


@register("sessions", "List all saved sessions")
def _cmd_sessions(args: str, state: State):
    rows = session.list_all()
    if not rows:
        ui.console.print("[dim]no sessions yet[/dim]\n")
        return
    ui.console.print()
    ui.console.print("[bold]Sessions[/bold]")
    for r in rows:
        marker = "[bold green]●[/bold green]" if r["session_id"] == state.session_id else "[dim]○[/dim]"
        sid = r["session_id"]
        if r.get("error"):
            ui.console.print(f"  {marker} session-{sid} [red](corrupt)[/red]")
            continue
        ui.console.print(f"  {marker} [cyan]session-{sid:<3}[/cyan] "
                         f"[dim]model={r['model_key']:8} msgs={r['msg_count']:<4} "
                         f"updated={r['updated_at']}[/dim]")
    ui.console.print()


@register("agents", "List available subagents")
def _cmd_agents(args: str, state: State):
    from agent import subagent
    agents = subagent.load_agents()
    if not agents:
        ui.console.print("[dim]no subagents found[/dim]\n")
        return
    ui.console.print()
    ui.console.print("[bold]Available subagents[/bold]")
    for name, spec in agents.items():
        ui.console.print(f"  [cyan]{name:12}[/cyan] [dim]({spec.source}) "
                         f"maxturn={spec.maxturn} {spec.description}[/dim]")
    ui.console.print()


@register("subagent", "Force next task to be delegated to a subagent. Optional: /subagent <model>")
def _cmd_subagent(args: str, state: State):
    if args:
        if args not in config.AVAILABLE_MODELS:
            ui.print_error(f"Unknown model '{args}'. See /model-list.")
            return
        state.force_subagent_model = args
    state.force_subagent = True
    pinned = f" pinned to model={args}" if args else ""
    ui.console.print(f"[dim]next task will be delegated to a subagent{pinned}[/dim]\n")
