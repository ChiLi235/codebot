from botocore.exceptions import ClientError, BotoCoreError

from agent import client, config, ui, prompt, info, commands, session, subagent
from agent.tools import TOOLS, TOOL_MAP
from agent.tools.guard import check_valid
from agent.tools.human_check import check_human_eval, REJECT_MESSAGE, REJECT_MESSAGE_WITH_REASON_PREFIX
from agent.tools.shell import categorize


FORCE_SUBAGENT_PREFIX = (
    "you must spawn a subagent to complete the following task. "
    "If you couldn't find a dedicated agent for this task, you should spawn a "
    "general purpose agent. Tasks: "
)


def _format_api_error(e: Exception) -> tuple[str, str]:
    if isinstance(e, ClientError):
        return e.response["Error"]["Code"], e.response["Error"]["Message"]
    if isinstance(e, BotoCoreError):
        return type(e).__name__, str(e)
    return type(e).__name__, str(e)


def _compact_tool_results(messages: list) -> None:
    """Replace successful toolResult content with 'success' to shrink history.
    Errors keep their text. Run after a turn fully ends so current turn still
    sees full output, but next turn's context window stays small."""
    for msg in messages:
        if msg.get("role") != "user":
            continue
        for block in msg.get("content", []):
            tr = block.get("toolResult")
            if not tr:
                continue
            if tr.get("status") == "success":
                tr["content"] = [{"text": "success"}]


def _rollback_to_clean(messages: list) -> None:
    """Pop messages until last is assistant-text-only (no pending tool_use), or empty."""
    while messages:
        last = messages[-1]
        if last["role"] == "user":
            messages.pop()
            continue
        if any("toolUse" in b for b in last["content"]):
            messages.pop()
            continue
        return


def _process_single_tool(tu: dict) -> tuple[str, str, bool]:
    """Run guard + human approval + tool. Returns (text, status, user_denied)."""
    name = tu["name"]
    fn = TOOL_MAP.get(name)
    if fn is None:
        return f"Error: unknown tool '{name}'", "error", False

    guard_err = check_valid(name, tu["input"].get("path"), tu["input"].get("command"))
    if guard_err:
        return guard_err, "error", False

    verdict, instr = check_human_eval(name, tu["input"])
    if verdict == "denied":
        return REJECT_MESSAGE, "error", True
    if verdict == "instructed":
        return REJECT_MESSAGE_WITH_REASON_PREFIX + (instr or ""), "error", False

    try:
        return fn(**tu["input"]), "success", False
    except Exception as e:
        return f"Error: {e}", "error", False


def _print_tool_label(tu: dict) -> None:
    name = tu["name"]
    if name == "spawn_agent":
        preview = f"{tu['input'].get('subagent_type','?')}: {tu['input'].get('description','')[:60]}"
        ui.print_tool_call("spawn_agent", preview, "unknown")
        return
    category = categorize(tu["input"].get("command", "")) if name == "bash" else "unknown"
    preview = tu["input"].get("command") or tu["input"].get("path") or ""
    ui.print_tool_call(name, preview[:80], category)


def _dispatch_tool_batch(tool_uses: list) -> tuple[list, bool]:
    """Run a tool_use batch. spawn_agent calls go through the parallel dispatcher;
    other tools run sequentially. Returns (tool_results in original order, user_denied)."""
    spawn_uses = [tu for tu in tool_uses if tu["name"] == "spawn_agent"]
    other_uses = [tu for tu in tool_uses if tu["name"] != "spawn_agent"]

    results_by_id: dict = {}
    user_denied = False

    # print labels for spawn calls before dispatch (so user sees them upfront)
    for tu in spawn_uses:
        _print_tool_label(tu)

    if spawn_uses:
        spawn_results = subagent.dispatch_spawn_batch(spawn_uses, parent_depth=99)
        for tuid, pair in spawn_results.items():
            results_by_id[tuid] = pair

    for tu in other_uses:
        text, status, denied = _process_single_tool(tu)
        if denied:
            user_denied = True
        _print_tool_label(tu)
        results_by_id[tu["toolUseId"]] = (text, status)

    tool_results = []
    for tu in tool_uses:
        text, status = results_by_id[tu["toolUseId"]]
        tool_results.append({
            "toolResult": {
                "toolUseId": tu["toolUseId"],
                "content": [{"text": text}],
                "status": status,
            }
        })
    return tool_results, user_denied


def run(model_key: str = config.DEFAULT_MODEL):
    model_id = config.AVAILABLE_MODELS.get(model_key)
    if not model_id:
        ui.print_error(f"Unknown model '{model_key}'. Available: {', '.join(config.AVAILABLE_MODELS)}")
        return

    session.ensure_history()
    latest = session.latest_session_id()
    state = commands.State(model_key=model_key, model_id=model_id)
    if latest is not None:
        try:
            data = session.load(latest)
            state.session_id = latest
            state.messages.extend(data.get("messages", []))
            _compact_tool_results(state.messages)
            ui.console.print(f"[dim]resumed session {latest} "
                             f"({len(state.messages)} messages)[/dim]")
        except Exception:
            state.session_id = session.next_session_id()
    else:
        state.session_id = 1
        session.save(1, [], state.model_key)

    last_skills = info.scan_skills()
    system = prompt.build_system(tools=TOOLS)
    ui.print_header(state.model_key)

    messages = state.messages

    if messages:
        ui.render_history(messages, state.session_id)

    while True:
        try:
            user_input = ui.get_input(state.session_id).strip()
        except (EOFError, KeyboardInterrupt):
            _compact_tool_results(state.messages)
            session.save(state.session_id, state.messages, state.model_key)
            ui.console.print("\n[dim]Bye.[/dim]")
            break

        if not user_input:
            continue

        ui.print_user_message(user_input, state.session_id)

        if user_input.lower() == "exit":
            _compact_tool_results(state.messages)
            session.save(state.session_id, state.messages, state.model_key)
            ui.console.print("[dim]Bye.[/dim]")
            break

        if commands.handle(user_input, state):
            continue

        # force-subagent directive
        force_model_id = None
        if state.force_subagent:
            user_input = FORCE_SUBAGENT_PREFIX + user_input
            if state.force_subagent_model:
                force_model_id = config.AVAILABLE_MODELS[state.force_subagent_model]
            state.force_subagent = False
            state.force_subagent_model = None

        subagent.configure(state.model_id, state.model_key, force_model_id=force_model_id)

        # skill state diff
        current_skills = info.scan_skills()
        diff_text = info.format_skills_diff(*info.diff_skills(last_skills, current_skills), current_skills)
        if diff_text:
            messages.append({"role": "user", "content": [{"text": diff_text}]})
            ui.console.print(f"[dim]{diff_text.splitlines()[0]}[/dim]")
        last_skills = current_skills

        messages.append({"role": "user", "content": [{"text": user_input}]})

        tool_uses: list = []
        stop_reason = "end_turn"
        text_buffer = ""

        ui.console.print()
        ui.print_model_label(state.model_key)

        api_failed = False
        try:
            with ui.stream_response() as write:
                for event in client.converse_stream(messages, TOOLS, system, state.model_id):
                    if event["type"] == "text":
                        text_buffer += event["text"]
                        write(event["text"])
                    elif event["type"] == "tool_use":
                        tool_uses.append(event["tool_use"])
                    elif event["type"] == "done":
                        stop_reason = event["stop_reason"]
        except KeyboardInterrupt:
            ui.print_interrupted()
            if text_buffer:
                messages.append({"role": "assistant", "content": [{"text": text_buffer}]})
            subagent.clear_force_model()
            continue
        except (ClientError, BotoCoreError, Exception) as e:
            code, msg = _format_api_error(e)
            ui.print_api_error(code, msg)
            _rollback_to_clean(messages)
            api_failed = True

        if api_failed:
            subagent.clear_force_model()
            continue

        ui.console.print()
        assistant_content = []
        if text_buffer:
            assistant_content.append({"text": text_buffer})
        for tu in tool_uses:
            assistant_content.append({"toolUse": {
                "toolUseId": tu["toolUseId"],
                "name": tu["name"],
                "input": tu["input"],
            }})
        if assistant_content:
            messages.append({"role": "assistant", "content": assistant_content})

        # tool use loop
        iterations = 0
        while stop_reason == "tool_use" and tool_uses and iterations < config.MAX_ITERATIONS:
            iterations += 1
            tool_results, user_denied = _dispatch_tool_batch(tool_uses)
            messages.append({"role": "user", "content": tool_results})

            if user_denied:
                ui.console.print("[dim]turn ended by user[/dim]\n")
                break

            tool_uses = []
            stop_reason = "end_turn"
            text_buffer = ""

            ui.console.print()
            ui.print_model_label(state.model_key)

            inner_failed = False
            try:
                with ui.stream_response() as write:
                    for event in client.converse_stream(messages, TOOLS, system, state.model_id):
                        if event["type"] == "text":
                            text_buffer += event["text"]
                            write(event["text"])
                        elif event["type"] == "tool_use":
                            tool_uses.append(event["tool_use"])
                        elif event["type"] == "done":
                            stop_reason = event["stop_reason"]
            except KeyboardInterrupt:
                ui.print_interrupted()
                break
            except (ClientError, BotoCoreError, Exception) as e:
                code, msg = _format_api_error(e)
                ui.print_api_error(code, msg)
                _rollback_to_clean(messages)
                inner_failed = True

            if inner_failed:
                break

            ui.console.print()
            assistant_content = []
            if text_buffer:
                assistant_content.append({"text": text_buffer})
            for tu in tool_uses:
                assistant_content.append({"toolUse": {
                    "toolUseId": tu["toolUseId"],
                    "name": tu["name"],
                    "input": tu["input"],
                }})
            if assistant_content:
                messages.append({"role": "assistant", "content": assistant_content})

        # turn complete
        subagent.clear_force_model()
        _compact_tool_results(state.messages)
        session.save(state.session_id, state.messages, state.model_key)
