import uuid
from botocore.exceptions import ClientError, BotoCoreError

from agent import client, config, ui, prompt, info, commands, session, subagent, skills
from agent.memory import get_memory_dir, fire_extract, recall_memories, MemoryStore, MemoryCursor
from agent.tools import TOOLS, TOOL_MAP
from agent.tools.guard import check_valid
from agent.tools.human_check import check_human_eval, set_headless
from agent.tools.shell import categorize
from agent.messages import (
    UserMessage, AssistantMessage, SystemMessage,
    normalizeMessagesForAPI, recordTranscript, load_transcript, record_token_usage,
)
from agent.messages.types import AnyMessage
from agent.context.pipeline import prepare_query
from agent.context.budget import rebuild_decision_map


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


def _rollback_to_clean(mutableMessages: list[AnyMessage]) -> None:
    """Pop messages until last assistant has no pending tool_use, or list is empty.
    NOTE: already-recorded JSONL lines are NOT removed — JSONL is append-only."""
    while mutableMessages:
        last = mutableMessages[-1]
        if isinstance(last, UserMessage):
            mutableMessages.pop()
            continue
        if isinstance(last, AssistantMessage):
            content = last.message.get("content", [])
            if any("toolUse" in b for b in content):
                mutableMessages.pop()
                continue
        return


def _process_single_tool(tu: dict) -> tuple[str, str, bool, str | None]:
    """Run guard + human approval + tool. Returns (text, status, user_denied, instruction)."""
    name = tu["name"]
    fn = TOOL_MAP.get(name)
    if fn is None:
        return f"Error: unknown tool '{name}'", "error", False, None

    guard_err = check_valid(name, tu["input"].get("path"), tu["input"].get("command"))
    if guard_err:
        return guard_err, "error", False, None

    verdict, instr = check_human_eval(name, tu["input"])
    if verdict == "denied":
        return "Cancelled", "error", True, None
    if verdict == "instructed":
        return "Cancelled", "error", False, instr or ""

    try:
        result = fn(**tu["input"])
    except Exception as e:
        return f"Error: {e}", "error", False, None

    return result, "success", False, None


def _print_cache_stats(usage: dict) -> None:
    read = usage.get("cacheReadInputTokens", 0)
    write = usage.get("cacheWriteInputTokens", 0)
    if read or write:
        ui.console.print(f"[dim]cache: read={read} write={write}[/dim]")


def _print_tool_label(tu: dict) -> None:
    name = tu["name"]
    if name == "spawn_agent":
        preview = f"{tu['input'].get('subagent_type','?')}: {tu['input'].get('description','')[:60]}"
        ui.print_tool_call("spawn_agent", preview, "unknown")
        return
    category = categorize(tu["input"].get("command", "")) if name == "bash" else "unknown"
    preview = tu["input"].get("command") or tu["input"].get("path") or ""
    ui.print_tool_call(name, preview[:80], category)


def _dispatch_tool_batch(tool_uses: list) -> tuple[list, bool, str | None]:
    """Run a tool_use batch. Returns (raw tool_result dicts, user_denied, instruction_text)."""
    spawn_uses = [tu for tu in tool_uses if tu["name"] == "spawn_agent"]
    other_uses = [tu for tu in tool_uses if tu["name"] != "spawn_agent"]

    results_by_id: dict = {}
    user_denied = False
    instruction: str | None = None

    for tu in spawn_uses:
        _print_tool_label(tu)

    if spawn_uses:
        spawn_results = subagent.dispatch_spawn_batch(spawn_uses, parent_depth=99)
        for tuid, pair in spawn_results.items():
            results_by_id[tuid] = pair

    for tu in other_uses:
        if user_denied or instruction is not None:
            results_by_id[tu["toolUseId"]] = ("Cancelled", "error")
            continue
        text, status, denied, instr = _process_single_tool(tu)
        if denied:
            user_denied = True
        if instr is not None:
            instruction = instr
        _print_tool_label(tu)
        results_by_id[tu["toolUseId"]] = (text, status)

    tool_results = []
    for tu in tool_uses:
        text, status = results_by_id[tu["toolUseId"]]
        tool_results.append({
            "toolUseId": tu["toolUseId"],
            "content": [{"text": text}],
            "status": status,
        })
    return tool_results, user_denied, instruction


def _run_turn(
    user_input: str,
    state: commands.State,
    mutableMessages: list[AnyMessage],
    system: list,
    model_id: str,
    last_asst_uuid: str | None,
    last_skills,
    mem_cursor: MemoryCursor,
):
    """Process one user turn: setup, model response, full tool-use loop.

    Shared by the interactive REPL (run) and the single-shot headless
    entrypoint (run_headless). Returns (last_asst_uuid, last_skills).
    """
    # force-subagent directive
    force_model_id = None
    if state.force_subagent:
        user_input = FORCE_SUBAGENT_PREFIX + user_input
        if state.force_subagent_model:
            force_model_id = config.AVAILABLE_MODELS[state.force_subagent_model]
        state.force_subagent = False
        state.force_subagent_model = None

    subagent.configure(state.model_id, state.model_key, force_model_id=force_model_id)

    # skill state diff — recorded as meta user message
    current_skills = skills.scan_skills()
    diff_text = skills.format_skills_diff(*skills.diff_skills(last_skills, current_skills), current_skills)
    if diff_text:
        meta_msg = UserMessage(
            message={"role": "user", "content": [{"text": diff_text}]},
            isMeta=True,
        )
        mutableMessages.append(meta_msg)
        recordTranscript(state.session_id, meta_msg)
        ui.console.print(f"[dim]{diff_text.splitlines()[0]}[/dim]")
    last_skills = current_skills

    # recall relevant memories and append to user turn as system-reminder
    _recalled = recall_memories(user_input)
    if _recalled:
        _recall_block = "\n\n---\n\n".join(_recalled)
        _turn_text = (
            f"{user_input}\n\n"
            f"<system-reminder>\n{_recall_block}\n</system-reminder>"
        )
        ui.console.print(f"[dim]recalled {len(_recalled)} memory file(s)[/dim]")
    else:
        _turn_text = user_input

    # record the human turn — parent is last assistant
    last_user_uuid = str(uuid.uuid4())
    user_msg = UserMessage(
        uuid=last_user_uuid,
        message={"role": "user", "content": [{"text": _turn_text}]},
        parentUuid=last_asst_uuid,
    )
    mutableMessages.append(user_msg)
    recordTranscript(state.session_id, user_msg)

    tool_uses: list = []
    stop_reason = "end_turn"
    text_buffer = ""
    asst_msg_uuid = str(uuid.uuid4())

    ui.console.print()
    ui.print_model_label(state.model_key)

    api_failed = False
    usage: dict = {}
    try:
        query_messages = prepare_query(
            state.session_id, mutableMessages, state.decision_map, state.model_id
        )
        thinking = False
        with ui.stream_response() as write:
            for event in client.converse_stream(query_messages, TOOLS, system, state.model_id):
                if event["type"] == "thinking":
                    thinking = True
                    ui.print_thinking_chunk(event["text"])
                elif event["type"] == "text":
                    if thinking:
                        ui.console.print()
                        thinking = False
                    text_buffer += event["text"]
                    write(event["text"])
                elif event["type"] == "tool_use":
                    tool_uses.append(event["tool_use"])
                elif event["type"] == "done":
                    stop_reason = event["stop_reason"]
                    usage = event.get("usage", {})
                    _print_cache_stats(usage)
        if usage:
            record_token_usage(state.session_id, usage, asst_msg_uuid, model_id)
    except KeyboardInterrupt:
        ui.print_interrupted()
        if text_buffer:
            asst_msg = AssistantMessage(
                uuid=asst_msg_uuid,
                message={"id": asst_msg_uuid, "role": "assistant", "model": model_id,
                         "content": [{"text": text_buffer}], "stop_reason": "interrupted"},
            )
            mutableMessages.append(asst_msg)
            recordTranscript(state.session_id, asst_msg)
        subagent.clear_force_model()
        return last_asst_uuid, last_skills
    except (ClientError, BotoCoreError, Exception) as e:
        code, msg = _format_api_error(e)
        ui.print_api_error(code, msg)
        _rollback_to_clean(mutableMessages)
        api_failed = True

    if api_failed:
        subagent.clear_force_model()
        return last_asst_uuid, last_skills

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
        asst_msg = AssistantMessage(
            uuid=asst_msg_uuid,
            parentUuid=last_user_uuid,
            message={"id": asst_msg_uuid, "role": "assistant", "model": model_id,
                     "content": assistant_content, "stop_reason": stop_reason},
            inputTokens=usage.get("inputTokens"),
            outputTokens=usage.get("outputTokens"),
        )
        mutableMessages.append(asst_msg)
        recordTranscript(state.session_id, asst_msg)
        last_asst_uuid = asst_msg_uuid

    # tool use loop
    iterations = 0
    while stop_reason == "tool_use" and tool_uses and iterations < config.MAX_ITERATIONS:
        iterations += 1

        raw_results, user_denied, instruction = _dispatch_tool_batch(tool_uses)

        # one UserMessage per tool result; parentUuid = the toolUseId it answers
        last_tr_uuid: str | None = None
        for tr in raw_results:
            tr_msg = UserMessage(
                message={"role": "user", "content": [{"toolResult": tr}]},
                toolResult=[tr],
                parentUuid=tr["toolUseId"],
                sourceToolAssistantUuid=asst_msg_uuid,
            )
            mutableMessages.append(tr_msg)
            recordTranscript(state.session_id, tr_msg)
            last_tr_uuid = tr_msg.uuid

        if user_denied:
            ui.console.print("[dim]turn ended by user[/dim]\n")
            break

        if instruction is not None:
            # bundle instruction as text alongside cancelled tool results;
            # normalizeMessagesForAPI merges consecutive UserMessages into one
            # API turn: [tool_result A: Cancelled, tool_result B: Cancelled, text: ...]
            instr_msg = UserMessage(
                message={"role": "user", "content": [{"text": instruction}]},
                parentUuid=last_tr_uuid,
            )
            mutableMessages.append(instr_msg)
            recordTranscript(state.session_id, instr_msg)
            last_tr_uuid = instr_msg.uuid
            ui.print_user_message(instruction, state.session_id)

        tool_uses = []
        stop_reason = "end_turn"
        text_buffer = ""
        asst_msg_uuid = str(uuid.uuid4())

        ui.console.print()
        ui.print_model_label(state.model_key)

        inner_failed = False
        usage = {}
        try:
            query_messages = normalizeMessagesForAPI(mutableMessages)
            thinking = False
            with ui.stream_response() as write:
                for event in client.converse_stream(query_messages, TOOLS, system, state.model_id):
                    if event["type"] == "thinking":
                        thinking = True
                        ui.print_thinking_chunk(event["text"])
                    elif event["type"] == "text":
                        if thinking:
                            ui.console.print()
                            thinking = False
                        text_buffer += event["text"]
                        write(event["text"])
                    elif event["type"] == "tool_use":
                        tool_uses.append(event["tool_use"])
                    elif event["type"] == "done":
                        stop_reason = event["stop_reason"]
                        usage = event.get("usage", {})
                        _print_cache_stats(usage)
        except KeyboardInterrupt:
            ui.print_interrupted()
            break
        except (ClientError, BotoCoreError, Exception) as e:
            code, msg = _format_api_error(e)
            ui.print_api_error(code, msg)
            _rollback_to_clean(mutableMessages)
            inner_failed = True

        if usage:
            record_token_usage(state.session_id, usage, asst_msg_uuid, model_id)

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
            asst_msg = AssistantMessage(
                uuid=asst_msg_uuid,
                parentUuid=last_tr_uuid,
                message={"id": asst_msg_uuid, "role": "assistant", "model": model_id,
                         "content": assistant_content, "stop_reason": stop_reason},
                inputTokens=usage.get("inputTokens"),
                outputTokens=usage.get("outputTokens"),
            )
            mutableMessages.append(asst_msg)
            recordTranscript(state.session_id, asst_msg)
            last_asst_uuid = asst_msg_uuid

    # fire background memory extraction — non-blocking, daemon thread
    fire_extract(mutableMessages, mem_cursor)
    subagent.clear_force_model()
    return last_asst_uuid, last_skills


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
            state.session_id = latest
            state.messages.extend(load_transcript(latest))
            ui.console.print(f"[dim]resumed session {latest} "
                             f"({len(state.messages)} messages)[/dim]")
        except Exception:
            state.session_id = session.next_session_id()
    else:
        state.session_id = 1

    # rebuild budget decision map from loaded transcript
    state.decision_map = rebuild_decision_map(state.messages)

    # memory subsystem — cursor persists across sessions
    _mem_dir = get_memory_dir()
    MemoryStore(_mem_dir).ensure_dir()
    _mem_cursor = MemoryCursor(_mem_dir)
    _mem_cursor.load()

    ui.init_history(state.session_id)

    last_skills = skills.scan_skills()
    system = prompt.build_system(tools=TOOLS)
    ui.print_header(state.model_key)

    mutableMessages = state.messages
    last_asst_uuid: str | None = None  # tracks uuid of most recent AssistantMessage

    if mutableMessages:
        # seed last_asst_uuid from loaded history
        for _m in reversed(mutableMessages):
            if isinstance(_m, AssistantMessage):
                last_asst_uuid = _m.uuid
                break
        ui.render_history(mutableMessages, state.session_id)

    while True:
        try:
            user_input = ui.get_input(state.session_id).strip()
        except (EOFError, KeyboardInterrupt):
            ui.console.print("\n[dim]Bye.[/dim]")
            break

        if not user_input:
            continue

        ui.print_user_message(user_input, state.session_id)

        if user_input.lower() == "exit":
            ui.console.print("[dim]Bye.[/dim]")
            break

        if commands.handle(user_input, state):
            continue

        last_asst_uuid, last_skills = _run_turn(
            user_input, state, mutableMessages, system, model_id,
            last_asst_uuid, last_skills, _mem_cursor,
        )


def run_headless(model_key: str = config.DEFAULT_MODEL, task_text: str = "") -> int:
    """Single-shot, non-interactive entrypoint for benchmark harnesses (eg. Terminal-Bench).

    Runs one task to completion (model response + full tool-use loop) and
    returns. No REPL, no approval prompts — the calling sandbox is assumed
    disposable. Returns process exit code.
    """
    model_id = config.AVAILABLE_MODELS.get(model_key)
    if not model_id:
        ui.print_error(f"Unknown model '{model_key}'. Available: {', '.join(config.AVAILABLE_MODELS)}")
        return 1

    task_text = task_text.strip()
    if not task_text:
        ui.print_error("No task instruction provided (empty stdin / --task-file).")
        return 1

    set_headless(True)

    session.ensure_history()
    state = commands.State(model_key=model_key, model_id=model_id)
    state.session_id = session.next_session_id()
    state.decision_map = rebuild_decision_map(state.messages)

    _mem_dir = get_memory_dir()
    MemoryStore(_mem_dir).ensure_dir()
    _mem_cursor = MemoryCursor(_mem_dir)
    _mem_cursor.load()

    last_skills = skills.scan_skills()
    system = prompt.build_system(tools=TOOLS)

    _run_turn(
        task_text, state, state.messages, system, model_id,
        None, last_skills, _mem_cursor,
    )
    return 0
