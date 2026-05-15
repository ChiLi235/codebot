import uuid
from botocore.exceptions import ClientError, BotoCoreError

from agent import client, config, ui, prompt, info, commands, session, subagent
from agent.tools import TOOLS, TOOL_MAP
from agent.tools.guard import check_valid
from agent.tools.human_check import check_human_eval
from agent.tools.shell import categorize
from agent.messages import (
    UserMessage, AssistantMessage, SystemMessage,
    normalizeMessagesForAPI, recordTranscript, load_transcript,
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
        return fn(**tu["input"]), "success", False, None
    except Exception as e:
        return f"Error: {e}", "error", False, None


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

    last_skills = info.scan_skills()
    system = prompt.build_system(tools=TOOLS)
    ui.print_header(state.model_key)

    # mutableMessages is state.messages — the in-memory list for next turn context
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
        current_skills = info.scan_skills()
        diff_text = info.format_skills_diff(*info.diff_skills(last_skills, current_skills), current_skills)
        if diff_text:
            meta_msg = UserMessage(
                message={"role": "user", "content": [{"text": diff_text}]},
                isMeta=True,
            )
            mutableMessages.append(meta_msg)
            recordTranscript(state.session_id, meta_msg)
            ui.console.print(f"[dim]{diff_text.splitlines()[0]}[/dim]")
        last_skills = current_skills

        # record the human turn — parent is last assistant
        last_user_uuid = str(uuid.uuid4())
        user_msg = UserMessage(
            uuid=last_user_uuid,
            message={"role": "user", "content": [{"text": user_input}]},
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
            with ui.stream_response() as write:
                for event in client.converse_stream(query_messages, TOOLS, system, state.model_id):
                    if event["type"] == "text":
                        text_buffer += event["text"]
                        write(event["text"])
                    elif event["type"] == "tool_use":
                        tool_uses.append(event["tool_use"])
                    elif event["type"] == "done":
                        stop_reason = event["stop_reason"]
                        usage = event.get("usage", {})
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
            continue
        except (ClientError, BotoCoreError, Exception) as e:
            code, msg = _format_api_error(e)
            ui.print_api_error(code, msg)
            _rollback_to_clean(mutableMessages)
            api_failed = True

        if api_failed:
            subagent.clear_force_model()
            continue

        ui.console.print()

        # build assistant content and record it
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
                with ui.stream_response() as write:
                    for event in client.converse_stream(query_messages, TOOLS, system, state.model_id):
                        if event["type"] == "text":
                            text_buffer += event["text"]
                            write(event["text"])
                        elif event["type"] == "tool_use":
                            tool_uses.append(event["tool_use"])
                        elif event["type"] == "done":
                            stop_reason = event["stop_reason"]
                            usage = event.get("usage", {})
            except KeyboardInterrupt:
                ui.print_interrupted()
                break
            except (ClientError, BotoCoreError, Exception) as e:
                code, msg = _format_api_error(e)
                ui.print_api_error(code, msg)
                _rollback_to_clean(mutableMessages)
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

        subagent.clear_force_model()
