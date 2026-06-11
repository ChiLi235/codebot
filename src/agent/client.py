import json
import time
import boto3
import openai
from openai import OpenAI
from botocore.exceptions import ClientError

from agent import config


def _make_client(profile: str, region: str):
    session = boto3.Session(profile_name=profile, region_name=region)
    return session.client("bedrock-runtime")


_client_cache: dict = {}


def _get_client(profile: str = config.AWS_PROFILE, region: str = config.REGION):
    key = (profile, region)
    if key not in _client_cache:
        _client_cache[key] = _make_client(profile, region)
    return _client_cache[key]


def _with_retry(fn, max_retries: int = 5):
    retryable = {"ThrottlingException", "ModelStreamErrorException", "ServiceUnavailableException"}
    delay = 1.0
    for attempt in range(max_retries):
        try:
            return fn()
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if code not in retryable or attempt == max_retries - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 30)


def _supports_cache(model_id: str) -> bool:
    return "anthropic.claude" in model_id


def _cached_system(system: list) -> list:
    return list(system) + [{"cachePoint": {"type": "default"}}]


def _cached_tool_config(tools: list) -> dict:
    return {"tools": list(tools) + [{"cachePoint": {"type": "default"}}]}


def _cached_messages(messages: list) -> list:
    if not messages:
        return messages
    result = list(messages)
    for i in reversed(range(len(result))):
        if result[i].get("role") == "user":
            msg = dict(result[i])
            content = list(msg.get("content", []))
            content.append({"cachePoint": {"type": "default"}})
            msg["content"] = content
            result[i] = msg
            break
    return result


# ── DeepSeek (own API, OpenAI-compatible — not Bedrock) ──────────────────────

_deepseek_client_cache: OpenAI | None = None

_FINISH_REASON_MAP = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "length": "max_tokens",
}


def _is_deepseek_model(model_id: str) -> bool:
    return model_id.startswith("deepseek")


def _get_deepseek_client() -> OpenAI:
    global _deepseek_client_cache
    if _deepseek_client_cache is None:
        _deepseek_client_cache = OpenAI(
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.DEEPSEEK_BASE_URL,
        )
    return _deepseek_client_cache


def _with_retry_deepseek(fn, max_retries: int = 5):
    retryable = (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError)
    delay = 1.0
    for attempt in range(max_retries):
        try:
            return fn()
        except retryable:
            if attempt == max_retries - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 30)


def _messages_to_openai(messages: list, system: list) -> list[dict]:
    """Convert Bedrock-style messages/system into OpenAI chat format."""
    out: list[dict] = []
    sys_text = "\n\n".join(b["text"] for b in system if "text" in b)
    if sys_text:
        out.append({"role": "system", "content": sys_text})

    for msg in messages:
        content = msg.get("content", [])
        if msg["role"] == "assistant":
            text_parts = []
            tool_calls = []
            for block in content:
                if "text" in block:
                    text_parts.append(block["text"])
                elif "toolUse" in block:
                    tu = block["toolUse"]
                    tool_calls.append({
                        "id": tu["toolUseId"],
                        "type": "function",
                        "function": {
                            "name": tu["name"],
                            "arguments": json.dumps(tu.get("input", {})),
                        },
                    })
            entry: dict = {"role": "assistant", "content": "".join(text_parts) or None}
            if tool_calls:
                entry["tool_calls"] = tool_calls
            out.append(entry)
        else:
            text_parts = []
            for block in content:
                if "toolResult" in block:
                    tr = block["toolResult"]
                    tr_text = "".join(b.get("text", "") for b in tr.get("content", []) if "text" in b)
                    out.append({"role": "tool", "tool_call_id": tr["toolUseId"], "content": tr_text})
                elif "text" in block:
                    text_parts.append(block["text"])
            if text_parts:
                out.append({"role": "user", "content": "".join(text_parts)})
    return out


def _tools_to_openai(tools: list) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["toolSpec"]["name"],
                "description": t["toolSpec"]["description"],
                "parameters": t["toolSpec"]["inputSchema"]["json"],
            },
        }
        for t in tools
    ]


def _deepseek_thinking_kwargs(thinking: str | None, reasoning_effort: str | None) -> dict:
    mode = thinking or config.DEEPSEEK_THINKING
    out: dict = {"extra_body": {"thinking": {"type": mode}}}
    if mode == "enabled":
        effort = reasoning_effort or config.DEEPSEEK_REASONING_EFFORT
        if effort:
            out["reasoning_effort"] = effort
    return out


def _deepseek_usage(usage) -> dict:
    if usage is None:
        return {}
    return {
        "inputTokens": usage.prompt_tokens,
        "outputTokens": usage.completion_tokens,
        "cacheReadInputTokens": getattr(usage, "prompt_cache_hit_tokens", 0) or 0,
        "cacheWriteInputTokens": 0,
    }


def _deepseek_converse(messages: list, tools: list, system: list, model_id: str,
                       max_tokens: int | None = None,
                       thinking: str | None = None,
                       reasoning_effort: str | None = None) -> dict:
    client = _get_deepseek_client()
    kwargs = dict(
        model=model_id,
        messages=_messages_to_openai(messages, system),
        max_tokens=max_tokens or config.MAX_TOKENS,
    )
    kwargs.update(_deepseek_thinking_kwargs(thinking, reasoning_effort))
    if tools:
        kwargs["tools"] = _tools_to_openai(tools)

    def _call():
        return client.chat.completions.create(**kwargs)

    response = _with_retry_deepseek(_call)
    choice = response.choices[0]

    content_blocks: list[dict] = []
    if choice.message.content:
        content_blocks.append({"text": choice.message.content})
    for tc in choice.message.tool_calls or []:
        try:
            parsed = json.loads(tc.function.arguments) if tc.function.arguments else {}
        except json.JSONDecodeError:
            parsed = {"raw": tc.function.arguments}
        content_blocks.append({"toolUse": {
            "toolUseId": tc.id, "name": tc.function.name, "input": parsed,
        }})

    return {
        "output": {"message": {"role": "assistant", "content": content_blocks}},
        "stopReason": _FINISH_REASON_MAP.get(choice.finish_reason, "end_turn"),
        "usage": _deepseek_usage(response.usage),
    }


def _deepseek_converse_stream(messages: list, tools: list, system: list, model_id: str,
                              thinking: str | None = None,
                              reasoning_effort: str | None = None):
    """Yield reconstructed stream events: text chunks and complete tool_use blocks."""
    client = _get_deepseek_client()
    kwargs = dict(
        model=model_id,
        messages=_messages_to_openai(messages, system),
        max_tokens=config.MAX_TOKENS,
        stream=True,
        stream_options={"include_usage": True},
    )
    kwargs.update(_deepseek_thinking_kwargs(thinking, reasoning_effort))
    if tools:
        kwargs["tools"] = _tools_to_openai(tools)

    def _call():
        return client.chat.completions.create(**kwargs)

    stream = _with_retry_deepseek(_call)

    tool_calls: dict[int, dict] = {}
    stop_reason = "end_turn"
    usage: dict = {}

    for chunk in stream:
        if chunk.usage:
            usage = _deepseek_usage(chunk.usage)
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        delta = choice.delta
        if delta:
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                yield {"type": "thinking", "text": reasoning}
        if delta and delta.content:
            yield {"type": "text", "text": delta.content}
        if delta and delta.tool_calls:
            for tc in delta.tool_calls:
                acc = tool_calls.setdefault(tc.index, {"id": None, "name": None, "arguments": ""})
                if tc.id:
                    acc["id"] = tc.id
                if tc.function:
                    if tc.function.name:
                        acc["name"] = tc.function.name
                    if tc.function.arguments:
                        acc["arguments"] += tc.function.arguments
        if choice.finish_reason:
            stop_reason = _FINISH_REASON_MAP.get(choice.finish_reason, "end_turn")

    for idx in sorted(tool_calls):
        tc = tool_calls[idx]
        try:
            parsed = json.loads(tc["arguments"]) if tc["arguments"] else {}
        except json.JSONDecodeError:
            parsed = {"raw": tc["arguments"]}
        yield {"type": "tool_use", "tool_use": {"toolUseId": tc["id"], "name": tc["name"], "input": parsed}}

    yield {"type": "done", "stop_reason": stop_reason, "usage": usage}


# ── Bedrock ───────────────────────────────────────────────────────────────────

def converse(messages: list, tools: list, system: list, model_id: str,
             max_tokens: int | None = None,
             thinking: str | None = None,
             reasoning_effort: str | None = None) -> dict:
    if _is_deepseek_model(model_id):
        return _deepseek_converse(messages, tools, system, model_id, max_tokens,
                                  thinking, reasoning_effort)

    client = _get_client()
    kwargs = dict(
        modelId=model_id,
        messages=messages,
        system=system,
        inferenceConfig={"maxTokens": max_tokens or config.MAX_TOKENS},
    )
    if tools:
        kwargs["toolConfig"] = {"tools": tools}

    def _call():
        return client.converse(**kwargs)

    return _with_retry(_call)


def converse_stream(messages: list, tools: list, system: list, model_id: str,
                    thinking: str | None = None,
                    reasoning_effort: str | None = None):
    """Yield reconstructed stream events: text chunks and complete tool_use blocks."""
    if _is_deepseek_model(model_id):
        yield from _deepseek_converse_stream(messages, tools, system, model_id,
                                             thinking, reasoning_effort)
        return

    client = _get_client()

    if _supports_cache(model_id):
        system = _cached_system(system)
        messages = _cached_messages(messages)

    kwargs = dict(
        modelId=model_id,
        messages=messages,
        system=system,
        inferenceConfig={"maxTokens": config.MAX_TOKENS},
    )
    if tools:
        kwargs["toolConfig"] = _cached_tool_config(tools) if _supports_cache(model_id) else {"tools": tools}

    def _call():
        return client.converse_stream(**kwargs)

    response = _with_retry(_call)
    stream = response["stream"]

    current_tool: dict | None = None
    current_tool_input = ""
    stop_reason = "end_turn"
    usage = {}

    for event in stream:
        if "messageStart" in event:
            pass

        elif "contentBlockStart" in event:
            block = event["contentBlockStart"].get("start", {})
            if "toolUse" in block:
                current_tool = {
                    "toolUseId": block["toolUse"]["toolUseId"],
                    "name": block["toolUse"]["name"],
                }
                current_tool_input = ""

        elif "contentBlockDelta" in event:
            delta = event["contentBlockDelta"]["delta"]
            if "text" in delta:
                yield {"type": "text", "text": delta["text"]}
            elif "toolUse" in delta:
                current_tool_input += delta["toolUse"].get("input", "")

        elif "contentBlockStop" in event:
            if current_tool is not None:
                try:
                    parsed = json.loads(current_tool_input) if current_tool_input else {}
                except json.JSONDecodeError:
                    parsed = {"raw": current_tool_input}
                current_tool["input"] = parsed
                yield {"type": "tool_use", "tool_use": current_tool}
                current_tool = None
                current_tool_input = ""

        elif "messageStop" in event:
            stop_reason = event["messageStop"].get("stopReason", "end_turn")

        elif "metadata" in event:
            usage = event["metadata"].get("usage", {})

    yield {"type": "done", "stop_reason": stop_reason, "usage": usage}
