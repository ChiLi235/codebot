import time
import boto3
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


def converse(messages: list, tools: list, system: list, model_id: str,
             max_tokens: int | None = None) -> dict:
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


def converse_stream(messages: list, tools: list, system: list, model_id: str):
    """Yield reconstructed stream events: text chunks and complete tool_use blocks."""
    client = _get_client()
    kwargs = dict(
        modelId=model_id,
        messages=messages,
        system=system,
        inferenceConfig={"maxTokens": config.MAX_TOKENS},
    )
    if tools:
        kwargs["toolConfig"] = {"tools": tools}

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
                import json
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
