import os

AWS_PROFILE = os.environ.get("AWS_PROFILE", "default")
REGION = os.environ.get("AWS_REGION", "us-east-1")
MAX_TOKENS = 8096
MAX_ITERATIONS = 50
SUBAGENT_MAX_ITERATIONS = 30
MAX_PARALLEL_SUBAGENTS = 4

AVAILABLE_MODELS = {
    "sonnet": "us.anthropic.claude-sonnet-4-6",
    "haiku": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "opus": "us.anthropic.claude-opus-4-7",

    "nova-pro": "us.amazon.nova-pro-v1:0",
    "nova-lite": "us.amazon.nova-2-lite-v1:0",

    "chatgpt-120b": "openai.gpt-oss-120b-1:0",
    "chatgpt-20b": "openai.gpt-oss-20b-1:0",

    "deepseek": "deepseek.v3.2"
}
 
DEFAULT_MODEL = "sonnet"
