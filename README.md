# codebot

Personal CLI coding agent powered by Claude and other models via Amazon Bedrock.

## What it does

Interactive terminal agent that can read, write, and edit files, run shell commands, search code, and spawn subagents — all from a chat interface. Supports multiple models and auto-manages context when conversations get long.

## Requirements

- Python 3.11+
- AWS credentials configured with Bedrock access
- [uv](https://github.com/astral-sh/uv)

## Install

```bash
git clone https://github.com/ChiLi235/codebot.git
cd codebot
uv sync
```

## Run

```bash
uv run codebot
```

Or with a specific model:

```bash
uv run codebot --model haiku
```

## Available models

| Key | Model |
|-----|-------|
| `sonnet` | Claude Sonnet (default) |
| `haiku` | Claude Haiku |
| `opus` | Claude Opus |
| `nova-pro` | Amazon Nova Pro |
| `nova-lite` | Amazon Nova Lite |
| `deepseek` | DeepSeek V3 |

## Commands

Type these inside the chat:

| Command | Description |
|---------|-------------|
| `/model <key>` | Switch model mid-session |
| `exit` | Quit |

## AWS setup

codebot uses your default AWS profile. Override with environment variables:

```bash
AWS_PROFILE=myprofile AWS_REGION=us-west-2 uv run codebot
```

Make sure your profile has `bedrock:InvokeModel` and `bedrock:InvokeModelWithResponseStream` permissions.
