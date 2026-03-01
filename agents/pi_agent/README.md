# pi Agent for HAL

This is a HAL harness wrapper for [pi](https://github.com/mariozechner/pi) - the AI coding assistant.

## Setup

Install pi globally:
```bash
npm install -g @mariozechner/pi-coding-agent
```

Verify installation:
```bash
pi --version
```

## Supported Benchmarks

- **GAIA** - General Q&A tasks (pi can search the web and run code)
- **swebench_verified_mini** - Code fix tasks
- Any other benchmark with code/problem tasks

## Basic Usage

### GAIA with pi
```bash
hal-eval --benchmark gaia \
  --agent_dir agents/pi_agent \
  --agent_function main.run \
  --agent_name "pi_gaia" \
  -A model_name="gpt-4o" \
  -A provider="openai" \
  --max_tasks 5
```

### swebench_verified_mini with pi
```bash
hal-eval --benchmark swebench_verified_mini \
  --agent_dir agents/pi_agent \
  --agent_function main.run \
  --agent_name "pi_swebench" \
  -A model_name="claude-opus-4-5" \
  -A provider="anthropic" \
  --max_tasks 3
```

### With local model
```bash
OPENAI_API_KEY="local" \
OPENAI_BASE_URL="http://localhost:9999/v1" \
hal-eval --benchmark gaia \
  --agent_dir agents/pi_agent \
  --agent_function main.run \
  --agent_name "pi_local" \
  -A model_name="Qwen/Qwen3-Next-80B-A3B-Instruct-AutoRound" \
  -A provider="openai" \
  --max_tasks 5
```

## Agent Arguments (-A)

| Argument | Description | Example |
|----------|-------------|---------|
| `model_name` | **Required**. Model ID or pattern | `gpt-4o`, `claude-opus-4-5` |
| `provider` | **Optional**. Provider override | `openai`, `anthropic`, `google` |
| `thinking` | **Optional**. Thinking level for reasoning | `off`, `low`, `medium`, `high`, `xhigh` |
| `tools` | **Optional**. Comma-separated tools to enable | `read,bash,edit,write,grep,find,ls` |
| `system_prompt` | **Optional**. Custom system prompt | `"You are an expert Python developer..."` |

## Examples

### With thinking (for reasoning models)
```bash
hal-eval --benchmark gaia \
  --agent_dir agents/pi_agent \
  --agent_function main.run \
  --agent_name "pi_gaia_thinking" \
  -A model_name="o1" \
  -A provider="openai" \
  -A thinking="high" \
  --max_tasks 5
```

### Custom tools
```bash
hal-eval --benchmark swebench_verified_mini \
  --agent_dir agents/pi_agent \
  --agent_function main.run \
  --agent_name "pi_swebench" \
  -A model_name="gpt-4o" \
  -A tools="read,bash,edit,write" \
  --max_tasks 3
```

### With custom system prompt
```bash
hal-eval --benchmark gaia \
  --agent_dir agents/pi_agent \
  --agent_function main.run \
  --agent_name "pi_custom" \
  -A model_name="claude-opus-4-5" \
  -A system_prompt="You are a world-class software engineer. Be concise." \
  --max_tasks 5
```

## How It Works

The wrapper:
1. Extracts the task prompt from the benchmark input (handles GAIA, swebench, etc.)
2. Builds a pi CLI command with the specified model and options
3. Runs pi in non-interactive mode (`-p`) and captures output
4. Returns the result to hal-eval for scoring

## Notes

- Sessions are disabled (`--no-session`) to avoid file clutter during evaluation
- Output is captured as plain text
- Timeouts are set to 600 seconds (10 minutes) per task
- For local models, set `OPENAI_API_KEY` and `OPENAI_BASE_URL` env vars before running
