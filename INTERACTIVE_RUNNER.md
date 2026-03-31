# Interactive Runner

Use `interactive_runner.py` to control the robot from a terminal.

## Run

- Live hardware mode:

```bash
python interactive_runner.py --mode local --llm-provider openai
```

- Dry-run mode (no hardware):

```bash
python interactive_runner.py --mode dry-run --llm-provider openai
```

- Remote host mode (LLM on host, motion on robot device):

```bash
HEXAPOD_ROBOT_URL=http://<robot-ip>:8080 python interactive_runner.py --mode remote --llm-provider openai
```

Backwards-compatible shortcut for dry run still works:

```bash
python interactive_runner.py --dry-run --llm-provider openai
```

## LLM Providers

- OpenAI (`--llm-provider openai`)
  - Required env var: `OPENAI_API_KEY`
  - Optional model override: `--llm-model gpt-4o-mini`

- Gemini (`--llm-provider gemini`)
  - Required env var: `GEMINI_API_KEY`
  - Optional model override: `--llm-model gemini-2.0-flash`

- Ollama (`--llm-provider ollama`)
  - Requires local Ollama server + model available
  - Optional model override: `--llm-model llava`

## CLI Commands vs Natural Language

- Input starting with `/` is treated as a direct command:
  - `/help`, `/state`, `/stop`, `/relax`, `/balance`
  - `/snapshot [output_path]` (captures the current camera frame to a file)
  - `/walk <y> [ttl] [speed] [gait_type]`
  - `/rotate <angle> [ttl] [speed]`
  - `/attitude <roll> <pitch> <yaw>`
  - `/position <x> <y> <z>`
  - `/quit`

- Any input without `/` is treated as a natural-language goal and forwarded to the LLM controller.

## Command History

- The prompt supports terminal history navigation with `↑`/`↓` arrows.
- History is persisted across runs in `~/.hexapod_history`.

## Example Goal Prompts

- `Move forward slowly until you are about 30 cm from obstacles, then stop.`
- `Rotate right about 45 degrees and hold position.`
- `Take a cautious step backward and stop if space behind is limited.`
- `Find a safer stance and relax once stable.`

## Useful Flags

- `--max-iterations 20` sets max LLM action iterations per goal.
- `--llm-model <name>` overrides the default model for the selected provider.
- `--mode local|dry-run|remote` controls execution backend.
- `--robot-url <url>` sets robot service URL in remote mode.

## Env Vars

- `HEXAPOD_ROBOT_URL` sets default robot URL for remote mode (`http://127.0.0.1:8080` by default).
