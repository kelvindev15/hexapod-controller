# Interactive Runner & Robot/Host Architecture

## Interactive Runner

Use `interactive_runner.py` to control the robot from a terminal.

### Run

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

### LLM Providers

- OpenAI (`--llm-provider openai`)
  - Required env var: `OPENAI_API_KEY`
  - Optional model override: `--llm-model gpt-4o-mini`

- Gemini (`--llm-provider gemini`)
  - Required env var: `GEMINI_API_KEY`
  - Optional model override: `--llm-model gemini-2.0-flash`

- Ollama (`--llm-provider ollama`)
  - Requires local Ollama server + model available
  - Optional model override: `--llm-model llava`

### LangSmith Tracing

- Set `LANGSMITH_API_KEY` to enable LangSmith traces for LLM calls.
- Optional: set `LANGSMITH_PROJECT` to group runs under a custom project name.
- The runner enables LangSmith tracing automatically when an API key is present.

### CLI Commands vs Natural Language

- Input starting with `/` is treated as a direct command:
  - `/help`, `/state`, `/stop`, `/relax`, `/balance`
  - `/snapshot [output_path]` (captures the current camera frame to a file)
  - `/walk <y> [ttl] [speed] [gait_type]`
  - `/rotate <angle> [ttl] [speed]`
  - `/attitude <roll> <pitch> <yaw>`
  - `/position <x> <y> <z>`
  - `/quit`

- Any input without `/` is treated as a natural-language goal and forwarded to the LLM controller.

### Command History

- The prompt supports terminal history navigation with `↑`/`↓` arrows.
- History is persisted across runs in `~/.hexapod_history`.

### Example Goal Prompts

- `Move forward slowly until you are about 30 cm from obstacles, then stop.`
- `Rotate right about 45 degrees and hold position.`
- `Take a cautious step backward and stop if space behind is limited.`
- `Find a safer stance and relax once stable.`

### Useful Flags

- `--max-iterations 20` sets max LLM action iterations per goal.
- `--chat-history-turns <n>` controls how many previous chat turns are retained (`default: full history`, `0: current turn only`).
- `--llm-model <name>` overrides the default model for the selected provider.
- `--mode local|dry-run|remote` controls execution backend.
- `--robot-url <url>` sets robot service URL in remote mode.

### Env Vars

- `HEXAPOD_ROBOT_URL` sets default robot URL for remote mode (`http://127.0.0.1:8080` by default).
- `LANGSMITH_API_KEY` enables LangSmith tracing.
- `LANGSMITH_PROJECT` sets the LangSmith project name used by traces.

## Robot/Host Architecture Split

### Roles

- Robot device (`server/robot_service.py`):
  - Owns one `MotionExecutor` instance
  - Executes hardware motion via `Control` + drivers
  - Enforces runtime safety checks before/while executing actions
- Host machine (`interactive_runner.py --mode remote`):
  - Runs LLM reasoning and orchestration (`LLMRobotController`)
  - Sends motion actions to robot over HTTP

### Shared API Contract

Implemented in `server/core/api_contract.py`.

- `Action` JSON payload:
  - `type`: action type string (existing `ActionType` values, e.g. `walk`, `stop`, `balance`, `complete`)
  - `params`: object
  - `ttl`: number
  - `metadata`: object
- `WorldState` JSON payload:
  - `roll`, `pitch`, `yaw`, `distance`, `is_balancing`, `timestamp`, `is_safe`, `safety_reason`
  - `current_action`: action type string or `null`

### Robot Service Endpoints

- `POST /actions` -> submit `Action` payload
- `POST /actions/stop` -> force stop action
- `GET /state` -> current `WorldState`
- `GET /health` -> basic service health

### Run Robot Service (device)

```bash
python server/robot_service.py --host 0.0.0.0 --port 8080
```

### Run Host Controller (machine)

```bash
HEXAPOD_ROBOT_URL=http://<robot-ip>:8080 python interactive_runner.py --mode remote --llm-provider openai
```

Optional explicit URL flag:

```bash
python interactive_runner.py --mode remote --robot-url http://<robot-ip>:8080 --llm-provider openai
```

Optional system prompt file:

```bash
python interactive_runner.py --mode remote --robot-url http://<robot-ip>:8080 --llm-provider openai --system-prompt-file ./system_prompt.txt
```

### Environment / Config

- `HEXAPOD_ROBOT_URL`: robot endpoint base URL for host remote mode (default `http://127.0.0.1:8080`)
- `HEXAPOD_SYSTEM_PROMPT_FILE`: optional path to a text file used as the LLM system prompt
- `HEXAPOD_SYSTEM_PROMPT`: optional inline system prompt text (takes precedence over file)
- LLM provider keys remain unchanged:
  - OpenAI: `OPENAI_API_KEY`
  - Gemini: `GEMINI_API_KEY`
  - Ollama: local server/model availability
