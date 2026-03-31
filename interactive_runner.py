#!/usr/bin/env python3
import argparse
import asyncio
import atexit
import os
import readline
import shlex
import sys
import time
from dataclasses import asdict
from datetime import datetime

import cv2
import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from common.utils.logging_config import bootstrap_logging
from server.core.motion_schema import Action, ActionType, WorldState


class DryRunExecutor:
    def __init__(self):
        self._state = WorldState()
        self._running = False

    def start(self):
        self._running = True

    def stop(self):
        self._running = False

    def submit_action(self, action: Action):
        if not self._running:
            raise RuntimeError("Executor is not running")
        self._state.current_action = action.type
        self._state.timestamp = time.time()
        self._state.is_safe = True
        self._state.safety_reason = ""

    def get_state(self) -> WorldState:
        return self._state


class InteractiveRobotBridge:
    def __init__(self, executor):
        self.executor = executor

    def goFront(self, distance: float = 1.0) -> None:
        self.executor.submit_action(
            Action(
                type=ActionType.WALK,
                params={"x": 0, "y": abs(distance), "angle": 0, "gait_type": "1", "speed": 5},
                ttl=max(0.25, min(4.0, abs(distance) / 15.0)),
            )
        )

    def goBack(self, distance: float = 1.0) -> None:
        self.executor.submit_action(
            Action(
                type=ActionType.WALK,
                params={"x": 0, "y": -abs(distance), "angle": 0, "gait_type": "1", "speed": 5},
                ttl=max(0.25, min(4.0, abs(distance) / 15.0)),
            )
        )

    def rotateRight(self, angle: float = 45.0) -> None:
        self.executor.submit_action(
            Action(
                type=ActionType.WALK,
                params={"x": 0, "y": 0, "angle": abs(angle), "gait_type": "1", "speed": 4},
                ttl=max(0.25, min(4.0, abs(angle) / 15.0)),
            )
        )

    def rotateLeft(self, angle: float = 45.0) -> None:
        self.executor.submit_action(
            Action(
                type=ActionType.WALK,
                params={"x": 0, "y": 0, "angle": -abs(angle), "gait_type": "1", "speed": 4},
                ttl=max(0.25, min(4.0, abs(angle) / 15.0)),
            )
        )

    def stop(self) -> None:
        self.executor.submit_action(Action(type=ActionType.STOP))

    def getCameraImage(self):
        try:
            from server.drivers.camera import Camera
            camera = Camera()
            image = camera.capture_array()
            camera.close()
            if image is not None:
                return image
            return np.zeros((64, 64, 3), dtype=np.uint8)
        except Exception as exc:
            # Fallback to black image if camera is not available
            return np.zeros((64, 64, 3), dtype=np.uint8)

    def getLidarImage(self, fov_degrees: int, offset_degrees: int = 0):
        _ = offset_degrees
        state = self.executor.get_state()
        distance = float(state.distance if state.distance is not None else 999.0)
        count = max(1, int(fov_degrees))
        return [distance] * count

    def getFrontLidarImage(self):
        return self.getLidarImage(90, 0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactive controller for the hexapod motion executor",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use an in-memory executor instead of hardware motion executor",
    )
    parser.add_argument(
        "--mode",
        choices=["local", "dry-run", "remote"],
        default="local",
        help="Runner mode: local hardware executor, dry-run executor, or remote robot service",
    )
    parser.add_argument(
        "--robot-url",
        default=os.environ.get("HEXAPOD_ROBOT_URL", "http://127.0.0.1:8080"),
        help="Robot service base URL used in --mode remote",
    )
    parser.add_argument(
        "--llm-provider",
        choices=["openai", "gemini", "ollama"],
        default="openai",
        help="LLM backend used for natural language goals",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="Optional model override for the selected LLM provider",
    )
    parser.add_argument(
        "--system-prompt-file",
        default=os.environ.get("HEXAPOD_SYSTEM_PROMPT_FILE", None),
        help="Path to a text file used as LLM system prompt",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=20,
        help="Max LLM reasoning/action iterations per natural language goal",
    )
    return parser


def print_help() -> None:
    print(
        """
Commands:
    /help
    /state
    /snapshot [output_path]
    /stop
    /relax
    /balance
    /walk <y> [ttl] [speed] [gait_type]
    /rotate <angle> [ttl] [speed]
    /attitude <roll> <pitch> <yaw>
    /position <x> <y> <z>
    /quit | /exit

Natural language goals:
    Any input that does not start with '/' is sent to the LLM controller.
    Press Ctrl+C while an LLM goal is running to interrupt that session.

Examples:
    Move forward carefully until you are 30 cm from the obstacle.
    Turn right and then stop.
    /walk 20 3 5 1
    /state
""".strip()
    )


def format_state(state: WorldState) -> str:
    state_dict = asdict(state)
    action = state_dict.get("current_action")
    if action is not None:
        state_dict["current_action"] = action.value
    return (
        f"action={state_dict['current_action']} "
        f"distance={state_dict['distance']:.1f}cm "
        f"roll={state_dict['roll']:.1f} pitch={state_dict['pitch']:.1f} yaw={state_dict['yaw']:.1f} "
        f"safe={state_dict['is_safe']} reason='{state_dict['safety_reason']}'"
    )


def parse_command(line: str) -> list[str]:
    return shlex.split(line.strip())


def save_snapshot(robot_bridge, output_path: str | None = None) -> str:
    image = robot_bridge.getCameraImage()
    if image is None:
        raise RuntimeError("Camera image is unavailable")

    if not isinstance(image, np.ndarray):
        image = np.array(image, dtype=np.uint8)

    if image.size == 0:
        raise RuntimeError("Camera image is empty")

    if output_path:
        snapshot_path = os.path.abspath(os.path.expanduser(output_path))
        directory = os.path.dirname(snapshot_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
    else:
        snapshots_dir = os.path.join(PROJECT_ROOT, "snapshots")
        os.makedirs(snapshots_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_path = os.path.join(snapshots_dir, f"snapshot_{timestamp}.jpg")

    ok = cv2.imwrite(snapshot_path, image)
    if not ok:
        raise RuntimeError(f"Failed to write snapshot to {snapshot_path}")

    return snapshot_path


def to_float(raw: str, name: str) -> float:
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid {name}: {raw}") from exc


def make_action(tokens: list[str]) -> Action:
    command = tokens[0].lower()

    if command == "stop":
        return Action(type=ActionType.STOP)
    if command == "relax":
        return Action(type=ActionType.RELAX)
    if command == "balance":
        return Action(type=ActionType.BALANCE)

    if command == "walk":
        if len(tokens) < 2:
            raise ValueError("Usage: walk <y> [ttl] [speed] [gait_type]")
        y = to_float(tokens[1], "y")
        ttl = to_float(tokens[2], "ttl") if len(tokens) >= 3 else 0.0
        speed = int(to_float(tokens[3], "speed")) if len(tokens) >= 4 else 5
        gait_type = str(tokens[4]) if len(tokens) >= 5 else "1"
        return Action(
            type=ActionType.WALK,
            params={"x": 0, "y": y, "angle": 0, "gait_type": gait_type, "speed": speed},
            ttl=ttl,
        )

    if command == "rotate":
        if len(tokens) < 2:
            raise ValueError("Usage: rotate <angle> [ttl] [speed]")
        angle = to_float(tokens[1], "angle")
        ttl = to_float(tokens[2], "ttl") if len(tokens) >= 3 else 0.0
        speed = int(to_float(tokens[3], "speed")) if len(tokens) >= 4 else 4
        return Action(
            type=ActionType.WALK,
            params={"x": 0, "y": 0, "angle": angle, "gait_type": "1", "speed": speed},
            ttl=ttl,
        )

    if command == "attitude":
        if len(tokens) != 4:
            raise ValueError("Usage: attitude <roll> <pitch> <yaw>")
        return Action(
            type=ActionType.ATTITUDE,
            params={
                "roll": to_float(tokens[1], "roll"),
                "pitch": to_float(tokens[2], "pitch"),
                "yaw": to_float(tokens[3], "yaw"),
            },
        )

    if command == "position":
        if len(tokens) != 4:
            raise ValueError("Usage: position <x> <y> <z>")
        return Action(
            type=ActionType.POSITION,
            params={
                "x": to_float(tokens[1], "x"),
                "y": to_float(tokens[2], "y"),
                "z": to_float(tokens[3], "z"),
            },
        )

    raise ValueError(f"Unknown command: {tokens[0]}")


def create_chat(provider: str, model_name: str | None):
    from common.llm.chats import GeminiChat, OllamaChat, OpenAIChat

    if provider == "openai":
        return OpenAIChat(model_name=model_name or "gpt-4o-mini")
    if provider == "gemini":
        return GeminiChat(model_name=model_name or "gemini-2.0-flash")
    return OllamaChat(model_name=model_name or "llava")


def setup_cli_history() -> None:
    history_path = os.path.expanduser("~/.hexapod_history")
    try:
        if os.path.exists(history_path):
            readline.read_history_file(history_path)
    except Exception:
        return

    readline.set_history_length(1000)

    def _save_history() -> None:
        try:
            readline.write_history_file(history_path)
        except Exception:
            pass

    atexit.register(_save_history)


def run_llm_goal(
    llm_event_loop: asyncio.AbstractEventLoop,
    llm_controller,
    goal: str,
    max_iterations: int,
) -> bool:
    task = llm_event_loop.create_task(llm_controller.ask(goal, maxIterations=max_iterations))
    interrupted = False
    try:
        llm_event_loop.run_until_complete(task)
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupt requested. Stopping current LLM session...")
        try:
            llm_controller.interrupt()
        except Exception:
            pass
        try:
            llm_event_loop.run_until_complete(task)
        except asyncio.CancelledError:
            pass
        except Exception:
            pass
    return interrupted


def main() -> int:
    bootstrap_logging()
    setup_cli_history()
    args = build_parser().parse_args()

    mode = "dry-run" if args.dry_run else args.mode

    if mode == "dry-run":
        executor = DryRunExecutor()
        print("Started in dry-run mode (no hardware access).")
        robot_bridge = InteractiveRobotBridge(executor)
    elif mode == "remote":
        from common.robot.RemoteRobotController import RemoteRobotController

        executor = RemoteRobotController(base_url=args.robot_url)
        robot_bridge = executor
        print(f"Started in remote mode (robot service: {args.robot_url}).")
    else:
        try:
            from server.core.motion_executor import MotionExecutor

            executor = MotionExecutor()
            print("Started in live mode (hardware motion executor).")
            robot_bridge = InteractiveRobotBridge(executor)
        except Exception as exc:
            print(f"Failed to initialize live executor: {exc}")
            print("Retry with --dry-run to test interactively without hardware.")
            return 1

    if hasattr(executor, "start") and callable(getattr(executor, "start", None)):
        executor.start()

    llm_controller = None
    llm_event_loop = None
    llm_init_error = None
    try:
        from common.robot.LLMRobotController import LLMRobotController

        llm_event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(llm_event_loop)
        chat = create_chat(args.llm_provider, args.llm_model)
        llm_controller = LLMRobotController(
            robotController=robot_bridge,
            chat=chat,
            motionExecutor=executor,
            system_prompt_file=args.system_prompt_file,
        )
        model_name = chat.get_model_name()
        print(f"LLM control enabled: provider={args.llm_provider} model={model_name}")
    except Exception as exc:
        llm_init_error = f"{type(exc).__name__}: {exc}"
        print(f"LLM initialization failed ({llm_init_error}).")
        if args.llm_provider == "gemini":
            print("Gemini setup: set GEMINI_API_KEY (or GOOGLE_API_KEY) and install langchain-google-genai.")
        elif args.llm_provider == "openai":
            print("OpenAI setup: set OPENAI_API_KEY and install langchain-openai.")
        elif args.llm_provider == "ollama":
            print("Ollama setup: run local Ollama service and ensure the model exists.")
        print("Manual slash commands remain available.")

    print_help()

    try:
        while True:
            line = input("hexapod> ").strip()
            if not line:
                continue

            if line.startswith("/"):
                tokens = parse_command(line[1:])
                if not tokens:
                    continue
                command = tokens[0].lower()

                if command in {"quit", "exit"}:
                    break
                if command == "help":
                    print_help()
                    continue
                if command == "state":
                    print(format_state(executor.get_state()))
                    continue
                if command == "snapshot":
                    if len(tokens) > 2:
                        print("Usage: /snapshot [output_path]")
                        continue
                    output_path = tokens[1] if len(tokens) == 2 else None
                    saved_path = save_snapshot(robot_bridge, output_path)
                    print(f"saved snapshot to {saved_path}")
                    continue

                action = make_action(tokens)
                executor.submit_action(action)
                print(f"submitted {action.type.value} params={action.params} ttl={action.ttl}")
                print(format_state(executor.get_state()))
                continue

            if llm_controller is None:
                if llm_init_error:
                    print(f"LLM unavailable: {llm_init_error}")
                else:
                    print("LLM is not configured. Use slash commands or fix LLM setup.")
                continue

            print(f"goal> {line}")
            interrupted = run_llm_goal(llm_event_loop, llm_controller, line, args.max_iterations)
            if interrupted:
                print("LLM session interrupted.")
            print(format_state(executor.get_state()))
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except EOFError:
        print("\nInput closed.")
    except Exception as exc:
        print(f"Error: {exc}")
        return 1
    finally:
        if llm_event_loop is not None:
            try:
                llm_event_loop.close()
            except Exception:
                pass
            asyncio.set_event_loop(None)
        try:
            executor.submit_action(Action(type=ActionType.STOP))
        except Exception:
            pass
        if hasattr(executor, "stop") and callable(getattr(executor, "stop", None)):
            executor.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())