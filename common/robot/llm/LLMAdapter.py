from common.llm.chats import LLMChat
from common.utils.llm import create_message
from common.utils.misc import extractJSON
import logging
import json
import os
from jsonschema import validate
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Optional, TypeVar
from server.core.motion_schema import Action, ActionType

T = TypeVar("T")

logger = logging.getLogger(__name__)


@dataclass
class Result(Generic[T]):
    ok: bool
    value: Optional[T] = None
    error: Optional[Exception] = None

    @classmethod
    def success(cls, value: T) -> "Result[T]":
        return cls(ok=True, value=value)

    @classmethod
    def failure(cls, error: Exception) -> "Result[T]":
        return cls(ok=False, error=error)


class InvalidJSON(Exception):
    """Raised when the LLM response doesn't contain valid JSON."""
    def __init__(self, raw: str):
        super().__init__("Invalid JSON from model")
        self.raw = raw


class SchemaValidationError(Exception):
    """Raised when the parsed JSON doesn't match the expected schema."""
    def __init__(self, original: Exception):
        super().__init__("Response did not match schema")
        self.original = original


class LLMAdapter:
    def __init__(self, chat: LLMChat, system_prompt: str | None = None, system_prompt_file: str | None = None):
        self.chat = chat
        self.responseSchema = {
            "type": "object",
            "properties": {
                "goal": { "type": "string" },
                "scene_description": { "type": "string" },
                "reasoning": { "type": "string" },
                "action": {
                    "type": "object",
                    "properties": {
                        "command": { "type": "string" },
                        "params": {
                            "oneOf": [
                                {"type": "number"},
                                {
                                    "type": "object",
                                    "properties": {
                                        "value": {"type": "number"},
                                        "forward": {"type": "number"},
                                        "turn": {"type": "number"},
                                        "angle": {"type": "number"},
                                        "distance": {"type": "number"},
                                    },
                                    "additionalProperties": True,
                                }
                            ]
                        },
                        "parameters": {
                            "oneOf": [
                                {"type": "number"},
                                {
                                    "type": "object",
                                    "properties": {
                                        "value": {"type": "number"},
                                        "forward": {"type": "number"},
                                        "turn": {"type": "number"},
                                        "angle": {"type": "number"},
                                        "distance": {"type": "number"},
                                    },
                                    "additionalProperties": True,
                                }
                            ]
                        },
                    },
                    "required": ["command"],
                }
            },
            "required": ["goal", "scene_description", "reasoning", "action"],
        }
        self._install_system_instruction(system_prompt=system_prompt, system_prompt_file=system_prompt_file)

    def _default_system_instruction(self) -> str:
        return (
            "You are a motion planner for a hexapod robot. "
            "Always respond with exactly one valid JSON object and no extra text. "
            "Allowed action.command values: FRONT, BACK, ROTATE_LEFT, ROTATE_RIGHT, STOP, RELAX, BALANCE, COMPLETE. "
            "Use this JSON shape: "
            '{"goal":"...","scene_description":"...","reasoning":"...",'
            '"action":{"command":"FRONT","params":{"value":10}}}. '
            "For movement, params.value must be numeric. "
            "When goal is complete, use command COMPLETE."
        )

    def _try_read_prompt_file(self, file_path: str | None) -> str | None:
        if not file_path:
            return None

        candidate = Path(file_path).expanduser()
        roots = [Path.cwd(), Path(__file__).resolve().parents[3]]
        resolved_paths = [candidate] if candidate.is_absolute() else [root / candidate for root in roots]

        for path in resolved_paths:
            try:
                if path.exists() and path.is_file():
                    content = path.read_text(encoding="utf-8").strip()
                    if content:
                        logger.info("Loaded LLM system prompt from file: %s", path)
                        return content
            except Exception:
                logger.warning("Failed reading system prompt file: %s", path, exc_info=True)
        return None

    def _resolve_system_instruction(self, system_prompt: str | None, system_prompt_file: str | None) -> str:
        if system_prompt is not None and system_prompt.strip():
            return system_prompt.strip()

        env_inline = os.environ.get("HEXAPOD_SYSTEM_PROMPT")
        if env_inline and env_inline.strip():
            return env_inline.strip()

        prompt_text = self._try_read_prompt_file(system_prompt_file)
        if prompt_text:
            return prompt_text

        env_file = os.environ.get("HEXAPOD_SYSTEM_PROMPT_FILE")
        prompt_text = self._try_read_prompt_file(env_file)
        if prompt_text:
            return prompt_text

        return self._default_system_instruction()

    def _install_system_instruction(self, system_prompt: str | None = None, system_prompt_file: str | None = None) -> None:
        if not hasattr(self.chat, "set_system_instruction"):
            return
        if not callable(getattr(self.chat, "set_system_instruction", None)):
            return

        self.chat.set_system_instruction(
            self._resolve_system_instruction(system_prompt=system_prompt, system_prompt_file=system_prompt_file)
        )

    def _to_float_or_none(self, value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _canonical_command(self, command: str, raw_params) -> str:
        token = str(command or "").strip().upper().replace("-", "_").replace(" ", "_")
        aliases = {
            "FORWARD": "FRONT",
            "MOVE_FORWARD": "FRONT",
            "GO_FORWARD": "FRONT",
            "BACKWARD": "BACK",
            "MOVE_BACK": "BACK",
            "MOVE_BACKWARD": "BACK",
            "GO_BACK": "BACK",
            "TURN_LEFT": "ROTATE_LEFT",
            "LEFT": "ROTATE_LEFT",
            "TURN_RIGHT": "ROTATE_RIGHT",
            "RIGHT": "ROTATE_RIGHT",
            "HALT": "STOP",
            "DONE": "COMPLETE",
            "FINISH": "COMPLETE",
            "SUCCESS": "COMPLETE",
        }
        if token in aliases:
            return aliases[token]

        if token in {"MOVE", "WALK", "TURN", "ROTATE"} and isinstance(raw_params, dict):
            turn = self._to_float_or_none(raw_params.get("turn"))
            if turn is None:
                turn = self._to_float_or_none(raw_params.get("angle"))
            forward = self._to_float_or_none(raw_params.get("forward"))
            if forward is None:
                forward = self._to_float_or_none(raw_params.get("distance"))

            if turn is not None and abs(turn) > 1e-6:
                return "ROTATE_RIGHT" if turn > 0 else "ROTATE_LEFT"
            if forward is not None and abs(forward) > 1e-6:
                return "FRONT" if forward > 0 else "BACK"

        return token

    def _extract_value(self, command: str, raw_params) -> float:
        numeric = self._to_float_or_none(raw_params)
        if numeric is not None:
            return numeric

        if not isinstance(raw_params, dict):
            return 0.0

        command_keys = {
            "FRONT": ["value", "forward", "distance", "amount", "cm", "meters", "m"],
            "BACK": ["value", "forward", "distance", "amount", "cm", "meters", "m"],
            "ROTATE_LEFT": ["value", "turn", "angle", "degrees", "deg"],
            "ROTATE_RIGHT": ["value", "turn", "angle", "degrees", "deg"],
        }

        for key in command_keys.get(command, ["value", "forward", "turn", "angle", "distance"]):
            if key in raw_params:
                candidate = self._to_float_or_none(raw_params.get(key))
                if candidate is not None:
                    return candidate

        return 0.0

    def _canonicalize_response(self, action_obj: dict) -> dict:
        if not isinstance(action_obj, dict):
            raise ValueError("LLM response must be a JSON object")

        action_raw = action_obj.get("action")
        if not isinstance(action_raw, dict):
            raise ValueError("LLM response action field must be an object")

        raw_params = action_raw.get("params")
        if raw_params is None:
            raw_params = action_raw.get("parameters")
        if raw_params is None and "value" in action_raw:
            raw_params = action_raw.get("value")

        command = self._canonical_command(action_raw.get("command", ""), raw_params)
        value = self._extract_value(command, raw_params)
        if command in {"BACK", "ROTATE_LEFT"}:
            value = abs(value)

        canonical = dict(action_obj)
        canonical.setdefault("goal", "")
        canonical.setdefault("scene_description", "")
        canonical.setdefault("reasoning", "")
        canonical["action"] = {
            **action_raw,
            "command": command,
            "params": {"value": value},
        }
        return canonical

    def clear(self):
        self.chat.clear_chat()

    def _normalize_action(self, command: str, raw_params, metadata: dict) -> Action:
        params = raw_params if isinstance(raw_params, dict) else {"value": float(raw_params)}
        value = float(params.get("value", 0.0))
        ttl = max(0.25, min(4.0, abs(value) / 15.0))
        upper_command = command.upper()

        if upper_command == "FRONT":
            return Action(
                type=ActionType.WALK,
                params={"x": 0, "y": abs(value), "angle": 0, "gait_type": "1", "speed": 5},
                ttl=ttl,
                metadata={**metadata, "command": upper_command},
            )
        if upper_command == "BACK":
            return Action(
                type=ActionType.WALK,
                params={"x": 0, "y": -abs(value), "angle": 0, "gait_type": "1", "speed": 5},
                ttl=ttl,
                metadata={**metadata, "command": upper_command},
            )
        if upper_command == "ROTATE_LEFT":
            return Action(
                type=ActionType.WALK,
                params={"x": 0, "y": 0, "angle": -abs(value), "gait_type": "1", "speed": 4},
                ttl=ttl,
                metadata={**metadata, "command": upper_command},
            )
        if upper_command == "ROTATE_RIGHT":
            return Action(
                type=ActionType.WALK,
                params={"x": 0, "y": 0, "angle": abs(value), "gait_type": "1", "speed": 4},
                ttl=ttl,
                metadata={**metadata, "command": upper_command},
            )
        if upper_command == "STOP":
            return Action(type=ActionType.STOP, metadata={**metadata, "command": upper_command})
        if upper_command == "RELAX":
            return Action(type=ActionType.RELAX, metadata={**metadata, "command": upper_command})
        if upper_command == "BALANCE":
            return Action(type=ActionType.BALANCE, metadata={**metadata, "command": upper_command})
        if upper_command == "COMPLETE":
            return Action(type=ActionType.COMPLETE, metadata={**metadata, "command": upper_command})

        raise ValueError(f"Unknown command: {command}")

    async def iterate(self, prompt, image) -> Result[Action]:
        """Send prompt+image to the chat, extract and validate JSON, and return a Result.

        Success: Result.success(Action)
        Failure: Result.failure(Exception) with one of: InvalidJSON, SchemaValidationError, or other runtime error
        """
        response = await self.chat.send_message(create_message(prompt, image))

        # Attempt to extract JSON string from the model response
        try:
            extracted = extractJSON(response)
            action_obj = json.loads(extracted)
            action_obj = self._canonicalize_response(action_obj)
        except Exception as e:
            logger.warning("Failed to extract/parse JSON from LLM response", exc_info=True)
            return Result.failure(InvalidJSON(response))

        # Validate against schema
        try:
            validate(action_obj, self.responseSchema)
        except Exception as e:
            logger.warning("Schema validation failed for LLM response", exc_info=True)
            return Result.failure(SchemaValidationError(e))

        # Build canonical Action and return
        try:
            cmd = action_obj["action"]["command"]
            raw_parameters = action_obj["action"].get("params")
            if raw_parameters is None:
                raw_parameters = action_obj["action"]["parameters"]
            reasoning = action_obj["reasoning"]
            subgoal = action_obj["goal"]
            scene_description = action_obj["scene_description"]
            metadata = {
                "goal": subgoal,
                "reasoning": reasoning,
                "scene_summary": scene_description,
                "raw_parameters": raw_parameters,
            }
            return Result.success(
                self._normalize_action(
                    command=cmd,
                    raw_params=raw_parameters,
                    metadata=metadata,
                )
            )
        except Exception as e:
            logger.debug("Failed to construct Action", exc_info=True)
            return Result.failure(e)
