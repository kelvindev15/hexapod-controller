from common.llm.chats import LLMChat
from common.utils.llm import create_message
from common.utils.misc import extractJSON
import logging
import json
from jsonschema import validate
from dataclasses import dataclass
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
    def __init__(self, chat: LLMChat):
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
                                        "value": {"type": "number"}
                                    },
                                    "required": ["value"],
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
                                        "value": {"type": "number"}
                                    },
                                    "required": ["value"],
                                    "additionalProperties": True,
                                }
                            ]
                        },
                    },
                    "required": ["command"],
                    "anyOf": [
                        {"required": ["params"]},
                        {"required": ["parameters"]}
                    ],
                }
            },
            "required": ["goal", "scene_description", "reasoning", "action"],
        }

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
