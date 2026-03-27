from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict

from .motion_schema import Action, ActionType, WorldState


class ContractValidationError(ValueError):
    pass


def action_to_dict(action: Action) -> Dict[str, Any]:
    return {
        "type": action.type.value,
        "params": dict(action.params or {}),
        "ttl": float(action.ttl),
        "metadata": dict(action.metadata or {}),
    }


def action_from_dict(payload: Dict[str, Any]) -> Action:
    if not isinstance(payload, dict):
        raise ContractValidationError("Action payload must be a JSON object")

    raw_type = payload.get("type")
    if not isinstance(raw_type, str):
        raise ContractValidationError("Action field 'type' must be a string")

    try:
        action_type = ActionType(raw_type)
    except Exception as exc:
        raise ContractValidationError(f"Unsupported action type: {raw_type}") from exc

    params = payload.get("params", {})
    if not isinstance(params, dict):
        raise ContractValidationError("Action field 'params' must be an object")

    raw_ttl = payload.get("ttl", 0.0)
    try:
        ttl = float(raw_ttl)
    except Exception as exc:
        raise ContractValidationError("Action field 'ttl' must be numeric") from exc

    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ContractValidationError("Action field 'metadata' must be an object")

    return Action(type=action_type, params=params, ttl=ttl, metadata=metadata)


def world_state_to_dict(state: WorldState) -> Dict[str, Any]:
    payload = asdict(state)
    if state.current_action is not None:
        payload["current_action"] = state.current_action.value
    return payload


def world_state_from_dict(payload: Dict[str, Any]) -> WorldState:
    if not isinstance(payload, dict):
        raise ContractValidationError("WorldState payload must be a JSON object")

    action_token = payload.get("current_action")
    current_action = None
    if action_token is not None:
        if not isinstance(action_token, str):
            raise ContractValidationError("WorldState field 'current_action' must be a string or null")
        try:
            current_action = ActionType(action_token)
        except Exception as exc:
            raise ContractValidationError(f"Unsupported current_action: {action_token}") from exc

    return WorldState(
        roll=float(payload.get("roll", 0.0)),
        pitch=float(payload.get("pitch", 0.0)),
        yaw=float(payload.get("yaw", 0.0)),
        distance=float(payload.get("distance", 999.0)),
        is_balancing=bool(payload.get("is_balancing", False)),
        current_action=current_action,
        timestamp=float(payload.get("timestamp", 0.0)),
        is_safe=bool(payload.get("is_safe", True)),
        safety_reason=str(payload.get("safety_reason", "")),
    )