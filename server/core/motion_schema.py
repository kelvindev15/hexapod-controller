from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any

class ActionType(Enum):
    RELAX = "relax"
    STOP = "stop"
    WALK = "walk"
    POSITION = "position"
    ATTITUDE = "attitude"
    BALANCE = "balance"
    COMPLETE = "complete"

@dataclass
class Action:
    type: ActionType
    params: Dict[str, Any] = field(default_factory=dict)
    ttl: float = 0.0  # Time to live in seconds, 0 for infinite
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def parameters(self) -> Dict[str, Any]:
        return self.params

    @parameters.setter
    def parameters(self, value: Dict[str, Any]) -> None:
        self.params = value

    @property
    def parameter(self) -> float:
        value = self.params.get("value", 0.0)
        try:
            return float(value)
        except Exception:
            return 0.0


@dataclass(frozen=True)
class SafetyDecision:
    is_safe: bool
    reason: str = ""


@dataclass
class SafetyContext:
    distance: Optional[float] = None
    roll: float = 0.0
    pitch: float = 0.0
    has_range_data: bool = True


class ActionSafetyPolicy:
    TILT_THRESHOLD = 30.0
    STOP_DISTANCE = 15.0

    def evaluate_pre_execution(self, action: Action, context: SafetyContext) -> SafetyDecision:
        return self._evaluate(action, context)

    def evaluate_runtime(self, action: Action, context: SafetyContext) -> SafetyDecision:
        return self._evaluate(action, context)

    def evaluate(self, action: Action, context: SafetyContext) -> SafetyDecision:
        return self._evaluate(action, context)

    def _evaluate(self, action: Action, context: SafetyContext) -> SafetyDecision:
        if abs(context.roll) > self.TILT_THRESHOLD or abs(context.pitch) > self.TILT_THRESHOLD:
            return SafetyDecision(False, "Unsafe posture: tilt exceeds threshold")

        if action.type == ActionType.WALK:
            y_speed = float(action.params.get("y", 0.0))
            if y_speed > 0:
                if not context.has_range_data or context.distance is None:
                    return SafetyDecision(False, "Missing range data for forward motion safety check")
                if context.distance < self.STOP_DISTANCE:
                    return SafetyDecision(
                        False,
                        f"Obstacle too close for forward motion (distance={context.distance:.2f}cm, required>={self.STOP_DISTANCE:.2f}cm)",
                    )

        return SafetyDecision(True, "")

@dataclass
class WorldState:
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    distance: float = 999.0
    is_balancing: bool = False
    current_action: Optional[ActionType] = None
    timestamp: float = 0.0
    is_safe: bool = True
    safety_reason: str = ""
