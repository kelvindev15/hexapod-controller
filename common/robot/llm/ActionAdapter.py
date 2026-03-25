import asyncio
import logging
from common.robot.llm.RobotAction import RobotAction
import numpy as np
from dataclasses import dataclass
from enum import Enum
from server.core.motion_executor import MotionExecutor
from server.core.motion_schema import Action, ActionType, ActionSafetyPolicy, SafetyContext, SafetyDecision

logger = logging.getLogger(__name__)


class ActionStatus(Enum):
    SUCCESS = 1
    FAILURE = 2
    OBSTACLE_DETECTED = 3
    
@dataclass
class ActionResult:
    status: ActionStatus
    message: str = ""

class ActionAdapter:

    def __init__(self, motion_executor: MotionExecutor):
        self.motion_executor = motion_executor
        self.safety_policy = ActionSafetyPolicy()

    def to_motion_action(self, action: RobotAction) -> Action:
        command = action.command.upper()
        value = float(action.params.get("value", 0.0))
        ttl = max(0.25, min(4.0, abs(value) / 15.0))

        if command == "FRONT":
            return Action(
                type=ActionType.WALK,
                params={"x": 0, "y": abs(value), "angle": 0, "gait_type": "1", "speed": 5},
                ttl=ttl,
                metadata=action.to_log_context(),
            )
        if command == "BACK":
            return Action(
                type=ActionType.WALK,
                params={"x": 0, "y": -abs(value), "angle": 0, "gait_type": "1", "speed": 5},
                ttl=ttl,
                metadata=action.to_log_context(),
            )
        if command == "ROTATE_LEFT":
            return Action(
                type=ActionType.WALK,
                params={"x": 0, "y": 0, "angle": -abs(value), "gait_type": "1", "speed": 4},
                ttl=ttl,
                metadata=action.to_log_context(),
            )
        if command == "ROTATE_RIGHT":
            return Action(
                type=ActionType.WALK,
                params={"x": 0, "y": 0, "angle": abs(value), "gait_type": "1", "speed": 4},
                ttl=ttl,
                metadata=action.to_log_context(),
            )
        if command == "STOP":
            return Action(type=ActionType.STOP, metadata=action.to_log_context())
        if command == "COMPLETE":
            return Action(type=ActionType.COMPLETE, metadata=action.to_log_context())

        raise ValueError(f"Unknown command: {command}")

    def normalize_action(self, action: Action | RobotAction) -> Action:
        if isinstance(action, Action):
            return action
        return self.to_motion_action(action)

    async def execute(self, action: Action | RobotAction) -> ActionResult:
        try:
            motion_action = self.normalize_action(action)
            if motion_action.type == ActionType.COMPLETE:
                return ActionResult(status=ActionStatus.SUCCESS, message="Goal marked complete")

            self.motion_executor.submit_action(motion_action)
            await asyncio.sleep(0)
            logger.info("Submitted action to MotionExecutor", extra={"action": motion_action.metadata})
            return ActionResult(
                status=ActionStatus.SUCCESS,
                message=f"Submitted {motion_action.type.value} with params {motion_action.params}",
            )
        except ValueError as exc:
            logger.warning("Rejected unknown action command")
            return ActionResult(status=ActionStatus.FAILURE, message=str(exc))
        except Exception as exc:
            logger.exception("Failed to submit action")
            return ActionResult(status=ActionStatus.FAILURE, message=str(exc))
        
    def assess_safety(self, action: Action | RobotAction, lidar: list[float] | None = None) -> SafetyDecision:
        try:
            motion_action = self.normalize_action(action)
        except ValueError as exc:
            return SafetyDecision(False, str(exc))

        if motion_action.type == ActionType.COMPLETE:
            return SafetyDecision(True, "")

        has_range_data = lidar is not None and len(lidar) > 0
        min_distance = float(np.min(lidar)) if has_range_data else None

        decision = self.safety_policy.evaluate_pre_execution(
            motion_action,
            SafetyContext(distance=min_distance, roll=0.0, pitch=0.0, has_range_data=has_range_data),
        )
        if not decision.is_safe:
            logger.warning("Unsafe action rejected", extra={"action": motion_action.metadata, "reason": decision.reason})
        return decision

    def checkSafety(self, action: Action | RobotAction, lidar: list[float] | None = None) -> bool:
        return self.assess_safety(action, lidar).is_safe