from dataclasses import dataclass, field
from typing import Any, Dict

@dataclass
class Motivation:
    subgoal: str = ""
    reasoning: str = ""
    scene_description: str = ""

@dataclass(init=False)
class RobotAction:
    command: str
    params: Dict[str, Any] = field(default_factory=dict)
    motivation: Motivation = field(default_factory=Motivation)
    goal: str = ""
    reasoning: str = ""
    scene_summary: str = ""

    def __init__(
        self,
        command: str,
        params: Dict[str, Any] | None = None,
        parameters: Dict[str, Any] | None = None,
        motivation: Motivation | None = None,
        goal: str = "",
        reasoning: str = "",
        scene_summary: str = "",
    ):
        self.command = command
        self.params = params if params is not None else (parameters or {})
        self.motivation = motivation if motivation is not None else Motivation()
        self.goal = goal
        self.reasoning = reasoning
        self.scene_summary = scene_summary

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

    def to_log_context(self) -> Dict[str, Any]:
        return {
            "command": self.command,
            "params": self.params,
            "goal": self.goal or self.motivation.subgoal,
            "reasoning": self.reasoning or self.motivation.reasoning,
            "scene_summary": self.scene_summary or self.motivation.scene_description,
        }
    
    def __repr__(self):
        return (
            "RobotAction("
            f"command={self.command}, "
            f"params={self.params}, "
            f"motivation={self.motivation}, "
            f"goal={self.goal}, "
            f"reasoning={self.reasoning}, "
            f"scene_summary={self.scene_summary}"
            ")"
        )
