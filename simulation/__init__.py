from typing import List
import json
from dataclasses import dataclass, field
import dataclasses
import os
import re
from datetime import datetime

@dataclass
class RobotPose:
    position: List[float]  # [x, y, z]
    rotation: List[float]  # [x, y, z, w] quaternion

@dataclass
class RobotPosition:
    x: float
    y: float
    heading: float

@dataclass    
class RobotTarget:
    name: str
    x: float
    y: float

@dataclass
class RobotStatus:
    position: RobotPosition
    pose: RobotPose    

@dataclass
class TargetScoringData:
    target: RobotTarget
    distance: float
    angle: float
    
@dataclass
class RobotAction:
    name: str
    parameter: float

@dataclass
class IterationData:
    message: str
    img: str  # base64 encoded image
    response: str
    action: RobotAction 
    scoringData: List[TargetScoringData]
    endRobotStatus: RobotStatus
    latency: float = None
    actionSuccess: bool = False

class LLMSession:
    def __init__(self):
        self.prompt = ""
        self.model = ""
        self.id = ""
        self.initialRobotStatus: RobotStatus = None
        self.systemPrompt: str = ""
        self.targets: List[RobotTarget] = []
        self.iterations: List[IterationData] = []

        self.jsonErrors = 0
        self.safetyTriggers = 0

        self.goalCompleted = False
        self.simulationAborted = False
        self.abortionReason = None
        self.sessionEnded = False

    def setPrompt(self, prompt: str):
        self.prompt = prompt

    def setModel(self, model: str):
        self.model = model

    def setId(self, id: str):
        self.id = id

    def setInitialRobotStatus(self, status: RobotStatus):
        self.initialRobotStatus = status

    def setSystemPrompt(self, systemPrompt: str):
        self.systemPrompt = systemPrompt

    def setTargets(self, targets: List[RobotTarget]):
        self.targets = targets    

    def addIteration(self, iteration: IterationData):
        self.iterations.append(iteration)

    def incrementJsonErrors(self):
        self.jsonErrors += 1

    def incrementSafetyTriggers(self):
        self.safetyTriggers += 1                
    
    def asObject(self):
        return {
            "id": self.id,
            "model": self.model,
            "prompt": self.prompt,
            "systemPrompt": self.systemPrompt,
            "initialRobotPose": self.initialRobotStatus,
            "targets": self.targets,
            "iterations": self.iterations,
            "jsonErrors": self.jsonErrors,
            "safetyTriggers": self.safetyTriggers,
            "goalCompleted": self.goalCompleted,
            "simulationAborted": self.simulationAborted,
            "abortionReason": self.abortionReason,
        }
    
    def toJSON(self):
        return json.dumps(self.asObject(), indent=4)
    
    def completeGoal(self):
        self.goalCompleted = True
        self.sessionEnded = True

    def abortSession(self, reason: str):
        self.simulationAborted = True
        self.abortionReason = reason
        self.sessionEnded = True

    def isGoalCompleted(self):
        return self.sessionEnded and self.goalCompleted and not self.simulationAborted
    
    def hasSessionEnded(self):
        return self.sessionEnded
    
    def isAborted(self):
        return self.sessionEnded and self.simulationAborted
    
    def getAbortionReason(self):
        return self.abortionReason if self.simulationAborted else None
    
    def __repr__(self):
        return f"LLMSession(model={self.model}, prompt={self.prompt}, id={self.id}, iterations={len(self.iterations)}, goalCompleted={self.goalCompleted}, simulationAborted={self.simulationAborted})"

    def _serialize(self, obj):
        """Recursively convert dataclasses and common containers to JSON-serializable structures."""
        # dataclass instance
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        # basic types
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        # list/tuple
        if isinstance(obj, (list, tuple)):
            return [self._serialize(v) for v in obj]
        # dict
        if isinstance(obj, dict):
            return {k: self._serialize(v) for k, v in obj.items()}
        # fallback: try to convert to str
        try:
            return str(obj)
        except Exception:
            return None

    def save(self, out_dir: str = "experiments", final: bool = False, name: str = None):
        """Save session to a JSON file.

        Behavior:
        - In partial mode (final=False) overwrite a stable partial filename: experiment_<model>_partial.json
        - When final=True, write a timestamped file: experiment_<model>_<YYYYmmdd-HHMMSS>.json
        - Writes atomically by writing to a temporary file then replacing the target file.
        - Best-effort: exceptions are caught and logged to stderr but not raised.
        """
        try:
            os.makedirs(out_dir, exist_ok=True)

            safe_model = re.sub(r"[^0-9A-Za-z_-]", "_", (self.model or "unknown_model"))
            if name:
                base = name
            else:
                if final:
                    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
                    base = f"experiment_{safe_model}_{ts}"
                else:
                    base = f"experiment_{safe_model}_partial"

            path = os.path.join(out_dir, f"{base}.json")
            tmp_path = path + ".tmp"

            # build serializable object
            obj = self.asObject()
            serializable = self._serialize(obj)

            # write atomically
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(serializable, f, indent=4, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())

            os.replace(tmp_path, path)

            # If final, also update a copy named 'latest' for quick access
            if final:
                latest_path = os.path.join(out_dir, f"experiment_{safe_model}_latest.json")
                try:
                    with open(latest_path + ".tmp", "w", encoding="utf-8") as f:
                        json.dump(serializable, f, indent=4, ensure_ascii=False)
                        f.flush()
                        os.fsync(f.fileno())
                    os.replace(latest_path + ".tmp", latest_path)
                except Exception:
                    # non-fatal
                    pass

            return path
        except Exception as e:
            # best-effort: don't raise, just return None
            try:
                import sys
                print(f"Failed to save LLMSession: {e}", file=sys.stderr)
            except Exception:
                pass
            return None
