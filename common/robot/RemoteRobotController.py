from __future__ import annotations

import json
from urllib import error, request

import numpy as np

from common.robot.RobotController import RobotController
from server.core.api_contract import action_to_dict, world_state_from_dict
from server.core.motion_schema import Action, ActionType, WorldState


class RemoteRobotController(RobotController):
    def __init__(self, base_url: str, timeout_seconds: float = 5.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _request(self, method: str, path: str, payload=None):
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = request.Request(self._url(path), method=method, data=data, headers=headers)
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
                if not raw:
                    return {}
                return json.loads(raw)
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8") if exc.fp else ""
            raise RuntimeError(f"Robot service HTTP {exc.code}: {body or exc.reason}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"Robot service unreachable: {exc.reason}") from exc

    def start(self):
        health = self._request("GET", "/health")
        if health.get("status") != "ok":
            raise RuntimeError(f"Robot service unhealthy: {health}")

    def submit_action(self, action: Action):
        self._request("POST", "/actions", action_to_dict(action))

    def get_state(self) -> WorldState:
        payload = self._request("GET", "/state")
        return world_state_from_dict(payload)

    def goFront(self, distance: float = 1.0) -> None:
        self.submit_action(
            Action(
                type=ActionType.WALK,
                params={"x": 0, "y": abs(distance), "angle": 0, "gait_type": "1", "speed": 5},
                ttl=max(0.25, min(4.0, abs(distance) / 15.0)),
            )
        )

    def goBack(self, distance: float = 1.0) -> None:
        self.submit_action(
            Action(
                type=ActionType.WALK,
                params={"x": 0, "y": -abs(distance), "angle": 0, "gait_type": "1", "speed": 5},
                ttl=max(0.25, min(4.0, abs(distance) / 15.0)),
            )
        )

    def rotateRight(self, angle: float = 45.0) -> None:
        self.submit_action(
            Action(
                type=ActionType.WALK,
                params={"x": 0, "y": 0, "angle": abs(angle), "gait_type": "1", "speed": 4},
                ttl=max(0.25, min(4.0, abs(angle) / 15.0)),
            )
        )

    def rotateLeft(self, angle: float = 45.0) -> None:
        self.submit_action(
            Action(
                type=ActionType.WALK,
                params={"x": 0, "y": 0, "angle": -abs(angle), "gait_type": "1", "speed": 4},
                ttl=max(0.25, min(4.0, abs(angle) / 15.0)),
            )
        )

    def stop(self) -> None:
        self._request("POST", "/actions/stop", {})

    def getCameraImage(self):
        return np.zeros((64, 64, 3), dtype=np.uint8)

    def getLidarImage(self, fov_degrees: int, offset_degrees: int = 0):
        _ = offset_degrees
        state = self.get_state()
        distance = float(state.distance if state.distance is not None else 999.0)
        count = max(1, int(fov_degrees))
        return [distance] * count

    def getFrontLidarImage(self):
        return self.getLidarImage(90, 0)