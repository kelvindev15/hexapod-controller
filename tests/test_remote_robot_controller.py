import json
import unittest
from unittest.mock import patch

from common.robot.RemoteRobotController import RemoteRobotController
from server.core.motion_schema import Action, ActionType


class FakeHTTPResponse:
    def __init__(self, status: int, payload):
        self.status = status
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class RemoteRobotControllerTests(unittest.TestCase):
    def test_submit_and_read_state_over_http(self):
        captured = []

        def fake_urlopen(req, timeout=5.0):
            method = req.get_method()
            url = req.full_url
            body = req.data.decode("utf-8") if req.data else ""
            parsed = json.loads(body) if body else None
            captured.append((method, url, parsed, timeout))

            if url.endswith("/health"):
                return FakeHTTPResponse(200, {"status": "ok", "running": True})
            if url.endswith("/actions"):
                return FakeHTTPResponse(202, {"accepted": True})
            if url.endswith("/state"):
                return FakeHTTPResponse(
                    200,
                    {
                        "roll": 0.0,
                        "pitch": 0.0,
                        "yaw": 0.0,
                        "distance": 65.0,
                        "is_balancing": False,
                        "current_action": "walk",
                        "timestamp": 1.0,
                        "is_safe": True,
                        "safety_reason": "",
                    },
                )
            if url.endswith("/actions/stop"):
                return FakeHTTPResponse(202, {"accepted": True})
            return FakeHTTPResponse(404, {"error": "Not found"})

        with patch("common.robot.RemoteRobotController.request.urlopen", side_effect=fake_urlopen):
            client = RemoteRobotController("http://robot.local:8080", timeout_seconds=3.0)
            client.start()
            client.submit_action(Action(type=ActionType.WALK, params={"x": 0, "y": 8, "angle": 0, "gait_type": "1", "speed": 5}, ttl=1.0))
            state = client.get_state()
            client.stop()

        self.assertEqual(state.distance, 65.0)
        self.assertEqual(state.current_action, ActionType.WALK)
        self.assertEqual(captured[0][0], "GET")
        self.assertTrue(captured[1][1].endswith("/actions"))
        self.assertEqual(captured[1][2]["type"], "walk")
        self.assertEqual(captured[-1][1].endswith("/actions/stop"), True)


if __name__ == "__main__":
    unittest.main()