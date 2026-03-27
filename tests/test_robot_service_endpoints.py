import json
import threading
import unittest
from urllib import request

from server.core.motion_schema import Action, ActionType, WorldState
from server.robot_service import RobotHTTPServer, create_handler


class FakeExecutor:
    def __init__(self):
        self.submitted = []
        self.running = True
        self.state = WorldState(distance=88.0, current_action=ActionType.STOP)

    def submit_action(self, action: Action):
        self.submitted.append(action)
        self.state.current_action = action.type

    def get_state(self):
        return self.state


class RobotServiceEndpointTests(unittest.TestCase):
    def setUp(self):
        self.executor = FakeExecutor()
        self.server = RobotHTTPServer(("127.0.0.1", 0), create_handler(), self.executor)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.server.server_address
        self.base_url = f"http://{host}:{port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _request(self, method: str, path: str, payload=None):
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = request.Request(f"{self.base_url}{path}", method=method, data=data, headers=headers)
        with request.urlopen(req, timeout=2.0) as response:
            return response.status, json.loads(response.read().decode("utf-8"))

    def test_health_and_state(self):
        status, health = self._request("GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(health["status"], "ok")

        status, state = self._request("GET", "/state")
        self.assertEqual(status, 200)
        self.assertEqual(state["distance"], 88.0)
        self.assertEqual(state["current_action"], "stop")

    def test_submit_action_accepts_valid_payload(self):
        status, payload = self._request(
            "POST",
            "/actions",
            {
                "type": "walk",
                "params": {"x": 0, "y": 10, "angle": 0, "gait_type": "1", "speed": 5},
                "ttl": 0.75,
                "metadata": {"source": "test"},
            },
        )

        self.assertEqual(status, 202)
        self.assertTrue(payload["accepted"])
        self.assertEqual(len(self.executor.submitted), 1)
        self.assertEqual(self.executor.submitted[0].type, ActionType.WALK)

    def test_submit_action_rejects_invalid_payload(self):
        req = request.Request(
            f"{self.base_url}/actions",
            method="POST",
            data=json.dumps({"type": "bad"}).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )

        with self.assertRaises(Exception) as ctx:
            request.urlopen(req, timeout=2.0)

        err_body = ctx.exception.read().decode("utf-8")
        parsed = json.loads(err_body)
        self.assertIn("Unsupported action type", parsed["error"])

    def test_stop_endpoint_submits_stop(self):
        status, payload = self._request("POST", "/actions/stop", {})
        self.assertEqual(status, 202)
        self.assertTrue(payload["accepted"])
        self.assertEqual(self.executor.submitted[-1].type, ActionType.STOP)


if __name__ == "__main__":
    unittest.main()