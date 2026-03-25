import sys
import types
import unittest

fake_numpy = types.ModuleType("numpy")
fake_numpy.min = min
sys.modules.setdefault("numpy", fake_numpy)

# Avoid importing heavy vision model during tests.
fake_images = types.ModuleType("common.utils.images")
fake_images.toBase64Image = lambda image: "stub-image"
sys.modules.setdefault("common.utils.images", fake_images)

fake_robot_utils = types.ModuleType("common.utils.robot")
fake_robot_utils.getDistancesFromLidar = lambda _readings, _fov, _sections: {"front_distance": 100.0, "sections": []}
fake_robot_utils.getDistanceDescription = lambda _distances: "Lidar distances: clear"
sys.modules.setdefault("common.utils.robot", fake_robot_utils)

fake_chats = types.ModuleType("common.llm.chats")


class _StubLLMChat:
    pass


fake_chats.LLMChat = _StubLLMChat
sys.modules.setdefault("common.llm.chats", fake_chats)

fake_llm_utils = types.ModuleType("common.utils.llm")
fake_llm_utils.create_message = lambda text, image=None: {"text": text, "image": image}
sys.modules.setdefault("common.utils.llm", fake_llm_utils)

fake_jsonschema = types.ModuleType("jsonschema")
fake_jsonschema.validate = lambda _obj, _schema: None
sys.modules.setdefault("jsonschema", fake_jsonschema)

fake_motion_executor = types.ModuleType("server.core.motion_executor")


class _StubMotionExecutor:
    pass


fake_motion_executor.MotionExecutor = _StubMotionExecutor
sys.modules.setdefault("server.core.motion_executor", fake_motion_executor)

from common.robot.llm.LLMAdapter import LLMAdapter, InvalidJSON
from common.robot.llm.RobotAction import RobotAction
from common.robot.llm.ActionAdapter import ActionAdapter, ActionStatus
from common.robot.LLMRobotController import LLMRobotController
from server.core.motion_schema import ActionType


class FakeChat:
    def __init__(self, responses):
        self.responses = list(responses)
        self.chat_id = None

    def clear_chat(self):
        return None

    def set_chat_id(self, chat_id: str):
        self.chat_id = chat_id

    async def send_message(self, _message):
        if not self.responses:
            return "{}"
        return self.responses.pop(0)


class FakeRobot:
    def goFront(self, distance=1.0):
        return None

    def goBack(self, distance=1.0):
        return None

    def rotateRight(self, angle=45.0):
        return None

    def rotateLeft(self, angle=45.0):
        return None

    def stop(self):
        return None

    def getCameraImage(self):
        return [[0]]

    def getLidarImage(self, fov_degrees: int, offset_degrees: int = 0):
        return [100.0] * 30

    def getFrontLidarImage(self):
        return [100.0] * 30


class FakeMotionExecutor:
    def __init__(self):
        self.started = False
        self.submitted = []

    def start(self):
        self.started = True

    def submit_action(self, action):
        self.submitted.append(action)


class FailingMotionExecutor(FakeMotionExecutor):
    def submit_action(self, action):
        raise RuntimeError("submit failed")


class LLMExecutionFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_action_normalization_from_numeric_parameters(self):
        chat = FakeChat([
            '{"goal": "move", "scene_description": "clear", "reasoning": "safe", "action": {"command": "FRONT", "parameters": 12}}'
        ])
        adapter = LLMAdapter(chat)

        result = await adapter.iterate("goal", "image")

        self.assertTrue(result.ok)
        self.assertEqual(result.value.type, ActionType.WALK)
        self.assertEqual(result.value.metadata["command"], "FRONT")
        self.assertEqual(result.value.params["y"], 12.0)

    async def test_malformed_llm_response_returns_failure(self):
        chat = FakeChat(["this is not json"])
        adapter = LLMAdapter(chat)

        with self.assertLogs("common.robot.llm.LLMAdapter", level="WARNING") as cm:
            result = await adapter.iterate("goal", "image")

        self.assertFalse(result.ok)
        self.assertIsInstance(result.error, InvalidJSON)
        self.assertTrue(any("Failed to extract/parse JSON" in entry for entry in cm.output))

    async def test_unsafe_action_rejection(self):
        executor = FakeMotionExecutor()
        adapter = ActionAdapter(executor)
        action = RobotAction(command="FRONT", parameters={"value": 20})

        with self.assertLogs("common.robot.llm.ActionAdapter", level="WARNING") as cm:
            safe = adapter.checkSafety(action, lidar=[5.0, 6.0, 7.0])

        self.assertFalse(safe)
        self.assertTrue(any("Unsafe action rejected" in entry for entry in cm.output))

    async def test_executor_submission_path(self):
        executor = FakeMotionExecutor()
        adapter = ActionAdapter(executor)
        action = RobotAction(command="BACK", parameters={"value": 10})

        result = await adapter.execute(action)

        self.assertEqual(result.status, ActionStatus.SUCCESS)
        self.assertEqual(len(executor.submitted), 1)
        self.assertEqual(executor.submitted[0].type, ActionType.WALK)

    async def test_executor_submission_failure_logs_exception(self):
        executor = FailingMotionExecutor()
        adapter = ActionAdapter(executor)
        action = RobotAction(command="BACK", parameters={"value": 10})

        with self.assertLogs("common.robot.llm.ActionAdapter", level="ERROR") as cm:
            result = await adapter.execute(action)

        self.assertEqual(result.status, ActionStatus.FAILURE)
        self.assertIn("submit failed", result.message)
        self.assertTrue(any("Failed to submit action" in entry for entry in cm.output))

    async def test_orchestration_loop_progresses_until_complete(self):
        chat = FakeChat([
            '{"goal": "advance", "scene_description": "clear", "reasoning": "move now", "action": {"command": "FRONT", "parameters": {"value": 10}}}',
            '{"goal": "advance", "scene_description": "done", "reasoning": "goal reached", "action": {"command": "COMPLETE", "parameters": {"value": 0}}}'
        ])
        robot = FakeRobot()
        executor = FakeMotionExecutor()
        controller = LLMRobotController(robotController=robot, chat=chat, motionExecutor=executor)

        await controller.ask("reach target", maxIterations=3)

        self.assertTrue(executor.started)
        self.assertEqual(len(executor.submitted), 1)
        self.assertEqual(executor.submitted[0].type, ActionType.WALK)


if __name__ == "__main__":
    unittest.main()
