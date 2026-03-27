import asyncio
import uuid
import logging
from typing import TYPE_CHECKING
from common.utils.logging_config import bootstrap_logging
from common.llm.chats import LLMChat
from common.utils.images import toBase64Image
from common.utils.robot import getDistancesFromLidar, getDistanceDescription
from common.robot.RobotController import RobotController
from common.robot.llm.ActionAdapter import ActionAdapter, ActionStatus
from common.robot.llm.LLMAdapter import LLMAdapter
from server.core.motion_schema import ActionType

if TYPE_CHECKING:
    from server.core.motion_executor import MotionExecutor

logger = logging.getLogger(__name__)

class LLMRobotController:

    def __init__(self, robotController: RobotController, chat: LLMChat, motionExecutor: "MotionExecutor | None" = None):
        bootstrap_logging()
        self.robot = robotController
        self.chat = chat
        self.llmAdapter = LLMAdapter(chat)
        if motionExecutor is not None:
            self.motionExecutor = motionExecutor
        elif hasattr(robotController, "submit_action"):
            self.motionExecutor = robotController
        else:
            from server.core.motion_executor import MotionExecutor

            self.motionExecutor = MotionExecutor()

        if hasattr(self.motionExecutor, "start") and callable(getattr(self.motionExecutor, "start", None)):
            self.motionExecutor.start()
        self.actionAdapter = ActionAdapter(self.motionExecutor)
        self.sessionLock = asyncio.Lock()

    def __buildSceneDescription(self, prompt = None) -> str:
        preamble = f"Goal: {prompt}" if prompt is not None else "Here is the current view of the robot."
        if hasattr(self.robot, "getFrontLidarImage") and callable(getattr(self.robot, "getFrontLidarImage", None)):
            view_description = f"""
                {preamble}
                {getDistanceDescription(getDistancesFromLidar(self.robot.getFrontLidarImage(), 90, 8))}
                """.strip()
            return view_description
        return "Current view:"

    async def __getCameraImageBase64(self) -> str:
        return await asyncio.to_thread(lambda: toBase64Image(self.robot.getCameraImage()))

    async def __getSceneDescription(self, prompt: str | None = None) -> str:
        return await asyncio.to_thread(self.__buildSceneDescription, prompt)

    async def __getLidarImage(self, fov_degrees: int, offset_degrees: int = 0):
        return await asyncio.to_thread(self.robot.getLidarImage, fov_degrees, offset_degrees)
        
    async def ask(self, prompt: str, maxIterations: int = 30):
        if self.sessionLock.locked():
            logger.warning("LLMRobotController: Unable to acquire action lock, another session is in progress.")
            return

        await self.sessionLock.acquire()
        chat_id = str(uuid.uuid4())
        self.llmAdapter.clear()
        self.chat.set_chat_id(chat_id)
        logger.info("LLM session started chat_id=%s max_iterations=%d", chat_id, maxIterations)
        iterations = 0
        try:
            response = await self.llmAdapter.iterate(await self.__getSceneDescription(prompt), await self.__getCameraImageBase64())
            while iterations < maxIterations and not(response.ok and response.value.type == ActionType.COMPLETE):
                iterations += 1
                logger.info("LLM iteration chat_id=%s iteration=%d/%d", chat_id, iterations, maxIterations)
                if response.ok:
                    decision = self.actionAdapter.assess_safety(response.value, await self.__getLidarImage(30, 0))
                    if decision.is_safe:
                        try:
                            actionTask = asyncio.create_task(self.actionAdapter.execute(response.value))
                            await asyncio.wait_for(actionTask, 30)
                        except asyncio.TimeoutError:
                            logger.warning("Action execution timed out")
                            break

                        actionResult = await actionTask
                        if actionResult.status == ActionStatus.SUCCESS:
                            response = await self.llmAdapter.iterate(await self.__getSceneDescription(prompt), await self.__getCameraImageBase64())
                        else:
                            logger.warning("Action execution failed chat_id=%s reason=%s", chat_id, actionResult.message)
                            break
                    else:
                        logger.warning(
                            "Action rejected by safety chat_id=%s action=%s reason=%s params=%s",
                            chat_id,
                            response.value.type.value,
                            decision.reason,
                            response.value.params,
                        )
                        message = (
                            f"The action {response.value.type.value} with params {response.value.params} "
                            f"is unsafe and was rejected: {decision.reason}. "
                            "Please provide a different safe action."
                        )
                        response = await self.llmAdapter.iterate(await self.__getSceneDescription(f"{prompt}\n{message}"), await self.__getCameraImageBase64())
                elif not response.ok:
                    logger.warning("LLM response invalid chat_id=%s error=%s", chat_id, type(response.error).__name__)
                    message = (
                        "The response format is invalid. Please respond again following the schema.\n"
                        f"Expected Schema: {self.llmAdapter.responseSchema}\n"
                        f"Received Error: {response.error}\n"
                    )
                    response = await self.llmAdapter.iterate(await self.__getSceneDescription(f"{prompt}\n{message}"), await self.__getCameraImageBase64())
        except Exception:
            logger.exception("Error in LLMRobotController.ask chat_id=%s", chat_id)
        finally:
            # always release the lock
            self.sessionLock.release()
            logger.info("LLM session ended chat_id=%s iterations=%d", chat_id, iterations)
    