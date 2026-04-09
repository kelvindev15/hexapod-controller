import asyncio
import uuid
import logging
from typing import TYPE_CHECKING, Any
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
    _CAMERA_CAPTURE_STABILIZATION_SECONDS = 0.75

    def __init__(
        self,
        robotController: RobotController,
        chat: LLMChat,
        motionExecutor: "MotionExecutor | None" = None,
        system_prompt_file: str | None = None,
    ):
        bootstrap_logging()
        self.robot = robotController
        self.chat = chat
        self.llmAdapter = LLMAdapter(chat, system_prompt_file=system_prompt_file)
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
        self._interrupt_requested = False
        self._session_task: asyncio.Task | None = None

    def interrupt(self) -> bool:
        was_running = self.sessionLock.locked()
        self._interrupt_requested = True
        if self._session_task is not None and not self._session_task.done():
            self._session_task.cancel()
        return was_running

    def _to_positive_int_or_none(self, value: Any) -> int | None:
        try:
            candidate = int(value)
        except (TypeError, ValueError):
            return None
        return candidate if candidate > 0 else None

    def _read_distance_sensor_profile(self) -> dict[str, Any]:
        profile_getter = getattr(self.robot, "getDistanceSensorProfile", None)
        if callable(profile_getter):
            try:
                profile = profile_getter()
                if isinstance(profile, dict):
                    return profile
            except Exception:
                logger.debug("Failed to read distance sensor profile", exc_info=True)
        return {}

    def _resolve_section_count(self, readings: list[float], sensor_profile: dict[str, Any]) -> int:
        reading_count = len(readings)
        if reading_count <= 1:
            return 1

        for key in ("sections", "channels", "samples"):
            resolved = self._to_positive_int_or_none(sensor_profile.get(key))
            if resolved is not None:
                return max(1, min(resolved, reading_count))

        return max(1, min(8, reading_count))

    def __buildSceneDescription(self, prompt = None) -> str:
        preamble = f"Goal: {prompt}" if prompt is not None else "Here is the current view of the robot."
        if hasattr(self.robot, "getFrontLidarImage") and callable(getattr(self.robot, "getFrontLidarImage", None)):
            readings = self.robot.getFrontLidarImage()
            sensor_profile = self._read_distance_sensor_profile()
            section_count = self._resolve_section_count(readings, sensor_profile)
            sensor_label = str(sensor_profile.get("label") or sensor_profile.get("type") or "Distance sensor")
            view_description = f"""
                {preamble}
                {getDistanceDescription(getDistancesFromLidar(readings, 90, section_count), sensor_label=sensor_label)}
                """.strip()
            return view_description
        return "Current view:"

    async def __getCameraImageBase64(self) -> str:
        # Let motion settle briefly to reduce blur in LLM-bound camera frames.
        await asyncio.sleep(self._CAMERA_CAPTURE_STABILIZATION_SECONDS)
        return await asyncio.to_thread(lambda: toBase64Image(self.robot.getCameraImage()))

    async def __getSceneDescription(self, prompt: str | None = None) -> str:
        return await asyncio.to_thread(self.__buildSceneDescription, prompt)

    async def __getLidarImage(self, fov_degrees: int, offset_degrees: int = 0):
        return await asyncio.to_thread(self.robot.getLidarImage, fov_degrees, offset_degrees)

    async def __gather_scene_and_image(self, prompt: str | None = None) -> tuple[str, str]:
        """Fetch scene description and camera image in parallel."""
        scene, image = await asyncio.gather(
            self.__getSceneDescription(prompt),
            self.__getCameraImageBase64()
        )
        return scene, image
        
    async def ask(self, prompt: str, maxIterations: int = 30):
        if self.sessionLock.locked():
            logger.warning("LLMRobotController: Unable to acquire action lock, another session is in progress.")
            return

        await self.sessionLock.acquire()
        self._interrupt_requested = False
        self._session_task = asyncio.current_task()
        chat_id = str(uuid.uuid4())
        self.llmAdapter.clear()
        self.chat.set_chat_id(chat_id)
        logger.info("LLM session started chat_id=%s max_iterations=%d", chat_id, maxIterations)
        iterations = 0
        try:
            if self._interrupt_requested:
                return
            scene, image = await self.__gather_scene_and_image(prompt)
            response = await self.llmAdapter.iterate(scene, image)
            while (
                iterations < maxIterations
                and not self._interrupt_requested
                and not (response.ok and response.value.type == ActionType.COMPLETE)
            ):
                iterations += 1
                logger.info("LLM iteration chat_id=%s iteration=%d/%d", chat_id, iterations, maxIterations)
                if response.ok:
                    decision = self.actionAdapter.assess_safety(response.value, await self.__getLidarImage(30, 0))
                    if decision.is_safe:
                        try:
                            actionTask = asyncio.create_task(self.actionAdapter.execute(response.value))
                            await asyncio.wait_for(actionTask, 30)
                        except asyncio.CancelledError:
                            logger.info("LLM action execution interrupted chat_id=%s", chat_id)
                            break
                        except asyncio.TimeoutError:
                            logger.warning("Action execution timed out")
                            break

                        actionResult = await actionTask
                        if self._interrupt_requested:
                            break
                        if actionResult.status == ActionStatus.SUCCESS:
                            scene, image = await self.__gather_scene_and_image(prompt)
                            response = await self.llmAdapter.iterate(scene, image)
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
                        scene, image = await self.__gather_scene_and_image(f"{prompt}\n{message}")
                        response = await self.llmAdapter.iterate(scene, image)
                elif not response.ok:
                    logger.warning("LLM response invalid chat_id=%s error=%s", chat_id, type(response.error).__name__)
                    message = (
                        "The response format is invalid. Please respond again following the schema.\n"
                        f"Expected Schema: {self.llmAdapter.responseSchema}\n"
                        f"Received Error: {response.error}\n"
                    )
                    scene, image = await self.__gather_scene_and_image(f"{prompt}\n{message}")
                    response = await self.llmAdapter.iterate(scene, image)
            if self._interrupt_requested:
                logger.info("LLM session interrupted chat_id=%s", chat_id)
        except asyncio.CancelledError:
            self._interrupt_requested = True
            logger.info("LLM session cancelled chat_id=%s", chat_id)
        except Exception:
            logger.exception("Error in LLMRobotController.ask chat_id=%s", chat_id)
        finally:
            self._session_task = None
            # always release the lock
            if self.sessionLock.locked():
                self.sessionLock.release()
            logger.info("LLM session ended chat_id=%s iterations=%d", chat_id, iterations)
    