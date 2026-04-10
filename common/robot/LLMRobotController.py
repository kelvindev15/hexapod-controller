import asyncio
import uuid
import logging
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any
from common.utils.logging_config import bootstrap_logging
from common.llm.chats import LLMChat
from common.utils.images import toBase64Image
from common.utils.robot import getDistancesFromLidar, getDistanceDescription
from common.robot.RobotController import RobotController
from common.robot.llm.ActionAdapter import ActionAdapter, ActionStatus
from common.robot.llm.LLMAdapter import LLMAdapter
from common.utils.experiment_reporter import ExperimentReporter
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
        reporter: ExperimentReporter | None = None,
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
        self.reporter = reporter
        self.last_session_outcome: dict[str, Any] | None = None

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

    def _is_provider_throttle_error(self, error: Exception | None) -> bool:
        if error is None:
            return False
        name = type(error).__name__.lower()
        message = str(error).lower()
        return (
            "llmratelimiterror" in name
            or "resourceexhausted" in name
            or "ratelimit" in name
            or "resource exhausted" in message
            or "rate limit" in message
            or "429" in message
        )

    def _safe_world_state(self) -> dict[str, Any] | None:
        state_getter = getattr(self.motionExecutor, "get_state", None)
        if not callable(state_getter):
            return None
        try:
            state = state_getter()
            return self._to_serializable(state)
        except Exception:
            logger.debug("Failed to read world state", exc_info=True)
            return None

    def _to_serializable(self, value: Any) -> Any:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, Enum):
            return value.value
        if is_dataclass(value):
            return self._to_serializable(asdict(value))
        if isinstance(value, dict):
            return {str(key): self._to_serializable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._to_serializable(item) for item in value]
        return str(value)

    def _report_event(self, event_type: str, **payload: Any) -> None:
        if self.reporter is None:
            return
        try:
            self.reporter.log_event(event_type, **payload)
        except Exception:
            logger.warning("Experiment reporter failed for event_type=%s", event_type, exc_info=True)

    def __buildSceneDescription(self, prompt = None) -> str:
        preamble = "This is the new scene."
        if prompt is not None:
            preamble = f"{preamble} Goal: {prompt}"
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
        return preamble

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
        
    async def ask(self, prompt: str, maxIterations: int = 30, reporter: ExperimentReporter | None = None):
        if self.sessionLock.locked():
            logger.warning("LLMRobotController: Unable to acquire action lock, another session is in progress.")
            return

        await self.sessionLock.acquire()
        self._interrupt_requested = False
        self._session_task = asyncio.current_task()
        chat_id = str(uuid.uuid4())
        session_reporter = reporter if reporter is not None else self.reporter
        self.llmAdapter.clear()
        self.chat.set_chat_id(chat_id)
        logger.info("LLM session started chat_id=%s max_iterations=%d", chat_id, maxIterations)
        iterations = 0
        termination_reason = "unknown"

        if session_reporter is not None:
            self.reporter = session_reporter
        self._report_event(
            "session_start",
            goal_text=prompt,
            chat_id=chat_id,
            session_id=chat_id,
            iteration_index=0,
        )
        try:
            if self._interrupt_requested:
                termination_reason = "interrupted_before_start"
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
                self._report_event(
                    "llm_iteration",
                    goal_text=prompt,
                    chat_id=chat_id,
                    session_id=chat_id,
                    iteration_index=iterations,
                )
                if response.ok:
                    world_state_before = self._safe_world_state()
                    self._report_event(
                        "action_proposed",
                        goal_text=prompt,
                        chat_id=chat_id,
                        session_id=chat_id,
                        iteration_index=iterations,
                        action_proposed=self._to_serializable(response.value),
                        world_state_before=world_state_before,
                    )
                    decision = self.actionAdapter.assess_safety(response.value, await self.__getLidarImage(30, 0))
                    if decision.is_safe:
                        try:
                            actionTask = asyncio.create_task(self.actionAdapter.execute(response.value))
                            await asyncio.wait_for(actionTask, 30)
                        except asyncio.CancelledError:
                            logger.info("LLM action execution interrupted chat_id=%s", chat_id)
                            termination_reason = "interrupted_during_execution"
                            self._report_event(
                                "session_interrupted",
                                goal_text=prompt,
                                chat_id=chat_id,
                                session_id=chat_id,
                                iteration_index=iterations,
                            )
                            break
                        except asyncio.TimeoutError:
                            logger.warning("Action execution timed out")
                            termination_reason = "action_timeout"
                            self._report_event(
                                "action_executed",
                                goal_text=prompt,
                                chat_id=chat_id,
                                session_id=chat_id,
                                iteration_index=iterations,
                                action_executed=self._to_serializable(response.value),
                                execution_status="timeout",
                                world_state_before=world_state_before,
                                world_state_after=self._safe_world_state(),
                            )
                            break

                        actionResult = await actionTask
                        self._report_event(
                            "action_executed",
                            goal_text=prompt,
                            chat_id=chat_id,
                            session_id=chat_id,
                            iteration_index=iterations,
                            action_executed=self._to_serializable(response.value),
                            execution_status=(
                                "success" if actionResult.status == ActionStatus.SUCCESS else "failure"
                            ),
                            error_message=actionResult.message if actionResult.status != ActionStatus.SUCCESS else None,
                            world_state_before=world_state_before,
                            world_state_after=self._safe_world_state(),
                        )
                        if self._interrupt_requested:
                            termination_reason = "interrupted"
                            self._report_event(
                                "session_interrupted",
                                goal_text=prompt,
                                chat_id=chat_id,
                                session_id=chat_id,
                                iteration_index=iterations,
                            )
                            break
                        if actionResult.status == ActionStatus.SUCCESS:
                            scene, image = await self.__gather_scene_and_image(prompt)
                            response = await self.llmAdapter.iterate(scene, image)
                        else:
                            logger.warning("Action execution failed chat_id=%s reason=%s", chat_id, actionResult.message)
                            termination_reason = "action_execution_failed"
                            break
                    else:
                        logger.warning(
                            "Action rejected by safety chat_id=%s action=%s reason=%s params=%s",
                            chat_id,
                            response.value.type.value,
                            decision.reason,
                            response.value.params,
                        )
                        self._report_event(
                            "action_rejected",
                            goal_text=prompt,
                            chat_id=chat_id,
                            session_id=chat_id,
                            iteration_index=iterations,
                            action_proposed=self._to_serializable(response.value),
                            safety_decision="rejected",
                            safety_reason=decision.reason,
                            world_state_before=world_state_before,
                        )
                        message = (
                            f"The action {response.value.type.value} with params {response.value.params} "
                            f"is unsafe and was rejected: {decision.reason}. "
                            "Please provide a different safe action."
                        )
                        scene, image = await self.__gather_scene_and_image(f"{prompt}\n{message}")
                        response = await self.llmAdapter.iterate(scene, image)
                elif not response.ok:
                    if self._is_provider_throttle_error(response.error):
                        logger.warning(
                            "LLM provider throttled chat_id=%s error=%s; ending session gracefully",
                            chat_id,
                            type(response.error).__name__,
                        )
                        termination_reason = "provider_throttle"
                        self._report_event(
                            "provider_throttle",
                            goal_text=prompt,
                            chat_id=chat_id,
                            session_id=chat_id,
                            iteration_index=iterations,
                            error_type=type(response.error).__name__,
                            error_message=str(response.error),
                        )
                        break
                    logger.warning("LLM response invalid chat_id=%s error=%s", chat_id, type(response.error).__name__)
                    self._report_event(
                        "llm_invalid_response",
                        goal_text=prompt,
                        chat_id=chat_id,
                        session_id=chat_id,
                        iteration_index=iterations,
                        error_type=type(response.error).__name__,
                        error_message=str(response.error),
                    )
                    message = (
                        "The response format is invalid. Please respond again following the schema.\n"
                        f"Expected Schema: {self.llmAdapter.responseSchema}\n"
                        f"Received Error: {response.error}\n"
                    )
                    scene, image = await self.__gather_scene_and_image(f"{prompt}\n{message}")
                    response = await self.llmAdapter.iterate(scene, image)
            if termination_reason == "unknown":
                if self._interrupt_requested:
                    termination_reason = "interrupted"
                elif response.ok and response.value.type == ActionType.COMPLETE:
                    termination_reason = "goal_complete"
                elif iterations >= maxIterations:
                    termination_reason = "max_iterations"
                else:
                    termination_reason = "ended"
            if self._interrupt_requested:
                logger.info("LLM session interrupted chat_id=%s", chat_id)
                self._report_event(
                    "session_interrupted",
                    goal_text=prompt,
                    chat_id=chat_id,
                    session_id=chat_id,
                    iteration_index=iterations,
                )
        except asyncio.CancelledError:
            self._interrupt_requested = True
            termination_reason = "cancelled"
            logger.info("LLM session cancelled chat_id=%s", chat_id)
            self._report_event(
                "session_interrupted",
                goal_text=prompt,
                chat_id=chat_id,
                session_id=chat_id,
                iteration_index=iterations,
            )
        except Exception:
            termination_reason = "error"
            logger.exception("Error in LLMRobotController.ask chat_id=%s", chat_id)
            self._report_event(
                "session_error",
                goal_text=prompt,
                chat_id=chat_id,
                session_id=chat_id,
                iteration_index=iterations,
            )
        finally:
            self._session_task = None
            # always release the lock
            if self.sessionLock.locked():
                self.sessionLock.release()
            logger.info("LLM session ended chat_id=%s iterations=%d", chat_id, iterations)
            self.last_session_outcome = {
                "chat_id": chat_id,
                "iterations": iterations,
                "termination_reason": termination_reason,
                "success": termination_reason == "goal_complete",
            }
            self._report_event(
                "session_end",
                goal_text=prompt,
                chat_id=chat_id,
                session_id=chat_id,
                iteration_index=iterations,
                termination_reason=termination_reason,
                success=termination_reason == "goal_complete",
            )
    