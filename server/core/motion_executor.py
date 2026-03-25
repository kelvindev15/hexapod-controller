import time
import threading
import queue
import copy
import math
import logging
from typing import Optional
from common.utils.logging_config import bootstrap_logging
from .control import Control
from .motion_schema import Action, ActionType, WorldState, ActionSafetyPolicy, SafetyContext
from ..drivers.ultrasonic import Ultrasonic


logger = logging.getLogger(__name__)

class MotionExecutor:
    """
    Thread-safe, preemptible motion executor for the hexapod.
    Runs at a fixed tick rate and handles safety overrides.
    """
    TICK_RATE = 50.0  # Hz
    TICK_INTERVAL = 1.0 / TICK_RATE
    
    def __init__(self):
        self.control = Control()
        self.ultrasonic = Ultrasonic()
        self.action_queue = queue.Queue()
        self.current_action: Optional[Action] = None
        self.action_start_time = 0.0
        
        self.state = WorldState()
        self.state_lock = threading.Lock()
        self.running = False
        self._thread = None
        
        # Gait/Balance state for tick-based execution
        self.gait_step = 0
        self.gait_params = {}
        self.is_balancing = False
        self.safety_policy = ActionSafetyPolicy()

    def start(self):
        if self._thread is not None:
            return
        bootstrap_logging()
        self.running = True
        self._thread = threading.Thread(target=self._main_loop, daemon=True)
        self._thread.start()
        logger.info("MotionExecutor started tick_rate_hz=%.1f", self.TICK_RATE)

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join()
            self._thread = None
        logger.info("MotionExecutor stopped")

    def submit_action(self, action: Action):
        """Submit a new action, preempting the current one."""
        # Validate/Clamp parameters here if needed
        self.action_queue.put(action)

    def cancel_current_action(self):
        self.submit_action(Action(type=ActionType.STOP))

    def get_state(self) -> WorldState:
        with self.state_lock:
            return copy.deepcopy(self.state)

    def _main_loop(self):
        while self.running:
            start_tick = time.time()
            try:
                # 1. Update Sensors & Safety
                self._update_sensors()
                safety_triggered = self._check_safety()

                # 2. Process New Actions
                try:
                    # Always check for new actions, even if current is running (preemption)
                    new_action = self.action_queue.get_nowait()
                    self._handle_new_action(new_action)
                except queue.Empty:
                    pass

                # 3. Check TTL/Expirations
                self._check_action_expiry()

                # 4. Execute Tick
                if not safety_triggered:
                    self._execute_tick()
                else:
                    self._handle_safety_shutdown()
            except Exception:
                logger.exception("Unhandled exception in MotionExecutor tick")

            # 5. Maintain Tick Rate
            elapsed = time.time() - start_tick
            sleep_time = max(0, self.TICK_INTERVAL - elapsed)
            time.sleep(sleep_time)

    def _update_sensors(self):
        try:
            roll, pitch, yaw = self.control.imu.update_imu_state()
        except Exception:
            logger.exception("IMU update failed")
            with self.state_lock:
                roll, pitch, yaw = self.state.roll, self.state.pitch, self.state.yaw

        try:
            dist = self.ultrasonic.get_distance()
        except Exception:
            logger.exception("Ultrasonic update failed")
            dist = None
        
        with self.state_lock:
            self.state.roll = roll
            self.state.pitch = pitch
            self.state.yaw = yaw
            self.state.distance = dist if dist is not None else 999.0
            self.state.timestamp = time.time()

    def _check_safety(self) -> bool:
        action = self.current_action or Action(type=ActionType.STOP)
        with self.state_lock:
            distance = self.state.distance
            roll = self.state.roll
            pitch = self.state.pitch
        decision = self.safety_policy.evaluate_runtime(
            action,
            SafetyContext(
                distance=distance,
                roll=roll,
                pitch=pitch,
                has_range_data=distance is not None,
            ),
        )

        with self.state_lock:
            self.state.is_safe = decision.is_safe
            self.state.safety_reason = decision.reason
        return not decision.is_safe

    def _handle_new_action(self, action: Action):
        previous_action = self.current_action.type.value if self.current_action else None
        self.current_action = action
        self.action_start_time = time.time()
        with self.state_lock:
            self.state.current_action = action.type

        logger.info(
            "Action accepted action=%s ttl=%.2f preempted=%s params=%s",
            action.type.value,
            action.ttl,
            previous_action,
            action.params,
        )
        
        # Reset internal tick states
        self.gait_step = 0
        self.is_balancing = (action.type == ActionType.BALANCE)
        
        if action.type == ActionType.RELAX:
            self.control.relax(True)
        elif action.type == ActionType.STOP:
            self.control.relax(False) # Just go to neutral stance
        elif action.type == ActionType.POSITION:
            x = self.control.restrict_value(action.params.get('x', 0), -40, 40)
            y = self.control.restrict_value(action.params.get('y', 0), -40, 40)
            z = self.control.restrict_value(action.params.get('z', 0), -20, 20)
            self.control.move_position(x, y, z)
        elif action.type == ActionType.ATTITUDE:
            r = self.control.restrict_value(action.params.get('roll', 0), -15, 15)
            p = self.control.restrict_value(action.params.get('pitch', 0), -15, 15)
            y = self.control.restrict_value(action.params.get('yaw', 0), -15, 15)
            points = self.control.calculate_posture_balance(r, p, y)
            self.control.transform_coordinates(points)
            self.control.set_leg_angles()

    def _check_action_expiry(self):
        if self.current_action and self.current_action.ttl > 0:
            if time.time() - self.action_start_time > self.current_action.ttl:
                logger.info(
                    "Action expired action=%s ttl=%.2f",
                    self.current_action.type.value,
                    self.current_action.ttl,
                )
                self.submit_action(Action(type=ActionType.STOP))

    def _execute_tick(self):
        if not self.current_action:
            return

        if self.current_action.type == ActionType.WALK:
            self._tick_gait()
        elif self.current_action.type == ActionType.BALANCE:
            self._tick_balance()

    def _tick_gait(self):
        """Non-blocking tick for gait movement."""
        if not self.current_action:
            return
            
        # 1. Action Setup on first tick
        if self.gait_step == 0:
            data = self.current_action.params
            self.gait_params['gait_type'] = data.get('gait_type', "1")
            self.gait_params['x'] = self.control.restrict_value(int(data.get('x', 0)), -35, 35)
            self.gait_params['y'] = self.control.restrict_value(int(data.get('y', 0)), -35, 35)
            self.gait_params['angle'] = int(data.get('angle', 0))
            
            # Map speed (2-10) to Frame count (F)
            speed = int(data.get('speed', 5))
            if self.gait_params['gait_type'] == "1":
                self.gait_params['F'] = round(self.control.map_value(speed, 2, 10, 126, 22))
            else:
                self.gait_params['F'] = round(self.control.map_value(speed, 2, 10, 171, 45))
                
            self.gait_params['Z'] = 40.0
            self.gait_params['z_step_size'] = self.gait_params['Z'] / self.gait_params['F']
            self.gait_params['points'] = copy.deepcopy(self.control.body_points)
            
            # Pre-calculate xy increments
            F = self.gait_params['F']
            angle = self.gait_params['angle']
            x = self.gait_params['x']
            y = self.gait_params['y']
            points = self.gait_params['points']
            
            self.gait_params['xy_inc'] = [[0, 0] for _ in range(6)]
            for i in range(6):
                self.gait_params['xy_inc'][i][0] = ((points[i][0] * math.cos(angle / 180 * math.pi) + points[i][1] * math.sin(angle / 180 * math.pi) - points[i][0]) + x) / F
                self.gait_params['xy_inc'][i][1] = ((-points[i][0] * math.sin(angle / 180 * math.pi) + points[i][1] * math.cos(angle / 180 * math.pi) - points[i][1]) + y) / F

        # 2. Sequential Step Logic
        F = self.gait_params['F']
        j = self.gait_step
        points = self.gait_params['points']
        xy = self.gait_params['xy_inc']
        z_step = self.gait_params['z_step_size']
        Z = self.gait_params['Z']
        body_h = self.control.body_height

        if self.gait_params['gait_type'] == "1":
            # Tripod Gait logic from run_gait()
            for i in range(3):
                if j < (F / 8):
                    points[2 * i][0] -= 4 * xy[2 * i][0]
                    points[2 * i][1] -= 4 * xy[2 * i][1]
                    points[2 * i + 1][0] += 8 * xy[2 * i + 1][0]
                    points[2 * i + 1][1] += 8 * xy[2 * i + 1][1]
                    points[2 * i + 1][2] = Z + body_h
                elif j < (F / 4):
                    points[2 * i][0] -= 4 * xy[2 * i][0]
                    points[2 * i][1] -= 4 * xy[2 * i][1]
                    points[2 * i + 1][2] -= z_step * 8
                elif j < (3 * F / 8):
                    points[2 * i][2] += z_step * 8
                    points[2 * i + 1][0] -= 4 * xy[2 * i + 1][0]
                    points[2 * i + 1][1] -= 4 * xy[2 * i + 1][1]
                elif j < (5 * F / 8):
                    points[2 * i][0] += 8 * xy[2 * i][0]
                    points[2 * i][1] += 8 * xy[2 * i][1]
                    points[2 * i + 1][0] -= 4 * xy[2 * i + 1][0]
                    points[2 * i + 1][1] -= 4 * xy[2 * i + 1][1]
                elif j < (3 * F / 4):
                    points[2 * i][2] -= z_step * 8
                    points[2 * i + 1][0] -= 4 * xy[2 * i + 1][0]
                    points[2 * i + 1][1] -= 4 * xy[2 * i + 1][1]
                elif j < (7 * F / 8):
                    points[2 * i][0] -= 4 * xy[2 * i][0]
                    points[2 * i][1] -= 4 * xy[2 * i][1]
                    points[2 * i + 1][2] += z_step * 8
                elif j < (F):
                    points[2 * i][0] -= 4 * xy[2 * i][0]
                    points[2 * i][1] -= 4 * xy[2 * i][1]
                    points[2 * i + 1][0] += 8 * xy[2 * i + 1][0]
                    points[2 * i + 1][1] += 8 * xy[2 * i + 1][1]
        else:
            # Quadruped Gait logic (gait "2")
            number = [5, 2, 1, 0, 3, 4]
            phase_len = int(F / 6)
            leg_idx = j // phase_len
            step_in_phase = j % phase_len
            
            if leg_idx < 6:
                target_leg = number[leg_idx]
                for k in range(6):
                    if target_leg == k:
                        if step_in_phase < int(F / 18):
                            points[k][2] += 18 * z_step
                        elif step_in_phase < int(F / 9):
                            points[k][0] += 30 * xy[k][0]
                            points[k][1] += 30 * xy[k][1]
                        elif step_in_phase < int(F / 6):
                            points[k][2] -= 18 * z_step
                    else:
                        points[k][0] -= 2 * xy[k][0]
                        points[k][1] -= 2 * xy[k][1]

        # 3. Apply changes
        self.control.transform_coordinates(points)
        self.control.set_leg_angles()
        
        # 4. Increment and Loop
        self.gait_step += 1
        if self.gait_step >= F:
            self.gait_step = 0

    def _tick_balance(self):
        roll, pitch, _ = self.control.imu.update_imu_state()
        r = self.control.pid_controller.pid_calculate(roll)
        p = self.control.pid_controller.pid_calculate(pitch)
        points = self.control.calculate_posture_balance(r, p, 0)
        self.control.transform_coordinates(points)
        self.control.set_leg_angles()

    def _handle_safety_shutdown(self):
        if self.current_action and self.current_action.type != ActionType.STOP:
            with self.state_lock:
                reason = self.state.safety_reason
            logger.warning(
                "Safety stop action=%s reason=%s",
                self.current_action.type.value,
                reason,
            )
            self.control.relax(False) # Immediate stop to neutral
            self.current_action = Action(type=ActionType.STOP)
            with self.state_lock:
                self.state.current_action = ActionType.STOP
