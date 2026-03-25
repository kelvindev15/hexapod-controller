#!/usr/bin/env python3
import time
import logging
from common.utils.logging_config import bootstrap_logging
from server.core.motion_executor import MotionExecutor
from server.core.motion_schema import Action, ActionType


logger = logging.getLogger(__name__)

def main():
    """
    Example runner for the MotionExecutor.
    Shows how to submit actions and monitor state from an external loop.
    """
    bootstrap_logging()
    executor = MotionExecutor()
    executor.start()
    
    logger.info("Motion Executor started state=%s", executor.get_state())
    
    try:
        # 1. Walk Forward for 5 seconds
        logger.info("Action requested action=%s ttl=5.0", ActionType.WALK.value)
        walk_action = Action(
            type=ActionType.WALK,
            params={'y': 25, 'gait_type': '1', 'speed': 5},
            ttl=5.0
        )
        executor.submit_action(walk_action)
        
        while True:
            state = executor.get_state()
            logger.info(
                "state distance_cm=%.1f roll=%.1f pitch=%.1f active_action=%s",
                state.distance,
                state.roll,
                state.pitch,
                state.current_action,
            )
            
            if not state.is_safe:
                logger.warning("Safety triggered reason=%s", state.safety_reason)
            
            # Check if current action finished (was walk but now something else/none)
            if state.current_action != ActionType.WALK and state.timestamp - executor.action_start_time > 5.1:
                break
                
            time.sleep(0.5)

        # 2. Set Attitude
        logger.info("Action requested action=%s", ActionType.ATTITUDE.value)
        lean_action = Action(
            type=ActionType.ATTITUDE,
            params={'roll': 10, 'pitch': 0, 'yaw': 0}
        )
        executor.submit_action(lean_action)
        time.sleep(2)

        # 3. Enter Balance Mode
        logger.info("Action requested action=%s", ActionType.BALANCE.value)
        balance_action = Action(type=ActionType.BALANCE)
        executor.submit_action(balance_action)
        time.sleep(5)

        # 4. Relax
        logger.info("Action requested action=%s", ActionType.RELAX.value)
        executor.submit_action(Action(type=ActionType.RELAX))
        time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Stopping motion runner")
    finally:
        executor.submit_action(Action(type=ActionType.STOP))
        executor.stop()

if __name__ == "__main__":
    main()
