import unittest

from server.core.api_contract import (
    ContractValidationError,
    action_from_dict,
    action_to_dict,
    world_state_from_dict,
    world_state_to_dict,
)
from server.core.motion_schema import Action, ActionType, WorldState


class APIContractTests(unittest.TestCase):
    def test_action_round_trip(self):
        action = Action(
            type=ActionType.WALK,
            params={"x": 0, "y": 12.0, "angle": 0, "gait_type": "1", "speed": 5},
            ttl=0.8,
            metadata={"source": "test"},
        )

        payload = action_to_dict(action)
        reconstructed = action_from_dict(payload)

        self.assertEqual(reconstructed.type, ActionType.WALK)
        self.assertEqual(reconstructed.params["y"], 12.0)
        self.assertEqual(reconstructed.ttl, 0.8)
        self.assertEqual(reconstructed.metadata["source"], "test")

    def test_invalid_action_type_rejected(self):
        with self.assertRaises(ContractValidationError):
            action_from_dict({"type": "INVALID", "params": {}})

    def test_world_state_round_trip(self):
        state = WorldState(
            roll=1.5,
            pitch=-2.5,
            yaw=0.5,
            distance=42.0,
            is_balancing=True,
            current_action=ActionType.BALANCE,
            timestamp=123.456,
            is_safe=True,
            safety_reason="",
        )

        payload = world_state_to_dict(state)
        reconstructed = world_state_from_dict(payload)

        self.assertEqual(reconstructed.current_action, ActionType.BALANCE)
        self.assertEqual(reconstructed.distance, 42.0)
        self.assertTrue(reconstructed.is_balancing)


if __name__ == "__main__":
    unittest.main()