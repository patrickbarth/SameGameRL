"""
Shared test utilities for SameGameRL tests.

Following YAGNI principles, this module contains only what is actually used:
- Static board configurations (TEST_BOARD_CONFIGS)
- Test models (TinyTestModel)
- Assertion utilities for validation
"""

import numpy as np
import torch.nn as nn

TEST_BOARD_CONFIGS = {
    "empty_2x2": [[0, 0], [0, 0]],
    "single_color_2x2": [[1, 1], [1, 1]],
    "checkerboard_2x2": [[1, 2], [2, 1]],
    "mixed_3x3": [[1, 1, 2], [2, 1, 2], [2, 2, 1]],
    "complex_4x4": [[1, 1, 2, 3], 
                    [1, 2, 2, 3], 
                    [2, 2, 3, 3], 
                    [3, 1, 1, 2]],
    "singles_only": [[1, 2, 3], [2, 3, 1], [3, 1, 2]],
    "winning_scenario": [[1, 1], [1, 0]],  # Can win in one move
}


class TinyTestModel(nn.Module):
    """Minimal neural network for fast testing"""

    def __init__(self, input_size: int = 4, output_size: int = 4):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        return self.fc(x_flat)


def assert_valid_observation(obs: np.ndarray, expected_shape: tuple[int, ...]):
    """Assert that an observation has the correct shape and properties"""
    assert isinstance(obs, np.ndarray), "Observation must be numpy array"
    assert (
        obs.shape == expected_shape
    ), f"Observation shape {obs.shape} != expected {expected_shape}"
    assert obs.dtype == np.float32, f"Observation dtype {obs.dtype} != float32"
    assert np.all(
        (obs == 0) | (obs == 1)
    ), "Observation should be one-hot encoded (only 0s and 1s)"


def assert_valid_step_return(step_return: tuple[object, ...]):
    """Assert that a step return follows the (obs, reward, done, info) pattern"""
    assert len(step_return) == 4, "Step return should be (obs, reward, done, info)"
    obs, reward, done, info = step_return
    assert isinstance(obs, np.ndarray), "Observation should be numpy array"
    assert isinstance(reward, (int, float)), "Reward should be numeric"
    assert isinstance(done, bool), "Done should be boolean"
    assert isinstance(info, dict), "Info should be dictionary"


def assert_valid_action_range(action: int, min_action: int, max_action: int):
    """Assert that action is within valid range"""
    assert (
        min_action <= action <= max_action
    ), f"Action {action} outside range [{min_action}, {max_action}]"
    assert isinstance(action, int), "Action should be integer"


def assert_reward_bounds(
    reward: float, min_reward: float = -10.0, max_reward: float = 10.0
):
    """Assert that reward is within reasonable bounds"""
    assert (
        min_reward <= reward <= max_reward
    ), f"Reward {reward} outside bounds [{min_reward}, {max_reward}]"
