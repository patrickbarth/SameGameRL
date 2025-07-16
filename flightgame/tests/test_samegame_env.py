import numpy as np
import pytest
from flightgame.environments.samegame_env import SameGameEnv

def test_trainable_game():
    env = SameGameEnv(num_colors=3, num_rows=2, num_cols=2)
    board = [
        [0, 1],
        [2, 1]
    ]

    tensor = env._trainable_game(board)

    expected = np.array([
        [[1, 0],
         [0, 0]],

         [[0, 1],
          [0, 1]],

          [[0, 0],
           [1, 0]]
    ], dtype=np.float32)

    assert np.array_equal(tensor, expected), "One-hot encoding failed"

def test_env_initialization():
    env = SameGameEnv()
    assert env.num_colors > 0
    assert env.num_colors > 0
    assert env.num_rows > 0
    assert not env.done
    obs = env.get_observation()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (env.num_colors, env.num_rows, env.num_cols)

def test_reset_resets_state():
    env = SameGameEnv()
    env.done = True
    env.reset()
    assert not env.done
    obs = env.get_observation()
    assert obs.shape == (env.num_colors, env.num_rows, env.num_cols)

def test_step_returns_valid_output():
    env = SameGameEnv()
    initial_obs = env.get_observation()
    action = (0,0)

    obs, reward, done, info = env.step(action)

    assert isinstance(obs, np.ndarray)
    assert obs.shape == (env.num_colors, env.num_rows, env.num_cols)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)

def test_step_after_done_raises_error():
    env = SameGameEnv()
    env.done = True
    with pytest.raises(RuntimeError, match="Episode done"):
        env.step((0, 0))

def test_reward_matches_left_count():
    env = SameGameEnv()
    env.reset()
    _, reward, _, _ = env.step((0,0))
    assert reward == env.game.left

        