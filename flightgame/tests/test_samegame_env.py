import numpy as np
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