import numpy as np

from samegamerl.game.game import Game
from samegamerl.game.game_config import GameConfig, GameFactory


class SameGameEnv:
    def __init__(
        self,
        config: GameConfig | None = None,
        completion_reward: float = 10.0,
        partial_completion_base: float = 1.0,
        invalid_move_penalty: float = -0.01,
    ):
        if config is None:
            config = GameFactory.default()

        self.config = config
        self.num_colors = config.num_colors
        self.num_rows = config.num_rows
        self.num_cols = config.num_cols

        # Reward function parameters
        self.completion_reward = completion_reward
        self.partial_completion_base = partial_completion_base
        self.invalid_move_penalty = invalid_move_penalty

        self.game = Game(config)
        self.done = self.game.done()
        self.reset()

    def reset(self, board: None | list[list[int]] = None):
        self.game = Game(self.config)
        if board:
            self.game.set_board(board)

        self.done = self.game.done()
        return self.get_observation()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        if self.done:
            raise RuntimeError("Episode done. Call reset()")

        prev_left = self.game.left
        prev_singles = self.game.get_singles()

        row, col = self._to_2d(action)
        self.game.move((row, col))
        self.done = self.game.done()  # might get adjusted during reward calculation

        cur_left = self.game.left
        cur_singles = self.game.get_singles()

        reward = self.compute_reward(
            prev_left, cur_left, prev_singles, cur_singles, (row, col)
        )

        return self.get_observation(), reward, self.done, {}

    def compute_reward(
        self, prev_left, cur_left, prev_singles, cur_singles, action
    ) -> float:
        """Simple sparse reward function with configurable parameters.

        Rewards:
        - Full board completion: high positive reward
        - Game end (no moves left): smaller positive reward based on remaining tiles
        - Invalid moves: small negative penalty
        - All other moves: zero reward
        """
        # Full board completion - highest reward
        if cur_left == 0:
            return float(self.completion_reward)

        # Invalid move penalty
        if prev_left == cur_left:
            return float(self.invalid_move_penalty)

        # Game end due to no valid moves (only singles remaining after move)
        if cur_singles == cur_left and cur_left > 0:
            self.done = True
            # Partial completion reward based on how many tiles cleared
            tiles_cleared = self.config.total_cells - cur_left
            completion_ratio = tiles_cleared / self.config.total_cells
            return float(self.partial_completion_base * completion_ratio)

        # All other moves get zero reward - pure sparse reward signal
        return 0.0

    def get_observation(self) -> np.ndarray:
        return self._trainable_game(self.game.get_board())

    def _trainable_game(self, board: None | list[list[int]]) -> np.ndarray:
        """Convert board to one-hot encoded tensor for CNN input."""
        if not board:
            board = self.game.get_board()
        board_np = np.array(board)
        obs = np.zeros(
            (self.num_colors, self.num_rows, self.num_cols), dtype=np.float32
        )
        for color in range(self.num_colors):
            obs[color] = board_np == color
        return obs

    def _reverse_trainable_game(self, board: np.ndarray) -> list[list[int]]:
        """Convert one-hot encoded tensor back to integer board representation."""
        new_board = [[0 for _ in range(self.num_cols)] for _ in range(self.num_rows)]

        for color in range(self.num_colors):
            for row in range(self.num_rows):
                for col in range(self.num_cols):
                    if board[color, row, col] == 1:
                        new_board[row][col] = color

        return new_board

    def _to_2d(self, action):
        return divmod(action, self.num_cols)
