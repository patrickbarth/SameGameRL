import numpy as np

from samegamerl.game.game import Game
from samegamerl.game.game_config import GameConfig, GameFactory


class SameGameEnv:
    def __init__(self, config: GameConfig | None = None):
        if config is None:
            config = GameFactory.default()

        self.config = config
        self.num_colors = config.num_colors
        self.num_rows = config.num_rows
        self.num_cols = config.num_cols
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
        self, prev_left, cur_left, prev_singles, cur_singles, _action
    ) -> float:
        """Multi-component reward function balancing progress and completion.
        
        Prioritizes reducing isolated singles over raw tile removal for better
        long-term strategy development. Large completion bonus incentivizes
        fully clearing the board.
        """
        if self.game.left == 0:
            return float(5)
        if prev_singles == prev_left:
            self.done = True
            return -cur_singles / 10
        if prev_left == cur_left:
            return -0.3

        single = prev_singles - cur_singles
        removed = prev_left - cur_left
        total = self.num_rows * self.num_cols
        
        if single > 0:
            return (single / prev_left) * 3
        else:
            return removed / (10 * total)

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
