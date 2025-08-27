from copy import deepcopy
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

    def reset(self, board: None | list[list[int]] = None, seed=42):
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
        # return removed / prev_left
        # return removed / 10
        if single > 0:
            return (single / prev_left) * 3
        else:
            return removed / (10 * total)
        # return single * ((total - prev_left) / total)

    def get_observation(self) -> np.ndarray:
        return self._trainable_game(self.game.get_board())

    def _trainable_game(self, board: None | list[list[int]]) -> np.ndarray:
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
        # Initialize the board with all zeros
        new_board = [[0] * self.num_cols] * self.num_rows

        # Iterate over colors, rows, and columns
        for color in range(self.num_colors):
            for row in range(self.num_rows):
                for col in range(self.num_cols):
                    # If the value in the layers array is 1, set the corresponding position on the board to the current color
                    if board[color, row, col] == 1:
                        new_board[row][col] = color

        return new_board

    def _to_2d(self, action):
        return divmod(action, self.num_cols)

        """
        layers = []
        for color in range(NUM_COLORS):
            layer = []
            for row in range(NUM_ROWS):
                srow = []
                for col in range(NUM_COLS):
                    if self.board[row][col] == color:
                        srow.append(1)
                    else:
                        srow.append(0)
                layer.append(srow)
            layers.append(layer)
        return np.array(layers)
        """

    """

    def reward_all(self, board):
        board = self.inverse_trainable_game(board)
        rewards = []
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                rewards.append(self.reward(row, col, board))
        return np.array(rewards)

    def reward(self, row, col, board):
        tile = (row, col)
        old_left = self.left
        new_left = self.left-1
        new_neighbours = [tile]
        old_neighbours = [tile]
        color = board[tile[0]][tile[1]]
        first = True

        if board[row][col] == 0:
            return -0.2

        # finding all the adjacent neighbours with the same color
        while first or len(old_neighbours) != len(new_neighbours):
            if not first:
                old_neighbours = new_neighbours.copy()
            for neighbour in self.get_neighbours(old_neighbours):
                 if (neighbour[0], neighbour[1]) not in new_neighbours and board[neighbour[0]][neighbour[1]] == color:
                    new_neighbours.append(neighbour)
                    new_left -= 1
            if len(new_neighbours) == 1:
                return -0.1
            first = False

        reward = (old_left - new_left) / 10

        return reward
        """
