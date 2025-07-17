import numpy as np

from flightgame.game.game import Game
from flightgame.game.game_params import NUM_COLS, NUM_ROWS, NUM_COLORS

class SameGameEnv:
    def __init__(self, num_colors=NUM_COLORS, num_rows=NUM_ROWS, num_cols=NUM_COLS):
        self.num_colors = num_colors
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.game = Game(num_colors=self.num_colors, num_rows=self.num_rows, num_cols=self.num_cols)
        self.done = self.game.done()
        self.reset()

    def reset(self, board : None | list[list[int]] = None, seed=42):
        self.game = Game(num_colors=self.num_colors, num_rows=self.num_rows, num_cols=self.num_cols)
        if board:
            self.game.set_board(board)

        self.done = self.game.done()
        return self.get_observation()
    
    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        if self.done:
            raise RuntimeError("Episode done. Call reset()")
        row, col = divmod(action, self.num_cols)
        valid = self.game.move((row, col))
        reward = self.compute_reward(valid)
        self.done = self.game.done()
        return self.get_observation(), reward, self.done, {}
    
    def compute_reward(self, valid) -> float:
        return float(self.game.left)

    def get_observation(self) -> np.ndarray:
        return self._trainable_game(self.game.get_board())
    
    def _trainable_game(self, board: list[list[int]]) -> np.ndarray:
        board_np = np.array(board)
        obs = np.zeros((self.num_colors, self.num_rows, self.num_cols), dtype=np.float32)
        for color in range(self.num_colors):
            obs[color] = (board_np == color)
        return obs
    
        '''
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
        '''

    '''

    def _inverse_trainable_game(self, board: np.ndarray) -> list[list[int]]:

        # Initialize the board with all zeros
        new_board = [[0]*self.num_cols]*self.num_rows

        # Iterate over colors, rows, and columns
        for color in range(self.num_colors):
            for row in range(self.num_rows):
                for col in range(self.num_cols):
                    # If the value in the layers array is 1, set the corresponding position on the board to the current color
                    if board[color, row, col] == 1:
                        new_board[row][col] = color

        return new_board

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
        '''