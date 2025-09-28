import random
from math import floor

import numpy as np

from samegamerl.game.game_config import GameConfig, GameFactory


class Game:

    def __init__(self, config: GameConfig = GameFactory.default(), seed: int | None = None):
        self.config = config
        self.neighbours = self.initialize_neighbours(self.config.num_rows, self.config.num_cols)
        self.board = self.create_board(seed=seed)
        self.left = self.config.num_rows * self.config.num_cols
        self.cols_left = self.config.num_cols

    def create_board(self, seed: int | None = None) -> list[list[int]]:
        """Create a new random board with optional seed for reproducibility."""
        if seed is not None:
            game_rng = random.Random(seed)
            randint_func = game_rng.randint
        else:
            randint_func = random.randint

        return [[randint_func(1, self.config.num_colors - 1)
                 for _ in range(self.config.num_cols)]
                 for _ in range(self.config.num_rows)]

    def get_board(self) -> list[list[int]]:
        """Return an immutable copy of the board to prevent external modifications"""
        return [row.copy() for row in self.board]

    def set_board(self, board: list[list[int]]):
        # Check for valid dimensions
        if len(board) != self.config.num_rows:
            raise ValueError(
                f"Board row count {len(board)} does not match expected {self.config.num_rows}"
            )
        if any(len(row) != self.config.num_cols for row in board):
            raise ValueError("All board rows must match expected number of columns.")

        # Check that all values are valid color indices
        empty_cells = 0
        for row in board:
            for val in row:
                if val == 0:
                    empty_cells += 1
                if not (0 <= val < self.config.num_colors):
                    raise ValueError(
                        f"Invalid color value {val}, must be in range 0 to {self.config.num_colors - 1}"
                    )

        # Set board and reset internal state
        self.board = board
        self.left = self.config.num_cols * self.config.num_rows - empty_cells

        self.sink()
        self.shrink()

    def get_nbhd_colors(self, row: int, col: int, board: list[list[int]]) -> list[int]:
        colors = []
        for neighbour in self.neighbours[row][col]:
            colors.append(board[neighbour[0]][neighbour[1]])
        return colors

    def move(self, tile: tuple[int, int]):
        new_neighbours = {tile}
        old_neighbours = {tile}
        color = self.board[tile[0]][tile[1]]
        first = True

        if color == 0:
            return self.board

        # finding all the adjacent neighbours with the same color
        while first or len(old_neighbours) != len(new_neighbours):
            if not first:
                old_neighbours = new_neighbours.copy()
            for neighbour in self.get_neighbours(old_neighbours):
                if self.board[neighbour[0]][neighbour[1]] == color:
                    new_neighbours.add(neighbour)
            if len(new_neighbours) == 1:
                return
            first = False

        # remove the tiles
        for neighbour in new_neighbours:
            self.left -= 1
            self.board[neighbour[0]][neighbour[1]] = 0

        self.sink()
        self.shrink()

        return self.board

    def movable(self, move):
        x = floor(move / self.config.num_rows)
        y = move % self.config.num_cols
        return (x, y)

    def shrink(self):
        for col in range(self.cols_left - 1):
            while self.board[self.config.num_rows - 1][col] == 0 and self.cols_left > col:
                for i in range(col, self.cols_left - 1):
                    self.shift(i)
                self.cols_left -= 1

    def shift(self, column):
        for row in range(self.config.num_rows):
            self.board[row][column] = self.board[row][column + 1]
            self.board[row][column + 1] = 0

    def sink(self):
        for col in range(0, self.config.num_cols):
            white = False
            color = False
            white_spaces = 0
            color_spaces = 0
            white_start = 0

            for row in range(self.config.num_rows - 1, -1, -1):

                # go up until you find the first white
                if self.board[row][col] == 0 and not color:
                    if not white:
                        white_start = row
                    white = True
                    white_spaces += 1

                # then count how much you have to go up until you find the next color
                if self.board[row][col] != 0 and white:
                    color = True
                    color_spaces += 1

                # then count how much you have to go up until you find the next white
                if (color and row == 0) or (self.board[row][col] == 0 and color):
                    color = False
                    self.move_block(col, white_start, white_spaces, color_spaces)
                    white_start = white_start - color_spaces
                    white_spaces += 1
                    color_spaces = 0

    def get_singles(self) -> int:
        single_counter = 0
        for row in range(self.config.num_rows):
            for column in range(self.config.num_cols):
                found = False
                for neighbour in self.neighbours[row][column]:
                    if (
                        self.board[neighbour[0]][neighbour[1]]
                        == self.board[row][column]
                    ):
                        found = True
                if not found:
                    single_counter += 1
        return single_counter

    def move_block(self, column, start, distance, block_size):
        block = []

        # get the block that needs to be moved and color them white
        for i in range(block_size):
            block.append(self.board[start - distance - i][column])

        for i in range(block_size):
            self.board[start - distance - i][column] = 0

        # color the old position all white
        for i in range(block_size):
            self.board[start - i][column] = block[i]

    def get_neighbours(self, tiles: set[tuple[int, int]]):
        neighbours = []
        for tile in tiles:
            for neighbour in self.neighbours[tile[0]][tile[1]]:
                if (neighbour not in neighbours) and (neighbour not in tiles):
                    neighbours.append(neighbour)

        return neighbours

    def done(self):
        return self.left == 0

    def initialize_neighbours(self, n, m):
        # Initialize a 2D list to store the neighbors for each square
        neighbor_list = []

        # Helper function to check if a square is within the field bounds
        def is_valid_square(row, col):
            return 0 <= row < n and 0 <= col < m

        # Iterate through each square
        for row in range(0, n):
            row_neighbors = []  # List to store neighbors for the current row
            for col in range(0, m):
                # Initialize a list to store the neighbors of the current square
                neighbors = []

                # Check and add neighbors
                for i in [-1, 1]:
                    if is_valid_square(row + i, col):
                        neighbors.append(((row + i), col))
                    if is_valid_square(row, col + i):
                        neighbors.append((row, (col + i)))

                row_neighbors.append(neighbors)

            neighbor_list.append(row_neighbors)
        return neighbor_list

    def trainable_game(self):
        layers = []
        for color in range(self.config.num_colors):
            layer = []
            for row in range(self.config.num_rows):
                srow = []
                for col in range(self.config.num_cols):
                    if self.board[row][col] == color:
                        srow.append(1)
                    else:
                        srow.append(0)
                layer.append(srow)
            layers.append(layer)
        return np.array(layers)
