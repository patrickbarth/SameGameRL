import random
from math import ceil
from math import floor

import numpy as np

from flightgame.game.game_params import NUM_COLS, NUM_ROWS, NUM_COLORS


class Game:

    def __init__(
        self, num_rows=NUM_ROWS, num_cols=NUM_COLS, num_colors=NUM_COLORS
    ):  # , screen):
        # self.screen = screen
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_colors = num_colors
        self.neighbours = self.initialize_neighbours(self.num_rows, self.num_cols)
        self.deterministic = False  # for debugging
        self.board = self.create_board(
            [[-1 for _ in range(self.num_cols)] for _ in range(self.num_rows)]
        )
        self.left = self.num_rows * self.num_cols
        self.cols_left = self.num_cols

    def create_board(self, board: list[list[int]]) -> list[list[int]]:
        for row in range(len(board)):
            for col in range(len(board[row])):
                board[row][col] = random.randint(1, self.num_colors - 1)

        if self.deterministic:
            board = [
                [1, 3, 3, 1, 3, 1, 2, 2],
                [3, 1, 1, 2, 1, 3, 1, 3],
                [2, 2, 1, 1, 3, 3, 2, 2],
                [3, 1, 1, 2, 3, 2, 3, 2],
                [3, 3, 2, 1, 1, 3, 3, 2],
                [3, 2, 3, 1, 2, 2, 1, 3],
                [3, 1, 3, 1, 2, 2, 2, 1],
                [2, 2, 3, 3, 2, 3, 1, 3],
            ]
        return board

    def get_board(self) -> list[list[int]]:
        return self.board

    def set_board(self, board: list[list[int]]):
        # Check for valid dimensions
        if len(board) != self.num_rows:
            raise ValueError(
                f"Board row count {len(board)} does not match expected {self.num_rows}"
            )
        if any(len(row) != self.num_cols for row in board):
            raise ValueError("All board rows must match expected number of columns.")

        # Check that all values are valid color indices
        empty_cells = 0
        for row in board:
            for val in row:
                if val == 0:
                    empty_cells += 1
                if not (0 <= val < self.num_colors):
                    raise ValueError(
                        f"Invalid color value {val}, must be in range 0 to {self.num_colors - 1}"
                    )

        # Set board and reset internal state
        self.board = board
        self.left = self.num_cols * self.num_rows - empty_cells

        self.sink()
        self.shrink()

    def get_nbhd_colors(self, row: int, col: int, board: list[list[int]]) -> list[int]:
        colors = []
        for neighbour in self.neighbours[row][col]:
            colors.append(board[neighbour[0]][neighbour[1]])
        return colors

    def move(self, tile: tuple[int, int]):
        old_left = self.left
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
        x = floor(move / self.num_rows)
        y = move % self.num_cols
        return (x, y)

    def shrink(self):
        for col in range(self.cols_left - 1):
            while self.board[self.num_rows - 1][col] == 0 and self.cols_left > col:
                for i in range(col, self.cols_left - 1):
                    self.shift(i)
                self.cols_left -= 1

    def shift(self, column):
        for row in range(self.num_rows):
            self.board[row][column] = self.board[row][column + 1]
            self.board[row][column + 1] = 0

    def sink(self):
        for col in range(0, self.num_cols):
            white = False
            color = False
            white_spaces = 0
            color_spaces = 0
            white_start = 0

            for row in range(self.num_rows - 1, -1, -1):

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
        for row in range(self.num_rows):
            for column in range(self.num_cols):
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

    def trainable_game_helper(self, board):
        layers = []
        for color in range(NUM_COLORS):
            layer = []
            for row in range(NUM_ROWS):
                srow = []
                for col in range(NUM_COLS):
                    if board[row][col] == color:
                        srow.append(1)
                    else:
                        srow.append(0)
                layer.append(srow)
            layers.append(layer)
        return np.array(layers)

    def inverse_trainable_game(self, board):

        # Initialize the board with all zeros
        new_board = np.zeros((NUM_ROWS, NUM_COLS), dtype=int)

        # Iterate over colors, rows, and columns
        for color in range(NUM_COLORS):
            for row in range(NUM_ROWS):
                for col in range(NUM_COLS):
                    # If the value in the layers array is 1, set the corresponding position on the board to the current color
                    if board[color, row, col] == 1:
                        new_board[row, col] = color

        return new_board

    def reward_all(self, board):
        board = self.inverse_trainable_game(board)
        rewards = []
        for row in range(NUM_ROWS):
            for col in range(NUM_COLS):
                rewards.append(self.reward(row, col, board))
        return np.array(rewards)

    def reward(self, row, col, board):
        tile = (row, col)
        old_left = self.left
        new_left = self.left-1
        # check which tiles will now belong to the current player
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

    """

    def rl_move_normal(self, move):
        tile = self.movable(move)
        old_left = self.left
        # check which tiles will now belong to the current player
        new_neighbours = [tile]
        old_neighbours = [tile]
        color = self.board[tile[0]][tile[1]]
        first = True

        if color == 0:
            return self.trainable_game(), -0.2, False

        # finding all the adjacent neighbours with the same color
        while first or len(old_neighbours) != len(new_neighbours):
            if not first:
                old_neighbours = new_neighbours.copy()
            for neighbour in self.get_neighbours(old_neighbours):
                if self.board[neighbour[0]][neighbour[1]] == color:
                    new_neighbours.append(neighbour)
            if len(new_neighbours) == 1:
                return self.trainable_game(), -0.1, False
            first = False

        # remove the tiles
        for neighbour in new_neighbours:
            self.left -= 1
            self.board[neighbour[0]][neighbour[1]] = 0

        self.sink()
        self.shrink()

        reward = (old_left - self.left) / 10
        if self.done():
            reward = 1

        return self.trainable_game(), reward, self.done()

    def rl_move(self, move):
        tile = self.movable(move)
        old_left = self.left
        # check which tiles will now belong to the current player
        new_neighbours = [tile]
        old_neighbours = [tile]
        color = self.board[tile[0]][tile[1]]
        first = True

        if color == 0:
            return self.trainable_game(), 0, False

        # finding all the adjacent neighbours with the same color
        while first or len(old_neighbours) != len(new_neighbours):
            if not first:
                old_neighbours = new_neighbours.copy()
            for neighbour in self.get_neighbours(old_neighbours):
                if self.board[neighbour[0]][neighbour[1]] == color:
                    new_neighbours.append(neighbour)
            if len(new_neighbours) == 1:
                return self.trainable_game(), 0, False
            first = False

        # remove the tiles
        for neighbour in new_neighbours:
            self.left -= 1
            self.board[neighbour[0]][neighbour[1]] = 0

        self.sink()
        self.shrink()

        singles = self.get_singles()

        if singles < 1:
            reward = 1
        else:
            reward = 1 / singles

        return self.trainable_game(), reward, self.done()



    def rl_move_left(self, move):
        tile = self.movable(move)
        old_left = self.left
        # check which tiles will now belong to the current player
        new_neighbours = [tile]
        old_neighbours = [tile]
        color = self.board[tile[0]][tile[1]]
        first = True

        if color == 0:
            return self.trainable_game(), 0, False

        # finding all the adjacent neighbours with the same color
        while first or len(old_neighbours) != len(new_neighbours):
            if not first:
                old_neighbours = new_neighbours.copy()
            for neighbour in self.get_neighbours(old_neighbours):
                if self.board[neighbour[0]][neighbour[1]] == color:
                    new_neighbours.append(neighbour)
            if len(new_neighbours) == 1:
                return self.trainable_game(), 0, False
            first = False

        # remove the tiles
        for neighbour in new_neighbours:
            self.left -= 1
            self.board[neighbour[0]][neighbour[1]] = 0

        self.sink()
        self.shrink()

        reward = 1/(2**((self.left*7)/(self.num_cols*self.num_rows)))

        return self.trainable_game(), reward, self.done()
    """
