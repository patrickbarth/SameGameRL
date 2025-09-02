"""
Game analysis utility functions for benchmark bots.

Pure functions for analyzing SameGame board states without dependency
on specific bot implementations or inheritance hierarchies.
"""

from copy import deepcopy
from samegamerl.game.game import Game
from samegamerl.game.game_config import GameConfig


def find_valid_moves(board: list[list[int]]) -> list[tuple[int, int]]:
    """
    Find all valid moves (groups of size > 1) on the board.
    
    For each connected group of same-colored tiles with size > 1, returns the
    top-left-most position (smallest row, then smallest column) within that group.
    
    Returns list of (row, col) positions where clicking would remove a group.
    """
    if not board or not board[0]:
        return []
    
    num_rows, num_cols = len(board), len(board[0])
    visited = [[False] * num_cols for _ in range(num_rows)]
    valid_moves = []
    
    for row in range(num_rows):
        for col in range(num_cols):
            if not visited[row][col] and board[row][col] != 0:
                group_size = _explore_group(board, row, col, visited)
                if group_size > 1:
                    valid_moves.append((row, col))
    
    return valid_moves


def calculate_group_size(board: list[list[int]], row: int, col: int) -> int:
    """
    Calculate the size of the connected group at the given position.
    
    Returns 0 for empty cells, 1+ for connected same-colored groups.
    """
    if not board or row < 0 or row >= len(board) or col < 0 or col >= len(board[0]):
        return 0
    
    if board[row][col] == 0:
        return 0
    
    num_rows, num_cols = len(board), len(board[0])
    visited = [[False] * num_cols for _ in range(num_rows)]
    
    return _explore_group(board, row, col, visited)


def simulate_move(board: list[list[int]], row: int, col: int) -> list[list[int]]:
    """
    Simulate a move without modifying the original board.
    
    Returns the board state that would result from clicking at (row, col).
    """
    if not board or row < 0 or row >= len(board) or col < 0 or col >= len(board[0]):
        return deepcopy(board)
    
    if board[row][col] == 0:
        return deepcopy(board)
    
    # Create a Game instance to leverage existing move mechanics
    num_rows, num_cols = len(board), len(board[0])
    max_color = max(max(row) for row in board if row)
    # Ensure num_colors is at least 3 to avoid randint(1,1) error in Game constructor
    num_colors = max(max_color + 1, 3)
    config = GameConfig(num_rows=num_rows, num_cols=num_cols, num_colors=num_colors)
    
    game = Game(config)
    game.set_board(deepcopy(board))
    game.move((row, col))
    
    return game.get_board()


def count_singles_after_move(board: list[list[int]], row: int, col: int) -> int:
    """
    Count the number of single isolated tiles after making the specified move.
    
    Returns the count of tiles that would have no same-colored neighbors.
    """
    simulated_board = simulate_move(board, row, col)
    
    # Create Game instance to use existing get_singles method
    if not simulated_board:
        return 0
        
    num_rows, num_cols = len(simulated_board), len(simulated_board[0])
    max_color = max((max(row) for row in simulated_board if row), default=0)
    # Ensure num_colors is at least 3 to avoid randint(1,1) error in Game constructor
    num_colors = max(max_color + 1, 3)
    config = GameConfig(num_rows=num_rows, num_cols=num_cols, num_colors=num_colors)
    
    game = Game(config)
    game.set_board(simulated_board)
    
    return game.get_singles()


def _explore_group(board: list[list[int]], row: int, col: int, visited: list[list[bool]]) -> int:
    """
    Recursively explore connected same-colored tiles using DFS.
    
    Marks visited cells and returns the total group size.
    """
    num_rows, num_cols = len(board), len(board[0])
    
    if (row < 0 or row >= num_rows or col < 0 or col >= num_cols or 
        visited[row][col] or board[row][col] == 0):
        return 0
    
    target_color = board[row][col]
    visited[row][col] = True
    size = 1
    
    # Explore all 4 directions
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    for dr, dc in directions:
        new_row, new_col = row + dr, col + dc
        if (0 <= new_row < num_rows and 0 <= new_col < num_cols and 
            not visited[new_row][new_col] and board[new_row][new_col] == target_color):
            size += _explore_group(board, new_row, new_col, visited)
    
    return size