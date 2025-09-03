"""
Game analysis utility functions for benchmark bots.

Pure functions for analyzing SameGame board states without dependency
on specific bot implementations or inheritance hierarchies.
"""



def find_valid_moves(board: list[list[int]]) -> list[tuple[int, int]]:
    """
    Find all valid moves (groups of size > 1) on the board using optimized flood-fill.
    
    For each connected group of same-colored tiles with size > 1, returns the
    top-left-most position (smallest row, then smallest column) within that group.
    
    Returns list of (row, col) positions where clicking would remove a group.
    """
    if not board or not board[0]:
        return []
    
    num_rows, num_cols = len(board), len(board[0])
    visited = set()
    valid_moves = []
    
    for row in range(num_rows):
        for col in range(num_cols):
            if (row, col) in visited or board[row][col] == 0:
                continue
                
            # Flood-fill to find entire group using stack
            group_cells = []
            stack = [(row, col)]
            color = board[row][col]
            
            while stack:
                r, c = stack.pop()
                if (r, c) in visited:
                    continue
                    
                visited.add((r, c))
                group_cells.append((r, c))
                
                # Check 4-directional neighbors
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < num_rows and 0 <= nc < num_cols 
                        and (nr, nc) not in visited 
                        and board[nr][nc] == color):
                        stack.append((nr, nc))
            
            # Only add if group size > 1, return top-left canonical position
            if len(group_cells) > 1:
                top_left = min(group_cells)  # Gets (min_row, min_col)
                valid_moves.append(top_left)
    
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
    Simulate a move without modifying the original board or creating Game objects.
    
    Returns the board state that would result from clicking at (row, col).
    Optimized implementation that replicates Game.move() logic directly.
    """
    if not board or row < 0 or row >= len(board) or col < 0 or col >= len(board[0]):
        return [row[:] for row in board]
    
    if board[row][col] == 0:
        return [row[:] for row in board]
    
    # Create working copy using list comprehension (faster than deepcopy)
    new_board = [row[:] for row in board]
    num_rows, num_cols = len(board), len(board[0])
    
    # Find all connected tiles of the same color using flood-fill
    target_color = board[row][col]
    connected_tiles = set()
    stack = [(row, col)]
    
    while stack:
        r, c = stack.pop()
        if (r, c) in connected_tiles:
            continue
        connected_tiles.add((r, c))
        
        # Check 4-directional neighbors
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if (0 <= nr < num_rows and 0 <= nc < num_cols 
                and (nr, nc) not in connected_tiles 
                and board[nr][nc] == target_color):
                stack.append((nr, nc))
    
    # Only proceed if group size > 1 (valid move)
    if len(connected_tiles) <= 1:
        return new_board
    
    # Remove the connected tiles
    for r, c in connected_tiles:
        new_board[r][c] = 0
    
    # Apply gravity (sink) - move non-zero tiles down
    _apply_gravity(new_board, num_rows, num_cols)
    
    # Apply shrinking - remove empty columns from right
    _apply_shrinking(new_board, num_rows, num_cols)
    
    return new_board


def _apply_gravity(board: list[list[int]], num_rows: int, num_cols: int) -> None:
    """Apply gravity to make tiles fall down, modifying board in-place."""
    for col in range(num_cols):
        # Collect non-zero tiles from bottom to top
        non_zero_tiles = []
        for row in range(num_rows - 1, -1, -1):
            if board[row][col] != 0:
                non_zero_tiles.append(board[row][col])
        
        # Clear the column
        for row in range(num_rows):
            board[row][col] = 0
        
        # Place non-zero tiles at bottom
        for i, tile in enumerate(non_zero_tiles):
            board[num_rows - 1 - i][col] = tile


def _apply_shrinking(board: list[list[int]], num_rows: int, num_cols: int) -> None:
    """Remove empty columns by shifting left, modifying board in-place."""
    write_col = 0
    
    for read_col in range(num_cols):
        # Check if column has any non-zero tiles
        has_tiles = any(board[row][read_col] != 0 for row in range(num_rows))
        
        if has_tiles:
            if write_col != read_col:
                # Move column content to the left
                for row in range(num_rows):
                    board[row][write_col] = board[row][read_col]
                    board[row][read_col] = 0
            write_col += 1
        elif write_col != read_col:
            # Clear the column if it's being skipped
            for row in range(num_rows):
                board[row][read_col] = 0


def count_singles(board: list[list[int]]) -> int:
    """
    Count the number of single isolated tiles using direct neighbor checking.
    
    Returns the count of tiles that have no same-colored neighbors.
    """
    if not board or not board[0]:
        return 0
    
    num_rows, num_cols = len(board), len(board[0])
    singles = 0
    
    for row in range(num_rows):
        for col in range(num_cols):
            if board[row][col] == 0:
                continue
                
            # Check if this cell is isolated (no same-color neighbors)
            color = board[row][col]
            has_neighbor = False
            
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = row + dr, col + dc
                if (0 <= nr < num_rows and 0 <= nc < num_cols 
                    and board[nr][nc] == color):
                    has_neighbor = True
                    break
            
            if not has_neighbor:
                singles += 1
                
    return singles


def count_singles_after_move(board: list[list[int]], row: int, col: int) -> int:
    """
    Count the number of single isolated tiles after making the specified move.
    
    Returns the count of tiles that would have no same-colored neighbors.
    """
    simulated_board = simulate_move(board, row, col)
    return count_singles(simulated_board)


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