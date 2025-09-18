"""
LargestGroupBot implementation for SameGame benchmarking.

Greedily selects the largest available group on each turn, providing
a simple heuristic-based benchmark for evaluating other agents.
"""

from samegamerl.agents.benchmark_bot_base import BenchmarkBotBase
from samegamerl.agents.bot_utils import find_valid_moves, calculate_group_size


class LargestGroupBot(BenchmarkBotBase):
    """
    Bot that always selects the largest available group.
    
    Provides a greedy baseline that prioritizes immediate large removals
    over strategic considerations.
    """
    
    name = "LargestGroupBot"  # Class attribute - accessible without instantiation

    def select_action(self, board: list[list[int]]) -> tuple[int, int] | None:
        """
        Select the position of the largest group on the board.
        
        When multiple groups tie for largest size, selects the first one
        encountered (deterministic tie-breaking by row-major order).
        
        Args:
            board: Current game board state
            
        Returns:
            (row, col) position of largest group, or None if no moves available
        """
        valid_moves = find_valid_moves(board)
        
        if not valid_moves:
            return None
        
        best_move = None
        best_size = 0
        
        for move in valid_moves:
            row, col = move
            group_size = calculate_group_size(board, row, col)
            
            if group_size > best_size:
                best_size = group_size
                best_move = move
        
        return best_move