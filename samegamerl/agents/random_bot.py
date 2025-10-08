"""
RandomBot implementation for SameGame benchmarking.

Selects random valid moves from available options, providing a baseline
for evaluating other agents' performance.
"""

import hashlib
import random
from samegamerl.agents.benchmark_bot_base import BenchmarkBotBase
from samegamerl.agents.bot_utils import find_valid_moves


class RandomBot(BenchmarkBotBase):
    """
    Bot that randomly selects from available valid moves.

    Provides a simple baseline for benchmarking other agents.
    Supports seeded random generation for reproducible testing.
    """

    name = "RandomBot"  # Class attribute - accessible without instantiation

    def __init__(self, seed: int = 42):
        """
        Initialize RandomBot with optional random seed.

        Args:
            seed: Random seed for reproducible behavior during testing
        """
        self.rng = random.Random(seed)

    def select_action(self, board: list[list[int]]) -> tuple[int, int] | None:
        """
        Randomly select from all valid moves on the board.
        Will select same random move for the same board.

        Args:
            board: Current game board state

        Returns:
            Random valid (row, col) position, or None if no moves available
        """
        valid_moves = find_valid_moves(board)

        if not valid_moves:
            return None

        board_str = str(board).encode('utf-8')
        stable_hash = int(hashlib.md5(board_str).hexdigest(), 16)
        self.rng.seed(stable_hash)

        return self.rng.choice(valid_moves)
