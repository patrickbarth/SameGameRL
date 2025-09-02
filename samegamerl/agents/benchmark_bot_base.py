"""
Base interface for benchmark bots.

Provides a focused interface for deterministic game-playing bots that don't
require learning, saving, or other RL-specific functionality.
"""

from abc import ABC, abstractmethod


class BenchmarkBotBase(ABC):
    """
    Abstract base class for benchmark bots that play SameGame deterministically.
    
    Unlike BaseAgent, this interface focuses solely on action selection
    without learning or persistence concerns.
    """

    @abstractmethod
    def select_action(self, board: list[list[int]]) -> tuple[int, int] | None:
        """
        Select the next action to take given the current board state.
        
        Args:
            board: 2D list representing the game board, where 0 = empty,
                   positive integers = colored tiles
        
        Returns:
            (row, col) tuple for the position to click, or None if no valid moves
        """
        pass