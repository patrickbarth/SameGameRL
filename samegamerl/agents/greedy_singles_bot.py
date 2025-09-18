"""
GreedySinglesBot implementation for SameGame benchmarking.

Selects moves that minimize the number of isolated single tiles on the board,
providing a strategic benchmark focused on board connectivity.
"""

from samegamerl.agents.benchmark_bot_base import BenchmarkBotBase
from samegamerl.agents.bot_utils import (
    find_valid_moves,
    count_singles_after_move,
)


class GreedySinglesBot(BenchmarkBotBase):
    """
    Bot that greedily minimizes the number of single isolated tiles.

    Evaluates each possible move by predicting the resulting singles count
    and selects the move that produces the fewest isolated tiles.
    """

    @property
    def name(self) -> str:
        """Return the display name of this bot"""
        return "GreedySinglesBot"

    def select_action(self, board: list[list[int]]) -> tuple[int, int] | None:
        """
        Select the move that results in the fewest single isolated tiles.

        Args:
            board: Current game board state

        Returns:
            (row, col) position that minimizes singles count, or None if no moves
        """
        valid_moves = find_valid_moves(board)

        if not valid_moves:
            return None

        best_moves = []
        best_singles_count = float("inf")

        # Evaluate each possible move
        for move in valid_moves:
            row, col = move
            singles_after = count_singles_after_move(board, row, col)

            if singles_after < best_singles_count:
                best_singles_count = singles_after
                best_moves = [move]
            elif singles_after == best_singles_count:
                best_moves.append(move)

        if len(best_moves) == 1:
            return best_moves[0]

        return self._break_ties(board, best_moves)

    def _break_ties(
        self, board: list[list[int]], tied_moves: list[tuple[int, int]]
    ) -> tuple[int, int]:
        """
        Break ties when multiple moves have the same singles reduction.

        Currently uses deterministic group most to the top, left, because this will affect the tiles before less.
        """

        return min(tied_moves)  # Deterministic: pick first by row-major order
