"""
Tests for bot utility functions.

These are pure function tests that validate game analysis logic independent
of any specific bot implementation.
"""

import pytest
from samegamerl.agents.bot_utils import (
    find_valid_moves,
    calculate_group_size,
    simulate_move,
    count_singles_after_move,
)


class TestFindValidMoves:
    """Test finding valid moves (groups of size > 1) on game boards"""

    def test_empty_board(self):
        board = [[0, 0], [0, 0]]
        result = find_valid_moves(board)
        assert result == []

    def test_singles_only_board(self):
        board = [[1, 2, 1], [2, 1, 2], [1, 2, 1]]
        result = find_valid_moves(board)
        expected = []  # Should be empty - no valid moves
        assert result == expected

    def test_multiple_groups_available(self):
        board = [[1, 1, 0, 0], [1, 2, 2, 1], [2, 1, 1, 2], [1, 2, 1, 1]]
        expected_moves = [
            (0, 0),  # top-left position of the group of 1s in top-left corner
            (1, 1),  # top-left position of the group of 2s in second row
            (2, 1),  # top-left position of the group of 1s in bottom area
        ]  # List of (row, col) positions for valid groups (one per group)
        result = find_valid_moves(board)
        assert len(result) == len(expected_moves)
        for move in expected_moves:
            assert move in result

    def test_mixed_board_with_singles_and_groups(self):
        board = [[1, 2, 3], [2, 1, 1], [3, 2, 1], [1, 3, 2]]
        expected_count = 1  # Number of valid groups expected (only the pair of 1s at (1,1) and (1,2))
        result = find_valid_moves(board)
        assert len(result) == expected_count


class TestCalculateGroupSize:
    """Test calculating connected group sizes"""

    def test_single_tile_group(self):
        board = [[2, 2, 3], [3, 1, 3], [3, 2, 2]]
        row, col = 1, 1  # Position of single tile
        result = calculate_group_size(board, row, col)
        assert result == 1

    def test_small_connected_group(self):
        board = [[2, 2, 3], [3, 1, 1], [3, 3, 1]]
        row, col = 0, 0
        result = calculate_group_size(board, row, col)
        assert result == 2

    def test_large_connected_group(self):
        board = [[1, 1, 0, 3], [3, 1, 2, 2], [3, 1, 2, 1], [2, 1, 1, 1]]
        row, col = 2, 1
        expected_size = 8  # Expected group size
        result = calculate_group_size(board, row, col)
        assert result == expected_size

    def test_empty_cell(self):
        board = [[0, 0], [1, 1]]
        result = calculate_group_size(board, 0, 0)
        assert result == 0


class TestSimulateMove:
    """Test move simulation without modifying original board"""

    def test_simulate_preserves_original(self):
        original_board = [[2, 2, 0], [1, 1, 0], [1, 2, 2]]
        row, col = 2, 0

        simulated_board = simulate_move(original_board, row, col)

        # Original should be unchanged
        assert original_board != simulated_board
        assert original_board == [[2, 2, 0], [1, 1, 0], [1, 2, 2]]

    def test_simulate_small_group_removal(self):
        board = [[2, 2, 0], [3, 3, 4], [1, 1, 4]]
        row, col = 2, 0  # Position of group to remove
        expected_board = [
            [0, 0, 0],  # Expected board state after move
            [2, 2, 4],
            [3, 3, 4],
        ]

        result = simulate_move(board, row, col)
        assert result == expected_board

    def test_simulate_with_gravity_and_column_collapse(self):
        board = [[1, 1, 0, 3], [3, 1, 2, 2], [3, 1, 2, 1], [2, 1, 1, 1]]
        row, col = 0, 0
        expected_board = [
            [0, 0, 0, 0],  # Expected final state after physics
            [3, 0, 0, 0],
            [3, 2, 3, 0],
            [2, 2, 2, 0],
        ]

        result = simulate_move(board, row, col)
        assert result == expected_board


class TestCountSinglesAfterMove:
    """Test predicting singles count after hypothetical moves"""

    def test_move_reduces_singles(self):
        board = [[0, 3, 0, 3], [3, 2, 3, 2], [2, 1, 2, 3], [1, 1, 1, 1]]
        row, col = 3, 0  # Move that reduces singles
        expected_singles = 3  # Expected singles count after move

        result = count_singles_after_move(board, row, col)
        assert result == expected_singles

    def test_move_increases_singles(self):
        board = [[0, 2, 0, 3], [3, 3, 3, 2], [2, 1, 2, 3], [1, 1, 1, 1]]
        row, col = 3, 0  # Move that creates more singles
        expected_singles = 9  # Expected singles count after move

        result = count_singles_after_move(board, row, col)
        assert result == expected_singles

    def test_move_maintains_singles_count(self):
        board = [[0, 3, 0, 3], [3, 3, 3, 2], [2, 2, 2, 3], [1, 1, 1, 1]]
        row, col = 3, 0  # Neutral move regarding singles
        original_singles_count = 3

        result = count_singles_after_move(board, row, col)
        assert result == original_singles_count
