import pytest
import numpy as np
from samegamerl.game.game import Game
from samegamerl.game.game_config import GameConfig, GameFactory
from .conftest import TEST_BOARD_CONFIGS


class TestGameInitialization:
    """Test game initialization and basic properties"""

    def test_default_initialization(self):
        game = Game(GameFactory.default())
        assert game.num_rows == 8
        assert game.num_cols == 8
        assert game.num_colors == 3
        assert game.left == 64  # 8 * 8
        assert game.cols_left == 8
        assert not game.done()

    def test_custom_dimensions(self):
        config = GameConfig(num_rows=5, num_cols=6, num_colors=3)
        game = Game(config)
        assert game.num_rows == 5
        assert game.num_cols == 6
        assert game.num_colors == 3
        assert game.left == 30  # 5 * 6
        assert game.cols_left == 6

    def test_board_creation_bounds(self):
        config = GameConfig(num_rows=3, num_cols=3, num_colors=4)
        game = Game(config)
        board = game.get_board()
        assert len(board) == 3
        assert all(len(row) == 3 for row in board)
        # Colors should be 1 to num_colors-1 (0 is empty)
        for row in board:
            for cell in row:
                assert 1 <= cell < 4


class TestBoardManagement:
    """Test board manipulation and validation"""

    def test_set_valid_board(self):
        game = Game(GameConfig(num_rows=2, num_cols=2, num_colors=3))
        test_board = TEST_BOARD_CONFIGS["checkerboard_2x2"]
        game.set_board(test_board)
        assert game.get_board() == test_board
        assert game.left == 4

    def test_set_board_invalid_dimensions(self):
        game = Game(GameConfig(num_rows=2, num_cols=2, num_colors=3))

        with pytest.raises(ValueError):
            game.set_board([[1, 2]])

        with pytest.raises(ValueError):
            game.set_board([[1, 2, 3], [1, 2]])

    def test_set_board_invalid_colors(self):
        game = Game(GameConfig(num_rows=2, num_cols=2, num_colors=3))

        with pytest.raises(ValueError):
            game.set_board([[1, 2], [3, 1]])

        with pytest.raises(ValueError):
            game.set_board([[1, 2], [-1, 1]])

    def test_set_board_with_empty_cells(self):
        game = Game(GameConfig(num_rows=3, num_cols=2, num_colors=3))
        board = [[1, 0], [2, 1], [0, 0]]
        game.set_board(board)
        assert game.left == 3

        expected_board = [[0, 0], [1, 0], [2, 1]]
        assert game.get_board() == expected_board


class TestMoveMechanics:
    """Test core game move mechanics"""

    def test_move_on_empty_cell(self):
        game = Game(GameConfig(num_rows=2, num_cols=2, num_colors=3))
        game.set_board(TEST_BOARD_CONFIGS["empty_2x2"])
        initial_board = game.get_board().copy()

        game.move((0, 1))
        assert game.get_board() == initial_board

    def test_move_single_tile(self):
        game = Game(GameConfig(num_rows=3, num_cols=3, num_colors=4))
        board = TEST_BOARD_CONFIGS["singles_only"]
        game.set_board(board)
        initial_board = game.get_board().copy()

        game.move((1, 1))
        assert game.get_board() == initial_board

    def test_move_connected_group(self):
        game = Game(GameConfig(num_rows=3, num_cols=3, num_colors=3))
        # Create 2x2 group of color 1 in top-left
        board = [[1, 1, 2], [1, 1, 2], [2, 2, 2]]
        game.set_board(board)
        initial_left = game.left

        game.move((0, 0))

        assert game.left == initial_left - 4

        new_board = game.get_board()
        expected_board = [[0, 0, 2], [0, 0, 2], [2, 2, 2]]
        assert new_board == expected_board

    def test_move_removes_correct_connected_component(self):
        game = Game(GameConfig(num_rows=2, num_cols=4, num_colors=3))
        # Two separate groups of color 1
        board = [[1, 1, 2, 1], [2, 2, 2, 1]]
        game.set_board(board)

        game.move((0, 0))
        new_board = game.get_board()
        expected_board = [[0, 0, 2, 1], [2, 2, 2, 1]]
        assert new_board == expected_board


class TestPhysics:
    """Test tile sinking and column shrinking"""

    def test_sink_mechanics(self):
        game = Game(GameConfig(num_rows=3, num_cols=2, num_colors=3))
        # Create floating tiles
        board = [[1, 2], [0, 0], [2, 1]]
        game.set_board(board)

        # After setting board, physics should apply
        final_board = game.get_board()
        expected_baord = [[0, 0], [1, 2], [2, 1]]

        assert final_board == expected_baord

    def test_column_shrinking(self):
        game = Game(GameConfig(num_rows=2, num_cols=3, num_colors=3))
        # Create board where middle column is empty
        board = [[1, 0, 2], [1, 0, 2]]
        game.set_board(board)

        final_board = game.get_board()
        expected_board = [[1, 2, 0], [1, 2, 0]]

        assert final_board == expected_board

    def test_complete_physics_after_move(self):
        game = Game(GameConfig(num_rows=3, num_cols=3, num_colors=3))
        board = [[1, 2, 2], [1, 1, 2], [2, 2, 2]]
        game.set_board(board)

        game.move((0, 0))

        expected_board = [[0, 0, 2], [0, 2, 2], [2, 2, 2]]
        actual_board = game.get_board()

        assert actual_board == expected_board


class TestGameState:
    """Test game state queries and utilities"""

    def test_get_singles_empty_board(self):
        game = Game(GameConfig(num_rows=2, num_cols=2, num_colors=3))
        game.set_board(TEST_BOARD_CONFIGS["empty_2x2"])
        assert game.get_singles() == 0

    def test_get_singles_all_different(self):
        game = Game(GameConfig(num_rows=2, num_cols=2, num_colors=5))
        game.set_board(TEST_BOARD_CONFIGS["checkerboard_2x2"])
        assert game.get_singles() == 4

    def test_get_singles_mixed(self):
        game = Game(GameConfig(num_rows=3, num_cols=3, num_colors=3))
        board = TEST_BOARD_CONFIGS["mixed_3x3"]
        game.set_board(board)
        singles = game.get_singles()
        assert singles == 1

    def test_done_condition(self):
        game = Game(GameConfig(num_rows=2, num_cols=2, num_colors=3))

        game.set_board(TEST_BOARD_CONFIGS["checkerboard_2x2"])
        assert not game.done()

        game.set_board(TEST_BOARD_CONFIGS["empty_2x2"])
        assert game.done()

        game.set_board(TEST_BOARD_CONFIGS["single_color_2x2"])
        assert not game.done()

        game.move((0, 0))
        assert game.done()

    def test_left_count_accuracy(self):
        game = Game(GameConfig(num_rows=3, num_cols=3, num_colors=3))
        board = [[1, 0, 2], [0, 1, 0], [2, 0, 1]]
        game.set_board(board)

        assert game.left == 5


class TestNeighborUtilities:
    """Test neighbor detection and utilities"""

    def test_initialize_neighbours(self):
        game = Game(GameConfig(num_rows=3, num_cols=3, num_colors=3))
        neighbors = game.neighbours

        assert len(neighbors[0][0]) == 2
        assert (0, 1) in neighbors[0][0]
        assert (1, 0) in neighbors[0][0]

        assert len(neighbors[0][1]) == 3
        assert (0, 0) in neighbors[0][1]
        assert (0, 2) in neighbors[0][1]
        assert (1, 1) in neighbors[0][1]

        assert len(neighbors[1][1]) == 4
        assert (0, 1) in neighbors[1][1]
        assert (1, 0) in neighbors[1][1]
        assert (1, 2) in neighbors[1][1]
        assert (2, 1) in neighbors[1][1]

    def test_get_neighbors_from_set(self):
        game = Game(GameConfig(num_rows=3, num_cols=3, num_colors=3))
        tiles = {(1, 1)}
        neighbors = game.get_neighbours(tiles)

        expected_neighbors = [(0, 1), (1, 0), (1, 2), (2, 1)]
        assert len(neighbors) == 4
        for neighbor in expected_neighbors:
            assert neighbor in neighbors

    def test_get_neighbors_excludes_input_tiles(self):
        game = Game(GameConfig(num_rows=3, num_cols=3, num_colors=3))
        tiles = {(1, 1), (1, 2)}
        neighbors = game.get_neighbours(tiles)

        assert (1, 1) not in neighbors
        assert (1, 2) not in neighbors

        assert (0, 1) in neighbors
        assert (1, 0) in neighbors
        assert (2, 1) in neighbors
        assert (2, 2) in neighbors


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_minimum_board_size(self):
        game = Game(GameConfig(num_rows=1, num_cols=1, num_colors=2))
        assert game.left == 1
        board = game.get_board()
        assert len(board) == 1
        assert len(board[0]) == 1

    def test_single_color_board(self):
        game = Game(GameConfig(num_rows=2, num_cols=2, num_colors=2))
        board = TEST_BOARD_CONFIGS["single_color_2x2"]
        game.set_board(board)

        game.move((0, 0))
        assert game.done()
        assert game.left == 0

    def test_move_on_boundary_positions(self):
        game = Game(GameConfig(num_rows=3, num_cols=3, num_colors=3))
        board = [[1, 1, 1], [1, 2, 1], [1, 1, 1]]
        game.set_board(board)

        game.move((0, 0))
        assert game.left == 1

        actual_board = game.get_board()
        expected_board = [[0, 0, 0], [0, 0, 0], [2, 0, 0]]
        assert actual_board == expected_board

    def test_complex_connected_component(self):
        game = Game(GameConfig(num_rows=4, num_cols=4, num_colors=4))
        board = TEST_BOARD_CONFIGS["complex_4x4"]
        game.set_board(board)

        game.move((1, 1))

        actual_board = game.get_board()
        expected_board = [[0, 0, 0, 3], [1, 0, 0, 3], [1, 1, 3, 3], [3, 1, 1, 2]]
        assert actual_board == expected_board


class TestDataStructureIntegrity:
    """Test that internal data structures remain consistent"""

    def test_board_immutability_through_get_board(self):
        """Test that get_board() returns an immutable copy that cannot affect game state"""
        game = Game(GameConfig(num_rows=2, num_cols=2, num_colors=3))
        game.set_board(TEST_BOARD_CONFIGS["checkerboard_2x2"])
        original_value = game.get_board()[0][0]

        # Get board and try to modify it
        returned_board = game.get_board()
        returned_board[0][0] = 999

        # Game's internal state should be unchanged
        current_board = game.get_board()
        assert current_board[0][0] == original_value

    def test_left_count_consistency_after_operations(self):
        game = Game(GameConfig(num_rows=3, num_cols=3, num_colors=3))
        board = [[1, 1, 2], [1, 2, 2], [2, 2, 1]]
        game.set_board(board)

        initial_left = game.left
        game.move((0, 0))
        intermediate_left = game.left
        game.move((1, 1))
        final_left = game.left

        assert initial_left == 9
        assert intermediate_left == 6
        assert final_left == 1

    def test_cols_left_consistency(self):
        game = Game(GameConfig(num_rows=2, num_cols=4, num_colors=3))
        board = [[1, 0, 0, 2], [1, 0, 0, 2]]
        game.set_board(board)

        final_board = game.get_board()

        non_empty_cols = 0
        for c in range(game.num_cols):
            if any(final_board[r][c] != 0 for r in range(game.num_rows)):
                non_empty_cols += 1

        assert game.cols_left == 2
