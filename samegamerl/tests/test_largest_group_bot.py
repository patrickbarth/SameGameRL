"""
Tests for LargestGroupBot behavior.

These tests focus on validating that LargestGroupBot consistently selects
the largest available group from the board.
"""

import pytest
from samegamerl.agents.largest_group_bot import LargestGroupBot


class TestLargestGroupBotInitialization:
    """Test LargestGroupBot creation and basic properties"""

    def test_bot_creation(self):
        bot = LargestGroupBot()
        assert bot is not None
        assert hasattr(bot, "select_action")


class TestLargestGroupBotActionSelection:
    """Test LargestGroupBot's greedy largest-group selection"""

    def test_clear_largest_group_choice(self):
        """When one group is clearly largest, bot must choose it"""
        bot = LargestGroupBot()
        board = [[0, 0, 0, 4], [0, 0, 3, 4], [0, 2, 3, 4], [1, 2, 3, 4]]
        largest_group_position = (0, 3)

        action = bot.select_action(board)
        assert action == largest_group_position

    def test_multiple_groups_same_max_size(self):
        """When multiple groups tie for largest, bot picks one of them"""
        bot = LargestGroupBot()
        board = [[0, 0, 0, 0], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
        expected_move = (1, 0)  # List of (row, col) for tied groups

        action = bot.select_action(board)
        assert action == expected_move

    def test_single_valid_group(self):
        """When only one group exists, bot must choose it regardless of size"""
        bot = LargestGroupBot()
        board = [[1, 3, 2, 3], [1, 2, 3, 4]]
        only_group_position = (0, 0)

        action = bot.select_action(board)
        assert action == only_group_position

    def test_no_valid_groups(self):
        """When no valid groups exist, bot returns None"""
        bot = LargestGroupBot()
        board = [[1, 2, 3], [2, 3, 1], [3, 1, 2]]

        action = bot.select_action(board)
        assert action is None

    def test_empty_board(self):
        """Empty board should return None"""
        bot = LargestGroupBot()
        board = [[0, 0], [0, 0]]

        action = bot.select_action(board)
        assert action is None

    def test_complex_board(self):
        """Complex board with two large groups"""
        bot = LargestGroupBot()
        board = [
            [1, 1, 1, 1],
            [1, 2, 2, 2],
            [1, 2, 1, 2],
            [1, 2, 1, 2],
            [1, 2, 2, 2],
            [1, 1, 1, 1],
        ]
        expected_move = (0, 0)

        action = bot.select_action(board)
        assert action == expected_move


class TestLargestGroupBotEdgeCases:
    """Test LargestGroupBot behavior in edge cases"""

    def test_single_cell_board(self):
        bot = LargestGroupBot()
        board = [[1]]
        expected_result = None

        action = bot.select_action(board)
        assert action == expected_result

    def test_all_same_color_board(self):
        """Entire board is one massive group"""
        bot = LargestGroupBot()
        board = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        expected_position = (0, 0)

        action = bot.select_action(board)
        assert action == expected_position
