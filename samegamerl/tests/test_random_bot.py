"""
Tests for RandomBot behavior.

These tests focus on validating that RandomBot selects moves appropriately
from available options, rather than testing specific game outcome predictions.
"""

import pytest
from samegamerl.agents.random_bot import RandomBot


class TestRandomBotInitialization:
    """Test RandomBot creation and basic properties"""

    def test_bot_creation(self):
        bot = RandomBot()
        assert bot is not None
        assert hasattr(bot, "select_action")

    def test_bot_with_custom_seed(self):
        bot1 = RandomBot(seed=42)
        bot2 = RandomBot(seed=42)

        board = [[2, 1, 3], [2, 1, 3], [3, 3, 1]]

        action1 = bot1.select_action(board)
        action2 = bot2.select_action(board)
        assert action1 == action2


class TestRandomBotActionSelection:
    """Test RandomBot's move selection logic"""

    def test_single_valid_move_scenario(self):
        """When only one move available, bot must choose it"""
        bot = RandomBot()

        board = [[3, 2, 1], [2, 3, 1], [3, 2, 3]]
        expected_move = (0, 2)

        action = bot.select_action(board)
        assert action == expected_move

    def test_multiple_valid_moves_scenario(self):
        """When multiple moves available, bot picks one of them"""
        bot = RandomBot()

        board = [[3, 2, 1], [3, 2, 1], [3, 2, 3]]
        valid_moves = [(0, 0), (0, 1), (0, 2)]

        action = bot.select_action(board)
        assert action in valid_moves

    def test_no_valid_moves_scenario(self):
        """When no moves available, bot returns None"""
        bot = RandomBot()

        board = [[3, 2, 1], [2, 1, 3], [1, 3, 2]]

        action = bot.select_action(board)
        assert action is None

    def test_empty_board_scenario(self):
        """Empty board should return None"""
        bot = RandomBot()
        board = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

        action = bot.select_action(board)
        assert action is None


class TestRandomBotDistribution:
    """Test that RandomBot explores different moves over multiple calls"""

    def test_explores_different_moves(self):
        """Over many trials, bot should pick different valid moves"""
        bot = RandomBot()
        board = [[3, 2, 1], [3, 2, 1], [3, 2, 3]]
        valid_moves = [(0, 0), (0, 1), (0, 2)]

        selected_moves = set()
        for _ in range(50):  # Sample many times
            action = bot.select_action(board)
            if action:
                selected_moves.add(action)

        # Should explore multiple options
        min_explored = 3
        assert len(selected_moves) == min_explored

    def test_uniform_distribution_approximation(self):
        """Over many trials, selection should be roughly uniform"""
        bot = RandomBot()
        board = [[3, 2, 0], [3, 2, 1], [3, 2, 3]]
        move1, move2 = (0, 0), (0, 1)

        counts = {move1: 0, move2: 0}
        trials = 1000

        for _ in range(trials):
            action = bot.select_action(board)
            if action in counts:
                counts[action] += 1

        # Each move should occur roughly 50% of the time (allow some variance)
        expected_per_move = trials // 2
        tolerance = trials * 0.1  # 10% tolerance

        assert (
            expected_per_move - tolerance
            <= counts[move1]
            <= expected_per_move + tolerance
        )


class TestRandomBotEdgeCases:
    """Test RandomBot behavior in edge cases"""

    def test_single_cell_board(self):
        bot = RandomBot()
        board = [[1]]
        expected_result = None

        action = bot.select_action(board)
        assert action == expected_result

    def test_checkerboard_pattern(self):
        """All cells isolated - no valid moves"""
        bot = RandomBot()

        board = [[1, 2, 1, 2], [2, 1, 2, 1], [1, 2, 1, 2], [2, 1, 2, 1]]

        action = bot.select_action(board)
        assert action is None

    def test_full_board_same_color(self):
        """Entire board is one big group"""
        bot = RandomBot()

        board = [[1, 1], [1, 1]]
        expected_move = (0, 0)

        action = bot.select_action(board)
        assert action == expected_move
