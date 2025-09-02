"""
Tests for GreedySinglesBot behavior.

These tests focus on validating that GreedySinglesBot consistently selects
moves that minimize the number of single isolated tiles on the board.
"""

import pytest
from samegamerl.agents.greedy_singles_bot import GreedySinglesBot


class TestGreedySinglesBotInitialization:
    """Test GreedySinglesBot creation and basic properties"""

    def test_bot_creation(self):
        bot = GreedySinglesBot()
        assert bot is not None
        assert hasattr(bot, "select_action")


class TestGreedySinglesBotSinglesReduction:
    """Test that bot prioritizes moves that reduce singles count"""

    def test_clear_singles_reduction_choice(self):
        """When one move clearly reduces singles more than others, choose it"""
        bot = GreedySinglesBot()

        board = [[0, 3, 0, 3], [3, 2, 3, 2], [2, 1, 2, 3], [1, 1, 1, 1]]

        best_singles_reduction_move = (2, 1)

        action = bot.select_action(board)
        assert action == best_singles_reduction_move

    def test_avoid_singles_increasing_moves(self):
        """Bot should avoid moves that increase singles count when better options exist"""
        bot = GreedySinglesBot()
        board = [[0, 1, 0], [0, 3, 1], [3, 2, 2], [1, 1, 3]]
        singles_reducing_moves = [(3, 0), (2, 1)]  # List of moves that reduce singles

        action = bot.select_action(board)
        assert action in singles_reducing_moves

    def test_neutral_singles_moves(self):
        """When no move changes singles count, bot should pick any valid move"""
        bot = GreedySinglesBot()
        board = [[0, 0, 1], [3, 3, 1], [2, 2, 3], [1, 1, 3]]
        valid_moves = [
            (1, 0),
            (2, 0),
            (3, 0),
            (0, 2),
            (2, 2),
        ]  # All valid moves (all equally good)

        action = bot.select_action(board)
        assert action in valid_moves

    def test_no_valid_moves(self):
        """When no valid moves exist, bot returns None"""
        bot = GreedySinglesBot()
        board = [[0, 1, 0], [0, 3, 1], [3, 1, 2], [1, 2, 3]]

        action = bot.select_action(board)
        assert action is None


class TestGreedySinglesBotTieBreaking:
    """Test bot's behavior when multiple moves have same singles impact"""

    def test_multiple_moves_same_singles_reduction(self):
        """When moves tie on singles reduction, bot should have consistent tie-breaking"""
        bot = GreedySinglesBot()
        board = [[0, 1, 0], [0, 3, 1], [3, 2, 2], [1, 1, 3]]
        tied_best_moves = [(3, 0), (2, 1)]  # List of moves with equal singles reduction

        action = bot.select_action(board)
        assert action in tied_best_moves

    def test_tie_breaking_consistency(self):
        """Same board should produce same choice when ties exist"""
        bot = GreedySinglesBot()
        board = [[0, 1, 0], [0, 3, 1], [3, 2, 2], [1, 1, 3]]

        # Should return same choice consistently
        first_choice = bot.select_action(board)
        for _ in range(5):
            action = bot.select_action(board)
            assert action == first_choice

    def test_tie_breaking_strategy_preference(self):
        """Test your preferred tie-breaking strategy when singles reduction ties"""
        bot = GreedySinglesBot()

        board = [[0, 1, 0], [0, 3, 1], [3, 2, 2], [1, 1, 3]]
        preferred_move = (2, 1)

        action = bot.select_action(board)
        assert action == preferred_move


class TestGreedySinglesBotComplexScenarios:
    """Test bot's analysis of complex board states"""

    def test_complex_board_analysis(self):
        """Bot correctly analyzes complex boards with many interconnected groups"""
        bot = GreedySinglesBot()
        board = [
            [2, 2, 3, 2, 2],
            [3, 2, 2, 3, 1],
            [3, 3, 3, 2, 1],
            [2, 2, 2, 3, 1],
            [2, 1, 1, 2, 2],
        ]

        optimal_singles_move = (3, 0)

        action = bot.select_action(board)
        assert action == optimal_singles_move


class TestGreedySinglesBotEdgeCases:
    """Test GreedySinglesBot behavior in edge cases"""

    def test_single_group_available(self):
        """When only one group exists, bot must choose it"""
        bot = GreedySinglesBot()
        board = [[2, 1, 0], [2, 3, 1], [3, 1, 2]]
        only_group_position = (0, 0)

        action = bot.select_action(board)
        assert action == only_group_position

    def test_all_moves_increase_singles(self):
        """When all moves increase singles, bot picks the least harmful one"""
        bot = GreedySinglesBot()
        board = [
            [5, 5, 5],
            [0, 4, 4],
            [4, 4, 3],
            [3, 2, 2],
            [1, 1, 3],
        ]  # Board where every move makes things worse

        board = [
            [1, 1, 1, 1, 1],
            [1, 2, 2, 2, 2],
            [1, 5, 5, 5, 2],
            [1, 2, 2, 4, 4],
            [1, 1, 3, 3, 1],
        ]

        least_harmful_move = (1, 1)  # Move that increases singles by smallest amount

        action = bot.select_action(board)
        assert action == least_harmful_move

    def test_empty_board(self):
        """Empty board should return None"""
        bot = GreedySinglesBot()
        board = [[0, 0], [0, 0]]

        action = bot.select_action(board)
        assert action is None

    def test_full_board_same_color(self):
        """Entire board same color - any position equally valid"""
        bot = GreedySinglesBot()
        board = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        best_move = (0, 0)  # All positions should be equally good

        action = bot.select_action(board)
        assert action == best_move
