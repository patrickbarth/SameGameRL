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


class TestRandomBotDeterminism:
    """Test that RandomBot behaves deterministically for the same board state"""

    def test_same_board_same_action(self):
        """Same board state should always produce the same action"""
        bot = RandomBot()
        board = [[3, 2, 1], [3, 2, 1], [3, 2, 3]]
        
        # Call multiple times with same board
        actions = [bot.select_action(board) for _ in range(10)]
        
        # All actions should be identical
        assert all(action == actions[0] for action in actions)
        # Action should be one of the valid moves
        valid_moves = [(0, 0), (0, 1), (0, 2)]
        assert actions[0] in valid_moves

    def test_different_boards_different_actions(self):
        """Different board states should produce different actions (usually)"""
        bot = RandomBot()
        
        board1 = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
        board2 = [[3, 2, 1], [3, 2, 1], [3, 2, 1]]
        board3 = [[2, 1, 3], [2, 1, 3], [2, 1, 3]]
        
        action1 = bot.select_action(board1)
        action2 = bot.select_action(board2)
        action3 = bot.select_action(board3)
        
        # At least some actions should be different
        actions = {action1, action2, action3}
        assert len(actions) > 1

    def test_determinism_across_bot_instances(self):
        """Different bot instances should produce same action for same board"""
        board = [[1, 1, 2], [1, 2, 2], [3, 3, 3]]
        
        bot1 = RandomBot()
        bot2 = RandomBot()
        
        action1 = bot1.select_action(board)
        action2 = bot2.select_action(board)
        
        assert action1 == action2

    def test_determinism_with_varying_board_sizes(self):
        """Determinism should work with different board configurations"""
        bot = RandomBot()
        
        # Small board
        small_board = [[1, 1], [2, 2]]
        small_action1 = bot.select_action(small_board)
        small_action2 = bot.select_action(small_board)
        assert small_action1 == small_action2
        
        # Larger board
        large_board = [
            [1, 2, 3, 1, 2],
            [2, 1, 2, 3, 1], 
            [3, 3, 1, 2, 2],
            [1, 2, 3, 1, 3]
        ]
        large_action1 = bot.select_action(large_board)
        large_action2 = bot.select_action(large_board)
        assert large_action1 == large_action2


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
