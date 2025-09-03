"""
Tests for the benchmark system components.

Validates dataset generation, bot evaluation, performance measurement,
and analysis functionality.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch

from samegamerl.evaluation.benchmark_dataset import BenchmarkDataset, GameSnapshot, BotPerformance
from samegamerl.evaluation.benchmark_runner import BenchmarkRunner
from samegamerl.evaluation.benchmark_analysis import BenchmarkAnalysis
from samegamerl.game.game_config import GameFactory
from samegamerl.agents.random_bot import RandomBot
from samegamerl.agents.benchmark_bot_base import BenchmarkBotBase


class MockBot(BenchmarkBotBase):
    """Mock bot for testing that always returns the same action"""
    
    def __init__(self, action_to_return=(0, 0)):
        self.action_to_return = action_to_return
        self.call_count = 0
    
    def select_action(self, board):
        self.call_count += 1
        if board[0][0] != 0:  # If position is valid
            return self.action_to_return
        return None  # No valid move


class TestBenchmarkDataset:
    """Test BenchmarkDataset functionality"""

    def test_dataset_initialization(self):
        dataset = BenchmarkDataset()
        assert len(dataset) == 0
        assert dataset.games == []
        assert dataset.results == {}

    def test_generate_games_reproducible(self):
        """Test that game generation is reproducible with same seed"""
        config = GameFactory.small()
        
        dataset1 = BenchmarkDataset()
        dataset1.generate_games(num_games=5, config=config, base_seed=42)
        
        dataset2 = BenchmarkDataset()
        dataset2.generate_games(num_games=5, config=config, base_seed=42)
        
        # Should generate identical games
        assert len(dataset1) == len(dataset2)
        for i in range(len(dataset1)):
            game1 = dataset1.get_game(i)
            game2 = dataset2.get_game(i)
            assert game1.board == game2.board
            assert game1.seed == game2.seed
            assert game1.game_id == game2.game_id

    def test_generate_games_different_seeds(self):
        """Test that different seeds generate different games"""
        config = GameFactory.small()
        
        dataset1 = BenchmarkDataset()
        dataset1.generate_games(num_games=5, config=config, base_seed=42)
        
        dataset2 = BenchmarkDataset()
        dataset2.generate_games(num_games=5, config=config, base_seed=123)
        
        # Should generate different games
        different_found = False
        for i in range(len(dataset1)):
            game1 = dataset1.get_game(i)
            game2 = dataset2.get_game(i)
            if game1.board != game2.board:
                different_found = True
                break
        
        assert different_found

    def test_save_load_dataset(self):
        """Test dataset persistence"""
        config = GameFactory.small()
        
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # Create and save dataset
            original_dataset = BenchmarkDataset()
            original_dataset.generate_games(num_games=3, config=config, base_seed=42)
            original_dataset.save_dataset(tmp_path)
            
            # Load dataset
            loaded_dataset = BenchmarkDataset()
            success = loaded_dataset.load_dataset(tmp_path)
            
            assert success
            assert len(loaded_dataset) == len(original_dataset)
            assert loaded_dataset.games[0].board == original_dataset.games[0].board
            
        finally:
            Path(tmp_path).unlink(missing_ok=True)


    def test_add_bot_results(self):
        """Test adding bot performance results"""
        dataset = BenchmarkDataset()
        
        performances = [
            BotPerformance("TestBot", 0, 15, 2, 5, False),
            BotPerformance("TestBot", 1, 20, 0, 8, True)
        ]
        
        dataset.add_bot_results("TestBot", performances)
        
        retrieved = dataset.get_bot_results("TestBot")
        assert len(retrieved) == 2
        assert retrieved[0].tiles_cleared == 15
        assert retrieved[1].completed is True

    def test_get_game_bounds(self):
        """Test game retrieval with bounds checking"""
        dataset = BenchmarkDataset()
        config = GameFactory.small()
        dataset.generate_games(num_games=3, config=config)
        
        # Valid access
        game = dataset.get_game(0)
        assert game.game_id == 0
        
        # Invalid access
        with pytest.raises(IndexError):
            dataset.get_game(5)
        
        with pytest.raises(IndexError):
            dataset.get_game(-1)


class TestBenchmarkRunner:
    """Test BenchmarkRunner functionality"""

    def test_runner_initialization(self):
        dataset = BenchmarkDataset()
        runner = BenchmarkRunner(dataset)
        assert runner.dataset is dataset
        assert "RandomBot" in runner.available_bots
        assert "LargestGroupBot" in runner.available_bots
        assert "GreedySinglesBot" in runner.available_bots

    def test_run_bot_on_game(self):
        """Test running a single bot on a single game"""
        # Create dataset with one game
        dataset = BenchmarkDataset()
        config = GameFactory.small()
        dataset.generate_games(num_games=1, config=config, base_seed=42)
        
        runner = BenchmarkRunner(dataset)
        mock_bot = MockBot((0, 0))
        
        game_snapshot = dataset.get_game(0)
        performance = runner.run_bot_on_game(mock_bot, game_snapshot)
        
        assert isinstance(performance, BotPerformance)
        assert performance.bot_name == "MockBot"
        assert performance.game_id == 0
        assert performance.moves_made >= 0
        assert mock_bot.call_count > 0

    def test_evaluate_bot(self):
        """Test evaluating a bot against all games"""
        dataset = BenchmarkDataset()
        config = GameFactory.small()
        dataset.generate_games(num_games=3, config=config, base_seed=42)
        
        runner = BenchmarkRunner(dataset)
        mock_bot = MockBot((0, 0))
        
        results = runner.evaluate_bot("MockBot", mock_bot)
        
        assert len(results) == 3
        assert all(isinstance(r, BotPerformance) for r in results)
        assert all(r.bot_name == "MockBot" for r in results)
        assert all(r.game_id == i for i, r in enumerate(results))

    def test_evaluate_unknown_bot(self):
        """Test evaluating an unknown bot name"""
        dataset = BenchmarkDataset()
        runner = BenchmarkRunner(dataset)
        
        with pytest.raises(ValueError, match="Unknown bot"):
            runner.evaluate_bot("NonexistentBot")

    def test_get_bot_summary(self):
        """Test bot performance summary calculation"""
        dataset = BenchmarkDataset()
        runner = BenchmarkRunner(dataset)
        
        # Add mock results
        performances = [
            BotPerformance("TestBot", 0, 15, 2, 5, False),
            BotPerformance("TestBot", 1, 20, 0, 8, True),
            BotPerformance("TestBot", 2, 18, 1, 6, False)
        ]
        dataset.add_bot_results("TestBot", performances)
        
        summary = runner.get_bot_summary("TestBot")
        
        assert summary["total_games"] == 3
        assert summary["completion_rate"] == 1/3  # 1 out of 3 completed
        assert summary["avg_tiles_cleared"] == (15 + 20 + 18) / 3  # Average tiles cleared
        assert "avg_moves_made" in summary
        assert "avg_singles_remaining" in summary

    def test_compare_bots(self):
        """Test bot comparison functionality"""
        dataset = BenchmarkDataset()
        runner = BenchmarkRunner(dataset)
        
        # Add results for two bots
        perf1 = [BotPerformance("Bot1", 0, 15, 2, 5, False)]
        perf2 = [BotPerformance("Bot2", 0, 20, 0, 8, True)]
        
        dataset.add_bot_results("Bot1", perf1)
        dataset.add_bot_results("Bot2", perf2)
        
        comparison = runner.compare_bots(["Bot1", "Bot2"])
        
        assert "Bot1" in comparison
        assert "Bot2" in comparison
        assert comparison["Bot1"]["avg_tiles_cleared"] == 15
        assert comparison["Bot2"]["avg_tiles_cleared"] == 20
        assert comparison["Bot2"]["completion_rate"] == 1.0


class TestBenchmarkAnalysis:
    """Test BenchmarkAnalysis functionality"""

    def test_analysis_initialization(self):
        dataset = BenchmarkDataset()
        analysis = BenchmarkAnalysis(dataset)
        assert analysis.dataset is dataset

    def test_head_to_head_analysis(self):
        """Test head-to-head bot comparison"""
        dataset = BenchmarkDataset()
        analysis = BenchmarkAnalysis(dataset)
        
        # Add results for two bots on same games
        bot1_results = [
            BotPerformance("Bot1", 0, 15, 2, 5, False),
            BotPerformance("Bot1", 1, 18, 1, 6, False)
        ]
        bot2_results = [
            BotPerformance("Bot2", 0, 20, 0, 8, True),
            BotPerformance("Bot2", 1, 16, 3, 7, False)
        ]
        
        dataset.add_bot_results("Bot1", bot1_results)
        dataset.add_bot_results("Bot2", bot2_results)
        
        head_to_head = analysis.head_to_head_analysis("Bot1", "Bot2")
        
        assert head_to_head["bot1"] == "Bot1"
        assert head_to_head["bot2"] == "Bot2"
        # Performance calculated as tiles_cleared + (1000 if completed else 0)
        # Game 0: Bot1=15, Bot2=1020 (20+1000) -> Bot2 wins
        # Game 1: Bot1=18, Bot2=16 -> Bot1 wins
        assert head_to_head["wins_bot1"] == 1
        assert head_to_head["wins_bot2"] == 1
        assert head_to_head["ties"] == 0

    def test_head_to_head_missing_bot(self):
        """Test head-to-head analysis with missing bot"""
        dataset = BenchmarkDataset()
        analysis = BenchmarkAnalysis(dataset)
        
        result = analysis.head_to_head_analysis("Bot1", "NonexistentBot")
        assert "error" in result

    def test_find_interesting_games_high_variance(self):
        """Test finding top 10% games with highest performance variance"""
        dataset = BenchmarkDataset()
        analysis = BenchmarkAnalysis(dataset)
        
        # Create dataset with 10 games
        config = GameFactory.small()
        dataset.generate_games(num_games=10, config=config, base_seed=42)
        
        # Add results with varying performance differences
        bot1_results = []
        bot2_results = []
        for i in range(10):
            # Game 0 has high variance (10 vs 30), others have low variance
            if i == 0:
                bot1_results.append(BotPerformance("Bot1", i, 10, 5, 3, False))
                bot2_results.append(BotPerformance("Bot2", i, 30, 0, 10, True))
            else:
                # Low variance games (similar performance)
                bot1_results.append(BotPerformance("Bot1", i, 15, 3, 5, False))
                bot2_results.append(BotPerformance("Bot2", i, 16, 2, 6, False))
        
        dataset.add_bot_results("Bot1", bot1_results)
        dataset.add_bot_results("Bot2", bot2_results)
        
        interesting = analysis.find_interesting_games()
        # Should return 1 game (top 10% of 10 games = 1), and game 0 should be it
        assert len(interesting) == 1
        assert 0 in interesting



