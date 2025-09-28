"""
Tests for the BenchmarkRepository class.

Validates data persistence, result validation, and merging functionality
in isolation from the main Benchmark orchestration logic.
"""

import pytest
import tempfile
from pathlib import Path

from samegamerl.evaluation.benchmark_repository import BenchmarkRepository
from samegamerl.evaluation.benchmark_data import (
    BenchmarkData,
    BotPerformance,
    GameSnapshot,
)
from samegamerl.game.game_config import GameFactory


class TestBenchmarkRepository:
    """Test BenchmarkRepository functionality"""

    def test_repository_initialization(self):
        """Test basic repository initialization"""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "test.pkl"
            repo = BenchmarkRepository(path)
            assert repo.benchmark_path == path
            assert not repo.data_exists()

    def test_save_and_load_data(self):
        """Test saving and loading benchmark data"""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "test.pkl"
            repo = BenchmarkRepository(path)

            # Create test data
            config = GameFactory.small()
            test_data = BenchmarkData(
                games=[
                    GameSnapshot(
                        board=[[1, 2], [2, 1]],
                        config=config,
                        seed=123,
                        game_id=0,
                    )
                ],
                results={
                    "TestBot": [
                        BotPerformance(
                            bot_name="TestBot",
                            game_id=0,
                            tiles_cleared=3,
                            singles_remaining=1,
                            moves_made=2,
                            completed=False,
                        )
                    ]
                },
                config=config,
                num_games=1,
                base_seed=42,
            )

            # Save and load
            repo.save_data(test_data)
            assert repo.data_exists()

            loaded_data = repo.load_data()
            assert loaded_data is not None
            assert loaded_data.config == config
            assert loaded_data.num_games == 1
            assert loaded_data.base_seed == 42
            assert len(loaded_data.games) == 1
            assert len(loaded_data.results["TestBot"]) == 1

    def test_load_nonexistent_data(self):
        """Test loading from nonexistent file"""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "nonexistent.pkl"
            repo = BenchmarkRepository(path)

            assert not repo.data_exists()
            assert repo.load_data() is None

    def test_is_compatible(self):
        """Test compatibility checking"""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "test.pkl"
            repo = BenchmarkRepository(path)

            config = GameFactory.small()
            test_data = BenchmarkData(
                games=[],
                results={},
                config=config,
                num_games=10,
                base_seed=42,
            )

            repo.save_data(test_data)

            # Same config and seed should be compatible
            assert repo.is_compatible(config, 42)

            # Different seed should not be compatible
            assert not repo.is_compatible(config, 99)

            # Different config should not be compatible
            different_config = GameFactory.medium()
            assert not repo.is_compatible(different_config, 42)

    def test_validate_results_valid_sequence(self):
        """Test validation of valid result sequence"""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "test.pkl"
            repo = BenchmarkRepository(path)

            valid_results = [
                BotPerformance("TestBot", 0, 10, 2, 5, False),
                BotPerformance("TestBot", 1, 15, 0, 8, True),
                BotPerformance("TestBot", 2, 12, 1, 6, False),
            ]

            assert repo.validate_results("TestBot", valid_results) == 3

    def test_validate_results_broken_sequence(self):
        """Test validation with broken game_id sequence"""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "test.pkl"
            repo = BenchmarkRepository(path)

            broken_results = [
                BotPerformance("TestBot", 0, 10, 2, 5, False),
                BotPerformance("TestBot", 2, 15, 0, 8, True),  # Missing game_id 1
                BotPerformance("TestBot", 3, 12, 1, 6, False),
            ]

            assert repo.validate_results("TestBot", broken_results) == 1

    def test_validate_results_wrong_bot_name(self):
        """Test validation with wrong bot name"""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "test.pkl"
            repo = BenchmarkRepository(path)

            mixed_results = [
                BotPerformance("TestBot", 0, 10, 2, 5, False),
                BotPerformance("DifferentBot", 1, 15, 0, 8, True),  # Wrong bot name
            ]

            assert repo.validate_results("TestBot", mixed_results) == 1

    def test_determine_missing_games(self):
        """Test determining which games are missing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "test.pkl"
            repo = BenchmarkRepository(path)

            # Bot with results for games 0 and 1
            results = {
                "TestBot": [
                    BotPerformance("TestBot", 0, 10, 2, 5, False),
                    BotPerformance("TestBot", 1, 15, 0, 8, True),
                ]
            }

            missing = repo.determine_missing_games("TestBot", results, 5)
            assert missing == [2, 3, 4]

            # Bot with no results
            missing = repo.determine_missing_games("NewBot", results, 3)
            assert missing == [0, 1, 2]

    def test_merge_results(self):
        """Test merging new results with existing ones"""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "test.pkl"
            repo = BenchmarkRepository(path)

            existing = [
                BotPerformance("TestBot", 0, 10, 2, 5, False),
                BotPerformance("TestBot", 1, 15, 0, 8, True),
            ]

            new_results = [
                BotPerformance("TestBot", 2, 12, 1, 6, False),
                BotPerformance("TestBot", 3, 8, 3, 4, False),
            ]

            merged = repo.merge_results(existing, new_results, 4, "TestBot")

            assert len(merged) == 4
            assert merged[0].game_id == 0
            assert merged[1].game_id == 1
            assert merged[2].game_id == 2
            assert merged[3].game_id == 3

            # Check that existing results are preserved
            assert merged[0].tiles_cleared == 10
            assert merged[1].tiles_cleared == 15

            # Check that new results are added
            assert merged[2].tiles_cleared == 12
            assert merged[3].tiles_cleared == 8

    def test_merge_results_with_invalid_existing(self):
        """Test merging when existing results have gaps"""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "test.pkl"
            repo = BenchmarkRepository(path)

            # Existing results with a gap (missing game_id 1)
            existing = [
                BotPerformance("TestBot", 0, 10, 2, 5, False),
                BotPerformance("TestBot", 2, 15, 0, 8, True),  # Gap here
            ]

            new_results = [
                BotPerformance("TestBot", 1, 12, 1, 6, False),
                BotPerformance("TestBot", 2, 8, 3, 4, False),
            ]

            merged = repo.merge_results(existing, new_results, 3, "TestBot")

            # Should only keep the first valid result, then add new ones
            assert len(merged) == 3
            assert merged[0].game_id == 0
            assert merged[0].tiles_cleared == 10  # From existing
            assert merged[1].game_id == 1
            assert merged[1].tiles_cleared == 12  # From new
            assert merged[2].game_id == 2
            assert merged[2].tiles_cleared == 8  # From new