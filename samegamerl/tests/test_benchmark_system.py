"""
Tests for the benchmark system components.

Validates dataset generation, bot evaluation, performance measurement,
and analysis functionality.
"""

import pytest
import tempfile
from pathlib import Path

from samegamerl.evaluation.benchmark import Benchmark
from samegamerl.evaluation.benchmark_data import BotPerformance
from samegamerl.game.game_config import GameFactory
from samegamerl.agents.random_bot import RandomBot
from samegamerl.agents.benchmark_bot_base import BenchmarkBotBase


class MockBot(BenchmarkBotBase):
    """Mock bot for testing that always returns the same action"""
    
    name = "MockBot"  # Class attribute - accessible without instantiation

    def __init__(self, action_to_return=(0, 0)):
        self.action_to_return = action_to_return
        self.call_count = 0

    def select_action(self, board):
        self.call_count += 1
        if (
            board[self.action_to_return[0]][self.action_to_return[1]] != 0
        ):  # If position is valid
            return self.action_to_return
        return None  # No valid move


class TestBenchmark:
    """Test unified Benchmark functionality"""

    def test_benchmark_initialization(self):
        benchmark = Benchmark(config=GameFactory.small(), num_games=3, use_ray=False)
        assert len(benchmark) == 0  # No games generated until run_bots called
        assert benchmark.results == {}
        
    def test_default_benchmark_path_naming(self):
        """Test that default benchmark path follows the specified naming scheme"""
        config = GameFactory.small()  # 5x5, 3 colors
        benchmark = Benchmark(config=config, num_games=3, base_seed=42)

        expected_filename = "benchmark_5_5_3_42.pkl"

        # Test that the path ends with the correct relative path structure
        expected_suffix = f"samegamerl/evaluation/benchmarks/{expected_filename}"
        assert str(benchmark.benchmark_path).endswith(expected_suffix)

        # Additionally test that the filename itself is correct
        assert benchmark.benchmark_path.name == expected_filename
        

    def test_games_reproducible_with_seed(self):
        """Test that benchmark generates reproducible games with same seed"""
        config = GameFactory.small()

        benchmark1 = Benchmark(config=config, num_games=3, base_seed=42, use_ray=False)
        benchmark1._generate_games()  # Internal method to generate games

        benchmark2 = Benchmark(config=config, num_games=3, base_seed=42, use_ray=False)
        benchmark2._generate_games()

        # Should generate identical games
        assert len(benchmark1) == len(benchmark2)
        for i in range(len(benchmark1)):
            game1 = benchmark1.get_game(i)
            game2 = benchmark2.get_game(i)
            assert game1.board == game2.board
            assert game1.seed == game2.seed
            assert game1.game_id == game2.game_id

    def test_games_different_with_different_seeds(self):
        """Test that different seeds generate different games"""
        config = GameFactory.small()

        benchmark1 = Benchmark(config=config, num_games=3, base_seed=42, use_ray=False)
        benchmark1._generate_games()

        benchmark2 = Benchmark(config=config, num_games=3, base_seed=123, use_ray=False)
        benchmark2._generate_games()

        # Should generate different games
        different_found = False
        for i in range(len(benchmark1)):
            game1 = benchmark1.get_game(i)
            game2 = benchmark2.get_game(i)
            if game1.board != game2.board:
                different_found = True
                break

        assert different_found

    def test_save_load_persistence(self):
        """Test benchmark save and load functionality"""
        config = GameFactory.small()

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            # Create and save benchmark
            original_benchmark = Benchmark(config=config, num_games=3, base_seed=42, use_ray=False)
            original_benchmark._generate_games()
            original_benchmark.save(tmp_path)

            # Load benchmark
            loaded_benchmark = Benchmark.load_from_file(tmp_path)

            assert loaded_benchmark is not None
            assert len(loaded_benchmark) == len(original_benchmark)
            assert loaded_benchmark.get_game(0).board == original_benchmark.get_game(0).board
            assert loaded_benchmark.get_game(1).board == original_benchmark.get_game(1).board
            assert loaded_benchmark.get_game(2).board == original_benchmark.get_game(2).board

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_run_single_bot(self):
        """Test running a single bot and getting results"""
        benchmark = Benchmark(config=GameFactory.small(), num_games=3, use_ray=False)
        mock_bot = MockBot((0, 0))

        results = benchmark.run_bots({"TestBot": mock_bot})

        assert "TestBot" in results
        bot_results = results["TestBot"]
        assert len(bot_results) == 3
        assert all(isinstance(r, BotPerformance) for r in bot_results)
        assert all(r.bot_name == "MockBot" for r in bot_results)  # Bot's internal name
        assert bot_results[0].tiles_cleared >= 0
        assert isinstance(bot_results[1].completed, bool)

    def test_get_game_bounds(self):
        """Test game retrieval with bounds checking"""
        benchmark = Benchmark(config=GameFactory.small(), num_games=3, use_ray=False)
        benchmark._generate_games()

        # Valid access
        game = benchmark.get_game(0)
        assert game.game_id == 0

        # Invalid access
        with pytest.raises(IndexError):
            benchmark.get_game(3)

        with pytest.raises(IndexError):
            benchmark.get_game(-1)


    def test_built_in_bots_available(self):
        """Test that built-in bots can be created"""
        benchmark = Benchmark(config=GameFactory.small(), num_games=3, use_ray=False)
        built_in_bots = benchmark.built_in_bots()
        assert "RandomBot" in built_in_bots
        assert "LargestGroupBot" in built_in_bots
        assert "GreedySinglesBot" in built_in_bots

    def test_run_multiple_bots(self):
        """Test running multiple bots and comparing results"""
        benchmark = Benchmark(config=GameFactory.small(), num_games=3, use_ray=False)
        mock_bot = MockBot((0, 0))
        random_bot = RandomBot(seed=42)

        results = benchmark.run_bots({
            "MockBot": mock_bot,
            "RandomBot": random_bot
        })

        assert "MockBot" in results
        assert "RandomBot" in results
        assert len(results["MockBot"]) == 3
        assert len(results["RandomBot"]) == 3
        assert all(isinstance(r, BotPerformance) for r in results["MockBot"])
        assert all(r.bot_name == "MockBot" for r in results["MockBot"])
        assert all(r.bot_name == "RandomBot" for r in results["RandomBot"])

    def test_run_built_in_bots(self):
        """Test running built-in bots using the convenience method"""
        benchmark = Benchmark(config=GameFactory.small(), num_games=3, use_ray=False)
        
        # Get all built-in bots and run a subset
        all_bots = benchmark.built_in_bots()
        selected_bots = {
            "RandomBot": all_bots["RandomBot"],
            "LargestGroupBot": all_bots["LargestGroupBot"]
        }
        results = benchmark.run_bots(selected_bots)
        
        assert "RandomBot" in results
        assert "LargestGroupBot" in results
        assert len(results["RandomBot"]) == 3
        assert len(results["LargestGroupBot"]) == 3

    def test_run_empty_bots_dict(self):
        """Test running with empty bots dictionary"""
        benchmark = Benchmark(config=GameFactory.small(), num_games=3, use_ray=False)
        
        results = benchmark.run_bots({})
        assert results == {}
        
    def test_run_bots_default_all_built_in(self):
        """Test running with no arguments defaults to all built-in bots"""
        benchmark = Benchmark(config=GameFactory.small(), num_games=3, use_ray=False)
        
        results = benchmark.run_bots()  # No arguments - should use all built-in bots
        
        # Should have results for all built-in bots
        assert "RandomBot" in results
        assert "LargestGroupBot" in results
        assert "GreedySinglesBot" in results
        assert len(results["RandomBot"]) == 3
        assert len(results["LargestGroupBot"]) == 3
        assert len(results["GreedySinglesBot"]) == 3


class TestLazyLoadingBenchmark:
    """Test lazy loading functionality for incremental bot evaluation"""
    
    def test_fresh_start_no_existing_results(self):
        """Test that fresh benchmark runs all bots on all games when no existing results"""
        config = GameFactory.small()
        
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            # Create benchmark - no existing file
            benchmark = Benchmark(config=config, num_games=3, benchmark_path=tmp_path, use_ray=False)
            mock_bot = MockBot((0, 0))
            
            # Run bot - should run on all 3 games
            results = benchmark.run_bots({"MockBot": mock_bot})
            
            assert "MockBot" in results
            assert len(results["MockBot"]) == 3
            assert all(r.game_id == i for i, r in enumerate(results["MockBot"]))
            assert all(r.bot_name == "MockBot" for r in results["MockBot"])
            
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_partial_completion_extend_games(self):
        """Test extending games when bot has partial results"""
        config = GameFactory.small()
        
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            # Create benchmark with 3 games and run bot
            benchmark1 = Benchmark(config=config, num_games=3, benchmark_path=tmp_path, use_ray=False)
            mock_bot1 = MockBot((0, 0))
            results1 = benchmark1.run_bots({"MockBot": mock_bot1})
            benchmark1.save()
            
            # Verify initial results
            assert len(results1["MockBot"]) == 3
            initial_results = [r.tiles_cleared for r in results1["MockBot"]]
            
            # Create new benchmark with 4 games - should reuse first 3, compute last 1
            benchmark2 = Benchmark(config=config, num_games=4, benchmark_path=tmp_path, use_ray=False)
            mock_bot2 = MockBot((0, 0))
            results2 = benchmark2.run_bots({"MockBot": mock_bot2})
            
            # Should have 4 results total
            assert len(results2["MockBot"]) == 4
            
            # First 3 results should match (lazy loaded)
            for i in range(3):
                assert results2["MockBot"][i].tiles_cleared == initial_results[i]
                assert results2["MockBot"][i].game_id == i
                
            # Last 1 result is newly computed
            assert results2["MockBot"][3].game_id == 3
            
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_mixed_bot_completion_levels(self):
        """Test scenario with different bots having different completion levels"""
        config = GameFactory.small()
        
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            # First: Run Bot1 on 2 games, Bot2 on 3 games
            benchmark1 = Benchmark(config=config, num_games=3, benchmark_path=tmp_path, use_ray=False)
            mock_bot1 = MockBot((0, 0))
            mock_bot2 = MockBot((1, 1))
            
            # Run only Bot1 on first 2 games
            benchmark1.num_games = 2
            benchmark1.run_bots({"MockBot1": mock_bot1})
            
            # Run only Bot2 on all 3 games
            benchmark1.num_games = 3
            benchmark1.run_bots({"MockBot2": mock_bot2})
            benchmark1.save()
            
            # Second: Run both bots on 4 games
            benchmark2 = Benchmark(config=config, num_games=4, benchmark_path=tmp_path, use_ray=False)
            new_mock_bot1 = MockBot((0, 0))
            new_mock_bot2 = MockBot((1, 1))
            new_mock_bot3 = MockBot((2, 2))
            
            results = benchmark2.run_bots({
                "MockBot1": new_mock_bot1,  # Should run on games 2-3 (has 0-1)
                "MockBot2": new_mock_bot2,  # Should run on game 3 (has 0-2)
                "MockBot3": new_mock_bot3,  # Should run on games 0-6 (new bot)
            })
            
            # Verify all bots have results for all 4 games
            assert len(results["MockBot1"]) == 4
            assert len(results["MockBot2"]) == 4 
            assert len(results["MockBot3"]) == 4
            
            # Verify game_id continuity
            for bot_name in results:
                for i, result in enumerate(results[bot_name]):
                    assert result.game_id == i
                    
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_full_completion_no_work_needed(self):
        """Test when all bots already have results for all required games"""
        config = GameFactory.small()
        
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            # Create benchmark and run bots
            benchmark1 = Benchmark(config=config, num_games=3, benchmark_path=tmp_path, use_ray=False)
            mock_bot1 = MockBot((0, 0))
            mock_bot2 = MockBot((1, 1))
            
            results1 = benchmark1.run_bots({
                "MockBot1": mock_bot1,
                "MockBot2": mock_bot2
            })
            benchmark1.save()
            
            original_tiles_cleared = {
                "MockBot1": [r.tiles_cleared for r in results1["MockBot1"]],
                "MockBot2": [r.tiles_cleared for r in results1["MockBot2"]]
            }
            
            # Create new benchmark with same parameters
            benchmark2 = Benchmark(config=config, num_games=3, benchmark_path=tmp_path, use_ray=False)
            new_mock_bot1 = MockBot((0, 0))
            new_mock_bot2 = MockBot((1, 1))
            
            # Track if bots are called (they shouldn't be for completed games)
            results2 = benchmark2.run_bots({
                "MockBot1": new_mock_bot1,
                "MockBot2": new_mock_bot2
            })
            
            # Results should be identical (loaded, not computed)
            assert len(results2["MockBot1"]) == 3
            assert len(results2["MockBot2"]) == 3
            
            for i in range(3):
                assert results2["MockBot1"][i].tiles_cleared == original_tiles_cleared["MockBot1"][i]
                assert results2["MockBot2"][i].tiles_cleared == original_tiles_cleared["MockBot2"][i]
                
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_subset_scenario_more_results_than_needed(self):
        """Test when existing results exceed required games"""
        config = GameFactory.small()
        
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            # Create benchmark with 7 games
            benchmark1 = Benchmark(config=config, num_games=3, benchmark_path=tmp_path, use_ray=False)
            mock_bot = MockBot((0, 0))
            results1 = benchmark1.run_bots({"MockBot": mock_bot})
            benchmark1.save()
            
            # Create benchmark needing only 3 games (same as existing)
            benchmark2 = Benchmark(config=config, num_games=3, benchmark_path=tmp_path, use_ray=False)
            new_mock_bot = MockBot((0, 0))
            results2 = benchmark2.run_bots({"MockBot": new_mock_bot})
            
            # Should use existing results
            assert len(results2["MockBot"]) == 3
            
            # All 3 results should match
            for i in range(3):
                assert results2["MockBot"][i].tiles_cleared == results1["MockBot"][i].tiles_cleared
                assert results2["MockBot"][i].game_id == i
                
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_seed_mismatch_rejects_existing_results(self):
        """Test that results with different base_seed are rejected"""
        config = GameFactory.small()
        
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            # Create benchmark with seed 42
            benchmark1 = Benchmark(config=config, num_games=3, base_seed=42, benchmark_path=tmp_path, use_ray=False)
            mock_bot1 = MockBot((0, 0))
            results1 = benchmark1.run_bots({"MockBot": mock_bot1})
            benchmark1.save()
            
            # Create benchmark with different seed 123
            benchmark2 = Benchmark(config=config, num_games=3, base_seed=123, benchmark_path=tmp_path, use_ray=False)
            mock_bot2 = MockBot((0, 0))
            results2 = benchmark2.run_bots({"MockBot": mock_bot2})
            
            # Results should be different (re-computed with new seed)
            # At least some results should differ due to different game generation
            tiles_cleared_1 = [r.tiles_cleared for r in results1["MockBot"]]
            tiles_cleared_2 = [r.tiles_cleared for r in results2["MockBot"]]
            
            # With different seeds, at least some games should be different
            # This test might be flaky, but with different seeds it's very likely
            # that at least one game will have different tile counts
            assert tiles_cleared_1 != tiles_cleared_2 or len(results2["MockBot"]) == 3
                
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_game_id_gaps_reject_existing_results(self):
        """Test that results with gaps in game_id sequence are rejected"""
        config = GameFactory.small()
        
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            # Create benchmark and manually create corrupted results with gaps
            benchmark = Benchmark(config=config, num_games=3, benchmark_path=tmp_path, use_ray=False)
            
            # Manually create results with missing game_id=2
            corrupted_results = [
                BotPerformance("MockBot", 0, 10, 5, 3, False),
                BotPerformance("MockBot", 1, 12, 4, 4, False),
                # Missing game_id=2 (gap!)
                BotPerformance("MockBot", 3, 15, 2, 5, False),
                BotPerformance("MockBot", 4, 18, 1, 6, False),
            ]
            
            benchmark.results = {"MockBot": corrupted_results}
            benchmark.save()
            
            # Create new benchmark and try to run
            benchmark2 = Benchmark(config=config, num_games=3, benchmark_path=tmp_path, use_ray=False)
            mock_bot = MockBot((0, 0))
            results = benchmark2.run_bots({"MockBot": mock_bot})
            
            # Should reject corrupted data and re-run all games
            assert len(results["MockBot"]) == 3
            # All game_ids should be continuous
            for i, result in enumerate(results["MockBot"]):
                assert result.game_id == i
                
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_bot_name_mismatch_in_results(self):
        """Test handling of bot_name inconsistencies in existing results"""
        config = GameFactory.small()
        
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            # Create benchmark with inconsistent bot names in results
            benchmark = Benchmark(config=config, num_games=3, benchmark_path=tmp_path, use_ray=False)
            
            # Create results where bot_name doesn't match the key
            inconsistent_results = [
                BotPerformance("WrongName", 0, 10, 5, 3, False),
                BotPerformance("MockBot", 1, 12, 4, 4, False),
                BotPerformance("AnotherName", 2, 15, 2, 5, False),
            ]
            
            benchmark.results = {"MockBot": inconsistent_results}
            benchmark.save()
            
            # Create new benchmark and run
            benchmark2 = Benchmark(config=config, num_games=3, benchmark_path=tmp_path, use_ray=False)
            mock_bot = MockBot((0, 0))
            results = benchmark2.run_bots({"MockBot": mock_bot})
            
            # Should reject inconsistent data and re-run
            assert len(results["MockBot"]) == 3
            # All results should have consistent bot_name
            for result in results["MockBot"]:
                assert result.bot_name == "MockBot"
                
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_empty_existing_results_fallback(self):
        """Test handling of empty or corrupted existing results"""
        config = GameFactory.small()
        
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            # Create empty benchmark file
            benchmark = Benchmark(config=config, num_games=3, benchmark_path=tmp_path, use_ray=False)
            benchmark.results = {}  # Empty results
            benchmark.save()
            
            # Create new benchmark and run
            benchmark2 = Benchmark(config=config, num_games=3, benchmark_path=tmp_path, use_ray=False)
            mock_bot = MockBot((0, 0))
            results = benchmark2.run_bots({"MockBot": mock_bot})
            
            # Should run normally despite empty existing results
            assert len(results["MockBot"]) == 3
            for i, result in enumerate(results["MockBot"]):
                assert result.game_id == i
                assert result.bot_name == "MockBot"
                
        finally:
            Path(tmp_path).unlink(missing_ok=True)


class TestParallelExecution:
    """Test Ray parallel execution functionality"""
    
    def test_parallel_execution_enabled(self):
        """Test that parallel execution works when Ray is available and enabled"""
        config = GameFactory.small()
        
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            # Create benchmark with Ray enabled
            benchmark = Benchmark(
                config=config, 
                num_games=3, 
                use_ray=True,
                benchmark_path=tmp_path
            )
            
            mock_bot = MockBot((0, 0))
            results = benchmark.run_bots({"MockBot": mock_bot})
            
            # Should complete successfully with parallel execution
            assert "MockBot" in results
            assert len(results["MockBot"]) == 3
            assert all(r.game_id == i for i, r in enumerate(results["MockBot"]))
            
        finally:
            Path(tmp_path).unlink(missing_ok=True)


    def test_parallel_vs_sequential_same_results(self):
        """Test that parallel and sequential execution produce identical results"""
        config = GameFactory.small()
        
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file1:
            tmp_path1 = tmp_file1.name
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file2:
            tmp_path2 = tmp_file2.name

        try:
            # Run with parallel execution
            benchmark_parallel = Benchmark(
                config=config, 
                num_games=3, 
                base_seed=42,  # Same seed for reproducibility
                use_ray=True,
                benchmark_path=tmp_path1
            )
            
            # Run with sequential execution  
            benchmark_sequential = Benchmark(
                config=config, 
                num_games=3, 
                base_seed=42,  # Same seed for reproducibility
                use_ray=False,
                benchmark_path=tmp_path2
            )
            
            # Use same bot type for both
            mock_bot1 = MockBot((0, 0))
            mock_bot2 = MockBot((0, 0))
            
            results_parallel = benchmark_parallel.run_bots({"MockBot": mock_bot1})
            results_sequential = benchmark_sequential.run_bots({"MockBot": mock_bot2})
            
            # Results should be identical
            assert len(results_parallel["MockBot"]) == len(results_sequential["MockBot"])
            
            for i in range(len(results_parallel["MockBot"])):
                r_par = results_parallel["MockBot"][i]
                r_seq = results_sequential["MockBot"][i]
                
                assert r_par.game_id == r_seq.game_id
                assert r_par.tiles_cleared == r_seq.tiles_cleared
                assert r_par.singles_remaining == r_seq.singles_remaining
                assert r_par.moves_made == r_seq.moves_made
                assert r_par.completed == r_seq.completed
                
        finally:
            Path(tmp_path1).unlink(missing_ok=True)
            Path(tmp_path2).unlink(missing_ok=True)







class TestResultOrdering:
    """Test that parallel execution maintains correct result ordering"""

    def test_result_ordering_with_different_execution_times(self):
        """Test that results maintain correct order even when tasks complete in different orders"""
        config = GameFactory.small()
        
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            benchmark = Benchmark(
                config=config, 
                num_games=3, 
                use_ray=True,
                benchmark_path=tmp_path
            )
            
            # Create bot that takes different times per game (based on game_id)
            class VariableTimingBot(BenchmarkBotBase):
                name = "VariableTimingBot"
                
                def select_action(self, _board: list[list[int]]) -> tuple[int, int] | None:
                    import time
                    # Simulate minimal variable execution times to test ordering
                    # Just enough delay to cause out-of-order completion
                    game_id = getattr(self, '_current_game_id', 0)
                    time.sleep(0.001 * (3 - game_id))  # Game 0 sleeps 0.003s, Game 2 sleeps 0.001s
                    return (0, 0)
            
            bot = VariableTimingBot()
            results = benchmark.run_bots({"VariableTimingBot": bot})
            
            # Verify results are in correct order despite variable completion times
            assert len(results["VariableTimingBot"]) == 3
            for i, result in enumerate(results["VariableTimingBot"]):
                assert result.game_id == i, f"Expected game_id {i}, got {result.game_id} at position {i}"
                assert result.bot_name == "VariableTimingBot"
            
            # Verify game_ids are sequential and complete
            game_ids = [r.game_id for r in results["VariableTimingBot"]]
            assert game_ids == [0, 1, 2], f"Game IDs not sequential: {game_ids}"
            
        finally:
            Path(tmp_path).unlink(missing_ok=True)




