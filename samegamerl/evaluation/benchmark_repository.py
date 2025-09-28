"""Repository for benchmark data persistence and validation."""

import pickle
from dataclasses import dataclass
from pathlib import Path

from samegamerl.evaluation.benchmark_data import (
    GameSnapshot,
    BotPerformance,
    BenchmarkData,
)
from samegamerl.game.game_config import GameConfig


class BenchmarkRepository:
    """Handles persistence and validation of benchmark data"""

    def __init__(self, benchmark_path: Path):
        self.benchmark_path = benchmark_path

    def save_data(self, data: BenchmarkData) -> None:
        """Save benchmark data to disk"""
        # Ensure parent directory exists
        self.benchmark_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert dataclass to dict for pickle compatibility
        data_dict = {
            "games": data.games,
            "results": data.results,
            "config": data.config,
            "num_games": data.num_games,
            "base_seed": data.base_seed,
        }

        with open(self.benchmark_path, "wb") as f:
            pickle.dump(data_dict, f)

    def load_data(self) -> BenchmarkData | None:
        """Load benchmark data from disk"""
        if not self.data_exists():
            return None

        try:
            with open(self.benchmark_path, "rb") as f:
                data_dict = pickle.load(f)

            return BenchmarkData(
                games=data_dict.get("games", []),
                results=data_dict.get("results", {}),
                config=data_dict["config"],
                num_games=data_dict.get("num_games", 1000),
                base_seed=data_dict.get("base_seed", 42),
            )
        except Exception:
            return None

    def validate_results(self, bot_name: str, results: list[BotPerformance]) -> int:
        """Validate existing results for a bot and return count of valid consecutive results"""
        if not results:
            return 0

        valid_count = 0
        expected_game_id = 0

        for result in results:
            # Check game_id continuity (must be sequential starting from 0)
            if result.game_id != expected_game_id:
                break

            # Check bot_name consistency
            if result.bot_name != bot_name:
                break

            # Check that result has all required fields
            if not self._is_valid_performance(result):
                break

            valid_count += 1
            expected_game_id += 1

        return valid_count

    def determine_missing_games(
        self, bot_name: str, results: dict[str, list[BotPerformance]], num_games: int
    ) -> list[int]:
        """Determine which games need to be computed for a bot"""
        existing_results = results.get(bot_name, [])
        valid_count = self.validate_results(bot_name, existing_results)

        # Return list of missing game_ids
        return list(range(valid_count, num_games))

    def merge_results(
        self,
        existing: list[BotPerformance],
        new: list[BotPerformance],
        num_games: int,
        bot_name: str,
    ) -> list[BotPerformance]:
        """Merge new results with existing validated results for a bot"""
        valid_count = self.validate_results(bot_name, existing)

        # Keep only the valid existing results
        validated_existing = existing[:valid_count]

        # Create a dictionary for fast lookup of new results by game_id
        new_results_dict = {result.game_id: result for result in new}

        # Build final results list maintaining order
        merged_results = []

        for game_id in range(num_games):
            if game_id < valid_count:
                # Use existing valid result
                merged_results.append(validated_existing[game_id])
            elif game_id in new_results_dict:
                # Use new result
                merged_results.append(new_results_dict[game_id])
            else:
                # This shouldn't happen if determine_missing_games worked correctly
                # but we'll handle it gracefully
                break

        return merged_results

    def data_exists(self) -> bool:
        """Check if benchmark data file exists"""
        return self.benchmark_path.exists()

    def is_compatible(self, config: GameConfig, base_seed: int) -> bool:
        """Check if existing data is compatible with current configuration"""
        data = self.load_data()
        if data is None:
            return False

        return data.config == config and data.base_seed == base_seed

    def _is_valid_performance(self, result: BotPerformance) -> bool:
        """Check if a performance result has valid data"""
        try:
            return (
                isinstance(result.tiles_cleared, int)
                and result.tiles_cleared >= 0
                and isinstance(result.singles_remaining, int)
                and result.singles_remaining >= 0
                and isinstance(result.moves_made, int)
                and result.moves_made >= 0
                and isinstance(result.completed, bool)
            )
        except AttributeError:
            return False