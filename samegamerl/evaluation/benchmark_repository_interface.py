"""Abstract interface for benchmark data repositories."""

from abc import ABC, abstractmethod
from pathlib import Path

from samegamerl.evaluation.benchmark_data import BenchmarkData, BotPerformance
from samegamerl.game.game_config import GameConfig


class BenchmarkRepositoryInterface(ABC):
    """Abstract interface for benchmark data persistence and validation."""

    @abstractmethod
    def save_data(self, data: BenchmarkData) -> None:
        """Save benchmark data to storage."""
        pass

    @abstractmethod
    def load_data(self) -> BenchmarkData | None:
        """Load benchmark data from storage."""
        pass

    @abstractmethod
    def validate_results(self, bot_name: str, results: list[BotPerformance]) -> int:
        """Validate existing results for a bot and return count of valid consecutive results."""
        pass

    @abstractmethod
    def determine_missing_games(
        self, bot_name: str, results: dict[str, list[BotPerformance]], num_games: int
    ) -> list[int]:
        """Determine which games need to be computed for a bot."""
        pass

    @abstractmethod
    def merge_results(
        self,
        existing: list[BotPerformance],
        new: list[BotPerformance],
        num_games: int,
        bot_name: str,
    ) -> list[BotPerformance]:
        """Merge new results with existing validated results for a bot."""
        pass

    @abstractmethod
    def data_exists(self) -> bool:
        """Check if benchmark data exists in storage."""
        pass

    @abstractmethod
    def is_compatible(self, config: GameConfig, base_seed: int) -> bool:
        """Check if existing data is compatible with current configuration."""
        pass