"""Factory for creating benchmark repositories based on storage type."""

from pathlib import Path

from samegamerl.evaluation.benchmark_repository import PickleBenchmarkRepository
from samegamerl.evaluation.database_benchmark_repository import DatabaseBenchmarkRepository
from samegamerl.evaluation.benchmark_repository_interface import BenchmarkRepositoryInterface
from samegamerl.game.game_config import GameConfig


class BenchmarkRepositoryFactory:
    """Factory for creating appropriate benchmark repository instances."""

    @staticmethod
    def create(
        storage_type: str = "pickle",
        config: GameConfig | None = None,
        base_seed: int | None = None,
        benchmark_path: Path | None = None,
    ) -> BenchmarkRepositoryInterface:
        """Create a benchmark repository based on storage type.

        Args:
            storage_type: Either "pickle" or "database"
            config: Game configuration (required for database)
            base_seed: Base seed for game generation (required for database)
            benchmark_path: Path for pickle storage (required for pickle)

        Returns:
            Appropriate repository instance

        Raises:
            ValueError: If required parameters are missing or storage_type is invalid
        """
        if storage_type == "pickle":
            if benchmark_path is None:
                raise ValueError("benchmark_path is required for pickle storage")
            return PickleBenchmarkRepository(benchmark_path)

        elif storage_type == "database":
            if config is None:
                raise ValueError("config is required for database storage")
            if base_seed is None:
                raise ValueError("base_seed is required for database storage")
            return DatabaseBenchmarkRepository(config, base_seed)

        else:
            raise ValueError(
                f"Invalid storage_type '{storage_type}'. Must be 'pickle' or 'database'"
            )

    @staticmethod
    def get_supported_types() -> list[str]:
        """Get list of supported storage types."""
        return ["pickle", "database"]