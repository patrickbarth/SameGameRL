"""Factory for creating benchmark repositories based on storage type."""

from pathlib import Path

from samegamerl.evaluation.benchmark_repository import PickleBenchmarkRepository
from samegamerl.evaluation.benchmark_repository_interface import BenchmarkRepositoryInterface
from samegamerl.game.game_config import GameConfig

# Try to import database support
try:
    from samegamerl.database.availability import DATABASE_AVAILABLE

    if DATABASE_AVAILABLE:
        from samegamerl.evaluation.database_benchmark_repository import (
            DatabaseBenchmarkRepository,
        )
except ImportError:
    DATABASE_AVAILABLE = False


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
            if not DATABASE_AVAILABLE:
                print(
                    "Warning: Database dependencies not available. "
                    "Falling back to pickle storage."
                )
                print(
                    "Install database support with: poetry install -E database"
                )
                # Fall back to pickle - need to generate path
                if benchmark_path is None:
                    if config is None or base_seed is None:
                        raise ValueError(
                            "benchmark_path or (config + base_seed) required for fallback"
                        )
                    # Generate default path based on config
                    benchmark_path = BenchmarkRepositoryFactory._generate_fallback_path(
                        config, base_seed
                    )
                return PickleBenchmarkRepository(benchmark_path)

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
        types = ["pickle"]
        if DATABASE_AVAILABLE:
            types.append("database")
        return types

    @staticmethod
    def is_database_available() -> bool:
        """Check if database dependencies are available."""
        return DATABASE_AVAILABLE

    @staticmethod
    def _generate_fallback_path(config: GameConfig, base_seed: int) -> Path:
        """Generate default benchmark path for fallback storage."""
        filename = f"benchmark_{config.num_rows}_{config.num_cols}_{config.num_colors}_{base_seed}.pkl"
        # Use absolute path relative to this file's location
        project_root = Path(__file__).parent.parent.parent
        return project_root / "samegamerl" / "evaluation" / "benchmarks" / filename