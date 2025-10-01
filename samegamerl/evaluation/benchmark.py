"""Benchmark system for consistent agent evaluation across standardized games."""

import random
from pathlib import Path

# Optional Ray import for parallelization
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

from samegamerl.game.game import Game
from samegamerl.game.game_config import GameConfig
from samegamerl.agents.benchmark_bot_base import BenchmarkBotBase
from samegamerl.agents.random_bot import RandomBot
from samegamerl.agents.largest_group_bot import LargestGroupBot
from samegamerl.agents.greedy_singles_bot import GreedySinglesBot
from samegamerl.evaluation.benchmark_data import (
    GameSnapshot,
    BotPerformance,
    BenchmarkData,
)
from samegamerl.evaluation.benchmark_repository_factory import BenchmarkRepositoryFactory
from samegamerl.evaluation.benchmark_execution_strategies import (
    ExecutionStrategyFactory,
)


class Benchmark:
    """Unified benchmark system for evaluating agents consistently.
    Provides a standardized set of games that bots and agents can be compared on.

    Main Interface:
    - run_bots(): Execute benchmarks on multiple bots
    - get_game(): Get specific game by ID
    - save()/load_from_file(): Persistence operations
    - built_in_bots(): Access to standard benchmark bots
    """

    # === INITIALIZATION ===

    def __init__(
        self,
        config: GameConfig,
        num_games: int = 1000,
        base_seed: int = 42,
        benchmark_path: str | None = None,
        use_ray: bool = True,
        ray_num_cpus: int | None = None,
        storage_type: str = "pickle",
    ):
        self.num_games = num_games
        self.config = config
        self.base_seed = base_seed
        self.storage_type = storage_type
        self.use_ray = use_ray and RAY_AVAILABLE
        self.ray_num_cpus = ray_num_cpus
        self._ray_initialized = False

        # Set up execution strategy
        self.execution_strategy = ExecutionStrategyFactory.create_strategy(
            use_ray, ray_num_cpus
        )

        # Set up repository based on storage type
        if storage_type == "pickle":
            self.benchmark_path = Path(self._get_benchmark_path(benchmark_path))
            self.repository = BenchmarkRepositoryFactory.create(
                storage_type="pickle",
                benchmark_path=self.benchmark_path
            )
        elif storage_type == "database":
            self.repository = BenchmarkRepositoryFactory.create(
                storage_type="database",
                config=config,
                base_seed=base_seed
            )
        else:
            raise ValueError(f"Invalid storage_type '{storage_type}'. Must be 'pickle' or 'database'")

        self.games: list[GameSnapshot] = []
        self.results: dict[str, list[BotPerformance]] = {}

    # === PUBLIC INTERFACE ===

    def run_bots(
        self, bots: dict[str, BenchmarkBotBase] | None = None
    ) -> dict[str, list[BotPerformance]]:
        """Execute benchmarks on multiple bots.

        Args:
            bots: Dict of bot_name -> bot_instance. Uses built-in bots if None.

        Returns:
            Dict of bot_name -> list of performance results
        """
        # Use all built-in bots if none specified
        if bots is None:
            bots = self.built_in_bots()

        # Initialize Ray if needed for parallel execution
        self._initialize_ray()

        try:
            # Try to load existing results for lazy loading
            self._load_existing_results()

            # Ensure games are generated
            self._generate_games()

            results = {}
            any_new_computation = False

            for bot_name, bot_instance in bots.items():
                # Determine which games need to be computed for this bot
                missing_game_ids = self.repository.determine_missing_games(
                    bot_name, self.results, self.num_games
                )

                if not missing_game_ids:
                    # Bot already has all required results - use existing
                    print(
                        f"{bot_name}: Using existing results for all {self.num_games} games"
                    )
                    results[bot_name] = self.results[bot_name][: self.num_games]
                    continue

                # Run bot only on missing games
                print(
                    f"{bot_name}: Computing {len(missing_game_ids)} missing games (total {self.num_games})"
                )
                any_new_computation = True

                # Run games using execution strategy
                new_results = self._run_games_for_bot(
                    bot_instance, missing_game_ids, bot_name
                )

                # Merge new results with existing validated results
                existing_results = self.results.get(bot_name, [])
                merged_results = self.repository.merge_results(
                    existing_results, new_results, self.num_games, bot_name
                )
                self.results[bot_name] = merged_results
                results[bot_name] = self.results[bot_name]

            # Save updated results if any new computation was done
            if any_new_computation:
                self.save()

            return results

        finally:
            # Clean up Ray if we initialized it
            self._cleanup_ray()

    def get_game(self, game_id: int) -> GameSnapshot:
        """Get a specific game by ID."""
        if not self.games:
            self._generate_games()

        if 0 <= game_id < len(self.games):
            return self.games[game_id]
        raise IndexError(f"Game ID {game_id} out of range")

    def save(self, filepath: str | None = None) -> None:
        """Save benchmark data to storage."""
        if filepath is None:
            # Use default repository
            repository = self.repository
        else:
            # Use custom filepath - only works for pickle storage
            if self.storage_type != "pickle":
                raise ValueError("Custom filepath only supported for pickle storage")
            repository = BenchmarkRepositoryFactory.create(
                storage_type="pickle",
                benchmark_path=Path(filepath)
            )

        data = BenchmarkData(
            games=self.games,
            results=self.results,
            config=self.config,
            num_games=self.num_games,
            base_seed=self.base_seed,
        )
        repository.save_data(data)

    @classmethod
    def load_from_file(cls, filepath: str, storage_type: str = "pickle") -> "Benchmark | None":
        """Load benchmark from storage, creating new instance with loaded config.

        Args:
            filepath: Path to pickle file (only used for pickle storage)
            storage_type: Either "pickle" or "database"
        """
        if storage_type == "pickle":
            repository = BenchmarkRepositoryFactory.create(
                storage_type="pickle",
                benchmark_path=Path(filepath)
            )
            data = repository.load_data()

            if data is None:
                return None

            # Create benchmark with loaded config
            benchmark = cls(
                config=data.config,
                num_games=data.num_games,
                base_seed=data.base_seed,
                benchmark_path=filepath,
                storage_type="pickle"
            )

            # Load the data
            benchmark.games = data.games
            benchmark.results = data.results

            return benchmark

        else:
            raise ValueError("load_from_file with database storage not yet implemented. Use load_from_database() instead.")

    def built_in_bots(self) -> dict[str, BenchmarkBotBase]:
        """Create instances of all built-in benchmark bots."""
        return {
            RandomBot.name: RandomBot(),
            LargestGroupBot.name: LargestGroupBot(),
            GreedySinglesBot.name: GreedySinglesBot(),
        }

    # === GAME MANAGEMENT ===

    def _generate_games(self) -> None:
        """Generate standardized games with reproducible initial states."""
        if len(self.games) == self.num_games:
            return  # Already generated

        rng = random.Random(self.base_seed)
        self.games = []

        for game_id in range(self.num_games):
            # Generate unique seed for each game
            game_seed = rng.randint(0, 2**31 - 1)

            # Create game with seeded board generation
            game = Game(self.config, seed=game_seed)

            snapshot = GameSnapshot(
                board=[row.copy() for row in game.board],
                config=self.config,
                seed=game_seed,
                game_id=game_id,
            )
            self.games.append(snapshot)

    def _load_existing_results(self) -> bool:
        """Load existing results for lazy loading, validating compatibility."""
        if not self.repository.data_exists():
            return False

        # Check compatibility first
        if not self.repository.is_compatible(self.config, self.base_seed):
            return False

        # Load compatible data
        data = self.repository.load_data()
        if data is None:
            return False

        # Load results and games
        self.results = data.results

        # Only load games if they don't exceed our current requirement
        if len(data.games) <= self.num_games:
            self.games = data.games
        else:
            # Use subset of games that match our requirement
            self.games = data.games[: self.num_games]

        return True

    # === BOT EXECUTION ===

    def _run_games_for_bot(
        self, bot: BenchmarkBotBase, missing_game_ids: list[int], bot_name: str
    ) -> list[BotPerformance]:
        """Run a bot on multiple games using configured execution strategy."""
        game_snapshots = [self.games[game_id] for game_id in missing_game_ids]
        return self.execution_strategy.run_games(bot, game_snapshots, bot_name)

    # === UTILITIES ===

    def _get_benchmark_path(self, benchmark_path: str | None) -> str:
        """Generate benchmark file path from configuration."""
        if benchmark_path is None:
            filename = f"benchmark_{self.config.num_rows}_{self.config.num_cols}_{self.config.num_colors}_{self.base_seed}.pkl"
            # Use absolute path relative to this file's location
            project_root = Path(
                __file__
            ).parent.parent.parent  # Up from samegamerl/evaluation/
            return str(
                project_root / "samegamerl" / "evaluation" / "benchmarks" / filename
            )

        # Handle custom benchmark paths
        if "/" in benchmark_path:
            return benchmark_path
        else:
            # Custom filename in default directory
            project_root = Path(__file__).parent.parent.parent
            return str(
                project_root
                / "samegamerl"
                / "evaluation"
                / "benchmarks"
                / benchmark_path
            )

    # === RAY MANAGEMENT ===

    def _initialize_ray(self) -> bool:
        """Initialize Ray if configured and available."""
        if not self.use_ray or self._ray_initialized:
            return self.use_ray

        try:
            if ray.is_initialized():
                # Ray is already initialized, use existing instance
                self._ray_initialized = True
                return True

            # Initialize Ray with optional CPU limit
            init_kwargs = {"ignore_reinit_error": True}
            if self.ray_num_cpus is not None:
                init_kwargs["num_cpus"] = self.ray_num_cpus

            ray.init(**init_kwargs)
            self._ray_initialized = True
            return True

        except Exception as e:
            print(f"Warning: Failed to initialize Ray: {e}")
            print("Falling back to sequential execution")
            self.use_ray = False
            return False

    def _cleanup_ray(self) -> None:
        """Clean up Ray resources if we initialized them."""
        if self._ray_initialized and ray.is_initialized():
            try:
                ray.shutdown()
                self._ray_initialized = False
            except Exception as e:
                print(f"Warning: Error during Ray cleanup: {e}")

    def __len__(self) -> int:
        """Number of games in benchmark."""
        return len(self.games)
