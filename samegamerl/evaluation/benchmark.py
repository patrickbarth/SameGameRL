"""Benchmark system for consistent agent evaluation across standardized games."""

import copy
import random
from pathlib import Path

from tqdm import tqdm

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
from samegamerl.evaluation.benchmark_repository import BenchmarkRepository



class Benchmark:
    """Unified benchmark system for evaluating agents consistently"""

    def __init__(
        self,
        config: GameConfig,
        num_games: int = 1000,
        base_seed: int = 42,
        benchmark_path: str | None = None,
        use_ray: bool = True,
        ray_num_cpus: int | None = None,
    ):
        self.num_games = num_games
        self.config = config
        self.base_seed = base_seed
        self.use_ray = use_ray and RAY_AVAILABLE  # Only use Ray if available
        self.ray_num_cpus = ray_num_cpus
        self._ray_initialized = False

        # Set up benchmark path and repository
        self.benchmark_path = Path(self._get_benchmark_path(benchmark_path))
        self.repository = BenchmarkRepository(self.benchmark_path)
        self.games: list[GameSnapshot] = []
        self.results: dict[str, list[BotPerformance]] = {}

    def get_game(self, game_id: int) -> GameSnapshot:
        """Get a specific game by ID"""
        if not self.games:
            self._generate_games()

        if 0 <= game_id < len(self.games):
            return self.games[game_id]
        raise IndexError(f"Game ID {game_id} out of range")

    def _get_benchmark_path(self, benchmark_path: str | None) -> str:
        if benchmark_path is None:
            filename = f"benchmark_{self.config.num_cols}_{self.config.num_rows}_{self.config.num_colors}_{self.base_seed}.pkl"
            return f"samegamerl/evaluation/benchmarks/{filename}"

        return (
            benchmark_path
            if "/" in benchmark_path
            else f"samegamerl/evaluation/benchmarks/{benchmark_path}"
        )

    def _is_valid_performance(self, result: BotPerformance) -> bool:
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

    def _initialize_ray(self) -> bool:
        """Initialize Ray if configured and available"""
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
        """Clean up Ray resources if we initialized them"""
        if self._ray_initialized and ray.is_initialized():
            try:
                ray.shutdown()
                self._ray_initialized = False
            except Exception as e:
                print(f"Warning: Error during Ray cleanup: {e}")

    def _generate_games(self) -> None:
        """Generate standardized games with reproducible initial states"""
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

    def built_in_bots(self) -> dict[str, BenchmarkBotBase]:
        """Create instances of all built-in benchmark bots"""
        return {
            RandomBot.name: RandomBot(),
            LargestGroupBot.name: LargestGroupBot(),
            GreedySinglesBot.name: GreedySinglesBot(),
        }

    def run_bots(
        self, bots: dict[str, BenchmarkBotBase] | None = None
    ) -> dict[str, list[BotPerformance]]:
        # Use all built-in bots if none specified
        if bots is None:
            bots = self.built_in_bots()

        # Initialize Ray if configured
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
                missing_game_ids = self._determine_missing_games(bot_name)

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

                # Run games in parallel or sequential based on configuration
                new_results = self._run_games_for_bot(
                    bot_instance, missing_game_ids, bot_name
                )

                # Merge new results with existing validated results
                self._merge_results(bot_name, new_results)
                results[bot_name] = self.results[bot_name]

            # Save updated results if any new computation was done
            if any_new_computation:
                self.save()

            return results

        finally:
            # Clean up Ray if we initialized it
            self._cleanup_ray()

    def _run_games_for_bot(
        self, bot: BenchmarkBotBase, missing_game_ids: list[int], bot_name: str
    ) -> list[BotPerformance]:
        """Run a bot on multiple games, using parallel or sequential execution"""
        game_snapshots = [self.games[game_id] for game_id in missing_game_ids]

        # Use parallel execution if Ray is available and configured
        if self.use_ray and RAY_AVAILABLE and ray.is_initialized():
            return self._run_games_parallel(bot, game_snapshots, bot_name)
        else:
            return self._run_games_sequential(bot, game_snapshots, bot_name)

    def _run_games_sequential(
        self, bot: BenchmarkBotBase, game_snapshots: list[GameSnapshot], bot_name: str
    ) -> list[BotPerformance]:
        """Run games sequentially with progress bar"""

        results = []
        for game_snapshot in tqdm(game_snapshots, desc=f"Running {bot_name}"):
            performance = _run_bot_on_game_sequential(bot, game_snapshot)
            results.append(performance)
        return results

    def _run_games_parallel(
        self, bot: BenchmarkBotBase, game_snapshots: list[GameSnapshot], bot_name: str
    ) -> list[BotPerformance]:
        """Run games in parallel using Ray"""

        print(f"  Using Ray parallel execution with {len(game_snapshots)} tasks")

        if RAY_AVAILABLE:
            # Put bot in object store once, share across tasks
            bot_ref = ray.put(bot)

            # Submit all tasks to Ray
            future_results = [
                _run_bot_on_game_parallel.remote(bot_ref, game_snapshot)
                for game_snapshot in game_snapshots
            ]

            # Collect results maintaining order using game_id
            results: list[BotPerformance | None] = [None] * len(game_snapshots)
            game_id_to_index = {gs.game_id: i for i, gs in enumerate(game_snapshots)}
            remaining_futures = future_results.copy()

            with tqdm(
                total=len(future_results), desc=f"Running {bot_name} (parallel)"
            ) as pbar:
                while remaining_futures:
                    try:
                        ready, remaining_futures = ray.wait(
                            remaining_futures, num_returns=1
                        )
                        for future in ready:
                            result = ray.get(future)
                            correct_index = game_id_to_index[result.game_id]
                            results[correct_index] = result
                        pbar.update(len(ready))
                    except Exception as e:
                        print(f"Warning: Task failed during {bot_name} execution: {e}")
                        pbar.update(1)

            # Filter out None values in case of failures
            return [r for r in results if r is not None]
        else:
            # Fallback to sequential if Ray not available
            return self._run_games_sequential(bot, game_snapshots, bot_name)

    def save(self, filepath: str | None = None) -> None:
        """Save benchmark data to disk"""
        if filepath is None:
            # Use default repository
            repository = self.repository
        else:
            # Use custom filepath
            repository = BenchmarkRepository(Path(filepath))

        data = BenchmarkData(
            games=self.games,
            results=self.results,
            config=self.config,
            num_games=self.num_games,
            base_seed=self.base_seed,
        )
        repository.save_data(data)

    @classmethod
    def load_from_file(cls, filepath: str) -> "Benchmark | None":
        """Load benchmark from file, creating new instance with loaded config"""
        repository = BenchmarkRepository(Path(filepath))
        data = repository.load_data()

        if data is None:
            return None

        # Create benchmark with loaded config
        benchmark = cls(
            config=data.config,
            num_games=data.num_games,
            base_seed=data.base_seed,
            benchmark_path=filepath,
        )

        # Load the data
        benchmark.games = data.games
        benchmark.results = data.results

        return benchmark

    def _load_existing_results(self) -> bool:
        """Load existing results for lazy loading, validating compatibility"""
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

    def _validate_existing_results(
        self, bot_name: str, results: list[BotPerformance]
    ) -> int:
        """Validate existing results for a bot and return count of valid consecutive results"""
        return self.repository.validate_results(bot_name, results)

    def _determine_missing_games(self, bot_name: str) -> list[int]:
        """Determine which games need to be computed for a bot"""
        return self.repository.determine_missing_games(bot_name, self.results, self.num_games)

    def _merge_results(self, bot_name: str, new_results: list[BotPerformance]) -> None:
        """Merge new results with existing validated results for a bot"""
        existing_results = self.results.get(bot_name, [])
        merged_results = self.repository.merge_results(
            existing_results, new_results, self.num_games, bot_name
        )
        self.results[bot_name] = merged_results

    def __len__(self) -> int:
        """Number of games in benchmark"""
        return len(self.games)


# Helper functions for bot execution (sequential and parallel)


def _run_bot_on_game_sequential(
    bot: BenchmarkBotBase, game_snapshot: GameSnapshot
) -> BotPerformance:
    """Sequential version of bot game execution"""
    # Create fresh game instance from snapshot
    game = Game(game_snapshot.config)
    game.set_board(copy.deepcopy(game_snapshot.board))

    initial_tiles = game.left
    moves_made = 0
    max_moves = 500  # Safety limit

    # Play the game until completion or no valid moves
    while moves_made < max_moves:
        current_board = game.get_board()
        action = bot.select_action(current_board)

        if action is None:
            break

        # Make the move
        game.move(action)
        moves_made += 1

        if game.left == 0:
            break

    # Calculate final metrics
    tiles_cleared = initial_tiles - game.left
    singles_remaining = game.get_singles()
    completed = game.left == 0

    return BotPerformance(
        bot_name=bot.name,
        game_id=game_snapshot.game_id,
        tiles_cleared=tiles_cleared,
        singles_remaining=singles_remaining,
        moves_made=moves_made,
        completed=completed,
    )


# Ray remote function for parallel execution
if RAY_AVAILABLE:

    @ray.remote
    def _run_bot_on_game_parallel(
        bot: BenchmarkBotBase, game_snapshot: GameSnapshot
    ) -> BotPerformance:
        """Ray remote function for parallel bot game execution"""
        return _run_bot_on_game_sequential(bot, game_snapshot)
