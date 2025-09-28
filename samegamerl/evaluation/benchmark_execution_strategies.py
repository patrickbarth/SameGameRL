"""Execution strategies for running benchmark bots on games."""

import copy
from abc import ABC, abstractmethod

from tqdm import tqdm

# Optional Ray import for parallelization
try:
    import ray

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

from samegamerl.game.game import Game
from samegamerl.agents.benchmark_bot_base import BenchmarkBotBase
from samegamerl.evaluation.benchmark_data import GameSnapshot, BotPerformance


class ExecutionStrategy(ABC):
    """Abstract base class for bot execution strategies."""

    @abstractmethod
    def run_games(
        self, bot: BenchmarkBotBase, game_snapshots: list[GameSnapshot], bot_name: str
    ) -> list[BotPerformance]:
        """Run a bot on multiple games and return performance results."""
        pass


class SequentialExecutionStrategy(ExecutionStrategy):
    """Execute bot games sequentially with progress tracking."""

    def run_games(
        self, bot: BenchmarkBotBase, game_snapshots: list[GameSnapshot], bot_name: str
    ) -> list[BotPerformance]:
        """Run games sequentially with progress bar."""
        results = []
        for game_snapshot in tqdm(game_snapshots, desc=f"Running {bot_name}"):
            performance = _run_bot_on_game_sequential(bot, game_snapshot)
            results.append(performance)
        return results


class ParallelExecutionStrategy(ExecutionStrategy):
    """Execute bot games in parallel using Ray."""

    def __init__(self, ray_num_cpus: int | None = None):
        self.ray_num_cpus = ray_num_cpus
        self._ray_initialized = False

    def run_games(
        self, bot: BenchmarkBotBase, game_snapshots: list[GameSnapshot], bot_name: str
    ) -> list[BotPerformance]:
        """Run games in parallel using Ray."""
        if not RAY_AVAILABLE:
            # Fallback to sequential if Ray not available
            sequential_strategy = SequentialExecutionStrategy()
            return sequential_strategy.run_games(bot, game_snapshots, bot_name)

        # Initialize Ray if needed
        self._initialize_ray()

        if not ray.is_initialized():
            # Fallback to sequential if Ray initialization failed
            sequential_strategy = SequentialExecutionStrategy()
            return sequential_strategy.run_games(bot, game_snapshots, bot_name)

        try:
            return self._run_games_parallel(bot, game_snapshots, bot_name)
        finally:
            self._cleanup_ray()

    def _run_games_parallel(
        self, bot: BenchmarkBotBase, game_snapshots: list[GameSnapshot], bot_name: str
    ) -> list[BotPerformance]:
        """Internal parallel execution logic."""
        print(f"  Using Ray parallel execution with {len(game_snapshots)} tasks")

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

    def _initialize_ray(self) -> bool:
        """Initialize Ray if configured and available."""
        if self._ray_initialized:
            return True

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
            return False

    def _cleanup_ray(self) -> None:
        """Clean up Ray resources if we initialized them."""
        if self._ray_initialized and ray.is_initialized():
            try:
                ray.shutdown()
                self._ray_initialized = False
            except Exception as e:
                print(f"Warning: Error during Ray cleanup: {e}")


class ExecutionStrategyFactory:
    """Factory for creating execution strategies."""

    @staticmethod
    def create_strategy(
        use_ray: bool, ray_num_cpus: int | None = None
    ) -> ExecutionStrategy:
        """Create appropriate execution strategy based on configuration."""
        if use_ray and RAY_AVAILABLE:
            return ParallelExecutionStrategy(ray_num_cpus)
        else:
            return SequentialExecutionStrategy()


# Helper functions for bot execution


def _run_bot_on_game_sequential(
    bot: BenchmarkBotBase, game_snapshot: GameSnapshot
) -> BotPerformance:
    """Sequential version of bot game execution."""
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
        """Ray remote function for parallel bot game execution."""
        return _run_bot_on_game_sequential(bot, game_snapshot)
