"""
Benchmark system for agent evaluation.

Provides:
- Game creation: enables all agents and bots to be evaluated against the same games
- Comparison and evaluation: agents and bots can be compared in their performance on the test games
- Loading and saving: loads existing games and performance data in the benchmarks folder, so that they
  do not have to be created again with every execution

"""

import copy
import pickle
import random
import statistics
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from samegamerl.game.game import Game
from samegamerl.game.game_config import GameConfig
from samegamerl.agents.benchmark_bot_base import BenchmarkBotBase
from samegamerl.agents.random_bot import RandomBot
from samegamerl.agents.largest_group_bot import LargestGroupBot
from samegamerl.agents.greedy_singles_bot import GreedySinglesBot


@dataclass
class GameSnapshot:
    """Immutable representation of a game's initial state"""

    board: list[list[int]]
    config: GameConfig
    seed: int
    game_id: int


@dataclass
class BotPerformance:
    """Performance metrics for a single bot on a single game"""

    bot_name: str
    game_id: int
    tiles_cleared: int
    singles_remaining: int
    moves_made: int
    completed: bool


class Benchmark:
    """Unified benchmark system for evaluating agents consistently"""

    def __init__(
        self,
        config: GameConfig,
        num_games: int = 1000,
        base_seed: int = 42,
        benchmark_path: str | None = None,
    ):
        """
        Initialize benchmark system.

        Args:
            config: Game configuration
            num_games: Number of games to generate for benchmarking
            base_seed: Base seed for reproducible game generation
            benchmark_path: Path for saving/loading benchmark data
        """
        self.num_games = num_games
        self.config = config
        self.base_seed = base_seed

        # Set up benchmark path
        if benchmark_path is None:
            filename = f"benchmark_{self.config.num_cols}_{self.config.num_rows}_{self.config.num_colors}_{self.base_seed}.pkl"
            benchmark_path = f"samegamerl/evaluation/benchmarks/{filename}"
        elif "/" not in benchmark_path:
            benchmark_path = f"samegamerl/evaluation/benchmarks/{benchmark_path}"

        self.benchmark_path = Path(benchmark_path)
        self.games: list[GameSnapshot] = []
        self.results: dict[str, list[BotPerformance]] = {}

        # Available built-in bots
        self._available_bots = {
            RandomBot.name: RandomBot,
            LargestGroupBot.name: LargestGroupBot,
            GreedySinglesBot.name: GreedySinglesBot,
        }

    def _generate_games(self) -> None:
        """Generate standardized games with reproducible initial states"""
        if self.games and len(self.games) == self.num_games:
            return  # Already generated

        rng = random.Random(self.base_seed)
        self.games = []

        for game_id in range(self.num_games):
            # Generate unique seed for each game
            game_seed = rng.randint(0, 2**31 - 1)

            # Create game with specific seed
            game = Game(self.config)
            # Override random board generation with seeded version
            game_rng = random.Random(game_seed)
            for row in range(self.config.num_rows):
                for col in range(self.config.num_cols):
                    game.board[row][col] = game_rng.randint(
                        1, self.config.num_colors - 1
                    )

            snapshot = GameSnapshot(
                board=[row.copy() for row in game.board],
                config=self.config,
                seed=game_seed,
                game_id=game_id,
            )
            self.games.append(snapshot)

    def run_bots(
        self, bots: dict[str, BenchmarkBotBase] | list[str]
    ) -> dict[str, list[BotPerformance]]:
        """
        Run bots against all games and return performance results.

        Args:
            bots: Dictionary of {name: bot_instance} or list of built-in bot names

        Returns:
            Dictionary mapping bot names to their performance results
        """
        # Ensure games are generated
        self._generate_games()

        # Handle different input formats
        if isinstance(bots, list):
            # Convert list of bot names to instances
            bot_instances = {}
            for bot_name in bots:
                if bot_name not in self._available_bots:
                    raise ValueError(
                        f"Unknown bot: {bot_name}. Available: {list(self._available_bots.keys())}"
                    )
                bot_instances[bot_name] = self._available_bots[bot_name]()
            bots = bot_instances

        results = {}

        for bot_name, bot_instance in bots.items():
            bot_results = []
            print(f"Evaluating {bot_name} against {len(self.games)} games...")

            # Set the benchmark name on the bot instance
            bot_instance._benchmark_name = bot_name

            for game_snapshot in tqdm(self.games, desc=f"Running {bot_name}"):
                performance = self._run_bot_on_game(bot_instance, game_snapshot)
                bot_results.append(performance)

            results[bot_name] = bot_results
            self.results[bot_name] = bot_results

        return results

    def _run_bot_on_game(
        self, bot: BenchmarkBotBase, game_snapshot: GameSnapshot
    ) -> BotPerformance:
        """Run a single bot against a single game and return performance metrics"""
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
            bot_name=getattr(bot, "_benchmark_name", bot.__class__.__name__),
            game_id=game_snapshot.game_id,
            tiles_cleared=tiles_cleared,
            singles_remaining=singles_remaining,
            moves_made=moves_made,
            completed=completed,
        )

    def compare(
        self, bot_names: list[str] | None = None
    ) -> dict[str, dict[str, object]]:
        """
        Compare performance statistics across bots.

        Args:
            bot_names: List of bot names to compare (defaults to all available results)

        Returns:
            Dictionary mapping bot names to their summary statistics
        """
        if bot_names is None:
            bot_names = list(self.results.keys())

        comparison = {}
        for bot_name in bot_names:
            if bot_name in self.results:
                comparison[bot_name] = self._get_bot_summary(bot_name)

        return comparison

    def _get_bot_summary(self, bot_name: str) -> dict[str, object]:
        """Get summary statistics for a bot's performance"""
        results = self.results.get(bot_name, [])
        if not results:
            return {}

        tiles_cleared = [r.tiles_cleared for r in results]
        completion_rate = sum(1 for r in results if r.completed) / len(results)
        avg_moves = sum(r.moves_made for r in results) / len(results)
        avg_singles = sum(r.singles_remaining for r in results) / len(results)

        return {
            "total_games": len(results),
            "completion_rate": completion_rate,
            "avg_tiles_cleared": sum(tiles_cleared) / len(tiles_cleared),
            "avg_moves_made": avg_moves,
            "avg_singles_remaining": avg_singles,
        }

    def head_to_head_analysis(
        self, bot1: str, bot2: str
    ) -> dict[str, float | str | int]:
        """Compare two bots game-by-game"""
        results1 = self.results.get(bot1, [])
        results2 = self.results.get(bot2, [])

        if not results1 or not results2:
            return {"error": "One or both bots not found in results"}

        if len(results1) != len(results2):
            return {"error": "Bots have different number of games"}

        wins_bot1 = 0
        wins_bot2 = 0
        ties = 0
        score_differences = []

        for r1, r2 in zip(results1, results2):
            if r1.game_id != r2.game_id:
                return {"error": "Game ID mismatch between bot results"}

            # Compare based on tiles cleared (primary) and completion (secondary)
            score1 = r1.tiles_cleared + (1000 if r1.completed else 0)
            score2 = r2.tiles_cleared + (1000 if r2.completed else 0)

            if score1 > score2:
                wins_bot1 += 1
            elif score1 < score2:
                wins_bot2 += 1
            else:
                ties += 1

            score_differences.append(score1 - score2)

        return {
            "bot1": bot1,
            "bot2": bot2,
            "wins_bot1": wins_bot1,
            "wins_bot2": wins_bot2,
            "ties": ties,
            "win_rate_bot1": wins_bot1 / len(results1),
            "win_rate_bot2": wins_bot2 / len(results2),
            "avg_performance_difference": statistics.mean(score_differences),
            "median_performance_difference": statistics.median(score_differences),
        }

    def find_interesting_games(self) -> list[int]:
        """Find the top 10% of games with highest performance variance between bots"""
        bot_names = list(self.results.keys())
        if len(bot_names) < 2:
            return []

        game_variances = []

        for game_id in range(len(self.games)):
            tiles_cleared_scores = []
            for bot_name in bot_names:
                results = self.results[bot_name]
                if game_id < len(results):
                    tiles_cleared_scores.append(results[game_id].tiles_cleared)

            if len(tiles_cleared_scores) >= 2:
                variance = statistics.variance(tiles_cleared_scores)
                game_variances.append((game_id, variance))

        # Sort by variance (highest first) and return top 10%
        game_variances.sort(key=lambda x: x[1], reverse=True)
        top_10_percent = max(1, len(game_variances) // 10)

        return [game_id for game_id, _ in game_variances[:top_10_percent]]

    def get_available_bots(self) -> list[str]:
        """Get list of available built-in bot names"""
        return list(self._available_bots.keys())

    def get_game(self, game_id: int) -> GameSnapshot:
        """Get a specific game by ID"""
        if not self.games:
            self._generate_games()

        if 0 <= game_id < len(self.games):
            return self.games[game_id]
        raise IndexError(f"Game ID {game_id} out of range")

    def save(self, filepath: str | None = None) -> None:
        """Save benchmark data to disk"""
        if filepath is None:
            filepath = str(self.benchmark_path)

        # Ensure parent directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        data = {
            "games": self.games,
            "results": self.results,
            "config": self.config,
            "num_games": self.num_games,
            "base_seed": self.base_seed,
        }

        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load_from_file(cls, filepath: str) -> "Benchmark | None":
        """Load benchmark from file, creating new instance with loaded config"""
        if not Path(filepath).exists():
            return None

        try:
            with open(filepath, "rb") as f:
                data = pickle.load(f)

            # Create benchmark with loaded config
            benchmark = cls(
                config=data["config"],
                num_games=data.get("num_games", 1000),
                base_seed=data.get("base_seed", 42),
            )

            # Load the data
            benchmark.games = data["games"]
            benchmark.results = data.get("results", {})

            return benchmark
        except Exception:
            return None

    def load(self, filepath: str | None = None) -> bool:
        """Load benchmark data from disk into existing instance"""
        if filepath is None:
            filepath = str(self.benchmark_path)

        if not Path(filepath).exists():
            return False

        try:
            with open(filepath, "rb") as f:
                data = pickle.load(f)
                self.games = data["games"]
                self.results = data.get("results", {})
                self.config = data["config"]
                self.num_games = data.get("num_games", self.num_games)
                self.base_seed = data.get("base_seed", self.base_seed)
            return True
        except Exception:
            return False

    def plot_comparison(
        self,
        bot_names: list[str] | None = None,
        show: bool = True,
        save_path: str | None = None,
    ) -> None:
        """Plot comparison of bot performance (delegates to plotting module)"""
        from samegamerl.evaluation.benchmark_plotting import plot_comparison

        plot_comparison(self, bot_names, show, save_path)

    def plot_head_to_head(
        self, bot1: str, bot2: str, show: bool = True, save_path: str | None = None
    ) -> None:
        """Plot head-to-head comparison between two bots (delegates to plotting module)"""
        from samegamerl.evaluation.benchmark_plotting import plot_head_to_head

        plot_head_to_head(self, bot1, bot2, show, save_path)

    def generate_report(
        self, output_dir: str = "benchmark_report", bot_names: list[str] | None = None
    ) -> None:
        """Generate complete benchmark report with plots and statistics"""
        from samegamerl.evaluation.benchmark_plotting import generate_benchmark_report

        generate_benchmark_report(self, output_dir, bot_names)

    def __len__(self) -> int:
        """Number of games in benchmark"""
        return len(self.games)
