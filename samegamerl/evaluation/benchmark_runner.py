"""
Benchmark runner for consistent agent evaluation.

Runs benchmark bots against standardized game datasets and collects
detailed performance metrics for comparison.
"""

from tqdm import tqdm
from typing import Any
import copy

from samegamerl.game.game import Game
from samegamerl.agents.benchmark_bot_base import BenchmarkBotBase
from samegamerl.agents.random_bot import RandomBot
from samegamerl.agents.largest_group_bot import LargestGroupBot
from samegamerl.agents.greedy_singles_bot import GreedySinglesBot
from samegamerl.evaluation.benchmark_dataset import (
    BenchmarkDataset,
    GameSnapshot,
    BotPerformance,
)


class BenchmarkRunner:
    """Runs benchmark bots against standardized game datasets"""

    def __init__(self, dataset: BenchmarkDataset):
        self.dataset = dataset
        self.available_bots = {
            "RandomBot": RandomBot,
            "LargestGroupBot": LargestGroupBot,
            "GreedySinglesBot": GreedySinglesBot,
        }

    def run_bot_on_game(
        self, bot: BenchmarkBotBase, game_snapshot: GameSnapshot
    ) -> BotPerformance:
        """Run a single bot against a single game and return performance metrics"""
        # Create fresh game instance from snapshot
        game = Game(game_snapshot.config)
        game.set_board(copy.deepcopy(game_snapshot.board))

        initial_tiles = game.left
        moves_made = 0
        max_moves = 500  # Safety limit to prevent infinite loops

        # Play the game until completion or no valid moves
        while moves_made < max_moves:
            current_board = game.get_board()
            action = bot.select_action(current_board)

            if action is None:
                # No valid moves available
                break

            # Make the move
            game.move(action)
            moves_made += 1

            # Check if game is complete
            if game.left == 0:
                break

        # Calculate final metrics
        tiles_cleared = initial_tiles - game.left
        singles_remaining = game.get_singles()
        completed = game.left == 0

        return BotPerformance(
            bot_name=bot.__class__.__name__,
            game_id=game_snapshot.game_id,
            tiles_cleared=tiles_cleared,
            singles_remaining=singles_remaining,
            moves_made=moves_made,
            completed=completed,
        )

    def evaluate_bot(
        self, bot_name: str, bot_instance: BenchmarkBotBase | None = None
    ) -> list[BotPerformance]:
        """Evaluate a single bot against all games in the dataset"""
        if bot_instance is None:
            if bot_name not in self.available_bots:
                raise ValueError(
                    f"Unknown bot: {bot_name}. Available: {list(self.available_bots.keys())}"
                )
            bot_instance = self.available_bots[bot_name]()

        assert bot_instance is not None

        results = []

        print(f"Evaluating {bot_name} against {len(self.dataset)} games...")
        for game_snapshot in tqdm(self.dataset.games, desc=f"Running {bot_name}"):
            performance = self.run_bot_on_game(bot_instance, game_snapshot)
            results.append(performance)

        return results

    def run_full_benchmark(self) -> dict[str, list[BotPerformance]]:
        """Run all available benchmark bots against the dataset"""
        all_results = {}

        for bot_name in self.available_bots.keys():
            bot_results = self.evaluate_bot(bot_name)
            all_results[bot_name] = bot_results
            self.dataset.add_bot_results(bot_name, bot_results)

        return all_results

    def evaluate_custom_bot(
        self, bot_name: str, bot_instance: BenchmarkBotBase
    ) -> list[BotPerformance]:
        """Evaluate a custom bot instance (e.g., DQN agent) against the dataset"""
        results = self.evaluate_bot(bot_name, bot_instance)
        self.dataset.add_bot_results(bot_name, results)
        return results

    def get_bot_summary(self, bot_name: str) -> dict[str, Any]:
        """Get summary statistics for a bot's performance"""
        results = self.dataset.get_bot_results(bot_name)
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

    def compare_bots(
        self, bot_names: list[str] | None = None
    ) -> dict[str, dict[str, Any]]:
        """Compare multiple bots' performance"""
        if bot_names is None:
            bot_names = list(self.dataset.results.keys())

        comparison = {}
        for bot_name in bot_names:
            if bot_name in self.dataset.results:
                comparison[bot_name] = self.get_bot_summary(bot_name)

        return comparison
