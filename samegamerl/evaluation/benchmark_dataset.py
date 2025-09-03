"""
Benchmark dataset system for consistent agent evaluation.

Creates and manages standardized game sets for comparing agent performance
across identical initial board states.
"""

import pickle
from dataclasses import dataclass
from pathlib import Path
import random

from samegamerl.game.game import Game
from samegamerl.game.game_config import GameConfig, GameFactory


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


class BenchmarkDataset:
    """Manages a standardized set of games for consistent agent evaluation"""

    def __init__(self, dataset_path: str = "benchmark_games.pkl"):
        # Default to datasets folder if just filename given
        if "/" not in dataset_path:
            dataset_path = f"samegamerl/evaluation/datasets/{dataset_path}"
        self.dataset_path = Path(dataset_path)
        self.games: list[GameSnapshot] = []
        self.results: dict[str, list[BotPerformance]] = {}

    def generate_games(
        self,
        num_games: int = 1000,
        config: GameConfig | None = None,
        base_seed: int = 42,
    ) -> None:
        """Generate a standardized set of games with reproducible initial states"""

        if config is None:
            config = GameFactory.medium()

        # Generate filename based on config and preserve datasets directory
        filename = (
            "benchmark_"
            + str(config.num_cols)
            + "_"
            + str(config.num_rows)
            + "_"
            + str(config.num_colors)
            + "_"
            + str(num_games)
        )
        self.dataset_path = Path("samegamerl/evaluation/datasets") / filename

        # Use deterministic seeding for reproducible game generation
        rng = random.Random(base_seed)

        self.games = []
        for game_id in range(num_games):
            # Generate unique seed for each game
            game_seed = rng.randint(0, 2**31 - 1)

            # Create game with specific seed
            game = Game(config)
            # Override random board generation with seeded version
            game_rng = random.Random(game_seed)
            for row in range(config.num_rows):
                for col in range(config.num_cols):
                    game.board[row][col] = game_rng.randint(1, config.num_colors - 1)

            snapshot = GameSnapshot(
                board=[row.copy() for row in game.board],
                config=config,
                seed=game_seed,
                game_id=game_id,
            )
            self.games.append(snapshot)

    def save_dataset(self, filepath: str | None = None) -> None:
        """Save dataset to disk using pickle for efficiency"""
        if filepath is None:
            filepath = str(self.dataset_path)

        # Ensure parent directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        data = {"games": self.games, "results": self.results}

        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    def load_dataset(self, filepath: str | None = None) -> bool:
        """Load dataset from disk"""
        if filepath is None:
            filepath = str(self.dataset_path)

        if not Path(filepath).exists():
            return False

        try:
            with open(filepath, "rb") as f:
                data = pickle.load(f)
                self.games = data["games"]
                self.results = data.get("results", {})
            return True
        except Exception:
            return False

    def add_bot_results(
        self, bot_name: str, performances: list[BotPerformance]
    ) -> None:
        """Add performance results for a bot across all games"""
        self.results[bot_name] = performances

    def get_game(self, game_id: int) -> GameSnapshot:
        """Get a specific game by ID"""
        if 0 <= game_id < len(self.games):
            return self.games[game_id]
        raise IndexError(f"Game ID {game_id} out of range")

    def get_bot_results(self, bot_name: str) -> list[BotPerformance]:
        """Get all results for a specific bot"""
        return self.results.get(bot_name, [])

    def __len__(self) -> int:
        """Number of games in dataset"""
        return len(self.games)
