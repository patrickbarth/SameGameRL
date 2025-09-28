"""Data classes for benchmark system."""

from dataclasses import dataclass

from samegamerl.game.game_config import GameConfig


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


@dataclass
class BenchmarkData:
    """Complete benchmark dataset including games and results"""
    games: list[GameSnapshot]
    results: dict[str, list[BotPerformance]]
    config: GameConfig
    num_games: int
    base_seed: int