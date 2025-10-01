#!/usr/bin/env python3
"""
Simple benchmark entry point scripts.

Provides two essential benchmarking functions:
1. evaluate_agent() - Test a custom agent against built-in bots
2. benchmark_builtin_bots() - Get baseline performance of built-in bots
"""

from samegamerl.evaluation.benchmark import Benchmark
from samegamerl.game.game_config import GameConfig
from samegamerl.agents.base_agent import BaseAgent
from samegamerl.evaluation.benchmark_data import BotPerformance


def benchmark_agent(
    agent: BaseAgent, config: GameConfig, num_games: int, storage_type: str = "pickle"
) -> None:
    """
    Evaluate a custom agent against built-in bots for comparison.

    Args:
        agent: Custom agent instance to evaluate
        config: Game configuration to use
        num_games: Number of games to play
        storage_type: Storage backend - "pickle" (default) or "database"
    """
    print(f"Evaluating Custom Agent")
    print("=" * 40)
    print(f"Games: {num_games}")
    print()

    benchmark = Benchmark(config=config, num_games=num_games, storage_type=storage_type)

    # Run built-in bots for comparison
    print("Running built-in bots...")
    builtin_results = benchmark.run_bots(benchmark.built_in_bots())

    # Run custom agent
    print("Running agent...")
    custom_results = {agent.model_name: benchmark.evaluate_agent(agent)}

    # Compute comparison stats
    all_results = {**builtin_results, **custom_results}
    comparison = _compute_stats(all_results)

    # Display results
    print("\nPerformance Results:")
    print("-" * 30)

    # Sort by completion rate for ranking
    sorted_bots = sorted(
        comparison.items(), key=lambda x: x[1]["completion_rate"], reverse=True
    )

    for i, (bot_name, stats) in enumerate(sorted_bots, 1):
        print(f"{i}. {bot_name}")
        print(f"   Completion rate: {stats['completion_rate']:.1%}")
        print(f"   Avg tiles cleared: {stats['avg_tiles_cleared']:.1f}")
        print(f"   Avg moves made: {stats['avg_moves_made']:.1f}")
        print(f"   Avg singles remaining: {stats['avg_singles_remaining']:.1f}")
        print()


def benchmark_builtin_bots(
    config: GameConfig,
    num_games: int,
    verbose: bool = True,
    storage_type: str = "pickle",
) -> dict[str, dict[str, float]]:
    """
    Benchmark all built-in bots to get baseline performance.

    Args:
        config: Game configuration to use
        num_games: Number of games to play
        verbose: Whether to print detailed results
        storage_type: Storage backend - "pickle" (default) or "database"
    """
    if verbose:
        print("Built-in Bots Benchmark")
        print("=" * 40)
        print(
            f"Config: {config.num_rows}x{config.num_cols}, {config.num_colors} colors"
        )
        print(f"Games: {num_games}")
        print(f"Storage: {storage_type}")
        print()

    benchmark = Benchmark(config=config, num_games=num_games, storage_type=storage_type)

    if verbose:
        print("Running built-in bots...")

    results = benchmark.run_bots(benchmark.built_in_bots())

    # Compute comparison stats
    comparison = _compute_stats(results)

    # Display results
    if verbose:
        print("\nBuilt-in Bot Performance:")
        print("-" * 30)

        # Sort by completion rate
        sorted_bots = sorted(
            comparison.items(), key=lambda x: x[1]["completion_rate"], reverse=True
        )

        for i, (bot_name, stats) in enumerate(sorted_bots, 1):
            print(f"{i}. {bot_name}:")
            print(f"   Completion rate: {stats['completion_rate']:.1%}")
            print(f"   Avg tiles cleared: {stats['avg_tiles_cleared']:.1f}")
            print(f"   Avg moves made: {stats['avg_moves_made']:.1f}")
            print(f"   Avg singles remaining: {stats['avg_singles_remaining']:.1f}")
            print()

    return comparison


def _compute_stats(
    results: dict[str, list[BotPerformance]],
) -> dict[str, dict[str, float]]:
    """Compute performance statistics for each bot."""
    stats = {}

    for bot_name, performances in results.items():
        if not performances:
            continue

        total_games = len(performances)
        completed_games = sum(1 for p in performances if p.completed)
        total_tiles_cleared = sum(p.tiles_cleared for p in performances)
        total_moves_made = sum(p.moves_made for p in performances)
        total_singles_remaining = sum(p.singles_remaining for p in performances)

        stats[bot_name] = {
            "completion_rate": (
                completed_games / total_games if total_games > 0 else 0.0
            ),
            "avg_tiles_cleared": (
                total_tiles_cleared / total_games if total_games > 0 else 0.0
            ),
            "avg_moves_made": (
                total_moves_made / total_games if total_games > 0 else 0.0
            ),
            "avg_singles_remaining": (
                total_singles_remaining / total_games if total_games > 0 else 0.0
            ),
        }

    return stats
