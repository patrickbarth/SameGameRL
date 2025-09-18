#!/usr/bin/env python3
"""
Simple benchmark entry point scripts.

Provides easy-to-use functions for common benchmarking tasks, replacing the 
complex benchmark_creation.py with straightforward examples.
"""

from pathlib import Path
from samegamerl.evaluation.benchmark import Benchmark
from samegamerl.game.game_config import GameFactory, GameConfig
from samegamerl.agents.benchmark_bot_base import BenchmarkBotBase


def run_standard_benchmark(
    num_games: int = 1000,
    config: GameConfig | None = None,
    save_results: bool = True,
    generate_plots: bool = True,
    output_dir: str = "benchmark_results"
) -> dict[str, object]:
    """
    Run standard benchmark with all built-in bots.
    
    Args:
        num_games: Number of games to generate for benchmarking
        config: Game configuration (defaults to large: 15x15, 5 colors)
        save_results: Whether to save results to disk
        generate_plots: Whether to generate visualization plots
        output_dir: Directory for saving results and plots
        
    Returns:
        Dictionary containing benchmark results and statistics
    """
    print("SameGameRL Standard Benchmark")
    print("=" * 50)
    
    # Configuration
    if config is None:
        config = GameFactory.large()  # 15x15 board with 5 colors
        
    print(f"Configuration: {config.num_rows}x{config.num_cols} board, {config.num_colors} colors")
    print(f"Number of games: {num_games}")
    print()
    
    # Create benchmark and run all built-in bots
    benchmark = Benchmark(config=config, num_games=num_games)
    
    print("Running benchmark with built-in bots...")
    results = benchmark.run_bots(["RandomBot", "LargestGroupBot", "GreedySinglesBot"])
    
    # Generate comparison
    comparison = benchmark.compare()
    
    # Display results
    print("\nPerformance Summary:")
    print("-" * 30)
    for bot_name in sorted(comparison.keys()):
        stats = comparison[bot_name]
        print(f"\n{bot_name}:")
        print(f"  Completion rate: {stats['completion_rate']:.1%}")
        print(f"  Avg tiles cleared: {stats['avg_tiles_cleared']:.1f}")
        print(f"  Avg moves made: {stats['avg_moves_made']:.1f}")
        print(f"  Avg singles remaining: {stats['avg_singles_remaining']:.1f}")
    
    # Save results
    if save_results:
        benchmark.save()
        print(f"\nResults saved to: {benchmark.benchmark_path}")
    
    # Generate plots and report
    if generate_plots:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        benchmark.generate_report(output_dir)
        print(f"Plots and report saved to: {output_dir}/")
    
    print("\nStandard benchmark complete!")
    
    return {
        "benchmark": benchmark,
        "results": results,
        "comparison": comparison,
        "num_games": num_games,
        "config": config
    }


def evaluate_custom_agent(
    agent: BenchmarkBotBase,
    agent_name: str,
    num_games: int = 1000,
    config: GameConfig | None = None,
    compare_with_builtin: bool = True,
    save_results: bool = True,
    generate_plots: bool = True,
    output_dir: str | None = None
) -> dict[str, object]:
    """
    Evaluate a custom trained agent against the benchmark.
    
    Args:
        agent: Custom agent instance to evaluate
        agent_name: Name for the custom agent
        num_games: Number of games to evaluate on
        config: Game configuration (defaults to medium: 8x8, 3 colors)
        compare_with_builtin: Whether to compare against built-in bots
        save_results: Whether to save results to disk
        generate_plots: Whether to generate visualization plots
        output_dir: Directory for saving results (defaults to agent name)
        
    Returns:
        Dictionary containing evaluation results and comparisons
    """
    print(f"SameGameRL Custom Agent Evaluation: {agent_name}")
    print("=" * 50)
    
    # Configuration
    if config is None:
        config = GameFactory.medium()  # 8x8 board with 3 colors
        
    if output_dir is None:
        output_dir = f"{agent_name}_evaluation"
        
    print(f"Configuration: {config.num_rows}x{config.num_cols} board, {config.num_colors} colors")
    print(f"Number of games: {num_games}")
    print()
    
    # Create benchmark
    benchmark = Benchmark(config=config, num_games=num_games)
    
    # Evaluate custom agent
    print(f"Evaluating {agent_name}...")
    bots_to_run = {agent_name: agent}
    
    # Add built-in bots for comparison if requested
    if compare_with_builtin:
        print("Including built-in bots for comparison...")
        builtin_results = benchmark.run_bots(["RandomBot", "LargestGroupBot", "GreedySinglesBot"])
        
    # Run custom agent
    custom_results = benchmark.run_bots(bots_to_run)
    
    # Combine results
    all_results = {**builtin_results} if compare_with_builtin else {}
    all_results.update(custom_results)
    
    # Generate comparison
    comparison = benchmark.compare()
    
    # Display results
    print(f"\n{agent_name} Performance:")
    print("-" * 30)
    if agent_name in comparison:
        stats = comparison[agent_name]
        print(f"  Completion rate: {stats['completion_rate']:.1%}")
        print(f"  Avg tiles cleared: {stats['avg_tiles_cleared']:.1f}")
        print(f"  Avg moves made: {stats['avg_moves_made']:.1f}")
        print(f"  Avg singles remaining: {stats['avg_singles_remaining']:.1f}")
    
    if compare_with_builtin:
        print("\nComparison with built-in bots:")
        print("-" * 30)
        # Sort by completion rate
        sorted_bots = sorted(comparison.items(), key=lambda x: x[1]['completion_rate'], reverse=True)
        for i, (bot_name, stats) in enumerate(sorted_bots, 1):
            marker = " <- Your agent" if bot_name == agent_name else ""
            print(f"{i}. {bot_name}: {stats['completion_rate']:.1%} completion{marker}")
    
    # Save results
    if save_results:
        benchmark.save()
        print(f"\nResults saved to: {benchmark.benchmark_path}")
    
    # Generate plots
    if generate_plots:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        benchmark.generate_report(output_dir)
        print(f"Evaluation report saved to: {output_dir}/")
    
    print(f"\n{agent_name} evaluation complete!")
    
    return {
        "benchmark": benchmark,
        "results": all_results,
        "comparison": comparison,
        "agent_name": agent_name,
        "custom_stats": comparison.get(agent_name, {}),
        "num_games": num_games,
        "config": config
    }


def quick_comparison(
    bot1_name: str, 
    bot1: BenchmarkBotBase,
    bot2_name: str,
    bot2: BenchmarkBotBase,
    num_games: int = 100,
    config: GameConfig | None = None
) -> dict[str, object]:
    """
    Quick head-to-head comparison between two agents.
    
    Args:
        bot1_name: Name for first bot
        bot1: First bot instance
        bot2_name: Name for second bot
        bot2: Second bot instance
        num_games: Number of games for comparison (defaults to 100 for speed)
        config: Game configuration (defaults to small: 5x5, 2 colors)
        
    Returns:
        Head-to-head comparison results
    """
    print(f"Quick Comparison: {bot1_name} vs {bot2_name}")
    print("=" * 50)
    
    # Use small config for quick comparison
    if config is None:
        config = GameFactory.small()  # 5x5 board with 2 colors
        
    print(f"Configuration: {config.num_rows}x{config.num_cols} board, {config.num_colors} colors")
    print(f"Number of games: {num_games}")
    print()
    
    # Create benchmark and run bots
    benchmark = Benchmark(config=config, num_games=num_games)
    results = benchmark.run_bots({bot1_name: bot1, bot2_name: bot2})
    
    # Get head-to-head analysis
    h2h = benchmark.head_to_head_analysis(bot1_name, bot2_name)
    
    # Display results
    if "error" not in h2h:
        print("Head-to-Head Results:")
        print("-" * 20)
        print(f"{bot1_name} wins: {h2h['wins_bot1']} ({h2h['win_rate_bot1']:.1%})")
        print(f"{bot2_name} wins: {h2h['wins_bot2']} ({h2h['win_rate_bot2']:.1%})")
        print(f"Ties: {h2h['ties']}")
        print(f"Avg performance difference: {h2h['avg_performance_difference']:.1f}")
        
        winner = bot1_name if h2h['wins_bot1'] > h2h['wins_bot2'] else bot2_name
        if h2h['wins_bot1'] != h2h['wins_bot2']:
            print(f"\nWinner: {winner}")
        else:
            print("\nResult: Tie")
    else:
        print(f"Error in comparison: {h2h['error']}")
    
    print(f"\nQuick comparison complete!")
    
    return {
        "benchmark": benchmark,
        "results": results,
        "head_to_head": h2h,
        "bot1_name": bot1_name,
        "bot2_name": bot2_name,
        "num_games": num_games,
        "config": config
    }


def load_and_analyze(dataset_path: str, generate_plots: bool = True, output_dir: str = "loaded_analysis") -> dict[str, object]:
    """
    Load existing benchmark results and generate analysis.
    
    Args:
        dataset_path: Path to saved benchmark dataset
        generate_plots: Whether to generate visualization plots
        output_dir: Directory for saving analysis plots
        
    Returns:
        Analysis results and statistics
    """
    print(f"Loading Benchmark Results: {dataset_path}")
    print("=" * 50)
    
    # Load benchmark
    benchmark = Benchmark.load_from_file(dataset_path)
    
    if benchmark is None:
        print(f"Error: Could not load benchmark data from {dataset_path}")
        return {"error": f"Could not load {dataset_path}"}
    
    print(f"Loaded benchmark with {len(benchmark)} games")
    print(f"Available bots: {list(benchmark.results.keys())}")
    print()
    
    # Generate comparison
    comparison = benchmark.compare()
    
    # Display results
    print("Performance Summary:")
    print("-" * 30)
    for bot_name in sorted(comparison.keys()):
        stats = comparison[bot_name]
        print(f"\n{bot_name}:")
        print(f"  Completion rate: {stats['completion_rate']:.1%}")
        print(f"  Avg tiles cleared: {stats['avg_tiles_cleared']:.1f}")
        print(f"  Avg moves made: {stats['avg_moves_made']:.1f}")
    
    # Generate plots
    if generate_plots and len(benchmark.results) > 0:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        benchmark.generate_report(output_dir)
        print(f"\nAnalysis plots saved to: {output_dir}/")
    
    print("Analysis complete!")
    
    return {
        "benchmark": benchmark,
        "comparison": comparison,
        "benchmark_path": dataset_path,
        "num_games": len(benchmark),
        "config": benchmark.config
    }


# Example usage functions for documentation
def main():
    """Example usage of benchmark scripts"""
    print("SameGameRL Benchmark Scripts")
    print("=" * 40)
    print("Available functions:")
    print("1. run_standard_benchmark() - Run all built-in bots")
    print("2. evaluate_custom_agent() - Test your trained agent")
    print("3. quick_comparison() - Head-to-head comparison")
    print("4. load_and_analyze() - Analyze saved results")
    print()
    print("Example usage:")
    print("  from samegamerl.evaluation.benchmark_scripts import run_standard_benchmark")
    print("  results = run_standard_benchmark(num_games=100)")


if __name__ == "__main__":
    main()