#!/usr/bin/env python3
"""
Creating benchmark dataset

This script:
1. Generates a standardized dataset of 1000 games
2. Evaluates all three benchmark bots against the dataset
3. Analyses and compares their performance
4. Saves results for future comparisons
"""


from pathlib import Path

from samegamerl.evaluation.benchmark_dataset import BenchmarkDataset
from samegamerl.evaluation.benchmark_runner import BenchmarkRunner
from samegamerl.evaluation.benchmark_analysis import BenchmarkAnalysis
from samegamerl.game.game_config import GameFactory


def main():
    print("SameGameRL Benchmark Creation")
    print("=" * 50)

    # Configuration
    num_games = 1000
    config = GameFactory.large()  # 8x8 board with 4 colors (3 playable + white spaces)

    print(
        f"Configuration: {config.num_rows}x{config.num_cols} board, {config.num_colors} colors"
    )
    print(f"Number of games: {num_games}")
    print()

    # Step 1: Create or load benchmark dataset
    print("Step 1: Creating benchmark dataset...")
    dataset = BenchmarkDataset()

    if dataset.load_dataset():
        print(f"Loaded existing dataset with {len(dataset)} games")
    else:
        print(f"Generating new dataset with {num_games} games...")
        dataset.generate_games(num_games=num_games, config=config, base_seed=42)
        dataset.save_dataset()
        print(f"Dataset saved")
    print()

    # Step 2: Run benchmark evaluation
    print("Step 2: Evaluating benchmark bots...")
    runner = BenchmarkRunner(dataset)

    # Check if we already have results
    if not dataset.results:
        print("Running full benchmark evaluation...")
        results = runner.run_full_benchmark()

        # Save updated dataset with results
        dataset.save_dataset()
        print("Results saved to dataset")
    else:
        print("Using existing benchmark results")
        results = dataset.results
    print()

    # Step 3: Display performance summaries
    print("Step 3: Performance Summary")
    print("-" * 30)

    for bot_name in sorted(results.keys()):
        summary = runner.get_bot_summary(bot_name)
        print(f"\n{bot_name}:")
        print(f"  Games played: {summary['total_games']}")
        print(f"  Completion rate: {summary['completion_rate']:.1%}")
        print(f"  Avg tiles cleared: {summary['avg_tiles_cleared']:.1f}")
        print(f"  Avg moves made: {summary['avg_moves_made']:.1f}")
        print(f"  Avg singles remaining: {summary['avg_singles_remaining']:.1f}")
    print()

    # Step 4: Comparative analysis
    print("Step 4: Comparative Analysis")
    print("-" * 30)

    analysis = BenchmarkAnalysis(dataset)

    # Head-to-head comparisons
    bot_names = list(results.keys())
    if len(bot_names) >= 2:
        for i, bot1 in enumerate(bot_names):
            for bot2 in bot_names[i + 1 :]:
                h2h = analysis.head_to_head_analysis(bot1, bot2)
                if "error" not in h2h:
                    print(f"\n{bot1} vs {bot2}:")
                    print(
                        f"  {bot1} wins: {h2h['wins_bot1']} ({h2h['win_rate_bot1']:.1%})"
                    )
                    print(
                        f"  {bot2} wins: {h2h['wins_bot2']} ({h2h['win_rate_bot2']:.1%})"
                    )
                    print(f"  Ties: {h2h['ties']}")
                    print(
                        f"  Avg performance difference: {h2h['avg_performance_difference']:.1f}"
                    )
    print()

    # Step 5: Generate analysis
    print("Step 5: Generate Analysis")
    print("-" * 30)

    # Generate comprehensive report  
    report_path = Path("samegamerl/evaluation/datasets") / "benchmark_report.txt"
    analysis.generate_performance_report(str(report_path))

    # Show comparison plots (if matplotlib available)
    analysis.show_comparison_plots()

    print("\nFiles generated:")
    print(f"  {dataset.dataset_path} - Dataset with results")
    print(f"  {report_path} - Performance report")
    print(
        "\nPlots displayed interactively (use analysis.save_comparison_plots() to save them)"
    )

    # Step 6: Find interesting games for analysis
    print("\nStep 6: Finding Interesting Games")
    print("-" * 30)

    interesting_games = analysis.find_interesting_games()

    print(
        f"Top 10% most interesting games (highest variance): {len(interesting_games)}"
    )
    if interesting_games:
        print(f"  Examples: {interesting_games[:5]}")

    print("\nBenchmark analysis complete!")
    print(
        f"Dataset contains {len(dataset)} standardized games ready for RL agent evaluation."
    )


if __name__ == "__main__":
    main()
