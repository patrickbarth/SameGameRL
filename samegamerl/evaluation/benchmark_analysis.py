"""
Benchmark analysis utilities for comparing agent performance.

Provides statistical analysis, reporting, and visualization tools for benchmark
results across multiple agents.
"""

from pathlib import Path
import statistics

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

from samegamerl.evaluation.benchmark_dataset import BenchmarkDataset, BotPerformance


class BenchmarkAnalysis:
    """Analysis and visualization tools for benchmark results"""

    def __init__(self, dataset: BenchmarkDataset):
        self.dataset = dataset

    def generate_performance_report(
        self, output_file: str = "benchmark_report.txt"
    ) -> None:
        """Generate comprehensive text report of all bot performances"""
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("SAMEGAME BENCHMARK PERFORMANCE REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Dataset: {len(self.dataset)} games")
        report_lines.append("")

        # Get all bot names
        bot_names = list(self.dataset.results.keys())
        if not bot_names:
            report_lines.append("No benchmark results available.")
            with open(output_file, "w") as f:
                f.write("\n".join(report_lines))
            return

        # Individual bot analysis
        for bot_name in sorted(bot_names):
            results = self.dataset.get_bot_results(bot_name)
            if not results:
                continue

            report_lines.append(f"BOT: {bot_name}")
            report_lines.append("-" * 40)

            stats = self._calculate_detailed_stats(results)

            report_lines.append(f"Games Played: {stats['total_games']}")
            report_lines.append(f"Completion Rate: {stats['completion_rate']:.1%}")
            report_lines.append(
                f"Average Tiles Cleared: {stats['avg_tiles_cleared']:.1f}"
            )
            report_lines.append(f"Average Moves: {stats['avg_moves_made']:.1f}")
            report_lines.append(
                f"Average Singles Left: {stats['avg_singles_remaining']:.1f}"
            )
            report_lines.append("")

        # Comparative analysis
        if len(bot_names) > 1:
            report_lines.append("COMPARATIVE ANALYSIS")
            report_lines.append("-" * 40)

            # Create comparison table
            comparison_data = []
            for bot_name in sorted(bot_names):
                results = self.dataset.get_bot_results(bot_name)
                if results:
                    stats = self._calculate_detailed_stats(results)
                    comparison_data.append(
                        (
                            bot_name,
                            stats["completion_rate"],
                            stats["avg_tiles_cleared"],
                            stats["avg_moves_made"],
                        )
                    )

            # Sort by completion rate (descending)
            comparison_data.sort(key=lambda x: x[1], reverse=True)

            report_lines.append("Ranking by Completion Rate:")
            for i, (
                bot_name,
                completion_rate,
                avg_tiles,
                avg_moves,
            ) in enumerate(comparison_data, 1):
                report_lines.append(
                    f"  {i}. {bot_name}: {completion_rate:.1%} completion, "
                    f"{avg_tiles:.1f} avg tiles cleared, {avg_moves:.1f} avg moves"
                )

        with open(output_file, "w") as f:
            f.write("\n".join(report_lines))

        print(f"Performance report saved to {output_file}")

    def show_comparison_plots(self) -> None:
        """Display interactive visualization plots comparing bot performance"""
        if not PLOTTING_AVAILABLE:
            print("Matplotlib/Seaborn not available. Install with: pip install matplotlib seaborn")
            return

        bot_names = list(self.dataset.results.keys())
        if len(bot_names) < 2:
            print("Need at least 2 bots for comparison plots")
            return

        # Prepare data for plotting
        plot_data = {
            "completion_rates": [],
            "tiles_cleared": [],
            "bot_names": [],
        }

        for bot_name in bot_names:
            results = self.dataset.get_bot_results(bot_name)
            if results:
                tiles_cleared = [r.tiles_cleared for r in results]
                completion_rate = sum(1 for r in results if r.completed) / len(results)

                plot_data["tiles_cleared"].extend(tiles_cleared)
                plot_data["completion_rates"].append(completion_rate)
                plot_data["bot_names"].extend([bot_name] * len(tiles_cleared))

        # Tiles cleared distribution comparison
        plt.figure(figsize=(12, 6))
        sns.boxplot(
            data={"Bot": plot_data["bot_names"], "Tiles Cleared": plot_data["tiles_cleared"]},
            x="Bot",
            y="Tiles Cleared",
        )
        plt.title("Tiles Cleared Distribution by Bot")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # Completion rate comparison
        plt.figure(figsize=(10, 6))
        bot_names_unique = list(set(plot_data["bot_names"]))
        completion_rates = []
        for bot_name in bot_names_unique:
            results = self.dataset.get_bot_results(bot_name)
            completion_rate = sum(1 for r in results if r.completed) / len(results)
            completion_rates.append(completion_rate)

        plt.bar(bot_names_unique, completion_rates)
        plt.title("Game Completion Rate by Bot")
        plt.ylabel("Completion Rate")
        plt.xlabel("Bot")
        plt.xticks(rotation=45)
        for i, rate in enumerate(completion_rates):
            plt.text(i, rate + 0.01, f"{rate:.1%}", ha="center")
        plt.tight_layout()
        plt.show()

    def save_comparison_plots(self, output_dir: str = "benchmark_plots") -> None:
        """Save visualization plots to files (for when you want to keep them)"""
        if not PLOTTING_AVAILABLE:
            print("Matplotlib/Seaborn not available. Install with: pip install matplotlib seaborn")
            return

        Path(output_dir).mkdir(exist_ok=True)

        bot_names = list(self.dataset.results.keys())
        if len(bot_names) < 2:
            print("Need at least 2 bots for comparison plots")
            return

        # Prepare data for plotting
        plot_data = {
            "completion_rates": [],
            "tiles_cleared": [],
            "bot_names": [],
        }

        for bot_name in bot_names:
            results = self.dataset.get_bot_results(bot_name)
            if results:
                tiles_cleared = [r.tiles_cleared for r in results]
                completion_rate = sum(1 for r in results if r.completed) / len(results)

                plot_data["tiles_cleared"].extend(tiles_cleared)
                plot_data["completion_rates"].append(completion_rate)
                plot_data["bot_names"].extend([bot_name] * len(tiles_cleared))

        # Tiles cleared distribution comparison
        plt.figure(figsize=(12, 6))
        sns.boxplot(
            data={"Bot": plot_data["bot_names"], "Tiles Cleared": plot_data["tiles_cleared"]},
            x="Bot",
            y="Tiles Cleared",
        )
        plt.title("Tiles Cleared Distribution by Bot")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/tiles_cleared_distribution.png")
        plt.close()

        # Completion rate comparison
        plt.figure(figsize=(10, 6))
        bot_names_unique = list(set(plot_data["bot_names"]))
        completion_rates = []
        for bot_name in bot_names_unique:
            results = self.dataset.get_bot_results(bot_name)
            completion_rate = sum(1 for r in results if r.completed) / len(results)
            completion_rates.append(completion_rate)

        plt.bar(bot_names_unique, completion_rates)
        plt.title("Game Completion Rate by Bot")
        plt.ylabel("Completion Rate")
        plt.xlabel("Bot")
        plt.xticks(rotation=45)
        for i, rate in enumerate(completion_rates):
            plt.text(i, rate + 0.01, f"{rate:.1%}", ha="center")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/completion_rates.png")
        plt.close()

        print(f"Plots saved to {output_dir}/")

    def head_to_head_analysis(self, bot1: str, bot2: str) -> dict[str, float | str | int]:
        """Compare two bots game-by-game"""
        results1 = self.dataset.get_bot_results(bot1)
        results2 = self.dataset.get_bot_results(bot2)

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
        bot_names = list(self.dataset.results.keys())
        if len(bot_names) < 2:
            return []

        game_variances = []

        for game_id in range(len(self.dataset)):
            tiles_cleared_scores = []
            for bot_name in bot_names:
                results = self.dataset.get_bot_results(bot_name)
                if game_id < len(results):
                    tiles_cleared_scores.append(results[game_id].tiles_cleared)

            if len(tiles_cleared_scores) >= 2:
                variance = statistics.variance(tiles_cleared_scores)
                game_variances.append((game_id, variance))

        # Sort by variance (highest first) and return top 10%
        game_variances.sort(key=lambda x: x[1], reverse=True)
        top_10_percent = max(1, len(game_variances) // 10)
        
        return [game_id for game_id, _ in game_variances[:top_10_percent]]

    def _calculate_detailed_stats(
        self, results: list[BotPerformance]
    ) -> dict[str, float | int]:
        """Calculate detailed statistics for a set of results"""
        if not results:
            return {}

        tiles_cleared = [r.tiles_cleared for r in results]
        moves_made = [r.moves_made for r in results]
        singles_remaining = [r.singles_remaining for r in results]

        return {
            "total_games": len(results),
            "completion_rate": sum(1 for r in results if r.completed) / len(results),
            "avg_tiles_cleared": statistics.mean(tiles_cleared),
            "median_tiles_cleared": statistics.median(tiles_cleared),
            "max_tiles_cleared": max(tiles_cleared),
            "min_tiles_cleared": min(tiles_cleared),
            "tiles_cleared_std_dev": statistics.stdev(tiles_cleared) if len(tiles_cleared) > 1 else 0,
            "avg_moves_made": statistics.mean(moves_made),
            "avg_singles_remaining": statistics.mean(singles_remaining),
        }