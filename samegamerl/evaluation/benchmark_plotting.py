"""
Benchmark-specific plotting functionality.

Provides visualization tools for comparing bot performance, separate from
training progress visualization.
"""

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from samegamerl.evaluation.benchmark import Benchmark

try:
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


def plot_comparison(
    benchmark: "Benchmark", 
    bot_names: list[str] | None = None,
    show: bool = True,
    save_path: str | None = None
) -> None:
    """
    Display comparison plots for bot performance.
    
    Args:
        benchmark: Benchmark instance with results
        bot_names: List of bot names to include (defaults to all)
        show: Whether to display plots interactively
        save_path: Optional directory to save plots
    """
    if not PLOTTING_AVAILABLE:
        print("Install matplotlib for plotting: pip install matplotlib")
        return
        
    if bot_names is None:
        bot_names = list(benchmark.results.keys())
        
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
        results = benchmark.results.get(bot_name, [])
        if results:
            tiles_cleared = [r.tiles_cleared for r in results]
            completion_rate = sum(1 for r in results if r.completed) / len(results)
            
            plot_data["tiles_cleared"].extend(tiles_cleared)
            plot_data["completion_rates"].append(completion_rate)
            plot_data["bot_names"].extend([bot_name] * len(tiles_cleared))
    
    # Create tiles cleared box plot
    plt.figure(figsize=(12, 6))
    
    # Simple box plot without seaborn dependency
    bot_data = {}
    for bot_name in bot_names:
        results = benchmark.results.get(bot_name, [])
        if results:
            bot_data[bot_name] = [r.tiles_cleared for r in results]
    
    if bot_data:
        plt.boxplot(
            bot_data.values(),
            labels=bot_data.keys(),
            patch_artist=True
        )
        plt.title("Tiles Cleared Distribution by Bot")
        plt.xlabel("Bot")
        plt.ylabel("Tiles Cleared")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            Path(save_path).mkdir(parents=True, exist_ok=True)
            plt.savefig(f"{save_path}/tiles_cleared_distribution.png")
            
        if show:
            plt.show()
        else:
            plt.close()
    
    # Create completion rate bar chart
    plt.figure(figsize=(10, 6))
    completion_rates = []
    
    for bot_name in bot_names:
        results = benchmark.results.get(bot_name, [])
        if results:
            completion_rate = sum(1 for r in results if r.completed) / len(results)
            completion_rates.append(completion_rate)
        else:
            completion_rates.append(0)
    
    bars = plt.bar(bot_names, completion_rates)
    plt.title("Game Completion Rate by Bot")
    plt.ylabel("Completion Rate")
    plt.xlabel("Bot")
    plt.xticks(rotation=45)
    
    # Add percentage labels on bars
    for i, (bar, rate) in enumerate(zip(bars, completion_rates)):
        plt.text(bar.get_x() + bar.get_width()/2, rate + 0.01, 
                f"{rate:.1%}", ha="center", va="bottom")
    
    plt.ylim(0, 1.1)  # Set y-axis to show full percentage range
    plt.tight_layout()
    
    if save_path:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{save_path}/completion_rates.png")
        
    if show:
        plt.show()
    else:
        plt.close()
    
    if save_path:
        print(f"Plots saved to {save_path}/")


def plot_head_to_head(
    benchmark: "Benchmark",
    bot1: str,
    bot2: str,
    show: bool = True,
    save_path: str | None = None
) -> None:
    """
    Plot head-to-head comparison between two bots.
    
    Args:
        benchmark: Benchmark instance with results
        bot1: First bot name
        bot2: Second bot name
        show: Whether to display plot interactively
        save_path: Optional directory to save plot
    """
    if not PLOTTING_AVAILABLE:
        print("Install matplotlib for plotting: pip install matplotlib")
        return
        
    h2h_results = benchmark.head_to_head_analysis(bot1, bot2)
    
    if "error" in h2h_results:
        print(f"Error: {h2h_results['error']}")
        return
        
    # Create win/loss pie chart
    plt.figure(figsize=(8, 6))
    
    wins_bot1 = h2h_results["wins_bot1"]
    wins_bot2 = h2h_results["wins_bot2"]
    ties = h2h_results["ties"]
    
    labels = [f"{bot1} Wins", f"{bot2} Wins", "Ties"]
    sizes = [wins_bot1, wins_bot2, ties]
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    
    # Filter out zero values
    filtered_data = [(label, size, color) for label, size, color 
                     in zip(labels, sizes, colors) if size > 0]
    
    if filtered_data:
        labels, sizes, colors = zip(*filtered_data)
        
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title(f"Head-to-Head: {bot1} vs {bot2}")
        plt.axis('equal')
        
        if save_path:
            Path(save_path).mkdir(parents=True, exist_ok=True)
            plt.savefig(f"{save_path}/head_to_head_{bot1}_vs_{bot2}.png")
            
        if show:
            plt.show()
        else:
            plt.close()
    else:
        print("No valid data for head-to-head comparison")


def plot_performance_distribution(
    benchmark: "Benchmark",
    bot_names: list[str] | None = None,
    metric: str = "tiles_cleared",
    show: bool = True,
    save_path: str | None = None
) -> None:
    """
    Plot distribution of a specific performance metric.
    
    Args:
        benchmark: Benchmark instance with results
        bot_names: List of bot names to include (defaults to all)
        metric: Performance metric to plot ('tiles_cleared', 'moves_made', 'singles_remaining')
        show: Whether to display plot interactively
        save_path: Optional directory to save plot
    """
    if not PLOTTING_AVAILABLE:
        print("Install matplotlib for plotting: pip install matplotlib")
        return
        
    if bot_names is None:
        bot_names = list(benchmark.results.keys())
        
    plt.figure(figsize=(12, 6))
    
    metric_map = {
        "tiles_cleared": "Tiles Cleared",
        "moves_made": "Moves Made", 
        "singles_remaining": "Singles Remaining"
    }
    
    if metric not in metric_map:
        print(f"Unknown metric: {metric}. Available: {list(metric_map.keys())}")
        return
        
    ylabel = metric_map[metric]
    
    for i, bot_name in enumerate(bot_names):
        results = benchmark.results.get(bot_name, [])
        if results:
            values = [getattr(r, metric) for r in results]
            plt.hist(values, bins=20, alpha=0.7, label=bot_name, 
                    color=plt.cm.Set3(i))
    
    plt.title(f"{ylabel} Distribution Comparison")
    plt.xlabel(ylabel)
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{save_path}/{metric}_distribution.png")
        
    if show:
        plt.show()
    else:
        plt.close()


def generate_benchmark_report(
    benchmark: "Benchmark",
    output_dir: str = "benchmark_report",
    bot_names: list[str] | None = None
) -> None:
    """
    Generate a complete benchmark report with all plots and statistics.
    
    Args:
        benchmark: Benchmark instance with results
        output_dir: Directory to save report files
        bot_names: List of bot names to include (defaults to all)
    """
    if not PLOTTING_AVAILABLE:
        print("Install matplotlib for plotting: pip install matplotlib")
        return
        
    if bot_names is None:
        bot_names = list(benchmark.results.keys())
        
    if len(bot_names) == 0:
        print("No bot results available for report generation")
        return
        
    print(f"Generating benchmark report in {output_dir}...")
    
    # Generate all plots
    plot_comparison(benchmark, bot_names, show=False, save_path=output_dir)
    
    for metric in ["tiles_cleared", "moves_made", "singles_remaining"]:
        plot_performance_distribution(
            benchmark, bot_names, metric, show=False, save_path=output_dir
        )
    
    # Generate head-to-head plots for all pairs
    if len(bot_names) >= 2:
        for i, bot1 in enumerate(bot_names):
            for bot2 in bot_names[i + 1:]:
                plot_head_to_head(benchmark, bot1, bot2, show=False, save_path=output_dir)
    
    # Generate text summary
    comparison = benchmark.compare(bot_names)
    
    summary_lines = []
    summary_lines.append("BENCHMARK PERFORMANCE SUMMARY")
    summary_lines.append("=" * 50)
    summary_lines.append(f"Games per bot: {len(benchmark.games)}")
    summary_lines.append(f"Game configuration: {benchmark.config.num_rows}x{benchmark.config.num_cols}, {benchmark.config.num_colors} colors")
    summary_lines.append("")
    
    for bot_name in sorted(bot_names):
        if bot_name in comparison:
            stats = comparison[bot_name]
            summary_lines.append(f"{bot_name}:")
            summary_lines.append(f"  Completion rate: {stats['completion_rate']:.1%}")
            summary_lines.append(f"  Avg tiles cleared: {stats['avg_tiles_cleared']:.1f}")
            summary_lines.append(f"  Avg moves made: {stats['avg_moves_made']:.1f}")
            summary_lines.append(f"  Avg singles remaining: {stats['avg_singles_remaining']:.1f}")
            summary_lines.append("")
    
    # Save text summary
    summary_path = Path(output_dir) / "summary.txt"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))
    
    print(f"Report generated: {len(bot_names)} bots analyzed")
    print(f"Files saved in: {output_dir}/")