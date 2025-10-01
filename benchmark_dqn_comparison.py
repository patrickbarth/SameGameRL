"""
Example script demonstrating how to benchmark trained DQN agents.

Shows how to:
1. Load a trained DQN agent from saved weights
2. Evaluate agent performance using the benchmark system
3. Compare DQN agent against baseline bots
4. Visualize comparative performance
"""

from pathlib import Path

import torch
from torch import nn

from samegamerl.agents.dqn_agent import DqnAgent
from samegamerl.evaluation.benchmark import Benchmark
from samegamerl.evaluation.benchmark_plotting import BenchmarkPlotter
from samegamerl.game.game_config import GameConfig, GameFactory


# Define the model architecture (must match training)
class PyramidModel(nn.Module):
    """Fully connected pyramid architecture for DQN."""

    def __init__(self, config: GameConfig):
        super().__init__()
        self.config = config
        self.flatten = nn.Flatten()
        input_size = config.num_rows * config.num_cols * config.num_colors
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, config.action_space_size),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def load_trained_agent(model_name: str, config: GameConfig) -> DqnAgent:
    """
    Load a trained DQN agent from saved weights.

    Args:
        model_name: Name of the model file (without .pth extension)
        config: Game configuration used during training

    Returns:
        Loaded DQN agent ready for evaluation
    """
    # Create model with same architecture as training
    model = PyramidModel(config)

    # Create agent (hyperparameters don't matter for evaluation)
    agent = DqnAgent(
        model=model,
        config=config,
        model_name=model_name,
        learning_rate=0.001,  # Not used for evaluation
        initial_epsilon=0.0,   # Will be set to 0 by adapter anyway
        epsilon_decay=0.0,
        final_epsilon=0.0,
    )

    # Load trained weights
    agent.load(name=model_name)

    print(f"Loaded agent: {model_name}")
    return agent


def example_ephemeral_evaluation():
    """Example: Quick evaluation without saving results."""
    print("=== Example 1: Ephemeral Evaluation ===\n")

    config = GameFactory.medium()

    # Create benchmark with small number of games for quick testing
    benchmark = Benchmark(
        config=config,
        num_games=10,
        base_seed=42,
        use_ray=False  # Sequential for easier debugging
    )

    # Load trained agent
    model_name = "pyramid"  # Change to your trained model name
    model_path = Path(f"samegamerl/models/{model_name}.pth")

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Please train a model first or update the model_name variable.")
        return

    agent = load_trained_agent(model_name, config)

    # Evaluate without saving (default behavior)
    print("\nEvaluating agent (ephemeral mode)...")
    results = benchmark.evaluate_agent(agent, save_results=False)

    # Analyze results
    tiles_cleared = [r.tiles_cleared for r in results]
    completed = [r.completed for r in results]

    print(f"\nResults over {len(results)} games:")
    print(f"  Average tiles cleared: {sum(tiles_cleared) / len(tiles_cleared):.1f}")
    print(f"  Completion rate: {sum(completed) / len(completed) * 100:.1f}%")
    print(f"  Average moves: {sum(r.moves_made for r in results) / len(results):.1f}")


def example_persistent_evaluation():
    """Example: Evaluation with saved results for later comparison."""
    print("\n\n=== Example 2: Persistent Evaluation ===\n")

    config = GameFactory.medium()

    # Create benchmark with more games for robust comparison
    benchmark = Benchmark(
        config=config,
        num_games=100,
        base_seed=42,
        use_ray=True  # Parallel execution for faster benchmarking
    )

    # Load trained agent
    model_name = "pyramid"
    model_path = Path(f"samegamerl/models/{model_name}.pth")

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Skipping persistent evaluation example.")
        return

    agent = load_trained_agent(model_name, config)

    # Evaluate and save results
    print("\nEvaluating agent (persistent mode)...")
    results = benchmark.evaluate_agent(agent, save_results=True)

    print(f"\nResults saved! The agent's performance is now stored in the benchmark.")
    print("You can load this benchmark later to compare against other agents.")


def example_compare_multiple_checkpoints():
    """Example: Compare performance across training checkpoints."""
    print("\n\n=== Example 3: Training Progress Comparison ===\n")

    config = GameFactory.medium()

    benchmark = Benchmark(
        config=config,
        num_games=50,
        base_seed=42,
        use_ray=True
    )

    # Checkpoints saved during training (example names)
    checkpoint_names = [
        "pyramid_epoch1000",
        "pyramid_epoch5000",
        "pyramid_epoch10000",
        "pyramid_final"
    ]

    print("Evaluating multiple training checkpoints...")

    for checkpoint_name in checkpoint_names:
        model_path = Path(f"samegamerl/models/{checkpoint_name}.pth")

        if not model_path.exists():
            print(f"  Skipping {checkpoint_name} (not found)")
            continue

        agent = load_trained_agent(checkpoint_name, config)
        results = benchmark.evaluate_agent(agent, save_results=True)

        # Quick summary
        avg_cleared = sum(r.tiles_cleared for r in results) / len(results)
        completion_rate = sum(r.completed for r in results) / len(results) * 100
        print(f"  {checkpoint_name}: {avg_cleared:.1f} tiles, {completion_rate:.0f}% complete")

    print("\nAll checkpoints evaluated and saved!")


def example_compare_against_baselines():
    """Example: Compare DQN agent against rule-based baseline bots."""
    print("\n\n=== Example 4: DQN vs Baseline Bots ===\n")

    config = GameFactory.medium()

    benchmark = Benchmark(
        config=config,
        num_games=100,
        base_seed=42,
        use_ray=True
    )

    # First, run baseline bots
    print("Running baseline bots...")
    baseline_results = benchmark.run_bots()  # Uses built-in bots

    # Then evaluate DQN agent
    model_name = "pyramid"
    model_path = Path(f"samegamerl/models/{model_name}.pth")

    if model_path.exists():
        print("\nEvaluating DQN agent...")
        agent = load_trained_agent(model_name, config)
        dqn_results = benchmark.evaluate_agent(agent, save_results=True)

        # Compare performance
        print("\n=== Performance Comparison ===")
        print(f"{'Bot Name':<25} | Avg Tiles | Completion % | Avg Moves")
        print("-" * 70)

        # Show baseline bots
        for bot_name, results in baseline_results.items():
            avg_tiles = sum(r.tiles_cleared for r in results) / len(results)
            completion = sum(r.completed for r in results) / len(results) * 100
            avg_moves = sum(r.moves_made for r in results) / len(results)
            print(f"{bot_name:<25} | {avg_tiles:9.1f} | {completion:12.1f}% | {avg_moves:9.1f}")

        # Show DQN agent (extract name from first result)
        if dqn_results:
            dqn_name = dqn_results[0].bot_name
            avg_tiles = sum(r.tiles_cleared for r in dqn_results) / len(dqn_results)
            completion = sum(r.completed for r in dqn_results) / len(dqn_results) * 100
            avg_moves = sum(r.moves_made for r in dqn_results) / len(dqn_results)
            print(f"{dqn_name:<25} | {avg_tiles:9.1f} | {completion:12.1f}% | {avg_moves:9.1f}")

        # Create visualization
        print("\nGenerating comparison plot...")
        plotter = BenchmarkPlotter(benchmark)
        plotter.plot_performance_comparison()
        print("Plot saved!")
    else:
        print(f"\nDQN model not found: {model_path}")
        print("Only showing baseline bot results.")


def main():
    """Run all examples (comment out those you don't need)."""

    # Quick evaluation without saving
    example_ephemeral_evaluation()

    # Evaluation with persistent storage
    # example_persistent_evaluation()

    # Compare multiple training checkpoints
    # example_compare_multiple_checkpoints()

    # Full comparison: DQN vs baseline bots
    # example_compare_against_baselines()

    print("\n" + "="*70)
    print("Examples complete!")
    print("\nTo run other examples, uncomment them in the main() function.")


if __name__ == "__main__":
    main()
